from __future__ import annotations

import os
import io
import struct
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xarray.core.utils import Frozen
from typing import Optional, Tuple, Union, BinaryIO

from attr import define, field

from pymech import HexaData

PathLike = Union[str, os.PathLike]


@define
class Header:
    # get word size: single or double precision
    wdsz: int = field(converter=int)
    # get polynomial order
    orders: Tuple[int, ...] = field()
    # get number of elements
    nb_elems: int = field(converter=int)
    # get number of elements in the file
    nb_elems_file: int = field(converter=int)
    # get current time
    time: float = field(converter=float)
    # get current time step
    istep: int = field(converter=int)
    # get file id
    fid: int = field(converter=int)
    # get tot number of files
    nb_files: int = field(converter=int)

    # get variables [XUPTS[01-99]]
    variables: str = field(factory=str)
    # floating point precision
    realtype: str = field(factory=str)
    # compute total number of points per element
    nb_pts_elem: int = field(factory=int)
    # get number of physical dimensions
    nb_dims: int = field(factory=int)
    # get number of variables
    nb_vars: Tuple[int, ...] = field(factory=tuple)

    def __attrs_post_init__(self):
        # get word size: single or double precision
        wdsz = self.wdsz
        if not self.realtype:
            if wdsz == 4:
                self.realtype = "f"
            elif wdsz == 8:
                self.realtype = "d"
            else:
                print(f"Could not interpret real type (wdsz = {wdsz})")

        self.orders = [int(x) for x in self.orders]
        orders = self.orders
        if not self.nb_pts_elem:
            self.nb_pts_elem = np.prod(orders)

        if not self.nb_dims:
            self.nb_dims = 2 + int(orders[2] > 1)

        if not self.variables and not self.nb_vars:
            raise ValueError("Both variables and nb_vars cannot be uninitialized.")
        elif self.variables:
            self.nb_vars = self._variables_to_nb_vars()
        elif self.nb_vars:
            self.variables = self._nb_vars_to_variables()

    def _variables_to_nb_vars(self) -> Optional[Tuple[int, ...]]:
        # get variables [XUPTS[01-99]]
        variables = self.variables
        nb_dims = self.nb_dims

        if not variables:
            print("Failed to convert variables to nb_vars")
            return None

        if not nb_dims:
            print("Unintialized nb_dims")
            return None

        def nb_scalars():
            index_s = variables.index("S")
            return int(variables[index_s + 1 :])

        variables = str(variables)

        nb_vars = (
            nb_dims if "X" in variables else 0,
            nb_dims if "U" in variables else 0,
            1 if "P" in variables else 0,
            1 if "T" in variables else 0,
            nb_scalars() if "S" in variables else 0,
        )

        return nb_vars

    def _nb_vars_to_variables(self) -> Optional[str]:
        nb_vars = self.nb_vars
        if not nb_vars:
            print("Failed to convert nb_vars to variables")
            return None

        str_vars = ("X", "U", "P", "T", f"S{nb_vars[4]:02d}")
        variables = (str_vars[i] if nb_vars[i] > 0 else "" for i in range(5))
        return "".join(variables)


def open_dataset(path, **kwargs):
    _open = _open_nek_dataset

    return _open(path, **kwargs)


def _open_nek_dataset(path, drop_variables=None):
    field = readnek(path)
    if isinstance(field, int):
        raise OSError(f"Failed to load {path}")

    elements = field.elem
    elem_stores = [_NekDataStore(elem) for elem in elements]
    try:
        elem_dsets = [
            xr.Dataset.load_store(store).set_coords(store.axes) for store in elem_stores
        ]
    except ValueError as err:
        raise NotImplementedError(
            "Opening dataset failed because you probably tried to open a field file "
            "with an unsupported mesh. "
            "The `pymech.open_dataset` function currently works only with cartesian "
            "box meshes. For more details on this, see "
            "https://github.com/eX-Mech/pymech/issues/31"
        ) from err

    # See: https://github.com/MITgcm/xmitgcm/pull/200
    ds = xr.combine_by_coords(elem_dsets, combine_attrs="drop")
    ds.coords.update({"time": field.time})

    if drop_variables:
        ds = ds.drop_vars(drop_variables)

    return ds


class _NekDataStore(xr.backends.common.AbstractDataStore):
    """Xarray store for a Nek field element.

    Parameters
    ----------
    elem: :class:`pymech.core.Elem`
        A Nek5000 element.

    """

    axes = ("z", "y", "x")

    def __init__(self, elem):
        self.elem = elem

    def meshgrid_to_dim(self, mesh):
        """Reverse of np.meshgrid. This method extracts one-dimensional
        coordinates from a cubical array format for every direction
        """
        dim = np.unique(np.round(mesh, 8))
        return dim

    def get_dimensions(self):
        return self.axes

    def get_attrs(self):
        elem = self.elem
        attrs = {
            "boundary_conditions": elem.bcs,
            "curvature": elem.curv,
            "curvature_type": elem.ccurv,
        }
        return Frozen(attrs)

    def get_variables(self):
        """Generate an xarray dataset from a single element."""
        ax = self.axes
        elem = self.elem

        data_vars = {
            ax[2]: self.meshgrid_to_dim(elem.pos[0]),  # x
            ax[1]: self.meshgrid_to_dim(elem.pos[1]),  # y
            ax[0]: self.meshgrid_to_dim(elem.pos[2]),  # z
            "xmesh": xr.Variable(ax, elem.pos[0]),
            "ymesh": xr.Variable(ax, elem.pos[1]),
            "zmesh": xr.Variable(ax, elem.pos[2]),
            "ux": xr.Variable(ax, elem.vel[0]),
            "uy": xr.Variable(ax, elem.vel[1]),
            "uz": xr.Variable(ax, elem.vel[2]),
        }
        if elem.pres.size:
            data_vars["pressure"] = xr.Variable(ax, elem.pres[0])

        if elem.temp.size:
            data_vars["temperature"] = xr.Variable(ax, elem.temp[0])

        if elem.scal.size:
            data_vars.update(
                {
                    "s{:02d}".format(iscalar + 1): xr.Variable(ax, elem.scal[iscalar])
                    for iscalar in range(elem.scal.shape[0])
                }
            )

        return Frozen(data_vars)


def readnek(fname, dtype="float64", skip_vars=()):
    try:
        infile = open(fname, "rb")
    except OSError as e:
        print(f"I/O error ({e.errno}): {e.strerror}")
        return -1

    # ---------------------------------------------------------------------------
    # READ HEADER
    # ---------------------------------------------------------------------------
    h = read_header(infile)
    #
    # identify endian encoding
    etagb = infile.read(4)
    etagL = struct.unpack("<f", etagb)[0]
    etagL = int(etagL * 1e5) / 1e5
    etagB = struct.unpack(">f", etagb)[0]
    etagB = int(etagB * 1e5) / 1e5
    if etagL == 6.54321:
        emode = "<"
    elif etagB == 6.54321:
        emode = ">"
    else:
        return -3

    # read element map for the file
    elmap = infile.read(4 * h.nb_elems_file)
    elmap = struct.unpack(emode + h.nb_elems_file * "i", elmap)

    # ---------------------------------------------------------------------------
    # READ DATA
    # ---------------------------------------------------------------------------
    # initialize data structure
    data = HexaData(h.nb_dims, h.nb_elems, h.orders, h.nb_vars, 0, dtype)
    data.time = h.time
    data.istep = h.istep
    data.wdsz = h.wdsz
    data.elmap = np.array(elmap, dtype=np.int32)
    if emode == "<":
        data.endian = "little"
    elif emode == ">":
        data.endian = "big"

    bytes_elem = h.nb_pts_elem * h.wdsz

    def read_file_into_data(data_var, index_var):
        """Read binary file into an array attribute of ``data.elem``"""
        fi = infile.read(bytes_elem)
        fi = np.frombuffer(fi, dtype=emode + h.realtype, count=h.nb_pts_elem)

        # Replace elem array in-place with
        # array read from file after reshaping as
        elem_shape = h.orders[::-1]  # lz, ly, lx
        data_var[index_var, ...] = fi.reshape(elem_shape)

    def skip_elements(nb_elements=1):
        infile.seek(bytes_elem * nb_elements, os.SEEK_CUR)

    # read geometry
    geometry_vars = "x", "y", "z"
    nb_vars = h.nb_vars[0]
    skip_condition = tuple(geometry_vars[idim] in skip_vars for idim in range(nb_vars))
    if nb_vars:
        if all(skip_condition):
            skip_elements(h.nb_elems * nb_vars)
        else:
            for iel in elmap:
                el = data.elem[iel - 1]
                for idim in range(nb_vars):
                    if skip_condition[idim]:
                        skip_elements()
                    else:
                        read_file_into_data(el.pos, idim)

    # read velocity
    velocity_vars1 = "ux", "uy", "uz"
    velocity_vars2 = "vx", "vy", "vz"
    nb_vars = h.nb_vars[1]
    skip_condition1 = tuple(
        velocity_vars1[idim] in skip_vars for idim in range(nb_vars)
    )
    skip_condition2 = tuple(
        velocity_vars2[idim] in skip_vars for idim in range(nb_vars)
    )

    if nb_vars:
        if all(skip_condition1) or all(skip_condition2):
            skip_elements(h.nb_elems * nb_vars)
        else:
            for iel in elmap:
                el = data.elem[iel - 1]
                for idim in range(nb_vars):
                    if skip_condition1[idim] or skip_condition2[idim]:
                        skip_elements()
                    else:
                        read_file_into_data(el.vel, idim)

    # read pressure
    nb_vars = h.nb_vars[2]
    skip_condition = any({"p", "pressure"}.intersection(skip_vars))
    if nb_vars:
        if skip_condition:
            skip_elements(h.nb_elems * nb_vars)
        else:
            for iel in elmap:
                el = data.elem[iel - 1]
                for ivar in range(nb_vars):
                    read_file_into_data(el.pres, ivar)

    # read temperature
    nb_vars = h.nb_vars[3]
    skip_condition = any({"t", "temperature"}.intersection(skip_vars))
    if nb_vars:
        if skip_condition:
            skip_elements(h.nb_elems * nb_vars)
        else:
            for iel in elmap:
                el = data.elem[iel - 1]
                for ivar in range(nb_vars):
                    read_file_into_data(el.temp, ivar)

    # read scalar fields
    nb_vars = h.nb_vars[4]
    scalar_vars = tuple(f"s{i:02d}" for i in range(1, nb_vars + 1))
    skip_condition = tuple(scalar_vars[ivar] in skip_vars for ivar in range(nb_vars))
    if nb_vars:
        if all(skip_condition):
            skip_elements(h.nb_elems * nb_vars)
        else:
            # NOTE: This is not a bug!
            # Unlike other variables, scalars are in the outer loop and elements
            # are in the inner loop
            for ivar in range(nb_vars):
                if skip_condition[ivar]:
                    skip_elements(h.nb_elems)
                else:
                    for iel in elmap:
                        el = data.elem[iel - 1]
                        read_file_into_data(el.scal, ivar)

    # close file
    infile.close()

    # output
    return data


def read_header(path_or_file_obj: Union[PathLike, BinaryIO]) -> Header:
    if isinstance(path_or_file_obj, (str, os.PathLike)):
        with Path(path_or_file_obj).open("rb") as fp:
            header = fp.read(132).split()
    elif isinstance(path_or_file_obj, io.BufferedReader):
        fp = path_or_file_obj
        header = fp.read(132).split()
    else:
        raise ValueError("Should be a path or opened file object in 'rb' mode.")

    if len(header) < 12:
        raise IOError("Header of the file was too short.")

    # Relying on attrs converter to type-cast. Mypy will complain
    return Header(header[1], header[2:5], *header[5:12])  # type: ignore[arg-type]


ds = open_dataset('rbc_16_p3.f00001')
print(ds)
T = ds['temperature']
x = ds['xmesh']
z = ds['zmesh']
plt.contourf(x[:, 64, :], z[:, 64, :], T[:, 64, :])
plt.show()
#ds.mean(['x', 'z']).ux.plot()
#plt.show()
