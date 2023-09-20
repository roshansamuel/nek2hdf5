#!/usr/bin/python3

import sys
import struct
import itertools
import numpy as np
import xarray as xr
from functools import partial
import matplotlib.pyplot as plt
from xarray.core.utils import Frozen

# repeat() can be used as a stand-in for range()
repeat = partial(itertools.repeat, None)


# ==============================================================================
class Elem:
    def __init__(self, lr1, dtype="float64"):
        #                    x,y,z   lz      ly      lx
        self.pos = np.zeros((3, lr1[2], lr1[1], lr1[0]), dtype=dtype)
        #                    u,v,w   lz      ly      lx
        self.vel = np.zeros((3, lr1[2], lr1[1], lr1[0]), dtype=dtype)
        #                    p       lz      ly      lx
        self.pres = np.zeros((1, lr1[2], lr1[1], lr1[0]), dtype=dtype)
        #                    T       lz      ly      lx
        self.temp = np.zeros((1, lr1[2], lr1[1], lr1[0]), dtype=dtype)


# ==============================================================================
class HexaData:
    def __init__(self, nel, lr1, dtype="float64"):
        self.nel = nel
        self.lr1 = lr1

        self.elem = [Elem(lr1, dtype) for _ in repeat(nel)]

# ==============================================================================
class _NekDataStore(xr.backends.common.AbstractDataStore):
    axes = ("z", "y", "x")

    def __init__(self, elem):
        self.elem = elem

    def meshgrid_to_dim(self, mesh):
        dim = np.unique(np.round(mesh, 8))
        return dim

    def get_attrs(self):
        elem = self.elem
        attrs = { }
        return Frozen(attrs)

    def get_variables(self):
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
    headData = infile.read(132).split()
    precSize = int(headData[1])
    polyOrder = tuple([int(x) for x in headData[2:5]])
    numElems = int(headData[5])
    solTime = float(headData[7])
    timeStep = int(headData[8])
    varList = str(headData[11])[2:-1]
    ptsPerElem = np.prod(polyOrder)

    if precSize == 4:
        rType = "f"
    elif precSize == 8:
        rType = "d"

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
    elmap = infile.read(4 * numElems)
    elmap = struct.unpack(emode + numElems * "i", elmap)

    # ---------------------------------------------------------------------------
    # READ DATA
    # ---------------------------------------------------------------------------
    # initialize data structure
    data = HexaData(numElems, polyOrder)

    bytes_elem = ptsPerElem * precSize

    def read_file_into_data(data_var, index_var=0):
        """Read binary file into an array attribute of ``data.elem``"""
        fi = infile.read(bytes_elem)
        fi = np.frombuffer(fi, dtype=emode + rType, count=ptsPerElem)

        # Replace elem array in-place with
        # array read from file after reshaping as
        elem_shape = polyOrder[::-1]  # lz, ly, lx
        data_var[index_var, ...] = fi.reshape(elem_shape)

    # read geometry
    if varList[0] == 'X':
        for iel in elmap:
            el = data.elem[iel - 1]
            for idim in range(3):
                read_file_into_data(el.pos, idim)
    else:
        # Read grid data from file
        try:
            gfName = sys.argv[2]
            g = open(gfName, 'rb')
        except:
            print("Please specify grid file :(\n")
            exit()

        g.read(136)

        gelMap = g.read(4 * numElems)
        gelMap = struct.unpack(emode + numElems * "i", gelMap)
        gelMap = np.array(gelMap, dtype=np.int32)

        if gelMap[0] != elmap[0]:
            print("Grid file is incompatible :(\n")
            exit()

        for iel in elmap:
            el = data.elem[iel - 1]
            for idim in range(3):
                fi = g.read(bytes_elem)
                fi = np.frombuffer(fi, dtype=emode + rType, count=ptsPerElem)
                elem_shape = polyOrder[::-1]  # lz, ly, lx
                el.pos[idim, ...] = fi.reshape(elem_shape)

        g.close()

    # read velocity
    for iel in elmap:
        el = data.elem[iel - 1]
        for idim in range(3):
            read_file_into_data(el.vel, idim)

    # read pressure
    for iel in elmap:
        el = data.elem[iel - 1]
        read_file_into_data(el.pres)

    # read temperature
    for iel in elmap:
        el = data.elem[iel - 1]
        read_file_into_data(el.temp)

    # close file
    infile.close()

    elements = data.elem
    elem_stores = [_NekDataStore(elem) for elem in elements]
    elem_dsets = [xr.Dataset.load_store(store).set_coords(store.axes) for store in elem_stores]

    ds = xr.combine_by_coords(elem_dsets, combine_attrs="drop")

    # output
    return ds


if __name__ == "__main__":
    try:
        fName = sys.argv[1]
    except:
        print("Could not read file :(\n")
        exit()

    ds = readnek(fName)

    x = np.array(ds['x'])
    y = np.array(ds['y'])
    z = np.array(ds['z'])

    #np.savetxt("grid_x.dat", x)
    #np.savetxt("grid_y.dat", y)
    #np.savetxt("grid_z.dat", z)

    T = ds['temperature']
    Nz, Ny, Nx = T.shape

    outData = np.zeros((5, Ny, Nx))

    dTdz = -(T[1,:,:] - T[0,:,:])/(z[1] - z[0])
    outData[0, :, :] = ds['ux'][1, :, :]
    outData[1, :, :] = ds['uy'][1, :, :]
    outData[2, :, :] = ds['uz'][1, :, :]
    outData[3, :, :] = ds['pressure'][0, :, :]
    outData[4, :, :] = dTdz[:, :]
    np.save(fName + ".b_plate", outData)

    dTdz = -(T[Nz-1,:,:] - T[Nz-2,:,:])/(z[Nz-1] - z[Nz-2])
    outData[0, :, :] = ds['ux'][Nz-2, :, :]
    outData[1, :, :] = ds['uy'][Nz-2, :, :]
    outData[2, :, :] = ds['uz'][Nz-2, :, :]
    outData[3, :, :] = ds['pressure'][Nz-2, :, :]
    outData[4, :, :] = dTdz[:, :]
    np.save(fName + ".t_plate", outData)

    #X = ds['xmesh']
    #Y = ds['ymesh']
    #plt.contourf(X[0, :, :], Y[0, :, :], dTdz)
    #plt.savefig("plot.png")
    #exit()
    #plt.contourf(X[:, 64, :], Z[:, 64, :], T[:, 64, :])
    #plt.show()
