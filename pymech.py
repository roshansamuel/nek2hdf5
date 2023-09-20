"""Core data structures for pymech"""
from textwrap import dedent, indent
import itertools
from itertools import product
from functools import reduce, partial

import numpy as np


"""Repeat N times. Pythonic idiom to use when the iterated value is discarded.

Example
-------
Instead of:

>>> [0 for _ in range(10)]

You could use:

>>> [0 for _ in repeat(10)]

"""
repeat = partial(itertools.repeat, None)


# ==============================================================================
class DataLims:
    """A class containing the extrema of all quantities stored in the mesh

    Attributes
    ----------
    - pos:  x,y,z   min,max
    - vel:  u,v,w   min,max
    - pres: p       min,max
    - temp: T       min,max
    - scal: s_i     min,max

    """

    def __init__(self, elements):
        self._variables = ("pos", "vel", "pres", "temp", "scal")

        aggregated_lims = reduce(self._lims_aggregator, elements)
        for var in self._variables:
            agg_lims_var = aggregated_lims[var]
            # set minimum, maximum of variables as a nested tuple
            setattr(self, var, tuple(zip(*agg_lims_var)))

        # prevent further mutation of attributes via __setattr__
        self._initialized = True

    def __repr__(self):
        return dedent(
            f"""\
          * x:         {self.pos[0]}
          * y:         {self.pos[1]}
          * z:         {self.pos[2]}"""
        )

    def __setattr__(self, name, value):
        if hasattr(self, "_initialized") and self._initialized:
            raise AttributeError(f"Setting attribute {name} is not permitted")
        else:
            super().__setattr__(name, value)

    def _lims_per_element(self, elem):
        """Get local limits for a given element."""
        if isinstance(elem, dict):
            return elem

        axis = (1, 2, 3)
        elem_lims = {
            var: (getattr(elem, var).min(axis), getattr(elem, var).max(axis))
            for var in self._variables
        }
        return elem_lims

    def _lims_aggregator(self, elem1, elem2):
        """Reduce local limits to global limits."""
        l1 = self._lims_per_element(elem1)
        l2 = self._lims_per_element(elem2)

        aggregated_lims = {
            var: (
                np.minimum(l1[var][0], l2[var][0]),
                np.maximum(l1[var][1], l2[var][1]),
            )
            for var in self._variables
        }
        return aggregated_lims


# ==============================================================================
class Elem:
    """A class containing one hexahedral element of Nek5000/SIMSON flow
    field.

    Parameters
    ----------
    var : iterable
        Iterable of integers of size 5, indicating how many variables are to be initialized
    lr1 : iterable
        Iterable of integers of size 3, defining the shape of an element as ``(lx, ly, lz)``
    nbc : int
        Number of boundary conditions
    dtype : str
        Floating point data type. Typical values are 'f4' or 'float32' for
        single precision, 'f8' or 'float64' for double precision

    """

    def __init__(self, var, lr1, nbc, dtype="float64"):
        #                    x,y,z   lz      ly      lx
        self.pos = np.zeros((3, lr1[2], lr1[1], lr1[0]), dtype=dtype)
        #                    one per edge
        self.curv = np.zeros((12, 5), dtype=dtype)
        #             curvature type
        self.ccurv = ["" for _ in repeat(12)]
        #                    u,v,w   lz      ly      lx
        self.vel = np.zeros((3, lr1[2], lr1[1], lr1[0]), dtype=dtype)
        #                    p       lz      ly      lx
        self.pres = np.zeros((var[2], lr1[2], lr1[1], lr1[0]), dtype=dtype)
        #                    T       lz      ly      lx
        self.temp = np.zeros((var[3], lr1[2], lr1[1], lr1[0]), dtype=dtype)
        #                    s_i     lz      ly      lx
        self.scal = np.zeros((var[4], lr1[2], lr1[1], lr1[0]), dtype=dtype)
        #                    list of 8 parameters, one per face
        #                    one column for velocity, one for temperature, and one for each scalar
        self.bcs = np.zeros((nbc, 6), dtype="U3, i4, i4" + f", {dtype}" * 5)

    def __repr__(self):
        message = f"<elem centered at {self.centroid}>"
        return message

    @property
    def centroid(self):
        return self.pos.mean(axis=(1, 2, 3))


# ==============================================================================
class HexaData:
    """A class containing data related to a hexahedral mesh"""

    def __init__(self, ndim, nel, lr1, var, nbc=0, dtype="float64"):
        self.ndim = ndim
        self.nel = nel
        self.ncurv = []
        self.nbc = nbc
        self.var = var
        self.lr1 = lr1
        self.time = []
        self.istep = []
        self.wdsz = []
        self.endian = []
        if isinstance(dtype, type):
            # For example np.float64 -> "float64"
            dtype = dtype.__name__

        self.elem = [Elem(var, lr1, nbc, dtype) for _ in repeat(nel)]
        self.elmap = np.linspace(1, nel, nel, dtype=np.int32)

    def __repr__(self):
        representation = dedent(
            f"""\
        <pymech.core.HexaData>
        Dimensions:    {self.ndim}
        Precision:     {self.wdsz} bytes
        Mesh limits:\n{indent(repr(self.lims), " "*10)}
        Time:
          * time:      {self.time}
          * istep:     {self.istep}
        Elements:
          * nel:       {self.nel}
          * elem:      [{self.elem[0]}
                        ...
                        {self.elem[-1]}]
        """
        )

        return representation

    @property
    def lims(self):
        return DataLims(self.elem)

