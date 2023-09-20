#!/usr/bin/python3

import sys
import struct
import itertools
import numpy as np
import xarray as xr
from functools import partial
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
    def __init__(self, nel, lr1):
        self.nel = nel
        self.lr1 = lr1

        self.elem = [Elem(lr1) for _ in repeat(nel)]


# ==============================================================================
class _xNekData(xr.backends.common.AbstractDataStore):
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

        data_vars["pressure"] = xr.Variable(ax, elem.pres[0])
        data_vars["temperature"] = xr.Variable(ax, elem.temp[0])

        return Frozen(data_vars)


if __name__ == "__main__":
    print()

    try:
        fName = sys.argv[1]
        f = open(fName, 'rb')
    except:
        print("Could not read file :(\n")
        exit()

    headData = f.read(132).split()

    try:
        polyOrder = tuple([int(x) for x in headData[2:5]])
    except:
        print("Invalid file :(\n")
        exit()

    precSize = int(headData[1])
    numElems = int(headData[5])
    solTime = float(headData[7])
    timeStep = int(headData[8])
    varList = str(headData[11])[2:-1]

    print("Polynomial order: ({0:2d}, {1:2d}, {2:2d})".format(*polyOrder))
    print("Number of elements: {0:<15d}".format(numElems))
    print("Solution time: {0:<11.5f}".format(solTime))
    print("Time-step count: {0:<15d}".format(timeStep))
    print("Variable list: {0:<5s}".format(varList))

    # Identifying endian encoding
    etagb = f.read(4)
    etagL = struct.unpack("<f", etagb)[0]
    etagL = int(etagL * 1e5) / 1e5
    etagB = struct.unpack(">f", etagb)[0]
    etagB = int(etagB * 1e5) / 1e5
    if etagL == 6.54321:
        emode = "<"
    elif etagB == 6.54321:
        emode = ">"
    else:
        print("Invalid endian :(\n")
        exit()

    # Set datatype of file
    if precSize == 4:
        rType = "f"
    elif precSize == 8:
        rType = "d"
    else:
        print("Invalid data precision :(\n")
        exit()

    fDType = emode + rType

    # Read element map for the file
    elMap = f.read(4 * numElems)
    elMap = struct.unpack(emode + numElems * "i", elMap)
    elMap = np.array(elMap, dtype=np.int32)

    ptsPerElem = np.prod(polyOrder)
    bytesPerElem = ptsPerElem * precSize

    # Create the HexaData class
    data = HexaData(numElems, polyOrder)


    def readData(dataArray):
        fi = f.read(bytesPerElem)
        fi = np.frombuffer(fi, dtype=fDType, count=ptsPerElem)

        elemShape = polyOrder[::-1]
        dataArray = fi.reshape(elemShape)


    if varList[0] == 'X':
        # Read X, Y, Z coordinates
        for elemInd in elMap:
            inpElem = data.elem[elemInd - 1]
            for iDim in range(3):
                #readData(f, inpElem.pos[2-iDim], polyOrder, bytesPerElem, ptsPerElem, fDType)
                readData(inpElem.pos[iDim])
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

        if gelMap[0] != elMap[0]:
            print("Grid file is incompatible :(\n")
            exit()

        for elemInd in elMap:
            inpElem = data.elem[elemInd - 1]
            for iDim in range(3):
                #readData(g, inpElem.pos[iDim], polyOrder, bytesPerElem, ptsPerElem, fDType)
                readData(inpElem.pos[iDim])

        g.close()

    # Read U, V, W velocities
    for elemInd in elMap:
        inpElem = data.elem[elemInd - 1]
        for iDim in range(3):
            #readData(f, inpElem.vel[iDim], polyOrder, bytesPerElem, ptsPerElem, fDType)
            readData(inpElem.vel[iDim])

    # Read pressure
    for elemInd in elMap:
        inpElem = data.elem[elemInd - 1]
        #readData(f, inpElem.pres, polyOrder, bytesPerElem, ptsPerElem, fDType)
        readData(inpElem.pres)

    # Read temperature
    for elemInd in elMap:
        inpElem = data.elem[elemInd - 1]
        #readData(f, inpElem.temp, polyOrder, bytesPerElem, ptsPerElem, fDType)
        readData(inpElem.temp)

    f.close()

    print(data.elem[0].pos[0])
    exit()

    # Convert to xarray
    elements = data.elem
    elemData = [_xNekData(elem) for elem in elements]
    elemDSet = [xr.Dataset.load_store(store).set_coords(store.axes) for store in elemData]

    #ds = xr.combine_by_coords(elemDSet, combine_attrs="drop")

    print()
