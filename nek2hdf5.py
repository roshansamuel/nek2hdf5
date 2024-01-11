import os
import gc
import sys
import struct
import h5py as hp
import numpy as np

# Optimize memory (sacrifice speed)
writeDisk = False

# Output memory trace (for debug only)
traceMem = False

if traceMem:
    import tracemalloc

nelx, nely, nelz = 100, 100, 64
ddtype = "float64"

def readnek(fname):
    global ddtype
    global nelx, nely, nelz

    # Open input file
    try:
        infile = open(fname, "rb")
    except OSError as e:
        print(f"I/O error ({e.errno}): {e.strerror}")
        return -1

    # Open output file
    try:
        outDest = sys.argv[3]
        h5fName = outDest + ".h5"
    except:
        h5fName = fName + ".h5"

    ############# READ HEADER #############

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
        ddtype = "float32"
    elif precSize == 8:
        rType = "d"
        ddtype = "float64"

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

    bytes_elem = ptsPerElem * precSize

    if (nelx*nely*nelz != numElems):
        print("ERROR: Element count is inconsistent")
        exit(0)

    ############# FINISHED HEADER - BEGIN DATA WRITE #############

    # Create the solution hdf5 file
    f = hp.File(h5fName, "w")
    dset = f.create_dataset("Time", data = solTime)
    dset = f.create_dataset("nelm", data = np.array([nelx, nely, nelz]))
    if writeDisk:
        f.close()

    # This function used only when writeDisk is enabled
    def addDataset(dName, dSet):
        f = hp.File(h5fName, "a")
        dset = f.create_dataset(dName, data = dSet)
        f.close()

    ############# FINISHED INITIAL - BEGIN DATA READ #############

    Nx = nelx*(polyOrder[0] - 1) + 1
    Ny = nely*(polyOrder[1] - 1) + 1
    Nz = nelz*(polyOrder[2] - 1) + 1

    # Function to read data from one element
    def read_elem_into_data(ifile):
        fi = ifile.read(bytes_elem)
        fi = np.frombuffer(fi, dtype=emode + rType, count=ptsPerElem)

        elem_shape = polyOrder[::-1]  # lz, ly, lx
        data_var = fi.reshape(elem_shape)

        return data_var

    # Function to read one scalar variable from file
    def read_file_into_data(ifile):
        fData = np.zeros((numElems, polyOrder[2], polyOrder[1], polyOrder[0]), dtype=ddtype)

        for iel in elmap:
            el = fData[iel - 1, ...]
            el[...] = read_elem_into_data(ifile)

        fData = np.swapaxes(fData, 1, 3)

        return fData

    # Function to read one vector variable from file
    def read_file_into_data_3c(ifile):
        fData = np.zeros((numElems, 3, polyOrder[2], polyOrder[1], polyOrder[0]), dtype=ddtype)

        for iel in elmap:
            el = fData[iel - 1, ...]
            for idim in range(3):
                el[idim, ...] = read_elem_into_data(ifile)

        # Get X and Z indices in correct places first
        fData = np.swapaxes(fData, 2, 4)
        if writeDisk:
            # Separate out the components by bringing the components index first
            fData = np.swapaxes(fData, 0, 1)

            # Write the 3 components into temp arrays
            np.save(fname + "_temp_x", fData[0, ...])
            np.save(fname + "_temp_y", fData[1, ...])
            np.save(fname + "_temp_z", fData[2, ...])
        else:
            fData = np.swapaxes(np.swapaxes(np.swapaxes(fData, 1, 2), 2, 3), 3, 4)
            return fData

    # Function to transfer scalar variable to 3D array
    def transfer_data(fData):
        oData = np.zeros((Nx, Ny, Nz), dtype=ddtype)

        for elz in range(nelz):
            strz = elz * (polyOrder[2] - 1)
            lenz = polyOrder[2] - 1
            if elz == nelz - 1:
                lenz = polyOrder[2]
            endz = strz + lenz

            for ely in range(nely):
                stry = ely * (polyOrder[1] - 1)
                leny = polyOrder[1] - 1
                if ely == nely - 1:
                    leny = polyOrder[1]
                endy = stry + leny

                for elx in range(nelx):
                    elNum = nelx*nely*elz + nelx*ely + elx

                    strx = elx * (polyOrder[0] - 1)
                    lenx = polyOrder[0] - 1
                    if elx == nelx - 1:
                        lenx = polyOrder[0]
                    endx = strx + lenx

                    oData[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz]

        return oData

    # Function to transfer one component of a vector variable to 3D array
    def transfer_data_3c(cInd, fData=None):
        if writeDisk:
            if cInd == 0:
                fData = np.load(fname + "_temp_x.npy")
            elif cInd == 1:
                fData = np.load(fname + "_temp_y.npy")
            elif cInd == 2:
                fData = np.load(fname + "_temp_z.npy")

        oData = np.zeros((Nx, Ny, Nz), dtype=ddtype)

        for elz in range(nelz):
            strz = elz * (polyOrder[2] - 1)
            lenz = polyOrder[2] - 1
            if elz == nelz - 1:
                lenz = polyOrder[2]
            endz = strz + lenz

            for ely in range(nely):
                stry = ely * (polyOrder[1] - 1)
                leny = polyOrder[1] - 1
                if ely == nely - 1:
                    leny = polyOrder[1]
                endy = stry + leny

                for elx in range(nelx):
                    elNum = nelx*nely*elz + nelx*ely + elx

                    strx = elx * (polyOrder[0] - 1)
                    lenx = polyOrder[0] - 1
                    if elx == nelx - 1:
                        lenx = polyOrder[0]
                    endx = strx + lenx

                    if writeDisk:
                        oData[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz]
                    else:
                        oData[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz, cInd]

        if writeDisk:
            if cInd == 0:
                os.remove(fname + "_temp_x.npy")
            elif cInd == 1:
                os.remove(fname + "_temp_y.npy")
            elif cInd == 2:
                os.remove(fname + "_temp_z.npy")

        return oData

    # Function to transfer x-grid to 1D array
    def transfer_xgrid(fData=None):
        if writeDisk:
            fData = np.load(fname + "_temp_x.npy")
        xPos = np.zeros(Nx, dtype=ddtype)

        ely, elz = 1, 1
        for elx in range(nelx):
            elNum = nelx*nely*elz + nelx*ely + elx

            strx = elx * (polyOrder[0] - 1)
            lenx = polyOrder[0] - 1
            if elx == nelx - 1:
                lenx = polyOrder[0]
            endx = strx + lenx

            if writeDisk:
                xPos[strx:endx] = fData[elNum, :lenx, 0, 0]
            else:
                xPos[strx:endx] = fData[elNum, :lenx, 0, 0, 0]

        if writeDisk:
            os.remove(fname + "_temp_x.npy")

        return xPos

    # Function to transfer y-grid to 1D array
    def transfer_ygrid(fData=None):
        if writeDisk:
            fData = np.load(fname + "_temp_y.npy")
        yPos = np.zeros(Ny, dtype=ddtype)

        elx, elz = 1, 1
        for ely in range(nely):
            elNum = nelx*nely*elz + nelx*ely + elx

            stry = ely * (polyOrder[1] - 1)
            leny = polyOrder[1] - 1
            if ely == nely - 1:
                leny = polyOrder[1]
            endy = stry + leny

            if writeDisk:
                yPos[stry:endy] = fData[elNum, 0, :leny, 0]
            else:
                yPos[stry:endy] = fData[elNum, 0, :leny, 0, 1]

        if writeDisk:
            os.remove(fname + "_temp_y.npy")

        return yPos

    # Function to transfer z-grid to 1D array
    def transfer_zgrid(fData=None):
        if writeDisk:
            fData = np.load(fname + "_temp_z.npy")
        zPos = np.zeros(Nz, dtype=ddtype)

        elx, ely = 1, 1
        for elz in range(nelz):
            elNum = nelx*nely*elz + nelx*ely + elx

            strz = elz * (polyOrder[2] - 1)
            lenz = polyOrder[2] - 1
            if elz == nelz - 1:
                lenz = polyOrder[2]
            endz = strz + lenz

            if writeDisk:
                zPos[strz:endz] = fData[elNum, 0, 0, :lenz]
            else:
                zPos[strz:endz] = fData[elNum, 0, 0, :lenz, 2]

        if writeDisk:
            os.remove(fname + "_temp_z.npy")

        return zPos

    # read geometry
    print("Processing grid data")

    if varList[0] == 'X':
        # Read XYZ Data
        if writeDisk:
            read_file_into_data_3c(infile)

            oData = transfer_xgrid()
            addDataset("X", oData)

            oData = transfer_ygrid()
            addDataset("Y", oData)

            oData = transfer_zgrid()
            addDataset("Z", oData)
        else:
            fData = read_file_into_data_3c(infile)

            oData = transfer_xgrid(fData)
            dset = f.create_dataset("X", data = oData)

            oData = transfer_ygrid(fData)
            dset = f.create_dataset("Y", data = oData)

            oData = transfer_zgrid(fData)
            dset = f.create_dataset("Z", data = oData)
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

        # Read XYZ Data
        if writeDisk:
            read_file_into_data_3c(g)

            oData = transfer_xgrid()
            addDataset("X", oData)

            oData = transfer_ygrid()
            addDataset("Y", oData)

            oData = transfer_zgrid()
            addDataset("Z", oData)
        else:
            fData = read_file_into_data_3c(g)

            oData = transfer_xgrid(fData)
            dset = f.create_dataset("X", data = oData)

            oData = transfer_ygrid(fData)
            dset = f.create_dataset("Y", data = oData)

            oData = transfer_zgrid(fData)
            dset = f.create_dataset("Z", data = oData)

        g.close()

    if writeDisk:
        # Free up memory
        del oData
        gc.collect()

    # Read Vx-Vy-Vz data
    print("Processing velocity data")
    if writeDisk:
        read_file_into_data_3c(infile)

        # Transfer Vx data
        oData = transfer_data_3c(0)
        addDataset("U", oData)
        del oData
        gc.collect()

        # Transfer Vy data
        oData = transfer_data_3c(1)
        addDataset("V", oData)
        del oData
        gc.collect()

        # Transfer Vz data
        oData = transfer_data_3c(2)
        addDataset("W", oData)
        del oData
        gc.collect()
    else:
        fData = read_file_into_data_3c(infile)

        # Transfer Vx data
        oData = transfer_data_3c(0, fData)
        dset = f.create_dataset("U", data = oData)

        # Transfer Vy data
        oData = transfer_data_3c(1, fData)
        dset = f.create_dataset("V", data = oData)

        # Transfer Vz data
        oData = transfer_data_3c(2, fData)
        dset = f.create_dataset("W", data = oData)

    # Read pressure
    print("Processing pressure data")
    fData = read_file_into_data(infile)
    oData = transfer_data(fData)
    if writeDisk:
        addDataset("P", oData)

        del fData
        del oData
        gc.collect()
    else:
        dset = f.create_dataset("P", data = oData)

    # Read temperature
    print("Processing temperature data")
    fData = read_file_into_data(infile)
    oData = transfer_data(fData)
    if writeDisk:
        addDataset("T", oData)

        del fData
        del oData
        gc.collect()
    else:
        dset = f.create_dataset("T", data = oData)

    # close file
    infile.close()

    if not writeDisk:
        f.close()

    print("Finished writing output file: ", h5fName)


if __name__ == "__main__":
    try:
        fName = sys.argv[1]
        print("Reading file " + fName)
    except:
        print("Could not read file :(\n")
        exit()

    if traceMem:
        tracemalloc.start()

    readnek(fName)

    if traceMem:
        cblocks, pblocks = tracemalloc.get_traced_memory()
        maxmem = pblocks/(1024*1024)
        print("Maximum memory used is: ", np.round(maxmem, 2), " MB")
        tracemalloc.stop()

