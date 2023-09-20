import sys
import struct
import h5py as hp
import numpy as np

nelx, nely, nelz = 100, 100, 64

def readnek(fname, dtype="float64"):
    global nelx, nely, nelz

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

    bytes_elem = ptsPerElem * precSize

    def read_file_into_data():
        """Read binary file into an array attribute of ``data.elem``"""
        fi = infile.read(bytes_elem)
        fi = np.frombuffer(fi, dtype=emode + rType, count=ptsPerElem)

        elem_shape = polyOrder[::-1]  # lz, ly, lx
        data_var = fi.reshape(elem_shape)

        return data_var

    fData = np.zeros((numElems, 8, polyOrder[2], polyOrder[1], polyOrder[0]), dtype="float64")

    # read geometry
    for iel in elmap:
        el = fData[iel - 1, 0:3, ...]
        for idim in range(3):
            el[idim, ...] = read_file_into_data()

    # read velocity
    for iel in elmap:
        el = fData[iel - 1, 3:6, ...]
        for idim in range(3):
            el[idim, ...] = read_file_into_data()

    # read pressure
    for iel in elmap:
        el = fData[iel - 1, 6, ...]
        el[...] = read_file_into_data()

    # read temperature
    for iel in elmap:
        el = fData[iel - 1, 7, ...]
        el[...] = read_file_into_data()

    # close file
    infile.close()

    print("Transferring data to 3D arrays")

    Nx = nelx*(polyOrder[0] - 1) + 1
    Ny = nely*(polyOrder[1] - 1) + 1
    Nz = nelz*(polyOrder[2] - 1) + 1

    # Swap x and z axes correctly
    fData = np.swapaxes(fData, 2, 4)

    # Bring all data to final axis
    fData = np.swapaxes(np.swapaxes(np.swapaxes(fData, 1, 2), 2, 3), 3, 4)

    xPos = np.zeros(Nx, dtype="float64")
    yPos = np.zeros(Ny, dtype="float64")
    zPos = np.zeros(Nz, dtype="float64")

    xVel = np.zeros((Nx, Ny, Nz), dtype="float64")
    yVel = np.zeros((Nx, Ny, Nz), dtype="float64")
    zVel = np.zeros((Nx, Ny, Nz), dtype="float64")

    prsr = np.zeros((Nx, Ny, Nz), dtype="float64")
    tmpr = np.zeros((Nx, Ny, Nz), dtype="float64")

    writex, writey, writez = True, True, True
    for elz in range(nelz):
        strz = elz * (polyOrder[2] - 1)
        lenz = polyOrder[2] - 1
        if elz == nelz - 1:
            lenz = polyOrder[2]
        endz = strz + lenz

        writez = True
        for ely in range(nely):
            stry = ely * (polyOrder[1] - 1)
            leny = polyOrder[1] - 1
            if ely == nely - 1:
                leny = polyOrder[1]
            endy = stry + leny

            writey = True
            for elx in range(nelx):
                elNum = nelx*nely*elz + nelx*ely + elx

                strx = elx * (polyOrder[0] - 1)
                lenx = polyOrder[0] - 1
                if elx == nelx - 1:
                    lenx = polyOrder[0]
                endx = strx + lenx

                if writex: xPos[strx:endx] = fData[elNum, :lenx, 0, 0, 0]
                if writey: yPos[stry:endy] = fData[elNum, 0, :leny, 0, 1]
                if writez: zPos[strz:endz] = fData[elNum, 0, 0, :lenz, 2]

                xVel[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz, 3]
                yVel[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz, 4]
                zVel[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz, 5]

                prsr[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz, 6]
                tmpr[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz, 7]

                writey, writez = False, False

            writex = False
        writey = False

    # output
    return xPos, yPos, zPos, xVel, yVel, zVel, prsr, tmpr


if __name__ == "__main__":
    try:
        fName = sys.argv[1]
        print("Reading file " + fName)
    except:
        print("Could not read file :(\n")
        exit()

    x, y, z, u, v, w, p, t = readnek(fName)

    h5fName = fName + ".h5"
    print("Writing output file: ", h5fName)

    f = hp.File(h5fName, "w")

    dset = f.create_dataset("X", data = x)
    dset = f.create_dataset("Y", data = y)
    dset = f.create_dataset("Z", data = z)
    dset = f.create_dataset("U", data = u)
    dset = f.create_dataset("V", data = v)
    dset = f.create_dataset("W", data = w)
    dset = f.create_dataset("P", data = p)
    dset = f.create_dataset("T", data = t)

    f.close()

