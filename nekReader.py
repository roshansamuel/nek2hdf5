import sys
import struct
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

    Nx = nelx*(polyOrder[0] - 1) + 1
    Ny = nely*(polyOrder[1] - 1) + 1
    Nz = nelz*(polyOrder[2] - 1) + 1

    #x = np.zeros((Nx, Ny, Nz), dtype="float64")
    #for i in range(nelx):
    #    for j in range(nely):
    #        for k in range(nelz):
    #            elNum = i*nelx + j*nely + k
    #print(Nx, Ny, Nz)
    #exit()

    # Swap x and z axes correctly
    fData = np.swapaxes(fData, 2, 4)

    # Bring all data to final axis
    fData = np.swapaxes(np.swapaxes(np.swapaxes(fData, 1, 2), 2, 3), 3, 4)

    xVel = np.zeros((Nx, Ny, Nz), dtype="float64")
    yVel = np.zeros((Nx, Ny, Nz), dtype="float64")
    zVel = np.zeros((Nx, Ny, Nz), dtype="float64")

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

                xVel[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz, 3]
                yVel[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz, 4]
                zVel[strx:endx, stry:endy, strz:endz] = fData[elNum, :lenx, :leny, :lenz, 5]

    print(xVel[-3:, -3:, -3:])
    exit()
    # Reshape to just the data values
    fData = fData.reshape(numElems*np.prod(polyOrder), 8)

    print(fData.shape)
    fData = np.unique(fData, axis=0)
    print(fData[:3, :])
    print(fData.shape)
    exit(0)

    x = fData[0, ...].flatten()
    y = fData[1, ...].flatten()
    z = fData[2, ...].flatten()
    u = fData[3, ...].flatten()
    v = fData[4, ...].flatten()
    w = fData[5, ...].flatten()
    p = fData[6, ...].flatten()
    t = fData[7, ...].flatten()

    print(u.shape)

    #print(fData.shape)
    #print(fData[-1, 3, -1, -1, :])
    #print(fData[-100, 3, 0, -1, :])
    exit()

    # output
    return np.array(fData)


if __name__ == "__main__":
    try:
        fName = sys.argv[1]
        print("Reading file " + fName)
    except:
        print("Could not read file :(\n")
        exit()

    ds = readnek(fName)
    x = ds[:,0,:,:,:]
    y = ds[:,1,:,:,:]
    z = ds[:,2,:,:,:]

    x = np.unique(x)
    y = np.unique(y)
    z = np.unique(z)

    np.savetxt("grid_x.dat", x)
    np.savetxt("grid_y.dat", y)
    np.savetxt("grid_z.dat", z)

