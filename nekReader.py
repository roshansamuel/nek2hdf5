import sys
import struct
import numpy as np

def readnek(fname, dtype="float64"):
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

    # read geometry
    #pos = [np.zeros((3, polyOrder[2], polyOrder[1], polyOrder[0]), dtype="float64") for i in range(numElems)]
    pos = np.zeros((numElems, 3, polyOrder[2], polyOrder[1], polyOrder[0]), dtype="float64")
    for iel in elmap:
        #el = pos[iel - 1]
        el = pos[iel - 1, ...]
        for idim in range(3):
            el[idim, ...] = read_file_into_data()

    # read velocity
    #vel = [np.zeros((3, polyOrder[2], polyOrder[1], polyOrder[0]), dtype="float64") for i in range(numElems)]
    vel = np.zeros((numElems, 3, polyOrder[2], polyOrder[1], polyOrder[0]), dtype="float64")
    for iel in elmap:
        #el = vel[iel - 1]
        el = vel[iel - 1, ...]
        for idim in range(3):
            el[idim, ...] = read_file_into_data()

    # read pressure
    #prs = [np.zeros((1, polyOrder[2], polyOrder[1], polyOrder[0]), dtype="float64") for i in range(numElems)]
    prs = np.zeros((numElems, 1, polyOrder[2], polyOrder[1], polyOrder[0]), dtype="float64")
    for iel in elmap:
        #el = prs[iel - 1]
        el = prs[iel - 1, ...]
        el[...] = read_file_into_data()

    # read temperature
    #tmp = [np.zeros((1, polyOrder[2], polyOrder[1], polyOrder[0]), dtype="float64") for i in range(numElems)]
    tmp = np.zeros((numElems, 1, polyOrder[2], polyOrder[1], polyOrder[0]), dtype="float64")
    for iel in elmap:
        #el = tmp[iel - 1]
        el = tmp[iel - 1, ...]
        el[...] = read_file_into_data()

    # close file
    infile.close()

    pos = np.swapaxes(np.array(pos), 2, 4)
    vel = np.swapaxes(np.array(vel), 2, 4)
    prs = np.swapaxes(np.array(prs), 2, 4)
    tmp = np.swapaxes(np.array(tmp), 2, 4)

    print(pos.shape)
    print(vel.shape)
    print(prs.shape)
    print(tmp.shape)
    print(vel[-1, 0, -1, -1, :])
    print(vel[-100, 0, 0, -1, :])
    exit()

    # output
    return np.array(pos)


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

