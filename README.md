# nek2hdf5 - Convert Nek5000 files to HDF5

The Python script converts Nek5000/NekRS solutions on rectilinear meshes into 3D NumPy arrays and writes them into HDF5 files.
These 3D arrays can then be easily post-processed in Python.
The understanding of Nek solution files with their elements and element maps was provided by the [PyMech](https://github.com/eX-Mech/pymech) package.
As a result, the basic elements of reading Nek solution files were derived from this package.
However, the reading of data has been highly optimized to use as little memory as possible,
allowing the script to process files larger than 1 TB with a memory footprint of around one-fourth the file-size.

The following Python modules are necessary for the script to run.

* h5py
* numpy

## Running the script

Some Nek solution files contain the grid data, while other do not.
However, each HDF5 file written by the script contains the 1D grid coordinates
(as mentioned earlier, the script is used only to process rectilinear mesh data).
If the solution file contains the grid data, only the filename is required as argument:

``python3 nek2hdf5.py sol0.f00001``

If the grid data is not there in the solution file however,
a file containing the grid data must be supplied as second argument:

``python3 nek2hdf5.py sol0.f00015 sol0.f00001``

Before performing this, the number of elements along x, y and z (namely ``nelx``, ``nely`` and ``nelz``), must be specifed within the script.

## Memory optimization vs speed

Additionally if memory optimization is not necessary, then the variable ``writeDisk`` in the script must be set to ``False``.
This will speed up the script by more than a factor of 2.
Finally, if one wishes to see the memory foot-print of the script, the variable ``traceMem`` can be set to ``True``.
However, this uses the ``tracemalloc`` module of Python, which can be memory intensive.
Therefore this flag is usually used for development purposes only.
