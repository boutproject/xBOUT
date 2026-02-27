"""
Fast lazy-loading of BOUT++ multi-file datasets.

Overview
--------
BOUT++ writes output distributed across many NetCDF files, one per processor.
The standard xarray.open_mfdataset opens every file to read metadata before
constructing the dataset, which is slow when there are hundreds of files on a
parallel filesystem where each file open incurs a metadata server round-trip.

This module avoids that overhead by reading metadata from a single file and
constructing a lazy dask-backed xarray Dataset without opening any other files.
Data is only read from disk when explicitly requested via .compute() or .load().

Method
------
All BOUT++ output files share the same array shapes and metadata. The processor
layout (NXPE x NYPE grid) and array dimensions are read from the first file
(BOUT.dmp.0.nc). From this, the slice of the global array stored in each
processor's file is determined, accounting for MXG/MYG guard cells.

A dask task graph is constructed as a Python dict, with one entry per processor
file per variable. Each task uses dask's internal getter() function, which
calls __getitem__ on a LazyFileArray object. LazyFileArray defers all file I/O
to h5py, reading only the requested hyperslab when the task is executed.

Slice fusion
------------
Dask's _optimize_slices pass (dask/array/optimization.py) recognizes chained
getter() tasks and fuses them into a single getter() call with composed slices.
Each variable's task graph uses two chained tasks per file:

    lazy task:  LazyFileArray object
    slice task: (getter, lazy_task_key, boundary_slices, False, False)

The boundary slice removes MXG/MYG guard cells from the raw file data as needed,
depending on keep_xboundaries and keep_yboundaries.
When the user slices the resulting DataArray, dask fuses the user slice with
the boundary slice into a single composed slice, which is passed through
getter() to LazyFileArray.__getitem__() and then directly to h5py as an HDF5
hyperslab selection. This means only the requested bytes are read from disk,
with a single file open per chunk per compute() call.

Usage
-----
    ds = lazy_open_boutdataset('/path/to/BOUT/dmp/files/')
    ds = lazy_open_boutdataset('/path/to/files/', keep_xboundaries=True)

    # Data is not read until here:
    data = ds['Ne'].isel(t=slice(10, 20)).compute()
"""

import xarray as xr
import dask
import numpy as np
import os
import h5py


class LazyFileArray:
    """Presents a numpy-like interface that defers reads to HDF5 hyperslabs."""

    def __init__(self, filepath, varname: str, shape, dtype, info: bool = False):
        """
        filepath: str or Path
            Full path to the file
        varname: str
            Name of the array to read
        shape: tuple
            The shape of the array in the file
        dtype
            Type of the elements in the array (e.g. numpy.float64)
        info: bool
            Print debugging information on read?
        """
        self.filepath = filepath
        self.varname = varname
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.info = info

    def __getitem__(self, slices):
        """
        Read data hyperslice from file.
        No cacheing is performed, so repeated access
        will open and read the file again.

        Uses h5py to access data because it has lower overhead
        when opening and closing files than NetCDF.
        """
        if self.info:
            print(f"Reading {self.filepath}:{self.varname}:{slices}")
        with h5py.File(self.filepath, "r") as f:
            return f[self.varname][slices]


from dask.array.core import getter


def make_chunkinfo(
    metadata: dict, keep_xboundaries: bool = True, keep_yboundaries: bool = True
):
    """
    Identify processor layout and the array slices to be extracted
    from each file. Handles single and double-null configurations.

    metadata: dict
        Dictionary of scalars read from the first processor's file
    keep_xboundaries: bool
        Keep the MXG cells on inner and outer X boundaries
    keep_yboundaries: bool
        Keep MYG cells on all Y boundaries (upper and lower targets if DN)

    Returns a dict that is used in make_lazy_array.
    """
    NXPE = metadata["NXPE"]
    NYPE = metadata["NYPE"]

    MXG = metadata["MXG"]
    MYG = metadata["MYG"]

    MXSUB = metadata["MXSUB"]
    MYSUB = metadata["MYSUB"]

    # Double null if it has upper legs
    is_double_null = metadata["jyseps2_1"] != metadata["jyseps1_2"]

    # Number of processors before the upper targets
    nyproc_inner = metadata["ny_inner"] // MYSUB

    # Size of the (x,y) array in each file
    nxsub = MXSUB + 2 * MXG
    nysub = MYSUB + 2 * MYG

    # Indices and size of each file's chunk
    xchunks = []
    xslices = []
    for i in range(NXPE):
        xslice = slice(
            0 if (i == 0 and keep_xboundaries) else MXG,  # Skip guard cells
            nxsub if (i == (NXPE - 1) and keep_xboundaries) else nxsub - MXG,
        )
        xchunks.append(xslice.stop - xslice.start)
        xslices.append(xslice)
    xchunks = tuple(xchunks)

    ychunks = []
    yslices = []
    for j in range(NYPE):
        yslice = slice(
            (
                0
                if (
                    keep_yboundaries
                    and (
                        j == 0  # Lower inner target
                        or (is_double_null and j == nyproc_inner)
                    )  # Upper outer target
                )
                else MYG
            ),  # Skip guard cells
            (
                nysub
                if (
                    keep_yboundaries
                    and (j == (NYPE - 1) or (is_double_null and j == nyproc_inner - 1))
                )
                else (nysub - MYG)
            ),
        )
        ychunks.append(yslice.stop - yslice.start)
        yslices.append(yslice)
    ychunks = tuple(ychunks)

    return {
        "NXPE": NXPE,  # Number of processors in X
        "NYPE": NYPE,  # Number of processors in Y
        "nxsub": nxsub,  # X size of the array in each file
        "nysub": nysub,  # Y size of the array in each file
        "xslices": xslices,  # List of slices in X
        "xchunks": xchunks,  # Tuple of slice sizes in X
        "yslices": yslices,  # List of slices in Y
        "ychunks": ychunks,  # Tuple of slice sizes in Y
    }


def make_lazy_array(
    datafilepath,
    ds,
    chunkinfo: dict,
    varname,
    prefix: str = "BOUT.dmp",
    info: bool = False,
):
    """
    Creates a lazy-loaded array, gathering data
    from a collection of NetCDF files.

    The array must have 'x' and 'y' dimensions but can
    have an arbitrary number of other dimensions.

    datafilepath : str or Path
        Directory containing BOUT.dmp.*.nc files
    ds : xarray.DataSet
        DataSet from one file
    chunkinfo : dict
        Describes processor layouts
    varname : str
        Name of the variable to read
    """
    NXPE = chunkinfo["NXPE"]
    NYPE = chunkinfo["NYPE"]
    xslices = chunkinfo["xslices"]
    yslices = chunkinfo["yslices"]
    xchunks = chunkinfo["xchunks"]
    ychunks = chunkinfo["ychunks"]

    # Get shape and type of the array in one file.
    # These are assumed to be the same for all files
    dtype = ds[varname].dtype
    file_shape = ds[varname].shape

    # Find x and y dimension indices
    xdim = ds[varname].dims.index("x")
    ydim = ds[varname].dims.index("y")
    ndims = len(file_shape)

    # Check x and y dimension sizes
    assert file_shape[xdim] == chunkinfo["nxsub"]
    assert file_shape[ydim] == chunkinfo["nysub"]

    # The name serves two purposes:
    # 1. Graph key prefix — it's the first element of every task key tuple
    #    (name, i, j, ...). Dask uses these keys to identify tasks in the graph,
    #    so the name must be unique across all arrays in a computation to avoid
    #    accidental key collisions between arrays that would cause one array's
    #    tasks to silently substitute for another's.
    # 2. Cache/fusion identity — when dask optimizes or fuses graphs, arrays with
    #    the same name are assumed to be identical. If two Array objects share
    #    a name, dask will treat them as the same array and only compute it once
    #    in a joint dask.compute() call. This is the mechanism behind deduplication.
    name = f"load-{varname}-{dask.base.tokenize(datafilepath, varname, chunkinfo)}"

    # Create a dict of tasks.
    dsk = {}
    for i in range(NXPE):
        xslice = xslices[i]
        for j in range(NYPE):
            yslice = yslices[j]

            filepath = os.path.join(datafilepath, f"{prefix}.{j * NXPE + i}.nc")
            # Create a lazy-loaded array
            lazy = LazyFileArray(filepath, varname, file_shape, dtype, info=info)

            # Store the lazy object as a separate key.
            # lazy_name resolves to an object, not an array of data.
            # Dask only reads data when a task returns a numpy array.
            lazy_name = (f"lazy-{name}", i, j)
            dsk[lazy_name] = lazy

            # The integer indices in the task key tuple
            # (name, i0, i1, i2, i3) directly map to chunk positions
            chunkpos = tuple(
                i if d == xdim else j if d == ydim else 0 for d in range(ndims)
            )

            # Keep all dimensions but slice in x and y
            slices = [
                slice(None),
            ] * ndims
            slices[xdim] = xslice
            slices[ydim] = yslice
            slices = tuple(slices)

            # Slice the LazyFileArray to remove boundary cells.
            # Dask can fuse slices only if they use dask.array.core.getter
            # The optimization is in _optimize_slices implemented here:
            #   https://github.com/dask/dask/blob/main/dask/array/optimization.py#L94
            # getter then passes slices through to LazyFileArray so that only the
            # required data is read from disk.
            dsk[(name, *chunkpos)] = (
                getter,
                lazy_name,
                slices,  # Remove boundary cells
                False,  # asarray
                False,  # lock
            )
    # Chunk sizes. Use file_shape except in 'x' and 'y' dimensions
    chunks = [(size,) for size in file_shape]
    chunks[xdim] = xchunks
    chunks[ydim] = ychunks
    chunks = tuple(chunks)

    return dask.array.Array(dsk, name, chunks, dtype=dtype)


def lazy_open_boutdataset(
    datapath,
    keep_xboundaries: bool = False,
    keep_yboundaries: bool = False,
    is_restart: bool = False,
    info: bool = False,
    **kwargs,
):
    """
    Open a multi-file dataset by only opening one file.
    Dask chunks are created for all processors using the
    metadata read from the first file.

    datapath : str or Path
        Directory containing the BOUT++ data files

    keep_xboundaries : bool, optional
        If true, keep x-direction boundary cells (the cells past the
        physical edges of the grid, where boundary conditions are
        set); increases the size of the x dimension in the returned
        data-set. If false, trim these cells.

    keep_yboundaries : bool, optional
        If true, keep y-direction boundary cells (the cells past the
        physical edges of the grid, where boundary conditions are
        set); increases the size of the y dimension in the returned
        data-set. If false, trim these cells.

    """
    prefix = "BOUT.restart" if is_restart else "BOUT.dmp"

    # Open first file to read metadata
    ds = xr.open_dataset(os.path.join(datapath, f"{prefix}.0.nc"))

    # Extract all scalars as metadata
    metadata = {
        name: var.item() for name, var in ds.data_vars.items() if len(var.dims) == 0
    }

    # Identify processor layout and the array slices from each file
    chunkinfo = make_chunkinfo(
        metadata, keep_xboundaries=keep_xboundaries, keep_yboundaries=keep_yboundaries
    )

    # Process all data variables
    data_vars = {}
    for name, var in ds.data_vars.items():
        if "x" in var.dims and "y" in var.dims:
            # Array distributed over processors in x and y
            data_vars[name] = xr.DataArray(
                make_lazy_array(
                    datapath, ds, chunkinfo, name, prefix=prefix, info=info
                ),
                dims=var.dims,
                attrs=var.attrs,
            )
        elif len(var.dims) == 0:
            continue  # scalars already in metadata
        elif ("x" not in var.dims) and ("y" not in var.dims):
            # Take DataArray from first processor
            data_vars[name] = var
        else:
            # If only 'x' or only 'y' dimension then skip
            warnings.warn(
                f"Variable '{name}' has only one of x/y dimensions and will be skipped"
            )

    coords = {}
    if "t_array" in ds:
        coords["t"] = ds["t_array"].values

    # Create a global dataset
    return xr.Dataset(data_vars, coords=coords, attrs={"metadata": metadata})
