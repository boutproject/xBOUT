"""
Contains an xarray + dask reimplementation of the BOUT++ collect function.

Original found at https://github.com/boutproject/BOUT-dev/blob/master/tools/pylib/boutdata/collect.py .
This version should be much faster (works in parallel), and more scalable (dask chunks limit memory usage).
"""

import xarray as xr
import numpy as np

import os
import sys
import glob

from xcollect.concatenate import concat_nd


def collect(varname, path='.', prefix='BOUT.dmp.', yguards=False, xguards=True,
            info=True, strict=False, lazy=False, chunks={}):
    """
    Collect a variable from a set of BOUT++ output files.

    Uses xarray + dask to load data lazily, parallelize operation and limit memory usage.

    Parameters
    ----------
    varname : str
        data variable to be collected
    path : str
    prefix : str

    xguards : bool
        Choice whether or not to keep domain guard cells in the x-direction.
    yguards : bool
        Choice whether or not to keep domain guard cells in the y-direction.
    info : bool
    strict : bool
    lazy : bool
    chunks : dict
        Dask chunks to split arrays into. Default is to load each dump file as one chunk.

    Returns
    -------
    var_da : dataarray containing collected variable
    """

    filepaths, datasets = open_all_dump_files(path, prefix, chunks=chunks)

    ds_grid, concat_dims = organise_dump_files(filepaths, datasets)

    ds_grid = trim_guard_cells(ds_grid, concat_dims, yguards, xguards)

    ds = concat_nd(ds_grid, concat_dims=concat_dims, data_vars='minimal')

    var_da = ds[varname]

    var_da = take_slice(var_da)

    if lazy:
        # Utilise xarray's lazy loading capabilities by returning a DataArray view to the data values.
        # Should always use lazy=True except for when old scripts are expecting an eager numpy array.
        return var_da
    else:
        # For backwards compatibility, return only the numpy array of data values
        # This will immediately load entire array into memory
        return var_da.values


def open_all_dump_files(path, prefix, chunks):

    filetype = determine_format(file)

    filepaths = glob.glob(os.path.join(path, prefix + "*.nc"))

    datasets = []
    for file in filepaths:
        ds = xr.open_dataset(file, engine=filetype, chunks=chunks)
        datasets.append(ds)

    return filepaths, datasets


def organise_dump_files(filepaths, datasets):
    """
    Arrange BOUT dump files into numpy ndarray structure which specifies the way they should be concatenated together.

    Parameters
    ----------
    filepaths
    datasets

    Returns
    -------
    ds_grid : numpy.ndarray
    concat_dims : list of str
    """

    nxpe, nype = read_parallelisation(filepaths[0])

    # This is where our knowledge of how BOUT does its parallelization is actually used
    ds_grid, concat_dims = construct_ds_grid(filepaths, datasets, nxpe, nype)

    return ds_grid, concat_dims
