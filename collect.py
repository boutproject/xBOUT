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

    filepaths, datasets = _open_all_dump_files(path, prefix, chunks=chunks)

    ds_grid, concat_dims = _organise_dump_files(filepaths, datasets)

    ds_grid = _trim(ds_grid, concat_dims,
                    guards={'x': mxg, 'y': myg}, ghosts={'x': mxg, 'y': myg},
                    keep_guards={'x': xguards, 'y': yguards})

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


def _open_all_dump_files(path, prefix, chunks):

    filetype = determine_format(file)

    filepaths = glob.glob(os.path.join(path, prefix + "*.nc"))

    datasets = []
    for file in filepaths:
        ds = xr.open_dataset(file, engine=filetype, chunks=chunks)
        datasets.append(ds)

    return filepaths, datasets


def _organise_dump_files(filepaths, datasets):
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

    nxpe, nype = _read_parallelisation(filepaths[0])

    # This is where our knowledge of how BOUT does its parallelization is actually used
    ds_grid, concat_dims = _construct_bout_ds_grid(filepaths, datasets, nxpe, nype)

    return ds_grid, concat_dims


def _construct_bout_ds_grid(filepaths, datasets, nxpe, nype):
    # For now assume that there is no splitting along t, and that all files are in current dir
    # Generalise later
    prefix = './BOUT.dmp.'

    concat_dims = []
    if nxpe > 0:
        concat_dims.append('x')
    if nxpe > 0:
        concat_dims.append('y')

    dataset_pieces = dict(zip(filepaths, datasets))

    # BOUT names files as num = nxpe*i + j
    # So use this knowledge to arrange files in the right shape for concatenation
    ds_grid = np.empty((nxpe, nype), dtype='object')
    for i in range(nxpe):
        for j in range(nype):
            file_num = (i + nxpe * j)
            filename = prefix + str(file_num) + '.nc'
            ds_grid[i, j] = dataset_pieces[filename]

    return ds_grid.squeeze(), concat_dims


def _trim(ds_grid, concat_dims, guards, ghosts, keep_guards):
    """
    Trims all ghost and guard cells off each dataset in ds_grid to prepare for concatenation.
    """
    for index, ds in np.ndenumerate(ds_grid):
        # Determine how many cells to trim off each dimension
        lower, upper = {}, {}
        for dim in concat_dims:
            lower[dim] = ghosts[dim]
            upper[dim] = -ghosts[dim]

            # If ds is at edge of grid trim guard cells instead of ghost cells
            dim_axis = concat_dims.index(dim)
            dim_max = ds_grid.shape[dim_axis]
            if keep_guards[dim]:
                if index[dim_axis] == 0:
                    lower[dim] = None
                if index[dim_axis] == dim_max:
                    upper[dim] = None
            else:
                if index[dim_axis] == 0:
                    lower[dim] = guards[dim]
                if index[dim_axis] == dim_max:
                    upper[dim] = -guards[dim]

        # Actually trim the dataset in-place
        selection = {dim: slice(lower[dim], upper[dim], None) for dim in ds.dims}
        ds_grid[index] = ds.isel(**selection)

    return ds_grid
