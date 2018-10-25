"""
Contains an xarray + dask reimplementation of the BOUT++ collect function.

Original found at https://github.com/boutproject/BOUT-dev/blob/master/tools/pylib/boutdata/collect.py .
This version should be much faster (works in parallel), and more scalable (dask chunks limit memory usage).
Although it would be relatively simple to alter this to maintain backwards compatibility for all input arguments,
we choose not to do so.
"""

import xarray as xr
import numpy as np

import re
from pathlib import Path

from xcollect.concatenate import _concat_nd


# TODO account for BOUT++ output files potentially containing attributes which we want to keep

def collect(vars='all', datapath='./BOUT.dmp.*.nc',
            slices={}, yguards=False, xguards=False,
            info=True, chunks={}):
    """
    Collect a variable from a set of BOUT++ output files.

    Uses xarray + dask to load data lazily, parallelize operation and limit memory usage.

    Parameters
    ----------
    var : str, optional
        Data variable to be collected. If one of the variables in the files then returns the data
        values as an xarray.DataArray. If 'all' then returns view to all data as an xarray.DataSet.
    path : str, optional
    prefix : str, optional
    slices : dict, optional
        Slices to return from all data variables. Using xarray this slicing can always be done later though.
        Should be a dictionary of slice objects, with dimension names as keys,
        e.g. {'t': slice(100,None,None), 'x': 20}
    xguards : bool, optional
        Choice whether or not to keep domain guard cells in the x-direction.
    yguards : bool, optional
        Choice whether or not to keep domain guard cells in the y-direction.
    info : bool, optional
    strict : bool, optional
    chunks : dict, optional
        Dask chunks to split arrays into. Default is to load each dump file as one chunk.

    Returns
    -------
    ds : xarray.Dataset
        View to dataset containing all data variables requested
    """

    # TODO implement optional argument info
    # TODO change the path/prefix system to use a glob

    path = Path(datapath)

    filepaths, datasets = _open_all_dump_files(path, chunks=chunks)

    # Open just one file to read processor splitting
    ds = xr.open_dataset(str(filepaths[0]))
    nxpe, nype = ds['NXPE'].values, ds['NYPE'].values
    mxg, myg = ds['MXG'].values, ds['MYG'].values
    # TODO check that BOUT doesn't ever set the number of guards to be different to the number of ghosts
    mxguards, myguards = mxg, myg

    ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe, nype)

    # TODO work out how to get numbers of guard cells in each dimension from output files
    ds_grid = _trim(ds_grid, concat_dims,
                    guards={'x': mxguards, 'y': myguards}, ghosts={'x': mxg, 'y': myg},
                    keep_guards={'x': xguards, 'y': yguards})

    ds = _concat_nd(ds_grid, concat_dims=concat_dims, data_vars=['minimal']*len(concat_dims))

    # TODO Check that none of the chunk sizes are zero!

    # Utilise xarray's lazy loading capabilities by returning a DataSet/DataArray view to the data values.
    if vars == 'all':
        return ds.isel(**slices)
    else:
        return ds[vars].isel(**slices)


def _open_all_dump_files(path, chunks={}):
    """Determines filetypes and opens all dump files."""

    filetype = _check_filetype(path)

    filepaths = _expand_wildcards(path)

    # Default chunks={} is for each file to be one chunk
    datasets = [xr.open_dataset(file, engine=filetype, chunks=chunks) for file in filepaths]

    return filepaths, datasets


def _check_filetype(path):
    if path.suffix == '.nc':
        filetype = 'netcdf4'
    elif path.suffix == '.h5netcdf':
        filetype = 'h5netcdf'
    else:
        raise IOError('Do not know how to read the supplied file extension: ' + path.suffix)

    return filetype


def _expand_wildcards(path):
    """Return list of filepaths matching wildcard"""

    # Find first parent directory which does not contain a wildcard
    base_dir = next(parent for parent in path.parents if '*' not in str(parent))

    # Find path relative to parent
    search_pattern = str(path.relative_to(base_dir))

    # Search this relative path from the parent directory for all files matching user input
    filepaths = list(base_dir.glob(search_pattern))

    # Sort by numbers in filepath before returning
    return sorted(filepaths, key=lambda filepath: str(filepath))


def _organise_files(filepaths, datasets, nxpe, nype):
    """
    Arranges given files into an ndarray so they can be concatenated.
    """

    # This is where our knowledge of how BOUT does its parallelization is actually used

    # For now assume that there is no splitting along t, and that all files are in current dir
    # TODO Generalise to deal with extra splitting along t later

    concat_dims = []
    if nxpe > 1:
        concat_dims.append('x')
    if nype > 1:
        concat_dims.append('y')

    dataset_pieces = dict(zip(filepaths, datasets))

    # BOUT names files as num = nxpe*i + j
    # So use this knowledge to arrange files in the right shape for concatenation
    ds_grid = np.empty((nxpe, nype), dtype='object')
    for i in range(nxpe):
        for j in range(nype):
            file_num = (i + nxpe * j)
            filename = Path(filepaths[0].parent / re.sub('\d+', str(file_num), filepaths[0].name))
            ds_grid[i, j] = {'key': dataset_pieces[filename]}

    return ds_grid.squeeze(), concat_dims


def _trim(ds_grid, concat_dims, guards, ghosts, keep_guards):
    """
    Trims all ghost and guard cells off each dataset in ds_grid to prepare for concatenation.

    Returns a new grid of datasets instead of trimming in-place, as lazy loading should mean this is cheap.
    """

    if not any(v > 0 for v in guards.values()) and not any(v > 0 for v in ghosts.values()):
        # Check that some kind of trimming is actually necessary
        print('No trimming was needed')
        return ds_grid
    else:
        # Create new numpy array to insert results into
        return_ds_grid = np.empty_like(ds_grid)

        # Loop over all datasets in grid
        for index, ds_dict in np.ndenumerate(ds_grid):
            # Unpack the dataset from the dict holding it
            ds = ds_dict['key']

            trimmed_ds = _trim_single_ds(index, ds, concat_dims, ds_grid.shape,
                                         guards, ghosts, keep_guards)

            # Insert into new dataset grid, contained in a dict
            return_ds_grid[index] = {'key': trimmed_ds}

        return return_ds_grid


def _trim_single_ds(index, ds, concat_dims, ds_grid_shape, guards, ghosts, keep_guards):

    # Determine how many cells to trim off each dimension
    lower, upper = {}, {}
    for dim in concat_dims:

        # Trime off ghost cells
        ghost = ghosts.get(dim, None)
        # This allows for no ghost cells to be specified as either ghosts = {'x': 0} or ghosts = {'x': None}
        if ghost == 0:
            lower[dim] = None
            upper[dim] = None
        else:
            lower[dim] = ghost
            upper[dim] = -ghost

        # If ds is at edge of grid trim guard cells instead of ghost cells
        if keep_guards[dim] is not None:  # This check is for unit testing/debugging purposes
            dim_axis = concat_dims.index(dim)
            dim_max = ds_grid_shape[dim_axis]
            if not keep_guards[dim] and guards[dim] > 0:
                if index[dim_axis] == 0:
                    lower[dim] = guards[dim]
                if index[dim_axis] == dim_max - 1:
                    upper[dim] = -guards[dim]
            else:
                if index[dim_axis] == 0:
                    lower[dim] = None
                if index[dim_axis] == dim_max - 1:
                    upper[dim] = None

    # Selection to use to trim the dataset
    selection = {dim: slice(lower[dim], upper[dim], None) for dim in concat_dims}

    # Return trimmed subset as a new object
    return ds.isel(**selection)
