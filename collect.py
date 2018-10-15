"""
Contains an xarray + dask reimplementation of the BOUT++ collect function.

Original found at https://github.com/boutproject/BOUT-dev/blob/master/tools/pylib/boutdata/collect.py .
This version should be much faster (works in parallel), and more scalable (dask chunks limit memory usage).
Although it would be relatively simple to alter this to maintain backwards compatibility for all input arguments,
we choose not to do so.
"""

import xarray as xr
import numpy as np

import os
import glob

from xcollect.concatenate import _concat_nd


def collect(vars='all', path='./', prefix='BOUT.dmp',
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

    filepaths, datasets = _open_all_dump_files(path, prefix, chunks=chunks)

    # Open just one file to read processor splitting
    ds = xr.open_dataset(filepaths[0])
    nxpe, nype = ds['NXPE'].values, ds['NYPE'].values
    mxg, myg = ds['MXG'].values, ds['MYG'].values

    ds_grid, concat_dims = _organise_files(filepaths, datasets, prefix, nxpe, nype)

    ds_grid = _trim(ds_grid, concat_dims,
                    guards={'x': mxg, 'y': myg}, ghosts={'x': mxg, 'y': myg},
                    keep_guards={'x': xguards, 'y': yguards})

    ds = _concat_nd(ds_grid, concat_dims=concat_dims, data_vars=['minimal']*len(concat_dims))

    # Utilise xarray's lazy loading capabilities by returning a DataSet/DataArray view to the data values.
    if vars == 'all':
        return ds.isel(**slices)
    else:
        return ds[vars].isel(**slices)


def _open_all_dump_files(path, prefix, chunks={}):
    """Determines filetypes and opens all dump files."""

    file_list_nc = glob.glob(os.path.join(path, prefix + ".*nc"))
    file_list_h5 = glob.glob(os.path.join(path, prefix + ".*hdf5"))
    if file_list_nc and file_list_h5:
        raise IOError("Both NetCDF and HDF5 files are present: do not know which to read.")
    elif file_list_h5:
        filetype = 'h5netcdf'
        file_list = file_list_h5
    elif file_list_nc:
        filetype = 'netcdf4'
        file_list = file_list_nc
    else:
        raise IOError("No data files found in path {0}".format(path))

    filepaths = sorted(file_list)

    # Default chunks={} is for each file to be one chunk
    datasets = [xr.open_dataset(file, engine=filetype, chunks=chunks) for file in filepaths]

    return filepaths, datasets


def _organise_files(filepaths, datasets, prefix, nxpe, nype):
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
    # TODO replace this kind of manipulation using the python path library?
    filestem = filepaths[0].rsplit('/', 1)[0]

    # BOUT names files as num = nxpe*i + j
    # So use this knowledge to arrange files in the right shape for concatenation
    ds_grid = np.empty((nxpe, nype), dtype='object')
    for i in range(nxpe):
        for j in range(nype):
            file_num = (i + nxpe * j)
            filename = filestem + '/' + prefix + '.' + str(file_num) + '.nc'
            ds_grid[i, j] = {'key': dataset_pieces[filename]}

    return ds_grid.squeeze(), concat_dims


def _trim(ds_grid, concat_dims, guards, ghosts, keep_guards):
    """
    Trims all ghost and guard cells off each dataset in ds_grid to prepare for concatenation.
    """

    if not any(v > 0 for v in guards.values()) and not any(v > 0 for v in ghosts.values()):
        # Check that some kind of trimming is actually necessary
        return ds_grid
    else:
        for index, ds_dict in np.ndenumerate(ds_grid):
            # Unpack the dataset from the dict holding it
            ds = ds_dict['key']

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

            # Selection to use to trim the dataset
            selection = {dim: slice(lower[dim], upper[dim], None) for dim in concat_dims}

            # Insert back, contained in a dict
            ds_grid[index] = {'key': ds.isel(**selection)}

        return ds_grid
