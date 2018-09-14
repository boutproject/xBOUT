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

from xcollect.concatenate import _concat_nd


def collect(var, xind=None, yind=None, zind=None, tind=None,
            path='./', prefix='BOUT.dmp.', yguards=False, xguards=True,
            info=True, strict=False, chunks={}):
    """
    Collect a variable from a set of BOUT++ output files.

    Uses xarray + dask to load data lazily, parallelize operation and limit memory usage.

    Parameters
    ----------
    var : str
        Data variable to be collected. If one of the variables in the files then returns the data
        values as a numpy ndarray as collect used to do. If 'all' then returns view to all data as
        an xarray.Dataset, which is chunked using dask.
    path : str
    prefix : str
    xguards : bool
        Choice whether or not to keep domain guard cells in the x-direction.
    yguards : bool
        Choice whether or not to keep domain guard cells in the y-direction.
    info : bool, optional
    strict : bool, optional
    chunks : dict
        Dask chunks to split arrays into. Default is to load each dump file as one chunk.

    Returns
    -------
    ds : xarray.Dataset
        View to dataset containing all data variables in files
    or
    var : np.ndarray
        Contains collected variable
    """

    # TODO implement optional arguments info & strict

    filepaths, datasets = _open_all_dump_files(path, prefix, chunks=chunks)

    ds = xr.open_dataset(filepaths[0])
    nxpe, nype = ds['NXPE'], ds['NYPE']
    mxg, myg = ds['MXG'], ds['MYG']

    ds_grid, concat_dims = _organise_files(filepaths, datasets, prefix, nxpe, nype)

    ds_grid = _trim(ds_grid, concat_dims,
                    guards={'x': mxg, 'y': myg}, ghosts={'x': mxg, 'y': myg},
                    keep_guards={'x': xguards, 'y': yguards})

    ds = _concat_nd(ds_grid, concat_dims=concat_dims, data_vars=['minimal']*len(concat_dims))

    if var == 'all':
        # Utilise xarray's lazy loading capabilities by returning a DataArray view to the data values.
        # Should always use this option except for when old scripts are expecting an eager numpy array.
        return ds
    else:
        # For backwards compatibility, return only the numpy array of data values
        # This will immediately load entire array into memory
        var_da = ds[var]
        var_da = _take_slice(var_da, xind, yind, zind, tind)
        return var_da.values


def _open_all_dump_files(path, prefix, chunks={}):
    """Determines filetypes and opens all dump files."""

    file_list_nc = glob.glob(os.path.join(path, prefix + ".*nc"))
    file_list_h5 = glob.glob(os.path.join(path, prefix + ".*hdf5"))
    if file_list_nc != [] and file_list_h5 != []:
        raise IOError("Error: Both NetCDF and HDF5 files are present: do not know which to read.")
    elif file_list_h5 != []:
        filetype = 'h5netcdf'
        file_list = file_list_h5
    else:
        filetype = 'netcdf4'
        file_list = file_list_nc

    if file_list == []:
        raise IOError("ERROR: No data files found in path {0}".format(path))

    filepaths = sorted(file_list)

    # Default chunks={} is for each file to be one chunk
    datasets = [xr.open_dataset(file, engine=filetype, chunks=chunks) for file in filepaths]

    return filepaths, datasets


def _organise_files(filepaths, datasets, prefix, nxpe, nype):
    # This is where our knowledge of how BOUT does its parallelization is actually used

    # For now assume that there is no splitting along t, and that all files are in current dir
    # TODO Generalise to deal with extra splitting along t later

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
            ds_grid[i, j] = {'key': dataset_pieces[filename]}

    return ds_grid.squeeze(), concat_dims


def _trim(ds_grid, concat_dims, guards, ghosts, keep_guards):
    """
    Trims all ghost and guard cells off each dataset in ds_grid to prepare for concatenation.
    """
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

        # Actually trim the dataset in-place
        selection = {dim: slice(lower[dim], upper[dim], None) for dim in ds.dims}

        # Insert back, contained in a dict
        ds_grid[index] = {'key': ds.isel(**selection)}

    return ds_grid


def _take_slice(da, xind=None, yind=None, zind=None, tind=None):
    """Just for backwards compatibility"""
    selection = {}
    if xind is not None:
        selection['x'] = slice(xind[0], xind[1])
    if yind is not None:
        selection['y'] = slice(yind[0], yind[1])
    if zind is not None:
        selection['z'] = slice(zind[0], zind[1])
    if tind is not None:
        selection['t'] = slice(tind[0], tind[1])
    return da.isel(**selection)
