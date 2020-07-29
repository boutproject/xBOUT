from copy import deepcopy
from itertools import chain

import numpy as np


def _set_attrs_on_all_vars(ds, key, attr_data, copy=False):
    ds.attrs[key] = attr_data
    if copy:
        for v in chain(ds.data_vars, ds.coords):
            ds[v].attrs[key] = deepcopy(attr_data)
    else:
        for v in chain(ds.data_vars, ds.coords):
            ds[v].attrs[key] = attr_data
    return ds


def _add_attrs_to_var(ds, varname, copy=False):
    if copy:
        for attr in ["metadata", "options", "geometry", "regions"]:
            if attr in ds.attrs and attr not in ds[varname].attrs:
                ds[varname].attrs[attr] = deepcopy(ds.attrs[attr])
    else:
        for attr in ["metadata", "options", "geometry", "regions"]:
            if attr in ds.attrs and attr not in ds[varname].attrs:
                ds[varname].attrs[attr] = ds.attrs[attr]


def _check_filetype(path):
    if path.suffix == '.nc':
        filetype = 'netcdf4'
    elif path.suffix == '.h5netcdf':
        filetype = 'h5netcdf'
    else:
        raise IOError("Do not know how to read file extension {}"
                      .format(path.suffix))
    return filetype


def _separate_metadata(ds):
    """
    Extract the metadata (nxpe, myg etc.) from the Dataset.

    Assumes that all scalar variables are metadata, not physical data!
    """

    # Find only the scalar variables
    variables = list(ds.variables)
    scalar_vars = [var for var in variables
                   if not any(dim in ['t', 'x', 'y', 'z'] for dim in ds[var].dims)]

    # Save metadata as a dictionary
    metadata_vals = [ds[var].values.item() for var in scalar_vars]
    metadata = dict(zip(scalar_vars, metadata_vals))

    return ds.drop(scalar_vars), metadata


def _update_metadata_increased_resolution(da, n):
    """
    Update the metadata variables to account for a y-direction resolution increased by a
    factor n.

    Parameters
    ----------
    da : DataArray
        The variable to update
    n : int
        The factor to increase the y-resolution by
    """

    # Take deepcopy to ensure we do not alter metadata of other variables
    da.attrs['metadata'] = deepcopy(da.metadata)

    def update_jyseps(name):
        # If any jyseps<=0, need to leave as is
        if da.metadata[name] > 0:
            da.metadata[name] = n*(da.metadata[name] + 1) - 1
    update_jyseps('jyseps1_1')
    update_jyseps('jyseps2_1')
    update_jyseps('jyseps1_2')
    update_jyseps('jyseps2_2')

    def update_ny(name):
        da.metadata[name] = n*da.metadata[name]
    update_ny('ny')
    update_ny('ny_inner')
    update_ny('MYSUB')

    return da
