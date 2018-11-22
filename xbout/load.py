from warnings import warn
from pathlib import Path

import numpy as np
import xarray

from natsort import natsorted


def _auto_open_mfboutdataset(datapath, chunks, info, keep_guards=True):
    path = Path(datapath)
    filepaths, filetype = _expand_filepaths(path)

    # Open just one file to read processor splitting
    nxpe, nype, mxg, myg, mxsub, mysub = _read_splitting(filepaths[0])

    paths_grid, concat_dims = _arrange_for_concatenation(filepaths, nxpe, nype)

    # TODO Special case needed for case of just one dump file?
    ds = xarray.open_mfdataset(paths_grid, concat_dims=concat_dims,
                               data_vars='minimal',
                               engine=filetype, chunks=chunks,
                               infer_order_from_coords=False)

    ds, metadata = _strip_metadata(ds)

    trimmed_ds = _trim(ds, ghosts={'x': mxg, 'y': myg},
                       proc_data_sizes={'x': mxsub, 'y': mysub},
                       proc_splitting={'x': nxpe, 'y': nype},
                       guards={'x': mxg, 'y': myg}, keep_guards=keep_guards)

    return trimmed_ds, metadata


def _expand_filepaths(path):
    """Determines filetypes and opens all dump files."""

    filetype = _check_filetype(path)

    filepaths = _expand_wildcards(path)

    if len(filepaths) > 128:
        warn("Trying to open a large number of files - setting xarray's"
             " `file_cache_maxsize` global option to {} to accommodate this. "
             "Recommend using `xr.set_options(file_cache_maxsize=NUM)`"
             " to explicitly set this to a large enough value."
             .format(str(len(filepaths))), UserWarning)
        xarray.set_options(file_cache_maxsize=len(filepaths))

    return filepaths, filetype


def _check_filetype(path):
    if path.suffix == '.nc':
        filetype = 'netcdf4'
    elif path.suffix == '.h5netcdf':
        filetype = 'h5netcdf'
    else:
        raise IOError("Do not know how to read the supplied file extension: "
                      "{}".format(path.suffix))

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
    # Use "natural" sort to avoid lexicographic ordering of numbers
    # e.g. ['0', '1', '10', '11', etc.]
    return natsorted(filepaths, key=lambda filepath: str(filepath))


def _read_splitting(filepath):
    # TODO make sure dask is being used here?
    ds = xarray.open_dataset(str(filepath))

    # TODO check that BOUT doesn't ever set the number of guards to be different to the number of ghosts

    nxpe, nype = ds['NXPE'].values, ds['NYPE'].values
    mxg, myg = ds['MXG'].values, ds['MYG'].values
    mxsub, mysub = ds['MXSUB'].values, ds['MYSUB'].values

    # Avoid trying to open this file twice
    ds.close()

    return nxpe, nype, mxg, myg, mxsub, mysub


def _arrange_for_concatenation(filepaths, nxpe=1, nype=1):
    """
    Arrange filepaths into a nested list-of-lists which represents their
    ordering across different processors and consecutive simulation runs.

    Filepaths must be a sorted list. Uses the fact that BOUT's output files are
    named as num = nxpe*i + j, and assumes that any consectutive simulation
    runs are in directories which when sorted are in the correct order
    (e.g. /run0/*, /run1/*,  ...).
    """

    nprocs = nxpe * nype
    n_runs = int(len(filepaths) / nprocs)
    if len(filepaths) % nprocs != 0:
        raise ValueError("Each run directory does not contain an equal number"
                         "of output files. If the parallelization scheme of "
                         "your simulation changed partway-through, then please"
                         "load each directory separately and concatenate them"
                         "along the time dimension with xarray.concat().")

    # Create list of lists of filepaths, so that xarray knows how they should
    # be concatenated by xarray.open_mfdataset()
    # Only possible with this Pull Request to xarray
    # https://github.com/pydata/xarray/pull/2553
    paths = iter(filepaths)
    paths_grid = [[[next(paths) for x in range(nxpe)]
                                for y in range(nype)]
                                for t in range(n_runs)]

    # Dimensions along which no concatenation is needed are still present as
    # single-element lists, so need to concatenation along dim=None for those
    concat_dims = [None, None, None]
    if len(filepaths) > nprocs:
        concat_dims[0] = 't'
    if nype > 1:
        concat_dims[1] = 'y'
    if nxpe > 1:
        concat_dims[2] = 'x'

    return paths_grid, concat_dims


def _trim(ds, ghosts={}, proc_splitting={}, proc_data_sizes={},
          guards={}, keep_guards=True):
    """
    Trims all ghost and guard cells off the combined dataset produced by
    `open_mfdataset()`.

    Parameters
    ----------
    ghosts : dict, optional

    proc_splitting : dict, optional

    guards : dict, optional

    keep_guards : dict, optional

    """

    # TODO generalise this function to handle guard cells being optional
    if not keep_guards:
        raise NotImplementedError

    selection = {}
    for dim in ds.dims:
        if ghosts.get(dim, False):
            single_proc_mask = [False]*ghosts[dim] \
                               + [True]*proc_data_sizes[dim] \
                               + [False]*ghosts[dim]

            selection[dim] = np.tile(np.array(single_proc_mask),
                                     reps=proc_splitting[dim])

    trimmed_ds = ds.isel(**selection)
    return trimmed_ds


def _strip_metadata(ds):
    """
    Extract the metadata (nxpe, myg etc.) from the Dataset.

    Assumes that all scalar variables are metadata, not physical data!
    """

    # Find only the scalar variables
    variables = list(ds.variables)
    scalar_vars = [var for var in variables
                   if not any(dim in ['t', 'x', 'y', 'z'] for dim in ds[var].dims)]

    # Save metadata as a dictionary
    metadata_vals = [np.asscalar(ds[var].values) for var in scalar_vars]
    metadata = dict(zip(scalar_vars, metadata_vals))

    return ds.drop(scalar_vars), metadata
