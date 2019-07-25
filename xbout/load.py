from warnings import warn
from pathlib import Path

import numpy as np
import xarray

from functools import partial

from natsort import natsorted

_BOUT_TIMING_VARIABLES = ['wall_time', 'wtime', 'wtime_rhs', 'wtime_invert',
                          'wtime_comms', 'wtime_io', 'wtime_per_rhs', 'wtime_per_rhs_e',
                          'wtime_per_rhs_i']

def _auto_open_mfboutdataset(datapath, chunks={}, info=True,
                             keep_xboundaries=False, keep_yboundaries=False):
    filepaths, filetype = _expand_filepaths(datapath)

    # Open just one file to read processor splitting
    nxpe, nype, mxg, myg, mxsub, mysub = _read_splitting(filepaths[0], info)

    paths_grid, concat_dims = _arrange_for_concatenation(filepaths, nxpe, nype)

    _preprocess = partial(_trim, guards={'x': mxg, 'y': myg},
                          keep_boundaries={'x': keep_xboundaries, 'y': keep_yboundaries},
                          nxpe=nxpe, nype=nype)

    # TODO warning message to make sure user knows if it's parallelized
    ds = xarray.open_mfdataset(paths_grid, concat_dim=concat_dims,
                               combine='nested', data_vars='minimal',
                               preprocess=_preprocess, engine=filetype,
                               chunks=chunks)

    ds, metadata = _strip_metadata(ds)

    return ds, metadata


def _expand_filepaths(datapath):
    """Determines filetypes and opens all dump files."""
    path = Path(datapath)

    filetype = _check_filetype(path)

    filepaths = _expand_wildcards(path)

    if not filepaths:
        raise IOError("No datafiles found matching datapath={}"
                      .format(datapath))

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
        raise IOError("Do not know how to read file extension "
                      "\"{path.suffix}\"")

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


def _read_splitting(filepath, info=True):
    ds = xarray.open_dataset(str(filepath))

    # Account for case of no parallelisation, when nxpe etc won't be in dataset
    def get_scalar(ds, key, default=1, info=True):
        if key in ds:
            return ds[key].values
        else:
            if info is True:
                print("{key} not found, setting to {default}".format(key=key, default=default))
            return default

    nxpe = get_scalar(ds, 'NXPE', default=1)
    nype = get_scalar(ds, 'NYPE', default=1)
    mxg = get_scalar(ds, 'MXG', default=2)
    myg = get_scalar(ds, 'MYG', default=0)
    mxsub = get_scalar(ds, 'MXSUB', default=ds.dims['x'] - 2 * mxg)
    mysub = get_scalar(ds, 'MYSUB', default=ds.dims['y'] - 2 * myg)

    # Avoid trying to open this file twice
    ds.close()

    return nxpe, nype, mxg, myg, mxsub, mysub


def _arrange_for_concatenation(filepaths, nxpe=1, nype=1):
    """
    Arrange filepaths into a nested list-of-lists which represents their
    ordering across different processors and consecutive simulation runs.

    Filepaths must be a sorted list. Uses the fact that BOUT's output files are
    named as num = nxpe*i + j, where i={0, ..., nype}, j={0, ..., nxpe}.
    Also assumes that any consecutive simulation runs are in directories which
    when sorted are in the correct order (e.g. /run0/*, /run1/*, ...).
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


def _trim(ds, *, guards, keep_boundaries, nxpe, nype):
    """
    Trims all guard (and optionally boundary) cells off a single dataset read from a
    single BOUT dump file, to prepare for concatenation.
    Also drops some variables that store timing information, which are different for each
    process and so cannot be concatenated.

    Parameters
    ----------
    guards : dict
        Number of guard cells along each dimension, e.g. {'x': 2, 't': 0}
    keep_boundaries : dict
        Whether or not to preserve the boundary cells along each dimension, e.g.
        {'x': True, 'y': False}
    """

    if any(keep_boundaries.values()):
        # Work out if this particular dataset contains any boundary cells
        # Relies on a change to xarray so datasets always have source encoding
        # See xarray GH issue #2550
        lower_boundaries, upper_boundaries = _infer_contains_boundaries(
            ds.encoding['source'], nxpe, nype)
    else:
        lower_boundaries, upper_boundaries = {}, {}

    selection = {}
    for dim in ds.dims:
        lower = _get_limit('lower', dim, keep_boundaries, lower_boundaries,
                           guards)
        upper = _get_limit('upper', dim, keep_boundaries, upper_boundaries,
                           guards)
        selection[dim] = slice(lower, upper)
    trimmed_ds = ds.isel(**selection)

    return trimmed_ds.drop(_BOUT_TIMING_VARIABLES, errors='ignore')


def _infer_contains_boundaries(filename, nxpe, nype):
    """
    Uses the name of the output file and the domain decomposition to work out
    whether this dataset contains boundary cells, and on which side.

    Uses knowledge that BOUT names its output files as /folder/prefix.num.nc,
    with a numbering scheme
    num = nxpe*i + j, where i={0, ..., nype}, j={0, ..., nxpe}
    """

    *prefix, filenum, extension = Path(filename).suffixes
    filenum = int(filenum.replace('.', ''))

    lower_boundaries, upper_boundaries = {}, {}

    lower_boundaries['x'] = filenum % nxpe == 0
    upper_boundaries['x'] = filenum % nxpe == nxpe-1

    lower_boundaries['y'] = filenum < nxpe
    upper_boundaries['y'] = filenum >= (nype-1)*nxpe

    return lower_boundaries, upper_boundaries


def _get_limit(side, dim, keep_boundaries, boundaries, guards):
    # Check for boundary cells, otherwise use guard cells, else leave alone

    if keep_boundaries.get(dim, False):
        if boundaries.get(dim, False):
            limit = None
        else:
            limit = guards[dim] if side is 'lower' else -guards[dim]
    elif guards.get(dim, False):
        limit = guards[dim] if side is 'lower' else -guards[dim]
    else:
        limit = None

    return limit


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
