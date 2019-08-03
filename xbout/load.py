from warnings import warn
from pathlib import Path
from functools import partial
import configparser

import xarray as xr

from natsort import natsorted

from .grid import open_grid
from .utils import _set_attrs_on_all_vars, _separate_metadata, _check_filetype


# This code should run whenever any function from this module is imported
# Set all attrs to survive all mathematical operations
# (see https://github.com/pydata/xarray/pull/2482)
try:
    xr.set_options(keep_attrs=True)
except ValueError:
    raise ImportError("For dataset attributes to be permanent you need to be "
                      "using the development version of xarray - found at "
                      "https://github.com/pydata/xarray/")
try:
    xr.set_options(file_cache_maxsize=256)
except ValueError:
    raise ImportError("For open and closing of netCDF files correctly you need"
                      " to be using the development version of xarray - found"
                      " at https://github.com/pydata/xarray/")


# TODO somehow check that we have access to the latest version of auto_combine


def open_boutdataset(datapath='./BOUT.dmp.*.nc', chunks={},
                     inputfilepath=None,
                     gridfilepath=None, geometry=None,
                     run_name=None, info=True):
    """
    Load a dataset from a set of BOUT output files, including the input options file.

    Parameters
    ----------
    datapath : str, optional
    prefix : str, optional
    slices : slice object, optional
    chunks : dict, optional
    inputfilepath : str, optional
    gridfilepath : str, optional
    run_name : str, optional
    info : bool, optional

    Returns
    -------
    ds : xarray.Dataset
    """

    # TODO handle possibility that we are loading a previously saved (and trimmed) dataset

    # Gather pointers to all numerical data from BOUT++ output files
    ds, metadata = _auto_open_mfboutdataset(datapath=datapath, chunks=chunks)
    ds = _set_attrs_on_all_vars(ds, 'metadata', metadata)

    if inputfilepath:
        # Use Ben's options class to store all input file options
        with open(inputfilepath, 'r') as f:
            config_string = "[dummysection]\n" + f.read()
        options = configparser.ConfigParser()
        options.read_string(config_string)
    else:
        options = None
    ds = _set_attrs_on_all_vars(ds, 'options', options)

    if gridfilepath:
        ds = open_grid(gridfilepath=gridfilepath, geometry=geometry, ds=ds)

    # TODO read and store git commit hashes from output files

    if run_name:
        ds.name = run_name

    if info is 'terse':
        print("Read in dataset from {}".format(str(Path(datapath))))
    elif info:
        print("Read in:\n{}".format(ds.bout))

    return ds


def _auto_open_mfboutdataset(datapath, chunks={}, info=True, keep_guards=True):
    filepaths, filetype = _expand_filepaths(datapath)

    # Open just one file to read processor splitting
    nxpe, nype, mxg, myg, mxsub, mysub = _read_splitting(filepaths[0], info)

    paths_grid, concat_dims = _arrange_for_concatenation(filepaths, nxpe, nype)

    _preprocess = partial(_trim, ghosts={'x': mxg, 'y': myg})

    ds = xr.open_mfdataset(paths_grid, concat_dim=concat_dims,
                           combine='nested', data_vars='minimal',
                           preprocess=_preprocess, engine=filetype,
                           chunks=chunks)

    ds, metadata = _separate_metadata(ds)

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
        xr.set_options(file_cache_maxsize=len(filepaths))

    return filepaths, filetype


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
    ds = xr.open_dataset(str(filepath))

    # TODO check that BOUT doesn't ever set the number of guards to be different to the number of ghosts

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


def _trim(ds, ghosts={}, keep_guards=True):
    """
    Trims all ghost and guard cells off a single dataset read from a single
    BOUT dump file, to prepare for concatenation.

    Parameters
    ----------
    ghosts : dict, optional
    guards : dict, optional
    keep_guards : dict, optional
    """

    # TODO generalise this function to handle guard cells being optional
    if not keep_guards:
        raise NotImplementedError

    selection = {}
    for dim in ds.dims:
        if ghosts.get(dim, False):
            selection[dim] = slice(ghosts[dim], -ghosts[dim])

    trimmed_ds = ds.isel(**selection)
    return trimmed_ds
