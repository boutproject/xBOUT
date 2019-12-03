from warnings import warn
from pathlib import Path
from functools import partial
import configparser

import xarray as xr

from natsort import natsorted

from . import geometries
from .utils import _set_attrs_on_all_vars, _separate_metadata, _check_filetype


_BOUT_PER_PROC_VARIABLES = ['wall_time', 'wtime', 'wtime_rhs', 'wtime_invert',
                            'wtime_comms', 'wtime_io', 'wtime_per_rhs',
                            'wtime_per_rhs_e', 'wtime_per_rhs_i', 'PE_XIND', 'PE_YIND',
                            'MYPE']


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


def open_boutdataset(datapath='./BOUT.dmp.*.nc', inputfilepath=None,
                     geometry=None, gridfilepath=None, chunks={},
                     keep_xboundaries=True, keep_yboundaries=False,
                     run_name=None, info=True):
    """
    Load a dataset from a set of BOUT output files, including the input options
    file. Can also load from a grid file.

    Parameters
    ----------
    datapath : str, optional
        Path to the data to open. Can point to either a set of one or more dump
        files, or a single grid file.

        To specify multiple dump files you must enter the path to them as a
        single glob, e.g. './BOUT.dmp.*.nc', or for multiple consecutive runs
        in different directories (in order) then './run*/BOUT.dmp.*.nc'.
    chunks : dict, optional
    inputfilepath : str, optional
    geometry : str, optional
        The geometry type of the grid data. This will specify what type of
        coordinates to add to the dataset, e.g. 'toroidal' or 'cylindrical'.

        If not specified then will attempt to read it from the file attrs.
        If still not found then a warning will be thrown, which can be
        suppressed by passing `info`=False.

        To define a new type of geometry you need to use the
        `register_geometry` decorator. You are encouraged to do this for your
        own BOUT++ physics module, to apply relevant normalisations.
    gridfilepath : str, optional
        The path to a grid file, containing any variables needed to apply the geometry
        specified by the 'geometry' option, which are not contained in the dump files.
    keep_xboundaries : bool, optional
        If true, keep x-direction boundary cells (the cells past the physical
        edges of the grid, where boundary conditions are set); increases the
        size of the x dimension in the returned data-set. If false, trim these
        cells.
    keep_yboundaries : bool, optional
        If true, keep y-direction boundary cells (the cells past the physical
        edges of the grid, where boundary conditions are set); increases the
        size of the y dimension in the returned data-set. If false, trim these
        cells.
    run_name : str, optional
        Name to give to the whole dataset, e.g. 'JET_ELM_high_resolution'.
        Useful if you are going to open multiple simulations and compare the
        results.
    info : bool, optional

    Returns
    -------
    ds : xarray.Dataset
    """

    # TODO handle possibility that we are loading a previously saved (and trimmed) dataset

    # Determine if file is a grid file or data dump files
    if _is_dump_files(datapath):
        # Gather pointers to all numerical data from BOUT++ output files
        ds = _auto_open_mfboutdataset(datapath=datapath, chunks=chunks,
                                      keep_xboundaries=keep_xboundaries,
                                      keep_yboundaries=keep_yboundaries)
    else:
        # Its a grid file
        ds = _open_grid(datapath, chunks=chunks,
                        keep_xboundaries=keep_xboundaries,
                        keep_yboundaries=keep_yboundaries)

    ds, metadata = _separate_metadata(ds)
    # Store as ints because netCDF doesn't support bools, so we can't save bool
    # attributes
    metadata['keep_xboundaries'] = int(keep_xboundaries)
    metadata['keep_yboundaries'] = int(keep_yboundaries)
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

    if geometry is None:
        if geometry in ds.attrs:
            geometry = ds.attrs.get('geometry')
    if geometry:
        if info:
            print("Applying {} geometry conventions".format(geometry))

        if gridfilepath is not None:
            grid = _open_grid(gridfilepath, chunks=chunks,
                              keep_xboundaries=keep_xboundaries,
                              keep_yboundaries=keep_yboundaries,
                              mxg=ds.metadata['MXG'])
        else:
            grid = None

        # Update coordinates to match particular geometry of grid
        ds = geometries.apply_geometry(ds, geometry, grid=grid)
    else:
        if info:
            warn("No geometry type found, no coordinates will be added")

    # TODO read and store git commit hashes from output files

    if run_name:
        ds.name = run_name

    if info is 'terse':
        print("Read in dataset from {}".format(str(Path(datapath))))
    elif info:
        print("Read in:\n{}".format(ds.bout))

    return ds


def _is_dump_files(datapath):
    """
    If there is only one file, and it's not got a time dimension, assume it's a
    grid file. Else assume we have one or more dump files.
    """

    filepaths, filetype = _expand_filepaths(datapath)

    if len(filepaths) == 1:
        ds = xr.open_dataset(filepaths[0], engine=filetype)
        dims = ds.dims
        ds.close()
        return True if 't' in dims else False
    else:
        return True


def _auto_open_mfboutdataset(datapath, chunks={}, info=True,
                             keep_xboundaries=False, keep_yboundaries=False):
    filepaths, filetype = _expand_filepaths(datapath)

    # Open just one file to read processor splitting
    nxpe, nype, mxg, myg, mxsub, mysub = _read_splitting(filepaths[0], info)

    paths_grid, concat_dims = _arrange_for_concatenation(filepaths, nxpe, nype)

    _preprocess = partial(_trim, guards={'x': mxg, 'y': myg},
                          keep_boundaries={'x': keep_xboundaries,
                                           'y': keep_yboundaries},
                          nxpe=nxpe, nype=nype)

    ds = xr.open_mfdataset(paths_grid, concat_dim=concat_dims, combine='nested',
                           data_vars='minimal', preprocess=_preprocess, engine=filetype,
                           chunks=chunks)

    return ds


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
             .format(str(len(filepaths))))
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

    # Account for case of no parallelisation, when nxpe etc won't be in dataset
    def get_scalar(ds, key, default=1, info=True):
        if key in ds:
            return ds[key].values
        else:
            if info is True:
                print("{key} not found, setting to {default}"
                      .format(key=key, default=default))
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
    nxpe : int
        Number of processors in x direction
    nype : int
        Number of processors in y direction
    """

    if any(keep_boundaries.values()):
        # Work out if this particular dataset contains any boundary cells
        # Relies on a change to xarray so datasets always have source encoding
        # See xarray GH issue #2550
        lower_boundaries, upper_boundaries = _infer_contains_boundaries(
            ds, nxpe, nype)
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

    # Ignore FieldPerps for now
    for name in trimmed_ds:
        if (trimmed_ds[name].dims == ('x', 'z')
                or trimmed_ds[name].dims == ('t', 'x', 'z')):
            trimmed_ds = trimmed_ds.drop(name)

    return trimmed_ds.drop(_BOUT_PER_PROC_VARIABLES, errors='ignore')


def _infer_contains_boundaries(ds, nxpe, nype):
    """
    Uses the processor indices and BOUT++'s topology indices to work out whether this
    dataset contains boundary cells, and on which side.
    """

    try:
        xproc = int(ds['PE_XIND'])
        yproc = int(ds['PE_YIND'])
    except KeyError:
        # output file from BOUT++ earlier than 4.3
        # Use knowledge that BOUT names its output files as /folder/prefix.num.nc, with a
        # numbering scheme
        # num = nxpe*i + j, where i={0, ..., nype}, j={0, ..., nxpe}
        filename = ds.encoding['source']
        *prefix, filenum, extension = Path(filename).suffixes
        filenum = int(filenum.replace('.', ''))
        xproc = filenum % nxpe
        yproc = filenum // nxpe

    lower_boundaries, upper_boundaries = {}, {}

    lower_boundaries['x'] = xproc == 0
    upper_boundaries['x'] = xproc == nxpe-1

    lower_boundaries['y'] = yproc == 0
    upper_boundaries['y'] = yproc == nype-1

    jyseps2_1 = int(ds['jyseps2_1'])
    jyseps1_2 = int(ds['jyseps1_2'])
    if jyseps1_2 > jyseps2_1:
        # second divertor present
        ny_inner = int(ds['ny_inner'])
        mysub = int(ds['MYSUB'])
        if mysub*(yproc + 1) == ny_inner:
            upper_boundaries['y'] = True
        elif mysub*yproc == ny_inner:
            lower_boundaries['y'] = True

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

    if limit == 0:
        # 0 would give incorrect result as an upper limit
        limit = None
    return limit


def _open_grid(datapath, chunks, keep_xboundaries, keep_yboundaries, mxg=2):
    """
    Opens a single grid file. Implements slightly different logic for
    boundaries to deal with different conventions in a BOUT grid file.
    """

    gridfilepath = Path(datapath)
    grid = xr.open_dataset(gridfilepath, engine=_check_filetype(gridfilepath),
                           chunks=chunks)

    # TODO find out what 'yup_xsplit' etc are in the doublenull storm file John gave me
    # For now drop any variables with extra dimensions
    acceptable_dims = ['t', 'x', 'y', 'z']
    unrecognised_dims = list(set(grid.dims) - set(acceptable_dims))
    if len(unrecognised_dims) > 0:
        # Weird string formatting is a workaround to deal with possible bug in
        # pytest warnings capture - doesn't match strings containing brackets
        warn(
            "Will drop all variables containing the dimensions {} because "
            "they are not recognised".format(str(unrecognised_dims)[1:-1]))
        grid = grid.drop_dims(unrecognised_dims)

    if not keep_xboundaries:
        xboundaries = mxg
        if xboundaries > 0:
            grid = grid.isel(x=slice(xboundaries, -xboundaries, None))
    if not keep_yboundaries:
        try:
            yboundaries = int(grid['y_boundary_guards'])
        except KeyError:
            # y_boundary_guards variable not in grid file - older grid files
            # never had y-boundary cells
            yboundaries = 0
        if yboundaries > 0:
            # Remove y-boundary cells from first divertor target
            grid = grid.isel(y=slice(yboundaries, -yboundaries, None))
            if grid['jyseps1_2'] > grid['jyseps2_1']:
                # There is a second divertor target, remove y-boundary cells
                # there too
                nin = int(grid['ny_inner'])
                grid_lower = grid.isel(y=slice(None, nin, None))
                grid_upper = grid.isel(
                    y=slice(nin + 2 * yboundaries, None, None))
                grid = xr.concat((grid_lower, grid_upper), dim='y',
                                 data_vars='minimal',
                                 compat='identical')
    return grid
