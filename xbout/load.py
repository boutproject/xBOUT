from copy import copy
from warnings import warn
from pathlib import Path
from functools import partial
from itertools import chain

from boutdata.data import BoutOptionsFile
import xarray as xr
from numpy import unique

from natsort import natsorted

from . import geometries
from .utils import _set_attrs_on_all_vars, _separate_metadata, _check_filetype, _is_path


_BOUT_PER_PROC_VARIABLES = [
    "wall_time",
    "wtime",
    "wtime_rhs",
    "wtime_invert",
    "wtime_comms",
    "wtime_io",
    "wtime_per_rhs",
    "wtime_per_rhs_e",
    "wtime_per_rhs_i",
    "PE_XIND",
    "PE_YIND",
    "MYPE",
]
_BOUT_TIME_DEPENDENT_META_VARS = ["iteration", "hist_hi", "tt"]


# This code should run whenever any function from this module is imported
# Set all attrs to survive all mathematical operations
# (see https://github.com/pydata/xarray/pull/2482)
try:
    xr.set_options(keep_attrs=True)
except ValueError:
    raise ImportError(
        "For dataset attributes to be permanent you need to be "
        "using the development version of xarray - found at "
        "https://github.com/pydata/xarray/"
    )
try:
    xr.set_options(file_cache_maxsize=256)
except ValueError:
    raise ImportError(
        "For open and closing of netCDF files correctly you need"
        " to be using the development version of xarray - found"
        " at https://github.com/pydata/xarray/"
    )


# TODO somehow check that we have access to the latest version of auto_combine


def open_boutdataset(
    datapath="./BOUT.dmp.*.nc",
    inputfilepath=None,
    geometry=None,
    gridfilepath=None,
    chunks=None,
    keep_xboundaries=True,
    keep_yboundaries=False,
    run_name=None,
    info=True,
    is_restart=None,
    **kwargs,
):
    """
    Load a dataset from a set of BOUT output files, including the input options
    file. Can also load from a grid file or from restart files.

    Note that when reloading a Dataset that was saved by xBOUT, the state of the saved
    Dataset is restored, and the values of `keep_xboundaries`, `keep_yboundaries`, and
    `run_name` are ignored. `geometry` is treated specially, and can be passed when
    reloading a Dataset (along with `gridfilepath` if needed).

    Troubleshooting
    ---------------
    Variable conflicts: sometimes, for example when loading data from multiple restarts,
    some variables may have conflicts (e.g. a source term was changed between some of
    the restarts, but the source term is saved as time-independent, without a
    t-dimension). In this case one workaround is to pass a list of variable names to the
    keyword argument `drop_vars` to ignore the variables with conflicts, e.g. if `"S1"`
    and `"S2"` have conflicts
    ```
    ds = open_boutdataset("data*/boutdata.nc", drop_variables=["S1", "S2"])
    ```
    will open a Dataset which is missing `"S1"` and `"S2"`.\
    [`drop_variables` is an argument of `xarray.open_dataset()` that is passed down
    through `kwargs`.]

    Parameters
    ----------
    datapath : str or (list or tuple of xr.Dataset), optional
        Path to the data to open. Can point to either a set of one or more dump
        files, or a single grid file.

        To specify multiple dump files you must enter the path to them as a
        single glob, e.g. './BOUT.dmp.*.nc', or for multiple consecutive runs
        in different directories (in order) then './run*/BOUT.dmp.*.nc'.

        If a list or tuple of xr.Dataset is passed, they will be combined with
        xr.combine_nested() instead of loading data from disk (intended for unit
        testing).
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
    info : bool or "terse", optional
    is_restart : bool, optional
        Restart files require some special handling (e.g. working around variables that
        are not present in restart files). By default, this special handling is enabled
        if the files do not have a time dimension and `restart` is present in the file
        name in `datapath`. This option can be set to True or False to explicitly enable
        or disable the restart file handling.
    kwargs : optional
        Keyword arguments are passed down to `xarray.open_mfdataset`, which in
        turn passes extra kwargs down to `xarray.open_dataset`.

    Returns
    -------
    ds : xarray.Dataset
    """

    if chunks is None:
        chunks = {}

    input_type = _check_dataset_type(datapath)

    if is_restart is None:
        is_restart = input_type == "restart"
    elif is_restart is True:
        input_type = "restart"

    if "reload" in input_type:
        if input_type == "reload":
            if isinstance(datapath, Path):
                # xr.open_mfdataset only accepts glob patterns as strings, not Path
                # objects
                datapath = str(datapath)
            ds = xr.open_mfdataset(
                datapath,
                chunks=chunks,
                combine="by_coords",
                data_vars="minimal",
                **kwargs,
            )
        elif input_type == "reload_fake":
            ds = xr.combine_by_coords(datapath, data_vars="minimal").chunk(chunks)
        else:
            raise ValueError(f"internal error: unexpected input_type={input_type}")

        def attrs_to_dict(obj, section):
            result = {}
            section = section + ":"
            sectionlength = len(section)
            for key in list(obj.attrs):
                if key[:sectionlength] == section:
                    val = obj.attrs.pop(key)
                    if isinstance(val, bytes):
                        val = val.decode()
                    result[key[sectionlength:]] = val
            return result

        def attrs_remove_section(obj, section):
            section = section + ":"
            sectionlength = len(section)
            has_metadata = False
            for key in list(obj.attrs):
                if key[:sectionlength] == section:
                    has_metadata = True
                    del obj.attrs[key]
            return has_metadata

        # Restore metadata from attrs
        metadata = attrs_to_dict(ds, "metadata")
        if "is_restart" not in metadata:
            # Loading data that was saved with a version of xbout from before
            # "is_restart" was added, so need to add it to the metadata.
            metadata["is_restart"] = int(is_restart)
        ds.attrs["metadata"] = metadata
        # Must do this for all variables and coordinates in dataset too
        for da in chain(ds.data_vars.values(), ds.coords.values()):
            if attrs_remove_section(da, "metadata"):
                da.attrs["metadata"] = metadata

        ds = _add_options(ds, inputfilepath)

        # If geometry was set, apply geometry again
        if geometry is not None:
            if "geometry" != ds.attrs.get("geometry", None):
                warn(
                    f'open_boutdataset() called with geometry="{geometry}", but we are '
                    f"reloading a Dataset that was saved after being loaded with "
                    f'geometry="{ds.attrs.get("geometry", None)}". Applying '
                    f'geometry="{geometry}" from the argument.'
                )
            if gridfilepath is not None:
                grid = _open_grid(
                    gridfilepath,
                    chunks=chunks,
                    keep_xboundaries=ds.metadata["keep_xboundaries"],
                    keep_yboundaries=ds.metadata["keep_yboundaries"],
                    mxg=ds.metadata["MXG"],
                )
            else:
                grid = None
            ds = geometries.apply_geometry(ds, geometry, grid=grid)
        elif "geometry" in ds.attrs:
            ds = geometries.apply_geometry(ds, ds.attrs["geometry"])
        else:
            ds = geometries.apply_geometry(ds, None)

        if info == "terse":
            print("Read in dataset from {}".format(str(Path(datapath))))
        elif info:
            print("Read in:\n{}".format(ds.bout))

        return ds

    # Determine if file is a grid file or data dump files
    remove_yboundaries = False
    if "dump" in input_type or "restart" in input_type:
        # Gather pointers to all numerical data from BOUT++ output files
        ds, remove_yboundaries = _auto_open_mfboutdataset(
            datapath=datapath,
            chunks=chunks,
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
            is_restart=is_restart,
            **kwargs,
        )
    elif "grid" in input_type:
        # Its a grid file
        ds = _open_grid(
            datapath,
            chunks=chunks,
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
            **kwargs,
        )
    else:
        raise ValueError(f"internal error: unexpected input_type={input_type}")

    if not is_restart:
        for var in _BOUT_TIME_DEPENDENT_META_VARS:
            if var in ds:
                # Assume different processors in x & y have same iteration etc.
                latest_top_left = {dim: 0 for dim in ds[var].dims}
                if "t" in ds[var].dims:
                    latest_top_left["t"] = -1
                ds[var] = ds[var].isel(latest_top_left).squeeze(drop=True)

    ds, metadata = _separate_metadata(ds)
    # Store as ints because netCDF doesn't support bools, so we can't save
    # bool attributes
    metadata["keep_xboundaries"] = int(keep_xboundaries)
    metadata["keep_yboundaries"] = int(keep_yboundaries)
    metadata["is_restart"] = int(is_restart)
    ds = _set_attrs_on_all_vars(ds, "metadata", metadata)

    if remove_yboundaries:
        # If remove_yboundaries is True, we need to keep y-boundaries when opening the
        # grid file, as they will be removed from the full Dataset below
        keep_yboundaries = True

    ds = _add_options(ds, inputfilepath)

    if geometry is None:
        if geometry in ds.attrs:
            geometry = ds.attrs.get("geometry")
    if geometry:
        if info:
            print("Applying {} geometry conventions".format(geometry))

        if gridfilepath is not None:
            grid = _open_grid(
                gridfilepath,
                chunks=chunks,
                keep_xboundaries=keep_xboundaries,
                keep_yboundaries=keep_yboundaries,
                mxg=ds.metadata["MXG"],
            )
        else:
            grid = None
    else:
        grid = None
        if info:
            warn("No geometry type found, no physical coordinates will be added")

    # Update coordinates to match particular geometry of grid
    ds = geometries.apply_geometry(ds, geometry, grid=grid)

    if remove_yboundaries:
        ds = ds.bout.remove_yboundaries()

    # TODO read and store git commit hashes from output files

    if run_name:
        ds.name = run_name

    # Set some default settings that are only used in post-processing by xBOUT, not by
    # BOUT++
    ds.bout.fine_interpolation_factor = 8

    if info == "terse":
        print("Read in dataset from {}".format(str(Path(datapath))))
    elif info:
        print("Read in:\n{}".format(ds.bout))

    return ds


def _add_options(ds, inputfilepath):
    if inputfilepath:
        # Use Ben's options class to store all input file options
        options = BoutOptionsFile(
            inputfilepath,
            nx=ds.metadata["nx"],
            ny=ds.metadata["ny"],
            nz=ds.metadata["nz"],
        )
    else:
        options = None
    ds = _set_attrs_on_all_vars(ds, "options", options)
    return ds


def collect(
    varname,
    xind=None,
    yind=None,
    zind=None,
    tind=None,
    path=".",
    yguards=False,
    xguards=True,
    info=True,
    prefix="BOUT.dmp",
):
    """

    Extract the data pertaining to a specified variable in a BOUT++ data set


    Parameters
    ----------
    varname : str
        Name of the variable
    xind, yind, zind, tind : int, slice or list of int, optional
        Range of X, Y, Z or time indices to collect. Either a single
        index to collect, a list containing [start, end] (inclusive
        end), or a slice object (usual python indexing). Default is to
        fetch all indices
    path : str, optional
        Path to data files (default: ".")
    prefix : str, optional
        File prefix (default: "BOUT.dmp")
    yguards : bool, optional
        Collect Y boundary guard cells? (default: False)
    xguards : bool, optional
        Collect X boundary guard cells? (default: True)
        (Set to True to be consistent with the definition of nx)
    info : bool, optional
        Print information about collect? (default: True)

    Notes
    ----------
    strict : This option found in boutdata.collect() is not present in this function
             it is assumed that the varname given is correct, if variable does not exist
             the function will fail
    tind_auto : This option is not required when using _auto_open_mfboutdataset as an
             automatic failure if datasets are different lengths is included

    Returns
    ----------
    ds : numpy.ndarray

    """
    from os.path import join

    datapath = join(path, prefix + "*.nc")

    ds, _ = _auto_open_mfboutdataset(
        datapath, keep_xboundaries=xguards, keep_yboundaries=yguards, info=info
    )

    if varname not in ds:
        raise KeyError("No variable, {} was found in {}.".format(varname, datapath))

    dims = list(ds.dims)
    inds = [tind, xind, yind, zind]

    selection = {}

    # Convert indexing values to an isel suitable format
    for dim, ind in zip(dims, inds):

        if isinstance(ind, int):
            indexer = [ind]
        elif isinstance(ind, list):
            start, end = ind
            indexer = slice(start, end + 1)
        elif ind is not None:
            indexer = ind
        else:
            indexer = None

        if indexer:
            selection[dim] = indexer

    try:
        version = ds["BOUT_VERSION"]
    except KeyError:
        # If BOUT Version is not saved in the dataset
        version = 0

    # Subtraction of z-dimensional data occurs in boutdata.collect
    # if BOUT++ version is old - same feature added here
    if (version < 3.5) and ("z" in dims):
        zsize = int(ds["nz"]) - 1
        ds = ds.isel(z=slice(zsize))

    if selection:
        ds = ds.isel(selection)

    result = ds[varname].values

    # Close netCDF files to ensure they are not locked if collect is called again
    ds.close()

    return result


def _check_dataset_type(datapath):
    """
    Check what type of files we have. Could be:
    (i) produced by xBOUT
        - one or several files, include metadata attributes, e.g.
          'metadata:keep_yboundaries'
    (ii) grid file
        - only one file, and no time dimension
    (iii) produced by BOUT++
        - one or several files
    (iv) restart files produced by BOUT++
        - one or several files, no time dimension, filenames include `restart`
    """

    if not _is_path(datapath):
        # not a filepath glob, so presumably Dataset or list of Datasets used for
        # testing
        if isinstance(datapath, xr.Dataset):
            if "metadata:keep_yboundaries" in datapath.attrs:
                # (i)
                return "reload_fake"
            elif "t" in datapath.dims:
                # (iii)
                return "dump_fake"
            else:
                # (ii)
                return "grid_fake"
        elif len(datapath) > 1:
            if "metadata:keep_yboundaries" in datapath[0].attrs:
                # (i)
                return "reload_fake"
            else:
                # (iii)
                return "dump_fake"
        else:
            # Single element list of Datasets, or nested list of Datasets
            return _check_dataset_type(datapath[0])

    filepaths, filetype = _expand_filepaths(datapath)

    ds = xr.open_dataset(filepaths[0], engine=filetype)
    ds.close()
    if "metadata:keep_yboundaries" in ds.attrs:
        # (i)
        return "reload"
    elif "t" in ds.dims:
        # (iii)
        return "dump"
    elif all(["restart" in Path(p).name for p in filepaths]):
        # (iv)
        return "restart"
    elif len(filepaths) == 1:
        # (ii)
        return "grid"
    else:
        # fall back to opening as dump files
        return "dump"


def _auto_open_mfboutdataset(
    datapath,
    chunks=None,
    info=True,
    keep_xboundaries=False,
    keep_yboundaries=False,
    is_restart=False,
    **kwargs,
):
    if chunks is None:
        chunks = {}

    if is_restart:
        data_vars = "minimal"
    else:
        data_vars = _BOUT_TIME_DEPENDENT_META_VARS

    if _is_path(datapath):
        filepaths, filetype = _expand_filepaths(datapath)

        # Open just one file to read processor splitting
        nxpe, nype, mxg, myg, mxsub, mysub, is_squashed_doublenull = _read_splitting(
            filepaths[0], info, keep_yboundaries
        )

        if is_squashed_doublenull:
            # Need to remove y-boundaries after loading: (i) in case we are loading a
            # squashed data-set, in which case we cannot easily remove the upper
            # boundary cells in _trim(); (ii) because using the remove_yboundaries()
            # method for non-squashed data-sets is simpler than replicating that logic
            # in _trim().
            remove_yboundaries = not keep_yboundaries
            keep_yboundaries = True
        else:
            remove_yboundaries = False

        _preprocess = partial(
            _trim,
            guards={"x": mxg, "y": myg},
            keep_boundaries={"x": keep_xboundaries, "y": keep_yboundaries},
            nxpe=nxpe,
            nype=nype,
            is_restart=is_restart,
        )

        paths_grid, concat_dims = _arrange_for_concatenation(filepaths, nxpe, nype)

        ds = xr.open_mfdataset(
            paths_grid,
            concat_dim=concat_dims,
            combine="nested",
            data_vars=data_vars,
            preprocess=_preprocess,
            engine=filetype,
            chunks=chunks,
            join="exact",
            **kwargs,
        )
    else:
        # datapath was nested list of Datasets

        if isinstance(datapath, xr.Dataset):
            # normalise as one-element list
            datapath = [datapath]

        mxg = int(datapath[0]["MXG"])
        myg = int(datapath[0]["MYG"])
        nxpe = int(datapath[0]["NXPE"])
        nype = int(datapath[0]["NYPE"])
        is_squashed_doublenull = (
            len(datapath) == 1
            and (datapath[0]["jyseps2_1"] != datapath[0]["jyseps1_2"]).values
        )

        if is_squashed_doublenull:
            # Need to remove y-boundaries after loading when loading a squashed
            # data-set, in which case we cannot easily remove the upper boundary cells
            # in _trim().
            remove_yboundaries = not keep_yboundaries
            keep_yboundaries = True
        else:
            remove_yboundaries = False

        _preprocess = partial(
            _trim,
            guards={"x": mxg, "y": myg},
            keep_boundaries={"x": keep_xboundaries, "y": keep_yboundaries},
            nxpe=nxpe,
            nype=nype,
            is_restart=is_restart,
        )

        datapath = [_preprocess(x) for x in datapath]

        ds_grid, concat_dims = _arrange_for_concatenation(datapath, nxpe, nype)

        ds = xr.combine_nested(
            ds_grid,
            concat_dim=concat_dims,
            data_vars=data_vars,
            join="exact",
            combine_attrs="no_conflicts",
        )

    if not is_restart:
        # Remove any duplicate time values from concatenation
        _, unique_indices = unique(ds["t_array"], return_index=True)
        ds = ds.isel(t=unique_indices)

    return ds, remove_yboundaries


def _expand_filepaths(datapath):
    """Determines filetypes and opens all dump files."""
    path = Path(datapath)

    filetype = _check_filetype(path)

    filepaths = _expand_wildcards(path)

    if not filepaths:
        raise IOError("No datafiles found matching datapath={}".format(datapath))

    if len(filepaths) > 128:
        warn(
            "Trying to open a large number of files - setting xarray's"
            " `file_cache_maxsize` global option to {} to accommodate this. "
            "Recommend using `xr.set_options(file_cache_maxsize=NUM)`"
            " to explicitly set this to a large enough value.".format(
                str(len(filepaths))
            )
        )
        xr.set_options(file_cache_maxsize=len(filepaths))

    return filepaths, filetype


def _expand_wildcards(path):
    """Return list of filepaths matching wildcard"""

    # Find first parent directory which does not contain a wildcard
    base_dir = Path(path.anchor)

    # Find path relative to parent
    search_pattern = str(path.relative_to(base_dir))

    # Search this relative path from the parent directory for all files matching user input
    filepaths = list(base_dir.glob(search_pattern))

    # Sort by numbers in filepath before returning
    # Use "natural" sort to avoid lexicographic ordering of numbers
    # e.g. ['0', '1', '10', '11', etc.]
    return natsorted(filepaths, key=lambda filepath: str(filepath))


def _read_splitting(filepath, info, keep_yboundaries):
    ds = xr.open_dataset(str(filepath))

    # Account for case of no parallelisation, when nxpe etc won't be in dataset
    def get_nonnegative_scalar(ds, key, default=1, info=True):
        if key in ds:
            val = ds[key].values
            if val < 0:
                raise ValueError(
                    f"{key} read from dump files is {val}, but negative"
                    f" values are not valid"
                )
            else:
                return val
        else:
            if info is True:
                print(f"{key} not found, setting to {default}")
            if default < 0:
                raise ValueError(
                    f"Default for {key} is {val}, but negative values are not valid"
                )
            return default

    nxpe = get_nonnegative_scalar(ds, "NXPE", default=1, info=info)
    nype = get_nonnegative_scalar(ds, "NYPE", default=1, info=info)
    mxg = get_nonnegative_scalar(ds, "MXG", default=2, info=info)
    myg = get_nonnegative_scalar(ds, "MYG", default=0, info=info)
    mxsub = get_nonnegative_scalar(
        ds, "MXSUB", default=ds.dims["x"] - 2 * mxg, info=info
    )
    mysub = get_nonnegative_scalar(
        ds, "MYSUB", default=ds.dims["y"] - 2 * myg, info=info
    )

    # Check whether this is a single file squashed from the multiple output files of a
    # parallel run (i.e. NXPE*NYPE > 1 even though there is only a single file to read).
    nx = ds["nx"].values
    ny = ds["ny"].values
    nx_file = ds.dims["x"]
    ny_file = ds.dims["y"]
    is_squashed_doublenull = False
    if nxpe > 1 or nype > 1:
        # if nxpe = nype = 1, was only one process anyway, so no need to check for
        # squashing
        if nx_file == nx or nx_file == nx - 2 * mxg:
            has_xboundaries = nx_file == nx
            if not has_xboundaries:
                mxg = 0

            # Check if there are two divertor targets present
            if ds["jyseps1_2"] > ds["jyseps2_1"]:
                upper_target_cells = myg
            else:
                upper_target_cells = 0
            if ny_file == ny or ny_file == ny + 2 * myg + 2 * upper_target_cells:
                # This file contains all the points, possibly including guard cells

                has_yboundaries = not (ny_file == ny)
                if not has_yboundaries:
                    myg = 0

                nxpe = 1
                nype = 1
                is_squashed_doublenull = (ds["jyseps2_1"] != ds["jyseps1_2"]).values
            elif ny_file == ny + 2 * myg:
                # Older squashed file from double-null grid but containing only lower
                # target boundary cells.
                if keep_yboundaries:
                    raise ValueError(
                        "Cannot keep y-boundary points: squashed file is missing upper "
                        "target boundary points."
                    )
                has_yboundaries = not (ny_file == ny)
                if not has_yboundaries:
                    myg = 0

                nxpe = 1
                nype = 1
                # For this case, do not need the special handling enabled by
                # is_squashed_doublenull=True, as keeping y-boundaries is not allowed
                is_squashed_doublenull = False

    # Avoid trying to open this file twice
    ds.close()

    return nxpe, nype, mxg, myg, mxsub, mysub, is_squashed_doublenull


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
        raise ValueError(
            "Each run directory does not contain an equal number "
            "of output files. If the parallelization scheme of "
            "your simulation changed partway-through, then please "
            "load each directory separately and concatenate them "
            "along the time dimension with xarray.concat()."
        )

    # Create list of lists of filepaths, so that xarray knows how they should
    # be concatenated by xarray.open_mfdataset()
    paths = iter(filepaths)
    paths_grid = [
        [[next(paths) for x in range(nxpe)] for y in range(nype)] for t in range(n_runs)
    ]

    # Dimensions along which no concatenation is needed are still present as
    # single-element lists, so need to concatenation along dim=None for those
    concat_dims = [None, None, None]
    if len(filepaths) > nprocs:
        concat_dims[0] = "t"
    if nype > 1:
        concat_dims[1] = "y"
    if nxpe > 1:
        concat_dims[2] = "x"

    return paths_grid, concat_dims


def _trim(ds, *, guards, keep_boundaries, nxpe, nype, is_restart):
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
    is_restart : bool
        Is data being loaded from restart files?
    """

    if any(keep_boundaries.values()):
        # Work out if this particular dataset contains any boundary cells
        lower_boundaries, upper_boundaries = _infer_contains_boundaries(ds, nxpe, nype)
    else:
        lower_boundaries, upper_boundaries = {}, {}

    selection = {}
    for dim in ds.dims:
        lower = _get_limit("lower", dim, keep_boundaries, lower_boundaries, guards)
        upper = _get_limit("upper", dim, keep_boundaries, upper_boundaries, guards)
        selection[dim] = slice(lower, upper)
    trimmed_ds = ds.isel(**selection)

    # Ignore FieldPerps for now
    for name in trimmed_ds:
        if trimmed_ds[name].dims == ("x", "z") or trimmed_ds[name].dims == (
            "t",
            "x",
            "z",
        ):
            trimmed_ds = trimmed_ds.drop_vars(name)

    to_drop = _BOUT_PER_PROC_VARIABLES

    return trimmed_ds.drop_vars(to_drop, errors="ignore")


def _infer_contains_boundaries(ds, nxpe, nype):
    """
    Uses the processor indices and BOUT++'s topology indices to work out whether this
    dataset contains boundary cells, and on which side.
    """

    if nxpe * nype == 1:
        # single file, always contains boundaries
        return {"x": True, "y": True}, {"x": True, "y": True}

    try:
        xproc = int(ds["PE_XIND"])
        yproc = int(ds["PE_YIND"])
    except KeyError:
        # output file from BOUT++ earlier than 4.3
        # Use knowledge that BOUT names its output files as /folder/prefix.num.nc, with a
        # numbering scheme
        # num = nxpe*i + j, where i={0, ..., nype}, j={0, ..., nxpe}
        filename = ds.encoding["source"]
        *prefix, filenum, extension = Path(filename).suffixes
        filenum = int(filenum.replace(".", ""))
        xproc = filenum % nxpe
        yproc = filenum // nxpe

    lower_boundaries, upper_boundaries = {}, {}

    lower_boundaries["x"] = xproc == 0
    upper_boundaries["x"] = xproc == nxpe - 1

    lower_boundaries["y"] = yproc == 0
    upper_boundaries["y"] = yproc == nype - 1

    jyseps2_1 = int(ds["jyseps2_1"])
    jyseps1_2 = int(ds["jyseps1_2"])
    if jyseps1_2 > jyseps2_1:
        # second divertor present
        ny_inner = int(ds["ny_inner"])
        mysub = int(ds["MYSUB"])
        if mysub * (yproc + 1) == ny_inner:
            upper_boundaries["y"] = True
        elif mysub * yproc == ny_inner:
            lower_boundaries["y"] = True

    return lower_boundaries, upper_boundaries


def _get_limit(side, dim, keep_boundaries, boundaries, guards):
    # Check for boundary cells, otherwise use guard cells, else leave alone

    if keep_boundaries.get(dim, False):
        if boundaries.get(dim, False):
            limit = None
        else:
            limit = guards[dim] if side == "lower" else -guards[dim]
    elif guards.get(dim, False):
        limit = guards[dim] if side == "lower" else -guards[dim]
    else:
        limit = None

    if limit == 0:
        # 0 would give incorrect result as an upper limit
        limit = None
    return limit


def _open_grid(datapath, chunks, keep_xboundaries, keep_yboundaries, mxg=2, **kwargs):
    """
    Opens a single grid file. Implements slightly different logic for
    boundaries to deal with different conventions in a BOUT grid file.
    """

    acceptable_dims = ["x", "y", "z"]

    # Passing 'chunks' with dimensions that are not present in the dataset causes an
    # error. A gridfile will be missing 't' and may be missing 'z' dimensions that dump
    # files have, so we must remove them from 'chunks'.
    grid_chunks = copy(chunks)
    unrecognised_chunk_dims = list(set(grid_chunks.keys()) - set(acceptable_dims))
    for dim in unrecognised_chunk_dims:
        del grid_chunks[dim]

    if _is_path(datapath):
        gridfilepath = Path(datapath)
        grid = xr.open_dataset(
            gridfilepath, engine=_check_filetype(gridfilepath), **kwargs
        )
    else:
        grid = datapath

    # TODO find out what 'yup_xsplit' etc are in the doublenull storm file John gave me
    # For now drop any variables with extra dimensions
    unrecognised_dims = list(set(grid.dims) - set(acceptable_dims))
    if len(unrecognised_dims) > 0:
        # Weird string formatting is a workaround to deal with possible bug in
        # pytest warnings capture - doesn't match strings containing brackets
        warn(
            "Will drop all variables containing the dimensions {} because "
            "they are not recognised".format(str(unrecognised_dims)[1:-1])
        )
        grid = grid.drop_dims(unrecognised_dims)

    if keep_xboundaries:
        # Set MXG so that it is picked up in metadata - needed for applying geometry,
        # etc.
        grid["MXG"] = mxg
    else:
        xboundaries = mxg
        if xboundaries > 0:
            grid = grid.isel(x=slice(xboundaries, -xboundaries, None))
        # Set MXG so that it is picked up in metadata - needed for applying geometry,
        # etc.
        grid["MXG"] = 0
    try:
        yboundaries = int(grid["y_boundary_guards"])
    except KeyError:
        # y_boundary_guards variable not in grid file - older grid files
        # never had y-boundary cells
        yboundaries = 0
    if keep_yboundaries:
        # Set MYG so that it is picked up in metadata - needed for applying geometry,
        # etc.
        grid["MYG"] = yboundaries
    else:
        if yboundaries > 0:
            # Remove y-boundary cells from first divertor target
            grid = grid.isel(y=slice(yboundaries, -yboundaries, None))
            if grid["jyseps1_2"] > grid["jyseps2_1"]:
                # There is a second divertor target, remove y-boundary cells
                # there too
                nin = int(grid["ny_inner"])
                grid_lower = grid.isel(y=slice(None, nin, None))
                grid_upper = grid.isel(y=slice(nin + 2 * yboundaries, None, None))
                grid = xr.concat(
                    (grid_lower, grid_upper),
                    dim="y",
                    data_vars="minimal",
                    compat="identical",
                    join="exact",
                )
        # Set MYG so that it is picked up in metadata - needed for applying geometry,
        # etc.
        grid["MYG"] = 0

    if "z" in grid_chunks and "z" not in grid.dims:
        del grid_chunks["z"]
    grid = grid.chunk(grid_chunks)

    return grid
