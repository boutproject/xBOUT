from copy import deepcopy
from boutdata.data import BoutOptionsFile
from gelidum import freeze
import inspect
import numpy as np
from pathlib import Path
from xarray import DataArray, Dataset


# Note, MYPE, PE_XIND and PE_YIND not included, since they are different for each
# processor and so are dropped when loading datasets.
METADATA_VARS = [
    "BOUT_VERSION",
    "NXPE",
    "NYPE",
    "NZPE",
    "MXG",
    "MYG",
    "MZG",
    "nx",
    "ny",
    "nz",
    "MZ",
    "MXSUB",
    "MYSUB",
    "MZSUB",
    "hist_hi",
    "iteration",
    "ixseps1",
    "ixseps2",
    "jyseps1_1",
    "jyseps1_2",
    "jyseps2_1",
    "jyseps2_2",
    "ny_inner",
    "tt",
    "zperiod",
    "ZMIN",
    "ZMAX",
    "use_metric_3d",
]


def load_example_input(name, *, nx=None, ny=None, nz=None):
    # Find input file in the 'inputs' subdirectory
    path = Path(__file__).parent.joinpath("inputs", name)

    return BoutOptionsFile(path, nx=nx, ny=ny, nz=nz)


def set_geometry_from_input_file(ds, name):
    ds = ds.copy()  # do not modify existing ds

    options = load_example_input(
        name, nx=ds.metadata["nx"], ny=ds.metadata["ny"], nz=ds.metadata["nz"]
    )

    # Set nx and ny in the options in case these are needed for expressions, as this is
    # not done automatically by BoutOptionsFile
    options["mesh:nx"] = ds.metadata["nx"]
    options["mesh:ny"] = ds.metadata["ny"]

    if not ds.metadata["MXG"] == options._keys.get("MXG", 2):
        raise ValueError(
            f"MXG does not match, {ds.metadata['MXG']} in ds and "
            f"{options._keys.get('MXG', 2)} in options"
        )
    if not ds.metadata["MYG"] == options._keys.get("MYG", 2):
        raise ValueError(
            f"MYG does not match, {ds.metadata['MYG']} in ds and "
            f"{options._keys.get('MYG', 2)} in options"
        )

    if ds.metadata["keep_xboundaries"] or ds.metadata["MXG"] == 0:
        slicex = slice(None)
    else:
        mxg = ds.metadata["MXG"]
        slicex = slice(mxg, -mxg)

    if ds.metadata["keep_yboundaries"] or ds.metadata["MYG"] == 0:
        slicey = slice(None)
    else:
        myg = ds.metadata["MYG"]
        slicey = slice(myg, -myg)

    slices = (slicex, slicey)

    # Replace values of fields already created by bout_xyt_example_files
    shape_2d = ds["g11"].shape
    for v in [
        "g11",
        "g22",
        "g33",
        "g12",
        "g13",
        "g23",
        "g_11",
        "g_22",
        "g_33",
        "g_12",
        "g_13",
        "g_23",
        "J",
        "Bxy",
        "zShift",
        "dx",
        "dy",
    ]:
        for location in ["CELL_CENTRE", "CELL_XLOW", "CELL_YLOW", "CELL_ZLOW"]:
            suffix = "" if location == "CELL_CENTRE" else f"_{location}"
            # Need all arrays returned from options.evaluate() to be the right shape.
            # Recommend adding '0*x' or '0*y' in the input file expressions if the
            # expression would be 1d otherwise.
            ds[v + suffix] = ds[v].copy(
                data=np.broadcast_to(
                    options.evaluate(f"mesh:{v}", location=location).squeeze(axis=2)[
                        slices
                    ],
                    shape_2d,
                )
            )

    # Set dz as it would be calculated by BOUT++ (but don't support zmin, zmax or
    # zperiod here)
    if "dz" in options["mesh"]:
        ds["dz"] = options.evaluate_scalar("mesh:dz")
    else:
        ds["dz"] = 2.0 * np.pi / ds.metadata["nz"]

    # Add extra fields needed by "toroidal" geometry
    for v in ["Rxy", "Zxy", "psixy"]:
        for location in ["CELL_CENTRE", "CELL_XLOW", "CELL_YLOW", "CELL_ZLOW"]:
            suffix = "" if location == "CELL_CENTRE" else f"_{location}"
            # Need all arrays returned from options.evaluate() to be the right shape.
            # Recommend adding '0*x' or '0*y' in the input file expressions if the
            # expression would be 1d otherwise.
            ds[v + suffix] = ds["g11"].copy(
                data=np.broadcast_to(
                    options.evaluate(f"mesh:{v}", location=location).squeeze(axis=2)[
                        slices
                    ],
                    shape_2d,
                )
            )

    # Set fields that don't have to be in input files to NaN
    for v in ["G1", "G2", "G3"]:
        if v in ds:
            ds[v] = ds[v].copy(data=np.broadcast_to(float("nan"), ds[v].shape))

    return ds, options


def _get_kwargs(ignore=None):
    """
    Get the arguments of a function as a frozendict. Extended version of code from here:
    https://stackoverflow.com/a/65927265

    Parameters
    ----------
    ignore : str or Sequence of str
        Arguments to drop when constructing the returned frozendict
    """
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    keys_to_ignore = ["self"]
    if ignore is not None:
        if isinstance(ignore, str):
            ignore = [ignore]
        keys_to_ignore += ignore
    for key in keys:
        if key not in keys_to_ignore:
            value = values[key]
            kwargs[key] = value

    return freeze(kwargs)


def create_bout_ds_list(
    prefix,
    lengths=(6, 2, 4, 7),
    nxpe=4,
    nype=2,
    nt=1,
    guards={},
    topology="core",
    syn_data_type="random",
    squashed=False,
    bout_v5=False,
    metric_3D=False,
):
    """
    Mocks up a set of BOUT-like datasets.

    Structured as though they were produced by a x-y parallelised run with multiple restarts.
    """

    if nt != 1:
        raise ValueError(
            "nt > 1 means the time dimension is split over several "
            + "directories. This is not implemented yet."
        )

    file_list = []
    ds_list = []
    for j in range(nype):
        for i in range(nxpe):
            num = i + nxpe * j
            filename = prefix + "." + str(num) + ".nc"
            file_list.append(filename)

            ds = create_bout_ds(
                syn_data_type=syn_data_type,
                num=num,
                lengths=lengths,
                nxpe=nxpe,
                nype=nype,
                xproc=i,
                yproc=j,
                guards=guards,
                topology=topology,
                squashed=squashed,
                bout_v5=bout_v5,
                metric_3D=metric_3D,
            )
            ds_list.append(ds)

    return ds_list, file_list


_create_bout_ds_cache = {}


def create_bout_ds(
    syn_data_type="random",
    lengths=(6, 2, 4, 7),
    num=0,
    nxpe=1,
    nype=1,
    xproc=0,
    yproc=0,
    guards=None,
    topology="core",
    squashed=False,
    bout_v5=False,
    metric_3D=False,
):
    call_args = _get_kwargs()

    try:
        # Has been called with the same signature before, just return the cached result
        return deepcopy(_create_bout_ds_cache[call_args])
    except KeyError:
        pass

    if metric_3D and not bout_v5:
        raise ValueError("3D metric requires BOUT++ v5")

    if guards is None:
        guards = {}

    # Set the shape of the data in this dataset
    t_length, x_length, y_length, z_length = lengths
    mxg = guards.get("x", 0)
    myg = guards.get("y", 0)
    x_length += 2 * mxg
    y_length += 2 * myg

    # calculate global nx, ny and nz
    nx = nxpe * lengths[1] + 2 * mxg
    ny = nype * lengths[2]
    nz = 1 * lengths[3]

    if squashed and "double-null" in topology:
        ny = ny + 2 * myg
        y_length = y_length + 2 * myg
    shape = (t_length, x_length, y_length, z_length)

    # Fill with some kind of synthetic data
    if syn_data_type == "random":
        # Each dataset contains unique random noise
        np.random.seed(seed=num)
        data = np.random.randn(*shape)
    elif syn_data_type == "linear":
        # Variables increase linearly across entire domain
        data = DataArray(-np.ones(shape), dims=("t", "x", "y", "z"))

        t_array = DataArray(
            (nx - 2 * mxg) * ny * nz * np.arange(t_length, dtype=float), dims="t"
        )
        x_array = DataArray(
            ny * nz * (xproc * lengths[1] + np.arange(lengths[1], dtype=float)),
            dims="x",
        )
        y_array = DataArray(
            nz * (yproc * lengths[2] + np.arange(lengths[2], dtype=float)), dims="y"
        )
        z_array = DataArray(np.arange(z_length, dtype=float), dims="z")

        data[:, mxg : x_length - mxg, myg : lengths[2] + myg, :] = (
            t_array + x_array + y_array + z_array
        )
    elif syn_data_type == "stepped":
        # Each dataset contains a different number depending on the filename
        data = np.ones(shape) * num
    elif isinstance(syn_data_type, int):
        data = np.ones(shape) * syn_data_type
    else:
        raise ValueError("Not a recognised choice of type of synthetic bout data.")

    T = DataArray(data, dims=["t", "x", "y", "z"])
    n = DataArray(data, dims=["t", "x", "y", "z"])
    S = DataArray(data[:, :, :, 0], dims=["t", "x", "y"])
    for v in [n, T]:
        v.attrs["direction_y"] = "Standard"
        v.attrs["cell_location"] = "CELL_CENTRE"
        v.attrs["direction_z"] = "Standard"
    for v in [S]:
        v.attrs["direction_y"] = "Standard"
        v.attrs["cell_location"] = "CELL_CENTRE"
        v.attrs["direction_z"] = "Average"
    ds = Dataset({"n": n, "T": T, "S": S})

    # BOUT_VERSION needed to deal with backwards incompatible changes:
    #
    # - v3 and earlier: number of points in z is MZ-1
    # - v4 and later: number of points in z is MZ
    # - v5 and later: metric components can be either 2D or 3D
    # - v5 and later: dz changed to be a Field2D/3D
    ds["BOUT_VERSION"] = 5.0 if bout_v5 else 4.3
    ds["use_metric_3d"] = int(metric_3D)

    # Include grid data
    ds["NXPE"] = nxpe
    ds["NYPE"] = nype
    ds["NZPE"] = 1
    ds["PE_XIND"] = xproc
    ds["PE_YIND"] = yproc
    ds["MYPE"] = num

    ds["MXG"] = mxg
    ds["MYG"] = myg
    ds["MZG"] = 0
    ds["nx"] = nx
    ds["ny"] = ny
    ds["nz"] = nz
    ds["MZ"] = 1 * lengths[3]
    if squashed:
        ds["MXSUB"] = lengths[1] // nxpe
        ds["MYSUB"] = lengths[2] // nype
        ds["MZSUB"] = lengths[3]
    else:
        ds["MXSUB"] = lengths[1]
        ds["MYSUB"] = lengths[2]
        ds["MZSUB"] = lengths[3]

    MYSUB = lengths[2]

    if topology == "core":
        ds["ixseps1"] = nx
        ds["ixseps2"] = nx
        ds["jyseps1_1"] = -1
        ds["jyseps2_1"] = ny // 2 - 1
        ds["jyseps1_2"] = ny // 2 - 1
        ds["jyseps2_2"] = ny - 1
        ds["ny_inner"] = ny // 2
    elif topology == "sol":
        ds["ixseps1"] = 0
        ds["ixseps2"] = 0
        ds["jyseps1_1"] = -1
        ds["jyseps2_1"] = ny // 2 - 1
        ds["jyseps1_2"] = ny // 2 - 1
        ds["jyseps2_2"] = ny - 1
        ds["ny_inner"] = ny // 2
    elif topology == "limiter":
        ds["ixseps1"] = nx // 2
        ds["ixseps2"] = nx
        ds["jyseps1_1"] = -1
        ds["jyseps2_1"] = ny // 2 - 1
        ds["jyseps1_2"] = ny // 2 - 1
        ds["jyseps2_2"] = ny - 1
        ds["ny_inner"] = ny // 2
    elif topology == "xpoint":
        if nype < 4 and not squashed:
            raise ValueError(f"Not enough processors for xpoint topology: nype={nype}")
        ds["ixseps1"] = nx // 2
        ds["ixseps2"] = nx // 2
        ds["jyseps1_1"] = MYSUB - 1
        ny_inner = 2 * MYSUB
        ds["ny_inner"] = ny_inner
        ds["jyseps2_1"] = MYSUB - 1
        ds["jyseps1_2"] = ny - MYSUB - 1
        ds["jyseps2_2"] = ny - MYSUB - 1
    elif topology == "single-null":
        if nype < 3 and not squashed:
            raise ValueError(f"Not enough processors for xpoint topology: nype={nype}")
        ds["ixseps1"] = nx // 2
        ds["ixseps2"] = nx
        ds["jyseps1_1"] = MYSUB - 1
        ds["jyseps2_1"] = ny // 2 - 1
        ds["jyseps1_2"] = ny // 2 - 1
        ds["jyseps2_2"] = ny - MYSUB - 1
        ds["ny_inner"] = ny // 2
    elif topology == "connected-double-null":
        if nype < 6 and not squashed:
            raise ValueError(
                "Not enough processors for connected-double-null topology: "
                f"nype={nype}"
            )
        ds["ixseps1"] = nx // 2
        ds["ixseps2"] = nx // 2
        ds["jyseps1_1"] = MYSUB - 1
        ny_inner = 3 * MYSUB
        ds["ny_inner"] = ny_inner
        ds["jyseps2_1"] = ny_inner - MYSUB - 1
        ds["jyseps1_2"] = ny_inner + MYSUB - 1
        ds["jyseps2_2"] = ny - MYSUB - 1
    elif topology == "lower-disconnected-double-null":
        if nype < 6 and not squashed:
            raise ValueError(
                "Not enough processors for lower-disconnected-double-null "
                f"topology: nype={nype}"
            )
        ds["ixseps1"] = nx // 2
        ds["ixseps2"] = nx // 2 + 4
        if ds["ixseps2"] >= nx:
            raise ValueError(
                "Not enough points in the x-direction. ixseps2="
                f'{ds["ixseps2"]} > nx={nx}'
            )
        ds["jyseps1_1"] = MYSUB - 1
        ny_inner = 3 * MYSUB
        ds["ny_inner"] = ny_inner
        ds["jyseps2_1"] = ny_inner - MYSUB - 1
        ds["jyseps1_2"] = ny_inner + MYSUB - 1
        ds["jyseps2_2"] = ny - MYSUB - 1
    elif topology == "upper-disconnected-double-null":
        if nype < 6 and not squashed:
            raise ValueError(
                "Not enough processors for upper-disconnected-double-null "
                f"topology: nype={nype}"
            )
        ds["ixseps2"] = nx // 2
        ds["ixseps1"] = nx // 2 + 4
        if ds["ixseps2"] >= nx:
            raise ValueError(
                "Not enough points in the x-direction. ixseps2="
                f'{ds["ixseps2"]} > nx={nx}'
            )
        ds["jyseps1_1"] = MYSUB - 1
        ny_inner = 3 * MYSUB
        ds["ny_inner"] = ny_inner
        ds["jyseps2_1"] = ny_inner - MYSUB - 1
        ds["jyseps1_2"] = ny_inner + MYSUB - 1
        ds["jyseps2_2"] = ny - MYSUB - 1
    else:
        raise ValueError(f"Unrecognised topology={topology}")

    if metric_3D:
        one = DataArray(np.ones((x_length, y_length, z_length)), dims=["x", "y", "z"])
        zero = DataArray(np.zeros((x_length, y_length, z_length)), dims=["x", "y", "z"])
    else:
        one = DataArray(np.ones((x_length, y_length)), dims=["x", "y"])
        zero = DataArray(np.zeros((x_length, y_length)), dims=["x", "y"])

    ds["zperiod"] = 1
    ds["ZMIN"] = 0.0
    ds["ZMAX"] = 1.0
    ds["g11"] = one
    ds["g22"] = one
    ds["g33"] = one
    ds["g12"] = zero
    ds["g13"] = zero
    ds["g23"] = zero
    ds["g_11"] = one
    ds["g_22"] = one
    ds["g_33"] = one
    ds["g_12"] = zero
    ds["g_13"] = zero
    ds["g_23"] = zero
    ds["G1"] = zero
    ds["G2"] = zero
    ds["G3"] = zero
    ds["J"] = one
    ds["Bxy"] = one
    ds["zShift"] = zero

    ds["dx"] = 0.5 * one
    ds["dy"] = 2.0 * one
    if bout_v5:
        ds["dz"] = 2.0 * one * np.pi / nz
    else:
        ds["dz"] = 2.0 * np.pi / nz

    ds["iteration"] = t_length - 1
    ds["hist_hi"] = t_length - 1
    ds["t_array"] = DataArray(np.arange(t_length, dtype=float) * 10.0, dims="t")
    ds["tt"] = ds["t_array"][-1]

    # xarray adds this encoding when opening a file. Emulate here as it may be used to
    # get the file number
    ds.encoding["source"] = f"BOUT.dmp.{num}.nc"

    _create_bout_ds_cache[call_args] = ds
    return deepcopy(ds)


_create_bout_grid_ds_cache = {}


def create_bout_grid_ds(xsize=2, ysize=4, guards={}, topology="core", ny_inner=0):
    call_args = _get_kwargs()

    try:
        # Has been called with the same signature before, just return the cached result
        return deepcopy(_create_bout_grid_ds_cache[call_args])
    except KeyError:
        pass

    # Set the shape of the data in this dataset
    mxg = guards.get("x", 0)
    myg = guards.get("y", 0)
    xsize += 2 * mxg
    ysize += 2 * myg

    # jyseps* from grid file only ever used to check topology when loading the grid file,
    # so do not need to be consistent with the main dataset
    jyseps2_1 = ysize // 2
    jyseps1_2 = jyseps2_1

    if "double-null" in topology or "xpoint" in topology:
        # Has upper target as well
        ysize += 2 * myg

        # make different from jyseps2_1 so double-null toplogy is recognised
        jyseps1_2 += 1

    shape = (xsize, ysize)

    data = DataArray(np.ones(shape), dims=["x", "y"])

    ds = Dataset(
        {
            "psixy": data,
            "Rxy": data,
            "Zxy": data,
            "hthe": data,
            "y_boundary_guards": myg,
            "jyseps2_1": jyseps2_1,
            "jyseps1_2": jyseps1_2,
            "ny_inner": ny_inner,
        }
    )

    _create_bout_grid_ds_cache[call_args] = ds
    return deepcopy(ds)
