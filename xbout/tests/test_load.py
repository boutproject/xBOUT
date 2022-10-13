from collections import namedtuple
from copy import deepcopy
import inspect
from pathlib import Path
import re
from functools import reduce
import operator

import pytest

import numpy as np

from xarray import DataArray, Dataset, concat
from xarray.tests.test_dataset import create_test_data
import xarray.testing as xrt

from natsort import natsorted

from xbout.load import (
    _check_filetype,
    _expand_wildcards,
    _expand_filepaths,
    _arrange_for_concatenation,
    _trim,
    _infer_contains_boundaries,
    open_boutdataset,
    _BOUT_PER_PROC_VARIABLES,
    _BOUT_TIME_DEPENDENT_META_VARS,
)
from xbout.utils import _separate_metadata
from xbout.tests.utils_for_tests import _get_kwargs


def test_check_extensions(tmp_path):
    files_dir = tmp_path.joinpath("data")
    files_dir.mkdir()
    example_nc_file = files_dir.joinpath("example.nc")
    example_nc_file.write_text("content_nc")

    filetype = _check_filetype(example_nc_file)
    assert filetype == "netcdf4"

    example_hdf5_file = files_dir.joinpath("example.h5netcdf")
    example_hdf5_file.write_text("content_hdf5")

    filetype = _check_filetype(example_hdf5_file)
    assert filetype == "h5netcdf"

    example_invalid_file = files_dir.joinpath("example.txt")
    example_hdf5_file.write_text("content_txt")
    with pytest.raises(IOError):
        filetype = _check_filetype(example_invalid_file)


def test_set_fci_coords(create_example_grid_file_fci, create_example_files_fci):
    grid = create_example_grid_file_fci
    data = create_example_files_fci

    ds = open_boutdataset(data, gridfilepath=grid, geometry="fci")
    assert "R" in ds
    assert "Z" in ds


class TestPathHandling:
    def test_glob_expansion_single(self, tmp_path):
        files_dir = tmp_path.joinpath("data")
        files_dir.mkdir()
        example_file = files_dir.joinpath("example.0.nc")
        example_file.write_text("content")

        path = example_file
        filepaths = _expand_wildcards(path)
        assert filepaths[0] == example_file

        path = files_dir.joinpath("example.*.nc")
        filepaths = _expand_wildcards(path)
        assert filepaths[0] == example_file

    @pytest.mark.parametrize(
        "ii, jj", [(1, 1), (1, 4), (3, 1), (5, 3), (12, 1), (1, 12), (121, 2), (3, 111)]
    )
    def test_glob_expansion_both(self, tmp_path, ii, jj):
        files_dir = tmp_path.joinpath("data")
        files_dir.mkdir()
        filepaths = []
        for i in range(ii):
            example_run_dir = files_dir.joinpath("run" + str(i))
            example_run_dir.mkdir()
            for j in range(jj):
                example_file = example_run_dir.joinpath("example." + str(j) + ".nc")
                example_file.write_text("content")
                filepaths.append(example_file)
        expected_filepaths = natsorted(filepaths, key=lambda filepath: str(filepath))

        path = files_dir.joinpath("run*/example.*.nc")
        actual_filepaths = _expand_wildcards(path)

        assert actual_filepaths == expected_filepaths

    @pytest.mark.parametrize(
        "ii, jj", [(1, 1), (1, 4), (3, 1), (5, 3), (1, 12), (3, 111)]
    )
    def test_glob_expansion_brackets(self, tmp_path, ii, jj):
        files_dir = tmp_path.joinpath("data")
        files_dir.mkdir()
        filepaths = []
        for i in range(ii):
            example_run_dir = files_dir.joinpath("run" + str(i))
            example_run_dir.mkdir()
            for j in range(jj):
                example_file = example_run_dir.joinpath("example." + str(j) + ".nc")
                example_file.write_text("content")
                filepaths.append(example_file)
        expected_filepaths = natsorted(filepaths, key=lambda filepath: str(filepath))

        path = files_dir.joinpath("run[1-9]/example.*.nc")
        actual_filepaths = _expand_wildcards(path)

        assert actual_filepaths == expected_filepaths[jj:]

    def test_no_files(self, tmp_path):
        files_dir = tmp_path.joinpath("data")
        files_dir.mkdir()

        with pytest.raises(IOError):
            path = files_dir.joinpath("run*/example.*.nc")
            actual_filepaths = _expand_filepaths(path)


@pytest.fixture()
def create_filepaths():
    return _create_filepaths


def _create_filepaths(nxpe=1, nype=1, nt=1):
    filepaths = []
    for t in range(nt):
        for i in range(nype):
            for j in range(nxpe):
                file_num = j + nxpe * i
                path = "./run{}".format(str(t)) + "/BOUT.dmp.{}.nc".format(
                    str(file_num)
                )
                filepaths.append(path)

    return filepaths


class TestArrange:
    def test_arrange_single(self, create_filepaths):
        paths = create_filepaths(nxpe=1, nype=1, nt=1)
        expected_path_grid = [[["./run0/BOUT.dmp.0.nc"]]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=1, nype=1
        )
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == [None, None, None]

    def test_arrange_along_x(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=1, nt=1)
        expected_path_grid = [
            [["./run0/BOUT.dmp.0.nc", "./run0/BOUT.dmp.1.nc", "./run0/BOUT.dmp.2.nc"]]
        ]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=1
        )
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == [None, None, "x"]

    def test_arrange_along_y(self, create_filepaths):
        paths = create_filepaths(nxpe=1, nype=3, nt=1)
        expected_path_grid = [
            [
                ["./run0/BOUT.dmp.0.nc"],
                ["./run0/BOUT.dmp.1.nc"],
                ["./run0/BOUT.dmp.2.nc"],
            ]
        ]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=1, nype=3
        )
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == [None, "y", None]

    def test_arrange_along_t(self, create_filepaths):
        paths = create_filepaths(nxpe=1, nype=1, nt=3)
        expected_path_grid = [
            [["./run0/BOUT.dmp.0.nc"]],
            [["./run1/BOUT.dmp.0.nc"]],
            [["./run2/BOUT.dmp.0.nc"]],
        ]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=1, nype=1
        )
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ["t", None, None]

    def test_arrange_along_xy(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=2, nt=1)
        expected_path_grid = [
            [
                [
                    "./run0/BOUT.dmp.0.nc",
                    "./run0/BOUT.dmp.1.nc",
                    "./run0/BOUT.dmp.2.nc",
                ],
                [
                    "./run0/BOUT.dmp.3.nc",
                    "./run0/BOUT.dmp.4.nc",
                    "./run0/BOUT.dmp.5.nc",
                ],
            ]
        ]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=2
        )
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == [None, "y", "x"]

    def test_arrange_along_xt(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=1, nt=2)
        expected_path_grid = [
            [["./run0/BOUT.dmp.0.nc", "./run0/BOUT.dmp.1.nc", "./run0/BOUT.dmp.2.nc"]],
            [["./run1/BOUT.dmp.0.nc", "./run1/BOUT.dmp.1.nc", "./run1/BOUT.dmp.2.nc"]],
        ]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=1
        )
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ["t", None, "x"]

    def test_arrange_along_xyt(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=2, nt=2)
        expected_path_grid = [
            [
                [
                    "./run0/BOUT.dmp.0.nc",
                    "./run0/BOUT.dmp.1.nc",
                    "./run0/BOUT.dmp.2.nc",
                ],
                [
                    "./run0/BOUT.dmp.3.nc",
                    "./run0/BOUT.dmp.4.nc",
                    "./run0/BOUT.dmp.5.nc",
                ],
            ],
            [
                [
                    "./run1/BOUT.dmp.0.nc",
                    "./run1/BOUT.dmp.1.nc",
                    "./run1/BOUT.dmp.2.nc",
                ],
                [
                    "./run1/BOUT.dmp.3.nc",
                    "./run1/BOUT.dmp.4.nc",
                    "./run1/BOUT.dmp.5.nc",
                ],
            ],
        ]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=2
        )
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ["t", "y", "x"]


@pytest.fixture(scope="session")
def bout_xyt_example_files(tmp_path_factory):
    return _bout_xyt_example_files


_bout_xyt_example_files_cache = {}


def _bout_xyt_example_files(
    tmp_path_factory,
    prefix="BOUT.dmp",
    lengths=(6, 2, 4, 7),
    nxpe=4,
    nype=2,
    nt=1,
    guards=None,
    syn_data_type="random",
    grid=None,
    squashed=False,
    topology="core",
    write_to_disk=False,
    bout_v5=False,
    metric_3D=False,
):
    """
    Mocks up a set of BOUT-like Datasets

    Either returns list of Datasets (if write_to_disk=False)
    or writes Datasets to netCDF files and returns the temporary test directory
    containing them, deleting the temporary directory once that test is done (if
    write_to_disk=True).
    """
    call_args = _get_kwargs(ignore="tmp_path_factory")

    try:
        # Has been called with the same signature before, just return the cached result
        return deepcopy(_bout_xyt_example_files_cache[call_args])
    except KeyError:
        pass

    if guards is None:
        guards = {}

    mxg = guards.get("x", 0)
    myg = guards.get("y", 0)

    if squashed:
        # create a single data-file, but alter the 'nxpe' and 'nype' variables, as if the
        # file had been created by combining a set of BOUT.dmp.*.nc files
        this_lengths = (
            lengths[0],
            lengths[1] * nxpe,
            lengths[2] * nype,
            lengths[3],
        )
        ds_list, file_list = create_bout_ds_list(
            prefix=prefix,
            lengths=this_lengths,
            nxpe=1,
            nype=1,
            nt=nt,
            guards=guards,
            topology=topology,
            syn_data_type=syn_data_type,
            squashed=True,
            bout_v5=bout_v5,
            metric_3D=metric_3D,
        )
        ds_list[0]["nxpe"] = nxpe
        ds_list[0]["nype"] = nype
    else:
        ds_list, file_list = create_bout_ds_list(
            prefix=prefix,
            lengths=lengths,
            nxpe=nxpe,
            nype=nype,
            nt=nt,
            guards=guards,
            topology=topology,
            syn_data_type=syn_data_type,
            bout_v5=bout_v5,
            metric_3D=metric_3D,
        )

    if grid is not None:
        xsize = lengths[1] * nxpe
        ysize = lengths[2] * nype
        grid_ds = create_bout_grid_ds(
            xsize=xsize,
            ysize=ysize,
            guards=guards,
            topology=topology,
            ny_inner=3 * lengths[2],
        )

    if not write_to_disk:
        if grid is None:
            _bout_xyt_example_files_cache[call_args] = ds_list
            return deepcopy(ds_list)
        else:
            _bout_xyt_example_files_cache[call_args] = ds_list, grid_ds
            return deepcopy((ds_list, grid_ds))
        raise ValueError("tmp_path_factory required when write_to_disk=True")

    save_dir = tmp_path_factory.mktemp("data")

    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(save_dir.joinpath(file_name))

    if grid is not None:
        grid_ds.to_netcdf(save_dir.joinpath(grid + ".nc"))

    # Return a glob-like path to all files created, which has all file numbers replaced
    # with a single asterix
    path = str(save_dir.joinpath(file_list[-1]))

    count = 1
    if nt > 1:
        count += 1
    # We have to reverse the path before limiting the number of numbers replaced so that the
    # tests don't get confused by pytest's persistent temporary directories (which are also designated
    # by different numbers)
    glob_pattern = Path((re.sub(r"\d+", "*", path[::-1], count=count))[::-1])
    _bout_xyt_example_files_cache[call_args] = glob_pattern
    return glob_pattern


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

            # Include guard cells
            upper_bndry_cells = {dim: guards.get(dim) for dim in guards.keys()}
            lower_bndry_cells = {dim: guards.get(dim) for dim in guards.keys()}

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

    extra_boundary_points = 0

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
            "y_boundary_guards": myg,
        }
    )

    _create_bout_grid_ds_cache[call_args] = ds
    return deepcopy(ds)


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


class TestStripMetadata:
    def test_strip_metadata(self):

        original = create_bout_ds()
        assert original["NXPE"] == 1

        ds, metadata = _separate_metadata(original)

        assert original.drop_vars(
            METADATA_VARS + _BOUT_PER_PROC_VARIABLES + _BOUT_TIME_DEPENDENT_META_VARS,
            errors="ignore",
        ).equals(ds)
        assert metadata["NXPE"] == 1


# TODO also test loading multiple files which have guard cells
class TestOpen:
    def test_single_file(self, tmp_path_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmp_path_factory, nxpe=1, nype=1, nt=1, write_to_disk=True
        )
        with pytest.warns(UserWarning):
            actual = open_boutdataset(datapath=path, keep_xboundaries=False)
        expected = create_bout_ds()
        expected = expected.set_coords(["t_array", "dx", "dy", "dz"]).rename(
            t_array="t"
        )
        xrt.assert_equal(
            actual.drop_vars(["x", "y", "z"]).load(),
            expected.drop_vars(
                METADATA_VARS
                + _BOUT_PER_PROC_VARIABLES
                + _BOUT_TIME_DEPENDENT_META_VARS,
                errors="ignore",
            ),
        )

        # check creation without writing to disk gives identical result
        fake_ds_list = bout_xyt_example_files(None, nxpe=1, nype=1, nt=1)
        with pytest.warns(UserWarning):
            fake = open_boutdataset(datapath=fake_ds_list, keep_xboundaries=False)
        xrt.assert_identical(actual, fake)

    def test_squashed_file(self, tmp_path_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmp_path_factory, nxpe=4, nype=3, nt=1, squashed=True, write_to_disk=True
        )
        with pytest.warns(UserWarning):
            actual = open_boutdataset(datapath=path, keep_xboundaries=False)
        expected = create_bout_ds(lengths=(6, 8, 12, 7))
        expected = expected.set_coords(["t_array", "dx", "dy", "dz"]).rename(
            t_array="t"
        )
        xrt.assert_equal(
            actual.drop_vars(["x", "y", "z"]).load(),
            expected.drop_vars(
                METADATA_VARS
                + _BOUT_PER_PROC_VARIABLES
                + _BOUT_TIME_DEPENDENT_META_VARS,
                errors="ignore",
            ),
        )

        # check creation without writing to disk gives identical result
        fake_ds_list = bout_xyt_example_files(None, nxpe=4, nype=3, nt=1, squashed=True)
        with pytest.warns(UserWarning):
            fake = open_boutdataset(datapath=fake_ds_list, keep_xboundaries=False)
        xrt.assert_identical(actual, fake)

    @pytest.mark.parametrize(
        "keep_xboundaries", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    @pytest.mark.parametrize(
        "keep_yboundaries", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_squashed_doublenull(
        self,
        tmp_path_factory,
        bout_xyt_example_files,
        keep_xboundaries,
        keep_yboundaries,
    ):
        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=4,
            nype=6,
            nt=1,
            lengths=(6, 2, 4, 7),
            guards={"x": 2, "y": 2},
            squashed=True,
            topology="lower-disconnected-double-null",
        )
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=path,
                keep_xboundaries=keep_xboundaries,
                keep_yboundaries=keep_yboundaries,
            )

        # bout_xyt_example_files when creating a 'squashed' file just makes it with
        # y-size nype*lengths[2]+2*myg, which is 6*4+4=28, so with upper and lower
        # boundaries removed, y-size should be 28-4*2=20.
        assert ds.sizes["t"] == 6
        assert ds.sizes["x"] == 12 if keep_xboundaries else 8
        assert ds.sizes["y"] == 32 if keep_yboundaries else 24
        assert ds.sizes["z"] == 7

    @pytest.mark.parametrize(
        "keep_xboundaries", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    @pytest.mark.parametrize(
        "keep_yboundaries", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_squashed_doublenull_file(
        self,
        tmp_path_factory,
        bout_xyt_example_files,
        keep_xboundaries,
        keep_yboundaries,
    ):
        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=4,
            nype=6,
            nt=1,
            lengths=(6, 4, 4, 7),
            guards={"x": 2, "y": 2},
            squashed=True,
            write_to_disk=True,
            topology="upper-disconnected-double-null",
        )
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=path,
                keep_xboundaries=keep_xboundaries,
                keep_yboundaries=keep_yboundaries,
            )

        # bout_xyt_example_files when creating a 'squashed' file just makes it with
        # y-size nype*lengths[2]+2*myg, which is 6*4+4=28, so with upper and lower
        # boundaries removed, y-size should be 28-4*2=20.
        assert ds.sizes["t"] == 6
        assert ds.sizes["x"] == 20 if keep_xboundaries else 16
        assert ds.sizes["y"] == 32 if keep_yboundaries else 24
        assert ds.sizes["z"] == 7

    def test_combine_along_x(self, tmp_path_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=4,
            nype=1,
            nt=1,
            syn_data_type="stepped",
            write_to_disk=True,
        )
        with pytest.warns(UserWarning):
            actual = open_boutdataset(datapath=path, keep_xboundaries=False)

        bout_ds = create_bout_ds
        expected = concat(
            [bout_ds(0), bout_ds(1), bout_ds(2), bout_ds(3)],
            dim="x",
            data_vars="minimal",
        )
        expected = expected.set_coords(["t_array", "dx", "dy", "dz"]).rename(
            t_array="t"
        )
        xrt.assert_equal(
            actual.drop_vars(["x", "y", "z"]).load(),
            expected.drop_vars(
                METADATA_VARS + _BOUT_PER_PROC_VARIABLES, errors="ignore"
            ),
        )

        # check creation without writing to disk gives identical result
        fake_ds_list = bout_xyt_example_files(
            None, nxpe=4, nype=1, nt=1, syn_data_type="stepped"
        )
        with pytest.warns(UserWarning):
            fake = open_boutdataset(datapath=fake_ds_list, keep_xboundaries=False)
        xrt.assert_identical(actual, fake)

    def test_combine_along_y(self, tmp_path_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=1,
            nype=3,
            nt=1,
            syn_data_type="stepped",
            write_to_disk=True,
        )
        with pytest.warns(UserWarning):
            actual = open_boutdataset(datapath=path, keep_xboundaries=False)

        bout_ds = create_bout_ds
        expected = concat(
            [bout_ds(0), bout_ds(1), bout_ds(2)], dim="y", data_vars="minimal"
        )
        expected = expected.set_coords(["t_array", "dx", "dy", "dz"]).rename(
            t_array="t"
        )
        xrt.assert_equal(
            actual.drop_vars(["x", "y", "z"]).load(),
            expected.drop_vars(
                METADATA_VARS + _BOUT_PER_PROC_VARIABLES, errors="ignore"
            ),
        )

        # check creation without writing to disk gives identical result
        fake_ds_list = bout_xyt_example_files(
            None, nxpe=1, nype=3, nt=1, syn_data_type="stepped"
        )
        with pytest.warns(UserWarning):
            fake = open_boutdataset(datapath=fake_ds_list, keep_xboundaries=False)
        xrt.assert_identical(actual, fake)

    @pytest.mark.skip
    def test_combine_along_t(self):
        ...

    @pytest.mark.parametrize(
        "bout_v5,metric_3D", [(False, False), (True, False), (True, True)]
    )
    @pytest.mark.parametrize("lengths", [(6, 2, 4, 7), (6, 2, 4, 1)])
    def test_combine_along_xy(
        self, tmp_path_factory, bout_xyt_example_files, bout_v5, metric_3D, lengths
    ):
        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=4,
            nype=3,
            nt=1,
            lengths=lengths,
            syn_data_type="stepped",
            write_to_disk=True,
            bout_v5=bout_v5,
            metric_3D=metric_3D,
        )
        with pytest.warns(UserWarning):
            actual = open_boutdataset(datapath=path, keep_xboundaries=False)

        def bout_ds(syn_data_type):
            return create_bout_ds(
                syn_data_type, bout_v5=bout_v5, metric_3D=metric_3D, lengths=lengths
            )

        line1 = concat(
            [bout_ds(0), bout_ds(1), bout_ds(2), bout_ds(3)],
            dim="x",
            data_vars="minimal",
        )
        line2 = concat(
            [bout_ds(4), bout_ds(5), bout_ds(6), bout_ds(7)],
            dim="x",
            data_vars="minimal",
        )
        line3 = concat(
            [bout_ds(8), bout_ds(9), bout_ds(10), bout_ds(11)],
            dim="x",
            data_vars="minimal",
        )
        expected = concat([line1, line2, line3], dim="y", data_vars="minimal")
        expected = expected.set_coords(["t_array", "dx", "dy", "dz"]).rename(
            t_array="t"
        )
        vars_to_drop = METADATA_VARS + _BOUT_PER_PROC_VARIABLES
        xrt.assert_equal(
            actual.drop_vars(["x", "y", "z"]).load(),
            expected.drop_vars(vars_to_drop, errors="ignore"),
        )

        # check creation without writing to disk gives identical result
        fake_ds_list = bout_xyt_example_files(
            None,
            nxpe=4,
            nype=3,
            nt=1,
            lengths=lengths,
            syn_data_type="stepped",
            bout_v5=bout_v5,
            metric_3D=metric_3D,
        )
        with pytest.warns(UserWarning):
            fake = open_boutdataset(datapath=fake_ds_list, keep_xboundaries=False)
        xrt.assert_identical(actual, fake)

    def test_toroidal(self, tmp_path_factory, bout_xyt_example_files):
        # actually write these to disk to test the loading fully
        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=3,
            nype=3,
            nt=1,
            syn_data_type="stepped",
            grid="grid",
            write_to_disk=True,
        )
        actual = open_boutdataset(
            datapath=path,
            geometry="toroidal",
            gridfilepath=path.parent.joinpath("grid.nc"),
        )

        # check dataset can be saved
        save_dir = tmp_path_factory.mktemp("data")
        actual.bout.save(save_dir.joinpath("boutdata.nc"))

        # check creation without writing to disk gives identical result
        fake_ds_list, fake_grid_ds = bout_xyt_example_files(
            None,
            nxpe=3,
            nype=3,
            nt=1,
            syn_data_type="stepped",
            grid="grid",
        )
        fake = open_boutdataset(
            datapath=fake_ds_list, geometry="toroidal", gridfilepath=fake_grid_ds
        )
        xrt.assert_identical(actual, fake)

    def test_salpha(self, tmp_path_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=3,
            nype=3,
            nt=1,
            syn_data_type="stepped",
            grid="grid",
            write_to_disk=True,
        )
        actual = open_boutdataset(
            datapath=path,
            geometry="s-alpha",
            gridfilepath=path.parent.joinpath("grid.nc"),
        )

        # check dataset can be saved
        save_dir = tmp_path_factory.mktemp("data")
        actual.bout.save(save_dir.joinpath("boutdata.nc"))

        # check creation without writing to disk gives identical result
        fake_ds_list, fake_grid_ds = bout_xyt_example_files(
            None, nxpe=3, nype=3, nt=1, syn_data_type="stepped", grid="grid"
        )
        fake = open_boutdataset(
            datapath=fake_ds_list, geometry="s-alpha", gridfilepath=fake_grid_ds
        )
        xrt.assert_identical(actual, fake)

    def test_drop_vars(self, tmp_path_factory, bout_xyt_example_files):
        datapath = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=4,
            nype=1,
            nt=1,
            syn_data_type="stepped",
            write_to_disk=True,
        )
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=datapath, keep_xboundaries=False, drop_variables=["T"]
            )

        assert "T" not in ds.keys()
        assert "n" in ds.keys()

    @pytest.mark.skip
    def test_combine_along_tx(self):
        ...

    def test_restarts(self):
        datapath = Path(__file__).parent.joinpath(
            "data", "restart", "BOUT.restart.*.nc"
        )
        ds = open_boutdataset(datapath, keep_xboundaries=True, keep_yboundaries=True)

        assert "T" in ds


_test_processor_layouts_list = [
    # No parallelization
    (0, 0, 1, 1, {"x": True, "y": True}, {"x": True, "y": True}),
    # 1d parallelization along x:
    # Left
    (0, 0, 3, 1, {"x": True, "y": True}, {"x": False, "y": True}),
    # Middle
    (1, 0, 3, 1, {"x": False, "y": True}, {"x": False, "y": True}),
    # Right
    (2, 0, 3, 1, {"x": False, "y": True}, {"x": True, "y": True}),
    # 1d parallelization along y:
    # Bottom
    (0, 0, 1, 3, {"x": True, "y": True}, {"x": True, "y": False}),
    # Middle
    (0, 1, 1, 3, {"x": True, "y": False}, {"x": True, "y": False}),
    # Top
    (0, 2, 1, 3, {"x": True, "y": False}, {"x": True, "y": True}),
    # 2d parallelization:
    # Bottom left corner
    (0, 0, 3, 4, {"x": True, "y": True}, {"x": False, "y": False}),
    # Bottom right corner
    (2, 0, 3, 4, {"x": False, "y": True}, {"x": True, "y": False}),
    # Top left corner
    (0, 3, 3, 4, {"x": True, "y": False}, {"x": False, "y": True}),
    # Top right corner
    (2, 3, 3, 4, {"x": False, "y": False}, {"x": True, "y": True}),
    # Centre
    (1, 2, 3, 4, {"x": False, "y": False}, {"x": False, "y": False}),
    # Left side
    (0, 2, 3, 4, {"x": True, "y": False}, {"x": False, "y": False}),
    # Right side
    (2, 2, 3, 4, {"x": False, "y": False}, {"x": True, "y": False}),
    # Bottom side
    (1, 0, 3, 4, {"x": False, "y": True}, {"x": False, "y": False}),
    # Top side
    (1, 3, 3, 4, {"x": False, "y": False}, {"x": False, "y": True}),
]

_test_processor_layouts_doublenull_list = [
    # 1d parallelization along y:
    # Bottom
    (0, 0, 1, 4, {"x": True, "y": True}, {"x": True, "y": False}),
    # Lower Middle
    (0, 1, 1, 4, {"x": True, "y": False}, {"x": True, "y": True}),
    # Upper Middle
    (0, 2, 1, 4, {"x": True, "y": True}, {"x": True, "y": False}),
    # Top
    (0, 3, 1, 4, {"x": True, "y": False}, {"x": True, "y": True}),
    # 2d parallelization:
    # Bottom left corner
    (0, 0, 3, 4, {"x": True, "y": True}, {"x": False, "y": False}),
    (1, 0, 3, 4, {"x": False, "y": True}, {"x": False, "y": False}),
    # Bottom right corner
    (2, 0, 3, 4, {"x": False, "y": True}, {"x": True, "y": False}),
    (0, 1, 3, 4, {"x": True, "y": False}, {"x": False, "y": True}),
    (1, 1, 3, 4, {"x": False, "y": False}, {"x": False, "y": True}),
    (2, 1, 3, 4, {"x": False, "y": False}, {"x": True, "y": True}),
    (0, 2, 3, 4, {"x": True, "y": True}, {"x": False, "y": False}),
    (1, 2, 3, 4, {"x": False, "y": True}, {"x": False, "y": False}),
    (2, 2, 3, 4, {"x": False, "y": True}, {"x": True, "y": False}),
    # Top left corner
    (0, 3, 3, 4, {"x": True, "y": False}, {"x": False, "y": True}),
    (1, 3, 3, 4, {"x": False, "y": False}, {"x": False, "y": True}),
    # Top right corner
    (2, 3, 3, 4, {"x": False, "y": False}, {"x": True, "y": True}),
]


class TestTrim:
    @pytest.mark.parametrize("is_restart", [False, True])
    def test_no_trim(self, is_restart):
        ds = create_test_data(0)
        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp.0.nc"
        actual = _trim(
            ds, guards={}, keep_boundaries={}, nxpe=1, nype=1, is_restart=is_restart
        )
        xrt.assert_equal(actual, ds)

    def test_trim_guards(self):
        ds = create_test_data(0)
        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp.0.nc"
        actual = _trim(
            ds, guards={"time": 2}, keep_boundaries={}, nxpe=1, nype=1, is_restart=False
        )
        selection = {"time": slice(2, -2)}
        expected = ds.isel(**selection)
        xrt.assert_equal(expected, actual)

    @pytest.mark.parametrize(
        "xproc, yproc, nxpe, nype, lower_boundaries, upper_boundaries",
        _test_processor_layouts_list,
    )
    def test_infer_boundaries_2d_parallelization(
        self, xproc, yproc, nxpe, nype, lower_boundaries, upper_boundaries
    ):
        """
        Numbering scheme for nxpe=3, nype=4

        y  9 10 11
        ^  6 7  8
        |  3 4  5
        |  0 1  2
         -----> x
        """

        ds = create_test_data(0)
        ds["jyseps2_1"] = 0
        ds["jyseps1_2"] = 0
        ds["PE_XIND"] = xproc
        ds["PE_YIND"] = yproc
        actual_lower_boundaries, actual_upper_boundaries = _infer_contains_boundaries(
            ds, nxpe, nype
        )

        assert actual_lower_boundaries == lower_boundaries
        assert actual_upper_boundaries == upper_boundaries

    @pytest.mark.parametrize(
        "xproc, yproc, nxpe, nype, lower_boundaries, upper_boundaries",
        _test_processor_layouts_doublenull_list,
    )
    def test_infer_boundaries_2d_parallelization_doublenull(
        self, xproc, yproc, nxpe, nype, lower_boundaries, upper_boundaries
    ):
        """
        Numbering scheme for nxpe=3, nype=4

        y  9 10 11
        ^  6 7  8
        |  3 4  5
        |  0 1  2
         -----> x
        """

        ds = create_test_data(0)
        ds["jyseps2_1"] = 3
        ds["jyseps1_2"] = 11
        ds["ny_inner"] = 8
        ds["MYSUB"] = 4
        ds["PE_XIND"] = xproc
        ds["PE_YIND"] = yproc
        actual_lower_boundaries, actual_upper_boundaries = _infer_contains_boundaries(
            ds, nxpe, nype
        )

        assert actual_lower_boundaries == lower_boundaries
        assert actual_upper_boundaries == upper_boundaries

    @pytest.mark.parametrize(
        "xproc, yproc, nxpe, nype, lower_boundaries, upper_boundaries",
        _test_processor_layouts_list,
    )
    def test_infer_boundaries_2d_parallelization_by_filenum(
        self, xproc, yproc, nxpe, nype, lower_boundaries, upper_boundaries
    ):
        """
        Numbering scheme for nxpe=3, nype=4

        y  9 10 11
        ^  6 7  8
        |  3 4  5
        |  0 1  2
         -----> x
        """

        filenum = yproc * nxpe + xproc

        ds = create_test_data(0)
        ds["jyseps2_1"] = 0
        ds["jyseps1_2"] = 0
        ds.encoding["source"] = "folder0/BOUT.dmp." + str(filenum) + ".nc"
        actual_lower_boundaries, actual_upper_boundaries = _infer_contains_boundaries(
            ds, nxpe, nype
        )

        assert actual_lower_boundaries == lower_boundaries
        assert actual_upper_boundaries == upper_boundaries

    @pytest.mark.parametrize(
        "xproc, yproc, nxpe, nype, lower_boundaries, upper_boundaries",
        _test_processor_layouts_doublenull_list,
    )
    def test_infer_boundaries_2d_parallelization_doublenull_by_filenum(
        self, xproc, yproc, nxpe, nype, lower_boundaries, upper_boundaries
    ):
        """
        Numbering scheme for nxpe=3, nype=4

        y  9 10 11
        ^  6 7  8
        |  3 4  5
        |  0 1  2
         -----> x
        """

        filenum = yproc * nxpe + xproc

        ds = create_test_data(0)
        ds["jyseps2_1"] = 3
        ds["jyseps1_2"] = 11
        ds["ny_inner"] = 8
        ds["MYSUB"] = 4
        ds.encoding["source"] = "folder0/BOUT.dmp." + str(filenum) + ".nc"
        actual_lower_boundaries, actual_upper_boundaries = _infer_contains_boundaries(
            ds, nxpe, nype
        )

        assert actual_lower_boundaries == lower_boundaries
        assert actual_upper_boundaries == upper_boundaries

    @pytest.mark.parametrize("is_restart", [False, True])
    def test_keep_xboundaries(self, is_restart):
        ds = create_test_data(0)
        ds = ds.rename({"dim2": "x"})

        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp.0.nc"

        ds["jyseps2_1"] = 8
        ds["jyseps1_2"] = 8

        actual = _trim(
            ds,
            guards={"x": 2},
            keep_boundaries={"x": True},
            nxpe=1,
            nype=1,
            is_restart=is_restart,
        )
        expected = ds  # Should be unchanged
        xrt.assert_equal(expected, actual)

    @pytest.mark.parametrize("is_restart", [False, True])
    def test_keep_yboundaries(self, is_restart):
        ds = create_test_data(0)
        ds = ds.rename({"dim2": "y"})

        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp.0.nc"

        ds["jyseps2_1"] = 8
        ds["jyseps1_2"] = 8

        actual = _trim(
            ds,
            guards={"y": 2},
            keep_boundaries={"y": True},
            nxpe=1,
            nype=1,
            is_restart=is_restart,
        )
        expected = ds  # Should be unchanged
        xrt.assert_equal(expected, actual)

    @pytest.mark.parametrize(
        "filenum, lower, upper",
        [(0, True, False), (1, False, True), (2, True, False), (3, False, True)],
    )
    @pytest.mark.parametrize("is_restart", [False, True])
    def test_keep_yboundaries_doublenull_by_filenum(
        self, filenum, lower, upper, is_restart
    ):
        ds = create_test_data(0)
        ds = ds.rename({"dim2": "y"})

        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp." + str(filenum) + ".nc"

        ds["jyseps2_1"] = 3
        ds["jyseps1_2"] = 11
        ds["ny_inner"] = 8
        ds["MYSUB"] = 4

        actual = _trim(
            ds,
            guards={"y": 2},
            keep_boundaries={"y": True},
            nxpe=1,
            nype=4,
            is_restart=is_restart,
        )
        expected = ds  # Should be unchanged
        if not lower:
            expected = expected.isel(y=slice(2, None, None))
        if not upper:
            expected = expected.isel(y=slice(None, -2, None))
        xrt.assert_equal(expected, actual)

    @pytest.mark.parametrize("is_restart", [False, True])
    def test_trim_timing_info(self, is_restart):
        ds = create_test_data(0)
        from xbout.load import _BOUT_PER_PROC_VARIABLES

        # remove a couple of entries from _BOUT_PER_PROC_VARIABLES so we test that _trim
        # does not fail if not all of them are present
        _BOUT_PER_PROC_VARIABLES = _BOUT_PER_PROC_VARIABLES[:-2]

        for v in _BOUT_PER_PROC_VARIABLES:
            ds[v] = 42.0
        ds = _trim(
            ds, guards={}, keep_boundaries={}, nxpe=1, nype=1, is_restart=is_restart
        )

        expected = create_test_data(0)
        xrt.assert_equal(ds, expected)


fci_shape = (2, 2, 3, 4)
fci_guards = (2, 2, 0)


@pytest.fixture
def create_example_grid_file_fci(tmp_path_factory):
    """
    Mocks up a FCI-like netCDF file, and return the temporary test
    directory containing them.

    Deletes the temporary directory once that test is done.
    """

    # Create grid dataset
    shape = (fci_shape[1] + 2 * fci_guards[0], *fci_shape[2:])
    arr = np.arange(reduce(operator.mul, shape, 1)).reshape(shape)
    grid = DataArray(data=arr, name="R", dims=["x", "y", "z"]).to_dataset()
    grid["Z"] = DataArray(np.random.random(shape), dims=["x", "y", "z"])
    grid["dy"] = DataArray(np.ones(shape), dims=["x", "y", "z"])
    grid = grid.set_coords(["dy"])

    # Create temporary directory
    save_dir = tmp_path_factory.mktemp("griddata")

    # Save
    filepath = save_dir.joinpath("fci.nc")
    grid.to_netcdf(filepath, engine="netcdf4")

    return filepath


@pytest.fixture
def create_example_files_fci(tmp_path_factory):

    return _bout_xyt_example_files(
        tmp_path_factory,
        lengths=fci_shape,
        nxpe=1,
        nype=1,
        # nt=1,
        guards={a: b for a, b in zip("xyz", fci_guards)},
        syn_data_type="random",
        grid=None,
        squashed=False,
        # topology="core",
        write_to_disk=False,
        bout_v5=True,
        metric_3D=True,
    )
