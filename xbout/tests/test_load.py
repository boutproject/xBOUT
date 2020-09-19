from pathlib import Path
import re

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


def test_check_extensions(tmpdir):
    files_dir = tmpdir.mkdir("data")
    example_nc_file = files_dir.join("example.nc")
    example_nc_file.write("content_nc")

    filetype = _check_filetype(Path(str(example_nc_file)))
    assert filetype == "netcdf4"

    example_hdf5_file = files_dir.join("example.h5netcdf")
    example_hdf5_file.write("content_hdf5")

    filetype = _check_filetype(Path(str(example_hdf5_file)))
    assert filetype == "h5netcdf"

    example_invalid_file = files_dir.join("example.txt")
    example_hdf5_file.write("content_txt")
    with pytest.raises(IOError):
        filetype = _check_filetype(Path(str(example_invalid_file)))


class TestPathHandling:
    def test_glob_expansion_single(self, tmpdir):
        files_dir = tmpdir.mkdir("data")
        example_file = files_dir.join("example.0.nc")
        example_file.write("content")

        path = Path(str(example_file))
        filepaths = _expand_wildcards(path)
        assert filepaths[0] == Path(str(example_file))

        path = Path(str(files_dir.join("example.*.nc")))
        filepaths = _expand_wildcards(path)
        assert filepaths[0] == Path(str(example_file))

    @pytest.mark.parametrize(
        "ii, jj", [(1, 1), (1, 4), (3, 1), (5, 3), (12, 1), (1, 12), (121, 2), (3, 111)]
    )
    def test_glob_expansion_both(self, tmpdir, ii, jj):
        files_dir = tmpdir.mkdir("data")
        filepaths = []
        for i in range(ii):
            example_run_dir = files_dir.mkdir("run" + str(i))
            for j in range(jj):
                example_file = example_run_dir.join("example." + str(j) + ".nc")
                example_file.write("content")
                filepaths.append(Path(str(example_file)))
        expected_filepaths = natsorted(filepaths, key=lambda filepath: str(filepath))

        path = Path(str(files_dir.join("run*/example.*.nc")))
        actual_filepaths = _expand_wildcards(path)

        assert actual_filepaths == expected_filepaths

    @pytest.mark.parametrize(
        "ii, jj", [(1, 1), (1, 4), (3, 1), (5, 3), (1, 12), (3, 111)]
    )
    def test_glob_expansion_brackets(self, tmpdir, ii, jj):
        files_dir = tmpdir.mkdir("data")
        filepaths = []
        for i in range(ii):
            example_run_dir = files_dir.mkdir("run" + str(i))
            for j in range(jj):
                example_file = example_run_dir.join("example." + str(j) + ".nc")
                example_file.write("content")
                filepaths.append(Path(str(example_file)))
        expected_filepaths = natsorted(filepaths, key=lambda filepath: str(filepath))

        path = Path(str(files_dir.join("run[1-9]/example.*.nc")))
        actual_filepaths = _expand_wildcards(path)

        assert actual_filepaths == expected_filepaths[jj:]

    def test_no_files(self, tmpdir):
        files_dir = tmpdir.mkdir("data")

        with pytest.raises(IOError):
            path = Path(str(files_dir.join("run*/example.*.nc")))
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


@pytest.fixture()
def bout_xyt_example_files(tmpdir_factory):
    return _bout_xyt_example_files


def _bout_xyt_example_files(
    tmpdir_factory,
    prefix="BOUT.dmp",
    lengths=(6, 2, 4, 7),
    nxpe=4,
    nype=2,
    nt=1,
    guards={},
    syn_data_type="random",
    grid=None,
    squashed=False,
    topology="core",
    write_to_disk=False,
):
    """
    Mocks up a set of BOUT-like Datasets

    Either returns list of Datasets (if write_to_disk=False)
    or writes Datasets to netCDF files and returns the temporary test directory
    containing them, deleting the temporary directory once that test is done (if
    write_to_disk=True).
    """

    if squashed:
        # create a single data-file, but alter the 'nxpe' and 'nype' variables, as if the
        # file had been created by combining a set of BOUT.dmp.*.nc files
        ds_list, file_list = create_bout_ds_list(
            prefix=prefix,
            lengths=lengths,
            nxpe=1,
            nype=1,
            nt=nt,
            guards=guards,
            topology=topology,
            syn_data_type=syn_data_type,
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
            return ds_list
        else:
            return ds_list, grid_ds
    elif tmpdir_factory is None:
        raise ValueError("tmpdir_factory required when write_to_disk=True")

    save_dir = tmpdir_factory.mktemp("data")

    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(str(save_dir.join(str(file_name))))

    if grid is not None:
        grid_ds.to_netcdf(str(save_dir.join(grid + ".nc")))

    # Return a glob-like path to all files created, which has all file numbers replaced
    # with a single asterix
    path = str(save_dir.join(str(file_list[-1])))

    count = 1
    if nt > 1:
        count += 1
    # We have to reverse the path before limiting the number of numbers replaced so that the
    # tests don't get confused by pytest's persistent temporary directories (which are also designated
    # by different numbers)
    glob_pattern = (re.sub(r"\d+", "*", path[::-1], count=count))[::-1]
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
            )
            ds_list.append(ds)

    return ds_list, file_list


def create_bout_ds(
    syn_data_type="random",
    lengths=(6, 2, 4, 7),
    num=0,
    nxpe=1,
    nype=1,
    xproc=0,
    yproc=0,
    guards={},
    topology="core",
):

    # Set the shape of the data in this dataset
    t_length, x_length, y_length, z_length = lengths
    mxg = guards.get("x", 0)
    myg = guards.get("y", 0)
    x_length += 2 * mxg
    y_length += 2 * myg
    shape = (t_length, x_length, y_length, z_length)

    # calculate global nx, ny and nz
    nx = nxpe * lengths[1] + 2 * mxg
    ny = nype * lengths[2]
    nz = 1 * lengths[3]

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

        data[:, mxg : x_length - mxg, myg : y_length - myg, :] = (
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
    for v in [n, T]:
        v.attrs["direction_y"] = "Standard"
        v.attrs["cell_location"] = "CELL_CENTRE"
    ds = Dataset({"n": n, "T": T})

    # BOUT_VERSION needed so that we know that number of points in z is MZ, not MZ-1 (as
    # it was in BOUT++ before v4.0
    ds["BOUT_VERSION"] = 4.3

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
        ds["jyseps2_2"] = ny
        ds["ny_inner"] = ny // 2
    elif topology == "sol":
        ds["ixseps1"] = 0
        ds["ixseps2"] = 0
        ds["jyseps1_1"] = -1
        ds["jyseps2_1"] = ny // 2 - 1
        ds["jyseps1_2"] = ny // 2 - 1
        ds["jyseps2_2"] = ny
        ds["ny_inner"] = ny // 2
    elif topology == "limiter":
        ds["ixseps1"] = nx // 2
        ds["ixseps2"] = nx
        ds["jyseps1_1"] = -1
        ds["jyseps2_1"] = ny // 2 - 1
        ds["jyseps1_2"] = ny // 2 - 1
        ds["jyseps2_2"] = ny
        ds["ny_inner"] = ny // 2
    elif topology == "xpoint":
        if nype < 4:
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
        if nype < 3:
            raise ValueError(f"Not enough processors for xpoint topology: nype={nype}")
        ds["ixseps1"] = nx // 2
        ds["ixseps2"] = nx
        ds["jyseps1_1"] = MYSUB - 1
        ds["jyseps2_1"] = ny // 2 - 1
        ds["jyseps1_2"] = ny // 2 - 1
        ds["jyseps2_2"] = ny - MYSUB - 1
        ds["ny_inner"] = ny // 2
    elif topology == "connected-double-null":
        if nype < 6:
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
    elif topology == "disconnected-double-null":
        if nype < 6:
            raise ValueError(
                "Not enough processors for disconnected-double-null "
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
    else:
        raise ValueError(f"Unrecognised topology={topology}")

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
    ds["dz"] = 2.0 * np.pi / nz

    ds["iteration"] = t_length
    ds["t_array"] = DataArray(np.arange(t_length, dtype=float) * 10.0, dims="t")

    # xarray adds this encoding when opening a file. Emulate here as it may be used to
    # get the file number
    ds.encoding["source"] = f"BOUT.dmp.{num}.nc"

    return ds


def create_bout_grid_ds(xsize=2, ysize=4, guards={}, topology="core", ny_inner=0):

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

    return ds


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
    "ixseps1",
    "ixseps2",
    "jyseps1_1",
    "jyseps1_2",
    "jyseps2_1",
    "jyseps2_2",
    "ny_inner",
    "zperiod",
    "ZMIN",
    "ZMAX",
    "dz",
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
    def test_single_file(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmpdir_factory, nxpe=1, nype=1, nt=1, write_to_disk=True
        )
        with pytest.warns(UserWarning):
            actual = open_boutdataset(datapath=path, keep_xboundaries=False)
        expected = create_bout_ds()
        expected = expected.set_coords("t_array").rename(t_array="t")
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

    def test_squashed_file(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmpdir_factory, nxpe=4, nype=3, nt=1, squashed=True, write_to_disk=True
        )
        with pytest.warns(UserWarning):
            actual = open_boutdataset(datapath=path, keep_xboundaries=False)
        expected = create_bout_ds()
        expected = expected.set_coords("t_array").rename(t_array="t")
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

    def test_combine_along_x(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmpdir_factory,
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
        expected = expected.set_coords("t_array").rename(t_array="t")
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

    def test_combine_along_y(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmpdir_factory,
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
        expected = expected.set_coords("t_array").rename(t_array="t")
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

    def test_combine_along_xy(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmpdir_factory,
            nxpe=4,
            nype=3,
            nt=1,
            syn_data_type="stepped",
            write_to_disk=True,
        )
        with pytest.warns(UserWarning):
            actual = open_boutdataset(datapath=path, keep_xboundaries=False)

        bout_ds = create_bout_ds
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
        expected = expected.set_coords("t_array").rename(t_array="t")
        xrt.assert_equal(
            actual.drop_vars(["x", "y", "z"]).load(),
            expected.drop_vars(
                METADATA_VARS + _BOUT_PER_PROC_VARIABLES, errors="ignore"
            ),
        )

        # check creation without writing to disk gives identical result
        fake_ds_list = bout_xyt_example_files(
            None, nxpe=4, nype=3, nt=1, syn_data_type="stepped"
        )
        with pytest.warns(UserWarning):
            fake = open_boutdataset(datapath=fake_ds_list, keep_xboundaries=False)
        xrt.assert_identical(actual, fake)

    def test_toroidal(self, tmpdir_factory, bout_xyt_example_files):
        # actually write these to disk to test the loading fully
        path = bout_xyt_example_files(
            tmpdir_factory,
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
            gridfilepath=Path(path).parent.joinpath("grid.nc"),
        )

        # check dataset can be saved
        save_dir = tmpdir_factory.mktemp("data")
        actual.bout.save(str(save_dir.join("boutdata.nc")))

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

    def test_salpha(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmpdir_factory,
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
            gridfilepath=Path(path).parent.joinpath("grid.nc"),
        )

        # check dataset can be saved
        save_dir = tmpdir_factory.mktemp("data")
        actual.bout.save(str(save_dir.join("boutdata.nc")))

        # check creation without writing to disk gives identical result
        fake_ds_list, fake_grid_ds = bout_xyt_example_files(
            None, nxpe=3, nype=3, nt=1, syn_data_type="stepped", grid="grid"
        )
        fake = open_boutdataset(
            datapath=fake_ds_list, geometry="s-alpha", gridfilepath=fake_grid_ds
        )
        xrt.assert_identical(actual, fake)

    def test_drop_vars(self, tmpdir_factory, bout_xyt_example_files):
        datapath = bout_xyt_example_files(
            tmpdir_factory,
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
    def test_no_trim(self):
        ds = create_test_data(0)
        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp.0.nc"
        actual = _trim(ds, guards={}, keep_boundaries={}, nxpe=1, nype=1)
        xrt.assert_equal(actual, ds)

    def test_trim_guards(self):
        ds = create_test_data(0)
        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp.0.nc"
        actual = _trim(ds, guards={"time": 2}, keep_boundaries={}, nxpe=1, nype=1)
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

    def test_keep_xboundaries(self):
        ds = create_test_data(0)
        ds = ds.rename({"dim2": "x"})

        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp.0.nc"

        ds["jyseps2_1"] = 8
        ds["jyseps1_2"] = 8

        actual = _trim(ds, guards={"x": 2}, keep_boundaries={"x": True}, nxpe=1, nype=1)
        expected = ds  # Should be unchanged
        xrt.assert_equal(expected, actual)

    def test_keep_yboundaries(self):
        ds = create_test_data(0)
        ds = ds.rename({"dim2": "y"})

        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp.0.nc"

        ds["jyseps2_1"] = 8
        ds["jyseps1_2"] = 8

        actual = _trim(ds, guards={"y": 2}, keep_boundaries={"y": True}, nxpe=1, nype=1)
        expected = ds  # Should be unchanged
        xrt.assert_equal(expected, actual)

    @pytest.mark.parametrize(
        "filenum, lower, upper",
        [(0, True, False), (1, False, True), (2, True, False), (3, False, True)],
    )
    def test_keep_yboundaries_doublenull_by_filenum(self, filenum, lower, upper):
        ds = create_test_data(0)
        ds = ds.rename({"dim2": "y"})

        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding["source"] = "folder0/BOUT.dmp." + str(filenum) + ".nc"

        ds["jyseps2_1"] = 3
        ds["jyseps1_2"] = 11
        ds["ny_inner"] = 8
        ds["MYSUB"] = 4

        actual = _trim(ds, guards={"y": 2}, keep_boundaries={"y": True}, nxpe=1, nype=4)
        expected = ds  # Should be unchanged
        if not lower:
            expected = expected.isel(y=slice(2, None, None))
        if not upper:
            expected = expected.isel(y=slice(None, -2, None))
        xrt.assert_equal(expected, actual)

    def test_trim_timing_info(self):
        ds = create_test_data(0)
        from xbout.load import _BOUT_PER_PROC_VARIABLES

        # remove a couple of entries from _BOUT_PER_PROC_VARIABLES so we test that _trim
        # does not fail if not all of them are present
        _BOUT_PER_PROC_VARIABLES = _BOUT_PER_PROC_VARIABLES[:-2]

        for v in _BOUT_PER_PROC_VARIABLES:
            ds[v] = 42.0
        ds = _trim(ds, guards={}, keep_boundaries={}, nxpe=1, nype=1)

        expected = create_test_data(0)
        xrt.assert_equal(ds, expected)
