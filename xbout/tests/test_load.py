from pathlib import Path

import pytest

from xarray import concat
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
)
from xbout.utils import _separate_metadata
from xbout.tests.utils_for_tests import create_bout_ds, METADATA_VARS

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
            _expand_filepaths(path)


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


class TestStripMetadata:
    def test_strip_metadata(self):
        original = create_bout_ds()
        assert original["NXPE"] == 1

        ds, metadata = _separate_metadata(original)

        xrt.assert_equal(
            original.drop_vars(
                METADATA_VARS + _BOUT_PER_PROC_VARIABLES, errors="ignore"
            ),
            ds,
        )
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
                METADATA_VARS + _BOUT_PER_PROC_VARIABLES, errors="ignore"
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
                METADATA_VARS + _BOUT_PER_PROC_VARIABLES, errors="ignore"
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
    def test_combine_along_t(self): ...

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
            None, nxpe=3, nype=3, nt=1, syn_data_type="stepped", grid="grid"
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
    def test_combine_along_tx(self): ...

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

        .. code:: text

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

        .. code:: text

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

        .. code:: text

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

        .. code:: text

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
