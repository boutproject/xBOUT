import pytest
import numpy.testing as npt

from xbout import open_boutdataset, collect as new_collect

from .test_load import create_bout_ds, create_bout_ds_list, METADATA_VARS

boutdata = pytest.importorskip("boutdata", reason="boutdata is not available")
old_collect = boutdata.collect


class TestAccuracyAgainstOldCollect:
    def test_single_file(self, tmp_path_factory):

        # Create temp directory for files
        test_dir = tmp_path_factory.mktemp("test_data")

        # Generate some test data
        generated_ds = create_bout_ds(syn_data_type="linear")
        generated_ds.to_netcdf(test_dir.joinpath("BOUT.dmp.0.nc"))

        var = "n"
        expected = old_collect(var, path=test_dir, xguards=True, yguards=False)

        # Test against new standard - open_boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(test_dir.joinpath("BOUT.dmp.0.nc"))
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)

        # Test against backwards compatible collect function
        actual = new_collect(var, path=test_dir, xguards=True, yguards=False)

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)

    def test_multiple_files_along_x(self, tmp_path_factory):

        # Create temp directory for files
        test_dir = tmp_path_factory.mktemp("test_data")

        # Generate some test data
        ds_list, file_list = create_bout_ds_list(
            "BOUT.dmp", nxpe=3, nype=1, syn_data_type="linear"
        )
        for temp_ds, file_name in zip(ds_list, file_list):
            temp_ds.to_netcdf(test_dir.joinpath(file_name))

        var = "n"
        expected = old_collect(var, path=test_dir, prefix="BOUT.dmp", xguards=True)

        # Test against new standard - open_boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(test_dir.joinpath("BOUT.dmp.*.nc"))
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)

        # Test against backwards compatible collect function
        actual = new_collect(var, path=test_dir, prefix="BOUT.dmp", xguards=True)

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)

    def test_multiple_files_along_y(self, tmp_path_factory):

        # Create temp directory for files
        test_dir = tmp_path_factory.mktemp("test_data")

        # Generate some test data
        ds_list, file_list = create_bout_ds_list(
            "BOUT.dmp", nxpe=1, nype=3, syn_data_type="linear"
        )
        for temp_ds, file_name in zip(ds_list, file_list):
            temp_ds.to_netcdf(test_dir.joinpath(file_name))

        var = "n"
        expected = old_collect(var, path=test_dir, prefix="BOUT.dmp", xguards=True)

        # Test against new standard - .open_boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(test_dir.joinpath("BOUT.dmp.*.nc"))
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)

        # Test against backwards compatible collect function
        actual = new_collect(var, path=test_dir, prefix="BOUT.dmp", xguards=True)

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)

    def test_multiple_files_along_xy(self, tmp_path_factory):

        # Create temp directory for files
        test_dir = tmp_path_factory.mktemp("test_data")

        # Generate some test data
        ds_list, file_list = create_bout_ds_list(
            "BOUT.dmp", nxpe=3, nype=3, syn_data_type="linear"
        )
        for temp_ds, file_name in zip(ds_list, file_list):
            temp_ds.to_netcdf(test_dir.joinpath(file_name))

        var = "n"
        expected = old_collect(var, path=test_dir, prefix="BOUT.dmp", xguards=True)

        # Test against new standard - .open_boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(test_dir.joinpath("BOUT.dmp.*.nc"))
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)

        # Test against backwards compatible collect function
        actual = new_collect(var, path=test_dir, prefix="BOUT.dmp", xguards=True)

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)

    def test_metadata(self, tmp_path_factory):
        # Create temp directory for files
        test_dir = tmp_path_factory.mktemp("test_data")

        # Generate some test data
        generated_ds = create_bout_ds(syn_data_type="linear")
        generated_ds.to_netcdf(test_dir.joinpath("BOUT.dmp.0.nc"))

        with pytest.warns(UserWarning):
            ds = open_boutdataset(test_dir.joinpath("BOUT.dmp.*.nc"))

        for v in METADATA_VARS:
            expected = old_collect(v, path=test_dir)
            # Check metadata against new standard - open_boutdataset
            actual = ds.bout.metadata[v]
            npt.assert_equal(actual, expected)

            # Check against backwards compatible collect function
            actual = new_collect(v, path=test_dir)
            npt.assert_equal(actual, expected)

    def test_new_collect_indexing_int(self, tmp_path_factory):
        # Create temp directory for files
        test_dir = tmp_path_factory.mktemp("test_data")

        # Generate some test data
        ds_list, file_list = create_bout_ds_list(
            "BOUT.dmp", nxpe=3, nype=3, syn_data_type="linear"
        )

        for temp_ds, file_name in zip(ds_list, file_list):
            temp_ds.to_netcdf(test_dir.joinpath(file_name))

        var = "n"
        indexers = ["tind", "xind", "yind", "zind"]
        ind_arg = 0

        for kwarg in indexers:
            # Extracting a the first index of each dimension for comparison
            expected = old_collect(var, path=test_dir, **{kwarg: ind_arg})

            # Test against backwards compatible collect function
            actual = new_collect(var, path=test_dir, **{kwarg: ind_arg})

            assert expected.shape == actual.shape
            npt.assert_equal(actual, expected)

    def test_new_collect_indexing_list(self, tmp_path_factory):
        # Create temp directory for files
        test_dir = tmp_path_factory.mktemp("test_data")

        # Generate some test data
        ds_list, file_list = create_bout_ds_list(
            "BOUT.dmp", nxpe=3, nype=3, syn_data_type="linear"
        )
        for temp_ds, file_name in zip(ds_list, file_list):
            temp_ds.to_netcdf(test_dir.joinpath(file_name))

        var = "n"
        indexers = ["tind", "xind", "yind", "zind"]
        ind_arg = [0, 4]

        for kwarg in indexers:
            # Extracting a the first index of each dimension for comparison
            expected = old_collect(var, path=test_dir, **{kwarg: ind_arg})

            # Test against backwards compatible collect function
            actual = new_collect(var, path=test_dir, **{kwarg: ind_arg})

            assert expected.shape == actual.shape
            npt.assert_equal(actual, expected)

    def test_new_collect_indexing_slice(self, tmp_path_factory):
        # Create temp directory for files
        test_dir = tmp_path_factory.mktemp("test_data")

        # Generate some test data
        ds_list, file_list = create_bout_ds_list(
            "BOUT.dmp", nxpe=3, nype=3, syn_data_type="linear"
        )

        for temp_ds, file_name in zip(ds_list, file_list):
            temp_ds.to_netcdf(test_dir.joinpath(file_name))

        var = "n"
        indexers = ["tind", "xind", "yind", "zind"]
        ind_list = [slice(0, 4, 2), slice(0, 4)]

        for kwarg in indexers:
            for ind_arg in ind_list:
                # Extracting a the first index of each dimension for comparison
                expected = old_collect(var, path=test_dir, **{kwarg: ind_arg})

                # Test against backwards compatible collect function
                actual = new_collect(var, path=test_dir, **{kwarg: ind_arg})

                assert expected.shape == actual.shape
                npt.assert_equal(actual, expected)


@pytest.mark.skip
class test_speed_against_old_collect:
    ...
