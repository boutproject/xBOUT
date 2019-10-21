import pytest

import numpy.testing as npt

from xbout.load import open_boutdataset
from .test_load import create_bout_ds, create_bout_ds_list, METADATA_VARS

from boutdata import collect

class TestAccuracyAgainstOldCollect:
    def test_single_file(self, tmpdir_factory):

        # Create temp directory for files
        test_dir = tmpdir_factory.mktemp("test_data")

        # Generate some test data
        generated_ds = create_bout_ds(syn_data_type="linear")
        generated_ds.to_netcdf(str(test_dir.join("BOUT.dmp.0.nc")))

        var = 'n'
        expected = collect(var, path=test_dir, xguards=True, yguards=False)

        ds = open_boutdataset(test_dir.join("BOUT.dmp.0.nc"))
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)


    def test_multiple_files_along_x(self, tmpdir_factory):

        # Create temp directory for files
        test_dir = tmpdir_factory.mktemp("test_data")

        # Generate some test data
        ds_list, file_list = create_bout_ds_list("BOUT.dmp", nxpe=3, nype=1,
                                                 syn_data_type="linear")
        for temp_ds, file_name in zip(ds_list, file_list):
            temp_ds.to_netcdf(str(test_dir.join(str(file_name))))

        var = 'n'
        expected = collect(var, path=test_dir,
                           prefix='BOUT.dmp', xguards=True)

        ds = open_boutdataset(test_dir.join('BOUT.dmp.*.nc'))
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)


    def test_multiple_files_along_y(self, tmpdir_factory):

        # Create temp directory for files
        test_dir = tmpdir_factory.mktemp("test_data")

        # Generate some test data
        ds_list, file_list = create_bout_ds_list("BOUT.dmp", nxpe=1, nype=3,
                                                 syn_data_type="linear")
        for temp_ds, file_name in zip(ds_list, file_list):
            temp_ds.to_netcdf(str(test_dir.join(str(file_name))))

        var = 'n'
        expected = collect(var, path=test_dir,
                           prefix='BOUT.dmp', xguards=True)

        ds = open_boutdataset(test_dir.join('BOUT.dmp.*.nc'))
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)


    def test_multiple_files_along_xy(self, tmpdir_factory):

        # Create temp directory for files
        test_dir = tmpdir_factory.mktemp("test_data")

        # Generate some test data
        ds_list, file_list = create_bout_ds_list("BOUT.dmp", nxpe=3, nype=3,
                                                 syn_data_type="linear")
        for temp_ds, file_name in zip(ds_list, file_list):
            temp_ds.to_netcdf(str(test_dir.join(str(file_name))))

        var = 'n'
        expected = collect(var, path=test_dir,
                           prefix='BOUT.dmp', xguards=True)

        ds = open_boutdataset(test_dir.join('BOUT.dmp.*.nc'))
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)


    def test_metadata(self, tmpdir_factory):
        # Create temp directory for files
        test_dir = tmpdir_factory.mktemp("test_data")

        # Generate some test data
        generated_ds = create_bout_ds(syn_data_type="linear")
        generated_ds.to_netcdf(str(test_dir.join("BOUT.dmp.0.nc")))

        ds = open_boutdataset(test_dir.join('BOUT.dmp.*.nc'))

        for v in METADATA_VARS:
            expected = collect(v, path=test_dir)
            actual = ds.bout.metadata[v]
            npt.assert_equal(actual, expected)


@pytest.mark.skip
class test_speed_against_old_collect:
    ...
