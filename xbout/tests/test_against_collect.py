import pytest

import numpy.testing as npt

from xbout import open_boutdataset
from .test_load import create_bout_ds_list

from xbout.load import collect as new_collect
from boutdata import collect as old_collect


@pytest.fixture
def create_test_file(tmpdir_factory):
    def _foo(nxpe, nype):
        # Create temp dir for test files
        save_dir = tmpdir_factory.mktemp("test_data")

        # Generate test data
        ds_list, file_list = create_bout_ds_list("data", nxpe=nxpe, nype=nype,
                                             syn_data_type="linear")

        for ds, file_name in zip(ds_list, file_list):
            ds.to_netcdf(str(save_dir.join(str(file_name))))

        return save_dir
    return _foo


class TestAccuracyAgainstOldCollect:
    # @pytest.mark.skip
    def test_single_file(self, create_test_file):

        save_dir = create_test_file(nxpe=1,nype=1)

        var = 'n'
        expected = old_collect(var, path=save_dir, prefix='data', xguards=False)

        actual = new_collect(var, path=save_dir, prefix='data', xguards=False)

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)


   # @pytest.mark.skip
    def test_multiple_files_along_x(self, create_test_file):

        save_dir = create_test_file(nxpe=3, nype=3)

        var = 'n'
        expected = old_collect(var, path=save_dir, prefix='data', xguards=False)

        actual = new_collect(var, path=save_dir, prefix='data', xguards=False)

        print(expected.shape, actual.shape)

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)


    @pytest.mark.skip
    def test_metadata(self):
        ...


@pytest.mark.skip
class test_speed_against_old_collect:
    ...
