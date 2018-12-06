import pytest

import numpy.testing as npt

from xbout.load import _auto_open_mfboutdataset

from boutdata import collect


class TestAccuracyAgainstOldCollect:
    @pytest.mark.xfail(reason="Collect tries to read 'nx' when it doesn't exist")
    def test_single_file(self):
        var = 'n'
        expected = collect(var, path='./tests/data/dump_files/single',
                           prefix='equilibrium', xguards=False)

        ds, metadata = _auto_open_mfboutdataset('./tests/data/dump_files/single/equilibrium.nc')
        print(ds)
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)

    @pytest.mark.xfail(reason="Collect says mz=129, then collects an array of "
                              "length 128?!")
    def test_multiple_files_along_x(self):
        var = 'n'
        expected = collect(var, path='./tests/data/dump_files/',
                           prefix='BOUT.dmp', xguards=False)

        ds, metadata = _auto_open_mfboutdataset('./tests/data/dump_files/BOUT.dmp.*.nc')
        actual = ds[var].values

        assert expected.shape == actual.shape
        npt.assert_equal(actual, expected)


    @pytest.mark.skip
    def test_multiple_files_along_x(self):
        ...

    @pytest.mark.skip
    def test_metadata(self):
        ...


@pytest.mark.skip
class test_speed_against_old_collect:
    ...
