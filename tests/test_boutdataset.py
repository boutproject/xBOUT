import pytest

from xarray import Dataset, DataArray
import xarray.testing as xrt
import numpy as np

from xcollect.boutdataset import BoutDataset


@pytest.fixture(scope='session')
def bout_example_file(tmpdir_factory):
    """
    Create single dataset containing variables like an unparallelised BOUT++ run.

    Saves it as a temporary NetCDF file and returns the file.
    """

    np.random.seed(seed=0)
    T = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])
    n = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])
    ds = Dataset({'n': n, 'T': T})

    filename = "BOUT.dmp.0.nc"
    save_path = tmpdir_factory.mktemp("data").join(filename)
    ds.to_netcdf(str(save_path))
    print(str(save_path))
    return save_path


class TestLoadData:
    def test_load_data(self, bout_example_file):
        bd = BoutDataset(str(bout_example_file))
        print(bd)
        actual = bd.data

        np.random.seed(seed=0)
        T = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])
        n = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])
        expected = Dataset({'n': n, 'T': T})

        xrt.assert_equal(expected, actual)

class TestDatasetMethods:
    pass


class TestLoadInputFile:
    pass


class TestLoadLogFile:
    pass


class TestSave:
    pass


class TestSaveRestart:
    pass
