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

    prefix = 'BOUT.dmp'
    filename = prefix + ".0.nc"
    save_dir = tmpdir_factory.mktemp("data")
    save_path = save_dir.join(filename)
    ds.to_netcdf(str(save_path))

    return save_dir, prefix


class TestLoadData:
    @pytest.mark.xfail
    def test_load_data(self, bout_example_file):
        save_path, prefix = bout_example_file

        bd = BoutDataset(datapath=str(save_path), prefix=prefix)
        print(bd)
        actual = bd.data

        np.random.seed(seed=0)
        T = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])
        n = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])
        expected = Dataset({'n': n, 'T': T})

        xrt.assert_equal(expected, actual)


class TestXarrayBehaviour:
    """Set of tests to check that BoutDatasets behave similarly to xarray Datasets."""
    def test_concat(self):
        pass


class TestBoutDatasetMethods:
    pass


class TestLoadInputFile:
    pass


class TestLoadLogFile:
    pass


class TestSave:
    pass


class TestSaveRestart:
    pass
