import pytest

from xarray import Dataset, DataArray, concat
import xarray.testing as xrt
import numpy as np

from test_collect import bout_xyt_example_files, create_bout_ds
from xcollect.boutdataset import BoutAccessor, load_boutdataset
from xcollect.collect import collect


@pytest.fixture(scope='session')
def bout_example_file(tmpdir_factory):
    """
    Create single dataset containing variables like an unparallelised BOUT++ run.

    Saves it as a temporary NetCDF file and returns the file.
    """

    np.random.seed(seed=0)
    T = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])
    n = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])

    #ds

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

    def test_load_from_single_file(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        actual = BoutDataset(datapath=path).data.compute()
        expected = create_bout_ds().drop(['NXPE', 'NYPE', 'MXG', 'MYG'])
        xrt.assert_equal(actual, expected)


class TestXarrayBehaviour:
    """Set of tests to check that BoutDatasets behave similarly to xarray Datasets."""

    @pytest.mark.xfail
    def test_concat(self, tmpdir_factory):
        path1 = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        bd1 = BoutDataset(datapath=path1)
        path2 = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        bd2 = BoutDataset(datapath=path1)
        print(concat([bd1, bd2], dim='run'))

    @pytest.mark.xfail
    def test_isel(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        bd = BoutDataset(datapath=path)
        actual = bd.isel(x=slice(None,None,2))
        expected = bd.bout.data.isel(x=slice(None,None,2))
        xrt.assert_equal(actual, expected)


class TestBoutDatasetMethods:
    def test_test_method(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        #bd = load_boutdataset(datapath=path)
        ds = collect(path=path)
        #bd = BoutAccessor(ds)
        print(ds)
        ds.bout.test_method()
        print(ds.bout.metadata)
        print(ds.isel(t=-1))
        assert False


class TestLoadInputFile:
    pass


class TestLoadLogFile:
    pass


class TestSave:
    pass


class TestSaveRestart:
    pass
