import pytest

from xarray import Dataset, DataArray, concat
import xarray.testing as xrt
import numpy as np

from boutdata.data import BoutOptionsFile, BoutOptions

from xcollect.tests.test_collect import bout_xyt_example_files, create_bout_ds
from xcollect.boutdataset import BoutAccessor, open_boutdataset
from xcollect.collect import collect


EXAMPLE_OPTIONS_FILE_PATH = './tests/data/options/BOUT.inp'


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

    ds['NXPE'], ds['NYPE'] = 1, 1
    ds['MXG'], ds['MYG'] = 2, 0

    prefix = 'BOUT.dmp'
    filename = prefix + ".0.nc"
    save_dir = tmpdir_factory.mktemp("data")
    save_path = save_dir.join(filename)
    ds.to_netcdf(str(save_path))

    return save_dir, prefix


class TestLoadData:
    def test_load_data(self, bout_example_file):
        save_path, prefix = bout_example_file

        bd = open_boutdataset(datapath=str(save_path), prefix=prefix, inputfilepath=None)
        print(bd)
        actual = bd

        np.random.seed(seed=0)
        T = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])
        n = DataArray(np.random.randn(5, 10, 20), dims=['t', 'x', 'z'])
        expected = Dataset({'n': n, 'T': T})

        xrt.assert_equal(expected, actual)

    def test_load_from_single_file(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        actual = open_boutdataset(datapath=path, inputfilepath=None).compute()
        expected = create_bout_ds().drop(['NXPE', 'NYPE', 'MXG', 'MYG'])
        xrt.assert_equal(actual, expected)


class TestXarrayBehaviour:
    """
    Set of tests to check that BoutDatasets behave similarly to xarray Datasets.
    (With the accessor approach these should pass trivially now.)
    """

    def test_concat(self, tmpdir_factory):
        path1 = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        bd1 = open_boutdataset(datapath=path1, inputfilepath=None)
        path2 = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        bd2 = open_boutdataset(datapath=path2, inputfilepath=None)
        result = concat([bd1, bd2], dim='run')
        assert result.dims == {**bd1.dims, 'run': 2}

    def test_isel(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        bd = open_boutdataset(datapath=path, inputfilepath=None)
        actual = bd.isel(x=slice(None,None,2))
        expected = bd.bout.data.isel(x=slice(None,None,2))
        xrt.assert_equal(actual, expected)


class TestBoutDatasetMethods:
    def test_test_method(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None)
        #ds = collect(path=path)
        #bd = BoutAccessor(ds)
        print(ds)
        #ds.bout.test_method()
        #print(ds.bout.options)
        #print(ds.bout.metadata)
        print(ds.isel(t=-1))

        #ds.bout.set_extra_data('stored')
        ds.bout.extra_data = 'stored'

        print(ds.bout.extra_data)


class TestLoadInputFile:
    def test_load_options(self):
        options = BoutOptionsFile(EXAMPLE_OPTIONS_FILE_PATH)
        assert isinstance(options, BoutOptions)
        # TODO Check it contains the same text

    def test_load_options_in_dataset(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=EXAMPLE_OPTIONS_FILE_PATH)
        assert isinstance(ds.options, BoutOptions)


class TestLoadLogFile:
    pass


class TestSave:
    pass


class TestSaveRestart:
    pass
