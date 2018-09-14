import pytest
import os

from xarray import Dataset, DataArray
import xarray.testing as xrt
import numpy as np

from xcollect.collect import collect, _open_all_dump_files


@pytest.fixture(scope='session')
def bout_xyt_example_files(tmpdir_factory, prefix='BOUT.dmp', nxpe=4, nype=2, nt=2):
    """
    Mocks up a set of BOUT-like netCDF files, and return the temporary test directory containing them.
    """

    save_dir = tmpdir_factory.mktemp("data")

    ds_list, file_list = create_bout_ds_list(prefix, nxpe, nype, nt)

    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(str(save_dir.join(str(file_name))))

    return str(save_dir)


def create_bout_ds_list(prefix, nxpe, nype, nt):
    """
    Mocks up a set of BOUT-like datasets.

    Structured as though they were produced by a x-y parallelised run with multiple restarts.
    """

    file_list = []
    ds_list = []
    for i in range(nxpe):
        for j in range(nype):
            num = (i + nxpe * j)
            filename = prefix + "." + str(num) + ".nc"
            file_list.append(filename)
            ds_list.append(create_bout_ds(seed=num))

    # Sort this in order of num to remove any BOUT-specific structure
    ds_list_sorted = [ds for filename, ds in sorted(zip(file_list, ds_list))]
    file_list_sorted = [filename for filename, ds in sorted(zip(file_list, ds_list))]

    return ds_list_sorted, file_list_sorted


def create_bout_ds(seed=0):
    np.random.seed(seed=seed)

    T = DataArray(np.random.randn(2, 4, 6), dims=['t', 'x', 'z'])
    n = DataArray(np.random.randn(2, 4, 6), dims=['t', 'x', 'z'])
    ds = Dataset({'n': n, 'T': T})
    return ds


class TestOpeningFiles:
    def test_open_single_file(self):
        pass

    def test_open_x_parallelized_files(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=1, nt=1)
        actual_filepaths, actual_datasets = _open_all_dump_files(path, 'BOUT.dmp', chunks=None)

        expected_datasets, expected_filepaths = create_bout_ds_list('BOUT.dmp', nxpe=4, nype=1, nt=1)

        for expected, actual in zip(expected_filepaths, actual_filepaths):
            actual_filename = os.path.split(actual)[-1]
            assert expected == actual_filename
        for expected, actual in zip(expected_datasets, actual_datasets):
            xrt.assert_equal(expected, actual)

    def test_open_xy_parallelized_files(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=3, nt=1)
        actual_filepaths, actual_datasets = _open_all_dump_files(path, 'BOUT.dmp', chunks=None)

        expected_datasets, expected_filepaths = create_bout_ds_list('BOUT.dmp', nxpe=4, nype=3, nt=1)

        for expected, actual in zip(expected_filepaths, actual_filepaths):
            actual_filename = os.path.split(actual)[-1]
            assert expected == actual_filename
        for expected, actual in zip(expected_datasets, actual_datasets):
            xrt.assert_equal(expected, actual)

    def test_open_t_parallelized_files(self):
        pass

    def test_open_xyt_parallelized_files(self):
        pass


class TestFileOrganisation:
    pass


class TestTrim:
    pass


class TestCollectData:
    pass


