import pytest
import os

from xarray import Dataset, DataArray
import xarray.testing as xrt
import numpy as np

from xcollect.collect import collect, _open_all_dump_files, _organise_files


@pytest.fixture(scope='session')
def bout_xyt_example_files(tmpdir_factory, prefix='BOUT.dmp', nxpe=4, nype=2, nt=2, syn_data_type='random'):
    """
    Mocks up a set of BOUT-like netCDF files, and return the temporary test directory containing them.
    """

    save_dir = tmpdir_factory.mktemp("data")

    ds_list, file_list = create_bout_ds_list(prefix, nxpe, nype, nt, syn_data_type=syn_data_type)

    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(str(save_dir.join(str(file_name))))

    return str(save_dir)


def create_bout_ds_list(prefix, nxpe, nype, nt=1, syn_data_type='random'):
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
            ds_list.append(create_bout_ds(syn_data_type, num))

    # Sort this in order of num to remove any BOUT-specific structure
    ds_list_sorted = [ds for filename, ds in sorted(zip(file_list, ds_list))]
    file_list_sorted = [filename for filename, ds in sorted(zip(file_list, ds_list))]

    return ds_list_sorted, file_list_sorted


def create_bout_ds(syn_data_type='random', num=0):
    shape = (2, 4, 6)

    if syn_data_type is 'random':
        # Each dataset contains the same random noise
        np.random.seed(seed=0)
        data = np.random.randn(*shape)
    elif syn_data_type is 'linear':
        # Variables increase linearly across entire domain
        #data =
        pass
    elif syn_data_type is 'stepped':
        # Each dataset contains a different number depending on the filename
        data = np.ones(shape) * num
    else:
        raise ValueError('Not a recognised choice of type of synthetic bout data.')

    T = DataArray(data, dims=['t', 'x', 'z'])
    n = DataArray(data, dims=['t', 'x', 'z'])
    ds = Dataset({'n': n, 'T': T})
    return ds


class TestOpeningFiles:
    def test_open_single_file(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        actual_filepath, actual_dataset = _open_all_dump_files(path, 'BOUT.dmp', chunks=None)

        expected_dataset, expected_filename = create_bout_ds_list('BOUT.dmp', nxpe=1, nype=1, nt=1)

        actual_filename = os.path.split(actual_filepath[0])[-1]
        assert expected_filename[0] == actual_filename
        xrt.assert_equal(expected_dataset[0], actual_dataset[0])

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
    def test_organise_x_parallelized_files(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=1, nt=1, syn_data_type='linear')
        prefix = 'BOUT.dmp'
        filepaths, datasets = _open_all_dump_files(path, prefix=prefix, chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, prefix=prefix, nxpe=4, nype=1)

        print(ds_grid)
        assert ds_grid.shape == (4,)
        # check datasets are in the right order somehow
        assert False

    def test_organise_y_parallelized_files(self):
        pass

    def test_organise_xy_parallelized_files(self):
        pass

    def test_organise_t_parallelized_files(self):
        pass

    def test_organise_xyt_parallelized_files(self):
        pass


class TestTrim:
    def test_no_trim(self):
        pass

    def test_trim_ghosts(self):
        pass

    def trim_guards(self):
        pass


class TestCollectData:
    @pytest.mark.xfail(reason='NotYetImplemented')
    def test_collect_from_single_file(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        actual = collect(vars='all', path=path)
        expected = create_bout_ds()
        xrt.assert_equal(actual, expected)

    @pytest.mark.xfail(reason='NotYetImplemented')
    def test_collect_single_variables(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        actual = collect(vars='n', path=path)
        expected = create_bout_ds()
        xrt.assert_equal(actual, expected['n'])

    def test_collect_multiple_files(self, tmpdir_factory):
        pass

