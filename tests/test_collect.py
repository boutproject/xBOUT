import pytest
import warnings
from pathlib import Path
import re

from xarray import Dataset, DataArray
import xarray.testing as xrt
import numpy as np
import numpy.testing as npt

from xcollect.collect import collect, _check_filetype, _expand_wildcards, _open_all_dump_files, _organise_files, _trim


@pytest.fixture(scope='session')
def bout_xyt_example_files(tmpdir_factory, prefix='BOUT.dmp', lengths=(2,4,1,6),
                           nxpe=4, nype=2, nt=1, ghosts={}, guards={}, syn_data_type='random'):
    """
    Mocks up a set of BOUT-like netCDF files, and return the temporary test directory containing them.

    Deletes the temporary directory once that test is done.
    """

    save_dir = tmpdir_factory.mktemp("data")

    ds_list, file_list = create_bout_ds_list(prefix=prefix, lengths=lengths, nxpe=nxpe, nype=nype, nt=nt,
                                             ghosts=ghosts, guards=guards, syn_data_type=syn_data_type)

    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(str(save_dir.join(str(file_name))))

    # Return a glob-like path to all files created, which has all file numbers replaced with a single asterix
    # We have to limit the number of numbers replaced and reverse the path so that the tests don't get confused by
    # persistent tempoarary directories (which are also designated by different numbers)
    path = str(save_dir.join(str(file_list[-1])))
    count = 1
    if nt > 1:
        count += 1
    glob_pattern = (re.sub('\d+', '*', path[::-1], count=count))[::-1]
    return glob_pattern

    #try:
    #yield glob_pattern
    # finally:
    #     # Clean up temporary directories
    #     print('Cleaning up after myself ')
    #     shutil.rmtree(save_dir)


def create_bout_ds_list(prefix, lengths=(2,4,1,6), nxpe=4, nype=2, nt=1, ghosts={}, guards={}, syn_data_type='random'):
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

            # Include ghost cells
            upper_bndry_cells = {dim: ghosts.get(dim) for dim in ghosts.keys()}
            lower_bndry_cells = {dim: ghosts.get(dim) for dim in ghosts.keys()}

            # Include guard cells
            for dim in ['x', 'y']:
                if dim in guards.keys():
                    if i == 0:
                        lower_bndry_cells[dim] = guards[dim]
                    if i == nxpe-1:
                        upper_bndry_cells[dim] = guards[dim]

            ds = create_bout_ds(syn_data_type=syn_data_type, num=num, lengths=lengths, nxpe=nxpe, nype=nype,
                                upper_bndry_cells=upper_bndry_cells, lower_bndry_cells=lower_bndry_cells, guards=guards)
            ds_list.append(ds)

    # Sort this in order of num to remove any BOUT-specific structure
    ds_list_sorted = [ds for filename, ds in sorted(zip(file_list, ds_list))]
    file_list_sorted = [filename for filename, ds in sorted(zip(file_list, ds_list))]

    return ds_list_sorted, file_list_sorted


def assert_dataset_grids_equal(ds_grid1, ds_grid2):
    assert ds_grid1.shape == ds_grid2.shape

    for index, ds_dict1 in np.ndenumerate(ds_grid1):
        ds1 = ds_dict1['key']
        ds2 = ds_grid2[index]['key']
        try:
            xrt.assert_equal(ds1, ds2)
        except AssertionError as error:
            print('Datasets in position ' + str(index) + ' are not equal.\n' + str(error))


def create_bout_ds(syn_data_type='random', lengths=(2,4,1,6), num=0, nxpe=1, nype=1,
                   upper_bndry_cells={}, lower_bndry_cells={}, guards={}):

    # Set the shape of the data in this dataset
    x_length, y_length, z_length, t_length = lengths
    x_length += upper_bndry_cells.get('x', 0) + lower_bndry_cells.get('x', 0)
    y_length += upper_bndry_cells.get('y', 0) + lower_bndry_cells.get('y', 0)
    z_length += upper_bndry_cells.get('z', 0) + lower_bndry_cells.get('z', 0)
    t_length += upper_bndry_cells.get('t', 0) + lower_bndry_cells.get('t', 0)
    shape = (x_length, y_length, z_length, t_length)

    # Fill with some kind of synthetic data
    if syn_data_type is 'random':
        # Each dataset contains the same random noise
        np.random.seed(seed=0)
        data = np.random.randn(*shape)
    elif syn_data_type is 'linear':
        # Variables increase linearly across entire domain
        raise NotImplementedError
    elif syn_data_type is 'stepped':
        # Each dataset contains a different number depending on the filename
        data = np.ones(shape) * num
    else:
        raise ValueError('Not a recognised choice of type of synthetic bout data.')

    T = DataArray(data, dims=['x', 'y', 'z', 't'])
    n = DataArray(data, dims=['x', 'y', 'z', 't'])
    ds = Dataset({'n': n, 'T': T}).squeeze()

    # Include the metadata about parallelization which collect requires
    ds['NXPE'] = nxpe
    ds['NYPE'] = nype
    ds['MXG'] = guards.get('x', 0)
    ds['MYG'] = guards.get('y', 0)

    return ds


def test_check_extensions(tmpdir):
    files_dir = tmpdir.mkdir("data")
    example_nc_file = files_dir.join('example.nc')
    example_nc_file.write("content_nc")

    filetype = _check_filetype(Path(str(example_nc_file)))
    assert filetype == 'netcdf4'

    example_hdf5_file = files_dir.join('example.h5netcdf')
    example_hdf5_file.write("content_hdf5")

    filetype = _check_filetype(Path(str(example_hdf5_file)))
    assert filetype == 'h5netcdf'

    example_invalid_file = files_dir.join('example.txt')
    example_hdf5_file.write("content_txt")

    with pytest.raises(IOError):
        filetype = _check_filetype(Path(str(example_invalid_file)))


class TestPathHandling:
    def test_glob_expansion_single(self, tmpdir):
        files_dir = tmpdir.mkdir("data")
        example_file = files_dir.join('example.0.nc')
        example_file.write("content")

        path = Path(str(example_file))
        filepaths = _expand_wildcards(path)
        assert filepaths[0] == Path(str(example_file))

        path = Path(str(files_dir.join('example.*.nc')))
        filepaths = _expand_wildcards(path)
        assert filepaths[0] == Path(str(example_file))

    @pytest.mark.parametrize("ii, jj", [(1, 1), (1, 4), (3, 1), (5, 3), (12, 1),
                                        (1, 12), (121, 2), (3, 111)])
    def test_glob_expansion_both(self, tmpdir, ii, jj):
        files_dir = tmpdir.mkdir("data")
        filepaths = []
        for i in range(ii):
            example_run_dir = files_dir.mkdir('run' + str(i))
            for j in range(jj):
                example_file = example_run_dir.join('example.' + str(j) + '.nc')
                example_file.write("content")
                filepaths.append(Path(str(example_file)))
        expected_filepaths = sorted(filepaths, key=lambda filepath: str(filepath))

        path = Path(str(files_dir.join('run*/example.*.nc')))
        actual_filepaths = _expand_wildcards(path)

        assert actual_filepaths == expected_filepaths


class TestOpeningFiles:
    def test_open_single_file(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        actual_filepath, actual_dataset = _open_all_dump_files(path=Path(path), chunks=None)

        expected_dataset, expected_filename = create_bout_ds_list('BOUT.dmp', nxpe=1, nype=1, nt=1)

        actual_filename = actual_filepath[0].parts[-1]
        assert expected_filename[0] == actual_filename
        xrt.assert_equal(expected_dataset[0], actual_dataset[0])

    def test_open_x_parallelized_files(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=1, nt=1)
        actual_filepaths, actual_datasets = _open_all_dump_files(path=Path(path), chunks=None)

        expected_datasets, expected_filepaths = create_bout_ds_list('BOUT.dmp', nxpe=4, nype=1, nt=1)

        assert len(expected_filepaths) == len(actual_filepaths)
        for expected, actual in zip(expected_filepaths, actual_filepaths):
            actual_filename = actual.parts[-1]
            assert expected == actual_filename
        for expected, actual in zip(expected_datasets, actual_datasets):
            xrt.assert_equal(expected, actual)

    def test_open_xy_parallelized_files(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=3, nt=1)
        actual_filepaths, actual_datasets = _open_all_dump_files(path=Path(path), chunks=None)

        expected_datasets, expected_filepaths = create_bout_ds_list('BOUT.dmp', nxpe=4, nype=3, nt=1)

        for expected, actual in zip(expected_filepaths, actual_filepaths):
            actual_filename = actual.parts[-1]
            assert expected == actual_filename
        for expected, actual in zip(expected_datasets, actual_datasets):
            xrt.assert_equal(expected, actual)

    def test_open_t_parallelized_files(self):
        pass

    def test_open_xyt_parallelized_files(self):
        pass

    def test_warn_on_opening_many_files(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=12, nype=14, nt=1)
        with pytest.warns(UserWarning):
            filepaths, datasets = _open_all_dump_files(path=Path(path), chunks=None)


class TestFileOrganisation:
    def test_organise_x_parallelized_files(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=1, nt=1, syn_data_type='stepped')
        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=4, nype=1)
        assert ds_grid.shape == (4,)
        assert concat_dims == ['x']

        # check datasets are in the right order
        contents = [np.unique(ds_dict['key']['n'].values) for ds_dict in ds_grid]
        expected = [np.array([i]) for i in range(4)]
        assert contents == expected

    def test_organise_y_parallelized_files(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=5, nt=1, syn_data_type='stepped')
        prefix = 'BOUT.dmp'
        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=1, nype=5)
        assert ds_grid.shape == (5,)
        assert concat_dims == ['y']

        # check datasets are in the right order
        contents = [np.unique(ds_dict['key']['n'].values) for ds_dict in ds_grid]
        expected = [np.array([i]) for i in range(5)]
        assert contents == expected

    def test_organise_xy_parallelized_files(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=6, nype=3, nt=1, syn_data_type='stepped')
        prefix = 'BOUT.dmp'
        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=3)
        assert ds_grid.shape == (6,3)
        assert concat_dims == ['x', 'y']

        # check datasets are in the right order
        contents = np.empty(ds_grid.shape)
        for index, ds_dict in np.ndenumerate(ds_grid):
            contents[index] = np.unique(ds_dict['key']['n'].values)
        expected = np.array([[i+6*j for j in range(3)] for i in range(6)])
        npt.assert_equal(contents, expected)

    def test_organise_t_parallelized_files(self):
        pass

    def test_organise_xyt_parallelized_files(self):
        pass


class TestTrim:
    def test_no_trim(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=6, nype=3, nt=1, syn_data_type='stepped')
        prefix = 'BOUT.dmp'
        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=3)

        not_trimmed = _trim(ds_grid, concat_dims=['x', 'y'],
                            guards={'x': 0, 'y': 0}, ghosts={'x': 0, 'y': 0}, keep_guards={'x': None, 'y': None})
        assert_dataset_grids_equal(not_trimmed, ds_grid)

    def test_trim_ghosts(self, tmpdir_factory):
        # Create data to trim
        prefix = 'BOUT.dmp'
        path = bout_xyt_example_files(tmpdir_factory, lengths=(6,10,1,6), nxpe=6, nype=3, nt=1, ghosts={'x': 2, 'y': 3},
                                      syn_data_type='stepped')

        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=3)

        # Trim data
        trimmed = _trim(ds_grid, concat_dims=['x', 'y'],
                        guards={'x': 0, 'y': 0}, ghosts={'x': 2, 'y': 3}, keep_guards={'x': None, 'y': None})

        # Create data that is already the size it should be after trimming, to compare to
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2,4,1,6), nxpe=6, nype=3, nt=1, ghosts={'x': 2, 'y': 3},
                                      syn_data_type='stepped')
        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        expected, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=3)

        assert_dataset_grids_equal(trimmed, expected)

    def test_trim_guards(self, tmpdir_factory):
        # Create data to trim
        prefix = 'BOUT.dmp'
        guards = {'x': 2}
        keep_guards = {'x': False}
        ghosts = {'x': 0, 'y': 0}
        lengths = (6, 8, 1, 6)

        path = bout_xyt_example_files(tmpdir_factory, lengths=lengths, nxpe=6, nype=1, nt=1,
                                      ghosts=ghosts, guards=guards,
                                      syn_data_type='stepped')

        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=1)

        # Trim data
        trimmed = _trim(ds_grid, concat_dims=['x'],
                        guards=guards, ghosts=ghosts, keep_guards=keep_guards)


        # Create data that is already the size it should be after trimming, to compare to
        path = bout_xyt_example_files(tmpdir_factory, lengths=lengths, nxpe=6, nype=1, nt=1,
                                      syn_data_type='stepped')
        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=1)
        ds_grid[0]['key'] = ds_grid[0]['key'].isel(**{'x': slice(2, None, None)})
        ds_grid[5]['key'] = ds_grid[5]['key'].isel(**{'x': slice(None, -2, None)})
        expected = ds_grid

        assert_dataset_grids_equal(trimmed, expected)

    def test_keep_guards(self, tmpdir_factory):
        # Create data to trim
        prefix = 'BOUT.dmp'
        guards = {'x': 2}
        keep_guards = {'x': True}
        ghosts = {'x': 0, 'y': 0}
        lengths = (6, 8, 1, 6)

        path = bout_xyt_example_files(tmpdir_factory, lengths=lengths, nxpe=6, nype=1, nt=1,
                                      ghosts=ghosts, guards=guards,
                                      syn_data_type='stepped')

        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=1)

        # Trim data
        trimmed = _trim(ds_grid, concat_dims=['x'],
                        guards=guards, ghosts=ghosts, keep_guards=keep_guards)


        # Create data that is already the size it should be after trimming, to compare to
        # Same size because no ghost cells and we're keeping the guard cells
        path = bout_xyt_example_files(tmpdir_factory, lengths=lengths, nxpe=6, nype=1, nt=1,
                                      syn_data_type='stepped')
        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=1)
        ds_grid[0]['key'] = ds_grid[0]['key'].isel(**{'x': slice(1, None, None)})
        ds_grid[5]['key'] = ds_grid[5]['key'].isel(**{'x': slice(None, -1, None)})
        expected = ds_grid

        assert_dataset_grids_equal(trimmed, expected)

    def test_trim_ghosts_and_guards(self, tmpdir_factory):
        # Create data to trim
        prefix = 'BOUT.dmp'
        guards = {'x': 3}
        keep_guards = {'x': False}
        ghosts = {'x': 2, 'y': 0}
        lengths = (6, 8, 1, 6)

        path = bout_xyt_example_files(tmpdir_factory, lengths=lengths, nxpe=6, nype=1, nt=1,
                                      ghosts=ghosts, guards=guards,
                                      syn_data_type='stepped')

        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=1)

        # Trim data
        trimmed = _trim(ds_grid, concat_dims=['x'],
                        guards=guards, ghosts=ghosts, keep_guards=keep_guards)

        # Create data that is already the size it should be after trimming, to compare to
        path = bout_xyt_example_files(tmpdir_factory, lengths=(4, 8, 1, 6), nxpe=6, nype=1, nt=1,
                                      syn_data_type='stepped')
        filepaths, datasets = _open_all_dump_files(Path(path), chunks=None)
        ds_grid, concat_dims = _organise_files(filepaths, datasets, nxpe=6, nype=1)
        expected = ds_grid

        assert_dataset_grids_equal(trimmed, expected)


class TestCollectData:
    def test_collect_from_single_file(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        actual = collect(vars='all', datapath=path)
        expected = create_bout_ds()
        xrt.assert_equal(actual, expected)

    def test_collect_single_variables(self, tmpdir_factory):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        actual = collect(vars='n', datapath=path)
        expected = create_bout_ds()
        xrt.assert_equal(actual, expected['n'])

    def test_collect_multiple_files(self, tmpdir_factory):
        # Create data to trim
        prefix = 'BOUT.dmp'
        guards = {'x': 2, 'y': 0}
        keep_guards = {'x': False, 'y': True}
        ghosts = {'x': 2, 'y': 0}
        lengths = (6, 8, 10, 6)  # remember these are lengths before any guard or ghost cells are added!
        nxpe = 6
        nype = 1

        path = bout_xyt_example_files(tmpdir_factory, lengths=lengths, nxpe=nxpe, nype=nype, nt=1,
                                      ghosts=ghosts, guards=guards,
                                      syn_data_type='stepped')

        actual = collect(datapath=path, xguards=keep_guards['x'], yguards=keep_guards['y'])

        assert actual.dims == {'x': lengths[0]*nxpe, 'y': lengths[1]*nype,
                               'z': lengths[2], 't': lengths[3]}

    @pytest.mark.skip
    def test_against_open_mfdataset(self):
        pass

    @pytest.mark.skip
    def test_without_glob(self):
        pass
