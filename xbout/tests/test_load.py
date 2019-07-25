from pathlib import Path
import re

import pytest

import numpy as np

from xarray import DataArray, Dataset, concat
from xarray.tests.test_dataset import create_test_data
import xarray.testing as xrt

from natsort import natsorted

from xbout.load import _check_filetype, _expand_wildcards, _expand_filepaths,\
    _arrange_for_concatenation, _trim, _strip_metadata, \
    _auto_open_mfboutdataset, _infer_contains_boundaries


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
        expected_filepaths = natsorted(filepaths,
                                       key=lambda filepath: str(filepath))

        path = Path(str(files_dir.join('run*/example.*.nc')))
        actual_filepaths = _expand_wildcards(path)

        assert actual_filepaths == expected_filepaths

    def test_no_files(self, tmpdir):
        files_dir = tmpdir.mkdir("data")

        with pytest.raises(IOError):
            path = Path(str(files_dir.join('run*/example.*.nc')))
            actual_filepaths = _expand_filepaths(path)
            print(actual_filepaths)


@pytest.fixture()
def create_filepaths():
    return _create_filepaths


def _create_filepaths(nxpe=1, nype=1, nt=1):
    filepaths = []
    for t in range(nt):
        for i in range(nype):
            for j in range(nxpe):
                file_num = (j + nxpe * i)
                path = './run{}'.format(str(t)) \
                       + '/BOUT.dmp.{}.nc'.format(str(file_num))
                filepaths.append(path)

    return filepaths


class TestArrange:
    def test_arrange_single(self, create_filepaths):
        paths = create_filepaths(nxpe=1, nype=1, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(paths, nxpe=1, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == [None, None, None]

    def test_arrange_along_x(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=1, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc',
                                './run0/BOUT.dmp.1.nc',
                                './run0/BOUT.dmp.2.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(paths, nxpe=3, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == [None, None, 'x']

    def test_arrange_along_y(self, create_filepaths):
        paths = create_filepaths(nxpe=1, nype=3, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc'],
                               ['./run0/BOUT.dmp.1.nc'],
                               ['./run0/BOUT.dmp.2.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=1, nype=3)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == [None, 'y', None]

    def test_arrange_along_t(self, create_filepaths):
        paths = create_filepaths(nxpe=1, nype=1, nt=3)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc']],
                              [['./run1/BOUT.dmp.0.nc']],
                              [['./run2/BOUT.dmp.0.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=1, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['t', None, None]

    def test_arrange_along_xy(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=2, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc', './run0/BOUT.dmp.1.nc', './run0/BOUT.dmp.2.nc'],
                               ['./run0/BOUT.dmp.3.nc', './run0/BOUT.dmp.4.nc', './run0/BOUT.dmp.5.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=2)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == [None, 'y', 'x']

    def test_arrange_along_xt(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=1, nt=2)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc', './run0/BOUT.dmp.1.nc', './run0/BOUT.dmp.2.nc']],
                              [['./run1/BOUT.dmp.0.nc', './run1/BOUT.dmp.1.nc', './run1/BOUT.dmp.2.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['t', None, 'x']

    def test_arrange_along_xyt(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=2, nt=2)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc', './run0/BOUT.dmp.1.nc', './run0/BOUT.dmp.2.nc'],
                               ['./run0/BOUT.dmp.3.nc', './run0/BOUT.dmp.4.nc', './run0/BOUT.dmp.5.nc']],
                              [['./run1/BOUT.dmp.0.nc', './run1/BOUT.dmp.1.nc', './run1/BOUT.dmp.2.nc'],
                               ['./run1/BOUT.dmp.3.nc', './run1/BOUT.dmp.4.nc', './run1/BOUT.dmp.5.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(paths, nxpe=3, nype=2)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['t', 'y', 'x']


@pytest.fixture()
def bout_xyt_example_files(tmpdir_factory):
    return _bout_xyt_example_files


def _bout_xyt_example_files(tmpdir_factory, prefix='BOUT.dmp', lengths=(2,4,7,6),
                            nxpe=4, nype=2, nt=1, guards={}, syn_data_type='random'):
    """
    Mocks up a set of BOUT-like netCDF files, and return the temporary test directory containing them.

    Deletes the temporary directory once that test is done.
    """

    save_dir = tmpdir_factory.mktemp("data")

    ds_list, file_list = create_bout_ds_list(prefix=prefix, lengths=lengths, nxpe=nxpe, nype=nype, nt=nt,
                                             guards=guards, syn_data_type=syn_data_type)

    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(str(save_dir.join(str(file_name))))

    # Return a glob-like path to all files created, which has all file numbers replaced with a single asterix
    path = str(save_dir.join(str(file_list[-1])))

    count = 1
    if nt > 1:
        count += 1
    # We have to reverse the path before limiting the number of numbers replaced so that the
    # tests don't get confused by pytest's persistent temporary directories (which are also designated
    # by different numbers)
    glob_pattern = (re.sub(r'\d+', '*', path[::-1], count=count))[::-1]
    return glob_pattern


def create_bout_ds_list(prefix, lengths=(2, 4, 7, 6), nxpe=4, nype=2, nt=1, guards={},
                        syn_data_type='random'):
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

            # Include guard cells
            upper_bndry_cells = {dim: guards.get(dim) for dim in guards.keys()}
            lower_bndry_cells = {dim: guards.get(dim) for dim in guards.keys()}

            # Include boundary cells
            for dim in ['x', 'y']:
                if dim in guards.keys():
                    if i == 0:
                        lower_bndry_cells[dim] = guards[dim]
                    if i == nxpe-1:
                        upper_bndry_cells[dim] = guards[dim]

            ds = create_bout_ds(syn_data_type=syn_data_type, num=num, lengths=lengths, nxpe=nxpe, nype=nype,
                                upper_bndry_cells=upper_bndry_cells, lower_bndry_cells=lower_bndry_cells,
                                guards=guards)
            ds_list.append(ds)

    # Sort this in order of num to remove any BOUT-specific structure
    ds_list_sorted = [ds for filename, ds in sorted(zip(file_list, ds_list))]
    file_list_sorted = [filename for filename, ds in sorted(zip(file_list, ds_list))]

    return ds_list_sorted, file_list_sorted


def create_bout_ds(syn_data_type='random', lengths=(2,4,7,6), num=0, nxpe=1, nype=1,
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
    elif isinstance(syn_data_type, int):
        data = np.ones(shape)* syn_data_type
    else:
        raise ValueError('Not a recognised choice of type of synthetic bout data.')

    T = DataArray(data, dims=['x', 'y', 'z', 't'])
    n = DataArray(data, dims=['x', 'y', 'z', 't'])
    ds = Dataset({'n': n, 'T': T})

    # Include metadata
    ds['NXPE'] = nxpe
    ds['NYPE'] = nype
    ds['MXG'] = guards.get('x', 0)
    ds['MYG'] = guards.get('y', 0)
    ds['nx'] = x_length
    ds['MXSUB'] = guards.get('x', 0)
    ds['MYSUB'] = guards.get('y', 0)
    ds['MZ'] = z_length

    return ds


METADATA_VARS = ['NXPE', 'NYPE', 'MXG', 'MYG', 'nx', 'MXSUB', 'MYSUB', 'MZ']


class TestStripMetadata():
    def test_strip_metadata(self):

        original = create_bout_ds()
        assert original['NXPE'] == 1

        ds, metadata = _strip_metadata(original)

        assert original.drop(METADATA_VARS).equals(ds)
        assert metadata['NXPE'] == 1


# TODO also test loading multiple files which have guard cells
class TestCombineNoTrim:
    def test_single_file(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        actual, metadata = _auto_open_mfboutdataset(datapath=path)
        expected = create_bout_ds()
        xrt.assert_equal(actual.load(), expected.drop(METADATA_VARS))

    def test_combine_along_x(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=1, nt=1,
                                      syn_data_type='stepped')
        actual, metadata = _auto_open_mfboutdataset(datapath=path)

        bout_ds = create_bout_ds
        expected = concat([bout_ds(0), bout_ds(1), bout_ds(2), bout_ds(3)], dim='x')
        xrt.assert_equal(actual.load(), expected.drop(METADATA_VARS))

    def test_combine_along_y(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=3, nt=1,
                                      syn_data_type='stepped')
        actual, metadata = _auto_open_mfboutdataset(datapath=path)

        bout_ds = create_bout_ds
        expected = concat([bout_ds(0), bout_ds(1), bout_ds(2)], dim='y')
        xrt.assert_equal(actual.load(), expected.drop(METADATA_VARS))

    @pytest.mark.skip
    def test_combine_along_t(self):
        ...

    def test_combine_along_xy(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=3, nt=1,
                                      syn_data_type='stepped')
        actual, metadata = _auto_open_mfboutdataset(datapath=path)

        bout_ds = create_bout_ds
        line1 = concat([bout_ds(0), bout_ds(1), bout_ds(2), bout_ds(3)], dim='x')
        line2 = concat([bout_ds(4), bout_ds(5), bout_ds(6), bout_ds(7)], dim='x')
        line3 = concat([bout_ds(8), bout_ds(9), bout_ds(10), bout_ds(11)], dim='x')
        expected = concat([line1, line2, line3], dim='y')
        xrt.assert_equal(actual.load(), expected.drop(METADATA_VARS))

    @pytest.mark.skip
    def test_combine_along_tx(self):
        ...


class TestTrim:
    def test_no_trim(self):
        ds = create_test_data(0)
        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding['source'] = 'folder0/BOUT.dmp.0.nc'
        actual = _trim(ds, guards={}, keep_boundaries={}, nxpe=1,
                       nype=1)
        xrt.assert_equal(actual, ds)

    def test_trim_guards(self):
        ds = create_test_data(0)
        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding['source'] = 'folder0/BOUT.dmp.0.nc'
        actual = _trim(ds, guards={'time': 2}, keep_boundaries={},
                       nxpe=1, nype=1)
        selection = {'time': slice(2, -2)}
        expected = ds.isel(**selection)
        xrt.assert_equal(expected, actual)

    @pytest.mark.parametrize("filenum, nxpe, nype, lower_boundaries, upper_boundaries",
                             # no parallelization
                             [(0,      1,    1,    {'x': True,  'y': True},
                                                   {'x': True,  'y': True}),

                              # 1d parallelization along x:
                              # Left
                              (0,      3,    1,    {'x': True,  'y': True},
                                                   {'x': False, 'y': True}),
                              # Middle
                              (1,      3,    1,    {'x': False, 'y': True},
                                                   {'x': False, 'y': True}),
                              # Right
                              (2,      3,    1,    {'x': False, 'y': True},
                                                   {'x': True,  'y': True}),

                              # 1d parallelization along y:
                              # Bottom
                              (0,      1,    3,    {'x': True,  'y': True},
                                                   {'x': True,  'y': False}),
                              # Middle
                              (1,      1,    3,    {'x': True,  'y': False},
                                                   {'x': True,  'y': False}),
                              # Top
                              (2,      1,    3,    {'x': True,  'y': False},
                                                   {'x': True,  'y': True}),

                              # 2d parallelization:
                              # Bottom left corner
                              (0,      3,    4,    {'x': True,  'y': True},
                                                   {'x': False, 'y': False}),
                              # Bottom right corner
                              (2,      3,    4,    {'x': False, 'y': True},
                                                   {'x': True,  'y': False}),
                              # Top left corner
                              (9,      3,    4,    {'x': True,  'y': False},
                                                   {'x': False, 'y': True}),
                              # Top right corner
                              (11,     3,    4,    {'x': False, 'y': False},
                                                   {'x': True,  'y': True}),
                              # Centre
                              (7,      3,    4,    {'x': False, 'y': False},
                                                   {'x': False, 'y': False}),
                              # Left side
                              (3,      3,    4,    {'x': True,  'y': False},
                                                   {'x': False, 'y': False}),
                              # Right side
                              (5,      3,    4,    {'x': False, 'y': False},
                                                   {'x': True,  'y': False}),
                              # Bottom side
                              (1,      3,    4,    {'x': False, 'y': True},
                                                   {'x': False, 'y': False}),
                              # Top side
                              (10,     3,    4,    {'x': False, 'y': False},
                                                   {'x': False, 'y': True})
                              ])
    def test_infer_boundaries_2d_parallelization(self, filenum, nxpe, nype,
                                                 lower_boundaries, upper_boundaries):
        """
        Numbering scheme for nxpe=3, nype=4

        y  9 10 11
        ^  6 7  8
        |  3 4  5
        |  0 1  2
         -----> x
        """

        filename = "folder0/BOUT.dmp." + str(filenum) + ".nc"
        actual_lower_boundaries, actual_upper_boundaries = _infer_contains_boundaries(
            filename, nxpe, nype)

        assert actual_lower_boundaries == lower_boundaries
        assert actual_upper_boundaries == upper_boundaries

    def test_keep_xboundaries(self):
        ds = create_test_data(0)
        ds = ds.rename({'dim2': 'x'})

        # Manually add filename - encoding normally added by xr.open_dataset
        ds.encoding['source'] = 'folder0/BOUT.dmp.0.nc'

        actual = _trim(ds, guards={'x': 2}, keep_boundaries={'x': True}, nxpe=1, nype=1)
        expected = ds  # Should be unchanged
        xrt.assert_equal(expected, actual)

    def test_trim_timing_info(self):
        ds = create_test_data(0)
        from xbout.load import _BOUT_TIMING_VARIABLES

        # remove a couple of entries from _BOUT_TIMING_VARIABLES so we test that _trim
        # does not fail if not all of them are present
        _BOUT_TIMING_VARIABLES = _BOUT_TIMING_VARIABLES[:-2]

        for v in _BOUT_TIMING_VARIABLES:
            ds[v] = 42.
        ds = _trim(ds, guards={}, keep_boundaries={}, nxpe=1, nype=1)

        expected = create_test_data(0)
        xrt.assert_equal(ds, expected)
