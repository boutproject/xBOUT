from pathlib import Path

import pytest


from xcollect.load import _check_filetype, _expand_wildcards, \
    _arrange_for_concatenation


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
        assert actual_concat_dims == []

    def test_arrange_along_x(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=1, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc',
                                './run0/BOUT.dmp.1.nc',
                                './run0/BOUT.dmp.2.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(paths, nxpe=3, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['x']

    def test_arrange_along_y(self, create_filepaths):
        paths = create_filepaths(nxpe=1, nype=3, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc'],
                               ['./run0/BOUT.dmp.1.nc'],
                               ['./run0/BOUT.dmp.2.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=1, nype=3)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['y']

    def test_arrange_along_t(self, create_filepaths):
        paths = create_filepaths(nxpe=1, nype=1, nt=3)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc']],
                              [['./run1/BOUT.dmp.0.nc']],
                              [['./run2/BOUT.dmp.0.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=1, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['t']

    def test_arrange_along_xy(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=2, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc', './run0/BOUT.dmp.1.nc', './run0/BOUT.dmp.2.nc'],
                               ['./run0/BOUT.dmp.3.nc', './run0/BOUT.dmp.4.nc', './run0/BOUT.dmp.5.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=2)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['x', 'y']

    def test_arrange_along_xt(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=1, nt=2)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc', './run0/BOUT.dmp.1.nc', './run0/BOUT.dmp.2.nc']],
                              [['./run1/BOUT.dmp.0.nc', './run1/BOUT.dmp.1.nc', './run1/BOUT.dmp.2.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['x', 't']

    def test_arrange_along_xyt(self, create_filepaths):
        paths = create_filepaths(nxpe=3, nype=2, nt=2)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc', './run0/BOUT.dmp.1.nc', './run0/BOUT.dmp.2.nc'],
                               ['./run0/BOUT.dmp.3.nc', './run0/BOUT.dmp.4.nc', './run0/BOUT.dmp.5.nc']],
                              [['./run1/BOUT.dmp.0.nc', './run1/BOUT.dmp.1.nc', './run1/BOUT.dmp.2.nc'],
                               ['./run1/BOUT.dmp.3.nc', './run1/BOUT.dmp.4.nc', './run1/BOUT.dmp.5.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(paths, nxpe=3, nype=2)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['x', 'y', 't']