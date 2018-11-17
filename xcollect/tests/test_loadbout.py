import pytest


from xcollect.loadbout import _arrange_for_concatenation


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