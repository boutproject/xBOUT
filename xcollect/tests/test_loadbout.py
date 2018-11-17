import pytest


from xcollect.loadbout import _grouper, _arrange_for_concatenation


@pytest.fixture(scope='module')
def create_filepaths(nxpe=1, nype=1, nt=1):

    filepaths = []
    for t in range(nt):
        for i in range(nxpe):
            for j in range(nype):
                file_num = (i + nxpe * j)
                path = './run{}'.format(str(t)) \
                       + '/BOUT.dmp.{}.nc'.format(str(file_num))
                filepaths.append(path)

    return filepaths


class TestArrange:
    @pytest.mark.skip
    def test_group_once(self):
        paths = create_filepaths(nxpe=4, nype=1, nt=1)

        grouped = _grouper(paths, 2)
        print(list(grouped))
        assert False

    def test_arrange_single(self):
        paths = create_filepaths(nxpe=1, nype=1, nt=1)
        expected_path_grid = [[[['./run0/BOUT.dmp.0.nc']]]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(paths, nxpe=1, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == []

    def test_arrange_along_x(self):
        paths = create_filepaths(nxpe=3, nype=1, nt=1)
        expected_path_grid = [[[['./run0/BOUT.dmp.0.nc'],
                                ['./run0/BOUT.dmp.1.nc'],
                                ['./run0/BOUT.dmp.2.nc']]]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(paths, nxpe=3, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['x']

    def test_arrange_along_y(self):
        paths = create_filepaths(nxpe=1, nype=3, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc']],
                              [['./run0/BOUT.dmp.1.nc']],
                              [['./run0/BOUT.dmp.2.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=1, nype=3)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['y']

    def test_arrange_along_t(self):
        paths = create_filepaths(nxpe=1, nype=1, nt=3)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc']],
                              [['./run1/BOUT.dmp.0.nc']],
                              [['./run2/BOUT.dmp.0.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=1, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['t']

    def test_arrange_along_xy(self):
        paths = create_filepaths(nxpe=3, nype=2, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc', './run0/BOUT.dmp.1.nc', './run0/BOUT.dmp.2.nc'],
                               ['./run0/BOUT.dmp.3.nc', './run0/BOUT.dmp.4.nc', './run0/BOUT.dmp.5.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=2)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['x']

    def test_arrange_along_xt(self):
        paths = create_filepaths(nxpe=3, nype=1, nt=2)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc'], ['./run0/BOUT.dmp.1.nc'], ['./run0/BOUT.dmp.2.nc']],
                              [['./run0/BOUT.dmp.3.nc'], ['./run0/BOUT.dmp.4.nc'], ['./run0/BOUT.dmp.5.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(
            paths, nxpe=3, nype=1)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['t', 'x']

    def test_arrange_along_xyt(self):
        paths = create_filepaths(nxpe=3, nype=2, nt=1)
        expected_path_grid = [[['./run0/BOUT.dmp.0.nc', './run0/BOUT.dmp.1.nc', './run0/BOUT.dmp.2.nc'],
                               ['./run0/BOUT.dmp.3.nc', './run0/BOUT.dmp.4.nc', './run0/BOUT.dmp.5.nc']],
                              [['./run1/BOUT.dmp.0.nc', './run1/BOUT.dmp.1.nc', './run1/BOUT.dmp.2.nc'],
                               ['./run1/BOUT.dmp.3.nc', './run1/BOUT.dmp.4.nc', './run1/BOUT.dmp.5.nc']]]
        actual_path_grid, actual_concat_dims = _arrange_for_concatenation(paths, nxpe=3, nype=2)
        assert expected_path_grid == actual_path_grid
        assert actual_concat_dims == ['t', 'x', 'y']