import numpy as np
import pytest
import xarray.tests as xrt
from xarray import Dataset, open_mfdataset, concat
from xarray.tests.test_dataset import create_test_data
from xarray.core.merge import MergeError

import os
import glob

from xcollect.concatenate import _concat_nd


class TestNumpyContainingXarray:
    """
    This test class is supposed to demonstrate why the function _concat_dicts() must be used instead of simply concat().
    """

    def test_numpy_containing_xarray(self):
        data = create_test_data()
        with pytest.raises(TypeError):
            data_grid = np.array(data, dtype='object')

    def test_numpy_containing_xarray_box(self):
        data = create_test_data()
        box = {'key': data}
        data_grid = np.array(box, dtype='object')
        actual = data_grid.item()['key']
        xrt.assert_equal(actual, data)


class TestConcatND:
    def test_no_concat_needed(self):
        data = create_test_data()
        data_grid = np.array({'key': data}, dtype=np.object)
        actual = _concat_nd(data_grid)
        xrt.assert_equal(data, actual)

    def test_concat_1d(self):
        data = create_test_data()

        split_data = [{'key': data.isel(dim1=slice(3))},
                      {'key': data.isel(dim1=slice(3, None))}]
        split_data_grid = np.array(split_data, dtype=np.object)

        reconstructed = _concat_nd(split_data_grid, concat_dims=['dim1'])
        xrt.assert_identical(data, reconstructed)

    def test_concat_2d(self):
        data = create_test_data()

        split_data = [[{'key': data.isel(dim1=slice(3),       dim2=slice(4))},
                       {'key': data.isel(dim1=slice(3),       dim2=slice(4, None))},],
                      [{'key': data.isel(dim1=slice(3, None), dim2=slice(4))},
                       {'key': data.isel(dim1=slice(3, None), dim2=slice(4, None))}]]
        split_data_grid = np.array(split_data, dtype=np.object)

        # Has to be minimal because otherwise the (dim1, dim3) array will be copied upon concatenation along dim2
        reconstructed = _concat_nd(split_data_grid, concat_dims=['dim1', 'dim2'], data_vars=['minimal'] * 2)
        xrt.assert_identical(data, reconstructed)

    def test_concat_3d(self):
        data = create_test_data()

        split_data = [[[{'key': data.isel(dim1=slice(3),       dim2=slice(4),       time=slice(10))},
                        {'key': data.isel(dim1=slice(3),       dim2=slice(4),       time=slice(10, None))}],
                       [{'key': data.isel(dim1=slice(3),       dim2=slice(4, None), time=slice(10))},
                        {'key': data.isel(dim1=slice(3),       dim2=slice(4, None), time=slice(10, None))}]],
                      [[{'key': data.isel(dim1=slice(3, None), dim2=slice(4),       time=slice(10))},
                        {'key': data.isel(dim1=slice(3, None), dim2=slice(4),       time=slice(10, None))}],
                       [{'key': data.isel(dim1=slice(3, None), dim2=slice(4, None), time=slice(10))},
                        {'key': data.isel(dim1=slice(3, None), dim2=slice(4, None), time=slice(10, None))}]]]
        split_data_grid = np.array(split_data, dtype=np.object)

        reconstructed = _concat_nd(split_data_grid, concat_dims= ['dim1', 'dim2', 'time'], data_vars=['minimal'] * 3)
        xrt.assert_identical(data, reconstructed)

    def test_concat_1d_dask(self):
        data = create_test_data()
        data = data.chunk({'dim1': 1})

        split_data = [{'key': data.isel(dim1=slice(3))},
                      {'key': data.isel(dim1=slice(3, None))}]
        split_data_grid = np.array(split_data, dtype=np.object)

        reconstructed = _concat_nd(split_data_grid, concat_dims=['dim1'])
        xrt.assert_identical(data, reconstructed)

    def test_concat_2d_dask(self):
        data = create_test_data()
        data = data.chunk({'dim1': 1, 'dim2': 2})

        split_data = [[{'key': data.isel(dim1=slice(3),       dim2=slice(4))},
                       {'key': data.isel(dim1=slice(3),       dim2=slice(4, None))},],
                      [{'key': data.isel(dim1=slice(3, None), dim2=slice(4))},
                       {'key': data.isel(dim1=slice(3, None), dim2=slice(4, None))}]]
        split_data_grid = np.array(split_data, dtype=np.object)

        # Has to be minimal because otherwise the (dim1, dim3) array will be copied upon concatenation along dim2
        reconstructed = _concat_nd(split_data_grid, concat_dims=['dim1', 'dim2'], data_vars=['minimal'] * 2)
        xrt.assert_identical(data, reconstructed)

    def test_concat_wrong_size_grid(self):
        grid = np.array([['a', 'b']])
        with pytest.raises(MergeError):
            _concat_nd(grid)

    def test_concat_new_dim(self):
        data1 = create_test_data(seed=1)
        data2 = create_test_data(seed=2)

        split_data = [{'key': data1},
                      {'key': data2}]
        split_data_grid = np.array(split_data, dtype=np.object)

        expected = concat([data1, data2], dim='run')
        reconstructed = _concat_nd(split_data_grid, concat_dims=['run'])

        xrt.assert_identical(expected, reconstructed)

    def test_concat_mixed_dims(self):
        data1 = create_test_data(seed=1)
        data2 = create_test_data(seed=2)

        split_data = [[{'key': data1.isel(dim1=slice(3))},
                       {'key': data2.isel(dim1=slice(3))}],
                      [{'key': data1.isel(dim1=slice(3, None))},
                       {'key': data2.isel(dim1=slice(3, None))}]]
        split_data_grid = np.array(split_data, dtype=np.object)

        reconstructed = _concat_nd(split_data_grid, concat_dims=['dim1', 'run'], data_vars=['minimal', 'all'])
        expected = concat([data1, data2], dim='run', data_vars='all')
        xrt.assert_identical(expected, reconstructed)


class TestOpenMFDatasetND:
    @pytest.mark.xfail(reason='Not yet implemented all the way up to open_mfdataset().')
    def test_motivating_test(self):
        """This is the failing test which motivated issue https://github.com/pydata/xarray/issues/2159"""

        # Create 4 datasets containing sections of contiguous (x,y) data
        for i, x in enumerate([1, 3]):
            for j, y in enumerate([10, 40]):
                ds = Dataset({'foo': (('x', 'y'), np.ones((2, 3)))},
                                coords={'x': [x, x + 1],
                                        'y': [y, y + 10, y + 20]})

                ds.to_netcdf('ds.' + str(i) + str(j) + '.nc')

        # Try to open them all in one go
        actual = open_mfdataset('ds.*.nc')

        expected = Dataset({'foo': (('x', 'y'), np.ones((4, 6)))},
                                coords={'x': [1, 2, 3, 4],
                                        'y': [10, 20, 30, 40, 50, 60]})

        os.remove(glob.glob('ds.*.nc'))
        xrt.assert_equal(expected, actual)
