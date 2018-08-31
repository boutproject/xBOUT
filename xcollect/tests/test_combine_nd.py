import os

import numpy as np
import pytest
import xarray.tests as xrt
from xarray import Dataset, open_mfdataset
from xarray.tests.test_dataset import create_test_data

from xcollect.concatenate import concat_nd


class TestConcatND:
    def test_numpy_containing_xarray(self):
        data = create_test_data()
        data_grid = np.array(data, dtype='object')
        print(data_grid.item())

        assert False

    def test_concat_1d(self):
        # drop the third dimension to keep things relatively understandable
        data = create_test_data()
        #for k in list(data.variables):
        #    if 'dim3' in data[k].dims:
        #        del data[k]

        split_data = [data.isel(dim1=slice(3)),
                      data.isel(dim1=slice(3, None))]

        #print(split_data)
        split_data_grid = np.array(split_data, dtype='object')

        print(split_data_grid)
        reconstructed = concat_nd(split_data_grid, ['dim1'])

        xrt.assert_identical(data, reconstructed)

    @pytest.mark.xfail(reason='Not yet implemented')
    def test_concat_2d(self):
        pass

    @pytest.mark.xfail(reason='Not yet implemented')
    def test_concat_3d(self):
        pass

    @pytest.mark.xfail(reason='Not yet implemented')
    def test_concat_bad_args(self):
        pass

    @pytest.mark.xfail(reason='Not yet implemented')
    def test_concat_new_dim(self):
        pass

    @pytest.mark.xfail(reason='Not yet implemented')
    def test_concat_mixed_dims(self):
        pass


class TestOpenMFDatasetND:
    @pytest.mark.xfail(reason='Not yet implemented all the way up open_mfdataset().')
    def motivating_test(self):
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

        xrt.assert_equal(expected, actual)

        os.remove('ds.*.nc')
