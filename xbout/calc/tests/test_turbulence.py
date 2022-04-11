import numpy as np
from xarray import DataArray
import pytest
import numpy.testing as npt

from xbout.calc.turbulence import rms


class TestRootMeanSquare:
    def test_no_dim(self):
        dat = np.array([5, 7, 3.2, -1, -4.4])
        orig = DataArray(dat, dims=["x"])

        # Need to supply dimension
        with pytest.raises(ValueError):
            rms(orig, dim=None)

    def test_1d(self):
        dat = np.array([5, 7, 3.2, -1, -4.4])
        orig = DataArray(dat, dims=["x"])

        sum_squares = np.sum(dat**2)
        mean_squares = sum_squares / dat.size
        rootmeansquare = np.sqrt(mean_squares)

        expected = rootmeansquare
        actual = rms(orig, dim="x").values
        npt.assert_equal(actual, expected)

    @pytest.mark.parametrize("dim, axis", [("t", 1), ("x", 0)])
    def test_reduce_2d(self, dim, axis):
        dat = np.array([[5, 7, 3.2, -1, -4.4], [-1, -2.5, 0, 8, 3.0]])
        orig = DataArray(dat, dims=["x", "t"])
        sum_squares = np.sum(dat**2, axis=axis)
        mean_squares = sum_squares / dat.shape[axis]
        rootmeansquare = np.sqrt(mean_squares)

        expected = rootmeansquare
        actual = rms(orig, dim=dim).values
        npt.assert_equal(actual, expected)

    def test_reduce_2d_dask(self):
        dat = np.array([[5, 7, 3.2, -1, -4.4], [-1, -2.5, 0, 8, 3.0]])
        orig = DataArray(dat, dims=["x", "t"])
        chunked = orig.chunk({"x": 1})
        axis = 1
        sum_squares = np.sum(dat**2, axis=axis)
        mean_squares = sum_squares / dat.shape[axis]
        rootmeansquare = np.sqrt(mean_squares)

        expected = rootmeansquare
        actual = rms(chunked, dim="t").values
        npt.assert_equal(actual, expected)
