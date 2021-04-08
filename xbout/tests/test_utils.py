import pytest

import numpy as np
import xarray as xr
import xarray.testing as xrt

from xbout.utils import (
    _set_attrs_on_all_vars,
    _update_metadata_increased_resolution,
    _1d_coord_from_spacing,
)


class TestUtils:
    def test__set_attrs_on_all_vars(self):
        ds = xr.Dataset()
        ds["a"] = xr.DataArray([0.0])
        ds["b"] = xr.DataArray([1.0])

        _set_attrs_on_all_vars(ds, "i", 42)
        _set_attrs_on_all_vars(ds, "metadata", {"x": 3})

        assert ds.i == 42
        assert ds["a"].i == 42
        assert ds["b"].i == 42

        ds.metadata["x"] = 5
        # no deepcopy done, so metadata on 'a' and 'b' should have changed too
        assert ds["a"].metadata["x"] == 5
        assert ds["b"].metadata["x"] == 5

    def test__set_attrs_on_all_vars_copy(self):
        ds = xr.Dataset()
        ds["a"] = xr.DataArray([0.0])
        ds["b"] = xr.DataArray([1.0])

        _set_attrs_on_all_vars(ds, "metadata", {"x": 3}, copy=True)

        ds.metadata["x"] = 5
        # deepcopy done, so metadata on 'a' and 'b' should not have changed
        assert ds.metadata["x"] == 5
        assert ds["a"].metadata["x"] == 3
        assert ds["b"].metadata["x"] == 3

    def test__update_metadata_increased_resolution(self):
        da = xr.DataArray()
        da.attrs["metadata"] = {
            "jyseps1_1": 1,
            "jyseps2_1": 2,
            "ny_inner": 3,
            "jyseps1_2": 4,
            "jyseps2_2": 5,
            "ny": 6,
            "MYSUB": 7,
        }

        da = _update_metadata_increased_resolution(da, 3)

        assert da.metadata["jyseps1_1"] == 5
        assert da.metadata["jyseps2_1"] == 8
        assert da.metadata["jyseps1_2"] == 14
        assert da.metadata["jyseps2_2"] == 17

        assert da.metadata["ny_inner"] == 9
        assert da.metadata["ny"] == 18
        assert da.metadata["MYSUB"] == 21

    @pytest.mark.parametrize(
        "origin_at", [None, "lower", "centre", "upper", "expectfail"]
    )
    def test__1d_coord_from_spacing_scalar(self, origin_at):
        ds = xr.Dataset()
        ds["foo"] = ("testdim", np.arange(10))

        if origin_at == "expectfail":
            # fail with no ds for scalar argument
            with pytest.raises(ValueError):
                coord = _1d_coord_from_spacing(0.2, "testdim")
            with pytest.raises(ValueError):
                coord = _1d_coord_from_spacing(0.2, "testdim", ds, origin_at=origin_at)
            return

        coord = _1d_coord_from_spacing(0.2, "testdim", ds, origin_at=origin_at)

        if origin_at is None or origin_at == "lower":
            xrt.assert_allclose(
                coord,
                xr.Variable("testdim", np.linspace(0.1, 1.9, 10)),
                rtol=2.0e-15,
                atol=0.0,
            )
        elif origin_at == "centre":
            xrt.assert_allclose(
                coord,
                xr.Variable("testdim", np.linspace(-0.9, 0.9, 10)),
                rtol=2.0e-15,
                atol=0.0,
            )
        elif origin_at == "upper":
            xrt.assert_allclose(
                coord,
                xr.Variable("testdim", np.linspace(-1.9, -0.1, 10)),
                rtol=2.0e-15,
                atol=0.0,
            )
        else:
            assert False

    @pytest.mark.parametrize(
        "origin_at", [None, "lower", "centre", "upper", "expectfail"]
    )
    def test__1d_coord_from_spacing_da(self, origin_at):
        da = xr.DataArray(0.2 * np.ones(10), dims="testdim")
        da.attrs["metadata"] = {
            "bout_xdim": "x",
            "bout_ydim": "y",
            "bout_zdim": "z",
            "keep_xboundaries": True,
            "keep_yboundaries": True,
        }

        if origin_at == "expectfail":
            with pytest.raises(ValueError):
                coord = _1d_coord_from_spacing(da, "testdim", origin_at=origin_at)
            return

        coord = _1d_coord_from_spacing(da, "testdim", origin_at=origin_at)

        if origin_at is None or origin_at == "lower":
            xrt.assert_allclose(
                coord,
                xr.Variable("testdim", np.linspace(0.1, 1.9, 10)),
                rtol=2.0e-15,
                atol=0.0,
            )
        elif origin_at == "centre":
            xrt.assert_allclose(
                coord,
                xr.Variable("testdim", np.linspace(-0.9, 0.9, 10)),
                rtol=2.0e-15,
                atol=0.0,
            )
        elif origin_at == "upper":
            xrt.assert_allclose(
                coord,
                xr.Variable("testdim", np.linspace(-1.9, -0.1, 10)),
                rtol=2.0e-15,
                atol=0.0,
            )
        else:
            assert False
