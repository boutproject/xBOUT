import numpy as np

from xarray import Dataset, DataArray
from xarray.testing import assert_equal
import pytest

from xbout.geometries import (
    apply_geometry,
    register_geometry,
    UnregisteredGeometryError,
    REGISTERED_GEOMETRIES,
)


class TestGeometryRegistration:
    def test_unregistered_geometry(self):
        with pytest.raises(
            UnregisteredGeometryError, match="tesseract is not a registered geometry"
        ):
            apply_geometry(ds=Dataset(), geometry_name="tesseract")

    def test_register_new_geometry(self):
        @register_geometry(name="Schwarzschild")
        def add_schwarzschild_coords(ds, coordinates=None):
            ds["event_horizon"] = 4.0
            return ds

        assert "Schwarzschild" in REGISTERED_GEOMETRIES.keys()

        original = Dataset()
        original["dy"] = DataArray(np.ones((3, 4)), dims=("x", "y"))
        metadata = {
            "bout_tdim": "t",
            "bout_xdim": "x",
            "bout_ydim": "y",
            "bout_zdim": "z",
            "keep_xboundaries": True,
            "keep_yboundaries": True,
        }
        original.attrs["metadata"] = metadata
        original["dy"].attrs["metadata"] = metadata
        updated = apply_geometry(ds=original, geometry_name="Schwarzschild")
        assert_equal(updated["event_horizon"], DataArray(4.0))

        # clean up
        del REGISTERED_GEOMETRIES["Schwarzschild"]
