from xbout.geometries import apply_geometry
from xbout.geometries import UnregisteredGeometryError

import pytest


class TestGeometryRegistration:
    def test_unregistered_geometry(self):
        with pytest.raises(UnregisteredGeometryError,
                           match="tesseract is not a registered geometry"):
            apply_geometry(ds=None, geometry_name="tesseract")

