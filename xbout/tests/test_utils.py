import pytest

from xarray import Dataset, DataArray

from xbout.utils import _set_attrs_on_all_vars, _update_metadata_increased_resolution


class TestUtils:
    def test__set_attrs_on_all_vars(self):
        ds = Dataset()
        ds["a"] = DataArray([0.0])
        ds["b"] = DataArray([1.0])

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
        ds = Dataset()
        ds["a"] = DataArray([0.0])
        ds["b"] = DataArray([1.0])

        _set_attrs_on_all_vars(ds, "metadata", {"x": 3}, copy=True)

        ds.metadata["x"] = 5
        # deepcopy done, so metadata on 'a' and 'b' should not have changed
        assert ds.metadata["x"] == 5
        assert ds["a"].metadata["x"] == 3
        assert ds["b"].metadata["x"] == 3

    def test__update_metadata_increased_resolution(self):
        da = DataArray()
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
