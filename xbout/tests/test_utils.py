import pytest

from xarray import Dataset, DataArray

from xbout.utils import _set_attrs_on_all_vars

class TestUtils:

    def test__set_attrs_on_all_vars(self):
        ds = Dataset()
        ds['a'] = DataArray([0.])
        ds['b'] = DataArray([1.])

        _set_attrs_on_all_vars(ds, 'i', 42)
        _set_attrs_on_all_vars(ds, 'metadata', {'x': 3})

        assert ds.i == 42
        assert ds['a'].i == 42
        assert ds['b'].i == 42

        ds.metadata['x'] = 5
        # no deepcopy done, so metadata on 'a' and 'b' should have changed too
        assert ds['a'].metadata['x'] == 5
        assert ds['b'].metadata['x'] == 5


    def test__set_attrs_on_all_vars_copy(self):
        ds = Dataset()
        ds['a'] = DataArray([0.])
        ds['b'] = DataArray([1.])

        _set_attrs_on_all_vars(ds, 'metadata', {'x': 3}, copy=True)

        ds.metadata['x'] = 5
        # deepcopy done, so metadata on 'a' and 'b' should not have changed
        assert ds.metadata['x'] == 5
        assert ds['a'].metadata['x'] == 3
        assert ds['b'].metadata['x'] == 3
