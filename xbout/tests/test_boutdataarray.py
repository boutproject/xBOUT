import pytest

from xarray.core.utils import dict_equiv

from xbout.tests.test_load import bout_xyt_example_files, create_bout_ds
from xbout import open_boutdataset

class TestBoutDataArrayMethods:

    def test_to_dataset(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=2)
        ds = open_boutdataset(datapath=path, inputfilepath=None, keep_xboundaries=False)
        da = ds['n']

        new_ds = da.bout.to_dataset()

        assert dict_equiv(ds.attrs, new_ds.attrs)
        assert dict_equiv(ds.metadata, new_ds.metadata)
