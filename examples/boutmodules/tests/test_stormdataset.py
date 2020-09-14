import pytest

from xarray import concat
from xbout import open_boutdataset
from xbout.tests.test_load import bout_xyt_example_files


@pytest.mark.skip
class TestStormDataset:
    @pytest.mark.xfail
    def test_storm_dataset(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None)
        print(ds.storm.normalisation)

        assert False

    @pytest.mark.xfail
    def test_storm_dataset_inheritance(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None)
        ds.storm.set_extra_data("options")
        print(ds.storm.extra_data)

        assert False

    @pytest.mark.xfail
    def test_object_permanence(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None)

        ds.storm.extra_info = "options"
        new_ds = ds.isel(t=-1)
        print(new_ds.storm.extra_info)

        assert False

    @pytest.mark.xfail
    def test_dataset_duck_typing(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None)

        result = concat([ds.bout, ds.bout], dim="run")
        print(result)
