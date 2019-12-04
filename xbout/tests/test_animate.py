import pytest

from xbout import open_boutdataset
from xbout.boutdataarray import BoutDataArrayAccessor
from .test_load import create_bout_ds_list

from animatplot.blocks import Imshow, Line


@pytest.fixture
def create_test_file(tmpdir_factory):

    # Create temp dir for output of animate1D/2D
    save_dir = tmpdir_factory.mktemp("test_data")

    # Generate some test data
    ds_list, file_list = create_bout_ds_list("BOUT.dmp", nxpe=3, nype=3,
                                             syn_data_type="linear")
    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(str(save_dir.join(str(file_name))))

    ds = open_boutdataset(save_dir.join("BOUT.dmp.*.nc"))  # Open test data

    return save_dir, ds


class TestAnimate:
    """
    Set of tests to check whether animate1D() and animate2D() are running properly
    and PillowWriter is saving each animation correctly
    """
    def test_animate2D(self, create_test_file):

        save_dir, ds = create_test_file
        animation = ds['n'].isel(y=1).bout.animate2D(y='z', save_as="%s/test" % save_dir)

        assert isinstance(animation, Imshow)

    def test_animate1D(self, create_test_file):

        save_dir, ds = create_test_file
        animation = ds['n'].isel(y=2, z=0).bout.animate1D(save_as="%s/test" % save_dir)

        assert isinstance(animation, Line)
