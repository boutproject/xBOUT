import pytest

from xbout import open_boutdataset
from xbout.boutdataarray import BoutDataArrayAccessor

from animatplot.blocks import Imshow, Line

DATA_PATH = './xbout/tests/data/dump_files/along_x/BOUT.dmp.*.nc' # Path to test dmp files

@pytest.fixture
def create_test_file(tmpdir_factory):

    save_dir = tmpdir_factory.mktemp("test_data") # Create temp dir for output of animate1D/2D
    ds = open_boutdataset(DATA_PATH).squeeze(drop=True) # Open test data

    return save_dir, ds

class TestAnimate:
    """
    Set of tests to check whether animate1D() and animate2D() are running properly
    and PillowWriter is saving each animation correctly
    """
    def test_animate2D(self, create_test_file):

        save_dir, ds = create_test_file
        animation = ds['T'].bout.animate2D(y='z', save_as="%s/test.gif" % save_dir)

        assert isinstance(animation, Imshow)

    def test_animate1D(self, create_test_file):

        save_dir, ds = create_test_file
        animation = ds['T'][:, :, 0].bout.animate1D(save_as="%s/test.gif" % save_dir)

        assert isinstance(animation, Line)

