import pytest
from os import remove

from xbout import open_boutdataset
from xbout.boutdataarray import BoutDataArrayAccessor

DATA_PATH = './xbout/tests/data/dump_files/along_x/BOUT.dmp.*.nc'


class TestAnimate:
    """
    Set of tests to check whether animate1D() and animate2D() are running properly
    and PillowWriter is saving each animation correctly
    """
    def test_animate2D(self):

        bd = open_boutdataset(DATA_PATH).squeeze(drop=True)

        anim_creator = str(bd['T'].bout.animate2D(y='z'))
        checker = '<animatplot.blocks.image_like.Imshow' in anim_creator

        remove('./T_over_t.gif')
        assert checker

    def test_animate1D(self):

        bd = open_boutdataset(DATA_PATH).squeeze(drop=True)

        anim_creator = str(bd['T'][:, :, 0].bout.animate1D())
        checker = '<animatplot.blocks.lineplots.Line' in anim_creator

        remove('./T_over_t.gif')
        assert checker
