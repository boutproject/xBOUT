import pytest

from pathlib import Path
from matplotlib import pyplot as plt

from xbout import open_boutdataset
from xbout.tests.test_load import bout_xyt_example_files
from xbout.tests.test_region import (params_guards, params_guards_values,
                                     params_boundaries, params_boundaries_values)


class TestPlot:

    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    @pytest.mark.parametrize('with_guards', [0, pytest.param(1, marks=pytest.mark.long),
                                             pytest.param(2, marks=pytest.mark.long)])
    def test_region_disconnecteddoublenull(self, tmpdir_factory, bout_xyt_example_files,
                                           guards, keep_xboundaries, keep_yboundaries,
                                           with_guards):
        # Note using more than MXG x-direction points and MYG y-direction points per
        # output file ensures tests for whether boundary cells are present do not fail
        # when using minimal numbers of processors
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 5, 4, 3), nxpe=3,
                                      nype=6, nt=1, guards=guards, grid='grid',
                                      topology='disconnected-double-null')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_xboundaries=keep_xboundaries,
                              keep_yboundaries=keep_yboundaries)

        n = ds['n'].isel(t=-1, zeta=0)

        n.bout.from_region('lower_inner_PFR', with_guards=with_guards).bout.pcolormesh()

        n.bout.from_region('lower_inner_intersep',
                           with_guards=with_guards).bout.contourf()

        n.bout.from_region('lower_inner_SOL', with_guards=with_guards).bout.contour()

        plt.figure()
        n.bout.from_region('inner_core', with_guards=with_guards).plot()

        n.bout.from_region('inner_intersep', with_guards=with_guards).bout.pcolormesh()

        n.bout.from_region('inner_SOL', with_guards=with_guards).bout.contourf()

        n.bout.from_region('upper_inner_PFR', with_guards=with_guards).bout.contour()

        plt.figure()
        n.bout.from_region('upper_inner_intersep', with_guards=with_guards).plot()

        n.bout.from_region('upper_inner_SOL', with_guards=with_guards).bout.pcolormesh()

        n.bout.from_region('upper_outer_PFR', with_guards=with_guards).bout.contourf()

        n.bout.from_region('upper_outer_intersep',
                           with_guards=with_guards).bout.contour()

        plt.figure()
        n.bout.from_region('upper_outer_SOL', with_guards=with_guards).plot()

        n.bout.from_region('outer_core', with_guards=with_guards).bout.pcolormesh()

        n.bout.from_region('outer_intersep', with_guards=with_guards).bout.contourf()

        n.bout.from_region('outer_SOL', with_guards=with_guards).bout.contour()

        plt.figure()
        n.bout.from_region('lower_outer_PFR', with_guards=with_guards).plot()

        n.bout.from_region('lower_outer_intersep',
                           with_guards=with_guards).bout.pcolormesh()

        n.bout.from_region('lower_outer_SOL', with_guards=with_guards).bout.contourf()
