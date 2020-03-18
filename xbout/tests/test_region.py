import pytest

from pathlib import Path

import xarray.testing as xrt

from xbout.tests.test_load import bout_xyt_example_files
from xbout import open_boutdataset

class TestRegion:

    params_guards = "guards"
    params_guards_values = [{'x': 0, 'y': 0}, {'x': 2, 'y': 0}, {'x': 0, 'y': 2},
                            {'x': 2, 'y': 2}]
    params_boundaries = "keep_xboundaries, keep_yboundaries"
    params_boundaries_values = [(False, False), (True, False), (False, True),
                                (True, True)]

    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    def test_region_core(self, tmpdir_factory, bout_xyt_example_files, guards,
                         keep_xboundaries, keep_yboundaries):
        # Note need to use more than (3*MXG,3*MYG) points per output file
        path = bout_xyt_example_files(tmpdir_factory, lengths=(3, 3, 4, 7), nxpe=3,
                                      nype=4, nt=1, guards=guards, grid='grid',
                                      topology='core')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_xboundaries=keep_xboundaries,
                              keep_yboundaries=keep_yboundaries)

        n = ds['n']

        if guards['y'] > 0 and not keep_yboundaries:
            # expect exception for core topology due to not having neighbour cells to get
            # coordinate values from
            with pytest.raises(ValueError):
                n_core = n.bout.fromRegion('core')
            return
        n_core = n.bout.fromRegion('core')

        # Remove attributes that are expected to be different
        del n_core.attrs['region']
        # Select only non-boundary data
        if keep_yboundaries:
            ybndry = guards['y']
        else:
            ybndry = 0
        xrt.assert_identical(n.isel(theta=slice(ybndry, -ybndry)),
                             n_core.isel(theta=slice(ybndry, -ybndry)))

    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    def test_region_sol(self, tmpdir_factory, bout_xyt_example_files, guards,
                        keep_xboundaries, keep_yboundaries):
        # Note need to use more than (3*MXG,3*MYG) points per output file
        path = bout_xyt_example_files(tmpdir_factory, lengths=(3, 3, 4, 7), nxpe=3,
                                      nype=4, nt=1, guards=guards, grid='grid',
                                      topology='sol')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_xboundaries=keep_xboundaries,
                              keep_yboundaries=keep_yboundaries)

        n = ds['n']

        print(n.regions)
        n_sol = n.bout.fromRegion('SOL')

        # Remove attributes that are expected to be different
        del n_sol.attrs['region']
        xrt.assert_identical(n, n_sol)
