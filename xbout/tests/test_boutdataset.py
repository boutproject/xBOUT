import pytest

import numpy.testing as npt
from xarray import Dataset, DataArray, concat, open_dataset, open_mfdataset
import xarray.testing as xrt
import numpy as np
from pathlib import Path

from xbout.tests.test_load import bout_xyt_example_files, create_bout_ds
from xbout.tests.test_region import (params_guards, params_guards_values,
                                     params_boundaries, params_boundaries_values)
from xbout import BoutDatasetAccessor, open_boutdataset, reload_boutdataset
from xbout.geometries import apply_geometry
from xbout.utils import _set_attrs_on_all_vars


EXAMPLE_OPTIONS_FILE_PATH = './xbout/tests/data/options/BOUT.inp'


class TestBoutDatasetIsXarrayDataset:
    """
    Set of tests to check that BoutDatasets behave similarly to xarray Datasets.
    (With the accessor approach these should pass trivially now.)
    """

    def test_concat(self, tmpdir_factory, bout_xyt_example_files):
        path1 = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        bd1 = open_boutdataset(datapath=path1, inputfilepath=None,
                               keep_xboundaries=False)
        path2 = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        bd2 = open_boutdataset(datapath=path2, inputfilepath=None,
                               keep_xboundaries=False)
        result = concat([bd1, bd2], dim='run')
        assert result.dims == {**bd1.dims, 'run': 2}

    def test_isel(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        bd = open_boutdataset(datapath=path, inputfilepath=None,
                              keep_xboundaries=False)
        actual = bd.isel(x=slice(None,None,2))
        expected = bd.bout.data.isel(x=slice(None,None,2))
        xrt.assert_equal(actual, expected)


class TestBoutDatasetMethods:
    @pytest.mark.skip
    def test_test_method(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None)
        #ds = collect(path=path)
        #bd = BoutAccessor(ds)
        print(ds)
        #ds.bout.test_method()
        #print(ds.bout.options)
        #print(ds.bout.metadata)
        print(ds.isel(t=-1))

        #ds.bout.set_extra_data('stored')
        ds.bout.extra_data = 'stored'

        print(ds.bout.extra_data)

    def test_get_field_aligned(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None, keep_xboundaries=False)

        ds['psixy'] = ds['x']
        ds['Rxy'] = ds['x']
        ds['Zxy'] = ds['y']

        ds = apply_geometry(ds, 'toroidal')

        n = ds['n']
        n.attrs['direction_y'] = 'Standard'
        n_aligned_from_array = n.bout.to_field_aligned()

        # check n_aligned does not exist yet
        assert 'n_aligned' not in ds

        n_aligned_from_ds = ds.bout.get_field_aligned('n')
        xrt.assert_allclose(n_aligned_from_ds, n_aligned_from_array)
        xrt.assert_allclose(ds['n_aligned'], n_aligned_from_array)

        # check getting the cached version
        ds['n_aligned'] = ds['T']
        ds['n_aligned'].attrs['direction_y'] = 'Aligned'
        xrt.assert_allclose(ds.bout.get_field_aligned('n'), ds['T'])

    @pytest.mark.parametrize("mxg", [0, pytest.param(2, marks=pytest.mark.long)])
    @pytest.mark.parametrize("myg", [pytest.param(0, marks=pytest.mark.long), 2])
    def test_remove_yboundaries(self, tmpdir_factory, bout_xyt_example_files, mxg, myg):
        path = bout_xyt_example_files(
            tmpdir_factory, lengths=(2, 3, 4, 3), nxpe=1, nype=6, nt=1, grid='grid',
            guards={'x': mxg, 'y': myg}, topology='connected-double-null',
            syn_data_type='linear'
        )

        ds = open_boutdataset(
            datapath=path, gridfilepath=Path(path).parent.joinpath('grid.nc'),
            geometry='toroidal', keep_yboundaries=True
        )

        path_no_yboundaries = bout_xyt_example_files(
            tmpdir_factory, lengths=(2, 3, 4, 3), nxpe=1, nype=6, nt=1, grid='grid',
            guards={'x': mxg, 'y': 0}, topology='connected-double-null',
            syn_data_type='linear'
        )

        ds_no_yboundaries = open_boutdataset(
            datapath=path_no_yboundaries,
            gridfilepath=Path(path).parent.joinpath('grid.nc'), geometry='toroidal',
            keep_yboundaries=False
        )

        ds = ds.bout.remove_yboundaries()

        assert ds.metadata['keep_yboundaries'] == 0
        for v in ds:
            assert ds[v].metadata['keep_yboundaries'] == 0

        # expect theta coordinate to be different, so ignore
        ds = ds.drop('theta')
        ds_no_yboundaries = ds_no_yboundaries.drop('theta')

        # expect MYG can be different, so ignore
        del ds.metadata['MYG']
        del ds_no_yboundaries.metadata['MYG']
        for v in ds:
            del ds[v].metadata['MYG']
            # metadata on variables is a reference to ds_no_yboundaries.metadata, so
            # don't need to delete MYG again

        xrt.assert_equal(ds, ds_no_yboundaries)

    def test_set_parallel_interpolation_factor(self):
        ds = Dataset()
        ds['a'] = DataArray()
        ds = _set_attrs_on_all_vars(ds, 'metadata', {})

        with pytest.raises(KeyError):
            ds.metadata['fine_interpolation_factor']
        with pytest.raises(KeyError):
            ds['a'].metadata['fine_interpolation_factor']

        ds.bout.fine_interpolation_factor = 42

        assert ds.metadata['fine_interpolation_factor'] == 42
        assert ds['a'].metadata['fine_interpolation_factor'] == 42

    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    @pytest.mark.parametrize(
        "vars_to_interpolate", [('n', 'T'), pytest.param(..., marks=pytest.mark.long)]
    )
    def test_interpolate_parallel(self, tmpdir_factory, bout_xyt_example_files,
                                  guards, keep_xboundaries, keep_yboundaries,
                                  vars_to_interpolate):
        # This test checks that the regions created in the new high-resolution Dataset by
        # interpolate_parallel are correct.
        # This test does not test the accuracy of the parallel interpolation (there are
        # other tests for that).

        # Note using more than MXG x-direction points and MYG y-direction points per
        # output file ensures tests for whether boundary cells are present do not fail
        # when using minimal numbers of processors
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 3, 4, 3), nxpe=3,
                                      nype=6, nt=1, guards=guards, grid='grid',
                                      topology='disconnected-double-null')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_xboundaries=keep_xboundaries,
                              keep_yboundaries=keep_yboundaries)

        # Get high parallel resolution version of ds, and check that
        ds = ds.bout.interpolate_parallel(vars_to_interpolate)

        mxg = guards['x']
        myg = guards['y']

        if keep_xboundaries:
            ixs1 = ds.metadata['ixseps1']
        else:
            ixs1 = ds.metadata['ixseps1'] - guards['x']

        if keep_xboundaries:
            ixs2 = ds.metadata['ixseps2']
        else:
            ixs2 = ds.metadata['ixseps2'] - guards['x']

        if keep_yboundaries:
            ybndry = guards['y']
        else:
            ybndry = 0
        jys11 = ds.metadata['jyseps1_1'] + ybndry
        jys21 = ds.metadata['jyseps2_1'] + ybndry
        ny_inner = ds.metadata['ny_inner'] + 2*ybndry
        jys12 = ds.metadata['jyseps1_2'] + 3*ybndry
        jys22 = ds.metadata['jyseps2_2'] + 3*ybndry
        ny = ds.metadata['ny'] + 4*ybndry

        for var in ['n', 'T']:
            v = ds[var]
            v_noregions = v.copy(deep=True)

            # Remove attributes that are expected to be different
            del v_noregions.attrs['regions']

            v_lower_inner_PFR = v.bout.from_region('lower_inner_PFR')

            # Remove attributes that are expected to be different
            del v_lower_inner_PFR.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 + mxg),
                                                  theta=slice(jys11 + 1)),
                                 v_lower_inner_PFR.isel(
                                     theta=slice(-myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(v_noregions.isel(x=slice(ixs1 + mxg),
                                        theta=slice(jys22 + 1, jys22 + 1 + myg)).values,
                                 v_lower_inner_PFR.isel(theta=slice(-myg, None)).values)

            v_lower_inner_intersep = v.bout.from_region('lower_inner_intersep')

            # Remove attributes that are expected to be different
            del v_lower_inner_intersep.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                                  theta=slice(jys11 + 1)),
                                 v_lower_inner_intersep.isel(
                                     theta=slice(-myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                         theta=slice(jys11 + 1, jys11 + 1 + myg)).values,
                        v_lower_inner_intersep.isel(theta=slice(-myg, None)).values)

            v_lower_inner_SOL = v.bout.from_region('lower_inner_SOL')

            # Remove attributes that are expected to be different
            del v_lower_inner_SOL.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs2 - mxg, None),
                                                  theta=slice(jys11 + 1)),
                                 v_lower_inner_SOL.isel(
                                     theta=slice(-myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs2 - mxg, None),
                                         theta=slice(jys11 + 1, jys11 + 1 + myg)).values,
                        v_lower_inner_SOL.isel(theta=slice(-myg, None)).values)

            v_inner_core = v.bout.from_region('inner_core')

            # Remove attributes that are expected to be different
            del v_inner_core.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 + mxg),
                                                  theta=slice(jys11 + 1, jys21 + 1)),
                                 v_inner_core.isel(
                                     theta=slice(myg, -myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 + mxg),
                                         theta=slice(jys22 + 1 - myg, jys22 + 1)).values,
                        v_inner_core.isel(theta=slice(myg)).values)
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 + mxg),
                                         theta=slice(jys12 + 1, jys12 + 1 + myg)).values,
                        v_inner_core.isel(theta=slice(-myg, None)).values)

            v_inner_intersep = v.bout.from_region('inner_intersep')

            # Remove attributes that are expected to be different
            del v_inner_intersep.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                                  theta=slice(jys11 + 1, jys21 + 1)),
                                 v_inner_intersep.isel(
                                     theta=slice(myg, -myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                         theta=slice(jys11 + 1 - myg, jys11 + 1)).values,
                        v_inner_intersep.isel(theta=slice(myg)).values)
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                         theta=slice(jys12 + 1, jys12 + 1 + myg)).values,
                        v_inner_intersep.isel(theta=slice(-myg, None)).values)

            v_inner_sol = v.bout.from_region('inner_SOL')

            # Remove attributes that are expected to be different
            del v_inner_sol.attrs['regions']
            xrt.assert_identical(
                    v_noregions.isel(x=slice(ixs2 - mxg, None),
                                     theta=slice(jys11 + 1, jys21 + 1)),
                    v_inner_sol.isel(theta=slice(myg, -myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs2 - mxg, None),
                                         theta=slice(jys11 + 1 - myg, jys11 + 1)).values,
                        v_inner_sol.isel(theta=slice(myg)).values)
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs2 - mxg, None),
                                         theta=slice(jys21 + 1, jys21 + 1 + myg)).values,
                        v_inner_sol.isel(theta=slice(-myg, None)).values)

            v_upper_inner_PFR = v.bout.from_region('upper_inner_PFR')

            # Remove attributes that are expected to be different
            del v_upper_inner_PFR.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 + mxg),
                                                  theta=slice(jys21 + 1, ny_inner)),
                                 v_upper_inner_PFR.isel(theta=slice(myg, None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 + mxg),
                                         theta=slice(jys12 + 1 - myg, jys12 + 1)).values,
                        v_upper_inner_PFR.isel(theta=slice(myg)).values)

            v_upper_inner_intersep = v.bout.from_region('upper_inner_intersep')

            # Remove attributes that are expected to be different
            del v_upper_inner_intersep.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                                  theta=slice(jys21 + 1, ny_inner)),
                                 v_upper_inner_intersep.isel(theta=slice(myg, None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                         theta=slice(jys12 + 1 - myg, jys12 + 1)).values,
                        v_upper_inner_intersep.isel(theta=slice(myg)).values)

            v_upper_inner_SOL = v.bout.from_region('upper_inner_SOL')

            # Remove attributes that are expected to be different
            del v_upper_inner_SOL.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs2 - mxg, None),
                                                  theta=slice(jys21 + 1, ny_inner)),
                                 v_upper_inner_SOL.isel(theta=slice(myg, None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs2 - mxg, None),
                                         theta=slice(jys21 + 1 - myg, jys21 + 1)).values,
                        v_upper_inner_SOL.isel(theta=slice(myg)).values)

            v_upper_outer_PFR = v.bout.from_region('upper_outer_PFR')

            # Remove attributes that are expected to be different
            del v_upper_outer_PFR.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 + mxg),
                                                  theta=slice(ny_inner, jys12 + 1)),
                                 v_upper_outer_PFR.isel(
                                     theta=slice(-myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 + mxg),
                                         theta=slice(jys21 + 1, jys21 + 1 + myg)).values,
                        v_upper_outer_PFR.isel(theta=slice(-myg, None)).values)

            v_upper_outer_intersep = v.bout.from_region('upper_outer_intersep')

            # Remove attributes that are expected to be different
            del v_upper_outer_intersep.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                                  theta=slice(ny_inner, jys12 + 1)),
                                 v_upper_outer_intersep.isel(
                                     theta=slice(-myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                         theta=slice(jys21 + 1, jys21 + 1 + myg)).values,
                        v_upper_outer_intersep.isel(theta=slice(-myg, None)).values)

            v_upper_outer_SOL = v.bout.from_region('upper_outer_SOL')

            # Remove attributes that are expected to be different
            del v_upper_outer_SOL.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs2 - mxg, None),
                                                  theta=slice(ny_inner, jys12 + 1)),
                                 v_upper_outer_SOL.isel(
                                     theta=slice(-myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs2 - mxg, None),
                                         theta=slice(jys12 + 1, jys12 + 1 + myg)).values,
                        v_upper_outer_SOL.isel(theta=slice(-myg, None)).values)

            v_outer_core = v.bout.from_region('outer_core')

            # Remove attributes that are expected to be different
            del v_outer_core.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 + mxg),
                                                  theta=slice(jys12 + 1, jys22 + 1)),
                                 v_outer_core.isel(
                                     theta=slice(myg, -myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 + mxg),
                                         theta=slice(jys21 + 1 - myg, jys21 + 1)).values,
                        v_outer_core.isel(theta=slice(myg)).values)
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 + mxg),
                                         theta=slice(jys11 + 1, jys11 + 1 + myg)).values,
                        v_outer_core.isel(theta=slice(-myg, None)).values)

            v_outer_intersep = v.bout.from_region('outer_intersep')

            # Remove attributes that are expected to be different
            del v_outer_intersep.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                                  theta=slice(jys12 + 1, jys22 + 1)),
                                 v_outer_intersep.isel(
                                     theta=slice(myg, -myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                         theta=slice(jys21 + 1 - myg, jys21 + 1)).values,
                        v_outer_intersep.isel(theta=slice(myg)).values)
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                         theta=slice(jys22 + 1, jys22 + 1 + myg)).values,
                        v_outer_intersep.isel(theta=slice(-myg, None)).values)

            v_outer_sol = v.bout.from_region('outer_SOL')

            # Remove attributes that are expected to be different
            del v_outer_sol.attrs['regions']
            xrt.assert_identical(
                    v_noregions.isel(x=slice(ixs2 - mxg, None),
                                     theta=slice(jys12 + 1, jys22 + 1)),
                    v_outer_sol.isel(theta=slice(myg, -myg if myg != 0 else None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs2 - mxg, None),
                                         theta=slice(jys12 + 1 - myg, jys12 + 1)).values,
                        v_outer_sol.isel(theta=slice(myg)).values)
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs2 - mxg, None),
                                         theta=slice(jys22 + 1, jys22 + 1 + myg)).values,
                        v_outer_sol.isel(theta=slice(-myg, None)).values)

            v_lower_outer_PFR = v.bout.from_region('lower_outer_PFR')

            # Remove attributes that are expected to be different
            del v_lower_outer_PFR.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 + mxg),
                                                  theta=slice(jys22 + 1, None)),
                                 v_lower_outer_PFR.isel(theta=slice(myg, None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 + mxg),
                                         theta=slice(jys11 + 1 - myg, jys11 + 1)).values,
                        v_lower_outer_PFR.isel(theta=slice(myg)).values)

            v_lower_outer_intersep = v.bout.from_region('lower_outer_intersep')

            # Remove attributes that are expected to be different
            del v_lower_outer_intersep.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                                  theta=slice(jys22 + 1, None)),
                                 v_lower_outer_intersep.isel(theta=slice(myg, None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs1 - mxg, ixs2 + mxg),
                                         theta=slice(jys22 + 1 - myg, jys22 + 1)).values,
                        v_lower_outer_intersep.isel(theta=slice(myg)).values)

            v_lower_outer_SOL = v.bout.from_region('lower_outer_SOL')

            # Remove attributes that are expected to be different
            del v_lower_outer_SOL.attrs['regions']
            xrt.assert_identical(v_noregions.isel(x=slice(ixs2 - mxg, None),
                                                  theta=slice(jys22 + 1, None)),
                                 v_lower_outer_SOL.isel(theta=slice(myg, None)))
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                        v_noregions.isel(x=slice(ixs2 - mxg, None),
                                         theta=slice(jys22 + 1 - myg, jys22 + 1)).values,
                        v_lower_outer_SOL.isel(theta=slice(myg)).values)

    def test_interpolate_parallel_all_variables_arg(self, tmpdir_factory,
                                                    bout_xyt_example_files):
        # Check that passing 'variables=...' to interpolate_parallel() does actually
        # interpolate all the variables
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 3, 4, 3), nxpe=1,
                                      nype=1, nt=1, grid='grid', topology='sol')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal')

        # Get high parallel resolution version of ds, and check that
        ds = ds.bout.interpolate_parallel(...)

        interpolated_variables = [v for v in ds]

        assert set(interpolated_variables) == set((
            'n', 'T', 'g11', 'g22', 'g33', 'g12', 'g13', 'g23', 'g_11', 'g_22', 'g_33',
            'g_12', 'g_13', 'g_23', 'G1', 'G2', 'G3', 'J', 'Bxy', 'dx', 'dy'
        ))


class TestLoadInputFile:
    @pytest.mark.skip
    def test_load_options(self):
        from boutdata.data import BoutOptionsFile, BoutOptions
        options = BoutOptionsFile(EXAMPLE_OPTIONS_FILE_PATH)
        assert isinstance(options, BoutOptions)
        # TODO Check it contains the same text

    @pytest.mark.skip
    def test_load_options_in_dataset(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=EXAMPLE_OPTIONS_FILE_PATH)
        assert isinstance(ds.options, BoutOptions)


@pytest.mark.skip(reason="Not yet implemented")
class TestLoadLogFile:
    pass

class TestSave:
    def test_save_all(self, tmpdir_factory, bout_xyt_example_files):
        # Create data
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=5, nt=1)

        # Load it as a boutdataset
        original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + 'temp_boutdata.nc'
        original.bout.save(savepath=savepath)

        # Load it again using bare xarray
        recovered = open_dataset(savepath)

        # Compare equal (not identical because attributes are changed when saving)
        xrt.assert_equal(original, recovered)

    @pytest.mark.parametrize("geometry", [None, "toroidal"])
    def test_reload_all(self, tmpdir_factory, bout_xyt_example_files, geometry):
        if geometry is not None:
            grid = "grid"
        else:
            grid = None

        # Create data
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=5, nt=1, grid=grid)

        if grid is not None:
            gridpath = str(Path(path).parent) + "/grid.nc"
        else:
            gridpath = None

        # Load it as a boutdataset
        original = open_boutdataset(
                       datapath=path,
                       inputfilepath=None,
                       geometry=geometry,
                       gridfilepath=gridpath,
                   )

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + 'temp_boutdata.nc'
        original.bout.save(savepath=savepath)

        # Load it again
        recovered = reload_boutdataset(savepath)

        xrt.assert_identical(original.load(), recovered.load())

    @pytest.mark.skip("saving and loading as float32 does not work")
    @pytest.mark.parametrize("save_dtype", [np.float64, np.float32])
    def test_save_dtype(self, tmpdir_factory, bout_xyt_example_files, save_dtype):

        # Create data
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)

        # Load it as a boutdataset
        original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + 'temp_boutdata.nc'
        original.bout.save(savepath=savepath, save_dtype=np.dtype(save_dtype))

        # Load it again using bare xarray
        recovered = open_dataset(savepath)

        assert recovered['n'].values.dtype == np.dtype(save_dtype)

    def test_save_separate_variables(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=1, nt=1)

        # Load it as a boutdataset
        original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + '/temp_boutdata.nc'
        original.bout.save(savepath=savepath, separate_vars=True)

        for var in ['n', 'T']:
            # Load it again using bare xarray
            savepath = str(Path(path).parent) + '/temp_boutdata_' + var + '.nc'
            recovered = open_dataset(savepath)

            # Compare equal (not identical because attributes are changed when saving)
            xrt.assert_equal(recovered[var], original[var])

    @pytest.mark.parametrize("geometry", [None, "toroidal"])
    def test_reload_separate_variables(
        self, tmpdir_factory, bout_xyt_example_files, geometry
    ):
        if geometry is not None:
            grid = "grid"
        else:
            grid = None

        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=1, nt=1, grid=grid)

        if grid is not None:
            gridpath = str(Path(path).parent) + "/grid.nc"
        else:
            gridpath = None

        # Load it as a boutdataset
        original = open_boutdataset(
                       datapath=path,
                       inputfilepath=None,
                       geometry=geometry,
                       gridfilepath=gridpath,
                   )

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + '/temp_boutdata.nc'
        original.bout.save(savepath=savepath, separate_vars=True)

        # Load it again
        savepath = str(Path(path).parent) + '/temp_boutdata_*.nc'
        recovered = reload_boutdataset(savepath, pre_squashed=True)

        # Compare
        xrt.assert_identical(recovered, original)


class TestSaveRestart:
    pass
