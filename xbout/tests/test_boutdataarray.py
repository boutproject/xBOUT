import pytest

import numpy as np
import numpy.testing as npt
from pathlib import Path

import xarray as xr
import xarray.testing as xrt
from xarray.core.utils import dict_equiv

from xbout.tests.test_load import bout_xyt_example_files, create_bout_ds
from xbout import open_boutdataset
from xbout.geometries import apply_geometry


class TestBoutDataArrayMethods:

    def test_to_dataset(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None, keep_xboundaries=False)
        da = ds['n']

        new_ds = da.bout.to_dataset()

        assert dict_equiv(ds.attrs, new_ds.attrs)
        assert dict_equiv(ds.metadata, new_ds.metadata)

    def test_toFieldAligned(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None, keep_xboundaries=False)

        ds['psixy'] = ds['x']
        ds['Rxy'] = ds['x']
        ds['Zxy'] = ds['y']

        ds = apply_geometry(ds, 'toroidal')

        # set up test variable
        n = ds['n'].load()
        zShift = ds['zShift'].load()
        for t in range(ds.sizes['t']):
            for x in range(ds.sizes['x']):
                for y in range(ds.sizes['theta']):
                    zShift[x, y] = (x*ds.sizes['theta'] + y) * 2.*np.pi/ds.sizes['zeta']
                    for z in range(ds.sizes['zeta']):
                        n[t, x, y, z] = 1000.*t + 100.*x + 10.*y + z

        n.attrs['direction_y'] = 'Standard'
        n_al = n.bout.toFieldAligned()
        for t in range(ds.sizes['t']):
            npt.assert_allclose(n_al[t, 0, 0, 0].values, 1000.*t + 0., rtol=1.e-15, atol=5.e-16)                # noqa: E501
            npt.assert_allclose(n_al[t, 0, 0, 1].values, 1000.*t + 1., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_al[t, 0, 0, 2].values, 1000.*t + 2., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_al[t, 0, 0, 3].values, 1000.*t + 3., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_al[t, 0, 0, 4].values, 1000.*t + 4., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_al[t, 0, 0, 5].values, 1000.*t + 5., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_al[t, 0, 0, 6].values, 1000.*t + 6., rtol=1.e-15, atol=0.)                    # noqa: E501

            npt.assert_allclose(n_al[t, 0, 1, 0].values, 1000.*t + 10.*1. + 1., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 1, 1].values, 1000.*t + 10.*1. + 2., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 1, 2].values, 1000.*t + 10.*1. + 3., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 1, 3].values, 1000.*t + 10.*1. + 4., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 1, 4].values, 1000.*t + 10.*1. + 5., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 1, 5].values, 1000.*t + 10.*1. + 6., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 1, 6].values, 1000.*t + 10.*1. + 0., rtol=1.e-15, atol=0.)           # noqa: E501

            npt.assert_allclose(n_al[t, 0, 2, 0].values, 1000.*t + 10.*2. + 2., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 2, 1].values, 1000.*t + 10.*2. + 3., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 2, 2].values, 1000.*t + 10.*2. + 4., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 2, 3].values, 1000.*t + 10.*2. + 5., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 2, 4].values, 1000.*t + 10.*2. + 6., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 2, 5].values, 1000.*t + 10.*2. + 0., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 2, 6].values, 1000.*t + 10.*2. + 1., rtol=1.e-15, atol=0.)           # noqa: E501

            npt.assert_allclose(n_al[t, 0, 3, 0].values, 1000.*t + 10.*3. + 3., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 3, 1].values, 1000.*t + 10.*3. + 4., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 3, 2].values, 1000.*t + 10.*3. + 5., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 3, 3].values, 1000.*t + 10.*3. + 6., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 3, 4].values, 1000.*t + 10.*3. + 0., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 3, 5].values, 1000.*t + 10.*3. + 1., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_al[t, 0, 3, 6].values, 1000.*t + 10.*3. + 2., rtol=1.e-15, atol=0.)           # noqa: E501

            npt.assert_allclose(n_al[t, 1, 0, 0].values, 1000.*t + 100.*1 + 10.*0. + 4., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 0, 1].values, 1000.*t + 100.*1 + 10.*0. + 5., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 0, 2].values, 1000.*t + 100.*1 + 10.*0. + 6., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 0, 3].values, 1000.*t + 100.*1 + 10.*0. + 0., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 0, 4].values, 1000.*t + 100.*1 + 10.*0. + 1., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 0, 5].values, 1000.*t + 100.*1 + 10.*0. + 2., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 0, 6].values, 1000.*t + 100.*1 + 10.*0. + 3., rtol=1.e-15, atol=0.)  # noqa: E501

            npt.assert_allclose(n_al[t, 1, 1, 0].values, 1000.*t + 100.*1 + 10.*1. + 5., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 1, 1].values, 1000.*t + 100.*1 + 10.*1. + 6., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 1, 2].values, 1000.*t + 100.*1 + 10.*1. + 0., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 1, 3].values, 1000.*t + 100.*1 + 10.*1. + 1., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 1, 4].values, 1000.*t + 100.*1 + 10.*1. + 2., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 1, 5].values, 1000.*t + 100.*1 + 10.*1. + 3., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 1, 6].values, 1000.*t + 100.*1 + 10.*1. + 4., rtol=1.e-15, atol=0.)  # noqa: E501

            npt.assert_allclose(n_al[t, 1, 2, 0].values, 1000.*t + 100.*1 + 10.*2. + 6., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 2, 1].values, 1000.*t + 100.*1 + 10.*2. + 0., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 2, 2].values, 1000.*t + 100.*1 + 10.*2. + 1., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 2, 3].values, 1000.*t + 100.*1 + 10.*2. + 2., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 2, 4].values, 1000.*t + 100.*1 + 10.*2. + 3., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 2, 5].values, 1000.*t + 100.*1 + 10.*2. + 4., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 2, 6].values, 1000.*t + 100.*1 + 10.*2. + 5., rtol=1.e-15, atol=0.)  # noqa: E501

            npt.assert_allclose(n_al[t, 1, 3, 0].values, 1000.*t + 100.*1 + 10.*3. + 0., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 3, 1].values, 1000.*t + 100.*1 + 10.*3. + 1., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 3, 2].values, 1000.*t + 100.*1 + 10.*3. + 2., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 3, 3].values, 1000.*t + 100.*1 + 10.*3. + 3., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 3, 4].values, 1000.*t + 100.*1 + 10.*3. + 4., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 3, 5].values, 1000.*t + 100.*1 + 10.*3. + 5., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_al[t, 1, 3, 6].values, 1000.*t + 100.*1 + 10.*3. + 6., rtol=1.e-15, atol=0.)  # noqa: E501

    def test_fromFieldAligned(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None, keep_xboundaries=False)

        ds['psixy'] = ds['x']
        ds['Rxy'] = ds['x']
        ds['Zxy'] = ds['y']

        ds = apply_geometry(ds, 'toroidal')

        # set up test variable
        n = ds['n'].load()
        zShift = ds['zShift'].load()
        for t in range(ds.sizes['t']):
            for x in range(ds.sizes['x']):
                for y in range(ds.sizes['theta']):
                    zShift[x, y] = (x*ds.sizes['theta'] + y) * 2.*np.pi/ds.sizes['zeta']
                    for z in range(ds.sizes['zeta']):
                        n[t, x, y, z] = 1000.*t + 100.*x + 10.*y + z

        n.attrs['direction_y'] = 'Aligned'
        n_nal = n.bout.fromFieldAligned()
        for t in range(ds.sizes['t']):
            npt.assert_allclose(n_nal[t, 0, 0, 0].values, 1000.*t + 0., rtol=1.e-15, atol=5.e-16)                # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 0, 1].values, 1000.*t + 1., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 0, 2].values, 1000.*t + 2., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 0, 3].values, 1000.*t + 3., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 0, 4].values, 1000.*t + 4., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 0, 5].values, 1000.*t + 5., rtol=1.e-15, atol=0.)                    # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 0, 6].values, 1000.*t + 6., rtol=1.e-15, atol=0.)                    # noqa: E501

            npt.assert_allclose(n_nal[t, 0, 1, 0].values, 1000.*t + 10.*1. + 6., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 1, 1].values, 1000.*t + 10.*1. + 0., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 1, 2].values, 1000.*t + 10.*1. + 1., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 1, 3].values, 1000.*t + 10.*1. + 2., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 1, 4].values, 1000.*t + 10.*1. + 3., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 1, 5].values, 1000.*t + 10.*1. + 4., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 1, 6].values, 1000.*t + 10.*1. + 5., rtol=1.e-15, atol=0.)           # noqa: E501

            npt.assert_allclose(n_nal[t, 0, 2, 0].values, 1000.*t + 10.*2. + 5., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 2, 1].values, 1000.*t + 10.*2. + 6., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 2, 2].values, 1000.*t + 10.*2. + 0., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 2, 3].values, 1000.*t + 10.*2. + 1., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 2, 4].values, 1000.*t + 10.*2. + 2., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 2, 5].values, 1000.*t + 10.*2. + 3., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 2, 6].values, 1000.*t + 10.*2. + 4., rtol=1.e-15, atol=0.)           # noqa: E501

            npt.assert_allclose(n_nal[t, 0, 3, 0].values, 1000.*t + 10.*3. + 4., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 3, 1].values, 1000.*t + 10.*3. + 5., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 3, 2].values, 1000.*t + 10.*3. + 6., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 3, 3].values, 1000.*t + 10.*3. + 0., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 3, 4].values, 1000.*t + 10.*3. + 1., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 3, 5].values, 1000.*t + 10.*3. + 2., rtol=1.e-15, atol=0.)           # noqa: E501
            npt.assert_allclose(n_nal[t, 0, 3, 6].values, 1000.*t + 10.*3. + 3., rtol=1.e-15, atol=0.)           # noqa: E501

            npt.assert_allclose(n_nal[t, 1, 0, 0].values, 1000.*t + 100.*1 + 10.*0. + 3., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 0, 1].values, 1000.*t + 100.*1 + 10.*0. + 4., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 0, 2].values, 1000.*t + 100.*1 + 10.*0. + 5., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 0, 3].values, 1000.*t + 100.*1 + 10.*0. + 6., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 0, 4].values, 1000.*t + 100.*1 + 10.*0. + 0., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 0, 5].values, 1000.*t + 100.*1 + 10.*0. + 1., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 0, 6].values, 1000.*t + 100.*1 + 10.*0. + 2., rtol=1.e-15, atol=0.)  # noqa: E501

            npt.assert_allclose(n_nal[t, 1, 1, 0].values, 1000.*t + 100.*1 + 10.*1. + 2., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 1, 1].values, 1000.*t + 100.*1 + 10.*1. + 3., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 1, 2].values, 1000.*t + 100.*1 + 10.*1. + 4., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 1, 3].values, 1000.*t + 100.*1 + 10.*1. + 5., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 1, 4].values, 1000.*t + 100.*1 + 10.*1. + 6., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 1, 5].values, 1000.*t + 100.*1 + 10.*1. + 0., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 1, 6].values, 1000.*t + 100.*1 + 10.*1. + 1., rtol=1.e-15, atol=0.)  # noqa: E501

            npt.assert_allclose(n_nal[t, 1, 2, 0].values, 1000.*t + 100.*1 + 10.*2. + 1., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 2, 1].values, 1000.*t + 100.*1 + 10.*2. + 2., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 2, 2].values, 1000.*t + 100.*1 + 10.*2. + 3., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 2, 3].values, 1000.*t + 100.*1 + 10.*2. + 4., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 2, 4].values, 1000.*t + 100.*1 + 10.*2. + 5., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 2, 5].values, 1000.*t + 100.*1 + 10.*2. + 6., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 2, 6].values, 1000.*t + 100.*1 + 10.*2. + 0., rtol=1.e-15, atol=0.)  # noqa: E501

            npt.assert_allclose(n_nal[t, 1, 3, 0].values, 1000.*t + 100.*1 + 10.*3. + 0., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 3, 1].values, 1000.*t + 100.*1 + 10.*3. + 1., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 3, 2].values, 1000.*t + 100.*1 + 10.*3. + 2., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 3, 3].values, 1000.*t + 100.*1 + 10.*3. + 3., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 3, 4].values, 1000.*t + 100.*1 + 10.*3. + 4., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 3, 5].values, 1000.*t + 100.*1 + 10.*3. + 5., rtol=1.e-15, atol=0.)  # noqa: E501
            npt.assert_allclose(n_nal[t, 1, 3, 6].values, 1000.*t + 100.*1 + 10.*3. + 6., rtol=1.e-15, atol=0.)  # noqa: E501

    @pytest.mark.long
    def test_highParallelResRegion_core(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 3, 16, 3), nxpe=1,
                                      nype=1, nt=1, grid='grid', guards={'y':2},
                                      topology='core')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_yboundaries=True)

        n = ds['n']

        thetalength = 2.*np.pi

        dtheta = thetalength/16.
        theta = xr.DataArray(np.linspace(0. - 1.5*dtheta, thetalength + 1.5*dtheta, 20),
                             dims='theta')

        dtheta_fine = thetalength/128.
        theta_fine = xr.DataArray(
                np.linspace(0. + dtheta_fine/2., thetalength - dtheta_fine/2., 128),
                dims='theta')

        def f(t):
            t = np.sin(t)
            return (t**3 - t**2 + t - 1.)

        n.data = f(theta).broadcast_like(n)

        n_highres = n.bout.highParallelResRegion('core')

        expected = f(theta_fine).broadcast_like(n_highres)

        npt.assert_allclose(n_highres.values, expected.values, rtol=0., atol=1.e-2)

    @pytest.mark.parametrize('res_factor', [pytest.param(2, marks=pytest.mark.long),
                                            3,
                                            pytest.param(7, marks=pytest.mark.long),
                                            pytest.param(18, marks=pytest.mark.long)])
    def test_highParallelResRegion_core_change_n(self, tmpdir_factory,
                                                 bout_xyt_example_files, res_factor):
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 3, 16, 3), nxpe=1,
                                      nype=1, nt=1, grid='grid', guards={'y':2},
                                      topology='core')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_yboundaries=True)

        n = ds['n']

        thetalength = 2.*np.pi

        dtheta = thetalength/16.
        theta = xr.DataArray(np.linspace(0. - 1.5*dtheta, thetalength + 1.5*dtheta, 20),
                             dims='theta')

        dtheta_fine = thetalength/res_factor/16.
        theta_fine = xr.DataArray(
                np.linspace(0. + dtheta_fine/2., thetalength - dtheta_fine/2.,
                            res_factor*16),
                dims='theta')

        def f(t):
            t = np.sin(t)
            return (t**3 - t**2 + t - 1.)

        n.data = f(theta).broadcast_like(n)

        n_highres = n.bout.highParallelResRegion('core', n=res_factor)

        expected = f(theta_fine).broadcast_like(n_highres)

        npt.assert_allclose(n_highres.values, expected.values, rtol=0., atol=1.e-2)

    @pytest.mark.long
    def test_highParallelResRegion_sol(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 3, 16, 3), nxpe=1,
                                      nype=1, nt=1, grid='grid', guards={'y':2},
                                      topology='sol')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_yboundaries=True)

        n = ds['n']

        thetalength = 2.*np.pi

        dtheta = thetalength/16.
        theta = xr.DataArray(np.linspace(0. - 1.5*dtheta, thetalength + 1.5*dtheta, 20),
                             dims='theta')

        dtheta_fine = thetalength/128.
        theta_fine = xr.DataArray(
                np.linspace(0. - 1.5*dtheta_fine, thetalength + 1.5*dtheta_fine, 132),
                dims='theta')

        def f(t):
            t = np.sin(t)
            return (t**3 - t**2 + t - 1.)

        n.data = f(theta).broadcast_like(n)

        n_highres = n.bout.highParallelResRegion('SOL')

        expected = f(theta_fine).broadcast_like(n_highres)

        npt.assert_allclose(n_highres.values, expected.values, rtol=0., atol=1.e-2)

    def test_highParallelResRegion_singlenull(self, tmpdir_factory,
                                              bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 3, 16, 3), nxpe=1,
                                      nype=3, nt=1, grid='grid', guards={'y':2},
                                      topology='single-null')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_yboundaries=True)

        n = ds['n']

        thetalength = 2.*np.pi

        dtheta = thetalength/48.
        theta = xr.DataArray(np.linspace(0. - 1.5*dtheta, thetalength + 1.5*dtheta, 52),
                             dims='theta')

        dtheta_fine = thetalength/3./128.
        theta_fine = xr.DataArray(
                np.linspace(0. + 0.5*dtheta_fine, thetalength - 0.5*dtheta_fine, 3*128),
                dims='theta')

        def f(t):
            t = np.sin(3.*t)
            return (t**3 - t**2 + t - 1.)

        n.data = f(theta).broadcast_like(n)

        f_fine = f(theta_fine)[:128]

        for region in ['inner_PFR', 'inner_SOL']:
            n_highres = n.bout.highParallelResRegion(region).isel(theta=slice(2, None))

            expected = f_fine.broadcast_like(n_highres)

            npt.assert_allclose(n_highres.values, expected.values, rtol=0., atol=1.e-2)

        for region in ['core', 'SOL']:
            n_highres = n.bout.highParallelResRegion(region)

            expected = f_fine.broadcast_like(n_highres)

            npt.assert_allclose(n_highres.values, expected.values, rtol=0., atol=1.e-2)

        for region in ['outer_PFR', 'outer_SOL']:
            n_highres = n.bout.highParallelResRegion(region).isel(theta=slice( -2))

            expected = f_fine.broadcast_like(n_highres)

            npt.assert_allclose(n_highres.values, expected.values, rtol=0., atol=1.e-2)

    def test_highParallelRes(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 3, 16, 3), nxpe=1,
                                      nype=3, nt=1, grid='grid', guards={'y':2},
                                      topology='single-null')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_yboundaries=True)

        n = ds['n']

        thetalength = 2.*np.pi

        dtheta = thetalength/48.
        theta = xr.DataArray(np.linspace(0. - 1.5*dtheta, thetalength + 1.5*dtheta, 52),
                             dims='theta')

        dtheta_fine = thetalength/3./128.
        theta_fine = xr.DataArray(
                np.linspace(0. + 0.5*dtheta_fine, thetalength - 0.5*dtheta_fine, 3*128),
                dims='theta')
        x = xr.DataArray(np.arange(3), dims='x')

        def f_y(t):
            t = np.sin(3.*t)
            return (t**3 - t**2 + t - 1.)

        f = f_y(theta) * (x + 1.)

        n.data = f.broadcast_like(n)

        f_fine = f_y(theta_fine)*(x + 1.)

        n_highres_ds = n.bout.highParallelRes().isel(theta=slice(2, -2))

        expected = f_fine.broadcast_like(n_highres_ds['n'])

        npt.assert_allclose(n_highres_ds['n'].values, expected.values,
                            rtol=0., atol=1.1e-2)

    def test_highParallelRes_toroidal_points(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 3, 16, 3), nxpe=1,
                                      nype=3, nt=1, grid='grid', guards={'y':2},
                                      topology='single-null')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_yboundaries=True)

        n_highres_ds = ds['n'].bout.highParallelRes()

        n_highres_ds_truncated = ds['n'].bout.highParallelRes(toroidal_points=2)

        xrt.assert_identical(n_highres_ds_truncated, n_highres_ds.isel(zeta=[0, 2]))

    def test_highParallelRes_toroidal_points_list(self, tmpdir_factory,
                                                  bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, lengths=(2, 3, 16, 3), nxpe=1,
                                      nype=3, nt=1, grid='grid', guards={'y':2},
                                      topology='single-null')

        ds = open_boutdataset(datapath=path,
                              gridfilepath=Path(path).parent.joinpath('grid.nc'),
                              geometry='toroidal', keep_yboundaries=True)

        n_highres_ds = ds['n'].bout.highParallelRes()

        points_list = [1, 2]

        n_highres_ds_truncated = ds['n'].bout.highParallelRes(
                toroidal_points=points_list)

        xrt.assert_identical(n_highres_ds_truncated, n_highres_ds.isel(zeta=points_list))
