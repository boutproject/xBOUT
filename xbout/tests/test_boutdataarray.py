import pytest

import numpy as np
from numpy.testing import assert_allclose

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

    @pytest.mark.parametrize('nz', [6, 7, 8, 9])
    def test_toFieldAligned(self, tmpdir_factory, bout_xyt_example_files, nz):
        path = bout_xyt_example_files(tmpdir_factory, lengths=(3, 3, 4, nz), nxpe=1,
                                      nype=1, nt=1)
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
                    for z in range(nz):
                        n[t, x, y, z] = 1000.*t + 100.*x + 10.*y + z

        n.attrs['direction_y'] = 'Standard'
        n_al = n.bout.toFieldAligned()
        for t in range(ds.sizes['t']):
            for z in range(nz):
                assert_allclose(n_al[t, 0, 0, z].values, 1000.*t + z % nz, rtol=1.e-15, atol=5.e-16)                      # noqa: E501

            for z in range(nz):
                assert_allclose(n_al[t, 0, 1, z].values, 1000.*t + 10.*1. + (z + 1) % nz, rtol=1.e-15, atol=0.)           # noqa: E501

            for z in range(nz):
                assert_allclose(n_al[t, 0, 2, z].values, 1000.*t + 10.*2. + (z + 2) % nz, rtol=1.e-15, atol=0.)           # noqa: E501

            for z in range(nz):
                assert_allclose(n_al[t, 0, 3, z].values, 1000.*t + 10.*3. + (z + 3) % nz, rtol=1.e-15, atol=0.)           # noqa: E501

            for z in range(nz):
                assert_allclose(n_al[t, 1, 0, z].values, 1000.*t + 100.*1 + 10.*0. + (z + 4) % nz, rtol=1.e-15, atol=0.)  # noqa: E501

            for z in range(nz):
                assert_allclose(n_al[t, 1, 1, z].values, 1000.*t + 100.*1 + 10.*1. + (z + 5) % nz, rtol=1.e-15, atol=0.)  # noqa: E501

            for z in range(nz):
                assert_allclose(n_al[t, 1, 2, z].values, 1000.*t + 100.*1 + 10.*2. + (z + 6) % nz, rtol=1.e-15, atol=0.)  # noqa: E501

            for z in range(nz):
                assert_allclose(n_al[t, 1, 3, z].values, 1000.*t + 100.*1 + 10.*3. + (z + 7) % nz, rtol=1.e-15, atol=0.)  # noqa: E501

    @pytest.mark.parametrize('nz', [6, 7, 8, 9])
    def test_fromFieldAligned(self, tmpdir_factory, bout_xyt_example_files, nz):
        path = bout_xyt_example_files(tmpdir_factory, lengths=(3, 3, 4, nz), nxpe=1,
                                      nype=1, nt=1)
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
            for z in range(nz):
                assert_allclose(n_nal[t, 0, 0, z].values, 1000.*t + z % nz, rtol=1.e-15, atol=5.e-16)                      # noqa: E501

            for z in range(nz):
                assert_allclose(n_nal[t, 0, 1, z].values, 1000.*t + 10.*1. + (z - 1) % nz, rtol=1.e-15, atol=0.)           # noqa: E501

            for z in range(nz):
                assert_allclose(n_nal[t, 0, 2, z].values, 1000.*t + 10.*2. + (z - 2) % nz, rtol=1.e-15, atol=0.)           # noqa: E501

            for z in range(nz):
                assert_allclose(n_nal[t, 0, 3, z].values, 1000.*t + 10.*3. + (z - 3) % nz, rtol=1.e-15, atol=0.)           # noqa: E501

            for z in range(nz):
                assert_allclose(n_nal[t, 1, 0, z].values, 1000.*t + 100.*1 + 10.*0. + (z - 4) % nz, rtol=1.e-15, atol=0.)  # noqa: E501

            for z in range(nz):
                assert_allclose(n_nal[t, 1, 1, z].values, 1000.*t + 100.*1 + 10.*1. + (z - 5) % nz, rtol=1.e-15, atol=0.)  # noqa: E501

            for z in range(nz):
                assert_allclose(n_nal[t, 1, 2, z].values, 1000.*t + 100.*1 + 10.*2. + (z - 6) % nz, rtol=1.e-15, atol=0.)  # noqa: E501

            for z in range(nz):
                assert_allclose(n_nal[t, 1, 3, z].values, 1000.*t + 100.*1 + 10.*3. + (z - 7) % nz, rtol=1.e-15, atol=0.)  # noqa: E501
