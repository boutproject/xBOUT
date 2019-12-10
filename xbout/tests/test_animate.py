import pytest
import numpy as np
import xarray as xr

from xbout import open_boutdataset
from xbout.boutdataarray import BoutDataArrayAccessor
from .test_load import create_bout_ds_list

from animatplot.blocks import Pcolormesh, Line


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

        animation = ds['n'].isel(x=1).bout.animate2D(save_as="%s/testyz" % save_dir)

        assert isinstance(animation, Pcolormesh)

        assert animation.ax.get_xlabel() == 'y'
        assert animation.ax.get_ylabel() == 'z'

        animation = ds['n'].isel(y=2).bout.animate2D(save_as="%s/testxz" % save_dir)

        assert isinstance(animation, Pcolormesh)
        assert animation.ax.get_xlabel() == 'x'
        assert animation.ax.get_ylabel() == 'z'

        animation = ds['n'].isel(z=3).bout.animate2D(save_as="%s/testxy" % save_dir)

        assert isinstance(animation, Pcolormesh)
        assert animation.ax.get_xlabel() == 'x'
        assert animation.ax.get_ylabel() == 'y'

    def test_animate1D(self, create_test_file):

        save_dir, ds = create_test_file
        animation = ds['n'].isel(y=2, z=0).bout.animate1D(save_as="%s/test" % save_dir)

        assert isinstance(animation, Line)

    def test_animate_list(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)])

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_1d_default(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(y=2, z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                          ds['n'].isel(y=1, z=2)])

        assert isinstance(animation.blocks[0], Line)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_animate_over(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(t=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    animate_over='x')

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_save_as(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    save_as="%s/test" % save_dir)

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_fps(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    fps=42)

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)
        assert animation.timeline.fps == 42

    def test_animate_list_nrows(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    nrows=2)

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_ncols(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    ncols=3)

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_not_enough_nrowsncols(self, create_test_file):

        save_dir, ds = create_test_file

        with pytest.raises(ValueError):
            animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                         ds['n'].isel(y=1, z=2)],
                                                        nrows=2, ncols=1)

    @pytest.mark.skip(reason='test data for plot_poloidal needs more work')
    def test_animate_list_poloidal_plot(self, create_test_file):

        save_dir, ds = create_test_file

        metadata = ds.metadata
        metadata['ixseps1'] = 2
        metadata['ixseps2'] = 4
        metadata['jyseps1_1'] = 1
        metadata['jyseps2_1'] = 2
        metadata['ny_inner'] = 3
        metadata['jyseps1_2'] = 4
        metadata['jyseps2_2'] = 5
        from ..geometries import apply_geometry
        from ..utils import _set_attrs_on_all_vars
        ds = _set_attrs_on_all_vars(ds, 'metadata', metadata)

        nx = ds.metadata['nx']
        ny = ds.metadata['ny']
        R = xr.DataArray(np.ones([nx, ny])*np.linspace(0, 1, nx)[:, np.newaxis],
                         dims=['x', 'y'])
        Z = xr.DataArray(np.ones([nx, ny])*np.linspace(0, 1, ny)[np.newaxis, :],
                         dims=['x', 'y'])
        ds['psixy'] = R
        ds['Rxy'] = R
        ds['Zxy'] = Z

        ds = apply_geometry(ds, 'toroidal')

        animation = ds.isel(zeta=3).bout.animate_list(['n', ds['T'].isel(zeta=3),
                                                        ds['n'].isel(theta=1, zeta=2)],
                                                       poloidal_plot=True)

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_subplots_adjust(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    subplots_adjust={'hspace': 4,
                                                                     'wspace': 5})

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_vmin(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                     vmin=-0.1)

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_vmin_list(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    vmin=[0., 0.1, 0.2])

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_vmax(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                     vmax=1.1)

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_vmax_list(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    vmax=[1., 1.1, 1.2])

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_logscale(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    logscale=True)

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_logscale_float(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    logscale=1.e-2)

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

    def test_animate_list_logscale_list(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(['n', ds['T'].isel(x=2),
                                                     ds['n'].isel(y=1, z=2)],
                                                    logscale=[True, 1.e-2, False])

        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)
