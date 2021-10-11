import pytest
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

from xbout import open_boutdataset
from xbout.boutdataarray import BoutDataArrayAccessor
from .test_load import create_bout_ds_list

from animatplot.blocks import Pcolormesh, Line


@pytest.fixture
def create_test_file(tmp_path_factory):

    # Create temp dir for output of animate1D/2D
    save_dir = tmp_path_factory.mktemp("test_data")

    # Generate some test data
    ds_list, file_list = create_bout_ds_list(
        "BOUT.dmp", nxpe=3, nype=3, syn_data_type="linear"
    )
    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(save_dir.joinpath(file_name))

    with pytest.warns(UserWarning):
        ds = open_boutdataset(save_dir.joinpath("BOUT.dmp.*.nc"))  # Open test data

    return save_dir, ds


class TestAnimate:
    """
    Set of tests to check whether animate1D() and animate2D() are running properly
    and PillowWriter is saving each animation correctly
    """

    def test_animate2D(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds["n"].isel(x=1).bout.animate2D(save_as="%s/testyz" % save_dir)

        assert len(animation.blocks) == 1
        block = animation.blocks[0]
        assert isinstance(block, Pcolormesh)

        assert block.ax.get_xlabel() == "y"
        assert block.ax.get_ylabel() == "z"

        plt.close()

        animation = ds["n"].isel(y=2).bout.animate2D(save_as="%s/testxz" % save_dir)

        assert len(animation.blocks) == 1
        block = animation.blocks[0]
        assert isinstance(block, Pcolormesh)
        assert block.ax.get_xlabel() == "x"
        assert block.ax.get_ylabel() == "z"

        plt.close()

        animation = ds["n"].isel(z=3).bout.animate2D(save_as="%s/testxy" % save_dir)

        assert len(animation.blocks) == 1
        block = animation.blocks[0]
        assert isinstance(block, Pcolormesh)
        assert block.ax.get_xlabel() == "x"
        assert block.ax.get_ylabel() == "y"

        plt.close()

    @pytest.mark.parametrize(
        "controls",
        [
            ("both", False),
            ("timeline", False),
            ("toggle", False),
            ("", False),
            (None, False),
            ("foo", True),
        ],
    )
    def test_animate2D_controls_arg(self, create_test_file, controls):
        controls, expect_error = controls

        save_dir, ds = create_test_file

        if expect_error:
            with pytest.raises(ValueError, match="Unrecognised value for controls"):
                animation = ds["n"].isel(x=1).bout.animate2D(controls=controls)
        else:
            animation = ds["n"].isel(x=1).bout.animate2D(controls=controls)

            assert len(animation.blocks) == 1
            assert isinstance(animation.blocks[0], Pcolormesh)
            if controls in ["both", "timeline"]:
                assert hasattr(animation, "slider")
            else:
                assert not hasattr(animation, "slider")
            if controls in ["both", "toggle"]:
                assert hasattr(animation, "button")
            else:
                assert not hasattr(animation, "button")

            plt.close()

    def test_animate1D(self, create_test_file):

        save_dir, ds = create_test_file
        animation = ds["n"].isel(y=2, z=0).bout.animate1D(save_as="%s/test" % save_dir)

        assert len(animation.blocks) == 1
        assert isinstance(animation.blocks[0], Line)

        plt.close()

    @pytest.mark.parametrize(
        "controls",
        [
            ("both", False),
            ("timeline", False),
            ("toggle", False),
            ("", False),
            (None, False),
            ("foo", True),
        ],
    )
    def test_animate1D_controls_arg(self, create_test_file, controls):
        controls, expect_error = controls

        save_dir, ds = create_test_file

        if expect_error:
            with pytest.raises(ValueError, match="Unrecognised value for controls"):
                animation = ds["n"].isel(y=2, z=0).bout.animate1D(controls=controls)
        else:
            animation = ds["n"].isel(y=2, z=0).bout.animate1D(controls=controls)

            assert len(animation.blocks) == 1
            assert isinstance(animation.blocks[0], Line)
            if controls in ["both", "timeline"]:
                assert hasattr(animation, "slider")
            else:
                assert not hasattr(animation, "slider")
            if controls in ["both", "toggle"]:
                assert hasattr(animation, "button")
            else:
                assert not hasattr(animation, "button")

            plt.close()

    def test_animate_list(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)]
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_1d_default(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(y=2, z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)]
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Line)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_1d_multiline(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(y=2, z=3).bout.animate_list(
            [["n", "T"], ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)]
        )

        assert len(animation.blocks) == 4
        assert isinstance(animation.blocks[0], Line)
        assert isinstance(animation.blocks[1], Line)
        assert isinstance(animation.blocks[2], Pcolormesh)
        assert isinstance(animation.blocks[3], Line)

        # check there were actually 3 subplots
        assert (
            len(
                [
                    x
                    for x in plt.gcf().get_axes()
                    if isinstance(x, matplotlib.axes.Subplot)
                ]
            )
            == 3
        )

        plt.close()

    def test_animate_list_animate_over(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(t=2), ds["n"].isel(y=1, z=2)], animate_over="x"
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_save_as(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)],
            save_as="%s/test" % save_dir,
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_fps(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], fps=42
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)
        assert animation.timeline.fps == 42

        plt.close()

    def test_animate_list_nrows(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], nrows=2
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_ncols(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], ncols=3
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_not_enough_nrowsncols(self, create_test_file):

        save_dir, ds = create_test_file

        with pytest.raises(ValueError):
            animation = ds.isel(z=3).bout.animate_list(
                ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], nrows=2, ncols=1
            )

    @pytest.mark.skip(reason="test data for plot_poloidal needs more work")
    def test_animate_list_poloidal_plot(self, create_test_file):

        save_dir, ds = create_test_file

        metadata = ds.metadata
        metadata["ixseps1"] = 2
        metadata["ixseps2"] = 4
        metadata["jyseps1_1"] = 1
        metadata["jyseps2_1"] = 2
        metadata["ny_inner"] = 3
        metadata["jyseps1_2"] = 4
        metadata["jyseps2_2"] = 5
        from ..geometries import apply_geometry
        from ..utils import _set_attrs_on_all_vars

        ds = _set_attrs_on_all_vars(ds, "metadata", metadata)

        nx = ds.metadata["nx"]
        ny = ds.metadata["ny"]
        R = xr.DataArray(
            np.ones([nx, ny]) * np.linspace(0, 1, nx)[:, np.newaxis], dims=["x", "y"]
        )
        Z = xr.DataArray(
            np.ones([nx, ny]) * np.linspace(0, 1, ny)[np.newaxis, :], dims=["x", "y"]
        )
        ds["psixy"] = R
        ds["Rxy"] = R
        ds["Zxy"] = Z

        ds = apply_geometry(ds, "toroidal")

        animation = ds.isel(zeta=3).bout.animate_list(
            ["n", ds["T"].isel(zeta=3), ds["n"].isel(theta=1, zeta=2)],
            poloidal_plot=True,
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_subplots_adjust(self, create_test_file):

        save_dir, ds = create_test_file

        with pytest.warns(UserWarning):
            animation = ds.isel(z=3).bout.animate_list(
                ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)],
                subplots_adjust={"hspace": 4, "wspace": 5},
            )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_vmin(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], vmin=-0.1
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_vmin_list(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], vmin=[0.0, 0.1, 0.2]
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_vmax(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], vmax=1.1
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_vmax_list(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], vmax=[1.0, 1.1, 1.2]
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_logscale(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], logscale=True
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_logscale_float(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], logscale=1.0e-2
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_logscale_list(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)],
            logscale=[True, 1.0e-2, False],
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert isinstance(animation.blocks[2], Line)

        plt.close()

    def test_animate_list_titles_list(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds.isel(z=3).bout.animate_list(
            ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], titles=["a", None, "b"]
        )

        assert len(animation.blocks) == 3
        assert isinstance(animation.blocks[0], Pcolormesh)
        assert animation.blocks[0].ax.title.get_text() == "a"
        assert isinstance(animation.blocks[1], Pcolormesh)
        assert animation.blocks[1].ax.title.get_text() == "T"
        assert isinstance(animation.blocks[2], Line)
        assert animation.blocks[2].ax.title.get_text() == "b"

        plt.close()

    @pytest.mark.parametrize(
        "controls",
        [
            ("both", False),
            ("timeline", False),
            ("toggle", False),
            ("", False),
            (None, False),
            ("foo", True),
        ],
    )
    def test_animate_list_controls_arg(self, create_test_file, controls):
        controls, expect_error = controls

        save_dir, ds = create_test_file

        if expect_error:
            with pytest.raises(ValueError, match="Unrecognised value for controls"):
                animation = ds.isel(z=3).bout.animate_list(
                    ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], controls=controls
                )
        else:
            animation = ds.isel(z=3).bout.animate_list(
                ["n", ds["T"].isel(x=2), ds["n"].isel(y=1, z=2)], controls=controls
            )

            assert len(animation.blocks) == 3
            assert isinstance(animation.blocks[0], Pcolormesh)
            assert isinstance(animation.blocks[1], Pcolormesh)
            assert isinstance(animation.blocks[2], Line)
            if controls in ["both", "timeline"]:
                assert hasattr(animation, "slider")
            else:
                assert not hasattr(animation, "slider")
            if controls in ["both", "toggle"]:
                assert hasattr(animation, "button")
            else:
                assert not hasattr(animation, "button")

            plt.close()
