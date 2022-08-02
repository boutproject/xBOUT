import pytest

import dask.array
import numpy as np
import numpy.testing as npt
from pathlib import Path

import xarray as xr
import xarray.testing as xrt
from xarray.core.utils import dict_equiv

from xbout.tests.test_load import bout_xyt_example_files, create_bout_ds
from xbout import open_boutdataset
from xbout.geometries import apply_geometry
from xbout.utils import _1d_coord_from_spacing


class TestBoutDataArrayMethods:
    def test_to_dataset(self, bout_xyt_example_files):
        dataset_list = bout_xyt_example_files(None, nxpe=3, nype=4, nt=1)
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )
        da = ds["n"]

        new_ds = da.bout.to_dataset()

        assert dict_equiv(ds.attrs, new_ds.attrs)
        assert dict_equiv(ds.metadata, new_ds.metadata)

    @pytest.mark.parametrize("mxg", [0, pytest.param(2, marks=pytest.mark.long)])
    @pytest.mark.parametrize("myg", [pytest.param(0, marks=pytest.mark.long), 2])
    @pytest.mark.parametrize(
        "remove_extra_upper", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_remove_yboundaries(
        self, bout_xyt_example_files, mxg, myg, remove_extra_upper
    ):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=1,
            nype=6,
            nt=1,
            grid="grid",
            guards={"x": mxg, "y": myg},
            topology="connected-double-null",
            syn_data_type="linear",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        dataset_list_no_yboundaries, grid_ds_no_yboundaries = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=1,
            nype=6,
            nt=1,
            grid="grid",
            guards={"x": mxg, "y": 0},
            topology="connected-double-null",
            syn_data_type="linear",
        )

        ds_no_yboundaries = open_boutdataset(
            datapath=dataset_list_no_yboundaries,
            gridfilepath=grid_ds_no_yboundaries,
            geometry="toroidal",
            keep_yboundaries=False,
        )

        if remove_extra_upper:
            ds_no_yboundaries = xr.concat(
                [
                    ds_no_yboundaries.isel(theta=slice(None, 11)),
                    ds_no_yboundaries.isel(theta=slice(12, -1)),
                ],
                dim="theta",
            )

        n = ds["n"].bout.remove_yboundaries(remove_extra_upper=remove_extra_upper)

        assert n.metadata["keep_yboundaries"] == 0
        npt.assert_equal(n.values, ds_no_yboundaries["n"].values)

    @pytest.mark.parametrize(
        "nz",
        [
            pytest.param(6, marks=pytest.mark.long),
            7,
            pytest.param(8, marks=pytest.mark.long),
            pytest.param(9, marks=pytest.mark.long),
        ],
    )
    @pytest.mark.parametrize(
        "permute_dims", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_to_field_aligned(self, bout_xyt_example_files, nz, permute_dims):
        dataset_list = bout_xyt_example_files(
            None, lengths=(3, 3, 4, nz), nxpe=1, nype=1, nt=1
        )
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )

        ds["psixy"] = ds["x"]
        ds["Rxy"] = ds["x"]
        ds["Zxy"] = ds["y"]

        ds = apply_geometry(ds, "toroidal")

        # set up test variable
        n = ds["n"].load()
        zShift = ds["zShift"].load()
        for t in range(ds.sizes["t"]):
            for x in range(ds.sizes["x"]):
                for y in range(ds.sizes["theta"]):
                    zShift[x, y] = (
                        (x * ds.sizes["theta"] + y) * 2.0 * np.pi / ds.sizes["zeta"]
                    )
                    for z in range(nz):
                        n[t, x, y, z] = 1000.0 * t + 100.0 * x + 10.0 * y + z

        n.attrs["direction_y"] = "Standard"

        if permute_dims:
            n = n.transpose("t", "zeta", "x", "theta").compute()

        n_al = n.bout.to_field_aligned()

        if permute_dims:
            n_al = n_al.transpose("t", "x", "theta", "zeta").compute()

        assert n_al.direction_y == "Aligned"

        for t in range(ds.sizes["t"]):
            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 0, 0, z].values,
                    1000.0 * t + z % nz,
                    rtol=1.0e-15,
                    atol=5.0e-16,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 0, 1, z].values,
                    1000.0 * t + 10.0 * 1.0 + (z + 1) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 0, 2, z].values,
                    1000.0 * t + 10.0 * 2.0 + (z + 2) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 0, 3, z].values,
                    1000.0 * t + 10.0 * 3.0 + (z + 3) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 1, 0, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 0.0 + (z + 4) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 1, 1, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 1.0 + (z + 5) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 1, 2, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 2.0 + (z + 6) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 1, 3, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 3.0 + (z + 7) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

    @pytest.mark.parametrize(
        "permute_dims", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_to_field_aligned_dask(self, bout_xyt_example_files, permute_dims):

        nz = 6

        dataset_list = bout_xyt_example_files(
            None, lengths=(3, 3, 4, nz), nxpe=1, nype=1, nt=1
        )
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )

        ds["psixy"] = ds["x"]
        ds["Rxy"] = ds["x"]
        ds["Zxy"] = ds["y"]

        ds = apply_geometry(ds, "toroidal")

        # set up test variable
        n = ds["n"].load()
        zShift = ds["zShift"].load()
        for t in range(ds.sizes["t"]):
            for x in range(ds.sizes["x"]):
                for y in range(ds.sizes["theta"]):
                    zShift[x, y] = (
                        (x * ds.sizes["theta"] + y) * 2.0 * np.pi / ds.sizes["zeta"]
                    )
                    for z in range(nz):
                        n[t, x, y, z] = 1000.0 * t + 100.0 * x + 10.0 * y + z

        # The above loop required the call to .load(), but that turned the data into a
        # numpy array. Now convert back to dask
        n = n.chunk({"t": 1})
        assert isinstance(n.data, dask.array.Array)

        n.attrs["direction_y"] = "Standard"

        if permute_dims:
            n = n.transpose("t", "zeta", "x", "theta").compute()

        n_al = n.bout.to_field_aligned()

        if permute_dims:
            n_al = n_al.transpose("t", "x", "theta", "zeta").compute()

        assert n_al.direction_y == "Aligned"

        for t in range(ds.sizes["t"]):
            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 0, 0, z].values,
                    1000.0 * t + z % nz,
                    rtol=1.0e-15,
                    atol=5.0e-16,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 0, 1, z].values,
                    1000.0 * t + 10.0 * 1.0 + (z + 1) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 0, 2, z].values,
                    1000.0 * t + 10.0 * 2.0 + (z + 2) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 0, 3, z].values,
                    1000.0 * t + 10.0 * 3.0 + (z + 3) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 1, 0, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 0.0 + (z + 4) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 1, 1, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 1.0 + (z + 5) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 1, 2, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 2.0 + (z + 6) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_al[t, 1, 3, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 3.0 + (z + 7) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

    @pytest.mark.parametrize(
        "nz",
        [
            pytest.param(6, marks=pytest.mark.long),
            7,
            pytest.param(8, marks=pytest.mark.long),
            pytest.param(9, marks=pytest.mark.long),
        ],
    )
    @pytest.mark.parametrize(
        "permute_dims", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_from_field_aligned(self, bout_xyt_example_files, nz, permute_dims):
        dataset_list = bout_xyt_example_files(
            None, lengths=(3, 3, 4, nz), nxpe=1, nype=1, nt=1
        )
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )

        ds["psixy"] = ds["x"]
        ds["Rxy"] = ds["x"]
        ds["Zxy"] = ds["y"]

        ds = apply_geometry(ds, "toroidal")

        # set up test variable
        n = ds["n"].load()
        zShift = ds["zShift"].load()
        for t in range(ds.sizes["t"]):
            for x in range(ds.sizes["x"]):
                for y in range(ds.sizes["theta"]):
                    zShift[x, y] = (
                        (x * ds.sizes["theta"] + y) * 2.0 * np.pi / ds.sizes["zeta"]
                    )
                    for z in range(ds.sizes["zeta"]):
                        n[t, x, y, z] = 1000.0 * t + 100.0 * x + 10.0 * y + z

        n.attrs["direction_y"] = "Aligned"

        if permute_dims:
            n = n.transpose("t", "zeta", "x", "theta").compute()

        n_nal = n.bout.from_field_aligned()

        if permute_dims:
            n_nal = n_nal.transpose("t", "x", "theta", "zeta").compute()

        assert n_nal.direction_y == "Standard"

        for t in range(ds.sizes["t"]):
            for z in range(nz):
                npt.assert_allclose(
                    n_nal[t, 0, 0, z].values,
                    1000.0 * t + z % nz,
                    rtol=1.0e-15,
                    atol=5.0e-16,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_nal[t, 0, 1, z].values,
                    1000.0 * t + 10.0 * 1.0 + (z - 1) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_nal[t, 0, 2, z].values,
                    1000.0 * t + 10.0 * 2.0 + (z - 2) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_nal[t, 0, 3, z].values,
                    1000.0 * t + 10.0 * 3.0 + (z - 3) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_nal[t, 1, 0, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 0.0 + (z - 4) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_nal[t, 1, 1, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 1.0 + (z - 5) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_nal[t, 1, 2, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 2.0 + (z - 6) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

            for z in range(nz):
                npt.assert_allclose(
                    n_nal[t, 1, 3, z].values,
                    1000.0 * t + 100.0 * 1 + 10.0 * 3.0 + (z - 7) % nz,
                    rtol=1.0e-15,
                    atol=0.0,
                )  # noqa: E501

    @pytest.mark.parametrize("stag_location", ["CELL_XLOW", "CELL_YLOW", "CELL_ZLOW"])
    @pytest.mark.parametrize(
        "permute_dims", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_to_field_aligned_staggered(
        self, bout_xyt_example_files, stag_location, permute_dims
    ):
        dataset_list = bout_xyt_example_files(
            None, lengths=(3, 3, 4, 8), nxpe=1, nype=1, nt=1
        )
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )

        ds["psixy"] = ds["x"]
        ds["Rxy"] = ds["x"]
        ds["Zxy"] = ds["y"]

        ds = apply_geometry(ds, "toroidal")

        # set up test variable
        n = ds["n"].load()
        zShift = ds["zShift"].load()
        for t in range(ds.sizes["t"]):
            for x in range(ds.sizes["x"]):
                for y in range(ds.sizes["theta"]):
                    zShift[x, y] = (
                        (x * ds.sizes["theta"] + y) * 2.0 * np.pi / ds.sizes["zeta"]
                    )
                    for z in range(ds.sizes["zeta"]):
                        n[t, x, y, z] = 1000.0 * t + 100.0 * x + 10.0 * y + z

        if permute_dims:
            n = n.transpose("t", "zeta", "x", "theta").compute()

        n_al = n.bout.to_field_aligned().copy(deep=True)

        if permute_dims:
            n_al = n_al.transpose("t", "x", "theta", "zeta").compute()

        assert n_al.direction_y == "Aligned"

        # make 'n' staggered
        ds["n"].attrs["cell_location"] = stag_location

        if stag_location != "CELL_ZLOW":
            with pytest.raises(ValueError):
                # Check exception raised when needed zShift_CELL_*LOW is not present
                ds["n"].bout.to_field_aligned()
            ds["zShift_" + stag_location] = zShift
            ds["zShift_" + stag_location].attrs["cell_location"] = stag_location
            ds = ds.set_coords("zShift_" + stag_location)
            ds = ds.drop_vars("zShift")

            with pytest.raises(ValueError):
                # Check shifting non-staggered field fails without zShift
                ds["T"].bout.to_field_aligned()

        n_stag_al = ds["n"].bout.to_field_aligned()

        npt.assert_equal(n_stag_al.values, n_al.values)

    @pytest.mark.parametrize("stag_location", ["CELL_XLOW", "CELL_YLOW", "CELL_ZLOW"])
    @pytest.mark.parametrize(
        "permute_dims", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_from_field_aligned_staggered(
        self, bout_xyt_example_files, stag_location, permute_dims
    ):
        dataset_list = bout_xyt_example_files(
            None, lengths=(3, 3, 4, 8), nxpe=1, nype=1, nt=1
        )
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )

        ds["psixy"] = ds["x"]
        ds["Rxy"] = ds["x"]
        ds["Zxy"] = ds["y"]

        ds = apply_geometry(ds, "toroidal")

        # set up test variable
        n = ds["n"].load()
        zShift = ds["zShift"].load()
        for t in range(ds.sizes["t"]):
            for x in range(ds.sizes["x"]):
                for y in range(ds.sizes["theta"]):
                    zShift[x, y] = (
                        (x * ds.sizes["theta"] + y) * 2.0 * np.pi / ds.sizes["zeta"]
                    )
                    for z in range(ds.sizes["zeta"]):
                        n[t, x, y, z] = 1000.0 * t + 100.0 * x + 10.0 * y + z
        n.attrs["direction_y"] = "Aligned"
        ds["T"].attrs["direction_y"] = "Aligned"

        if permute_dims:
            n = n.transpose("t", "zeta", "x", "theta").compute()

        n_nal = n.bout.from_field_aligned().copy(deep=True)

        if permute_dims:
            n_nal = n_nal.transpose("t", "x", "theta", "zeta").compute()

        assert n_nal.direction_y == "Standard"

        # make 'n' staggered
        ds["n"].attrs["cell_location"] = stag_location

        if stag_location != "CELL_ZLOW":
            with pytest.raises(ValueError):
                # Check exception raised when needed zShift_CELL_*LOW is not present
                ds["n"].bout.from_field_aligned()
            ds["zShift_" + stag_location] = zShift
            ds["zShift_" + stag_location].attrs["cell_location"] = stag_location
            ds = ds.set_coords("zShift_" + stag_location)
            ds = ds.drop_vars("zShift")

            with pytest.raises(ValueError):
                # Check shifting non-staggered field fails without zShift
                ds["T"].bout.from_field_aligned()

        n_stag_nal = ds["n"].bout.from_field_aligned()

        npt.assert_equal(n_stag_nal.values, n_nal.values)

    @pytest.mark.long
    def test_interpolate_parallel_region_core(self, bout_xyt_example_files):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 16, 3),
            nxpe=1,
            nype=1,
            nt=1,
            grid="grid",
            guards={"y": 2},
            topology="core",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        n = ds["n"]

        thetalength = 2.0 * np.pi

        dtheta = thetalength / 16.0
        theta = xr.DataArray(
            np.linspace(0.0 - 1.5 * dtheta, thetalength + 1.5 * dtheta, 20),
            dims="theta",
        )

        dtheta_fine = thetalength / 128.0
        theta_fine = xr.DataArray(
            np.linspace(0.0 + dtheta_fine / 2.0, thetalength - dtheta_fine / 2.0, 128),
            dims="theta",
        )

        def f(t):
            t = np.sin(t)
            return t**3 - t**2 + t - 1.0

        n.data = f(theta).broadcast_like(n)

        n_highres = n.bout.interpolate_parallel("core")

        expected = f(theta_fine).broadcast_like(n_highres)

        npt.assert_allclose(n_highres.values, expected.values, rtol=0.0, atol=1.0e-2)

    @pytest.mark.parametrize(
        "res_factor",
        [
            pytest.param(2, marks=pytest.mark.long),
            3,
            pytest.param(7, marks=pytest.mark.long),
            pytest.param(18, marks=pytest.mark.long),
        ],
    )
    def test_interpolate_parallel_region_core_change_n(
        self, bout_xyt_example_files, res_factor
    ):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 16, 3),
            nxpe=1,
            nype=1,
            nt=1,
            grid="grid",
            guards={"y": 2},
            topology="core",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        n = ds["n"]

        thetalength = 2.0 * np.pi

        dtheta = thetalength / 16.0
        theta = xr.DataArray(
            np.linspace(0.0 - 1.5 * dtheta, thetalength + 1.5 * dtheta, 20),
            dims="theta",
        )

        dtheta_fine = thetalength / res_factor / 16.0
        theta_fine = xr.DataArray(
            np.linspace(
                0.0 + dtheta_fine / 2.0,
                thetalength - dtheta_fine / 2.0,
                res_factor * 16,
            ),
            dims="theta",
        )

        def f(t):
            t = np.sin(t)
            return t**3 - t**2 + t - 1.0

        n.data = f(theta).broadcast_like(n)

        n_highres = n.bout.interpolate_parallel("core", n=res_factor)

        expected = f(theta_fine).broadcast_like(n_highres)

        npt.assert_allclose(n_highres.values, expected.values, rtol=0.0, atol=1.0e-2)

    @pytest.mark.long
    def test_interpolate_parallel_region_sol(self, bout_xyt_example_files):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 16, 3),
            nxpe=1,
            nype=1,
            nt=1,
            grid="grid",
            guards={"y": 2},
            topology="sol",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        n = ds["n"]

        thetalength = 2.0 * np.pi

        dtheta = thetalength / 16.0
        theta = xr.DataArray(
            np.linspace(0.0 - 1.5 * dtheta, thetalength + 1.5 * dtheta, 20),
            dims="theta",
        )

        dtheta_fine = thetalength / 128.0
        theta_fine = xr.DataArray(
            np.linspace(0.0 - 1.5 * dtheta_fine, thetalength + 1.5 * dtheta_fine, 132),
            dims="theta",
        )

        def f(t):
            t = np.sin(t)
            return t**3 - t**2 + t - 1.0

        n.data = f(theta).broadcast_like(n)

        n_highres = n.bout.interpolate_parallel("SOL")

        expected = f(theta_fine).broadcast_like(n_highres)

        npt.assert_allclose(n_highres.values, expected.values, rtol=0.0, atol=1.0e-2)

    def test_interpolate_parallel_region_singlenull(self, bout_xyt_example_files):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 16, 3),
            nxpe=1,
            nype=3,
            nt=1,
            grid="grid",
            guards={"y": 2},
            topology="single-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        n = ds["n"]

        thetalength = 2.0 * np.pi

        dtheta = thetalength / 48.0
        theta = xr.DataArray(
            np.linspace(0.0 - 1.5 * dtheta, thetalength + 1.5 * dtheta, 52),
            dims="theta",
        )

        dtheta_fine = thetalength / 3.0 / 128.0
        theta_fine = xr.DataArray(
            np.linspace(
                0.0 + 0.5 * dtheta_fine, thetalength - 0.5 * dtheta_fine, 3 * 128
            ),
            dims="theta",
        )

        def f(t):
            t = np.sin(3.0 * t)
            return t**3 - t**2 + t - 1.0

        n.data = f(theta).broadcast_like(n)

        f_fine = f(theta_fine)[:128]

        for region in ["inner_PFR", "inner_SOL"]:
            n_highres = n.bout.interpolate_parallel(region).isel(theta=slice(2, None))

            expected = f_fine.broadcast_like(n_highres)

            npt.assert_allclose(
                n_highres.values, expected.values, rtol=0.0, atol=1.0e-2
            )

        for region in ["core", "SOL"]:
            n_highres = n.bout.interpolate_parallel(region)

            expected = f_fine.broadcast_like(n_highres)

            npt.assert_allclose(
                n_highres.values, expected.values, rtol=0.0, atol=1.0e-2
            )

        for region in ["outer_PFR", "outer_SOL"]:
            n_highres = n.bout.interpolate_parallel(region).isel(theta=slice(-2))

            expected = f_fine.broadcast_like(n_highres)

            npt.assert_allclose(
                n_highres.values, expected.values, rtol=0.0, atol=1.0e-2
            )

    def test_interpolate_parallel(self, bout_xyt_example_files):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 16, 3),
            nxpe=1,
            nype=3,
            nt=1,
            grid="grid",
            guards={"y": 2},
            topology="single-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        n = ds["n"]

        thetalength = 2.0 * np.pi

        dtheta = thetalength / 48.0
        theta = xr.DataArray(
            np.linspace(0.0 - 1.5 * dtheta, thetalength + 1.5 * dtheta, 52),
            dims="theta",
        )

        dtheta_fine = thetalength / 3.0 / 128.0
        theta_fine = xr.DataArray(
            np.linspace(
                0.0 + 0.5 * dtheta_fine, thetalength - 0.5 * dtheta_fine, 3 * 128
            ),
            dims="theta",
        )
        x = xr.DataArray(np.arange(3), dims="x")

        def f_y(t):
            t = np.sin(3.0 * t)
            return t**3 - t**2 + t - 1.0

        f = f_y(theta) * (x + 1.0)

        n.data = f.broadcast_like(n)

        f_fine = f_y(theta_fine) * (x + 1.0)

        n_highres = n.bout.interpolate_parallel().isel(theta=slice(2, -2))

        expected = f_fine.broadcast_like(n_highres)

        npt.assert_allclose(n_highres.values, expected.values, rtol=0.0, atol=1.1e-2)

    def test_interpolate_parallel_sol(self, bout_xyt_example_files):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 16, 3),
            nxpe=1,
            nype=1,
            nt=1,
            grid="grid",
            guards={"y": 2},
            topology="sol",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        n = ds["n"]

        thetalength = 2.0 * np.pi

        dtheta = thetalength / 16.0
        theta = xr.DataArray(
            np.linspace(0.0 - 1.5 * dtheta, thetalength + 1.5 * dtheta, 20),
            dims="theta",
        )

        dtheta_fine = thetalength / 128.0
        theta_fine = xr.DataArray(
            np.linspace(0.0 + 0.5 * dtheta_fine, thetalength - 0.5 * dtheta_fine, 128),
            dims="theta",
        )
        x = xr.DataArray(np.arange(3), dims="x")

        def f_y(t):
            t = np.sin(t)
            return t**3 - t**2 + t - 1.0

        f = f_y(theta) * (x + 1.0)

        n.data = f.broadcast_like(n)

        f_fine = f_y(theta_fine) * (x + 1.0)

        n_highres = n.bout.interpolate_parallel().isel(theta=slice(2, -2))

        expected = f_fine.broadcast_like(n_highres)

        npt.assert_allclose(n_highres.values, expected.values, rtol=0.0, atol=1.1e-2)

    def test_interpolate_parallel_toroidal_points(self, bout_xyt_example_files):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 16, 3),
            nxpe=1,
            nype=3,
            nt=1,
            grid="grid",
            guards={"y": 2},
            topology="single-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        n_highres = ds["n"].bout.interpolate_parallel()

        n_highres_truncated = ds["n"].bout.interpolate_parallel(toroidal_points=2)

        xrt.assert_identical(n_highres_truncated, n_highres.isel(zeta=[0, 2]))

    def test_interpolate_parallel_toroidal_points_list(self, bout_xyt_example_files):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 16, 3),
            nxpe=1,
            nype=3,
            nt=1,
            grid="grid",
            guards={"y": 2},
            topology="single-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        n_highres = ds["n"].bout.interpolate_parallel()

        points_list = [1, 2]

        n_highres_truncated = ds["n"].bout.interpolate_parallel(
            toroidal_points=points_list
        )

        xrt.assert_identical(n_highres_truncated, n_highres.isel(zeta=points_list))

    def test_interpolate_to_cartesian(self, bout_xyt_example_files):
        dataset_list = bout_xyt_example_files(
            None, lengths=(2, 16, 17, 18), nxpe=1, nype=1, nt=1
        )
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )

        ds["psixy"] = ds["g11"].copy(deep=True)
        ds["Rxy"] = ds["g11"].copy(deep=True)
        ds["Zxy"] = ds["g11"].copy(deep=True)

        r = np.linspace(1.0, 2.0, ds.metadata["nx"])
        theta = np.linspace(0.0, 2.0 * np.pi, ds.metadata["ny"])
        R = r[:, np.newaxis] * np.cos(theta[np.newaxis, :])
        Z = r[:, np.newaxis] * np.sin(theta[np.newaxis, :])
        ds["Rxy"].values[:] = R
        ds["Zxy"].values[:] = Z

        ds = apply_geometry(ds, "toroidal")

        da = ds["n"]
        da.values[:] = 1.0

        nX = 30
        nY = 30
        nZ = 10
        da_cartesian = da.bout.interpolate_to_cartesian(nX, nY, nZ)

        # Check a point inside the original grid
        npt.assert_allclose(
            da_cartesian.isel(t=0, X=round(nX * 4 / 5), Y=nY // 2, Z=nZ // 2).item(),
            1.0,
            rtol=1.0e-15,
            atol=1.0e-15,
        )
        # Check a point outside the original grid
        assert np.isnan(da_cartesian.isel(t=0, X=0, Y=0, Z=0).item())
        # Check output is float32
        assert da_cartesian.dtype == np.float32

    def test_add_cartesian_coordinates(self, bout_xyt_example_files):
        dataset_list = bout_xyt_example_files(None, nxpe=1, nype=1, nt=1)
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )

        ds["psixy"] = ds["g11"].copy(deep=True)
        ds["Rxy"] = ds["g11"].copy(deep=True)
        ds["Zxy"] = ds["g11"].copy(deep=True)

        r = np.linspace(1.0, 2.0, ds.metadata["nx"])
        theta = np.linspace(0.0, 2.0 * np.pi, ds.metadata["ny"])
        R = r[:, np.newaxis] * np.cos(theta[np.newaxis, :])
        Z = r[:, np.newaxis] * np.sin(theta[np.newaxis, :])
        ds["Rxy"].values[:] = R
        ds["Zxy"].values[:] = Z

        ds = apply_geometry(ds, "toroidal")

        zeta = ds["zeta"].values

        da = ds["n"].bout.add_cartesian_coordinates()

        npt.assert_allclose(
            da["X_cartesian"],
            R[:, :, np.newaxis] * np.cos(zeta[np.newaxis, np.newaxis, :]),
        )
        npt.assert_allclose(
            da["Y_cartesian"],
            R[:, :, np.newaxis] * np.sin(zeta[np.newaxis, np.newaxis, :]),
        )
        npt.assert_allclose(
            da["Z_cartesian"],
            Z[:, :, np.newaxis] * np.ones(ds.metadata["nz"])[np.newaxis, np.newaxis, :],
        )

    def test_ddx(self, bout_xyt_example_files):

        nx = 64

        dataset_list = bout_xyt_example_files(
            None,
            lengths=(2, nx, 4, 3),
            nxpe=1,
            nype=1,
        )

        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list,
            )

        n = ds["n"]

        t = ds["t"].broadcast_like(n)
        ds["x_1d"] = _1d_coord_from_spacing(ds["dx"], "x")
        x = ds["x_1d"].broadcast_like(n)
        y = ds["y"].broadcast_like(n)
        z = ds["z"].broadcast_like(n)

        n.values[:] = (np.sin(12.0 * x / nx) * (1.0 + t + y + z)).values

        expected = 12.0 / nx * np.cos(12.0 * x / nx) * (1.0 + t + y + z)

        npt.assert_allclose(
            n.bout.ddx().isel(x=slice(1, -1)).values,
            expected.isel(x=slice(1, -1)).values,
            rtol=1.0e-2,
            atol=1.0e-13,
        )

    def test_ddy(self, bout_xyt_example_files):

        ny = 64

        dataset_list, gridfilepath = bout_xyt_example_files(
            None,
            lengths=(2, 3, ny, 4),
            nxpe=1,
            nype=1,
            grid="grid",
        )

        ds = open_boutdataset(
            datapath=dataset_list, geometry="toroidal", gridfilepath=gridfilepath
        )

        n = ds["n"]

        t = ds["t"].broadcast_like(n)
        x = ds["x"].broadcast_like(n)
        y = ds["theta"].broadcast_like(n)
        z = ds["zeta"].broadcast_like(n)

        n.values[:] = (np.sin(3.0 * y / ny) * (1.0 + t + x + z)).values

        expected = 3.0 / ny * np.cos(3.0 * y / ny) * (1.0 + t + x + z)

        npt.assert_allclose(
            n.bout.ddy().isel(theta=slice(1, -1)).values,
            expected.isel(theta=slice(1, -1)).values,
            rtol=1.0e-2,
            atol=1.0e-13,
        )

    def test_ddz(self, bout_xyt_example_files):
        dataset_list = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 64),
            nxpe=1,
            nype=1,
        )

        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list,
            )

        n = ds["n"]

        t = ds["t"].broadcast_like(n)
        x = ds["x"].broadcast_like(n)
        y = ds["y"].broadcast_like(n)
        z = ds["z"].broadcast_like(n)

        n.values[:] = (np.sin(z) * (1.0 + t + x + y)).values

        expected = np.cos(z) * (1.0 + t + x + y)

        npt.assert_allclose(
            n.bout.ddz().values, expected.values, rtol=1.0e-2, atol=1.0e-13
        )

    def test_derivatives_doublenull(self, bout_xyt_example_files):
        # Check function does not error on double-null topology
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=1,
            nype=6,
            nt=1,
            grid="grid",
            guards={"x": 2, "y": 2},
            topology="connected-double-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_yboundaries=True,
        )

        test_ddx = ds["n"].bout.ddx()
        test_ddy = ds["n"].bout.ddy()
        test_ddz = ds["n"].bout.ddz()
