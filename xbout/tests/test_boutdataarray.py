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
    def test_to_field_aligned(self, bout_xyt_example_files, nz):
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
        n_al = n.bout.to_field_aligned()
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

    def test_to_field_aligned_dask(self, bout_xyt_example_files):

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
        n_al = n.bout.to_field_aligned()
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
    def test_from_field_aligned(self, bout_xyt_example_files, nz):
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
        n_nal = n.bout.from_field_aligned()
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
    def test_to_field_aligned_staggered(self, bout_xyt_example_files, stag_location):
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

        n_al = n.bout.to_field_aligned().copy(deep=True)

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
    def test_from_field_aligned_staggered(self, bout_xyt_example_files, stag_location):
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

        n_nal = n.bout.from_field_aligned().copy(deep=True)

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

        n_stag_al = ds["n"].bout.from_field_aligned()

        npt.assert_equal(n_stag_al.values, n_nal.values)

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
            return t ** 3 - t ** 2 + t - 1.0

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
            return t ** 3 - t ** 2 + t - 1.0

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
            return t ** 3 - t ** 2 + t - 1.0

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
            return t ** 3 - t ** 2 + t - 1.0

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
            return t ** 3 - t ** 2 + t - 1.0

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
            return t ** 3 - t ** 2 + t - 1.0

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
