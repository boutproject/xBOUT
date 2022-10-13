import pytest

import numpy.testing as npt
from xarray import Dataset, DataArray, concat, open_dataset, open_mfdataset
import xarray.testing as xrt

import dask.array
import numpy as np
from pathlib import Path
from scipy.integrate import quad_vec

from xbout.tests.test_load import bout_xyt_example_files, create_bout_ds
from xbout.tests.test_region import (
    params_guards,
    params_guards_values,
    params_boundaries,
    params_boundaries_values,
)
from xbout import BoutDatasetAccessor, open_boutdataset
from xbout.geometries import apply_geometry
from xbout.utils import _set_attrs_on_all_vars, _1d_coord_from_spacing
from xbout.tests.utils_for_tests import set_geometry_from_input_file


EXAMPLE_OPTIONS_FILE_PATH = "./xbout/tests/data/options/BOUT.inp"


class TestBoutDatasetIsXarrayDataset:
    """
    Set of tests to check that BoutDatasets behave similarly to xarray Datasets.
    (With the accessor approach these should pass trivially now.)
    """

    def test_concat(self, bout_xyt_example_files):
        dataset_list1 = bout_xyt_example_files(None, nxpe=3, nype=4, nt=1)
        with pytest.warns(UserWarning):
            bd1 = open_boutdataset(
                datapath=dataset_list1, inputfilepath=None, keep_xboundaries=False
            )
        dataset_list2 = bout_xyt_example_files(None, nxpe=3, nype=4, nt=1)
        with pytest.warns(UserWarning):
            bd2 = open_boutdataset(
                datapath=dataset_list2, inputfilepath=None, keep_xboundaries=False
            )
        result = concat([bd1, bd2], dim="run")
        assert result.dims == {**bd1.dims, "run": 2}

    def test_isel(self, bout_xyt_example_files):
        dataset_list = bout_xyt_example_files(None, nxpe=1, nype=1, nt=1)
        with pytest.warns(UserWarning):
            bd = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )
        actual = bd.isel(x=slice(None, None, 2))
        expected = bd.bout.data.isel(x=slice(None, None, 2))
        xrt.assert_equal(actual, expected)


class TestBoutDatasetMethods:
    def test_get_field_aligned(self, bout_xyt_example_files):
        dataset_list = bout_xyt_example_files(None, nxpe=3, nype=4, nt=1)
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )

        ds["psixy"] = ds["x"]
        ds["Rxy"] = ds["x"]
        ds["Zxy"] = ds["y"]

        ds = apply_geometry(ds, "toroidal")

        n = ds["n"]
        n.attrs["direction_y"] = "Standard"
        n_aligned_from_array = n.bout.to_field_aligned()

        # check n_aligned does not exist yet
        assert "n_aligned" not in ds

        n_aligned_from_ds = ds.bout.get_field_aligned("n")
        xrt.assert_allclose(n_aligned_from_ds, n_aligned_from_array)
        xrt.assert_allclose(ds["n_aligned"], n_aligned_from_array)

        # check getting the cached version
        ds["n_aligned"] = ds["T"]
        ds["n_aligned"].attrs["direction_y"] = "Aligned"
        xrt.assert_allclose(ds.bout.get_field_aligned("n"), ds["T"])

    @pytest.mark.parametrize("mxg", [0, pytest.param(2, marks=pytest.mark.long)])
    @pytest.mark.parametrize("myg", [pytest.param(0, marks=pytest.mark.long), 2])
    def test_remove_yboundaries(self, bout_xyt_example_files, mxg, myg):
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

        ds = ds.bout.remove_yboundaries()

        assert ds.metadata["keep_yboundaries"] == 0
        for v in ds:
            assert ds[v].metadata["keep_yboundaries"] == 0

        # expect theta coordinate to be different, so ignore
        ds = ds.drop_vars("theta")
        ds_no_yboundaries = ds_no_yboundaries.drop_vars("theta")

        # expect MYG can be different, so ignore
        del ds.metadata["MYG"]
        del ds_no_yboundaries.metadata["MYG"]
        for v in ds:
            del ds[v].metadata["MYG"]
            # metadata on variables is a reference to ds_no_yboundaries.metadata, so
            # don't need to delete MYG again

        xrt.assert_equal(ds, ds_no_yboundaries)

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
        ds["n"] = n

        assert ds["n"].direction_y == "Standard"

        # Create field-aligned Dataset
        ds_al = ds.bout.to_field_aligned()

        n_al = ds_al["n"]

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
        ds = ds.chunk({"t": 1})
        assert isinstance(n.data, dask.array.Array)
        assert isinstance(ds["n"].data, dask.array.Array)

        n.attrs["direction_y"] = "Standard"
        ds["n"] = n

        assert ds["n"].direction_y == "Standard"

        # Create field-aligned Dataset
        ds_al = ds.bout.to_field_aligned()

        n_al = ds_al["n"]

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
        ds["n"] = n

        assert ds["n"].direction_y == "Aligned"

        # Create non-field-aligned Dataset
        ds_nal = ds.bout.from_field_aligned()

        n_nal = ds_nal["n"]

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

        ds["n"] = n

        assert ds["n"].direction_y == "Standard"

        # Create field-aligned Dataset
        ds_al = ds.bout.to_field_aligned()

        n_al = ds_al["n"].copy(deep=True)

        assert n_al.direction_y == "Aligned"

        # make 'n' staggered
        ds["n"].attrs["cell_location"] = stag_location

        if stag_location != "CELL_ZLOW":
            with pytest.raises(ValueError):
                # Check exception raised when needed zShift_CELL_*LOW is not present
                ds.bout.to_field_aligned()
            ds["zShift_" + stag_location] = zShift
            ds["zShift_" + stag_location].attrs["cell_location"] = stag_location
            ds = ds.set_coords("zShift_" + stag_location)

        # New field-aligned Dataset
        ds_al = ds.bout.to_field_aligned()

        n_stag_al = ds_al["n"]

        assert n_stag_al.direction_y == "Aligned"

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
        ds["n"] = n
        ds["T"].attrs["direction_y"] = "Aligned"

        # Make non-field-aligned Dataset
        ds_nal = ds.bout.from_field_aligned()

        n_nal = ds_nal["n"].copy(deep=True)

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

        # New non-field-aligned Dataset
        ds_nal = ds.bout.from_field_aligned()

        n_stag_nal = ds_nal["n"]

        assert n_stag_nal.direction_y == "Standard"

        npt.assert_equal(n_stag_nal.values, n_nal.values)

    def test_set_parallel_interpolation_factor(self):
        ds = Dataset()
        ds["a"] = DataArray()
        ds = _set_attrs_on_all_vars(ds, "metadata", {})

        with pytest.raises(KeyError):
            ds.metadata["fine_interpolation_factor"]
        with pytest.raises(KeyError):
            ds["a"].metadata["fine_interpolation_factor"]

        ds.bout.fine_interpolation_factor = 42

        assert ds.metadata["fine_interpolation_factor"] == 42
        assert ds["a"].metadata["fine_interpolation_factor"] == 42

    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    @pytest.mark.parametrize(
        "vars_to_interpolate", [("n", "T"), pytest.param(..., marks=pytest.mark.long)]
    )
    def test_interpolate_parallel(
        self,
        bout_xyt_example_files,
        guards,
        keep_xboundaries,
        keep_yboundaries,
        vars_to_interpolate,
    ):
        # This test checks that the regions created in the new high-resolution Dataset by
        # interpolate_parallel are correct.
        # This test does not test the accuracy of the parallel interpolation (there are
        # other tests for that).

        # Note using more than MXG x-direction points and MYG y-direction points per
        # output file ensures tests for whether boundary cells are present do not fail
        # when using minimal numbers of processors
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=6,
            nt=1,
            guards=guards,
            grid="grid",
            topology="lower-disconnected-double-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
        )

        # Get high parallel resolution version of ds, and check that
        ds = ds.bout.interpolate_parallel(vars_to_interpolate)

        mxg = guards["x"]
        myg = guards["y"]

        if keep_xboundaries:
            ixs1 = ds.metadata["ixseps1"]
        else:
            ixs1 = ds.metadata["ixseps1"] - guards["x"]

        if keep_xboundaries:
            ixs2 = ds.metadata["ixseps2"]
        else:
            ixs2 = ds.metadata["ixseps2"] - guards["x"]

        if keep_yboundaries:
            ybndry = guards["y"]
        else:
            ybndry = 0
        jys11 = ds.metadata["jyseps1_1"] + ybndry
        jys21 = ds.metadata["jyseps2_1"] + ybndry
        ny_inner = ds.metadata["ny_inner"] + 2 * ybndry
        jys12 = ds.metadata["jyseps1_2"] + 3 * ybndry
        jys22 = ds.metadata["jyseps2_2"] + 3 * ybndry
        ny = ds.metadata["ny"] + 4 * ybndry

        for var in ["n", "T"]:
            v = ds[var]
            v_noregions = v.copy(deep=True)

            # Remove attributes that are expected to be different
            del v_noregions.attrs["regions"]

            v_lower_inner_PFR = v.bout.from_region("lower_inner_PFR")

            # Remove attributes that are expected to be different
            del v_lower_inner_PFR.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(x=slice(ixs1 + mxg), theta=slice(jys11 + 1)),
                v_lower_inner_PFR.isel(theta=slice(-myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 + mxg), theta=slice(jys22 + 1, jys22 + 1 + myg)
                    ).values,
                    v_lower_inner_PFR.isel(theta=slice(-myg, None)).values,
                )

            v_lower_inner_intersep = v.bout.from_region("lower_inner_intersep")

            # Remove attributes that are expected to be different
            del v_lower_inner_intersep.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs1 - mxg, ixs2 + mxg), theta=slice(jys11 + 1)
                ),
                v_lower_inner_intersep.isel(theta=slice(-myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 - mxg, ixs2 + mxg),
                        theta=slice(jys11 + 1, jys11 + 1 + myg),
                    ).values,
                    v_lower_inner_intersep.isel(theta=slice(-myg, None)).values,
                )

            v_lower_inner_SOL = v.bout.from_region("lower_inner_SOL")

            # Remove attributes that are expected to be different
            del v_lower_inner_SOL.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(x=slice(ixs2 - mxg, None), theta=slice(jys11 + 1)),
                v_lower_inner_SOL.isel(theta=slice(-myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs2 - mxg, None),
                        theta=slice(jys11 + 1, jys11 + 1 + myg),
                    ).values,
                    v_lower_inner_SOL.isel(theta=slice(-myg, None)).values,
                )

            v_inner_core = v.bout.from_region("inner_core")

            # Remove attributes that are expected to be different
            del v_inner_core.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs1 + mxg), theta=slice(jys11 + 1, jys21 + 1)
                ),
                v_inner_core.isel(theta=slice(myg, -myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 + mxg), theta=slice(jys22 + 1 - myg, jys22 + 1)
                    ).values,
                    v_inner_core.isel(theta=slice(myg)).values,
                )
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 + mxg), theta=slice(jys12 + 1, jys12 + 1 + myg)
                    ).values,
                    v_inner_core.isel(theta=slice(-myg, None)).values,
                )

            v_inner_intersep = v.bout.from_region("inner_intersep")

            # Remove attributes that are expected to be different
            del v_inner_intersep.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs1 - mxg, ixs2 + mxg), theta=slice(jys11 + 1, jys21 + 1)
                ),
                v_inner_intersep.isel(theta=slice(myg, -myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 - mxg, ixs2 + mxg),
                        theta=slice(jys11 + 1 - myg, jys11 + 1),
                    ).values,
                    v_inner_intersep.isel(theta=slice(myg)).values,
                )
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 - mxg, ixs2 + mxg),
                        theta=slice(jys12 + 1, jys12 + 1 + myg),
                    ).values,
                    v_inner_intersep.isel(theta=slice(-myg, None)).values,
                )

            v_inner_sol = v.bout.from_region("inner_SOL")

            # Remove attributes that are expected to be different
            del v_inner_sol.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs2 - mxg, None), theta=slice(jys11 + 1, jys21 + 1)
                ),
                v_inner_sol.isel(theta=slice(myg, -myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs2 - mxg, None),
                        theta=slice(jys11 + 1 - myg, jys11 + 1),
                    ).values,
                    v_inner_sol.isel(theta=slice(myg)).values,
                )
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs2 - mxg, None),
                        theta=slice(jys21 + 1, jys21 + 1 + myg),
                    ).values,
                    v_inner_sol.isel(theta=slice(-myg, None)).values,
                )

            v_upper_inner_PFR = v.bout.from_region("upper_inner_PFR")

            # Remove attributes that are expected to be different
            del v_upper_inner_PFR.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(x=slice(ixs1 + mxg), theta=slice(jys21 + 1, ny_inner)),
                v_upper_inner_PFR.isel(theta=slice(myg, None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 + mxg), theta=slice(jys12 + 1 - myg, jys12 + 1)
                    ).values,
                    v_upper_inner_PFR.isel(theta=slice(myg)).values,
                )

            v_upper_inner_intersep = v.bout.from_region("upper_inner_intersep")

            # Remove attributes that are expected to be different
            del v_upper_inner_intersep.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs1 - mxg, ixs2 + mxg), theta=slice(jys21 + 1, ny_inner)
                ),
                v_upper_inner_intersep.isel(theta=slice(myg, None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 - mxg, ixs2 + mxg),
                        theta=slice(jys12 + 1 - myg, jys12 + 1),
                    ).values,
                    v_upper_inner_intersep.isel(theta=slice(myg)).values,
                )

            v_upper_inner_SOL = v.bout.from_region("upper_inner_SOL")

            # Remove attributes that are expected to be different
            del v_upper_inner_SOL.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs2 - mxg, None), theta=slice(jys21 + 1, ny_inner)
                ),
                v_upper_inner_SOL.isel(theta=slice(myg, None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs2 - mxg, None),
                        theta=slice(jys21 + 1 - myg, jys21 + 1),
                    ).values,
                    v_upper_inner_SOL.isel(theta=slice(myg)).values,
                )

            v_upper_outer_PFR = v.bout.from_region("upper_outer_PFR")

            # Remove attributes that are expected to be different
            del v_upper_outer_PFR.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(x=slice(ixs1 + mxg), theta=slice(ny_inner, jys12 + 1)),
                v_upper_outer_PFR.isel(theta=slice(-myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 + mxg), theta=slice(jys21 + 1, jys21 + 1 + myg)
                    ).values,
                    v_upper_outer_PFR.isel(theta=slice(-myg, None)).values,
                )

            v_upper_outer_intersep = v.bout.from_region("upper_outer_intersep")

            # Remove attributes that are expected to be different
            del v_upper_outer_intersep.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs1 - mxg, ixs2 + mxg), theta=slice(ny_inner, jys12 + 1)
                ),
                v_upper_outer_intersep.isel(theta=slice(-myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 - mxg, ixs2 + mxg),
                        theta=slice(jys21 + 1, jys21 + 1 + myg),
                    ).values,
                    v_upper_outer_intersep.isel(theta=slice(-myg, None)).values,
                )

            v_upper_outer_SOL = v.bout.from_region("upper_outer_SOL")

            # Remove attributes that are expected to be different
            del v_upper_outer_SOL.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs2 - mxg, None), theta=slice(ny_inner, jys12 + 1)
                ),
                v_upper_outer_SOL.isel(theta=slice(-myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs2 - mxg, None),
                        theta=slice(jys12 + 1, jys12 + 1 + myg),
                    ).values,
                    v_upper_outer_SOL.isel(theta=slice(-myg, None)).values,
                )

            v_outer_core = v.bout.from_region("outer_core")

            # Remove attributes that are expected to be different
            del v_outer_core.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs1 + mxg), theta=slice(jys12 + 1, jys22 + 1)
                ),
                v_outer_core.isel(theta=slice(myg, -myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 + mxg), theta=slice(jys21 + 1 - myg, jys21 + 1)
                    ).values,
                    v_outer_core.isel(theta=slice(myg)).values,
                )
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 + mxg), theta=slice(jys11 + 1, jys11 + 1 + myg)
                    ).values,
                    v_outer_core.isel(theta=slice(-myg, None)).values,
                )

            v_outer_intersep = v.bout.from_region("outer_intersep")

            # Remove attributes that are expected to be different
            del v_outer_intersep.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs1 - mxg, ixs2 + mxg), theta=slice(jys12 + 1, jys22 + 1)
                ),
                v_outer_intersep.isel(theta=slice(myg, -myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 - mxg, ixs2 + mxg),
                        theta=slice(jys21 + 1 - myg, jys21 + 1),
                    ).values,
                    v_outer_intersep.isel(theta=slice(myg)).values,
                )
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 - mxg, ixs2 + mxg),
                        theta=slice(jys22 + 1, jys22 + 1 + myg),
                    ).values,
                    v_outer_intersep.isel(theta=slice(-myg, None)).values,
                )

            v_outer_sol = v.bout.from_region("outer_SOL")

            # Remove attributes that are expected to be different
            del v_outer_sol.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs2 - mxg, None), theta=slice(jys12 + 1, jys22 + 1)
                ),
                v_outer_sol.isel(theta=slice(myg, -myg if myg != 0 else None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs2 - mxg, None),
                        theta=slice(jys12 + 1 - myg, jys12 + 1),
                    ).values,
                    v_outer_sol.isel(theta=slice(myg)).values,
                )
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs2 - mxg, None),
                        theta=slice(jys22 + 1, jys22 + 1 + myg),
                    ).values,
                    v_outer_sol.isel(theta=slice(-myg, None)).values,
                )

            v_lower_outer_PFR = v.bout.from_region("lower_outer_PFR")

            # Remove attributes that are expected to be different
            del v_lower_outer_PFR.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(x=slice(ixs1 + mxg), theta=slice(jys22 + 1, None)),
                v_lower_outer_PFR.isel(theta=slice(myg, None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 + mxg), theta=slice(jys11 + 1 - myg, jys11 + 1)
                    ).values,
                    v_lower_outer_PFR.isel(theta=slice(myg)).values,
                )

            v_lower_outer_intersep = v.bout.from_region("lower_outer_intersep")

            # Remove attributes that are expected to be different
            del v_lower_outer_intersep.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs1 - mxg, ixs2 + mxg), theta=slice(jys22 + 1, None)
                ),
                v_lower_outer_intersep.isel(theta=slice(myg, None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs1 - mxg, ixs2 + mxg),
                        theta=slice(jys22 + 1 - myg, jys22 + 1),
                    ).values,
                    v_lower_outer_intersep.isel(theta=slice(myg)).values,
                )

            v_lower_outer_SOL = v.bout.from_region("lower_outer_SOL")

            # Remove attributes that are expected to be different
            del v_lower_outer_SOL.attrs["regions"]
            xrt.assert_identical(
                v_noregions.isel(
                    x=slice(ixs2 - mxg, None), theta=slice(jys22 + 1, None)
                ),
                v_lower_outer_SOL.isel(theta=slice(myg, None)),
            )
            if myg > 0:
                # check y-guards, which were 'communicated' by from_region
                # Coordinates are not equal, so only compare array values
                npt.assert_equal(
                    v_noregions.isel(
                        x=slice(ixs2 - mxg, None),
                        theta=slice(jys22 + 1 - myg, jys22 + 1),
                    ).values,
                    v_lower_outer_SOL.isel(theta=slice(myg)).values,
                )

    def test_interpolate_parallel_all_variables_arg(self, bout_xyt_example_files):
        # Check that passing 'variables=...' to interpolate_parallel() does actually
        # interpolate all the variables
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=1,
            nype=1,
            nt=1,
            grid="grid",
            topology="sol",
        )

        ds = open_boutdataset(
            datapath=dataset_list, gridfilepath=grid_ds, geometry="toroidal"
        )

        # Get high parallel resolution version of ds, and check that
        ds = ds.bout.interpolate_parallel(...)

        interpolated_variables = [v for v in ds]

        assert set(interpolated_variables) == set(
            (
                "n",
                "T",
                "S",
                "g11",
                "g22",
                "g33",
                "g12",
                "g13",
                "g23",
                "g_11",
                "g_22",
                "g_33",
                "g_12",
                "g_13",
                "g_23",
                "G1",
                "G2",
                "G3",
                "J",
                "Bxy",
            )
        )

    def test_interpolate_parallel_limiter(
        self,
        bout_xyt_example_files,
    ):
        # This test checks that the regions created in the new high-resolution Dataset by
        # interpolate_parallel are correct.
        # This test does not test the accuracy of the parallel interpolation (there are
        # other tests for that).
        # Note using more than MXG x-direction points and MYG y-direction points per
        # output file ensures tests for whether boundary cells are present do not fail
        # when using minimal numbers of processors
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=6,
            nt=1,
            guards={"x": 2, "y": 2},
            grid="grid",
            topology="limiter",
        )
        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=True,
            keep_yboundaries=False,
        )
        # Get high parallel resolution version of ds, and check that
        ds = ds.bout.interpolate_parallel(["n", "T"])
        mxg = 2
        myg = 2
        ixs1 = ds.metadata["ixseps1"]
        for var in ["n", "T"]:
            v = ds[var]
            v_noregions = v.copy(deep=True)
            # Remove attributes that are expected to be different
            del v_noregions.attrs["regions"]
            v_core = v.bout.from_region("core", with_guards={"theta": 0})
            # Remove attributes that are expected to be different
            del v_core.attrs["regions"]
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 + mxg)), v_core)
            v_sol = v.bout.from_region("SOL")
            # Remove attributes that are expected to be different
            del v_sol.attrs["regions"]
            xrt.assert_identical(v_noregions.isel(x=slice(ixs1 - mxg, None)), v_sol)

    def test_integrate_midpoints_slab(self, bout_xyt_example_files):
        # Create data
        dataset_list = bout_xyt_example_files(
            None, lengths=(4, 100, 110, 120), nxpe=1, nype=1, nt=1, syn_data_type=1
        )
        ds = open_boutdataset(dataset_list)
        t = np.linspace(0.0, 8.0, 4)[:, np.newaxis, np.newaxis, np.newaxis]
        x = np.linspace(0.05, 9.95, 100)[np.newaxis, :, np.newaxis, np.newaxis]
        y = np.linspace(0.1, 21.9, 110)[np.newaxis, np.newaxis, :, np.newaxis]
        z = np.linspace(0.15, 35.85, 120)[np.newaxis, np.newaxis, np.newaxis, :]
        ds["t"].data[...] = t.squeeze()
        ds["dx"].data[...] = 0.1
        ds["dy"].data[...] = 0.2
        ds["dz"] = 0.3
        tfunc = 1.5 * t
        xfunc = x**2
        yfunc = 10.0 * y - y**2
        zfunc = 2.0 * z**2 - 30.0 * z
        ds["n"].data[...] = tfunc * xfunc * yfunc * zfunc
        tintegral = 48.0
        xintegral = 1000.0 / 3.0
        yintegral = 5.0 * 22.0**2 - 22.0**3 / 3.0
        zintegral = 2.0 * 36.0**3 / 3.0 - 15.0 * 36.0**2
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims="t"),
            (tintegral * xfunc * yfunc * zfunc).squeeze(),
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims="x"),
            (tfunc * xintegral * yfunc * zfunc).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims="y"),
            (tfunc * xfunc * yintegral * zfunc).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims="z"),
            (tfunc * xfunc * yfunc * zintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "x"]),
            (tintegral * xintegral * yfunc * zfunc).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "y"]),
            (tintegral * xfunc * yintegral * zfunc).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "z"]),
            (tintegral * xfunc * yfunc * zintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["x", "y"]),
            (tfunc * xintegral * yintegral * zfunc).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["x", "z"]),
            (tfunc * xintegral * yfunc * zintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["y", "z"]),
            (tfunc * xfunc * yintegral * zintegral).squeeze(),
            rtol=1.2e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "x", "y"]),
            (tintegral * xintegral * yintegral * zfunc).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "x", "z"]),
            (tintegral * xintegral * yfunc * zintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "y", "z"]),
            (tintegral * xfunc * yintegral * zintegral).squeeze(),
            rtol=1.2e-4,
        )
        # default dims
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n"),
            (tfunc * xintegral * yintegral * zintegral).squeeze(),
            rtol=1.4e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "x", "y", "z"]),
            (tintegral * xintegral * yintegral * zintegral),
            rtol=1.4e-4,
        )

        # Create and test Field2D
        ds["S"][...] = (tfunc * xfunc * yfunc).squeeze()

        # S is 'axisymmetric' so z-integral is just the length of the
        # z-dimension
        zintegral = 36.0

        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims="t"),
            (tintegral * xfunc * yfunc).squeeze(),
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims="x"),
            (tfunc * xintegral * yfunc).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims="y"),
            (tfunc * xfunc * yintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims="z"),
            (tfunc * xfunc * yfunc * zintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["t", "x"]),
            (tintegral * xintegral * yfunc).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["t", "y"]),
            (tintegral * xfunc * yintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["t", "z"]),
            (tintegral * xfunc * yfunc * zintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["x", "y"]),
            (tfunc * xintegral * yintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["x", "z"]),
            (tfunc * xintegral * yfunc * zintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["y", "z"]),
            (tfunc * xfunc * yintegral * zintegral).squeeze(),
            rtol=1.2e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["t", "x", "y"]),
            (tintegral * xintegral * yintegral),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["t", "x", "z"]),
            (tintegral * xintegral * yfunc * zintegral).squeeze(),
            rtol=1.0e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["t", "y", "z"]),
            (tintegral * xfunc * yintegral * zintegral).squeeze(),
            rtol=1.2e-4,
        )
        # default dims
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S"),
            (tfunc * xintegral * yintegral * zintegral).squeeze(),
            rtol=1.4e-4,
        )
        npt.assert_allclose(
            ds.bout.integrate_midpoints("S", dims=["t", "x", "y", "z"]),
            (tintegral * xintegral * yintegral * zintegral),
            rtol=1.4e-4,
        )

    @pytest.mark.parametrize(
        "location", ["CELL_CENTRE", "CELL_XLOW", "CELL_YLOW", "CELL_ZLOW"]
    )
    def test_integrate_midpoints_salpha(self, bout_xyt_example_files, location):
        # Create data
        nx = 100
        ny = 110
        nz = 120
        dataset_list = bout_xyt_example_files(
            None,
            lengths=(4, nx, ny, nz),
            nxpe=1,
            nype=1,
            nt=1,
            syn_data_type=1,
            guards={"x": 2, "y": 2},
        )
        ds = open_boutdataset(dataset_list)

        ds, options = set_geometry_from_input_file(ds, "s-alpha.inp")

        ds = apply_geometry(ds, "toroidal")

        # Integrate 1 so we just get volume, areas and lengths
        ds["n"].values[:] = 1.0
        ds["n"].attrs["cell_location"] = location

        # remove boundary cells (don't want to integrate over those)
        ds = ds.bout.remove_yboundaries()
        if ds.metadata["keep_xboundaries"] and ds.metadata["MXG"] > 0:
            mxg = ds.metadata["MXG"]
            xslice = slice(mxg, -mxg)
        else:
            xslice = slice(None)
        ds = ds.isel(x=xslice)

        # Test geometry has major radius R and goes between minor radii a-Lr/2 and
        # a+Lr/2
        R = options.evaluate_scalar("mesh:R0")
        a = options.evaluate_scalar("mesh:a")
        Lr = options.evaluate_scalar("mesh:Lr")
        rinner = a - Lr / 2.0
        router = a + Lr / 2.0
        r = options.evaluate("mesh:r").squeeze()[xslice]
        if location == "CELL_XLOW":
            rinner = rinner - Lr / (2.0 * nx)
            router = router - Lr / (2.0 * nx)
            r = r - Lr / (2.0 * nx)
        q = options.evaluate_scalar("mesh:q")
        T_total = (ds.sizes["t"] - 1) * (ds["t"][1] - ds["t"][0]).values
        T_cumulative = np.arange(ds.sizes["t"]) * (ds["t"][1] - ds["t"][0]).values

        # Volume of torus with circular cross-section of major radius R and minor radius
        # a is 2*pi*R*pi*a^2
        # https://en.wikipedia.org/wiki/Torus
        # Default is to integrate over all spatial dimensions
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n"),
            2.0 * np.pi * R * np.pi * (router**2 - rinner**2),
            rtol=1.0e-5,
            atol=0.0,
        )
        # Pass all spatial dims explicitly
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["x", "theta", "zeta"]),
            2.0 * np.pi * R * np.pi * (router**2 - rinner**2),
            rtol=1.0e-5,
            atol=0.0,
        )
        # Integrate in time too
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "x", "theta", "zeta"]),
            T_total * 2.0 * np.pi * R * np.pi * (router**2 - rinner**2),
            rtol=1.0e-5,
            atol=0.0,
        )
        # Integrate in time using dims=...
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=...),
            T_total * 2.0 * np.pi * R * np.pi * (router**2 - rinner**2),
            rtol=1.0e-5,
            atol=0.0,
        )
        # Cumulative integral in time
        npt.assert_allclose(
            ds.bout.integrate_midpoints(
                "n", dims=["t", "x", "theta", "zeta"], cumulative_t=True
            ),
            T_cumulative * 2.0 * np.pi * R * np.pi * (router**2 - rinner**2),
            rtol=1.0e-5,
            atol=0.0,
        )

        # Area of torus with circular cross-section of major radius R and minor radius a
        # is 2*pi*R*2*pi*a
        # https://en.wikipedia.org/wiki/Torus
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["theta", "zeta"]),
            (2.0 * np.pi * R * 2.0 * np.pi * r)[np.newaxis, :]
            * np.ones(ds.sizes["t"])[:, np.newaxis],
            rtol=1.0e-5,
            atol=0.0,
        )
        # Integrate in time too
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "theta", "zeta"]),
            T_total * 2.0 * np.pi * R * 2.0 * np.pi * r,
            rtol=1.0e-5,
            atol=0.0,
        )
        # Cumulative integral in time
        npt.assert_allclose(
            ds.bout.integrate_midpoints(
                "n", dims=["t", "theta", "zeta"], cumulative_t=True
            ),
            T_cumulative[:, np.newaxis]
            * (2.0 * np.pi * R * 2.0 * np.pi * r)[np.newaxis, :],
            rtol=1.0e-5,
            atol=0.0,
        )

        # Area of cross section in poloidal plane is difference of circle with radius
        # router and circle with radius rinner, pi*(router**2 - rinner**2)
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["x", "theta"]),
            np.pi * (router**2 - rinner**2),
            rtol=1.0e-5,
            atol=0.0,
        )
        # Integrate in time too
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "x", "theta"]),
            T_total * np.pi * (router**2 - rinner**2),
            rtol=1.0e-5,
            atol=0.0,
        )
        # Cumulative integral in time
        npt.assert_allclose(
            ds.bout.integrate_midpoints(
                "n", dims=["t", "x", "theta"], cumulative_t=True
            ),
            T_cumulative[:, np.newaxis]
            * np.pi
            * (router**2 - rinner**2)
            * np.ones(ds.sizes["zeta"])[np.newaxis, :],
            rtol=1.0e-5,
            atol=0.0,
        )

        # x-z planes are 'conical frustrums', with area pi*(Rinner + Router)*Lr
        # https://en.wikipedia.org/wiki/Frustum
        theta = _1d_coord_from_spacing(ds["dy"], "theta").values
        if location == "CELL_YLOW":
            theta = theta - 2.0 * np.pi / (2.0 * ny)
        Rinner = R + rinner * np.cos(theta)
        Router = R + router * np.cos(theta)
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["x", "zeta"]),
            (np.pi * (Rinner + Router) * Lr)[np.newaxis, :]
            * np.ones(ds.sizes["t"])[:, np.newaxis],
            rtol=1.0e-5,
            atol=0.0,
        )
        # Integrate in time too
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "x", "zeta"]),
            T_total * (np.pi * (Rinner + Router) * Lr),
            rtol=1.0e-5,
            atol=0.0,
        )
        # Cumulative integral in time
        npt.assert_allclose(
            ds.bout.integrate_midpoints(
                "n", dims=["t", "x", "zeta"], cumulative_t=True
            ),
            T_cumulative[:, np.newaxis]
            * (np.pi * (Rinner + Router) * Lr)
            * np.ones(ds.sizes["theta"])[np.newaxis, :],
            rtol=1.0e-5,
            atol=0.0,
        )

        # Radial lines have length Lr
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["x"]),
            Lr,
            rtol=1.0e-5,
            atol=0.0,
        )
        # Integrate in time too
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "x"]),
            T_total * Lr,
            rtol=1.0e-5,
            atol=0.0,
        )
        # Cumulative integral in time
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "x"], cumulative_t=True),
            T_cumulative[:, np.newaxis, np.newaxis]
            * Lr
            * np.ones([ds.sizes["theta"], ds.sizes["zeta"]])[np.newaxis, :, :],
            rtol=1.0e-5,
            atol=0.0,
        )

        # Poloidal lines have length 2*pi*r
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["theta"]),
            (2.0 * np.pi * r)[np.newaxis, :, np.newaxis]
            * np.ones(ds.sizes["t"])[:, np.newaxis, np.newaxis]
            * np.ones(ds.sizes["zeta"])[np.newaxis, np.newaxis, :],
            rtol=1.0e-5,
            atol=0.0,
        )
        # Integrate in time too
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "theta"]),
            T_total
            * (2.0 * np.pi * r)[:, np.newaxis]
            * np.ones(ds.sizes["zeta"])[np.newaxis, :],
            rtol=1.0e-5,
            atol=0.0,
        )
        # Cumulative integral in time
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "theta"], cumulative_t=True),
            T_cumulative[:, np.newaxis, np.newaxis]
            * (2.0 * np.pi * r)[np.newaxis, :, np.newaxis]
            * np.ones(ds.sizes["zeta"])[np.newaxis, np.newaxis, :],
            rtol=1.0e-5,
            atol=0.0,
        )

        # Field line length
        # -----------------
        #
        # Field lines, parameterised by poloidal angle theta, have (R, Z, zeta)
        # coordinates
        # X = (R + r*cos(theta),
        #      -r*sin(theta),
        #      -q*2*atan(sqrt((1-r/R0)/(1+r/R0))*tan(theta/2))
        #     )
        # using d(arctan(x))/dx = 1/(1 + x**2) and d(tan(x))/dx = 1/cos(x)**2
        # dX/dtheta = (r*sin(theta),
        #              + r*cos(theta),
        #              - q * 2
        #                * sqrt((1-r/R0)/(1+r/R0))/(2*cos(theta/2)**2)
        #                / (1 + (1-r/R0)/(1+r/R0)*tan(theta/2)**2)
        #             )
        #           = (r*sin(theta),
        #              + r*cos(theta),
        #              - q
        #                * sqrt((1-r/R0)/(1+r/R0))/cos(theta/2)**2
        #                / (1 + (1-r/R0)/(1+r/R0)*tan(theta/2)**2)
        #             )
        #           = (r*sin(theta),
        #              + r*cos(theta),
        #              - q
        #                * sqrt((1-r/R0)/(1+r/R0))
        #                / (cos(theta/2)**2 + (1-r/R0)/(1+r/R0)*sin(theta/2)**2)
        #             )
        # Line element dl = |dR + dZ + R dzeta|
        #                 = |dR + dZ + (R0 + r cos(theta)) dzeta|
        #                 = |dX/dtheta|*dtheta
        # |dX/dtheta| = |
        #             = sqrt(
        #                r**2*sin(theta)**2
        #                + r**2*cos(theta)**2
        #                + (R0 + r cos(theta))**2 * q**2
        #                  * (1-r/R0)/(1+r/R0)
        #                  / (cos(theta/2)**2 + (1-r/R0)/(1+r/R0)*sin(theta/2)**2)**2
        #               )
        #             = sqrt(
        #                r**2
        #                + (R0 + r cos(theta))**2 * q**2
        #                  * (1-r/R0)/(1+r/R0)
        #                  / (cos(theta/2)**2 + (1-r/R0)/(1+r/R0)*sin(theta/2)**2)**2
        #               )
        # field line length is int_{0}^{2 pi} dl
        a = r**2
        c = (1 - r / R) / (1 + r / R)
        b = q**2 * c

        def func(theta):
            return np.sqrt(
                a
                + b
                * (R + r * np.cos(theta)) ** 2
                / (np.cos(theta / 2.0) ** 2 + c * np.sin(theta / 2.0) ** 2) ** 2
            )

        integral, _ = quad_vec(func, 0.0, 2.0 * np.pi)

        ds["n_aligned"] = ds["n"].bout.to_field_aligned()
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n_aligned", dims=["theta"]),
            integral[np.newaxis, :, np.newaxis]
            * np.ones(ds.sizes["t"])[:, np.newaxis, np.newaxis]
            * np.ones(ds.sizes["zeta"])[np.newaxis, np.newaxis, :],
            rtol=1.0e-5,
            atol=0.0,
        )
        # Integrate in time too
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n_aligned", dims=["t", "theta"]),
            T_total
            * integral[:, np.newaxis]
            * np.ones(ds.sizes["zeta"])[np.newaxis, :],
            rtol=1.0e-5,
            atol=0.0,
        )
        # Cumulative integral in time
        npt.assert_allclose(
            ds.bout.integrate_midpoints(
                "n_aligned", dims=["t", "theta"], cumulative_t=True
            ),
            T_cumulative[:, np.newaxis, np.newaxis]
            * integral[np.newaxis, :, np.newaxis]
            * np.ones(ds.sizes["zeta"])[np.newaxis, np.newaxis, :],
            rtol=1.0e-5,
            atol=0.0,
        )

        # Toroidal lines have length 2*pi*Rxy
        if location == "CELL_CENTRE":
            R_2d = ds["R"]
        else:
            R_2d = ds[f"Rxy_{location}"]
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["zeta"]),
            (2.0 * np.pi * R_2d).values[np.newaxis, :, :]
            * np.ones(ds.sizes["t"])[:, np.newaxis, np.newaxis],
            rtol=1.0e-5,
            atol=0.0,
        )
        # Integrate in time too
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "zeta"]),
            T_total * (2.0 * np.pi * R_2d),
            rtol=1.0e-5,
            atol=0.0,
        )
        # Cumulative integral in time
        npt.assert_allclose(
            ds.bout.integrate_midpoints("n", dims=["t", "zeta"], cumulative_t=True),
            T_cumulative[:, np.newaxis, np.newaxis]
            * (2.0 * np.pi * R_2d).values[np.newaxis, :, :],
            rtol=1.0e-5,
            atol=0.0,
        )

    def test_interpolate_from_unstructured(self, bout_xyt_example_files):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=6,
            nt=1,
            grid="grid",
            topology="upper-disconnected-double-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list, gridfilepath=grid_ds, geometry="toroidal"
        )

        # Set up non-trivial R and Z coordinates so we have something to interpolate to
        r = ds["x"] / ds["x"][-1] * 0.1 + 0.1
        theta = ds["theta"] / ds["theta"][-1] * 2.0 * np.pi

        ds["R"] = r * np.cos(theta)
        ds["Z"] = 2.0 * r * np.sin(theta)

        ds["n"] = (ds["R"] + ds["Z"]).broadcast_like(ds["n"])

        n = ds["n"]

        NR = 23
        R_rect = np.linspace(-0.1, 0.1, NR)

        NZ = 37
        Z_rect = np.linspace(-0.2, 0.2, NZ)

        n_rect = n.bout.interpolate_from_unstructured(R=R_rect, Z=Z_rect)

        # check BoutDataset and BoutDataArray versions are consistent
        n_rect_from_ds = ds.bout.interpolate_from_unstructured(..., R=R_rect, Z=Z_rect)[
            "n"
        ]
        npt.assert_allclose(n_rect, n_rect_from_ds)

        # check non-NaN values are correct
        n_check = R_rect[:, np.newaxis] + Z_rect[np.newaxis, :]
        n_rect = n_rect.isel(t=0, zeta=0).transpose("R", "Z")
        mask = ~np.isnan(n_rect.values)
        npt.assert_allclose(n_rect.values[mask], n_check[mask], atol=1.0e-7)

        # Check there were non-nan values to compare in the previous assert_allclose
        assert int((~np.isnan(n_rect)).count()) == 851

    def test_interpolate_from_unstructured_unstructured_output(
        self, bout_xyt_example_files
    ):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=6,
            nt=1,
            grid="grid",
            topology="lower-disconnected-double-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list, gridfilepath=grid_ds, geometry="toroidal"
        )

        # Set up non-trivial R and Z coordinates so we have something to interpolate to
        r = ds["x"] / ds["x"][-1] * 0.1 + 0.1
        theta = ds["theta"] / ds["theta"][-1] * 2.0 * np.pi

        ds["R"] = r * np.cos(theta)
        ds["Z"] = 2.0 * r * np.sin(theta)

        ds["n"] = (ds["R"] + ds["Z"]).broadcast_like(ds["n"])

        n = ds["n"]

        # make a set of points within the domain
        # domain is 'double null' so leave a gap to avoid the 'upper target' boundaries
        t = np.concatenate(
            [
                np.linspace(0.2, np.pi - 0.2, 11),
                np.linspace(np.pi + 0.2, 2.0 * np.pi - 0.2, 13),
            ]
        )
        R_unstruct = (0.142679 + 0.013291 * t / (2.0 * np.pi)) * np.cos(t)
        Z_unstruct = (0.240837 + 0.113408 * t / (2.0 * np.pi)) * np.sin(t)

        # check input validation
        with pytest.raises(ValueError):
            # input length mismatch
            n_unstruct = n.bout.interpolate_from_unstructured(
                R=R_unstruct, Z=Z_unstruct[:-2], structured_output=False
            )
        with pytest.raises(ValueError):
            # input length mismatch
            n_unstruct = n.bout.interpolate_from_unstructured(
                R=R_unstruct[:-2], Z=Z_unstruct, structured_output=False
            )

        n_unstruct = n.bout.interpolate_from_unstructured(
            R=R_unstruct, Z=Z_unstruct, structured_output=False
        )

        # check BoutDataset and BoutDataArray versions are consistent
        n_unstruct_from_ds = ds.bout.interpolate_from_unstructured(
            ..., R=R_unstruct, Z=Z_unstruct, structured_output=False
        )["n"]
        npt.assert_allclose(n_unstruct, n_unstruct_from_ds)

        # check non-NaN values are correct
        n_check = R_unstruct + Z_unstruct
        n_unstruct = n_unstruct.isel(t=0, zeta=0)
        npt.assert_allclose(n_unstruct.values, n_check, atol=1.0e-7)

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

        ds["n"].values[:] = 1.0

        nX = 30
        nY = 30
        nZ = 10
        ds_cartesian = ds.bout.interpolate_to_cartesian(nX, nY, nZ)

        # Check a point inside the original grid
        npt.assert_allclose(
            ds_cartesian["n"]
            .isel(t=0, X=round(nX * 4 / 5), Y=nY // 2, Z=nZ // 2)
            .item(),
            1.0,
            rtol=1.0e-15,
            atol=1.0e-15,
        )
        # Check a point outside the original grid
        assert np.isnan(ds_cartesian["n"].isel(t=0, X=0, Y=0, Z=0).item())
        # Check output is float32
        assert ds_cartesian["n"].dtype == np.float32

    def test_add_cartesian_coordinates(self, bout_xyt_example_files):
        dataset_list = bout_xyt_example_files(None, nxpe=1, nype=1, nt=1)
        with pytest.warns(UserWarning):
            ds = open_boutdataset(
                datapath=dataset_list, inputfilepath=None, keep_xboundaries=False
            )

        ds["psixy"] = ds["g11"].copy(deep=True)
        ds["Rxy"] = ds["g11"].copy(deep=True)
        ds["Zxy"] = ds["g11"].copy(deep=True)

        R0 = 3.0
        r = np.linspace(1.0, 2.0, ds.metadata["nx"])
        theta = np.linspace(0.0, 2.0 * np.pi, ds.metadata["ny"])
        R = R0 + r[:, np.newaxis] * np.cos(theta[np.newaxis, :])
        Z = r[:, np.newaxis] * np.sin(theta[np.newaxis, :])
        ds["Rxy"].values[:] = R
        ds["Zxy"].values[:] = Z

        ds = apply_geometry(ds, "toroidal")

        zeta = ds["zeta"].values

        ds = ds.bout.add_cartesian_coordinates()

        npt.assert_allclose(
            ds["X_cartesian"],
            R[:, :, np.newaxis] * np.cos(zeta[np.newaxis, np.newaxis, :]),
        )
        npt.assert_allclose(
            ds["Y_cartesian"],
            R[:, :, np.newaxis] * np.sin(zeta[np.newaxis, np.newaxis, :]),
        )
        npt.assert_allclose(
            ds["Z_cartesian"],
            Z[:, :, np.newaxis] * np.ones(ds.metadata["nz"])[np.newaxis, np.newaxis, :],
        )


class TestLoadInputFile:
    @pytest.mark.skip
    def test_load_options(self):
        from boutdata.data import BoutOptionsFile, BoutOptions

        options = BoutOptionsFile(EXAMPLE_OPTIONS_FILE_PATH)
        assert isinstance(options, BoutOptions)
        # TODO Check it contains the same text

    @pytest.mark.skip
    def test_load_options_in_dataset(self, bout_xyt_example_files):
        dataset_list = bout_xyt_example_files(None, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(
            datapath=dataset_list, inputfilepath=EXAMPLE_OPTIONS_FILE_PATH
        )
        assert isinstance(ds.options, BoutOptions)


@pytest.mark.skip(reason="Not yet implemented")
class TestLoadLogFile:
    pass


class TestSave:
    def test_save_all(self, tmp_path_factory, bout_xyt_example_files):
        # Create data
        path = bout_xyt_example_files(
            tmp_path_factory, nxpe=4, nype=5, nt=1, write_to_disk=True
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savedir = tmp_path_factory.mktemp("test_save_all")
        savepath = savedir.joinpath("temp_boutdata.nc")
        original.bout.save(savepath=savepath)

        # Load it again using bare xarray
        recovered = open_dataset(savepath)

        # Compare equal (not identical because attributes are changed when saving)
        xrt.assert_equal(original, recovered)

    @pytest.mark.parametrize("geometry", [None, "toroidal"])
    def test_reload_all(self, tmp_path_factory, bout_xyt_example_files, geometry):
        # Create data
        path = bout_xyt_example_files(
            tmp_path_factory, nxpe=4, nype=5, nt=1, grid="grid", write_to_disk=True
        )

        gridpath = path.parent.joinpath("grid.nc")

        # Load it as a boutdataset
        if geometry is None:
            with pytest.warns(UserWarning):
                original = open_boutdataset(
                    datapath=path,
                    inputfilepath=None,
                    geometry=geometry,
                    gridfilepath=None if geometry is None else gridpath,
                )
        else:
            original = open_boutdataset(
                datapath=path,
                inputfilepath=None,
                geometry=geometry,
                gridfilepath=None if geometry is None else gridpath,
            )

        # Save it to a netCDF file
        savedir = tmp_path_factory.mktemp("test_reload_all")
        savepath = savedir.joinpath("temp_boutdata.nc")
        original.bout.save(savepath=savepath)

        # Load it again
        recovered = open_boutdataset(savepath)

        xrt.assert_identical(original.load(), recovered.load())

        # Check if we can load with a different geometry argument
        for reload_geometry in [None, "toroidal"]:
            if reload_geometry is None or geometry == reload_geometry:
                recovered = open_boutdataset(
                    savepath,
                    geometry=reload_geometry,
                    gridfilepath=None if reload_geometry is None else gridpath,
                )
                xrt.assert_identical(original.load(), recovered.load())
            else:
                # Expect a warning because we change the geometry
                print("here", gridpath)
                with pytest.warns(UserWarning):
                    recovered = open_boutdataset(
                        savepath, geometry=reload_geometry, gridfilepath=gridpath
                    )
                # Datasets won't be exactly the same because different geometry was
                # applied

    @pytest.mark.parametrize("save_dtype", [np.float64, np.float32])
    @pytest.mark.parametrize(
        "separate_vars", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_save_dtype(
        self, tmp_path_factory, bout_xyt_example_files, save_dtype, separate_vars
    ):

        # Create data
        path = bout_xyt_example_files(
            tmp_path_factory, nxpe=1, nype=1, nt=1, write_to_disk=True
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savedir = tmp_path_factory.mktemp("test_save_dtype")
        savepath = savedir.joinpath("temp_boutdata.nc")
        original.bout.save(
            savepath=savepath, save_dtype=save_dtype, separate_vars=separate_vars
        )

        # Load it again using bare xarray
        if separate_vars:
            for v in ["n", "T"]:
                savepath = savedir.joinpath(f"temp_boutdata_{v}.nc")
                recovered = open_dataset(savepath)
                assert recovered[v].values.dtype == np.dtype(save_dtype)
        else:
            recovered = open_dataset(savepath)

            for v in original:
                assert recovered[v].values.dtype == np.dtype(save_dtype)

    def test_save_separate_variables(self, tmp_path_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmp_path_factory, nxpe=4, nype=1, nt=1, write_to_disk=True
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savedir = tmp_path_factory.mktemp("test_save_separate_variables")
        savepath = savedir.joinpath("temp_boutdata.nc")
        original.bout.save(savepath=savepath, separate_vars=True)

        for var in ["n", "T"]:
            # Load it again using bare xarray
            savepath = savedir.joinpath(f"temp_boutdata_{var}.nc")
            recovered = open_dataset(savepath)

            # Compare equal (not identical because attributes are changed when saving)
            xrt.assert_equal(recovered[var], original[var])

        # test open_boutdataset() on dataset saved with separate_vars=True
        savepath = savedir.joinpath("temp_boutdata_*.nc")
        recovered = open_boutdataset(savepath)
        xrt.assert_identical(original, recovered)

    @pytest.mark.parametrize("geometry", [None, "toroidal"])
    def test_reload_separate_variables(
        self, tmp_path_factory, bout_xyt_example_files, geometry
    ):
        if geometry is not None:
            grid = "grid"
        else:
            grid = None

        path = bout_xyt_example_files(
            tmp_path_factory, nxpe=4, nype=1, nt=1, grid=grid, write_to_disk=True
        )

        if grid is not None:
            gridpath = path.parent.joinpath("grid.nc")
        else:
            gridpath = None

        # Load it as a boutdataset
        if geometry is None:
            with pytest.warns(UserWarning):
                original = open_boutdataset(
                    datapath=path,
                    inputfilepath=None,
                    geometry=geometry,
                    gridfilepath=gridpath,
                )
        else:
            original = open_boutdataset(
                datapath=path,
                inputfilepath=None,
                geometry=geometry,
                gridfilepath=gridpath,
            )

        # Save it to a netCDF file
        savedir = tmp_path_factory.mktemp("test_reload_separate_variables")
        savepath = savedir.joinpath("temp_boutdata.nc")
        original.bout.save(savepath=savepath, separate_vars=True)

        # Load it again
        savepath = savedir.joinpath("temp_boutdata_*.nc")
        recovered = open_boutdataset(savepath)

        # Compare
        xrt.assert_identical(recovered, original)

    @pytest.mark.parametrize("geometry", [None, "toroidal"])
    def test_reload_separate_variables_time_split(
        self, tmp_path_factory, bout_xyt_example_files, geometry
    ):
        if geometry is not None:
            grid = "grid"
        else:
            grid = None

        path = bout_xyt_example_files(
            tmp_path_factory, nxpe=4, nype=1, nt=1, grid=grid, write_to_disk=True
        )

        if grid is not None:
            gridpath = path.parent.joinpath("grid.nc")
        else:
            gridpath = None

        # Load it as a boutdataset
        if geometry is None:
            with pytest.warns(UserWarning):
                original = open_boutdataset(
                    datapath=path,
                    inputfilepath=None,
                    geometry=geometry,
                    gridfilepath=gridpath,
                )
        else:
            original = open_boutdataset(
                datapath=path,
                inputfilepath=None,
                geometry=geometry,
                gridfilepath=gridpath,
            )

        # Save it to a netCDF file
        tcoord = original.metadata.get("bout_tdim", "t")
        savedir = tmp_path_factory.mktemp("test_reload_separate_variables_time_split")
        savepath = savedir.joinpath("temp_boutdata_1.nc")
        original.isel({tcoord: slice(3)}).bout.save(
            savepath=savepath, separate_vars=True
        )
        savepath = savedir.joinpath("temp_boutdata_2.nc")
        original.isel({tcoord: slice(3, None)}).bout.save(
            savepath=savepath, separate_vars=True
        )

        # Load it again
        savepath = savedir.joinpath("temp_boutdata_*.nc")
        recovered = open_boutdataset(savepath)

        # Compare
        xrt.assert_identical(recovered, original)


class TestSaveRestart:
    @pytest.mark.parametrize("tind", [None, pytest.param(1, marks=pytest.mark.long)])
    def test_to_restart(self, tmp_path_factory, bout_xyt_example_files, tind):
        nxpe = 3
        nype = 2

        nt = 6

        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=nxpe,
            nype=nype,
            nt=1,
            lengths=[nt, 4, 4, 7],
            guards={"x": 2, "y": 2},
            write_to_disk=True,
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(datapath=path)

        nx = ds.metadata["nx"]
        ny = ds.metadata["ny"]

        # Save it to a netCDF file
        savepath = tmp_path_factory.mktemp("test_to_restart")
        if tind is None:
            ds.bout.to_restart(savepath=savepath, nxpe=nxpe, nype=nype)
        else:
            ds.bout.to_restart(savepath=savepath, nxpe=nxpe, nype=nype, tind=tind)

        mxsub = (nx - 4) // nxpe
        mysub = ny // nype
        for proc_yind in range(nype):
            for proc_xind in range(nxpe):
                num = nxpe * proc_yind + proc_xind

                restart_ds = open_dataset(savepath.joinpath(f"BOUT.restart.{num}.nc"))

                if tind is None:
                    t = -1
                else:
                    t = tind

                # ignore guard cells - they are filled with NaN in the created restart
                # files
                restart_ds = restart_ds.isel(x=slice(2, -2), y=slice(2, -2))

                check_ds = ds.isel(
                    t=t,
                    x=slice(2 + proc_xind * mxsub, 2 + (proc_xind + 1) * mxsub),
                    y=slice(proc_yind * mysub, (proc_yind + 1) * mysub),
                ).load()
                t_array = check_ds["t"]
                check_ds = check_ds.drop_vars(["t", "x", "y", "z"])

                # No coordinates saved in restart files, so unset them in check_ds
                check_ds = check_ds.reset_coords()

                # ignore variables that depend on the rank of the BOUT++ process - they
                # cannot be consistent with check_ds
                rank_dependent_vars = ["PE_XIND", "PE_YIND", "MYPE"]
                for v in restart_ds:
                    if v in check_ds:
                        xrt.assert_equal(restart_ds[v], check_ds[v])
                    else:
                        if v == "hist_hi":
                            if t >= 0:
                                assert restart_ds[v].values == t
                            else:
                                assert restart_ds[v].values == nt - 1
                        elif v == "tt":
                            assert restart_ds[v].values == t_array
                        elif v not in rank_dependent_vars:
                            assert restart_ds[v].values == check_ds.metadata[v]

    def test_to_restart_change_npe(self, tmp_path_factory, bout_xyt_example_files):
        nxpe_in = 3
        nype_in = 2

        nxpe = 2
        nype = 4

        nt = 6

        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=nxpe_in,
            nype=nype_in,
            nt=1,
            lengths=[nt, 4, 4, 7],
            guards={"x": 2, "y": 2},
            write_to_disk=True,
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(datapath=path)

        nx = ds.metadata["nx"]
        ny = ds.metadata["ny"]

        # Save it to a netCDF file
        savepath = tmp_path_factory.mktemp("test_to_restart_change_npe")
        ds.bout.to_restart(savepath=savepath, nxpe=nxpe, nype=nype)

        mxsub = (nx - 4) // nxpe
        mysub = ny // nype
        for proc_yind in range(nype):
            for proc_xind in range(nxpe):
                num = nxpe * proc_yind + proc_xind

                restart_ds = open_dataset(savepath.joinpath(f"BOUT.restart.{num}.nc"))

                # ignore guard cells - they are filled with NaN in the created restart
                # files
                restart_ds = restart_ds.isel(x=slice(2, -2), y=slice(2, -2))

                check_ds = ds.isel(
                    t=-1,
                    x=slice(2 + proc_xind * mxsub, 2 + (proc_xind + 1) * mxsub),
                    y=slice(proc_yind * mysub, (proc_yind + 1) * mysub),
                ).load()
                t_array = check_ds["t"]
                check_ds = check_ds.drop_vars(["t", "x", "y", "z"])

                # No coordinates saved in restart files, so unset them in check_ds
                check_ds = check_ds.reset_coords()

                # ignore variables that depend on the rank of the BOUT++ process - they
                # cannot be consistent with check_ds
                rank_dependent_vars = ["PE_XIND", "PE_YIND", "MYPE"]
                for v in restart_ds:
                    if v in check_ds:
                        xrt.assert_equal(restart_ds[v], check_ds[v])
                    else:
                        if v in ["NXPE", "NYPE", "MXSUB", "MYSUB"]:
                            pass
                        elif v == "hist_hi":
                            assert restart_ds[v].values == nt - 1
                        elif v == "tt":
                            assert restart_ds[v].values == t_array
                        elif v not in rank_dependent_vars:
                            assert restart_ds[v].values == check_ds.metadata[v]

    @pytest.mark.long
    def test_to_restart_change_npe_doublenull(
        self, tmp_path_factory, bout_xyt_example_files
    ):
        nxpe_in = 3
        nype_in = 6

        nxpe = 1
        nype = 12

        nt = 6

        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=nxpe_in,
            nype=nype_in,
            nt=1,
            guards={"x": 2, "y": 2},
            lengths=(nt, 5, 4, 7),
            topology="upper-disconnected-double-null",
            write_to_disk=True,
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(datapath=path)

        nx = ds.metadata["nx"]
        ny = ds.metadata["ny"]

        # Save it to a netCDF file
        savepath = tmp_path_factory.mktemp("test_to_restart_change_npe_doublenull")
        ds.bout.to_restart(savepath=savepath, nxpe=nxpe, nype=nype)

        mxsub = (nx - 4) // nxpe
        mysub = ny // nype
        for proc_yind in range(nype):
            for proc_xind in range(nxpe):
                num = nxpe * proc_yind + proc_xind

                restart_ds = open_dataset(savepath.joinpath(f"BOUT.restart.{num}.nc"))

                # ignore guard cells - they are filled with NaN in the created restart
                # files
                restart_ds = restart_ds.isel(x=slice(2, -2), y=slice(2, -2))

                check_ds = ds.isel(
                    t=-1,
                    x=slice(2 + proc_xind * mxsub, 2 + (proc_xind + 1) * mxsub),
                    y=slice(proc_yind * mysub, (proc_yind + 1) * mysub),
                ).load()
                t_array = check_ds["t"]
                check_ds = check_ds.drop_vars(["t", "x", "y", "z"])

                # No coordinates saved in restart files, so unset them in check_ds
                check_ds = check_ds.reset_coords()

                # ignore variables that depend on the rank of the BOUT++ process - they
                # cannot be consistent with check_ds
                rank_dependent_vars = ["PE_XIND", "PE_YIND", "MYPE"]
                for v in restart_ds:
                    if v in check_ds:
                        xrt.assert_equal(restart_ds[v], check_ds[v])
                    else:
                        if v in ["NXPE", "NYPE", "MXSUB", "MYSUB"]:
                            pass
                        elif v == "hist_hi":
                            assert restart_ds[v].values == nt - 1
                        elif v == "tt":
                            assert restart_ds[v].values == t_array
                        elif v not in rank_dependent_vars:
                            assert restart_ds[v].values == check_ds.metadata[v]

    @pytest.mark.long
    @pytest.mark.parametrize("npes", [(2, 6), (3, 4)])
    def test_to_restart_change_npe_doublenull_expect_fail(
        self, tmp_path_factory, bout_xyt_example_files, npes
    ):
        nxpe_in = 3
        nype_in = 6

        nxpe, nype = npes

        path = bout_xyt_example_files(
            tmp_path_factory,
            nxpe=nxpe_in,
            nype=nype_in,
            nt=1,
            guards={"x": 2, "y": 2},
            lengths=(6, 5, 4, 7),
            topology="lower-disconnected-double-null",
            write_to_disk=True,
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(datapath=path)

        nx = ds.metadata["nx"]
        ny = ds.metadata["ny"]

        # Save it to a netCDF file
        savepath = tmp_path_factory.mktemp(
            "test_to_restart_change_npe_doublenull_expect_fail"
        )
        with pytest.raises(ValueError):
            ds.bout.to_restart(savepath=savepath, nxpe=nxpe, nype=nype)

    def test_from_restart_to_restart(self, tmp_path):
        datapath = Path(__file__).parent.joinpath(
            "data", "restart", "BOUT.restart.*.nc"
        )
        ds = open_boutdataset(datapath, keep_xboundaries=True, keep_yboundaries=True)

        ds.bout.to_restart(savepath=tmp_path, nxpe=1, nype=4)
