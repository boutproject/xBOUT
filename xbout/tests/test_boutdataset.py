import pytest

import numpy.testing as npt
from xarray import Dataset, DataArray, concat, open_dataset, open_mfdataset
import xarray.testing as xrt
import numpy as np
from pathlib import Path

from xbout.tests.test_load import bout_xyt_example_files, create_bout_ds
from xbout.tests.test_region import (
    params_guards,
    params_guards_values,
    params_boundaries,
    params_boundaries_values,
)
from xbout import BoutDatasetAccessor, open_boutdataset
from xbout.geometries import apply_geometry
from xbout.utils import _set_attrs_on_all_vars


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
    @pytest.mark.skip
    def test_test_method(self, bout_xyt_example_files):
        dataset_list = bout_xyt_example_files(None, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=dataset_list, inputfilepath=None)
        # ds = collect(path=path)
        # bd = BoutAccessor(ds)
        print(ds)
        # ds.bout.test_method()
        # print(ds.bout.options)
        # print(ds.bout.metadata)
        print(ds.isel(t=-1))

        # ds.bout.set_extra_data('stored')
        ds.bout.extra_data = "stored"

        print(ds.bout.extra_data)

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
            topology="disconnected-double-null",
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
                "dx",
                "dy",
            )
        )

    def test_interpolate_from_unstructured(self, bout_xyt_example_files):
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=6,
            nt=1,
            grid="grid",
            topology="disconnected-double-null",
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
            topology="disconnected-double-null",
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
    def test_save_all(self, tmpdir_factory, bout_xyt_example_files):
        # Create data
        path = bout_xyt_example_files(
            tmpdir_factory, nxpe=4, nype=5, nt=1, write_to_disk=True
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + "temp_boutdata.nc"
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
        path = bout_xyt_example_files(
            tmpdir_factory, nxpe=4, nype=5, nt=1, grid=grid, write_to_disk=True
        )

        if grid is not None:
            gridpath = str(Path(path).parent) + "/grid.nc"
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
        savepath = str(Path(path).parent) + "/temp_boutdata.nc"
        original.bout.save(savepath=savepath)

        # Load it again
        recovered = open_boutdataset(savepath)

        xrt.assert_identical(original.load(), recovered.load())

    @pytest.mark.parametrize("save_dtype", [np.float64, np.float32])
    @pytest.mark.parametrize(
        "separate_vars", [False, pytest.param(True, marks=pytest.mark.long)]
    )
    def test_save_dtype(
        self, tmpdir_factory, bout_xyt_example_files, save_dtype, separate_vars
    ):

        # Create data
        path = bout_xyt_example_files(
            tmpdir_factory, nxpe=1, nype=1, nt=1, write_to_disk=True
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + "/temp_boutdata.nc"
        original.bout.save(
            savepath=savepath, save_dtype=save_dtype, separate_vars=separate_vars
        )

        # Load it again using bare xarray
        if separate_vars:
            for v in ["n", "T"]:
                savepath = str(Path(path).parent) + f"/temp_boutdata_{v}.nc"
                recovered = open_dataset(savepath)
                assert recovered[v].values.dtype == np.dtype(save_dtype)
        else:
            recovered = open_dataset(savepath)

            for v in original:
                assert recovered[v].values.dtype == np.dtype(save_dtype)

    def test_save_separate_variables(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(
            tmpdir_factory, nxpe=4, nype=1, nt=1, write_to_disk=True
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + "/temp_boutdata.nc"
        original.bout.save(savepath=savepath, separate_vars=True)

        for var in ["n", "T"]:
            # Load it again using bare xarray
            savepath = str(Path(path).parent) + "/temp_boutdata_" + var + ".nc"
            recovered = open_dataset(savepath)

            # Compare equal (not identical because attributes are changed when saving)
            xrt.assert_equal(recovered[var], original[var])

        # test open_boutdataset() on dataset saved with separate_vars=True
        savepath = str(Path(path).parent) + "/temp_boutdata_*.nc"
        recovered = open_boutdataset(savepath)
        xrt.assert_identical(original, recovered)

    @pytest.mark.parametrize("geometry", [None, "toroidal"])
    def test_reload_separate_variables(
        self, tmpdir_factory, bout_xyt_example_files, geometry
    ):
        if geometry is not None:
            grid = "grid"
        else:
            grid = None

        path = bout_xyt_example_files(
            tmpdir_factory, nxpe=4, nype=1, nt=1, grid=grid, write_to_disk=True
        )

        if grid is not None:
            gridpath = str(Path(path).parent) + "/grid.nc"
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
        savepath = str(Path(path).parent) + "/temp_boutdata.nc"
        original.bout.save(savepath=savepath, separate_vars=True)

        # Load it again
        savepath = str(Path(path).parent) + "/temp_boutdata_*.nc"
        recovered = open_boutdataset(savepath)

        # Compare
        xrt.assert_identical(recovered, original)

    @pytest.mark.parametrize("geometry", [None, "toroidal"])
    def test_reload_separate_variables_time_split(
        self, tmpdir_factory, bout_xyt_example_files, geometry
    ):
        if geometry is not None:
            grid = "grid"
        else:
            grid = None

        path = bout_xyt_example_files(
            tmpdir_factory, nxpe=4, nype=1, nt=1, grid=grid, write_to_disk=True
        )

        if grid is not None:
            gridpath = str(Path(path).parent) + "/grid.nc"
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
        savepath = str(Path(path).parent) + "/temp_boutdata_1.nc"
        original.isel({tcoord: slice(3)}).bout.save(
            savepath=savepath, separate_vars=True
        )
        savepath = str(Path(path).parent) + "/temp_boutdata_2.nc"
        original.isel({tcoord: slice(3, None)}).bout.save(
            savepath=savepath, separate_vars=True
        )

        # Load it again
        savepath = str(Path(path).parent) + "/temp_boutdata_*.nc"
        recovered = open_boutdataset(savepath)

        # Compare
        xrt.assert_identical(recovered, original)


class TestSaveRestart:
    @pytest.mark.parametrize("tind", [None, pytest.param(1, marks=pytest.mark.long)])
    def test_to_restart(self, tmpdir_factory, bout_xyt_example_files, tind):
        nxpe = 3
        nype = 2

        path = bout_xyt_example_files(
            tmpdir_factory,
            nxpe=nxpe,
            nype=nype,
            nt=1,
            lengths=[6, 4, 4, 7],
            guards={"x": 2, "y": 2},
            write_to_disk=True,
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(datapath=path)

        nx = ds.metadata["nx"]
        ny = ds.metadata["ny"]

        # Save it to a netCDF file
        savepath = Path(path).parent
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

                for v in restart_ds:
                    if v in check_ds:
                        xrt.assert_equal(restart_ds[v], check_ds[v])
                    else:
                        if v == "hist_hi":
                            assert restart_ds[v].values == -1
                        elif v == "tt":
                            assert restart_ds[v].values == t_array
                        else:
                            assert restart_ds[v].values == check_ds.metadata[v]

    def test_to_restart_change_npe(self, tmpdir_factory, bout_xyt_example_files):
        nxpe_in = 3
        nype_in = 2

        nxpe = 2
        nype = 4

        path = bout_xyt_example_files(
            tmpdir_factory,
            nxpe=nxpe_in,
            nype=nype_in,
            nt=1,
            lengths=[6, 4, 4, 7],
            guards={"x": 2, "y": 2},
            write_to_disk=True,
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(datapath=path)

        nx = ds.metadata["nx"]
        ny = ds.metadata["ny"]

        # Save it to a netCDF file
        savepath = Path(path).parent
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

                for v in restart_ds:
                    if v in check_ds:
                        xrt.assert_equal(restart_ds[v], check_ds[v])
                    else:
                        if v in ["NXPE", "NYPE", "MXSUB", "MYSUB"]:
                            pass
                        elif v == "hist_hi":
                            assert restart_ds[v].values == -1
                        elif v == "tt":
                            assert restart_ds[v].values == t_array
                        else:
                            assert restart_ds[v].values == check_ds.metadata[v]

    @pytest.mark.long
    def test_to_restart_change_npe_doublenull(
        self, tmpdir_factory, bout_xyt_example_files
    ):
        nxpe_in = 3
        nype_in = 6

        nxpe = 1
        nype = 12

        path = bout_xyt_example_files(
            tmpdir_factory,
            nxpe=nxpe_in,
            nype=nype_in,
            nt=1,
            guards={"x": 2, "y": 2},
            lengths=(6, 5, 4, 7),
            topology="disconnected-double-null",
            write_to_disk=True,
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(datapath=path)

        nx = ds.metadata["nx"]
        ny = ds.metadata["ny"]

        # Save it to a netCDF file
        savepath = Path(path).parent
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

                for v in restart_ds:
                    if v in check_ds:
                        xrt.assert_equal(restart_ds[v], check_ds[v])
                    else:
                        if v in ["NXPE", "NYPE", "MXSUB", "MYSUB"]:
                            pass
                        elif v == "hist_hi":
                            assert restart_ds[v].values == -1
                        elif v == "tt":
                            assert restart_ds[v].values == t_array
                        else:
                            assert restart_ds[v].values == check_ds.metadata[v]

    @pytest.mark.long
    @pytest.mark.parametrize("npes", [(2, 6), (3, 4)])
    def test_to_restart_change_npe_doublenull_expect_fail(
        self, tmpdir_factory, bout_xyt_example_files, npes
    ):
        nxpe_in = 3
        nype_in = 6

        nxpe, nype = npes

        path = bout_xyt_example_files(
            tmpdir_factory,
            nxpe=nxpe_in,
            nype=nype_in,
            nt=1,
            guards={"x": 2, "y": 2},
            lengths=(6, 5, 4, 7),
            topology="disconnected-double-null",
            write_to_disk=True,
        )

        # Load it as a boutdataset
        with pytest.warns(UserWarning):
            ds = open_boutdataset(datapath=path)

        nx = ds.metadata["nx"]
        ny = ds.metadata["ny"]

        # Save it to a netCDF file
        savepath = Path(path).parent
        with pytest.raises(ValueError):
            ds.bout.to_restart(savepath=savepath, nxpe=nxpe, nype=nype)
