import pytest

from pathlib import Path

import numpy.testing as npt
import xarray.testing as xrt

from xbout.tests.test_load import bout_xyt_example_files
from xbout import open_boutdataset


params_guards = "guards"
params_guards_values = [
    pytest.param({"x": 0, "y": 0}, marks=pytest.mark.long),
    pytest.param({"x": 2, "y": 0}, marks=pytest.mark.long),
    pytest.param({"x": 0, "y": 2}, marks=pytest.mark.long),
    {"x": 2, "y": 2},
]
params_boundaries = "keep_xboundaries, keep_yboundaries"
params_boundaries_values = [
    pytest.param(False, False, marks=pytest.mark.long),
    pytest.param(True, False, marks=pytest.mark.long),
    pytest.param(False, True, marks=pytest.mark.long),
    (True, True),
]


class TestRegion:
    @pytest.mark.long
    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    def test_region_core(
        self, bout_xyt_example_files, guards, keep_xboundaries, keep_yboundaries
    ):
        # Note need to use more than (3*MXG,3*MYG) points per output file
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=4,
            nt=1,
            guards=guards,
            grid="grid",
            topology="core",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
        )

        n = ds["n"]

        n_core = n.bout.from_region("core")

        # Remove attributes that are expected to be different
        del n_core.attrs["regions"]
        del n.attrs["regions"]

        # Select only non-boundary data
        if keep_yboundaries:
            ybndry = guards["y"]
        else:
            ybndry = 0
        xrt.assert_identical(
            n.isel(theta=slice(ybndry, -ybndry if ybndry != 0 else None)),
            n_core.isel(
                theta=slice(guards["y"], -guards["y"] if guards["y"] != 0 else None)
            ),
        )

    @pytest.mark.long
    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    def test_region_sol(
        self, bout_xyt_example_files, guards, keep_xboundaries, keep_yboundaries
    ):
        # Note need to use more than (3*MXG,3*MYG) points per output file
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=4,
            nt=1,
            guards=guards,
            grid="grid",
            topology="sol",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
        )

        n = ds["n"]

        n_sol = n.bout.from_region("SOL")

        # Remove attributes that are expected to be different
        del n_sol.attrs["regions"]
        del n.attrs["regions"]

        xrt.assert_identical(n, n_sol)

    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    @pytest.mark.parametrize("region_guards", [0, 1])
    def test_region_limiter(
        self,
        bout_xyt_example_files,
        guards,
        keep_xboundaries,
        keep_yboundaries,
        region_guards,
    ):
        # Note using more than MXG x-direction points and MYG y-direction points per
        # output file ensures tests for whether boundary cells are present do not fail
        # when using minimal numbers of processors
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=4,
            nt=1,
            guards=guards,
            grid="grid",
            topology="limiter",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
        )

        mxg = guards["x"]

        if keep_xboundaries:
            ixs = ds.metadata["ixseps1"]
        else:
            ixs = ds.metadata["ixseps1"] - guards["x"]

        # For selecting only non-boundary data
        if keep_yboundaries:
            ybndry = guards["y"]
        else:
            ybndry = 0

        n = ds["n"]
        n_noregions = n.copy(deep=False)

        # Remove attributes that are expected to be different
        del n_noregions.attrs["regions"]

        n_sol = n.bout.from_region("SOL", with_guards=region_guards)

        # Remove attributes that are expected to be different
        # Corners may be different because core region 'communicates' in y
        del n_sol.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs, None)),
            n_sol.isel(x=slice(region_guards, None)),
        )
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs - region_guards, ixs),
                theta=slice(ybndry, -ybndry if ybndry != 0 else None),
            ),
            n_sol.isel(
                x=slice(region_guards),
                theta=slice(ybndry, -ybndry if ybndry != 0 else None),
            ),
        )

        n_core = n.bout.from_region("core", with_guards=region_guards)

        # Remove attributes that are expected to be different
        del n_core.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs + region_guards),
                theta=slice(ybndry, -ybndry if ybndry != 0 else None),
            ),
            n_core.isel(
                theta=slice(
                    region_guards, -region_guards if region_guards != 0 else None
                )
            ),
        )

    @pytest.mark.long
    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    def test_region_xpoint(
        self, bout_xyt_example_files, guards, keep_xboundaries, keep_yboundaries
    ):
        # Note using more than MXG x-direction points and MYG y-direction points per
        # output file ensures tests for whether boundary cells are present do not fail
        # when using minimal numbers of processors
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=4,
            nt=1,
            guards=guards,
            grid="grid",
            topology="xpoint",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
        )

        mxg = guards["x"]
        myg = guards["y"]

        if keep_xboundaries:
            ixs = ds.metadata["ixseps1"]
        else:
            ixs = ds.metadata["ixseps1"] - guards["x"]

        if keep_yboundaries:
            ybndry = guards["y"]
        else:
            ybndry = 0
        jys1 = ds.metadata["jyseps1_1"] + ybndry
        ny_inner = ds.metadata["ny_inner"] + 2 * ybndry
        jys2 = ds.metadata["jyseps2_2"] + 3 * ybndry
        ny = ds.metadata["ny"] + 4 * ybndry

        n = ds["n"]
        n_noregions = n.copy(deep=True)

        # Remove attributes that are expected to be different
        del n_noregions.attrs["regions"]

        n_lower_inner_PFR = n.bout.from_region("lower_inner_PFR")

        # Remove attributes that are expected to be different
        del n_lower_inner_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys1 + 1)),
            n_lower_inner_PFR.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys2 + 1, jys2 + 1 + myg)
                ).values,
                n_lower_inner_PFR.isel(theta=slice(-myg, None)).values,
            )

        n_lower_inner_SOL = n.bout.from_region("lower_inner_SOL")

        # Remove attributes that are expected to be different
        del n_lower_inner_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs - mxg, None), theta=slice(jys1 + 1)),
            n_lower_inner_SOL.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys1 + 1, jys1 + 1 + myg)
                ).values,
                n_lower_inner_SOL.isel(theta=slice(-myg, None)).values,
            )

        n_upper_inner_PFR = n.bout.from_region("upper_inner_PFR")

        # Remove attributes that are expected to be different
        del n_upper_inner_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys1 + 1, ny_inner)),
            n_upper_inner_PFR.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys2 + 1 - myg, jys2 + 1)
                ).values,
                n_upper_inner_PFR.isel(theta=slice(myg)).values,
            )

        n_upper_inner_SOL = n.bout.from_region("upper_inner_SOL")

        # Remove attributes that are expected to be different
        del n_upper_inner_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs - mxg, None), theta=slice(jys1 + 1, ny_inner)),
            n_upper_inner_SOL.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys1 + 1 - myg, jys1 + 1)
                ).values,
                n_upper_inner_SOL.isel(theta=slice(myg)).values,
            )

        n_upper_outer_PFR = n.bout.from_region("upper_outer_PFR")

        # Remove attributes that are expected to be different
        del n_upper_outer_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(ny_inner, jys2 + 1)),
            n_upper_outer_PFR.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys1 + 1, jys1 + 1 + myg)
                ).values,
                n_upper_outer_PFR.isel(theta=slice(-myg, None)).values,
            )

        n_upper_outer_SOL = n.bout.from_region("upper_outer_SOL")

        # Remove attributes that are expected to be different
        del n_upper_outer_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs - mxg, None), theta=slice(ny_inner, jys2 + 1)),
            n_upper_outer_SOL.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys2 + 1, jys2 + 1 + myg)
                ).values,
                n_upper_outer_SOL.isel(theta=slice(-myg, None)).values,
            )

        n_lower_outer_PFR = n.bout.from_region("lower_outer_PFR")

        # Remove attributes that are expected to be different
        del n_lower_outer_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys2 + 1, None)),
            n_lower_outer_PFR.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys1 + 1 - myg, jys1 + 1)
                ).values,
                n_lower_outer_PFR.isel(theta=slice(myg)).values,
            )

        n_lower_outer_SOL = n.bout.from_region("lower_outer_SOL")

        # Remove attributes that are expected to be different
        del n_lower_outer_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs - mxg, None), theta=slice(jys2 + 1, None)),
            n_lower_outer_SOL.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys2 + 1 - myg, jys2 + 1)
                ).values,
                n_lower_outer_SOL.isel(theta=slice(myg)).values,
            )

    @pytest.mark.long
    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    def test_region_singlenull(
        self, bout_xyt_example_files, guards, keep_xboundaries, keep_yboundaries
    ):
        # Note using more than MXG x-direction points and MYG y-direction points per
        # output file ensures tests for whether boundary cells are present do not fail
        # when using minimal numbers of processors
        dataset_list, grid_ds = bout_xyt_example_files(
            None,
            lengths=(2, 3, 4, 3),
            nxpe=3,
            nype=4,
            nt=1,
            guards=guards,
            grid="grid",
            topology="single-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
        )

        mxg = guards["x"]
        myg = guards["y"]

        if keep_xboundaries:
            ixs = ds.metadata["ixseps1"]
        else:
            ixs = ds.metadata["ixseps1"] - guards["x"]

        if keep_yboundaries:
            ybndry = guards["y"]
        else:
            ybndry = 0
        jys1 = ds.metadata["jyseps1_1"] + ybndry
        jys2 = ds.metadata["jyseps2_2"] + ybndry
        ny = ds.metadata["ny"] + 2 * ybndry

        n = ds["n"]
        n_noregions = n.copy(deep=True)

        # Remove attributes that are expected to be different
        del n_noregions.attrs["regions"]

        n_inner_PFR = n.bout.from_region("inner_PFR")

        # Remove attributes that are expected to be different
        del n_inner_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys1 + 1)),
            n_inner_PFR.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys2 + 1, jys2 + 1 + myg)
                ).values,
                n_inner_PFR.isel(theta=slice(-myg, None)).values,
            )

        n_inner_SOL = n.bout.from_region("inner_SOL")

        # Remove attributes that are expected to be different
        del n_inner_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs - mxg, None), theta=slice(jys1 + 1)),
            n_inner_SOL.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys1 + 1, jys1 + 1 + myg)
                ).values,
                n_inner_SOL.isel(theta=slice(-myg, None)).values,
            )

        n_core = n.bout.from_region("core")

        # Remove attributes that are expected to be different
        del n_core.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys1 + 1, jys2 + 1)),
            n_core.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys2 + 1 - myg, jys2 + 1)
                ).values,
                n_core.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys1 + 1, jys1 + 1 + myg)
                ).values,
                n_core.isel(theta=slice(-myg, None)).values,
            )

        n_sol = n.bout.from_region("SOL")

        # Remove attributes that are expected to be different
        del n_sol.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs - mxg, None), theta=slice(jys1 + 1, jys2 + 1)),
            n_sol.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys1 + 1 - myg, jys1 + 1)
                ).values,
                n_sol.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys2 + 1, jys2 + 1 + myg)
                ).values,
                n_sol.isel(theta=slice(-myg, None)).values,
            )

        n_outer_PFR = n.bout.from_region("outer_PFR")

        # Remove attributes that are expected to be different
        del n_outer_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys2 + 1, None)),
            n_outer_PFR.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys1 + 1 - myg, jys1 + 1)
                ).values,
                n_outer_PFR.isel(theta=slice(myg)).values,
            )

        n_outer_SOL = n.bout.from_region("outer_SOL")

        # Remove attributes that are expected to be different
        del n_outer_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs - mxg, None), theta=slice(jys2 + 1, None)),
            n_outer_SOL.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys2 + 1 - myg, jys2 + 1)
                ).values,
                n_outer_SOL.isel(theta=slice(myg)).values,
            )

    @pytest.mark.long
    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    def test_region_connecteddoublenull(
        self, bout_xyt_example_files, guards, keep_xboundaries, keep_yboundaries
    ):
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
            topology="connected-double-null",
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
        )

        mxg = guards["x"]
        myg = guards["y"]

        if keep_xboundaries:
            ixs = ds.metadata["ixseps1"]
        else:
            ixs = ds.metadata["ixseps1"] - guards["x"]

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

        n = ds["n"]
        n_noregions = n.copy(deep=True)

        # Remove attributes that are expected to be different
        del n_noregions.attrs["regions"]

        n_lower_inner_PFR = n.bout.from_region("lower_inner_PFR")

        # Remove attributes that are expected to be different
        del n_lower_inner_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys11 + 1)),
            n_lower_inner_PFR.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys22 + 1, jys22 + 1 + myg)
                ).values,
                n_lower_inner_PFR.isel(theta=slice(-myg, None)).values,
            )

        n_lower_inner_SOL = n.bout.from_region("lower_inner_SOL")

        # Remove attributes that are expected to be different
        del n_lower_inner_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs - mxg, None), theta=slice(jys11 + 1)),
            n_lower_inner_SOL.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys11 + 1, jys11 + 1 + myg)
                ).values,
                n_lower_inner_SOL.isel(theta=slice(-myg, None)).values,
            )

        n_inner_core = n.bout.from_region("inner_core")

        # Remove attributes that are expected to be different
        del n_inner_core.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys11 + 1, jys21 + 1)),
            n_inner_core.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys22 + 1 - myg, jys22 + 1)
                ).values,
                n_inner_core.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys12 + 1, jys12 + 1 + myg)
                ).values,
                n_inner_core.isel(theta=slice(-myg, None)).values,
            )

        n_inner_sol = n.bout.from_region("inner_SOL")

        # Remove attributes that are expected to be different
        del n_inner_sol.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs - mxg, None), theta=slice(jys11 + 1, jys21 + 1)
            ),
            n_inner_sol.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys11 + 1 - myg, jys11 + 1)
                ).values,
                n_inner_sol.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys21 + 1, jys21 + 1 + myg)
                ).values,
                n_inner_sol.isel(theta=slice(-myg, None)).values,
            )

        n_upper_inner_PFR = n.bout.from_region("upper_inner_PFR")

        # Remove attributes that are expected to be different
        del n_upper_inner_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys21 + 1, ny_inner)),
            n_upper_inner_PFR.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys12 + 1 - myg, jys12 + 1)
                ).values,
                n_upper_inner_PFR.isel(theta=slice(myg)).values,
            )

        n_upper_inner_SOL = n.bout.from_region("upper_inner_SOL")

        # Remove attributes that are expected to be different
        del n_upper_inner_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs - mxg, None), theta=slice(jys21 + 1, ny_inner)
            ),
            n_upper_inner_SOL.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys21 + 1 - myg, jys21 + 1)
                ).values,
                n_upper_inner_SOL.isel(theta=slice(myg)).values,
            )

        n_upper_outer_PFR = n.bout.from_region("upper_outer_PFR")

        # Remove attributes that are expected to be different
        del n_upper_outer_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(ny_inner, jys12 + 1)),
            n_upper_outer_PFR.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys21 + 1, jys21 + 1 + myg)
                ).values,
                n_upper_outer_PFR.isel(theta=slice(-myg, None)).values,
            )

        n_upper_outer_SOL = n.bout.from_region("upper_outer_SOL")

        # Remove attributes that are expected to be different
        del n_upper_outer_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs - mxg, None), theta=slice(ny_inner, jys12 + 1)
            ),
            n_upper_outer_SOL.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys12 + 1, jys12 + 1 + myg)
                ).values,
                n_upper_outer_SOL.isel(theta=slice(-myg, None)).values,
            )

        n_outer_core = n.bout.from_region("outer_core")

        # Remove attributes that are expected to be different
        del n_outer_core.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys12 + 1, jys22 + 1)),
            n_outer_core.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys21 + 1 - myg, jys21 + 1)
                ).values,
                n_outer_core.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys11 + 1, jys11 + 1 + myg)
                ).values,
                n_outer_core.isel(theta=slice(-myg, None)).values,
            )

        n_outer_sol = n.bout.from_region("outer_SOL")

        # Remove attributes that are expected to be different
        del n_outer_sol.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs - mxg, None), theta=slice(jys12 + 1, jys22 + 1)
            ),
            n_outer_sol.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys12 + 1 - myg, jys12 + 1)
                ).values,
                n_outer_sol.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys22 + 1, jys22 + 1 + myg)
                ).values,
                n_outer_sol.isel(theta=slice(-myg, None)).values,
            )

        n_lower_outer_PFR = n.bout.from_region("lower_outer_PFR")

        # Remove attributes that are expected to be different
        del n_lower_outer_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs + mxg), theta=slice(jys22 + 1, None)),
            n_lower_outer_PFR.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs + mxg), theta=slice(jys11 + 1 - myg, jys11 + 1)
                ).values,
                n_lower_outer_PFR.isel(theta=slice(myg)).values,
            )

        n_lower_outer_SOL = n.bout.from_region("lower_outer_SOL")

        # Remove attributes that are expected to be different
        del n_lower_outer_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs - mxg, None), theta=slice(jys22 + 1, None)),
            n_lower_outer_SOL.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs - mxg, None), theta=slice(jys22 + 1 - myg, jys22 + 1)
                ).values,
                n_lower_outer_SOL.isel(theta=slice(myg)).values,
            )

    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    @pytest.mark.parametrize(
        "dnd_type", ["lower-disconnected-double-null", "upper-disconnected-double-null"]
    )
    def test_region_disconnecteddoublenull(
        self,
        bout_xyt_example_files,
        guards,
        keep_xboundaries,
        keep_yboundaries,
        dnd_type,
    ):
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
            topology=dnd_type,
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
        )

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

        if dnd_type == "lower-disconnected-double-null":
            ixs_inner = ixs1
            ixs_outer = ixs2
        elif dnd_type == "upper-disconnected-double-null":
            ixs_inner = ixs2
            ixs_outer = ixs1
        else:
            raise ValueError(f"Unsupported dnd_type: '{dnd_type}'")

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

        n = ds["n"]
        n_noregions = n.copy(deep=True)

        # Remove attributes that are expected to be different
        del n_noregions.attrs["regions"]

        n_lower_inner_PFR = n.bout.from_region("lower_inner_PFR")

        # Remove attributes that are expected to be different
        del n_lower_inner_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs_inner + mxg), theta=slice(jys11 + 1)),
            n_lower_inner_PFR.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + mxg), theta=slice(jys22 + 1, jys22 + 1 + myg)
                ).values,
                n_lower_inner_PFR.isel(theta=slice(-myg, None)).values,
            )

        n_lower_inner_intersep = n.bout.from_region("lower_inner_intersep")

        # Remove attributes that are expected to be different
        del n_lower_inner_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - mxg, ixs_outer + mxg), theta=slice(jys11 + 1)
            ),
            n_lower_inner_intersep.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in inner SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys11 + 1, jys11 + 1 + myg),
                    ).values,
                    n_lower_inner_intersep.isel(theta=slice(-myg, None)).values,
                )
            else:
                # Connects to intersep in lower outer divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys22 + 1, jys22 + 1 + myg),
                    ).values,
                    n_lower_inner_intersep.isel(theta=slice(-myg, None)).values,
                )

        n_lower_inner_SOL = n.bout.from_region("lower_inner_SOL")

        # Remove attributes that are expected to be different
        del n_lower_inner_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs_outer - mxg, None), theta=slice(jys11 + 1)),
            n_lower_inner_SOL.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - mxg, None),
                    theta=slice(jys11 + 1, jys11 + 1 + myg),
                ).values,
                n_lower_inner_SOL.isel(theta=slice(-myg, None)).values,
            )

        n_inner_core = n.bout.from_region("inner_core")

        # Remove attributes that are expected to be different
        del n_inner_core.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner + mxg), theta=slice(jys11 + 1, jys21 + 1)
            ),
            n_inner_core.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + mxg), theta=slice(jys22 + 1 - myg, jys22 + 1)
                ).values,
                n_inner_core.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + mxg), theta=slice(jys12 + 1, jys12 + 1 + myg)
                ).values,
                n_inner_core.isel(theta=slice(-myg, None)).values,
            )

        n_inner_intersep = n.bout.from_region("inner_intersep")

        # Remove attributes that are expected to be different
        del n_inner_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - mxg, ixs_outer + mxg),
                theta=slice(jys11 + 1, jys21 + 1),
            ),
            n_inner_intersep.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in lower inner divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys11 + 1 - myg, jys11 + 1),
                    ).values,
                    n_inner_intersep.isel(theta=slice(myg)).values,
                )

                # Connects to intersep in outer SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys12 + 1, jys12 + 1 + myg),
                    ).values,
                    n_inner_intersep.isel(theta=slice(-myg, None)).values,
                )
            else:
                # Connects to intersep in outer SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys22 + 1 - myg, jys22 + 1),
                    ).values,
                    n_inner_intersep.isel(theta=slice(myg)).values,
                )

                # Connects to intersep in upper inner divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys21 + 1, jys21 + 1 + myg),
                    ).values,
                    n_inner_intersep.isel(theta=slice(-myg, None)).values,
                )

        n_inner_sol = n.bout.from_region("inner_SOL")

        # Remove attributes that are expected to be different
        del n_inner_sol.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - mxg, None), theta=slice(jys11 + 1, jys21 + 1)
            ),
            n_inner_sol.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - mxg, None),
                    theta=slice(jys11 + 1 - myg, jys11 + 1),
                ).values,
                n_inner_sol.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - mxg, None),
                    theta=slice(jys21 + 1, jys21 + 1 + myg),
                ).values,
                n_inner_sol.isel(theta=slice(-myg, None)).values,
            )

        n_upper_inner_PFR = n.bout.from_region("upper_inner_PFR")

        # Remove attributes that are expected to be different
        del n_upper_inner_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner + mxg), theta=slice(jys21 + 1, ny_inner)
            ),
            n_upper_inner_PFR.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + mxg), theta=slice(jys12 + 1 - myg, jys12 + 1)
                ).values,
                n_upper_inner_PFR.isel(theta=slice(myg)).values,
            )

        n_upper_inner_intersep = n.bout.from_region("upper_inner_intersep")

        # Remove attributes that are expected to be different
        del n_upper_inner_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - mxg, ixs_outer + mxg),
                theta=slice(jys21 + 1, ny_inner),
            ),
            n_upper_inner_intersep.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in upper outer divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys12 + 1 - myg, jys12 + 1),
                    ).values,
                    n_upper_inner_intersep.isel(theta=slice(myg)).values,
                )
            else:
                # Connects to intersep in inner SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys21 + 1 - myg, jys21 + 1),
                    ).values,
                    n_upper_inner_intersep.isel(theta=slice(myg)).values,
                )

        n_upper_inner_SOL = n.bout.from_region("upper_inner_SOL")

        # Remove attributes that are expected to be different
        del n_upper_inner_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - mxg, None), theta=slice(jys21 + 1, ny_inner)
            ),
            n_upper_inner_SOL.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - mxg, None),
                    theta=slice(jys21 + 1 - myg, jys21 + 1),
                ).values,
                n_upper_inner_SOL.isel(theta=slice(myg)).values,
            )

        n_upper_outer_PFR = n.bout.from_region("upper_outer_PFR")

        # Remove attributes that are expected to be different
        del n_upper_outer_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner + mxg), theta=slice(ny_inner, jys12 + 1)
            ),
            n_upper_outer_PFR.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + mxg), theta=slice(jys21 + 1, jys21 + 1 + myg)
                ).values,
                n_upper_outer_PFR.isel(theta=slice(-myg, None)).values,
            )

        n_upper_outer_intersep = n.bout.from_region("upper_outer_intersep")

        # Remove attributes that are expected to be different
        del n_upper_outer_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - mxg, ixs_outer + mxg),
                theta=slice(ny_inner, jys12 + 1),
            ),
            n_upper_outer_intersep.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in upper inner divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys21 + 1, jys21 + 1 + myg),
                    ).values,
                    n_upper_outer_intersep.isel(theta=slice(-myg, None)).values,
                )
            else:
                # Connects to intersep in outer SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys12 + 1, jys12 + 1 + myg),
                    ).values,
                    n_upper_outer_intersep.isel(theta=slice(-myg, None)).values,
                )

        n_upper_outer_SOL = n.bout.from_region("upper_outer_SOL")

        # Remove attributes that are expected to be different
        del n_upper_outer_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - mxg, None), theta=slice(ny_inner, jys12 + 1)
            ),
            n_upper_outer_SOL.isel(theta=slice(-myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - mxg, None),
                    theta=slice(jys12 + 1, jys12 + 1 + myg),
                ).values,
                n_upper_outer_SOL.isel(theta=slice(-myg, None)).values,
            )

        n_outer_core = n.bout.from_region("outer_core")

        # Remove attributes that are expected to be different
        del n_outer_core.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner + mxg), theta=slice(jys12 + 1, jys22 + 1)
            ),
            n_outer_core.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + mxg), theta=slice(jys21 + 1 - myg, jys21 + 1)
                ).values,
                n_outer_core.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + mxg), theta=slice(jys11 + 1, jys11 + 1 + myg)
                ).values,
                n_outer_core.isel(theta=slice(-myg, None)).values,
            )

        n_outer_intersep = n.bout.from_region("outer_intersep")

        # Remove attributes that are expected to be different
        del n_outer_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - mxg, ixs_outer + mxg),
                theta=slice(jys12 + 1, jys22 + 1),
            ),
            n_outer_intersep.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in inner SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys21 + 1 - myg, jys21 + 1),
                    ).values,
                    n_outer_intersep.isel(theta=slice(myg)).values,
                )
                # Connects to intersep in lower outer divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys22 + 1, jys22 + 1 + myg),
                    ).values,
                    n_outer_intersep.isel(theta=slice(-myg, None)).values,
                )
            else:
                # Connects to intersep in upper outer divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys12 + 1 - myg, jys12 + 1),
                    ).values,
                    n_outer_intersep.isel(theta=slice(myg)).values,
                )
                # Connects to intersep in inner SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys11 + 1, jys11 + 1 + myg),
                    ).values,
                    n_outer_intersep.isel(theta=slice(-myg, None)).values,
                )

        n_outer_sol = n.bout.from_region("outer_SOL")

        # Remove attributes that are expected to be different
        del n_outer_sol.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - mxg, None), theta=slice(jys12 + 1, jys22 + 1)
            ),
            n_outer_sol.isel(theta=slice(myg, -myg if myg != 0 else None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - mxg, None),
                    theta=slice(jys12 + 1 - myg, jys12 + 1),
                ).values,
                n_outer_sol.isel(theta=slice(myg)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - mxg, None),
                    theta=slice(jys22 + 1, jys22 + 1 + myg),
                ).values,
                n_outer_sol.isel(theta=slice(-myg, None)).values,
            )

        n_lower_outer_PFR = n.bout.from_region("lower_outer_PFR")

        # Remove attributes that are expected to be different
        del n_lower_outer_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs_inner + mxg), theta=slice(jys22 + 1, None)),
            n_lower_outer_PFR.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + mxg), theta=slice(jys11 + 1 - myg, jys11 + 1)
                ).values,
                n_lower_outer_PFR.isel(theta=slice(myg)).values,
            )

        n_lower_outer_intersep = n.bout.from_region("lower_outer_intersep")

        # Remove attributes that are expected to be different
        del n_lower_outer_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - mxg, ixs_outer + mxg), theta=slice(jys22 + 1, None)
            ),
            n_lower_outer_intersep.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in outer SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys22 + 1 - myg, jys22 + 1),
                    ).values,
                    n_lower_outer_intersep.isel(theta=slice(myg)).values,
                )
            else:
                # Connects to intersep in lower inner divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - mxg, ixs_outer + mxg),
                        theta=slice(jys11 + 1 - myg, jys11 + 1),
                    ).values,
                    n_lower_outer_intersep.isel(theta=slice(myg)).values,
                )

        n_lower_outer_SOL = n.bout.from_region("lower_outer_SOL")

        # Remove attributes that are expected to be different
        del n_lower_outer_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - mxg, None), theta=slice(jys22 + 1, None)
            ),
            n_lower_outer_SOL.isel(theta=slice(myg, None)),
        )
        if myg > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - mxg, None),
                    theta=slice(jys22 + 1 - myg, jys22 + 1),
                ).values,
                n_lower_outer_SOL.isel(theta=slice(myg)).values,
            )

    @pytest.mark.parametrize(params_guards, params_guards_values)
    @pytest.mark.parametrize(params_boundaries, params_boundaries_values)
    @pytest.mark.parametrize(
        "with_guards", [0, {"x": 1}, {"theta": 1}, {"x": 1, "theta": 1}, 1]
    )
    @pytest.mark.parametrize(
        "dnd_type", ["lower-disconnected-double-null", "upper-disconnected-double-null"]
    )
    def test_region_disconnecteddoublenull_get_one_guard(
        self,
        bout_xyt_example_files,
        guards,
        keep_xboundaries,
        keep_yboundaries,
        with_guards,
        dnd_type,
    ):
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
            topology=dnd_type,
        )

        ds = open_boutdataset(
            datapath=dataset_list,
            gridfilepath=grid_ds,
            geometry="toroidal",
            keep_xboundaries=keep_xboundaries,
            keep_yboundaries=keep_yboundaries,
        )

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

        if dnd_type == "lower-disconnected-double-null":
            ixs_inner = ixs1
            ixs_outer = ixs2
        elif dnd_type == "upper-disconnected-double-null":
            ixs_inner = ixs2
            ixs_outer = ixs1
        else:
            raise ValueError(f"Unsupported dnd_type: '{dnd_type}'")

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

        if isinstance(with_guards, dict):
            xguards = with_guards.get("x", mxg)
            yguards = with_guards.get("theta", myg)
        else:
            xguards = with_guards
            yguards = with_guards

        n = ds["n"]
        n_noregions = n.copy(deep=True)

        # Remove attributes that are expected to be different
        del n_noregions.attrs["regions"]

        n_lower_inner_PFR = n.bout.from_region(
            "lower_inner_PFR", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_lower_inner_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(x=slice(ixs_inner + xguards), theta=slice(jys11 + 1)),
            n_lower_inner_PFR.isel(theta=slice(-yguards if yguards != 0 else None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + xguards),
                    theta=slice(jys22 + 1, jys22 + 1 + yguards),
                ).values,
                n_lower_inner_PFR.isel(theta=slice(-yguards, None)).values,
            )

        n_lower_inner_intersep = n.bout.from_region(
            "lower_inner_intersep", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_lower_inner_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - xguards, ixs_outer + xguards),
                theta=slice(jys11 + 1),
            ),
            n_lower_inner_intersep.isel(
                theta=slice(-yguards if yguards != 0 else None)
            ),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in inner SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys11 + 1, jys11 + 1 + yguards),
                    ).values,
                    n_lower_inner_intersep.isel(theta=slice(-yguards, None)).values,
                )
            else:
                # Connects to intersep in lower outer divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys22 + 1, jys22 + 1 + yguards),
                    ).values,
                    n_lower_inner_intersep.isel(theta=slice(-yguards, None)).values,
                )

        n_lower_inner_SOL = n.bout.from_region(
            "lower_inner_SOL", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_lower_inner_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - xguards, None), theta=slice(jys11 + 1)
            ),
            n_lower_inner_SOL.isel(theta=slice(-yguards if yguards != 0 else None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - xguards, None),
                    theta=slice(jys11 + 1, jys11 + 1 + yguards),
                ).values,
                n_lower_inner_SOL.isel(theta=slice(-yguards, None)).values,
            )

        n_inner_core = n.bout.from_region("inner_core", with_guards=with_guards)

        # Remove attributes that are expected to be different
        del n_inner_core.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner + xguards), theta=slice(jys11 + 1, jys21 + 1)
            ),
            n_inner_core.isel(theta=slice(yguards, -yguards if yguards != 0 else None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + xguards),
                    theta=slice(jys22 + 1 - yguards, jys22 + 1),
                ).values,
                n_inner_core.isel(theta=slice(yguards)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + xguards),
                    theta=slice(jys12 + 1, jys12 + 1 + yguards),
                ).values,
                n_inner_core.isel(theta=slice(-yguards, None)).values,
            )

        n_inner_intersep = n.bout.from_region("inner_intersep", with_guards=with_guards)

        # Remove attributes that are expected to be different
        del n_inner_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - xguards, ixs_outer + xguards),
                theta=slice(jys11 + 1, jys21 + 1),
            ),
            n_inner_intersep.isel(
                theta=slice(yguards, -yguards if yguards != 0 else None)
            ),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in lower inner divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys11 + 1 - yguards, jys11 + 1),
                    ).values,
                    n_inner_intersep.isel(theta=slice(yguards)).values,
                )

                # Connects to intersep in outer SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys12 + 1, jys12 + 1 + yguards),
                    ).values,
                    n_inner_intersep.isel(theta=slice(-yguards, None)).values,
                )
            else:
                # Connects to intersep in outer SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys22 + 1 - yguards, jys22 + 1),
                    ).values,
                    n_inner_intersep.isel(theta=slice(yguards)).values,
                )

                # Connects to intersep in upper inner divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys21 + 1, jys21 + 1 + yguards),
                    ).values,
                    n_inner_intersep.isel(theta=slice(-yguards, None)).values,
                )

        n_inner_sol = n.bout.from_region("inner_SOL", with_guards=with_guards)

        # Remove attributes that are expected to be different
        del n_inner_sol.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - xguards, None), theta=slice(jys11 + 1, jys21 + 1)
            ),
            n_inner_sol.isel(theta=slice(yguards, -yguards if yguards != 0 else None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - xguards, None),
                    theta=slice(jys11 + 1 - yguards, jys11 + 1),
                ).values,
                n_inner_sol.isel(theta=slice(yguards)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - xguards, None),
                    theta=slice(jys21 + 1, jys21 + 1 + yguards),
                ).values,
                n_inner_sol.isel(theta=slice(-yguards, None)).values,
            )

        n_upper_inner_PFR = n.bout.from_region(
            "upper_inner_PFR", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_upper_inner_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner + xguards), theta=slice(jys21 + 1, ny_inner)
            ),
            n_upper_inner_PFR.isel(theta=slice(yguards, None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + xguards),
                    theta=slice(jys12 + 1 - yguards, jys12 + 1),
                ).values,
                n_upper_inner_PFR.isel(theta=slice(yguards)).values,
            )

        n_upper_inner_intersep = n.bout.from_region(
            "upper_inner_intersep", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_upper_inner_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - xguards, ixs_outer + xguards),
                theta=slice(jys21 + 1, ny_inner),
            ),
            n_upper_inner_intersep.isel(theta=slice(yguards, None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in upper outer divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys12 + 1 - yguards, jys12 + 1),
                    ).values,
                    n_upper_inner_intersep.isel(theta=slice(yguards)).values,
                )
            else:
                # Connects to intersep in inner SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys21 + 1 - yguards, jys21 + 1),
                    ).values,
                    n_upper_inner_intersep.isel(theta=slice(yguards)).values,
                )

        n_upper_inner_SOL = n.bout.from_region(
            "upper_inner_SOL", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_upper_inner_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - xguards, None), theta=slice(jys21 + 1, ny_inner)
            ),
            n_upper_inner_SOL.isel(theta=slice(yguards, None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - xguards, None),
                    theta=slice(jys21 + 1 - yguards, jys21 + 1),
                ).values,
                n_upper_inner_SOL.isel(theta=slice(yguards)).values,
            )

        n_upper_outer_PFR = n.bout.from_region(
            "upper_outer_PFR", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_upper_outer_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner + xguards), theta=slice(ny_inner, jys12 + 1)
            ),
            n_upper_outer_PFR.isel(theta=slice(-yguards if yguards != 0 else None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + xguards),
                    theta=slice(jys21 + 1, jys21 + 1 + yguards),
                ).values,
                n_upper_outer_PFR.isel(theta=slice(-yguards, None)).values,
            )

        n_upper_outer_intersep = n.bout.from_region(
            "upper_outer_intersep", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_upper_outer_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - xguards, ixs_outer + xguards),
                theta=slice(ny_inner, jys12 + 1),
            ),
            n_upper_outer_intersep.isel(
                theta=slice(-yguards if yguards != 0 else None)
            ),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in upper inner divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys21 + 1, jys21 + 1 + yguards),
                    ).values,
                    n_upper_outer_intersep.isel(theta=slice(-yguards, None)).values,
                )
            else:
                # Connects to intersep in outer SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys12 + 1, jys12 + 1 + yguards),
                    ).values,
                    n_upper_outer_intersep.isel(theta=slice(-yguards, None)).values,
                )

        n_upper_outer_SOL = n.bout.from_region(
            "upper_outer_SOL", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_upper_outer_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - xguards, None), theta=slice(ny_inner, jys12 + 1)
            ),
            n_upper_outer_SOL.isel(theta=slice(-yguards if yguards != 0 else None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - xguards, None),
                    theta=slice(jys12 + 1, jys12 + 1 + yguards),
                ).values,
                n_upper_outer_SOL.isel(theta=slice(-yguards, None)).values,
            )

        n_outer_core = n.bout.from_region("outer_core", with_guards=with_guards)

        # Remove attributes that are expected to be different
        del n_outer_core.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner + xguards), theta=slice(jys12 + 1, jys22 + 1)
            ),
            n_outer_core.isel(theta=slice(yguards, -yguards if yguards != 0 else None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + xguards),
                    theta=slice(jys21 + 1 - yguards, jys21 + 1),
                ).values,
                n_outer_core.isel(theta=slice(yguards)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + xguards),
                    theta=slice(jys11 + 1, jys11 + 1 + yguards),
                ).values,
                n_outer_core.isel(theta=slice(-yguards, None)).values,
            )

        n_outer_intersep = n.bout.from_region("outer_intersep", with_guards=with_guards)

        # Remove attributes that are expected to be different
        del n_outer_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - xguards, ixs_outer + xguards),
                theta=slice(jys12 + 1, jys22 + 1),
            ),
            n_outer_intersep.isel(
                theta=slice(yguards, -yguards if yguards != 0 else None)
            ),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in inner SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys21 + 1 - yguards, jys21 + 1),
                    ).values,
                    n_outer_intersep.isel(theta=slice(yguards)).values,
                )

                # Connects to intersep in lower outer divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys22 + 1, jys22 + 1 + yguards),
                    ).values,
                    n_outer_intersep.isel(theta=slice(-yguards, None)).values,
                )
            else:
                # Connects to intersep in upper outer divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys12 + 1 - yguards, jys12 + 1),
                    ).values,
                    n_outer_intersep.isel(theta=slice(yguards)).values,
                )

                # Connects to intersep in inner SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys11 + 1, jys11 + 1 + yguards),
                    ).values,
                    n_outer_intersep.isel(theta=slice(-yguards, None)).values,
                )

        n_outer_sol = n.bout.from_region("outer_SOL", with_guards=with_guards)

        # Remove attributes that are expected to be different
        del n_outer_sol.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - xguards, None), theta=slice(jys12 + 1, jys22 + 1)
            ),
            n_outer_sol.isel(theta=slice(yguards, -yguards if yguards != 0 else None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - xguards, None),
                    theta=slice(jys12 + 1 - yguards, jys12 + 1),
                ).values,
                n_outer_sol.isel(theta=slice(yguards)).values,
            )
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - xguards, None),
                    theta=slice(jys22 + 1, jys22 + 1 + yguards),
                ).values,
                n_outer_sol.isel(theta=slice(-yguards, None)).values,
            )

        n_lower_outer_PFR = n.bout.from_region(
            "lower_outer_PFR", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_lower_outer_PFR.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner + xguards), theta=slice(jys22 + 1, None)
            ),
            n_lower_outer_PFR.isel(theta=slice(yguards, None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_inner + xguards),
                    theta=slice(jys11 + 1 - yguards, jys11 + 1),
                ).values,
                n_lower_outer_PFR.isel(theta=slice(yguards)).values,
            )

        n_lower_outer_intersep = n.bout.from_region(
            "lower_outer_intersep", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_lower_outer_intersep.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_inner - xguards, ixs_outer + xguards),
                theta=slice(jys22 + 1, None),
            ),
            n_lower_outer_intersep.isel(theta=slice(yguards, None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            if dnd_type == "lower-disconnected-double-null":
                # Connects to intersep in outer SOL
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys22 + 1 - yguards, jys22 + 1),
                    ).values,
                    n_lower_outer_intersep.isel(theta=slice(yguards)).values,
                )
            else:
                # Connects to intersep in lower inner divertor leg
                npt.assert_equal(
                    n_noregions.isel(
                        x=slice(ixs_inner - xguards, ixs_outer + xguards),
                        theta=slice(jys11 + 1 - yguards, jys11 + 1),
                    ).values,
                    n_lower_outer_intersep.isel(theta=slice(yguards)).values,
                )

        n_lower_outer_SOL = n.bout.from_region(
            "lower_outer_SOL", with_guards=with_guards
        )

        # Remove attributes that are expected to be different
        del n_lower_outer_SOL.attrs["regions"]
        xrt.assert_identical(
            n_noregions.isel(
                x=slice(ixs_outer - xguards, None), theta=slice(jys22 + 1, None)
            ),
            n_lower_outer_SOL.isel(theta=slice(yguards, None)),
        )
        if yguards > 0:
            # check y-guards, which were 'communicated' by from_region
            # Coordinates are not equal, so only compare array values
            npt.assert_equal(
                n_noregions.isel(
                    x=slice(ixs_outer - xguards, None),
                    theta=slice(jys22 + 1 - yguards, jys22 + 1),
                ).values,
                n_lower_outer_SOL.isel(theta=slice(yguards)).values,
            )
