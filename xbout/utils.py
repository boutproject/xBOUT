from copy import deepcopy
from itertools import chain
from pathlib import Path

import numpy as np
import xarray as xr


def _set_attrs_on_all_vars(ds, key, attr_data, copy=False):
    ds.attrs[key] = attr_data
    if copy:
        for v in chain(ds.data_vars, ds.coords):
            ds[v].attrs[key] = deepcopy(attr_data)
    else:
        for v in chain(ds.data_vars, ds.coords):
            ds[v].attrs[key] = attr_data
    return ds


def _add_attrs_to_var(ds, varname, copy=False):
    if copy:
        for attr in ["metadata", "options", "geometry", "regions"]:
            if attr in ds.attrs and attr not in ds[varname].attrs:
                ds[varname].attrs[attr] = deepcopy(ds.attrs[attr])
    else:
        for attr in ["metadata", "options", "geometry", "regions"]:
            if attr in ds.attrs and attr not in ds[varname].attrs:
                ds[varname].attrs[attr] = ds.attrs[attr]


def _check_filetype(path):
    if path.suffix == ".nc":
        filetype = "netcdf4"
    elif path.suffix == ".h5netcdf":
        filetype = "h5netcdf"
    else:
        raise IOError("Do not know how to read file extension {}".format(path.suffix))
    return filetype


def _is_path(p):
    try:
        Path(p)
        return True
    except TypeError:
        return False


def _separate_metadata(ds):
    """
    Extract the metadata (nxpe, myg etc.) from the Dataset.

    Assumes that all scalar variables are metadata, not physical data!
    """

    # Find only the scalar variables
    variables = list(ds.variables)

    # Remove dz from metadata if it's present. Allows treating dz more consistently
    # whether it is scalar or 2d/3d array.
    exclude = ["dz"]

    scalar_vars = [
        var
        for var in variables
        if not any(dim in ["t", "x", "y", "z"] for dim in ds[var].dims)
        and var not in exclude
    ]

    # Save metadata as a dictionary
    metadata_vals = [ds[var].values.item() for var in scalar_vars]
    metadata_vals = [x.decode() if isinstance(x, bytes) else x for x in metadata_vals]
    metadata = dict(zip(scalar_vars, metadata_vals))

    # Add default values for dimensions to metadata. These may be modified later by
    # apply_geometry()
    metadata["bout_tdim"] = "t"
    metadata["bout_xdim"] = "x"
    metadata["bout_ydim"] = "y"
    metadata["bout_zdim"] = "z"

    return ds.drop_vars(scalar_vars), metadata


def _update_metadata_increased_resolution(da, n):
    """
    Update the metadata variables to account for a y-direction resolution increased by a
    factor n.

    Parameters
    ----------
    da : DataArray
        The variable to update
    n : int
        The factor to increase the y-resolution by
    """

    # Take deepcopy to ensure we do not alter metadata of other variables
    da.attrs["metadata"] = deepcopy(da.metadata)

    def update_jyseps(name):
        # If any jyseps<=0, need to leave as is
        if da.metadata[name] > 0:
            da.metadata[name] = n * (da.metadata[name] + 1) - 1

    update_jyseps("jyseps1_1")
    update_jyseps("jyseps2_1")
    update_jyseps("jyseps1_2")
    update_jyseps("jyseps2_2")

    def update_ny(name):
        da.metadata[name] = n * da.metadata[name]

    update_ny("ny")
    update_ny("ny_inner")
    update_ny("MYSUB")

    # Update attrs of coordinates to be consistent with da
    for coord in da.coords:
        da[coord].attrs = {}
        _add_attrs_to_var(da, coord)

    return da


def _1d_coord_from_spacing(spacing, dim, ds=None, *, origin_at=None):
    """
    Create a 1d coordinate varying along the dimension 'dim' from the grid spacing
    'spacing', with the grid points in the centres of the cells with width given by
    'spacing'.

    Parameters
    ----------
    spacing : DataArray or scalar
        The grid spacing. If a DataArray, must include dimension 'dim' and must be
        constant in any other dimensions.
    dim : str
        The dimension to create a coordinate for
    ds : DataArray or Dataset, optional
        If spacing is a scalar, a Dataset or DataArray containing dimension 'dim' must
        be passed to determine the size of the dimension
    origin_at : string, {["lower"], "centre", "upper"}
        Where to put the origin of the coordinate. Can be at lower edge, centre, or
        upper edge of the grid
    """
    if origin_at is None:
        origin_at = "lower"

    if not isinstance(spacing, xr.DataArray):
        # spacing is a scalar
        if ds is None:
            raise ValueError("ds must be passed when spacing is a scalar")
        n = ds.sizes[dim]
        coord_values = (np.arange(n, dtype=float) + 0.5) * spacing
        total_length = n * spacing
    else:
        other_dims = set(spacing.dims) - set([dim])
        if not np.all(spacing.min(dim=other_dims) == spacing.max(dim=other_dims)):
            raise ValueError(
                f"Spacing is not constant in dimensions other than {dim}. Cannot "
                f"create coordinate"
            )

        point_to_use = {
            spacing.metadata["bout_xdim"]: spacing.metadata.get("MXG", 0)
            if spacing.metadata["keep_xboundaries"]
            else 0,
            spacing.metadata["bout_ydim"]: spacing.metadata.get("MYG", 0)
            if spacing.metadata["keep_yboundaries"]
            else 0,
            spacing.metadata["bout_zdim"]: spacing.metadata.get("MZG", 0),
        }

        # make spacing 1d
        spacing = spacing.isel({d: point_to_use[d] for d in other_dims})

        # xarray stores coordinates as numpy (not dask) arrays anyway, so use .values
        # here to evaluate the task-graph (if there is one)
        spacing = spacing.values

        coord_values = spacing.cumsum()
        total_length = coord_values[-1]

        # adjust to cell mid-point positions
        coord_values = coord_values - 0.5 * spacing

    if origin_at == "lower":
        pass
    elif origin_at == "centre":
        coord_values = coord_values - 0.5 * total_length
    elif origin_at == "upper":
        coord_values = coord_values - total_length
    else:
        raise ValueError(f"Unrecognised argument origin_at={origin_at}")

    return xr.Variable(dim, coord_values)


def _add_cartesian_coordinates(ds):
    # Add Cartesian X and Y coordinates if they do not exist already
    # Works on either BoutDataset or BoutDataArray

    if ds.geometry == "toroidal":
        R = ds["R"]
        Z = ds["Z"]
        zeta = ds[ds.metadata["bout_zdim"]]
        if "X_cartesian" not in ds.coords:
            X = R * np.cos(zeta)
            ds = ds.assign_coords(X_cartesian=X)
        if "Y_cartesian" not in ds.coords:
            Y = R * np.sin(zeta)
            ds = ds.assign_coords(Y_cartesian=Y)
        if "Z_cartesian" not in ds.coords:
            zcoord = ds.metadata["bout_zdim"]
            nz = len(ds[zcoord])
            ds = ds.assign_coords(Z_cartesian=Z.expand_dims({zcoord: nz}, axis=-1))
    else:
        raise ValueError(
            f"Adding Cartesian coordinates to geometry={ds.geometry} is not supported"
        )

    return ds


def _check_new_nxpe(ds, nxpe):
    # Check nxpe is valid

    nx = ds.metadata["nx"] - 2 * ds.metadata["MXG"]
    mxsub = nx // nxpe

    if nx % nxpe != 0:
        raise ValueError(
            f"nxpe={nxpe} must divide total number of points "
            f"nx-2*MXG={ds.metadata['nx'] - 2*ds.metadata['MXG']}"
        )


def _check_new_nype(ds, nype):
    # Check nype is valid

    ny = ds.metadata["ny"]
    mysub = ny // nype

    if ny % nype != 0:
        raise ValueError(
            f"nype={nype} must divide total number of points ny={ds.metadata['ny']}"
        )
    if (
        ds.metadata["jyseps1_1"] >= 0
        and ds.metadata["jyseps1_1"] <= ny
        and (ds.metadata["jyseps1_1"] + 1) % mysub != 0
    ):
        raise ValueError(
            f"mysub={mysub} must divide jyseps1_1+1={ds.metadata['jyseps1_1'] + 1}"
        )
    if (
        ds.metadata["jyseps2_1"] >= 0
        and ds.metadata["jyseps2_1"] <= ny
        and ds.metadata["jyseps1_2"] != ds.metadata["jyseps2_1"]
        and (ds.metadata["jyseps2_1"] + 1) % mysub != 0
    ):
        raise ValueError(
            f"mysub={mysub} must divide jyseps2_1+1={ds.metadata['jyseps2_1'] + 1}"
        )
    if (
        ds.metadata["jyseps1_2"] != ds.metadata["jyseps2_1"]
        and ds.metadata["ny_inner"] % mysub != 0
    ):
        raise ValueError(
            f"mysub={mysub} must divide ny_inner={ds.metadata['ny_inner']}"
        )
    if (
        ds.metadata["jyseps1_2"] >= 0
        and ds.metadata["jyseps1_2"] <= ny
        and ds.metadata["jyseps1_2"] != ds.metadata["jyseps2_1"]
        and (ds.metadata["jyseps1_2"] + 1) % mysub != 0
    ):
        raise ValueError(
            f"mysub={mysub} must divide jyseps1_2+1={ds.metadata['jyseps1_2'] + 1}"
        )
    if (
        ds.metadata["jyseps2_2"] >= 0
        and ds.metadata["jyseps2_2"] < ny
        and (ds.metadata["jyseps2_2"] + 1) % mysub != 0
    ):
        raise ValueError(
            f"mysub={mysub} must divide jyseps2_2+1={ds.metadata['jyseps2_2'] + 1}"
        )


def _pad_x_boundaries(ds):
    # pad Dataset if boundaries are not already present

    mxg = ds.metadata["MXG"]
    xcoord = ds.metadata.get("bout_xdim", "x")

    if not ds.metadata["keep_xboundaries"] and mxg > 0:
        boundary_pad = ds.isel({xcoord: slice(mxg)}).copy(deep=True).load()
        for v in boundary_pad:
            if xcoord in boundary_pad[v].dims:
                boundary_pad[v].values[...] = np.nan
        ds = xr.concat(
            [boundary_pad, ds.load(), boundary_pad], dim=xcoord, data_vars="minimal"
        )

    return ds


def _pad_y_boundaries(ds):
    # pad Dataset if boundaries are not already present

    myg = ds.metadata["MYG"]
    ycoord = ds.metadata.get("bout_ydim", "y")
    has_second_divertor = ds.metadata["jyseps2_1"] != ds.metadata["jyseps1_2"]

    if not ds.metadata["keep_yboundaries"] and myg > 0:
        boundary_pad = ds.isel({ycoord: slice(myg)}).copy(deep=True).load()
        for v in boundary_pad:
            if ycoord in boundary_pad[v].dims:
                boundary_pad[v].values[...] = np.nan
        if not has_second_divertor:
            ds = xr.concat(
                [boundary_pad, ds.load(), boundary_pad], dim=ycoord, data_vars="minimal"
            )
        else:
            # Include second divertor
            ny_inner = ds.metadata["ny_inner"]
            ds = xr.concat(
                [
                    boundary_pad,
                    ds.isel({ycoord: slice(ny_inner)}).load(),
                    boundary_pad,
                    boundary_pad,
                    ds.isel({ycoord: slice(ny_inner, None)}).load(),
                    boundary_pad,
                ],
                dim=ycoord,
                data_vars="minimal",
            )

    return ds


def _split_into_restarts(ds, variables, savepath, nxpe, nype, tind, prefix, overwrite):
    # Rename dimension coordinates to BOUT++ standard names
    ds = ds.copy()
    for d in ["t", "x", "y", "z"]:
        key = f"bout_{d}dim"
        if key in ds.metadata and ds.metadata[key] in ds.dims:
            ds = ds.rename({ds.metadata[key]: d})

    _check_new_nxpe(ds, nxpe)
    _check_new_nype(ds, nype)

    # Create paths for the files. Do this first so we can check if files exist before
    # spending time creating restart Datasets
    paths = []
    for xproc in range(nxpe):
        for yproc in range(nype):
            # Global processor number
            i = yproc * nxpe + xproc

            paths.append(Path(savepath).joinpath(f"{prefix}.{i}.nc"))
            if not overwrite and paths[-1].exists():
                raise ValueError(
                    f"File {paths[-1]} already exists. Pass overwrite=True to overwrite."
                )

    mxg = ds.metadata["MXG"]
    myg = ds.metadata["MYG"]

    ny_inner = ds.metadata["ny_inner"]

    # These metadata variables are created by xBOUT, so should not be saved to restart
    # files
    restart_exclude_metadata_vars = [
        "bout_tdim",
        "bout_xdim",
        "bout_ydim",
        "bout_zdim",
        "fine_interpolation_factor",
        "is_restart",
        "keep_xboundaries",
        "keep_yboundaries",
    ]

    if variables is None:
        if "t" in ds.dims:
            # If variables to be saved were not specified, add all time-evolving
            # variables
            variables = [v for v in ds if "t" in ds[v].dims]
        else:
            # No time dimension in Dataset, so cannot distinguish time-evolving
            # variables: just include all variables
            variables = [v for v in ds]

    # Add extra variables always needed
    for v in [
        "dx",
        "dy",
        "dz",
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
        "J",
        "zShift",
    ]:
        if v not in variables and v in ds:
            variables.append(v)

    # number of points in the domain on each processor, not including guard or boundary
    # points
    mxsub = (ds.metadata["nx"] - 2 * mxg) // nxpe
    mysub = ds.metadata["ny"] // nype

    # hist_hi represents the number of iterations before the restart. Attempt to
    # reconstruct here
    final_hist_hi = ds.metadata.get("hist_hi", -1)
    if "t" in ds.dims:
        nt = ds.sizes["t"]
        if tind < 0:
            absolute_tind = tind + nt
            if absolute_tind < 0:
                absolute_tind = 0
        elif tind >= nt:
            absolute_tind = nt - 1
        else:
            absolute_tind = tind
        hist_hi = final_hist_hi - (nt - 1 - absolute_tind)
        if hist_hi < 0:
            hist_hi = -1
    else:
        # Either input is a set of restart files, in which case we should just use
        # `hist_hi` if it exists, or the user has already selected a single time-point,
        # and the best guess we have left is the existing `hist_hi` (although it might
        # not be right).
        hist_hi = final_hist_hi

    has_second_divertor = ds.metadata["jyseps2_1"] != ds.metadata["jyseps1_2"]

    if "t" in ds.dims:
        # select desired time-index for the restart files
        ds = ds.isel({"t": tind}).persist()
        tt = ds["t"].values.flatten()[0]
    else:
        if "tt" in ds.metadata:
            # If loaded from restart files, "tt" should be a scalar in metadata
            tt = ds.metadata["tt"]
        else:
            tt = 0.0

    ds = _pad_x_boundaries(ds)
    ds = _pad_y_boundaries(ds)

    restart_datasets = []
    for xproc in range(nxpe):
        xslice = slice(xproc * mxsub, (xproc + 1) * mxsub + 2 * mxg)
        for yproc in range(nype):
            print(f"creating {xproc*nype + yproc + 1}/{nxpe*nype}", end="\r")
            if has_second_divertor and yproc * mysub >= ny_inner:
                yslice = slice(yproc * mysub + 2 * myg, (yproc + 1) * mysub + 4 * myg)
            else:
                yslice = slice(yproc * mysub, (yproc + 1) * mysub + 2 * myg)

            ds_slice = ds.isel({"x": xslice, "y": yslice})

            restart_ds = xr.Dataset()
            for v in variables:
                data_variable = ds_slice[v].variable

                # delete attrs so we don't try to save metadata dict to restart files
                data_variable.attrs = {}

                restart_ds[v] = data_variable
            for v in ds.metadata:
                if v not in restart_exclude_metadata_vars:
                    restart_ds[v] = ds.metadata[v]

            # These variables need to be altered, because they depend on the number of
            # files and/or the rank of this file.
            restart_ds["MXSUB"] = mxsub
            restart_ds["MYSUB"] = mysub
            restart_ds["NXPE"] = nxpe
            restart_ds["NYPE"] = nype
            restart_ds["PE_XIND"] = xproc
            restart_ds["PE_YIND"] = yproc
            restart_ds["hist_hi"] = hist_hi
            restart_ds["PE_XIND"] = xproc
            restart_ds["PE_YIND"] = yproc
            restart_ds["MYPE"] = yproc * nxpe + xproc

            # tt is the simulation time where the restart happens
            restart_ds["tt"] = tt

            restart_datasets.append(restart_ds)

    return restart_datasets, paths


# Utility methods for _get_bounding_surfaces()
##############################################

_bounding_surface_checks = {}


def _check_upper_y(ds_region, boundary_points, xbndry, ybndry, Rcoord, Zcoord):
    region = list(ds_region.bout._regions.values())[0]
    xcoord = ds_region.metadata["bout_xdim"]
    ycoord = ds_region.metadata["bout_ydim"]

    if region.connection_upper_y is None:
        xinner = (
            xbndry - 1 if region.connection_inner_x is None and xbndry > 0 else None
        )
        xouter = (
            -xbndry - 1 if region.connection_outer_x is None and xbndry > 0 else None
        )

        # Reverse order of points to go clockwise around region
        boundary_slice = {xcoord: slice(xinner, xouter, -1), ycoord: -1 - ybndry}
        new_boundary_points = np.stack(
            [
                ds_region[Rcoord].isel(boundary_slice).values,
                ds_region[Zcoord].isel(boundary_slice).values,
            ],
            axis=-1,
        )

        boundary_points = np.concatenate([boundary_points, new_boundary_points])

        return boundary_points, region.connection_inner_x, "inner_x"
    else:
        return None


_bounding_surface_checks["upper_y"] = _check_upper_y


def _check_inner_x(ds_region, boundary_points, xbndry, ybndry, Rcoord, Zcoord):
    region = list(ds_region.bout._regions.values())[0]
    xcoord = ds_region.metadata["bout_xdim"]
    ycoord = ds_region.metadata["bout_ydim"]

    if region.connection_inner_x is None:
        ylower = (
            ybndry - 1 if region.connection_lower_y is None and ybndry > 0 else None
        )
        yupper = (
            -ybndry - 1 if region.connection_upper_y is None and ybndry > 0 else None
        )

        # Reverse order of points to go clockwise around region
        boundary_slice = {xcoord: xbndry, ycoord: slice(yupper, ylower, -1)}
        new_boundary_points = np.stack(
            [
                ds_region[Rcoord].isel(boundary_slice).values,
                ds_region[Zcoord].isel(boundary_slice).values,
            ],
            axis=-1,
        )

        boundary_points = np.concatenate([boundary_points, new_boundary_points])

        return boundary_points, region.connection_lower_y, "lower_y"
    else:
        return None


_bounding_surface_checks["inner_x"] = _check_inner_x


def _check_lower_y(ds_region, boundary_points, xbndry, ybndry, Rcoord, Zcoord):
    region = list(ds_region.bout._regions.values())[0]
    xcoord = ds_region.metadata["bout_xdim"]
    ycoord = ds_region.metadata["bout_ydim"]

    if region.connection_lower_y is None:
        xinner = xbndry if region.connection_inner_x is None else None
        xouter = -xbndry if region.connection_outer_x is None and xbndry > 0 else None

        boundary_slice = {xcoord: slice(xinner, xouter), ycoord: ybndry}
        new_boundary_points = np.stack(
            [
                ds_region[Rcoord].isel(boundary_slice).values,
                ds_region[Zcoord].isel(boundary_slice).values,
            ],
            axis=-1,
        )

        boundary_points = np.concatenate([boundary_points, new_boundary_points])

        return boundary_points, region.connection_outer_x, "outer_x"
    else:
        return None


_bounding_surface_checks["lower_y"] = _check_lower_y


def _check_outer_x(ds_region, boundary_points, xbndry, ybndry, Rcoord, Zcoord):
    region = list(ds_region.bout._regions.values())[0]
    xcoord = ds_region.metadata["bout_xdim"]
    ycoord = ds_region.metadata["bout_ydim"]

    if region.connection_outer_x is None:
        ylower = ybndry if region.connection_lower_y is None else None
        yupper = -ybndry if region.connection_upper_y is None and ybndry > 0 else None

        boundary_slice = {xcoord: -1 - xbndry, ycoord: slice(ylower, yupper)}
        new_boundary_points = np.stack(
            [
                ds_region[Rcoord].isel(boundary_slice).values,
                ds_region[Zcoord].isel(boundary_slice).values,
            ],
            axis=-1,
        )

        boundary_points = np.concatenate([boundary_points, new_boundary_points])

        return boundary_points, region.connection_upper_y, "upper_y"
    else:
        return None


_bounding_surface_checks["outer_x"] = _check_outer_x


def _follow_boundary(ds, start_region, start_direction, xbndry, ybndry, Rcoord, Zcoord):
    # Define lists of boundaries to cycle through, depending on which direction we
    # entered the region in
    check_order = {
        "upper_y": ["outer_x", "upper_y", "inner_x", "lower_y"],
        "inner_x": ["upper_y", "inner_x", "lower_y", "outer_x"],
        "lower_y": ["inner_x", "lower_y", "outer_x", "upper_y"],
        "outer_x": ["lower_y", "outer_x", "upper_y", "inner_x"],
    }

    this_region = start_region
    direction = start_direction
    visited_regions = []
    boundary_points = np.empty([0, 2])
    while True:
        # Follow around the boundary from start_region

        visited_regions.append(this_region)

        ds_region = ds.bout.from_region(this_region, with_guards=0)
        region = ds.bout._regions[this_region]

        # Get all boundary points from this region, and decide which region to go to next
        this_region = None

        for boundary in check_order[direction]:
            result = _bounding_surface_checks[boundary](
                ds_region,
                boundary_points,
                xbndry,
                ybndry,
                Rcoord,
                Zcoord,
            )
            if result is not None:
                boundary_points, this_region, direction = result
            else:
                # Only add the boundary now if it's connected to the boundary we are
                # following
                break

        if this_region is None:
            raise ValueError(f"this_region={this_region} has no boundaries")
        elif this_region == start_region:
            break

    return boundary_points, visited_regions


def _get_bounding_surfaces(ds, coords):
    """
    Get boundary surfaces from a Dataset or DataArray

    Makes some assumptions about possible topologies that are true for currently
    supported single-null/double-null tokamaks, but might not be for very general
    topologies:
    - If there are y-boundaries, there is a region with both a lower-y boundary and an
      outer-x boundary.

    Outer boundary goes clockwise, inner boundary (if present) goes anti-clockwise.

    Note the 'boundary points' are the grid points adjacent to the boundary, not the grid
    boundary half way between grid point and boundary point. Boundary cells are not
    always present, so tricky to implement something 'better'.

    Returns
    -------
    tuple of (N,2) float arrays of vertices on the boundaries
    """

    # coords do not have to be R and Z, but usually will be.
    Rcoord, Zcoord = coords

    # Get number of boundary cells
    if ds.metadata["keep_xboundaries"]:
        xbndry = ds.metadata["MXG"]
    else:
        xbndry = 0
    if ds.metadata["keep_yboundaries"]:
        ybndry = ds.metadata["MYG"]
    else:
        ybndry = 0

    xcoord = ds.metadata["bout_xdim"]
    ycoord = ds.metadata["bout_ydim"]

    # First find the outer boundary
    start_region = None
    for name, region in ds.bout._regions.items():
        if region.connection_lower_y is None and region.connection_outer_x is None:
            start_region = name
            break
    if start_region is None:
        # No y-boundary region found, presumably is core-only simulation. Start on any
        # region with an outer-x boundary
        for name, region in ds.bout._regions.items():
            if region.connection_outer_x is None:
                start_region = name
    if start_region is None:
        raise ValueError(
            "Failed to find bounding surfaces - no region with outer boundary found"
        )

    # First region has outer-x boundary, but we only visit start_region once, so need to
    # add all boundaries in it the first time.
    region = ds.bout._regions[start_region]
    if region.connection_upper_y is None:
        start_direction = "inner_x"
    elif region.connection_inner_x is None and region.connection_lower_y is None:
        start_direction = "lower_y"
    elif region.connection_lower_y is None:
        start_direction = "outer_x"
    else:
        start_direction = "upper_y"

    boundary, checked_regions = _follow_boundary(
        ds,
        start_region,
        start_direction,
        xbndry,
        ybndry,
        Rcoord,
        Zcoord,
    )
    boundaries = [boundary]

    # Look for an inner boundary
    ############################
    remaining_regions = set(ds.bout._regions) - set(checked_regions)
    start_region = None
    if not remaining_regions:
        # Check for separate inner-x boundary on any of the already visited regions.
        # If inner-x boundary was part of the outer boundary (e.g. for SOL-only
        # simulation) then any region without y-boundaries will be visited twice,
        # otherwise any region that has an inner-x boundary and no y-boundary must be on
        # a separate inner boundary
        for r in checked_regions:
            if (
                ds.bout._regions[r].connection_inner_x is None
                and ds.bout._regions[r].connection_lower_y is not None
                and ds.bout._regions[r].connection_upper_y is not None
                and checked_regions.count(r) < 2
            ):
                start_region = r
                break
    else:
        for r in remaining_regions:
            if (
                ds.bout._regions[r].connection_inner_x is None
                and ds.bout._regions[r].connection_lower_y is not None
                and ds.bout._regions[r].connection_upper_y is not None
            ):
                start_region = r
                break

    if start_region is not None:
        # Inner boundary found
        # Inner boundary should have only an inner_x boundary, so set start_direction to
        # check that first
        start_direction = "lower_y"

        boundary, more_checked_regions = _follow_boundary(
            ds, start_region, start_direction, xbndry, ybndry, Rcoord, Zcoord
        )
        boundaries.append(boundary)
        checked_regions += more_checked_regions

        remaining_regions = set(ds.bout._regions) - set(checked_regions)

    # If there are any remaining regions, they should not have any boundaries
    for r in remaining_regions:
        region = ds.bout._regions[r]
        if (
            region.connection_lower_y is None
            or region.connection_outer_x is None
            or region.connection_upper_y is None
            or region.connection_inner_x is None
        ):
            raise ValueError(
                f"Region '{r}' has a boundary, but was missed when getting boundary "
                f"points"
            )

    # Pack the result into a DataArray
    result = [
        xr.DataArray(
            boundary,
            dims=("boundary", "coord"),
            coords={"coord": [Rcoord, Zcoord]},
        )
        for boundary in boundaries
    ]

    return result


def _set_as_coord(ds, name):
    try:
        ds = ds.set_coords(name)
    except ValueError:
        pass
    try:
        ds = ds.set_coords(f"{name}_CELL_XLOW")
    except ValueError:
        pass
    try:
        ds = ds.set_coords(f"{name}_CELL_YLOW")
    except ValueError:
        pass
    try:
        ds = ds.set_coords(f"{name}_CELL_ZLOW")
    except ValueError:
        pass
    return ds
