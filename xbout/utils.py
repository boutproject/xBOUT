from copy import deepcopy
from itertools import chain

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
    if path.suffix == '.nc':
        filetype = 'netcdf4'
    elif path.suffix == '.h5netcdf':
        filetype = 'h5netcdf'
    else:
        raise IOError("Do not know how to read file extension {}"
                      .format(path.suffix))
    return filetype


def _separate_metadata(ds):
    """
    Extract the metadata (nxpe, myg etc.) from the Dataset.

    Assumes that all scalar variables are metadata, not physical data!
    """

    # Find only the scalar variables
    variables = list(ds.variables)
    scalar_vars = [var for var in variables
                   if not any(dim in ['t', 'x', 'y', 'z'] for dim in ds[var].dims)]

    # Save metadata as a dictionary
    metadata_vals = [ds[var].values.item() for var in scalar_vars]
    metadata = dict(zip(scalar_vars, metadata_vals))

    return ds.drop(scalar_vars), metadata


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
    da.attrs['metadata'] = deepcopy(da.metadata)

    def update_jyseps(name):
        # If any jyseps<=0, need to leave as is
        if da.metadata[name] > 0:
            da.metadata[name] = n*(da.metadata[name] + 1) - 1
    update_jyseps('jyseps1_1')
    update_jyseps('jyseps2_1')
    update_jyseps('jyseps1_2')
    update_jyseps('jyseps2_2')

    def update_ny(name):
        da.metadata[name] = n*da.metadata[name]
    update_ny('ny')
    update_ny('ny_inner')
    update_ny('MYSUB')

    # Update attrs of coordinates to be consistent with da
    for coord in da.coords:
        da[coord].attrs = {}
        _add_attrs_to_var(da, coord)

    return da


# Utility methods for _get_bounding_surfaces()
##############################################

_bounding_surface_checks = {}


def _check_upper_y(ds_region, boundary_points, xbndry, ybndry, Rcoord, Zcoord):
    region = list(ds_region.regions.values())[0]
    xcoord = ds_region.metadata["bout_xdim"]
    ycoord = ds_region.metadata["bout_ydim"]

    if region.connection_upper_y is None:
        xinner = (
            xbndry - 1
            if region.connection_inner_x is None and xbndry > 0
            else None
        )
        xouter = (
            -xbndry - 1
            if region.connection_outer_x is None and xbndry > 0
            else None
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

        boundary_points = np.concatenate(
            [boundary_points, new_boundary_points]
        )

        return boundary_points, region.connection_inner_x, "inner_x"
    else:
        return None


_bounding_surface_checks["upper_y"] = _check_upper_y


def _check_inner_x(ds_region, boundary_points, xbndry, ybndry, Rcoord, Zcoord):
    region = list(ds_region.regions.values())[0]
    xcoord = ds_region.metadata["bout_xdim"]
    ycoord = ds_region.metadata["bout_ydim"]

    if region.connection_inner_x is None:
        ylower = (
            ybndry - 1
            if region.connection_lower_y is None and ybndry > 0
            else None
        )
        yupper = (
            -ybndry - 1
            if region.connection_upper_y is None and ybndry > 0
            else None
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

        boundary_points = np.concatenate(
            [boundary_points, new_boundary_points]
        )

        return boundary_points, region.connection_lower_y, "lower_y"
    else:
        return None


_bounding_surface_checks["inner_x"] = _check_inner_x


def _check_lower_y(ds_region, boundary_points, xbndry, ybndry, Rcoord, Zcoord):
    region = list(ds_region.regions.values())[0]
    xcoord = ds_region.metadata["bout_xdim"]
    ycoord = ds_region.metadata["bout_ydim"]

    if region.connection_lower_y is None:
        xinner = xbndry if region.connection_inner_x is None else None
        xouter = (
            -xbndry
            if region.connection_outer_x is None and xbndry > 0
            else None
        )

        boundary_slice = {xcoord: slice(xinner, xouter), ycoord: ybndry}
        new_boundary_points = np.stack(
            [
                ds_region[Rcoord].isel(boundary_slice).values,
                ds_region[Zcoord].isel(boundary_slice).values,
            ],
            axis=-1,
        )

        boundary_points = np.concatenate(
            [boundary_points, new_boundary_points]
        )

        return boundary_points, region.connection_outer_x, "outer_x"
    else:
        return None


_bounding_surface_checks["lower_y"] = _check_lower_y


def _check_outer_x(ds_region, boundary_points, xbndry, ybndry, Rcoord, Zcoord):
    region = list(ds_region.regions.values())[0]
    xcoord = ds_region.metadata["bout_xdim"]
    ycoord = ds_region.metadata["bout_ydim"]

    if region.connection_outer_x is None:
        ylower = ybndry if region.connection_lower_y is None else None
        yupper = (
            -ybndry
            if region.connection_upper_y is None and ybndry > 0
            else None
        )

        boundary_slice = {xcoord: -1 - xbndry, ycoord: slice(ylower, yupper)}
        new_boundary_points = np.stack(
            [
                ds_region[Rcoord].isel(boundary_slice).values,
                ds_region[Zcoord].isel(boundary_slice).values,
            ],
            axis=-1,
        )

        boundary_points = np.concatenate(
            [boundary_points, new_boundary_points]
        )

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
        region = ds.regions[this_region]

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
    for name, region in ds.regions.items():
        if region.connection_lower_y is None and region.connection_outer_x is None:
            start_region = name
            break
    if start_region is None:
        # No y-boundary region found, presumably is core-only simulation. Start on any
        # region with an outer-x boundary
        for name, region in ds.regions.items():
            if region.connection_outer_x is None:
                start_region = name
    if start_region is None:
        raise ValueError(
            "Failed to find bounding surfaces - no region with outer boundary found"
        )

    # First region has outer-x boundary, but we only visit start_region once, so need to
    # add all boundaries in it the first time.
    region = ds.regions[start_region]
    if region.connection_upper_y is None:
        start_direction = "inner_x"
    elif region.connection_inner_x is None and region.connection_lower_y is None:
        start_direction = "lower_y"
    elif region.connection_lower_y is None:
        start_direction = "outer_x"
    else:
        start_direction = "upper_y"

    boundary, checked_regions = _follow_boundary(
        ds, start_region, start_direction, xbndry, ybndry, Rcoord, Zcoord,
    )
    boundaries = [boundary]

    # Look for an inner boundary
    ############################
    remaining_regions = set(ds.regions) - set(checked_regions)
    start_region = None
    if not remaining_regions:
        # Check for separate inner-x boundary on any of the already visited regions.
        # If inner-x boundary was part of the outer boundary (e.g. for SOL-only
        # simulation) then any region without y-boundaries will be visited twice,
        # otherwise any region that has an inner-x boundary and no y-boundary must be on
        # a separate inner boundary
        for r in checked_regions:
            if (
                ds.regions[r].connection_inner_x is None
                and ds.regions[r].connection_lower_y is not None
                and ds.regions[r].connection_upper_y is not None
                and checked_regions.count(r) < 2
            ):
                start_region = r
                break
    else:
        for r in remaining_regions:
            if (
                ds.regions[r].connection_inner_x is None
                and ds.regions[r].connection_lower_y is not None
                and ds.regions[r].connection_upper_y is not None
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

        remaining_regions = set(ds.regions) - set(checked_regions)

    # If there are any remaining regions, they should not have any boundaries
    for r in remaining_regions:
        region = ds.regions[r]
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
    result = [xr.DataArray(
        boundary,
        dims=("boundary", "coord"),
        coords={"coord": [Rcoord, Zcoord]},
    ) for boundary in boundaries]

    return result
