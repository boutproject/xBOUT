import warnings

import matplotlib as mpl
import numpy as np
import xarray as xr


def _create_norm(logscale, norm, vmin, vmax):
    if logscale:
        if norm is not None:
            raise ValueError(
                "norm and logscale cannot both be passed at the same time."
            )
        if vmin * vmax > 0:
            # vmin and vmax have the same sign, so can use standard log-scale
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            # vmin and vmax have opposite signs, so use symmetrical logarithmic scale
            if not isinstance(logscale, bool):
                linear_scale = logscale
            else:
                linear_scale = 1.0e-5
            linear_threshold = min(abs(vmin), abs(vmax)) * linear_scale
            norm = mpl.colors.SymLogNorm(linear_threshold, vmin=vmin, vmax=vmax)
    elif norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return norm


def plot_separatrix(da, sep_pos, ax, radial_coord="x"):
    """
    Plots the separatrix as a black dotted line.

    Should plot in the correct place regardless of the choice of coordinates
    for the plot axes, and the type of plot. sep_x needs to be supplied as the
    integer index of the grid point location of the separatrix.
    """

    # TODO Maybe use a decorator to do this for any type of plot?

    # 2D domain needs to intersect the separatrix plane to be able to plot it
    dims = da.dims
    if radial_coord not in dims:

        warnings.warn(
            "Cannot plot separatrix as domain does not cross "
            "separatrix, as it does not have a radial dimension",
            Warning,
        )
        return

    else:
        # Determine the separatrix position in terms of the radial coordinate
        # being used in the plot
        x_coord_vals = da.coords[radial_coord].values
        sep_x_pos = x_coord_vals[sep_pos]

        # Plot a vertical line at that location on the plot
        ax.axvline(x=sep_x_pos, linewidth=2, color="black", linestyle="--")

        return ax


def _decompose_regions(da):

    if da.geometry == "fci":
        return {region: da for region in da.bout._regions}
    return {
        region: da.bout.from_region(region, with_guards=1)
        for region in da.bout._regions
    }


def _is_core_only(da):

    nx = da.metadata["nx"]
    ix1 = da.metadata["ixseps1"]
    ix2 = da.metadata["ixseps2"]

    return ix1 >= nx and ix2 >= nx


def plot_separatrices(da, ax, *, x="R", y="Z"):
    """Plot separatrices"""

    if not isinstance(da, dict):
        da_regions = _decompose_regions(da)
    else:
        da_regions = da

    da0 = list(da_regions.values())[0]

    xcoord = da0.metadata["bout_xdim"]
    ycoord = da0.metadata["bout_ydim"]

    for da_region in da_regions.values():
        inner = list(da_region.bout._regions.values())[0].connection_inner_x
        if inner in da_regions:
            da_inner = da_regions[inner]

            try:
                da_region, da_inner = xr.align(da_region, da_inner)
            except ValueError:
                # For geometries with a limiter, the closed field-line region may have
                # guard cells while the open field line region does not. Also the
                # closed-field line guard cells may (if the region is connected to
                # itself) have duplicated coordinate values, which xr.align() cannot
                # handle. Use np.unique() to remove the duplicated coordinate values
                _, unique_yinds = np.unique(da_inner[ycoord], return_index=True)
                da_inner = da_inner.isel(**{ycoord: unique_yinds})

            # Put da_inner second as the unique_yinds selection may mess up the order of
            # points. xarray will align the coordinates with the first argument (to the
            # addition here).
            x_sep = 0.5 * (
                da_region[x].isel(**{xcoord: 0}) + da_inner[x].isel(**{xcoord: -1})
            )
            y_sep = 0.5 * (
                da_region[y].isel(**{xcoord: 0}) + da_inner[y].isel(**{xcoord: -1})
            )
            ax.plot(x_sep, y_sep, "k--")


def plot_targets(da, ax, *, x="R", y="Z", hatching=True):
    """Plot divertor and limiter target plates"""

    if not isinstance(da, dict):
        da_regions = _decompose_regions(da)
    else:
        da_regions = da

    da0 = list(da_regions.values())[0]

    xcoord = da0.metadata["bout_xdim"]
    ycoord = da0.metadata["bout_ydim"]

    if da0.metadata["keep_yboundaries"]:
        y_boundary_guards = da0.metadata["MYG"]
    else:
        y_boundary_guards = 0

    for da_region in da_regions.values():
        if list(da_region.bout._regions.values())[0].connection_lower_y is None:
            # lower target exists
            x_target = da_region.coords[x].isel(**{ycoord: y_boundary_guards})
            y_target = da_region.coords[y].isel(**{ycoord: y_boundary_guards})
            [line] = ax.plot(x_target, y_target, "k-", linewidth=2)
            if hatching:
                _add_hatching(line, ax)
        if list(da_region.bout._regions.values())[0].connection_upper_y is None:
            # upper target exists
            x_target = da_region.coords[x].isel(**{ycoord: -y_boundary_guards - 1})
            y_target = da_region.coords[y].isel(**{ycoord: -y_boundary_guards - 1})
            [line] = ax.plot(x_target, y_target, "k-", linewidth=2)
            if hatching:
                _add_hatching(line, ax, reversed=True)


def _add_hatching(line, ax, reversed=False):
    """
    Adds a series of angled ticks to target plate lines to give a hatching
    effect, indicative of a solid surface
    """

    x = line.get_xdata()
    y = line.get_ydata()

    if reversed:
        x = np.flip(x)
        y = np.flip(y)

    # TODO redo this to evenly space ticks by physical distance along line
    num_hatchings = 3
    step = len(x) // num_hatchings
    hatch_inds = np.arange(0, len(x), step)

    vx, vy = x.max() - x.min(), y.max() - y.min()
    limiter_line_length = np.linalg.norm((vx, vy))
    hatch_line_length = limiter_line_length / num_hatchings

    # For each hatching
    for ind in hatch_inds[:-1]:
        # Compute local perpendicular vector
        dx, dy = _get_perp_vec(
            (x[ind], y[ind]), (x[ind + 1], y[ind + 1]), magnitude=hatch_line_length
        )

        # Rotate by 60 degrees
        t = -np.pi / 3
        new_dx = dx * np.cos(t) - dy * np.sin(t)
        new_dy = dx * np.sin(t) + dy * np.cos(t)

        # Draw
        ax.plot([x[ind], x[ind] + new_dx], [y[ind], y[ind] + new_dy], "k-")


def _get_perp_vec(u1, u2, magnitude=0.04):
    """Return the vector perpendicular to the vector u2-u1."""

    x1, y1 = u1
    x2, y2 = u2
    vx, vy = x2 - x1, y2 - y1
    v = np.linalg.norm((vx, vy))
    wx, wy = -vy / v * magnitude, vx / v * magnitude
    return wx, wy


def _make_structured_triangulation(m, n):
    # Construct a 'triangulation' of an (m x n) grid
    # Returns an array of triples of (flattened) indices of the corners of the triangles
    indices = np.zeros((m - 1, 2 * (n - 1), 3), dtype=np.uint32)
    # indices[0] = [0, 1, n]
    # indices[1] = [1, n+1, n]

    # Each quadrilateral grid cell with corners (i,j), (i,j+1), (i+1,j+1), (i+1,j), at
    # flattened index I=i*n+j is split into two triangles, one with corners (i,j),
    # (i,j+1), (i+1,j) and the other with corners (i,j+1) (i+1,j+1), (i+1,j),

    lower_left_indices = (
        n * np.arange(m - 1, dtype=np.uint32)[:, np.newaxis]
        + np.arange((n - 1), dtype=np.uint32)[np.newaxis, :]
    )
    # Lower left triangles
    # (i,j) corner
    indices[:, ::2, 0] = lower_left_indices
    # (i,j+1) corner
    indices[:, ::2, 1] = lower_left_indices + 1
    # (i+1,j) corner
    indices[:, ::2, 2] = lower_left_indices + n

    # Upper right triangles
    # (i,j+1) corner
    indices[:, 1::2, 0] = lower_left_indices + 1
    # (i+1,j+1) corner
    indices[:, 1::2, 1] = lower_left_indices + n + 1
    # (i+1,j) corner
    indices[:, 1::2, 2] = lower_left_indices + n

    return indices.reshape((-1, 3))


def _k3d_plot_isel(da_region, isel, vmin, vmax, **kwargs):
    import k3d

    da_sel = da_region.isel(isel)
    X = da_sel["X_cartesian"].values.flatten().astype(np.float32)
    Y = da_sel["Y_cartesian"].values.flatten().astype(np.float32)
    Z = da_sel["Z_cartesian"].values.flatten().astype(np.float32)
    data = da_sel.values.flatten().astype(np.float32)

    indices = _make_structured_triangulation(*da_sel.shape)

    return k3d.mesh(
        np.vstack([X, Y, Z]).T,
        indices,
        attribute=data,
        color_range=[vmin, vmax],
        **kwargs
    )
