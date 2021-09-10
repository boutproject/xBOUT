import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings

import xarray as xr

from .utils import (
    _create_norm,
    _decompose_regions,
    _is_core_only,
    plot_separatrices,
    plot_targets,
)


if (
    "pcolor.shading" in matplotlib.rcParams
    and matplotlib.rcParams["pcolor.shading"] == "flat"
):
    # "flat" was the old matplotlib default which discarded the last row and column if
    # X, Y and Z were all equal in size. The new "auto" should be better. Need to set
    # this explicitly because "flat" is the default during a deprecation cycle.
    matplotlib.rcParams["pcolor.shading"] = "auto"


def plot_regions(da, ax=None, **kwargs):
    """
    Plots each logical plotting region as a different color for debugging.

    Uses matplotlib.pcolormesh
    """

    x = kwargs.pop("x", "R")
    y = kwargs.pop("y", "Z")

    if len(da.dims) != 2:
        raise ValueError("da must be 2D (x,y)")

    if ax is None:
        fig, ax = plt.subplots()

    da_regions = _decompose_regions(da)

    colored_regions = [
        xr.full_like(da_region, fill_value=num / len(regions))
        for num, da_region in enumerate(da_regions.values())
    ]

    with warnings.catch_warnings():
        # The coordinates we pass are a logically rectangular grid, so should be fine
        # even if this warning is triggered.
        warnings.filterwarnings(
            "ignore",
            "The input coordinates to pcolormesh are interpreted as cell centers, but "
            "are not monotonically increasing or decreasing. This may lead to "
            "incorrectly calculated cell edges, in which case, please supply explicit "
            "cell edges to pcolormesh.",
            UserWarning,
        )
        result = [
            region.plot.pcolormesh(
                x=x,
                y=y,
                vmin=0,
                vmax=1,
                cmap="tab20",
                infer_intervals=False,
                add_colorbar=False,
                ax=ax,
                **kwargs
            )
            for region in colored_regions
        ]

    return result


def plot2d_wrapper(
    da,
    method,
    *,
    ax=None,
    separatrix=True,
    targets=True,
    add_limiter_hatching=True,
    gridlines=None,
    cmap=None,
    norm=None,
    logscale=None,
    vmin=None,
    vmax=None,
    aspect=None,
    extend=None,
    **kwargs
):
    """
    Make a 2D plot using an xarray method, taking into account branch cuts (X-points).

    Wraps `xarray.plot` methods, so automatically adds labels.

    Parameters
    ----------
    da : xarray.DataArray
        A 2D (x,y) DataArray of data to plot
    method : xarray.plot.*
        An xarray plotting method to use
    ax : Axes, optional
        A matplotlib axes instance to plot to. If None, create a new
        figure and axes, and plot to that
    separatrix : bool, optional
        Add dashed lines showing separatrices
    targets : bool, optional
        Draw solid lines at the target surfaces
    add_limiter_hatching : bool, optional
        Draw hatched areas at the targets
    gridlines : bool, int or slice or dict of bool, int or slice, optional
        If True, draw grid lines on the plot. If an int is passed, it is used as the
        stride when plotting grid lines (to reduce the number on the plot). If a slice is
        passed it is used to select the grid lines to plot.
        If a dict is passed, the 'x' entry (bool, int or slice) is used for the radial
        grid-lines and the 'y' entry for the poloidal grid lines.
    cmap : Matplotlib colormap, optional
        Color map to use for the plot
    norm : matplotlib.colors.Normalize instance, optional
        Normalization to use for the color scale.
        Cannot be set at the same time as 'logscale'
    logscale : bool or float, optional
        If True, default to a logarithmic color scale instead of a linear one.
        If a non-bool type is passed it is treated as a float used to set the linear
        threshold of a symmetric logarithmic scale as
        linthresh=min(abs(vmin),abs(vmax))*logscale, defaults to 1e-5 if True is passed.
        Cannot be set at the same time as 'norm'
    vmin : float, optional
        Minimum value for the color scale
    vmax : float, optional
        Maximum value for the color scale
    aspect : str or float, optional
        Passed to ax.set_aspect(). By default 'equal' is used.
    extend : str or None, optional
        Passed to fig.colorbar()
    levels : int or iterable, optional
        Only used by contour or contourf, sets the number of levels (if int) or the level
        values (if iterable)
    **kwargs : optional
        Additional arguments are passed on to method

    Returns
    -------
    artists
        List of the artist instances
    """

    # TODO generalise this
    x = kwargs.pop("x", "R")
    y = kwargs.pop("y", "Z")

    if len(da.dims) != 2:
        raise ValueError("da must be 2D (x,y)")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if aspect is None:
        aspect = "equal"
    ax.set_aspect(aspect)

    if vmin is None:
        vmin = da.min().values
    if vmax is None:
        vmax = da.max().values

    if extend is None:
        # Replicate default for older matplotlib that does not handle extend=None
        # matplotlib-3.3 definitely does not need this. Not sure about 3.0, 3.1, 3.2.
        extend = "neither"

    # set up 'levels' if needed
    if method is xr.plot.contourf or method is xr.plot.contour:
        levels = kwargs.get("levels", 7)
        if isinstance(levels, np.int):
            levels = np.linspace(vmin, vmax, levels, endpoint=True)
            # put levels back into kwargs
            kwargs["levels"] = levels
        else:
            levels = np.array(list(levels))
            kwargs["levels"] = levels
            vmin = np.min(levels)
            vmax = np.max(levels)

        levels = kwargs.get("levels", 7)
        if isinstance(levels, np.int):
            levels = np.linspace(vmin, vmax, levels, endpoint=True)
            # put levels back into kwargs
            kwargs["levels"] = levels
        else:
            levels = np.array(list(levels))
            kwargs["levels"] = levels
            vmin = np.min(levels)
            vmax = np.max(levels)

    # Need to create a colorscale that covers the range of values in the whole array.
    # Using the add_colorbar argument would create a separate color scale for each
    # separate region, which would not make sense.
    if method is xr.plot.contourf:
        # create colorbar
        norm = _create_norm(logscale, norm, vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        # make colorbar have only discrete levels
        # average the levels so that colors in the colorbar represent the intervals
        # between the levels, as contourf colors filled regions between the given levels.
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "discrete cmap",
            sm.to_rgba(0.5 * (levels[:-1] + levels[1:])),
            len(levels) - 1,
        )
        # re-make sm with new cmap
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ticks=levels, ax=ax, extend=extend)
    elif method is xr.plot.contour:
        # create colormap to be shared by all regions
        norm = matplotlib.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cmap = matplotlib.colors.ListedColormap(
            sm.to_rgba(levels), name="discrete cmap"
        )
    else:
        # pass vmin and vmax through kwargs as they are not used for contourf or contour
        # plots
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

        # create colorbar
        norm = _create_norm(logscale, norm, vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cmap = sm.get_cmap()
        fig.colorbar(sm, ax=ax, extend=extend)

    if method is xr.plot.pcolormesh:
        if "infer_intervals" not in kwargs:
            kwargs["infer_intervals"] = False

    da_regions = _decompose_regions(da)

    # Plot all regions on same axis
    add_labels = [True] + [False] * (len(da_regions) - 1)
    with warnings.catch_warnings():
        # The coordinates we pass are a logically rectangular grid, so should be fine
        # even if this warning is triggered by pcolor or pcolormesh
        warnings.filterwarnings(
            "ignore",
            "The input coordinates to pcolormesh are interpreted as cell centers, but "
            "are not monotonically increasing or decreasing. This may lead to "
            "incorrectly calculated cell edges, in which case, please supply explicit "
            "cell edges to pcolormesh.",
            UserWarning,
        )
        artists = [
            method(
                region,
                x=x,
                y=y,
                ax=ax,
                add_colorbar=False,
                add_labels=add_label,
                cmap=cmap,
                **kwargs
            )
            for region, add_label in zip(da_regions.values(), add_labels)
        ]

    if method is xr.plot.contour:
        fig.colorbar(artists[0], ax=ax)

    if gridlines is not None:
        # convert gridlines to dict
        if not isinstance(gridlines, dict):
            gridlines = {"x": gridlines, "y": gridlines}

        for key, value in gridlines.items():
            if value is True:
                gridlines[key] = slice(None)
            elif isinstance(value, int):
                gridlines[key] = slice(0, None, value)
            elif value is not None:
                if not isinstance(value, slice):
                    raise ValueError(
                        "Argument passed to gridlines must be bool, int or "
                        "slice. Got a " + type(value) + ", " + str(value)
                    )

        x_regions = [da_region[x] for da_region in da_regions.values()]
        y_regions = [da_region[y] for da_region in da_regions.values()]

        for x, y in zip(x_regions, y_regions):
            if (
                not da.metadata["bout_xdim"] in x.dims
                and not da.metadata["bout_ydim"] in x.dims
            ) or (
                not da.metadata["bout_xdim"] in y.dims
                and not da.metadata["bout_ydim"] in y.dims
            ):
                # Small regions around X-point do not have segments in x- or y-directions,
                # so skip
                # Currently this region does not exist, but there is a small white gap at
                # the X-point, so we might add it back in future
                continue
            if gridlines.get("x") is not None:
                # transpose in case Dataset or DataArray has been transposed away from the usual
                # form
                dim_order = (da.metadata["bout_xdim"], da.metadata["bout_ydim"])
                yarg = {da.metadata["bout_ydim"]: gridlines["x"]}
                plt.plot(
                    x.isel(**yarg).transpose(*dim_order, transpose_coords=True),
                    y.isel(**yarg).transpose(*dim_order, transpose_coords=True),
                    color="k",
                    lw=0.1,
                )
            if gridlines.get("y") is not None:
                xarg = {da.metadata["bout_xdim"]: gridlines["y"]}
                # Need to plot transposed arrays to make gridlines that go in the
                # y-direction
                dim_order = (da.metadata["bout_ydim"], da.metadata["bout_xdim"])
                plt.plot(
                    x.isel(**xarg).transpose(*dim_order, transpose_coords=True),
                    y.isel(**yarg).transpose(*dim_order, transpose_coords=True),
                    color="k",
                    lw=0.1,
                )

    ax.set_title(da.name)

    if _is_core_only(da):
        separatrix = False
        targets = False

    if separatrix:
        plot_separatrices(da_regions, ax, x=x, y=y)

    if targets:
        plot_targets(da_regions, ax, x=x, y=y, hatching=add_limiter_hatching)

    return artists
