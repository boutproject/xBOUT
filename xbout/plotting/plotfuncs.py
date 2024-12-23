from collections.abc import Sequence
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import xarray as xr

from .utils import (
    _create_norm,
    _decompose_regions,
    _is_core_only,
    _k3d_plot_isel,
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
        xr.full_like(da_region, fill_value=num / len(da_regions))
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
                **kwargs,
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
    **kwargs,
):
    """
    Make a 2D plot using an xarray method, taking into account branch cuts (X-points).

    Wraps `xarray.DataArray.plot` methods, so automatically adds labels.

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
        if isinstance(levels, Sequence):
            levels = np.array(list(levels))
            kwargs["levels"] = levels
            vmin = np.min(levels)
            vmax = np.max(levels)
        else:
            levels = np.linspace(vmin, vmax, levels, endpoint=True)
            # put levels back into kwargs
            kwargs["levels"] = levels

        levels = kwargs.get("levels", 7)
        if isinstance(levels, np.int64):
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
        cbar = fig.colorbar(sm, ax=ax, extend=extend)
        if "long_name" in da.attrs:
            cbar_label = da.long_name
        else:
            cbar_label = da.name
        if "units" in da.attrs:
            cbar_label += f" [{da.units}]"
        cbar.ax.set_ylabel(cbar_label)

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
                norm=norm,
                **kwargs,
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
                da.metadata["bout_xdim"] not in x.dims
                and da.metadata["bout_ydim"] not in x.dims
            ) or (
                da.metadata["bout_xdim"] not in y.dims
                and da.metadata["bout_ydim"] not in y.dims
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


def plot3d(
    da,
    style="surface",
    engine="k3d",
    levels=None,
    outputgrid=(100, 100, 25),
    color_map=None,
    colorbar=True,
    colorbar_font_size=None,
    plot=None,
    save_as=None,
    surface_xinds=None,
    surface_yinds=None,
    surface_zinds=None,
    fps=20.0,
    mayavi_figure=None,
    mayavi_figure_args=None,
    mayavi_view=None,
    **kwargs,
):
    """
    Make a 3d plot

    Warnings
    --------

    3d plotting functionality is still a bit of a work in progress. Bugs are likely, and
    help developing is welcome!

    Parameters
    ----------
    style : {'surface', 'poloidal planes'}
        Type of plot to make:
        - 'surface' plots the outer surface of the DataArray
        - 'poloidal planes' plots each poloidal plane in the DataArray
    engine : {'k3d', 'mayavi'}
        3d plotting library to use
    levels : sequence of (float, float)
        For isosurface, the pairs of (level-value, opacity) to plot
    outputgrid : (int, int, int) or None, optional
        For isosurface or volume plots, the number of points to use in the Cartesian
        (X,Y,Z) grid, that data is interpolated onto for plotting. If None, then do not
        interpolate and treat "bout_xdim", "bout_ydim" and "bout_zdim" coordinates as
        Cartesian (useful for slab simulations).
    color_map : k3d color map, optional
        Color map for k3d plots
    colorbar : bool or dict, default True
        Add a color bar. If a dict is passed, it is passed on to the colorbar() function
        as keyword arguments.
    colorbar_font_size : float, default None
        Set the font size used by the colorbar (for engine="mayavi")
    plot : k3d plot instance, optional
        Existing plot to add new plots to
    save_as : str
        Filename to save figure to. Animations will be saved as a sequence of
        numbered files.
    surface_xinds : (int, int), default None
        Indices to select when plotting radial surfaces. These indices are local to the
        region being plotted, so values will be strange. Recommend using values relative
        to the radial boundaries (i.e. positive for inner boundary and negative for
        outer boundary).
    surface_yinds : (int, int), default None
        Indices to select when plotting poloidal surfaces. These indices are local to the
        region being plotted, so values will be strange. Recommend using values relative
        to the poloidal boundaries (i.e. positive for lower boundaries and negative for
        upper boundaries).
    surface_zinds : (int, int), default None
        Indices to select when plotting toroidal surfaces
    fps : float, default 20
        Frames per second to use when creating an animation.
    mayavi_figure : mayavi.core.scene.Scene, default None
        Existing Mayavi figure to add this plot to.
    mayavi_figure_args : dict, default None
        Arguments to use when creating a new Mayavi figure. Ignored if ``mayavi_figure``
        is passed.
    mayavi_view : (float, float, float), default None
        If set, arguments are passed to `mayavi.mlab.view` to set the
        view when ``engine="mayavi"``
    vmin, vmax : float
        vmin and vmax are treated specially. If a float is passed, then it is used for
        vmin/vmax. If the arguments are not passed, then the minimum and maximum of the
        data are used. For an animation, to get minimum and/or maximum calculated
        separately for each frame, pass ``vmin=None`` and/or ``vmax=None`` explicitly.
    **kwargs
        Extra keyword arguments are passed to the backend plotting function
    """

    tcoord = da.metadata["bout_tdim"]
    xcoord = da.metadata["bout_xdim"]
    ycoord = da.metadata["bout_ydim"]
    zcoord = da.metadata["bout_zdim"]

    if tcoord in da.dims:
        animate = True
        if len(da.dims) != 4:
            raise ValueError(
                f"plot3d needs to be passed 3d spatial data. Got {da.dims}."
            )
    else:
        animate = False
        if len(da.dims) != 3:
            raise ValueError(
                f"plot3d needs to be passed 3d spatial data. Got {da.dims}."
            )

    da = da.bout.add_cartesian_coordinates()
    if "vmin" in kwargs:
        vmin = kwargs.pop("vmin")
    else:
        vmin = float(da.min().values)
    if "vmax" in kwargs:
        vmax = kwargs.pop("vmax")
    else:
        vmax = float(da.max().values)

    if engine == "k3d":
        if animate:
            raise ValueError(
                "animation not supported by k3d, do not pass time-dependent DataArray"
            )
        if save_as is not None:
            raise ValueError("save_as not supported by k3d implementation yet")
        if colorbar:
            warnings.warn("colorbar not added to k3d plots yet")

        try:
            import k3d
        except ImportError:
            raise ImportError(
                'Please install the `k3d` package for 3d plotting with `engine="k3d"`'
            )

        if color_map is None:
            color_map = k3d.matplotlib_color_maps.Viridis

        if plot is None:
            plot = k3d.plot()
            return_plot = True
        else:
            return_plot = False

        if style == "isosurface" or style == "volume":
            data = da.copy(deep=True).load()
            datamin = data.min().item()
            datamax = data.max().item()

            if outputgrid is None:
                Xmin = da[da.metadata["bout_xdim"]][0]
                Xmax = da[da.metadata["bout_xdim"]][-1]
                Ymin = da[da.metadata["bout_ydim"]][0]
                Ymax = da[da.metadata["bout_ydim"]][-1]
                Zmin = da[da.metadata["bout_zdim"]][0]
                Zmax = da[da.metadata["bout_zdim"]][-1]

                grid = da.astype(np.float32).values
            else:
                xpoints, ypoints, zpoints = outputgrid
                nx, ny, nz = data.shape

                # interpolate to Cartesian array
                Xmin = data["X_cartesian"].min()
                Xmax = data["X_cartesian"].max()
                Ymin = data["Y_cartesian"].min()
                Ymax = data["Y_cartesian"].max()
                Zmin = data["Z_cartesian"].min()
                Zmax = data["Z_cartesian"].max()
                Rmin = data["R"].min()
                Rmax = data["R"].max()
                Zmin = data["Z"].min()
                Zmax = data["Z"].max()
                newX = xr.DataArray(
                    np.linspace(Xmin, Xmax, xpoints), dims="x"
                ).expand_dims({"y": ypoints, "z": zpoints}, axis=[1, 0])
                newY = xr.DataArray(
                    np.linspace(Ymin, Ymax, ypoints), dims="y"
                ).expand_dims({"x": xpoints, "z": zpoints}, axis=[2, 0])
                newZ = xr.DataArray(
                    np.linspace(Zmin, Zmax, zpoints), dims="z"
                ).expand_dims({"x": xpoints, "y": ypoints}, axis=[2, 1])
                newR = np.sqrt(newX**2 + newY**2)
                newzeta = np.arctan2(newY, newX)  # .values

                from scipy.interpolate import (
                    RegularGridInterpolator,
                    griddata,
                )

                print("start interpolating")
                Rcyl = xr.DataArray(
                    np.linspace(Rmin, Rmax, 2 * zpoints), dims="r"
                ).expand_dims({"z": 2 * zpoints}, axis=1)
                Zcyl = xr.DataArray(
                    np.linspace(Zmin, Zmax, 2 * zpoints), dims="z"
                ).expand_dims({"r": 2 * zpoints}, axis=0)

                # Interpolate in two stages for efficiency. Unstructured 3d interpolation is
                # very slow. Unstructured 2d interpolation onto Cartesian (R, Z) grids,
                # followed by structured 3d interpolation onto the (X, Y, Z) grid, is much
                # faster.
                # Structured 3d interpolation straight from (psi, theta, zeta) to (X, Y, Z)
                # leaves artifacts in the output, because theta does not vary continuously
                # everywhere (has branch cuts).

                # order of dimensions does not really matter here - output only depends on
                # shape of newR, newZ, newzeta. Possibly more efficient to assign the 2d
                # results in the loop to the last two dimensions, so put zeta first.
                data_cyl = np.zeros((nz, 2 * zpoints, 2 * zpoints))
                print("interpolate poloidal planes")
                for z in range(nz):
                    data_cyl[z] = griddata(
                        (data["R"].values.flatten(), data["Z"].values.flatten()),
                        data.isel(zeta=z).values.flatten(),
                        (Rcyl.values, Zcyl.values),
                        method="cubic",
                        fill_value=datamin - 2.0 * (datamax - datamin),
                    )
                print("build 3d interpolator")
                interp = RegularGridInterpolator(
                    (data["zeta"].values, Rcyl.isel(z=0).values, Zcyl.isel(r=0).values),
                    data_cyl,
                    bounds_error=False,
                    fill_value=datamin - 2.0 * (datamax - datamin),
                )
                print("do 3d interpolation")
                grid = interp((newzeta, newR, newZ), method="linear")
                print("done interpolating")

            if style == "isosurface":
                if levels is None:
                    levels = [(0.5 * (datamin + datamax), 1.0)]
                for level, opacity in levels:
                    plot += k3d.marching_cubes(
                        grid.astype(np.float32),
                        bounds=[Xmin, Xmax, Ymin, Ymax, Zmin, Zmax],
                        level=level,
                        color_map=color_map,
                    )
            elif style == "volume":
                plot += k3d.volume(
                    grid.astype(np.float32),
                    color_range=[datamin, datamax],
                    bounds=[Xmin, Xmax, Ymin, Ymax, Zmin, Zmax],
                    color_map=color_map,
                )
            if return_plot:
                return plot
            else:
                return

        for region_name, da_region in _decompose_regions(da).items():
            npsi, ntheta, nzeta = da_region.shape

            if style == "surface":
                region = da_region.regions[region_name]

                if region.connection_inner_x is None:
                    # Plot the inner-x surface
                    plot += _k3d_plot_isel(
                        da_region,
                        {xcoord: 0},
                        vmin,
                        vmax,
                        color_map=color_map,
                        **kwargs,
                    )

                if region.connection_outer_x is None:
                    # Plot the outer-x surface
                    plot += _k3d_plot_isel(
                        da_region,
                        {xcoord: -1},
                        vmin,
                        vmax,
                        color_map=color_map,
                        **kwargs,
                    )

                if region.connection_lower_y is None:
                    # Plot the lower-y surface
                    plot += _k3d_plot_isel(
                        da_region,
                        {ycoord: 0},
                        vmin,
                        vmax,
                        color_map=color_map,
                        **kwargs,
                    )

                if region.connection_upper_y is None:
                    # Plot the upper-y surface
                    plot += _k3d_plot_isel(
                        da_region,
                        {ycoord: -1},
                        vmin,
                        vmax,
                        color_map=color_map,
                        **kwargs,
                    )

                # First z-surface
                plot += _k3d_plot_isel(
                    da_region, {zcoord: 0}, vmin, vmax, color_map=color_map, **kwargs
                )

                # Last z-surface
                plot += _k3d_plot_isel(
                    da_region, {zcoord: -1}, vmin, vmax, color_map=color_map, **kwargs
                )
            elif style == "poloidal planes":
                for zeta in range(nzeta):
                    plot += _k3d_plot_isel(
                        da_region,
                        {zcoord: zeta},
                        vmin,
                        vmax,
                        color_map=color_map,
                        **kwargs,
                    )
            else:
                raise ValueError(f"style='{style}' not implemented for engine='k3d'")

        if return_plot:
            return plot
        else:
            return

    elif engine == "mayavi":
        try:
            from mayavi import mlab
        except ImportError:
            raise ImportError(
                "Please install the `mayavi` package for 3d plotting with "
                '`engine="mayavi"`'
            )

        if mayavi_figure is None:
            if mayavi_figure_args is not None:
                mlab.figure(**mayavi_figure_args)
        else:
            mlab.figure(mayavi_figure)

        if style == "surface":

            def create_or_update_plot(plot_objects=None, tind=None, this_save_as=None):
                if plot_objects is None:
                    # Creating plot for first time
                    plot_objects_to_return = {}
                    this_da = da
                if tind is not None:
                    this_da = da.isel({tcoord: tind})

                for region_name, da_region in _decompose_regions(this_da).items():
                    region = da_region.regions[region_name]

                    # Always include z-surfaces
                    zstart_ind = 0 if surface_zinds is None else surface_zinds[0]
                    zend_ind = -1 if surface_zinds is None else surface_zinds[1]
                    surface_selections = [
                        {this_da.metadata["bout_zdim"]: zstart_ind},
                        {this_da.metadata["bout_zdim"]: zend_ind},
                    ]
                    if region.connection_inner_x is None:
                        # Plot the inner-x surface
                        xstart_ind = 0 if surface_xinds is None else surface_xinds[0]
                        surface_selections.append({xcoord: xstart_ind})
                    if region.connection_outer_x is None:
                        # Plot the outer-x surface
                        xend_ind = -1 if surface_xinds is None else surface_xinds[1]
                        surface_selections.append({xcoord: xend_ind})
                    if region.connection_lower_y is None:
                        # Plot the lower-y surface
                        ystart_ind = 0 if surface_yinds is None else surface_yinds[0]
                        surface_selections.append({ycoord: ystart_ind})
                    if region.connection_upper_y is None:
                        # Plot the upper-y surface
                        yend_ind = -1 if surface_yinds is None else surface_yinds[1]
                        surface_selections.append({ycoord: yend_ind})

                    for i, surface_sel in enumerate(surface_selections):
                        da_sel = da_region.isel(surface_sel)
                        X = da_sel["X_cartesian"].values
                        Y = da_sel["Y_cartesian"].values
                        Z = da_sel["Z_cartesian"].values
                        data = da_sel.values

                        if plot_objects is None:
                            plot_objects_to_return[region_name + str(i)] = mlab.mesh(
                                X, Y, Z, scalars=data, vmin=vmin, vmax=vmax, **kwargs
                            )
                        else:
                            plot_objects[region_name + str(i)].mlab_source.scalars = (
                                data
                            )

                if mayavi_view is not None:
                    mlab.view(*mayavi_view)

                if colorbar and (tind is None or tind == 0):
                    if isinstance(colorbar, dict):
                        colorbar_args = colorbar
                    else:
                        colorbar_args = {}
                    cb = mlab.colorbar(**colorbar_args)
                    if colorbar_font_size is not None:
                        cb.scalar_bar.unconstrained_font_size = True
                        cb.label_text_property.font_size = colorbar_font_size
                        cb.title_text_property.font_size = colorbar_font_size

                if this_save_as:
                    if tind is None:
                        mlab.savefig(this_save_as)
                    else:
                        name_parts = this_save_as.split(".")
                        name_parts = name_parts[:-1] + [str(tind)] + name_parts[-1:]
                        frame_save_as = ".".join(name_parts)
                        mlab.savefig(frame_save_as)
                if plot_objects is None:
                    return plot_objects_to_return

            if animate:
                orig_offscreen_option = mlab.options.offscreen
                mlab.options.offscreen = True

                try:
                    # resets mlab.options.offscreen when it finishes, even if there is
                    # an error
                    if save_as is None:
                        raise ValueError(
                            "Must pass `save_as` for a mayavi animation, or no output will "
                            "be created"
                        )
                    with TemporaryDirectory() as d:
                        nframes = da.sizes[tcoord]

                        # First create png files in the temporary directory
                        temp_path = Path(d)
                        temp_save_as = str(temp_path.joinpath("temp.png"))
                        print("tind=0")
                        plot_objects = create_or_update_plot(
                            tind=0, this_save_as=temp_save_as
                        )

                        # @mlab.animate # interative mayavi animation too slow
                        def animation_func():
                            for tind in range(1, nframes):
                                print(f"tind={tind}")
                                create_or_update_plot(
                                    plot_objects=plot_objects,
                                    tind=tind,
                                    this_save_as=temp_save_as,
                                )
                                # yield # needed for an interactive mayavi animation

                        a = animation_func()

                        # Use ImageMagick via the wand package to turn the .png files into
                        # an animation
                        try:
                            from wand.image import Image
                        except ImportError:
                            raise ImportError(
                                "Please install the `wand` package to save the 3d animation"
                            )
                        with Image() as animation:
                            for i in range(nframes):
                                filename = str(temp_path.joinpath(f"temp.{i}.png"))
                                with Image(filename=filename) as frame:
                                    animation.sequence.append(frame)
                            animation.type = "optimize"
                            animation.loop = 0
                            # Delay is in milliseconds
                            animation.delay = int(round(1000.0 / fps))
                            animation.save(filename=save_as)
                finally:
                    mlab.options.offscreen = orig_offscreen_option

                # mlab.show() # interative mayavi animation so slow that it's not useful
                return a
            else:
                create_or_update_plot(this_save_as=save_as)

        else:
            raise ValueError(f"style='{style}' not implemented for engine='mayavi'")

        plt.show()
    else:
        raise ValueError(f"Unrecognised plot3d() 'engine' argument: {engine}")


def plot2d_polygon(
    da,
    ax=None,
    cax=None,
    cmap="viridis",
    norm=None,
    logscale=False,
    antialias=False,
    vmin=None,
    vmax=None,
    extend="neither",
    add_colorbar=True,
    colorbar_label=None,
    separatrix=True,
    separatrix_kwargs={"color": "white", "linestyle": "-", "linewidth": 2},
    targets=False,
    add_limiter_hatching=False,
    grid_only=False,
    linewidth=0,
    linecolor="black",
):
    """
    Nice looking 2D plots which have no visual artifacts around the X-point.

    Parameters
    ----------
    da : xarray.DataArray
        A 2D (x,y) DataArray of data to plot
    ax :  Axes, optional
        Axes to plot on. If not provided, will make its own.
    cax : Axes, optional
        Axes to plot colorbar on. If not provided, will plot on the same axes as the plot.
    cmap : str or matplotlib.colors.Colormap, default "viridis"
        Colormap to use for the plot
    norm : matplotlib.colors.Normalize, optional
        Normalization to use for the color scale
    logscale : bool, default False
        If True, use a symlog color scale
    antialias : bool, default False
        Enables antialiasing. Note: this also shows mesh cell edges - it's unclear how to disable this.
    vmin : float, optional
        Minimum value for the color scale
    vmax : float, optional
        Maximum value for the color scale
    extend : str, optional, default "neither"
        Extend the colorbar. Options are "neither", "both", "min", "max"
    add_colorbar : bool, default True
        Enable colorbar in figure?
    colorbar_label : str, optional
        Label for the colorbar
    separatrix : bool, default True
        Add lines showing separatrices
    separatrix_kwargs : dict
        Keyword arguments to pass custom style to the separatrices plot
    targets : bool, default True
        Draw solid lines at the target surfaces
    add_limiter_hatching : bool, default True
        Draw hatched areas at the targets
    grid_only : bool, default False
        Only plot the grid, not the data. This sets all the polygons to have a white face.
    linewidth : float, default 0
        Width of the gridlines on cell edges
    linecolor : str, default "black"
        Color of the gridlines on cell edges
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 6), dpi=120)
    else:
        fig = ax.get_figure()

    if cax is None:
        cax = ax

    if vmin is None:
        vmin = np.nanmin(da.values)

    if vmax is None:
        vmax = np.nanmax(da.max().values)

    if colorbar_label is None:
        if "short_name" in da.attrs:
            colorbar_label = da.attrs["short_name"]
        elif da.name is not None:
            colorbar_label = da.name
        else:
            colorbar_label = ""

    if "units" in da.attrs:
        colorbar_label += f" [{da.attrs['units']}]"

    if "Rxy_lower_right_corners" in da.coords:
        r_nodes = [
            "R",
            "Rxy_lower_left_corners",
            "Rxy_lower_right_corners",
            "Rxy_upper_left_corners",
            "Rxy_upper_right_corners",
        ]
        z_nodes = [
            "Z",
            "Zxy_lower_left_corners",
            "Zxy_lower_right_corners",
            "Zxy_upper_left_corners",
            "Zxy_upper_right_corners",
        ]
        cell_r = np.concatenate(
            [np.expand_dims(da[x], axis=2) for x in r_nodes], axis=2
        )
        cell_z = np.concatenate(
            [np.expand_dims(da[x], axis=2) for x in z_nodes], axis=2
        )
    else:
        raise Exception("Cell corners not present in mesh, cannot do polygon plot")

    Nx = len(cell_r)
    Ny = len(cell_r[0])
    patches = []

    # https://matplotlib.org/2.0.2/examples/api/patch_collection.html

    idx = [np.array([1, 2, 4, 3, 1])]
    patches = []
    for i in range(Nx):
        for j in range(Ny):
            p = matplotlib.patches.Polygon(
                np.concatenate((cell_r[i][j][tuple(idx)], cell_z[i][j][tuple(idx)]))
                .reshape(2, 5)
                .T,
                fill=False,
                closed=True,
                facecolor=None,
            )
            patches.append(p)

    norm = _create_norm(logscale, norm, vmin, vmax)

    if grid_only is True:
        cmap = matplotlib.colors.ListedColormap(["white"])
    colors = da.data.flatten()
    polys = matplotlib.collections.PatchCollection(
        patches,
        alpha=1,
        norm=norm,
        cmap=cmap,
        antialiaseds=antialias,
        edgecolors=linecolor,
        linewidths=linewidth,
        joinstyle="bevel",
    )

    polys.set_array(colors)

    if add_colorbar:
        # This produces a "foolproof" colorbar which
        # is always the height of the plot
        # From https://joseph-long.com/writing/colorbars/
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(polys, cax=cax, label=colorbar_label, extend=extend)
        cax.grid(which="both", visible=False)

    ax.add_collection(polys)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_ylim(cell_z.min(), cell_z.max())
    ax.set_xlim(cell_r.min(), cell_r.max())
    ax.set_title(da.name)

    if separatrix:
        plot_separatrices(da, ax, x="R", y="Z", **separatrix_kwargs)

    if targets:
        plot_targets(da, ax, x="R", y="Z", hatching=add_limiter_hatching)
