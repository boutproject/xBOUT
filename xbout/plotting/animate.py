import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
import animatplot as amp

from .utils import (
    _create_norm,
    _decompose_regions,
    _is_core_only,
    plot_separatrices,
    plot_targets,
)
from matplotlib.animation import PillowWriter


if (
    "pcolor.shading" in matplotlib.rcParams
    and matplotlib.rcParams["pcolor.shading"] == "flat"
):
    # "flat" was the old matplotlib default which discarded the last row and column if
    # X, Y and Z were all equal in size. The new "auto" should be better. Need to set
    # this explicitly because "flat" is the default during a deprecation cycle.
    matplotlib.rcParams["pcolor.shading"] = "auto"


def _add_controls(anim, controls, t_label):
    if controls == "both":
        # Add both time slider and play/pause toggle
        anim.controls(timeline_slider_args={"text": t_label})
    elif controls == "timeline":
        # Add time slider
        anim.timeline_slider(text=t_label)
    elif controls == "toggle":
        # Add play/pause toggle
        anim.toggle()
    elif controls is None or controls == "":
        # Add no controls
        pass
    else:
        raise ValueError(f"Unrecognised value for controls={controls}")


def _parse_coord_option(coord, axis_coords, da):
    if isinstance(axis_coords, dict):
        option_value = axis_coords.get(coord, None)
    else:
        option_value = axis_coords

    if option_value is None:
        c = da[coord]
        if "long_name" in c.attrs:
            label = c.long_name
        else:
            label = coord
        if "units" in c.attrs:
            label = label + f" [{c.units}]"
        return c, label
    elif option_value == "index":
        return np.arange(da.sizes[coord]), f"{coord} index"
    elif isinstance(option_value, str):
        c = da[option_value]
        if "long_name" in c.attrs:
            label = c.long_name
        else:
            label = option_value
        if "units" in c.attrs:
            label = label + f" [{c.units}]"
        return c, label
    else:
        return option_value, None


def _normalise_time_coord(time_values):
    """
    amp.Timeline() does not do a good job of displaying time values that are very small
    (they get rounded to zero). This function scales the time values by a power of 10 to
    get nice values and modifies the units by the scale factor.
    """
    tmax = time_values.max()
    if tmax < 1.0e-2 or tmax > 1.0e6:
        scale_pow = int(np.floor(np.log10(tmax)))
        scale_factor = 10**scale_pow
        time_values = time_values / scale_factor
        suffix = f"e{scale_pow}"
    else:
        suffix = ""

    return time_values, suffix


def animate_poloidal(
    da,
    *,
    ax=None,
    cax=None,
    animate_over=None,
    separatrix=True,
    separatrix_kwargs=dict(),
    targets=True,
    add_limiter_hatching=True,
    cmap=None,
    axis_coords=None,
    vmin=None,
    vmax=None,
    logscale=False,
    animate=True,
    save_as=None,
    fps=10,
    controls="both",
    aspect=None,
    extend=None,
    **kwargs,
):
    """
    Make a 2D plot in R-Z coordinates using animatplotlib's Pcolormesh, taking into
    account branch cuts (X-points).

    Parameters
    ----------
    da : xarray.DataArray
        A 2D (x,y) DataArray of data to plot
    ax : Axes, optional
        A matplotlib axes instance to plot to. If None, create a new
        figure and axes, and plot to that
    cax : Axes, optional
        Matplotlib axes instance where the colorbar will be plotted. If None, the default
        position created by matplotlab.figure.Figure.colorbar() will be used.
    animate_over : str, optional
        Dimension over which to animate, defaults to the time dimension
    separatrix : bool, optional
        Add dashed lines showing separatrices
    separatrix_kwargs : dict, optional
        Options to pass to the separatrix plotter (e.g. line color)
    targets : bool, optional
        Draw solid lines at the target surfaces
    add_limiter_hatching : bool, optional
        Draw hatched areas at the targets
    cmap : matplotlib.colors.Colormap instance, optional
        Colors to use for the plot
    axis_coords : None, str, dict
        Coordinates to use for axis labelling. Only affects time coordinate.

        - None: Use the dimension coordinate for each axis, if it exists.
        - "index": Use the integer index values.
        - dict: keys are dimension names, values set axis_coords for each axis
          separately. Values can be: None, "index", the name of a 1d variable or
          coordinate (which must have the dimension given by 'key'), or a 1d
          numpy array, dask array or DataArray whose length matches the length of
          the dimension given by 'key'.
    vmin : float, optional
        Minimum value for the color scale
    vmax : float, optional
        Maximum value for the color scale
    logscale : bool or float, optional
        If True, default to a logarithmic color scale instead of a linear one.
        If a non-bool type is passed it is treated as a float used to set the linear
        threshold of a symmetric logarithmic scale as
        linthresh=min(abs(vmin),abs(vmax))*logscale, defaults to 1e-5 if True is
        passed.
    animate : bool, optional
        If set to false, do not create the animation, just return the blocks
    save_as : True or str, optional
        If str is passed, save the animation as save_as+'.gif'.
        If True is passed, save the animation with a default name,
        '<variable name>_over_<animate_over>.gif'
    fps : float, optional
        Frame rate for the animation
    controls : string or None, default "both"
        By default, add both the timeline and play/pause toggle to the animation. If
        "timeline" is passed add only the timeline, if "toggle" is passed add only the
        play/pause toggle. If None or an empty string is passed, add neither.
    aspect : str or None, optional
        Argument to set_aspect(), defaults to "equal"
    extend : str or None, optional
        Passed to fig.colorbar()
    **kwargs : optional
        Additional arguments are passed on to the animation method
        animatplot.blocks.Pcolormesh

    Returns
    -------
    animation or blocks
        If animate==True, returns an animatplot.Animation object, otherwise
        returns a list of animatplot.blocks.Pcolormesh instances.
    """

    if animate_over is None:
        animate_over = da.metadata.get("bout_tdim", "t")

    if aspect is None:
        aspect = "equal"

    # TODO generalise this
    x = kwargs.pop("x", "R")
    y = kwargs.pop("y", "Z")

    # Check plot is the right orientation
    spatial_dims = list(da.dims)

    try:
        spatial_dims.remove(animate_over)
    except ValueError:
        raise ValueError(
            "Dimension animate_over={} is not present in the data".format(animate_over)
        )

    if len(da.dims) != 3:
        raise ValueError("da must be 2+1D (t,x,y)")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if vmin is None:
        vmin = da.min().values
    if vmax is None:
        vmax = da.max().values

    if extend is None:
        # Replicate default for older matplotlib that does not handle extend=None
        # matplotlib-3.3 definitely does not need this. Not sure about 3.0, 3.1, 3.2.
        extend = "neither"

    # create colorbar
    norm = _create_norm(logscale, kwargs.pop("norm", None), vmin, vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cmap = sm.get_cmap()
    cbar = fig.colorbar(sm, ax=ax, cax=cax, extend=extend)
    if "long_name" in da.attrs:
        cbar_label = da.long_name
    else:
        cbar_label = da.name
    if "units" in da.attrs:
        cbar_label += f" [{da.units}]"
    cbar.ax.set_ylabel(cbar_label)

    ax.set_aspect(aspect)

    da_regions = _decompose_regions(da)

    # Plot all regions on same axis
    blocks = []
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
        for da_region in da_regions.values():
            # Load values eagerly otherwise for some reason the plotting takes
            # 100's of times longer - for some reason animatplot does not deal
            # well with dask arrays!
            blocks.append(
                amp.blocks.Pcolormesh(
                    da_region.coords[x].values,
                    da_region.coords[y].values,
                    da_region.values,
                    ax=ax,
                    cmap=cmap,
                    norm=norm,
                    **kwargs,
                )
            )

    ax.set_title(da.name)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if _is_core_only(da):
        separatrix = False
        targets = False

    if separatrix:
        plot_separatrices(da_regions, ax, x=x, y=y, **separatrix_kwargs)

    if targets:
        plot_targets(da_regions, ax, x=x, y=y, hatching=add_limiter_hatching)

    if animate:
        t_values, t_label = _parse_coord_option(animate_over, axis_coords, da)
        t_values, t_suffix = _normalise_time_coord(t_values)

        timeline = amp.Timeline(t_values, fps=fps, units=t_suffix)
        anim = amp.Animation(blocks, timeline)

        _add_controls(anim, controls, t_label)

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(da.name, animate_over)
            anim.save(save_as + ".gif", writer=PillowWriter(fps=fps))

        return anim

    return blocks


def animate_pcolormesh(
    data,
    animate_over=None,
    x=None,
    y=None,
    animate=True,
    axis_coords=None,
    vmin=None,
    vmax=None,
    vsymmetric=False,
    logscale=False,
    fps=10,
    save_as=None,
    ax=None,
    cax=None,
    aspect=None,
    extend=None,
    controls="both",
    **kwargs,
):
    """
    Plots a color plot which is animated with time over the specified
    coordinate.

    Currently only supports 2D+1 data, which it plots with animatplotlib's
    wrapping of matplotlib's pcolormesh.

    Parameters
    ----------
    data : xarray.DataArray
    animate_over : str, optional
        Dimension over which to animate, defaults to the time dimension
    x : str, optional
        Dimension to use on the x axis, default is None - then use the first spatial
        dimension of the data
    y : str, optional
        Dimension to use on the y axis, default is None - then use the second spatial
        dimension of the data
    animate : bool, optional
        If set to false, do not create the animation, just return the block
    axis_coords : None, str, dict
        Coordinates to use for axis labelling.

        - None: Use the dimension coordinate for each axis, if it exists.
        - "index": Use the integer index values.
        - dict: keys are dimension names, values set axis_coords for each axis
          separately. Values can be: None, "index", the name of a 1d variable or
          coordinate (which must have the dimension given by 'key'), or a 1d
          numpy array, dask array or DataArray whose length matches the length of
          the dimension given by 'key'.
    vmin : float, optional
        Minimum value to use for colorbar. Default is to use minimum value of
        data across whole timeseries.
    vmax : float, optional
        Maximum value to use for colorbar. Default is to use maximum value of
        data across whole timeseries.
    vsymmetric : bool, optional
        If set to true, make the color-scale symmetric
    logscale : bool or float, optional
        If True, default to a logarithmic color scale instead of a linear one.
        If a non-bool type is passed it is treated as a float used to set the linear
        threshold of a symmetric logarithmic scale as
        linthresh=min(abs(vmin),abs(vmax))*logscale, defaults to 1e-5 if True is
        passed.
    fps : int, optional
        Frames per second of resulting gif
    save_as : True or str, optional
        If str is passed, save the animation as save_as+'.gif'.
        If True is passed, save the animation with a default name,
        '<variable name>_over_<animate_over>.gif'
    ax : Axes, optional
        A matplotlib axes instance to plot to. If None, create a new
        figure and axes, and plot to that
    cax : Axes, optional
        Matplotlib axes instance where the colorbar will be plotted. If None, the default
        position created by matplotlab.figure.Figure.colorbar() will be used.
    aspect : str or None, optional
        Argument to set_aspect(), defaults to "auto"
    extend : str or None, optional
        Passed to fig.colorbar()
    controls : string or None, default "both"
        By default, add both the timeline and play/pause toggle to the animation. If
        "timeline" is passed add only the timeline, if "toggle" is passed add only the
        play/pause toggle. If None or an empty string is passed, add neither.
    kwargs : dict, optional
        Additional keyword arguments are passed on to the animation function
        animatplot.blocks.Pcolormesh

    Returns
    -------
    animation or block
        If animate==True, returns an animatplot.Animation object, otherwise
        returns an animatplot.blocks.Pcolormesh instance.
    """

    if animate_over is None:
        animate_over = data.metadata.get("bout_tdim", "t")

    if aspect is None:
        aspect = "auto"

    variable = data.name

    # Check plot is the right orientation
    spatial_dims = list(data.dims)

    if len(data.dims) != 3:
        raise ValueError("Data passed to animate_imshow must be 3-dimensional")

    try:
        spatial_dims.remove(animate_over)
    except ValueError:
        raise ValueError(
            "Dimension animate_over={} is not present in the data".format(animate_over)
        )

    if x is None and y is None:
        x, y = spatial_dims
    elif x is None:
        try:
            spatial_dims.remove(y)
        except ValueError:
            raise ValueError("Dimension {} is not present in the data".format(y))
        x = spatial_dims[0]
    elif y is None:
        try:
            spatial_dims.remove(x)
        except ValueError:
            raise ValueError("Dimension {} is not present in the data".format(x))
        y = spatial_dims[0]

    if extend is None:
        # Replicate default for older matplotlib that does not handle extend=None
        # matplotlib-3.3 definitely does not need this. Not sure about 3.0, 3.1, 3.2.
        extend = "neither"

    x_values, x_label = _parse_coord_option(x, axis_coords, data)
    y_values, y_label = _parse_coord_option(y, axis_coords, data)

    data = data.transpose(animate_over, y, x, transpose_coords=True)

    # Load values eagerly otherwise for some reason the plotting takes
    # 100's of times longer - for some reason animatplot does not deal
    # well with dask arrays!
    image_data = data.values

    # If not specified, determine max and min values across entire data series
    if vmax is None:
        vmax = np.max(image_data)
    if vmin is None:
        vmin = np.min(image_data)
    if vsymmetric:
        vmax = max(np.abs(vmin), np.abs(vmax))
        vmin = -vmax
    kwargs["norm"] = _create_norm(logscale, kwargs.get("norm", None), vmin, vmax)

    if not ax:
        fig, ax = plt.subplots()

    ax.set_aspect(aspect)

    # Note: animatplot's Pcolormesh gave strange outputs without passing
    # explicitly x- and y-value arrays, although in principle these should not
    # be necessary.
    ny, nx = image_data.shape[1:]
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
        pcolormesh_block = amp.blocks.Pcolormesh(
            x_values, y_values, image_data, ax=ax, **kwargs
        )

    if animate:
        t_values, t_label = _parse_coord_option(animate_over, axis_coords, data)
        t_values, t_suffix = _normalise_time_coord(t_values)

        timeline = amp.Timeline(t_values, fps=fps, units=t_suffix)
        anim = amp.Animation([pcolormesh_block], timeline)

    cbar = plt.colorbar(pcolormesh_block.quad, ax=ax, cax=cax, extend=extend)
    if "long_name" in data.attrs:
        cbar_label = data.long_name
    else:
        cbar_label = variable
    if "units" in data.attrs:
        cbar_label += f" [{data.units}]"
    cbar.ax.set_ylabel(cbar_label)

    # Add title and axis labels
    ax.set_title(variable)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if animate:
        _add_controls(anim, controls, t_label)

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(variable, animate_over)
            anim.save(save_as + ".gif", writer=PillowWriter(fps=fps))

        return anim

    return pcolormesh_block


def animate_line(
    data,
    animate_over=None,
    animate=True,
    axis_coords=None,
    vmin=None,
    vmax=None,
    logscale=False,
    fps=10,
    save_as=None,
    sep_pos=None,
    ax=None,
    aspect=None,
    controls="both",
    **kwargs,
):
    """
    Plots a line plot which is animated with time.

    Currently only supports 1D+1 data, which it plots with animatplot's Line animation.

    Parameters
    ----------
    data : xarray.DataArray
    animate_over : str, optional
        Dimension over which to animate, defaults to the time dimension
    animate : bool, optional
        If set to false, do not create the animation, just return the block
    axis_coords : None, str, dict
        Coordinates to use for axis labelling.

        - None: Use the dimension coordinate for each axis, if it exists.
        - "index": Use the integer index values.
        - dict: keys are dimension names, values set axis_coords for each axis
          separately. Values can be: None, "index", the name of a 1d variable or
          coordinate (which must have the dimension given by 'key'), or a 1d
          numpy array, dask array or DataArray whose length matches the length of
          the dimension given by 'key'.
    vmin : float, optional
        Minimum value to use for colorbar. Default is to use minimum value of
        data across whole timeseries.
    vmax : float, optional
        Maximum value to use for colorbar. Default is to use maximum value of
        data across whole timeseries.
    logscale : bool or float, optional
        If True, default to a logarithmic color scale instead of a linear one.
        If a non-bool type is passed it is treated as a float used to set the linear
        threshold of a symmetric logarithmic scale as
        linthresh=min(abs(vmin),abs(vmax))*logscale, defaults to 1e-5 if True is
        passed.
    fps : int, optional
        Frames per second of resulting gif
    save_as : True or str, optional
        If str is passed, save the animation as save_as+'.gif'.
        If True is passed, save the animation with a default name,
        '<variable name>_over_<animate_over>.gif'
    sep_pos : int, optional
        Radial position at which to plot the separatrix
    ax : Axes, optional
        A matplotlib axes instance to plot to. If None, create a new
        figure and axes, and plot to that
    aspect : str or None, optional
        Argument to set_aspect(), defaults to "auto"
    controls : string or None, default "both"
        By default, add both the timeline and play/pause toggle to the animation. If
        "timeline" is passed add only the timeline, if "toggle" is passed add only the
        play/pause toggle. If None or an empty string is passed, add neither.
    kwargs : dict, optional
        Additional keyword arguments are passed on to the plotting function
        animatplot.blocks.Line

    Returns
    -------
    animation or block
        If animate==True, returns an animatplot.Animation object, otherwise
        returns an animatplot.blocks.Line instance.
    """

    if animate_over is None:
        animate_over = data.metadata.get("bout_tdim", "t")

    if aspect is None:
        aspect = "auto"

    variable = data.name

    # Check plot is the right orientation
    t_read, x_read = data.dims
    if t_read == animate_over:
        x = x_read
    else:
        data = data.transpose(animate_over, t_read, transpose_coords=True)
        x = t_read

    # Load values eagerly otherwise for some reason the plotting takes
    # 100's of times longer - for some reason animatplot does not deal
    # well with dask arrays!
    image_data = data.values

    # If not specified, determine max and min values across entire data series
    if vmax is None:
        vmax = np.max(image_data)
    if vmin is None:
        vmin = np.min(image_data)

    x_values, x_label = _parse_coord_option(x, axis_coords, data)

    if not ax:
        fig, ax = plt.subplots()

    ax.set_aspect(aspect)

    # set range of plot
    ax.set_ylim([vmin, vmax])

    line_block = amp.blocks.Line(x_values, image_data, ax=ax, **kwargs)

    if animate:
        t_values, t_label = _parse_coord_option(animate_over, axis_coords, data)
        t_values, t_suffix = _normalise_time_coord(t_values)

        timeline = amp.Timeline(t_values, fps=fps, units=t_suffix)
        anim = amp.Animation([line_block], timeline)

    # Add title and axis labels
    ax.set_title(variable)
    ax.set_xlabel(x_label)
    if "long_name" in data.attrs:
        y_label = data.long_name
    else:
        y_label = variable
    if "units" in data.attrs:
        y_label = y_label + f" [{data.units}]"
    ax.set_ylabel(y_label)

    if logscale:
        if vmin * vmax > 0.0:
            ax.set_yscale("log")
        else:
            if not isinstance(logscale, bool):
                linear_scale = logscale
            else:
                linear_scale = 1.0e-5
            linear_threshold = min(abs(vmin), abs(vmax)) * linear_scale
            ax.set_yscale("symlog", linthresh=linear_threshold)

    # Plot separatrix
    if sep_pos:
        ax.plot_vline(sep_pos, "--")

    if animate:
        _add_controls(anim, controls, t_label)

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(variable, animate_over)
            anim.save(save_as + ".gif", writer=PillowWriter(fps=fps))

        return anim

    return line_block

def animate_polygon(
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
    animate=True, 
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
    
    colors = da.data[0,:,:].flatten()
    polys.set_array(colors)
    ax.add_collection(polys)
    # function to update the data plotted
    # assuming data in shape (t,x,y)
    def update(frame):
        colors = da.data[frame,:,:].flatten()
        polys.set_array(colors)
        
    if add_colorbar:
        # This produces a "foolproof" colorbar which
        # is always the height of the plot
        # From https://joseph-long.com/writing/colorbars/
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(polys, cax=cax, label=colorbar_label, extend=extend)
        cax.grid(which="both", visible=False)

    

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
    if animate:
        # make the animation by using FuncAnimation and update() to generate frames    
        ani = matplotlib.animation.FuncAnimation(fig=fig, func=update, frames=np.shape(da.data)[0], interval=30)
        return ani
    else:
        # return function for making the animation
        return update

