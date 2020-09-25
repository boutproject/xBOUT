import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

import animatplot as amp

from .utils import _decompose_regions, _is_core_only, plot_separatrices, plot_targets
from matplotlib.animation import PillowWriter


def animate_poloidal(
    da,
    *,
    ax=None,
    cax=None,
    animate_over="t",
    separatrix=True,
    targets=True,
    add_limiter_hatching=True,
    cmap=None,
    vmin=None,
    vmax=None,
    animate=True,
    save_as=None,
    fps=10,
    controls=True,
    aspect="equal",
    extend=None,
    **kwargs
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
    separatrix : bool, optional
        Add dashed lines showing separatrices
    targets : bool, optional
        Draw solid lines at the target surfaces
    add_limiter_hatching : bool, optional
        Draw hatched areas at the targets
    cmap : matplotlib.colors.Colormap instance, optional
        Colors to use for the plot
    vmin : float, optional
        Minimum value for the color scale
    vmax : float, optional
        Maximum value for the color scale
    animate : bool, optional
        If set to false, do not create the animation, just return the blocks
    save_as : True or str, optional
        If str is passed, save the animation as save_as+'.gif'.
        If True is passed, save the animation with a default name,
        '<variable name>_over_<animate_over>.gif'
    fps : float, optional
        Frame rate for the animation
    controls : bool, optional
        If False, do not add the timeline and pause button to the animation
    aspect : str or None, optional
        Argument to set_aspect()
    extend : str or None, optional
        Passed to fig.colorbar()
    **kwargs : optional
        Additional arguments are passed on to the animation method
        animatplot.blocks.Pcolormesh

    Returns
    -------
    blocks
        List of animatplot.blocks.Pcolormesh instances
    """

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
    norm = (
        kwargs["norm"]
        if "norm" in kwargs
        else matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cmap = sm.get_cmap()
    fig.colorbar(sm, ax=ax, cax=cax, extend=extend)

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
                    **kwargs
                )
            )

    ax.set_title(da.name)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if _is_core_only(da):
        separatrix = False
        targets = False

    if separatrix:
        plot_separatrices(da_regions, ax, x=x, y=y)

    if targets:
        plot_targets(da_regions, ax, x=x, y=y, hatching=add_limiter_hatching)

    if animate:
        timeline = amp.Timeline(np.arange(da.sizes[animate_over]), fps=fps)
        anim = amp.Animation(blocks, timeline)

        if controls:
            anim.controls(timeline_slider_args={"text": animate_over})

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(da.name, animate_over)
            anim.save(save_as + ".gif", writer=PillowWriter(fps=fps))

    return blocks


def animate_pcolormesh(
    data,
    animate_over="t",
    x=None,
    y=None,
    animate=True,
    vmin=None,
    vmax=None,
    vsymmetric=False,
    fps=10,
    save_as=None,
    ax=None,
    cax=None,
    extend=None,
    controls=True,
    **kwargs
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
        Dimension over which to animate
    x : str, optional
        Dimension to use on the x axis, default is None - then use the first spatial
        dimension of the data
    y : str, optional
        Dimension to use on the y axis, default is None - then use the second spatial
        dimension of the data
    animate : bool, optional
        If set to false, do not create the animation, just return the block
    vmin : float, optional
        Minimum value to use for colorbar. Default is to use minimum value of
        data across whole timeseries.
    vmax : float, optional
        Maximum value to use for colorbar. Default is to use maximum value of
        data across whole timeseries.
    vsymmetric : bool, optional
        If set to true, make the color-scale symmetric
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
    extend : str or None, optional
        Passed to fig.colorbar()
    controls : bool, optional
        If False, do not add the timeline and pause button to the animation
    kwargs : dict, optional
        Additional keyword arguments are passed on to the animation function
        animatplot.blocks.Pcolormesh
    """

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

    data = data.transpose(animate_over, y, x, transpose_coords=True)

    # Load values eagerly otherwise for some reason the plotting takes
    # 100's of times longer - for some reason animatplot does not deal
    # well with dask arrays!
    image_data = data.values

    if "norm" not in kwargs:
        # If not specified, determine max and min values across entire data series
        if vmax is None:
            vmax = np.max(image_data)
        if vmin is None:
            vmin = np.min(image_data)
        if vsymmetric:
            vmax = max(np.abs(vmin), np.abs(vmax))
            vmin = -vmax
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    if not ax:
        fig, ax = plt.subplots()

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
            np.arange(float(nx)), np.arange(float(ny)), image_data, ax=ax, **kwargs
        )

    if animate:
        timeline = amp.Timeline(np.arange(data.sizes[animate_over]), fps=fps)
        anim = amp.Animation([pcolormesh_block], timeline)

    cbar = plt.colorbar(pcolormesh_block.quad, ax=ax, cax=cax, extend=extend)
    cbar.ax.set_ylabel(variable)

    # Add title and axis labels
    ax.set_title(variable)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if animate:
        if controls:
            anim.controls(timeline_slider_args={"text": animate_over})

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(variable, animate_over)
            anim.save(save_as + ".gif", writer=PillowWriter(fps=fps))

    return pcolormesh_block


def animate_line(
    data,
    animate_over="t",
    animate=True,
    vmin=None,
    vmax=None,
    fps=10,
    save_as=None,
    sep_pos=None,
    ax=None,
    controls=True,
    **kwargs
):
    """
    Plots a line plot which is animated with time.

    Currently only supports 1D+1 data, which it plots with animatplot's Line animation.

    Parameters
    ----------
    data : xarray.DataArray
    animate_over : str, optional
        Dimension over which to animate
    animate : bool, optional
        If set to false, do not create the animation, just return the block
    vmin : float, optional
        Minimum value to use for colorbar. Default is to use minimum value of
        data across whole timeseries.
    vmax : float, optional
        Maximum value to use for colorbar. Default is to use maximum value of
        data across whole timeseries.
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
    controls : bool, optional
        If False, do not add the timeline and pause button to the animation
    kwargs : dict, optional
        Additional keyword arguments are passed on to the plotting function
        animatplot.blocks.Line
    """

    variable = data.name

    # Check plot is the right orientation
    t_read, x_read = data.dims
    if t_read is animate_over:
        pass
    else:
        data = data.transpose(animate_over, t_read, transpose_coords=True)

    # Load values eagerly otherwise for some reason the plotting takes
    # 100's of times longer - for some reason animatplot does not deal
    # well with dask arrays!
    image_data = data.values

    # If not specified, determine max and min values across entire data series
    if vmax is None:
        vmax = np.max(image_data)
    if vmin is None:
        vmin = np.min(image_data)

    if not ax:
        fig, ax = plt.subplots()

    # set range of plot
    ax.set_ylim([vmin, vmax])

    line_block = amp.blocks.Line(image_data, ax=ax, **kwargs)

    if animate:
        timeline = amp.Timeline(np.arange(data.sizes[animate_over]), fps=fps)
        anim = amp.Animation([line_block], timeline)

    # Add title and axis labels
    ax.set_title(variable)
    ax.set_xlabel(x_read)
    ax.set_ylabel(variable)

    # Plot separatrix
    if sep_pos:
        ax.plot_vline(sep_pos, "--")

    if animate:
        if controls:
            anim.controls(timeline_slider_args={"text": animate_over})

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(variable, animate_over)
            anim.save(save_as + ".gif", writer=PillowWriter(fps=fps))

    return line_block
