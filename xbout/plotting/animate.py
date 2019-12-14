import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import animatplot as amp

from .utils import _decompose_regions, _is_core_only, plot_separatrices, plot_targets
from matplotlib.animation import PillowWriter


def animate_poloidal(da, *, ax=None, cax=None, animate_over='t', separatrix=True,
                     targets=True, add_limiter_hatching=True, cmap=None, vmin=None,
                     vmax=None, animate=True, save_as=None, fps=10, controls=True,
                     **kwargs):
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
    separatrix : bool, optional
        Add dashed lines showing separatrices
    targets : bool, optional
        Draw solid lines at the target surfaces
    add_limiter_hatching : bool, optional
        Draw hatched areas at the targets
    save_as : True or str, optional
        If str is passed, save the animation as save_as+'.gif'.
        If True is passed, save the animation with a default name,
        '<variable name>_over_<animate_over>.gif'
    **kwargs : optional
        Additional arguments are passed on to method

    ###Returns
    ###-------
    ###artists
    ###    List of the contourf instances
    """

    # TODO generalise this
    x = kwargs.pop('x', 'R')
    y = kwargs.pop('y', 'Z')

    # Check plot is the right orientation
    spatial_dims = list(da.dims)

    try:
        spatial_dims.remove(animate_over)
    except ValueError:
        raise ValueError("Dimension animate_over={} is not present in the data"
                         .format(animate_over))

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

    # pass vmin and vmax through kwargs as they are not used for contour plots
    kwargs['vmin'] = vmin
    kwargs['vmax'] = vmax

    # create colorbar
    norm = (kwargs['norm'] if 'norm' in kwargs
            else matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cmap = sm.get_cmap()
    fig.colorbar(sm, ax=ax, cax=cax)

    ax.set_aspect('equal')

    regions = _decompose_regions(da)

    # Plot all regions on same axis
    blocks = []
    for region in regions:
        # Load values eagerly otherwise for some reason the plotting takes
        # 100's of times longer - for some reason animatplot does not deal
        # well with dask arrays!
        blocks.append(amp.blocks.Pcolormesh(region.coords[x].values,
                      region.coords[y].values, region.values, ax=ax, cmap=cmap,
                      **kwargs))

    ax.set_title(da.name)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if _is_core_only(da):
         separatrix = False
         targets = False

    if separatrix:
        plot_separatrices(da, ax)

    if targets:
        plot_targets(da, ax, hatching=add_limiter_hatching)

    if animate:
        timeline = amp.Timeline(np.arange(da.sizes[animate_over]), fps=fps)
        anim = amp.Animation(blocks, timeline)

        if controls:
            anim.controls(timeline_slider_args={'text': animate_over})

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(da.name, animate_over)
            anim.save(save_as + '.gif', writer=PillowWriter(fps=fps))

    return blocks


def animate_pcolormesh(data, animate_over='t', x=None, y=None, animate=True,
                       vmin=None, vmax=None, vsymmetric=False, fps=10, save_as=None,
                       ax=None, cax=None, controls=True, **kwargs):
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
    vmin : float, optional
        Minimum value to use for colorbar. Default is to use minimum value of
        data across whole timeseries.
    vmax : float, optional
        Maximum value to use for colorbar. Default is to use maximum value of
        data across whole timeseries.
    save_as : True or str, optional
        If str is passed, save the animation as save_as+'.gif'.
        If True is passed, save the animation with a default name,
        '<variable name>_over_<animate_over>.gif'
    fps : int, optional
        Frames per second of resulting gif
    kwargs : dict, optional
        Additional keyword arguments are passed on to the plotting function
        (e.g. imshow for 2D plots).
    """

    variable = data.name

    # Check plot is the right orientation
    spatial_dims = list(data.dims)

    if len(data.dims) != 3:
        raise ValueError('Data passed to animate_imshow must be 3-dimensional')

    try:
        spatial_dims.remove(animate_over)
    except ValueError:
        raise ValueError("Dimension animate_over={} is not present in the data"
                         .format(animate_over))

    if x is None and y is None:
        x, y = spatial_dims
    elif x is None:
        try:
            spatial_dims.remove(y)
        except ValueError:
            raise ValueError("Dimension {} is not present in the data" .format(y))
        x = spatial_dims[0]
    elif y is None:
        try:
            spatial_dims.remove(x)
        except ValueError:
            raise ValueError("Dimension {} is not present in the data" .format(x))
        y = spatial_dims[0]

    data = data.transpose(animate_over, y, x)

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

    if not ax:
        fig, ax = plt.subplots()

    # Note: animatplot's Pcolormesh gave strange outputs without passing
    # explicitly x- and y-value arrays, although in principle these should not
    # be necessary.
    ny, nx = image_data.shape[1:]
    pcolormesh_block = amp.blocks.Pcolormesh(np.arange(float(nx)), np.arange(float(ny)),
                                             image_data, vmin=vmin, vmax=vmax, ax=ax,
                                             **kwargs)

    if animate:
        timeline = amp.Timeline(np.arange(data.sizes[animate_over]), fps=fps)
        anim = amp.Animation([pcolormesh_block], timeline)

    cbar = plt.colorbar(pcolormesh_block.quad, ax=ax, cax=cax)
    cbar.ax.set_ylabel(variable)

    # Add title and axis labels
    ax.set_title(variable)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if animate:
        if controls:
            anim.controls(timeline_slider_args={'text': animate_over})

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(variable, animate_over)
            anim.save(save_as + '.gif', writer=PillowWriter(fps=fps))

    return pcolormesh_block


def animate_line(data, animate_over='t', animate=True,
                 vmin=None, vmax=None, fps=10, save_as=None, sep_pos=None, ax=None,
                 controls=True, **kwargs):
    """
    Plots a line plot which is animated with time.

    Currently only supports 1D+1 data, which it plots with animatplot's Line animation.

    Parameters
    ----------
    data : xarray.DataArray
    animate_over : str, optional
        Dimension over which to animate
    vmin : float, optional
        Minimum value to use for colorbar. Default is to use minimum value of
        data across whole timeseries.
    vmax : float, optional
        Maximum value to use for colorbar. Default is to use maximum value of
        data across whole timeseries.
    sep_pos : int, optional
        Radial position at which to plot the separatrix
    save_as : True or str, optional
        If str is passed, save the animation as save_as+'.gif'.
        If True is passed, save the animation with a default name,
        '<variable name>_over_<animate_over>.gif'
    fps : int, optional
        Frames per second of resulting gif
    kwargs : dict, optional
        Additional keyword arguments are passed on to the plotting function
        (e.g. imshow for 2D plots).
    """

    variable = data.name

    # Check plot is the right orientation
    t_read, x_read = data.dims
    if (t_read is animate_over):
        pass
    else:
        data = data.transpose(animate_over, t_read)

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
        ax.plot_vline(sep_pos, '--')

    if animate:
        if controls:
            anim.controls(timeline_slider_args={'text': animate_over})

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(variable, animate_over)
            anim.save(save_as + '.gif', writer=PillowWriter(fps=fps))

    return line_block
