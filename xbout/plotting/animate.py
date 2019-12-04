import numpy as np
import matplotlib.pyplot as plt

import animatplot as amp

from .utils import plot_separatrix
from matplotlib.animation import PillowWriter

def animate_imshow(data, animate_over='t', x=None, y=None, animate=True,
                   vmin=None, vmax=None, fps=10, save_as=None,
                   sep_pos=None, ax=None, **kwargs):
    """
    Plots a color plot which is animated with time over the specified
    coordinate.

    Currently only supports 2D+1 data, which it plots with xarray's
    wrapping of matplotlib's imshow.

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
    sep_pos : int, optional
        Radial position at which to plot the separatrix
    save_as: str, optional
        Filename to give to the resulting gif
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

    try:
        if x is None and y is None:
            x, y = spatial_dims
        elif x is None:
            spatial_dims.remove(y)
            x = spatial_dims[0]
        elif y is None:
            spatial_dims.remove(x)
            y = spatial_dims[0]

        # Use (y, x) here so we transpose by default for imshow
        data = data.transpose(animate_over, y, x)
    except ValueError:
        raise ValueError("Dimensions {} or {} are not present in the data"
                         .format(x, y))

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

    imshow_block = amp.blocks.Imshow(image_data, vmin=vmin, vmax=vmax,
                                     ax=ax, origin='lower', **kwargs)

    timeline = amp.Timeline(np.arange(data.sizes[animate_over]), fps=fps)

    if animate:
        anim = amp.Animation([imshow_block], timeline)

    cbar = plt.colorbar(imshow_block.im, ax=ax)
    cbar.ax.set_ylabel(variable)

    # Add title and axis labels
    ax.set_title("{} variation over {}".format(variable, animate_over))
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    # Plot separatrix
    if sep_pos:
        ax = plot_separatrix(data, sep_pos, ax)

    if animate:
        anim.controls(timeline_slider_args={'text': animate_over})

        if not save_as:
            save_as = "{}_over_{}".format(variable, animate_over)
        anim.save(save_as + '.gif', writer=PillowWriter(fps=fps))

    return imshow_block


def animate_line(data, animate_over='t', animate=True,
                 vmin=None, vmax=None, fps=10, save_as=None, sep_pos=None, ax=None,
                 **kwargs):
    """
    Plots a line plot which is animated with time.

    Currently only supports 1D+1 data, which it plots with xarray's
    wrapping of matplotlib's plot.

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
    save_as: str, optional
        Filename to give to the resulting gif
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
        data = data.transpose(x_read, animate_over)

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

    timeline = amp.Timeline(np.arange(data.sizes[animate_over]), fps=fps)

    if animate:
        anim = amp.Animation([line_block], timeline)

    # Add title and axis labels
    ax.set_title("{} variation over {}".format(variable, animate_over))
    ax.set_xlabel(x_read)
    ax.set_ylabel(variable)

    # Plot separatrix
    if sep_pos:
        ax.plot_vline(sep_pos, '--')

    if animate:
        anim.controls(timeline_slider_args={'text': animate_over})

        if not save_as:
            save_as = "{}_over_{}".format(variable, animate_over)
        anim.save(save_as + '.gif', writer=PillowWriter(fps=fps))

    return line_block
