import numpy as np
import matplotlib.pyplot as plt

import animatplot as amp

from .utils import plot_separatrix


def animate_imshow(data, animate_over='t', x='x', y='y', animate=True,
                   fps=10, save_as=None, sep_pos=None, ax=None, **kwargs):
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
        Dimension to use on the x axis, default is 'x'
    y : str, optional
        Dimension to use on the y axis, default is 'y'
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
    t_read, y_read, x_read = data.dims
    if (x_read is x) & (y_read is y):
        pass
    elif (x_read is y) & (y_read is x):
        data = data.transpose(animate_over, y, x)
    else:
        raise ValueError("Dimensions {} or {} are not present in the data"
                         .format(x, y))

    # Load values eagerly otherwise for some reason the plotting takes
    # 100's of times longer - for some reason animatplot does not deal
    # well with dask arrays!
    image_data = data.values

    # Determine max and min values across entire data series
    max = np.max(image_data)
    min = np.min(image_data)

    if not ax:
        fig, ax = plt.subplots()

    imshow_block = amp.blocks.Imshow(image_data, vmin=min, vmax=max,
                                     axis=ax, origin='lower', **kwargs)

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
        anim.controls(timeline_slider_args={'time_label': animate_over})

        if not save_as:
            save_as = "{}_over_{}".format(variable, animate_over)
        anim.save_gif(save_as)

    return imshow_block
