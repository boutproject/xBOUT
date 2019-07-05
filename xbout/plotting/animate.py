import numpy as np
import matplotlib.pyplot as plt

import animatplot as amp


def animate_imshow(data, animate_over='t', x='x', y='y', animate=True,
                   vmin='min', vmax='max', fps=10, save_as=None,
                   ax=None, **kwargs):
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
    vmin : float, optional
        Minimum value to use for colorbar. Default is to use minimum value of
        data across whole timeseries.
    vmax : float, optional
        Maximum value to use for colorbar. Default is to use maximum value of
        data across whole timeseries.
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

    # If not specified, determine max and min values across entire data series
    if vmax is 'max':
        vmax = np.max(image_data)
    if vmin is 'min':
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

    # TODO Plot separatrix

    if animate:
        anim.controls(timeline_slider_args={'text': animate_over})

        if not save_as:
            save_as = "{}_over_{}".format(variable, animate_over)
        # TODO save using PillowWriter instead once matplotlib 3.1 comes out
        # see https://github.com/t-makaro/animatplot/issues/24
        anim.save(save_as + '.gif', writer='imagemagick')

    return imshow_block


def animate_line(data, animate_over='t', x='x', y='y', animate=True,
                 fps=10, save_as=None, sep_pos=None, ax=None, **kwargs):
    variable = data.name

    t_read, x_read = data.dims
    raise NotImplementedError
