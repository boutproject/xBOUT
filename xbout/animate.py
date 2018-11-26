#!/usr/bin/env python

import warnings

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np


def draw_first_frame(data, x, y, anim_over_coord, sep_pos, aspect, ax, **kwargs):
    """
    Draws the first frame of the evolving color plot.

    Done because colorbar, axes, and separatrix only need to be plotted once.
    """

    print('Initialising background...')

    # Get data from first index of dimension to be animated over
    anim_over_dim, values = get_dim_of_coord(anim_over_coord, data)
    # frame_data = data[dict(anim_over_dim: 0)].transpose()
    frame_data = data.isel(**{anim_over_dim: 0}).transpose()

    # Set range of colorbar to max and min over all data, not just first frame
    extra_args = kwargs
    if 'robust' not in extra_args:
        if 'vmin' not in kwargs:
            vmin = data.values.min()  # Lowest value
            extra_args = dict(vmin=vmin, **extra_args)
        if 'vmax' not in kwargs:
            vmax = data.values.max()  # Highest value
            extra_args = dict(vmax=vmax, **extra_args)

    im = frame_data.plot.imshow(ax=ax, x=x, y=y, add_colorbar=True, animated=True, **extra_args)

    # Set aspect ratio of plot
    if aspect in ['auto', 'equal']:
        ax.set_aspect(aspect)
    else:
        raise ValueError('Unrecognised method for setting aspect ratio')

    if sep_pos:
        plot_separatrix(frame_data, sep_pos, ax)

    return im


def get_dim_of_coord(coord, da):
    """
    Returns the dimension corresponding to a given coordinate in the dataset, and also the exact values of that
    coordinate.
    """

    dims = da.coords[coord].dims
    dim, = dims

    if isinstance(dims, tuple) and len(dims) > 1:
         raise ValueError('Coordinate ' + coord + ' has multiple dimensions: ' + str(dims))
    values = da.coords[coord].values

    return dim, values


def plot_separatrix(da, sep_pos, ax):
    """
    Plots the separatrix as a black dotted line.

    Should plot in the correct place regardless of the choice of coordinates for the plot axes, and the type of plot.
    sep_x needs to be supplied as the integer index of the grid point location of the separatrix.
    """

    # TODO Maybe use a decorator to do this for any type of plot?

    # 2D domain needs to intersect the separatrix plane to be able to plot it
    dims = da.dims
    if 'x' not in dims:

        warnings.warn("Cannot plot separatrix as domain does not cross "
                      "separatrix, as it does not have a radial dimension",
                      Warning)
        return

    else:
        # Determine the separatrix position in terms of the radial coordinate being used in the plot
        x_coord_vals = da.coords['x'].values
        sep_x_pos = x_coord_vals[sep_pos]

        # Plot a vertical line at that location on the plot
        ax.axvline(x=sep_x_pos, linewidth=2, color="black", linestyle='--')

        return ax


def update_im(frame, im, anim_over_coord, dat, ax):
    """
    Updates the color plot by updating the values in the imshow array, and the title.
    """

    print('Drawing frame ' + str(frame+1) + ' ...', end='\r')

    # TODO find out how to pass the frame selection without unpacking a dictionary
    anim_over_dim, coord_values = get_dim_of_coord(anim_over_coord, dat)
    frame_data = dat.isel(**{anim_over_dim: frame}).transpose()
    im.set_array(frame_data.values)

    # Plot title with as much info as possible, including units (assumes coord passed has form like 'time_s')
    # TODO add support for units etc. found in attrs
    coord_name = anim_over_coord
    title = dat.name + ' at ' + coord_name + ' = ' + str(coord_values[frame])
    ax.set_title(title)

    return im,


def _promote_to_coords(ds, dims):
    for dim in dims:
        if dim not in ds.coords:
            values = np.arange(ds.dims[dim])
            new_coord = {dim: (dim, values)}
            ds = ds.assign_coords(**new_coord)
    return ds


def _animated_plot(ds, var, animate_over, x=None, y=None, sep_pos=None, aspect='auto',
                   fps=10, save_as=None, writer='imagemagick', **kwargs):
    """
    Plots a color plot which is animated with time over the specified coordinate.

    Currently only supports 2D+1 data, which it plots with xarray's warpping of matplotlib's imshow.

    Parameters
    ----------
    data : xarray.DataArray
    anim_over_coord : str
    x : str, optional
    y : str, optional
    sep_x : int, optional
        Radial position of the separatrix as an integer
    save_as: str, optional
        Filename to give to the resulting gif
    autoplay
    fps : int, optional
    aspect : {'auto', 'equal'} or int, optional
        Method for choosing aspect ratio of 2D plot. 'auto' uses the number of grid points, 'equal' uses
    kwargs : dict, optional
        Additional keyword arguments are passed on to the plotting function (e.g. imshow for 2D plots).

    """

    # TODO add option to pass a user-defined function in which just updates details about the plot
    # TODO add ability to autoplay the gif once it's been created

    ds = _promote_to_coords(ds, [animate_over, x, y])

    data = ds[var]

    variable = data.name
    n_dims = len(data.dims)
    if n_dims == 3:
        print("{} data passed has {} dimensions - will use "
              "xarray.plot.imshow()".format(variable, str(n_dims)))
    else:
        raise ValueError("Data passed has an unsupported number of dimensions "
                         "({})".format(str(n_dims)))

    num_frames = data[animate_over].size
    print("{} frames to plot".format(str(num_frames)))

    fig, ax = plt.subplots()

    im = draw_first_frame(data, x, y, animate_over, sep_pos, aspect, ax,
                          **kwargs)

    anim = FuncAnimation(fig, update_im, num_frames,
                         fargs=(im, animate_over, data, ax),
                         blit=False)

    if save_as is None:
        save_as = variable + '_over_' + animate_over + '.gif'
    anim.save(save_as, writer=writer, fps=fps)

    plt.close(fig)
