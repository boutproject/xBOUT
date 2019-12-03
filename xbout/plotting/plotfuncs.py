import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import xarray as xr

from .utils import _decompose_regions, plot_separatrices, plot_targets


def regions(da, ax=None, **kwargs):
    """
    Plots each logical plotting region as a different color for debugging.

    Uses matplotlib.pcolormesh
    """

    x = kwargs.pop('x', 'R')
    y = kwargs.pop('y', 'Z')

    if len(da.dims) != 2:
        raise ValueError("da must be 2D (x,y)")

    if ax is None:
        fig, ax = plt.subplots()

    regions = _decompose_regions(da)

    colored_regions = [xr.full_like(region, fill_value=num / len(regions))
                       for num, region in enumerate(regions)]

    first, *rest = colored_regions
    artists = [first.plot.pcolormesh(x=x, y=y, vmin=0, vmax=1, cmap='tab20',
                                     infer_intervals=False,
                                     add_colorbar=False, ax=ax, **kwargs)]
    if rest:
        for region in rest:
            artist = region.plot.pcolormesh(x=x, y=y, vmin=0, vmax=1,
                                            cmap='tab20',
                                            infer_intervals=False,
                                            add_colorbar=False,
                                            add_labels=False, ax=ax, **kwargs)
            artists.append(artist)
    return artists


def plot2d_wrapper(da, method, *, ax=None, separatrix=True, targets=True,
                   add_limiter_hatching=True, cmap=None, vmin=None, vmax=None,
                   aspect=None, **kwargs):
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
    cmap : Matplotlib colormap, optional
        Color map to use for the plot
    vmin : float, optional
        Minimum value for the color scale
    vmax : float, optional
        Maximum value for the color scale
    aspect : str or float, optional
        Passed to ax.set_aspect(). By default 'equal' is used.
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
    x = kwargs.pop('x', 'R')
    y = kwargs.pop('y', 'Z')

    if len(da.dims) != 2:
        raise ValueError("da must be 2D (x,y)")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if aspect is None:
        aspect = 'equal'
    ax.set_aspect(aspect)

    if vmin is None:
        vmin = da.min().values
    if vmax is None:
        vmax = da.max().values

    # Need to create a colorscale that covers the range of values in the whole array.
    # Using the add_colorbar argument would create a separate color scale for each
    # separate region, which would not make sense.
    if method is xr.plot.contourf:
        levels = kwargs.get('levels', 7)
        if isinstance(levels, np.int):
            levels = np.linspace(vmin, vmax, levels, endpoint=True)
            # put levels back into kwargs
            kwargs['levels'] = levels
        else:
            levels = np.array(list(levels))
            kwargs['levels'] = levels
            vmin = np.min(levels)
            vmax = np.max(levels)

        # create colorbar
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        # make colorbar have only discrete levels
        # average the levels so that colors in the colorbar represent the intervals
        # between the levels, as contourf colors filled regions between the given levels.
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'discrete cmap', sm.to_rgba(0.5*(levels[:-1] + levels[1:])),
                len(levels) - 1)
        # re-make sm with new cmap
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ticks=levels, ax=ax)
    elif method is xr.plot.contour:
        levels = kwargs.get('levels', 7)
        if isinstance(levels, np.int):
            vrange = vmax - vmin
            levels = np.linspace(vmin + vrange/(levels + 1), vmax - vrange/(levels + 1),
                                 levels, endpoint=True)
            # put levels back into kwargs
            kwargs['levels'] = levels
        else:
            levels = np.array(list(levels))
            kwargs['levels'] = levels
            vmin = np.min(levels)
            vmax = np.max(levels)

        # create colormap to be shared by all regions
        norm = matplotlib.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cmap = matplotlib.colors.ListedColormap(
                sm.to_rgba(levels), name='discrete cmap')
    else:
        # pass vmin and vmax through kwargs as they are not used for contourf or contour
        # plots
        kwargs['vmin'] = vmin
        kwargs['vmax'] = vmax

        # create colorbar
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cmap = sm.get_cmap()
        fig.colorbar(sm, ax=ax)

    if method is xr.plot.pcolormesh:
        if 'infer_intervals' not in kwargs:
            kwargs['infer_intervals'] = False

    regions = _decompose_regions(da)

    # Plot all regions on same axis
    add_labels = [True] + [False] * (len(regions) - 1)
    artists = [method(region, x=x, y=y, ax=ax, add_colorbar=False, add_labels=add_label,
        cmap=cmap, **kwargs) for region, add_label in zip(regions, add_labels)]

    if method is xr.plot.contour:
        # using extend='neither' guarantees that the ends of the colorbar will be
        # consistent, regardless of whether artists[0] happens to have any values below
        # vmin or above vmax. Unfortunately it does not seem to be possible to combine all
        # the QuadContourSet objects in artists to have this done properly. It would be
        # nicer to always draw triangular ends as if there
        # are always values below vmin and above vmax, but there does not seem to be an
        # option available to force this.
        extend = kwargs.get('extend', 'neither')
        fig.colorbar(artists[0], ax=ax, extend=extend)

    ax.set_title(da.name)

    if separatrix:
        plot_separatrices(da, ax)

    if targets:
        plot_targets(da, ax, hatching=add_limiter_hatching)

    return artists
