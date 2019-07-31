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
             add_limiter_hatching=True, cmap=None, vmin=None, vmax=None, **kwargs):
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
    levels : int or iterable, optional
        Only used by contour or contourf, sets the number of levels (if int) or the level
        values (if iterable)
    **kwargs : optional
        Additional arguments are passed on to method

    Returns
    -------
    artists
        List of the contourf instances
    """

    # TODO generalise this
    x = kwargs.pop('x', 'R')
    y = kwargs.pop('y', 'Z')

    if len(da.dims) != 2:
        raise ValueError("da must be 2D (x,y)")

    # TODO work out how to auto-set the aspect ration of the plot correctly
    height = da.coords[y].max() - da.coords[y].min()
    width = da.coords[x].max() - da.coords[x].min()
    aspect = height / width

    if ax is None:
        fig, ax = plt.subplots()

    if vmin is None:
        vmin = da.min().values
    if vmax is None:
        vmax = da.max().values

    if method is xr.plot.contour or method is xr.plot.contourf:
        levels = kwargs.get('levels', 7)
        if isinstance(levels, np.int):
            levels = np.linspace(vmin, vmax, levels, endpoint=True)
            # put levels back into kwargs
            kwargs['levels'] = levels
        else:
            levels = np.array(levels)
            vmin = np.min(levels)
            vmax = np.max(levels)

        # create colorbar
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        # make colorbar have only discrete levels
        # average the levels so that colors represent the intervals between the levels
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'discrete cmap', sm.to_rgba(0.5*(levels[:-1] + levels[1:])), len(levels) - 1)
        # re-make sm with new cmap
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ticks=levels)
    else:
        # pass vmin and vmax through kwargs as they are not used for contour plots
        kwargs['vmin'] = vmin
        kwargs['vmax'] = vmax

        # create colorbar
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cmap = sm.get_cmap()
        fig.colorbar(sm)

    if method is xr.plot.pcolormesh:
        if 'infer_intervals' not in kwargs:
            kwargs['infer_intervals'] = False

    regions = _decompose_regions(da)
    region_kwargs = {}

    # Plot all regions on same axis
    first, *rest = regions
    artists = [method(first, x=x, y=y, ax=ax, add_colorbar=False, cmap=cmap, **kwargs,
                      **region_kwargs)]
    if rest:
        for region in rest:
            artist = method(region, x=x, y=y, ax=ax, add_colorbar=False, add_labels=False,
                            cmap=cmap, **kwargs, **region_kwargs)
            artists.append(artist)

    ax.set_title(da.name)

    if separatrix:
        plot_separatrices(da, ax)

    if targets:
        plot_targets(da, ax, hatching=add_limiter_hatching)

    return artists
