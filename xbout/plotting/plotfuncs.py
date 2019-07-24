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



def contourf(da, levels=7, ax=None, separatrix=True, targets=True,
             add_limiter_hatching=True, cmap=None, **kwargs):
    """
    Plots a 2D filled contour plot, taking into account branch cuts (X-points).

    Wraps `xarray.plot.contourf`, so automatically adds labels.

    Parameters
    ----------
    da : xarray.DataArray
        A 2D (x,y) DataArray of data to plot
    ax : Axes, optional
        A matplotlib axes instance to plot to. If None, create a new
        figure and axes, and plot to that
    **kwargs : optional
        Additional arguments are passed on to xarray.plot.contourf

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

    if isinstance(levels, np.int):
        vmin = da.min().values
        vmax = da.max().values
        levels = np.linspace(vmin, vmax, levels, endpoint=True)
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
    fig.colorbar(sm, ticks=levels)

    regions = _decompose_regions(da)
    region_kwargs = {}

    # Plot all regions on same axis
    first, *rest = regions
    artists = [first.plot.contourf(x=x, y=y, ax=ax, levels=levels, add_colorbar=False,
                                   cmap=cmap, **kwargs, **region_kwargs)]
    if rest:
        for region in rest:
            artist = region.plot.contourf(x=x, y=y, ax=ax, levels=levels,
                                          add_colorbar=False, add_labels=False,
                                          cmap=cmap, **kwargs, **region_kwargs)
            artists.append(artist)

    if separatrix:
        plot_separatrices(da, ax)

    if targets:
        plot_targets(da, ax, hatching=add_limiter_hatching)

    return artists
