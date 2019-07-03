import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def contourf(da, nlevel=10, ax=None, **kwargs):
    """
    Plots a 2D contour plot, taking into account branch cuts (X-points).

    Parameters
    ----------
    da : xarray.DataArray
        A 2D (x,y) DataArray of data to plot
    nlevel : int, optional
        Number of levels in the contour plot
    ax : Axes, optional
        A matplotlib axes instance to plot to. If None, create a new
        figure and axes, and plot to that

    Returns
    -------
    con
        The contourf instance

    Examples
    --------
    To put a plot into an axis with a color bar:
    >>> fig, axis = plt.subplots()
    >>> c = gridcontourf(grid, data, show=False, ax=axis)
    >>> fig.colorbar(c, ax=axis)
    >>> plt.show()
    """

    print(da)

    # TODO generalise this
    x='R'
    y='Z'

    if len(da.dims) != 2:
        raise ValueError("da must be 2D (x,y)")

    grid = da.attrs['grid']
    j11 = grid['jyseps1_1']
    j12 = grid['jyseps1_2']
    j21 = grid['jyseps2_1']
    j22 = grid['jyseps2_2']
    ix1 = grid['ixseps1']
    ix2 = grid['ixseps2']
    nin = grid.get('ny_inner', j12)

    nx = grid['nx']
    ny = grid['ny']

    R = da.coords['R']
    Z = da.coords['R']

    add_colorbar = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        add_colorbar = True

    mind = da.min()
    maxd = da.max()

    levels = np.linspace(mind, maxd, nlevel, endpoint=True)

    ystart = 0  # Y index to start the next section
    if j11 >= 0:
        # plot lower inner leg
        region1 = da[:, ystart:(j11 + 1)]
        region1.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)

        yind = [j11, j22 + 1]
        region2 = da[:ix1, yind]
        region2.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)

        region3 = da[ix1:, j11: (j11 + 2)]
        region3.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)

        yind = [j22, j11 + 1]
        region4 = da[:ix1, yind]
        region4.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)

        ystart = j11 + 1

    # Inner SOL
    region5 = da[:, ystart:(j21 + 1)]
    region5.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                          add_colorbar=False)

    ystart = j21 + 1

    if j12 > j21:
        # Contains upper PF region

        # Inner leg
        region6 = da[ix1:, j21:(j21 + 2)]
        region6.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)
        region7 = da[:, ystart:nin]
        region7.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)

        # Outer leg
        region8 = da[:, nin:(j12 + 1)]
        region8.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)

        region9 = da[:, nin:(j12 + 1)]
        region9.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)

        region10 = da[ix1:, j12:(j12 + 2)]
        region10.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)

        yind = [j21, j12 + 1]

        region11 = da[:ix1, yind]
        region11.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                               add_colorbar=False)

        yind = [j21 + 1, j12]

        region12 = da[:ix1, yind]
        region12.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                               add_colorbar=False)

        ystart = j12 + 1
    else:
        ystart -= 1

    # Outer SOL
    region12 = da[:, ystart:(j22 + 1)]
    region12.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                           add_colorbar=False)


    ystart = j22 + 1

    if j22 + 1 < ny:
        # Outer leg

        region13 = da[ix1:, j22:(j22 + 2)]
        region13.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                               add_colorbar=False)

        region14 = da[:, ystart:ny]
        region14.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                               add_colorbar=False)

        # X-point regions
        xregion1 = da[ix1 - 1, j11]
        xregion2 = da[ix1, j11]
        xregion3 = da[ix1, j11 + 1]
        xregion4 = da[ix1 - 1, j11 + 1]

        xregion_lower = xr.concat([xregion1, xregion2, xregion3, xregion4],
                                  dim='x')

        xregion5 = da[ix1 - 1, j22 + 1]
        xregion6 = da[ix1, j22 + 1]
        xregion7 = da[ix1, j22]
        xregion8 = da[ix1 - 1, j22]

        xregion_upper = xr.concat([xregion5, xregion6, xregion7, xregion8],
                                  dim='x')

        xregion = xr.concat([xregion_lower.drop('theta'),
                             xregion_upper.drop('theta')], dim='theta')

        xregion.plot.contourf(x=x, y=y, levels=levels, ax=ax,
                              add_colorbar=False)

    return None