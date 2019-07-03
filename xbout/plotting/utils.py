import warnings

import xarray as xr


def plot_separatrix(da, sep_pos, ax, radial_coord='x'):
    """
    Plots the separatrix as a black dotted line.

    Should plot in the correct place regardless of the choice of coordinates
    for the plot axes, and the type of plot. sep_x needs to be supplied as the
    integer index of the grid point location of the separatrix.
    """

    # TODO Maybe use a decorator to do this for any type of plot?

    # 2D domain needs to intersect the separatrix plane to be able to plot it
    dims = da.dims
    if radial_coord not in dims:

        warnings.warn("Cannot plot separatrix as domain does not cross "
                      "separatrix, as it does not have a radial dimension",
                      Warning)
        return

    else:
        # Determine the separatrix position in terms of the radial coordinate
        # being used in the plot
        x_coord_vals = da.coords[radial_coord].values
        sep_x_pos = x_coord_vals[sep_pos]

        # Plot a vertical line at that location on the plot
        ax.axvline(x=sep_x_pos, linewidth=2, color="black", linestyle='--')

        return ax


def _decompose_regions(da):


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

    regions = []

    ystart = 0  # Y index to start the next section
    if j11 >= 0:
        # plot lower inner leg
        region1 = da[:, ystart:(j11 + 1)]

        yind = [j11, j22 + 1]
        region2 = da[:ix1, yind]

        region3 = da[ix1:, j11: (j11 + 2)]

        yind = [j22, j11 + 1]
        region4 = da[:ix1, yind]

        regions.extend([region1, region2, region3, region4])

        ystart = j11 + 1

    # Inner SOL
    region5 = da[:, ystart:(j21 + 1)]
    regions.append(region5)

    ystart = j21 + 1

    if j12 > j21:
        # Contains upper PF region

        # Inner leg
        region6 = da[ix1:, j21:(j21 + 2)]
        region7 = da[:, ystart:nin]

        # Outer leg
        region8 = da[:, nin:(j12 + 1)]
        region9 = da[ix1:, j12:(j12 + 2)]

        yind = [j21, j12 + 1]

        region10 = da[:ix1, yind]

        yind = [j21 + 1, j12]
        region11 = da[:ix1, yind]

        regions.extend([region6, region7, region8,
                        region9, region10, region11])

        ystart = j12 + 1
    else:
        ystart -= 1

    # Outer SOL
    region12 = da[:, ystart:(j22 + 1)]
    regions.append(region12)

    ystart = j22 + 1

    if j22 + 1 < ny:
        # Outer leg
        region13 = da[ix1:, j22:(j22 + 2)]
        region14 = da[:, ystart:ny]

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

        region15 = xr.concat([xregion_lower.drop('theta'),
                              xregion_upper.drop('theta')], dim='theta')

        regions.extend([region13, region14, region15])

    return regions