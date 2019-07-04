import warnings

from numpy import concatenate
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

    # TODO are we dealing with empty regions sensibly?

    j11, j12, j21, j22, ix1, ix2, nin, _, ny = _get_seps(da)
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
        corner1 = da[ix1 - 1, j11]
        corner2 = da[ix1, j11]
        corner3 = da[ix1, j11 + 1]
        corner4 = da[ix1 - 1, j11 + 1]

        xregion_lower = xr.concat([corner1, corner2, corner3, corner4],
                                  dim='dim1')

        corner5 = da[ix1 - 1, j22 + 1]
        corner6 = da[ix1, j22 + 1]
        corner7 = da[ix1, j22]
        corner8 = da[ix1 - 1, j22]

        xregion_upper = xr.concat([corner5, corner6, corner7, corner8],
                                  dim='dim1')

        region15 = xr.concat([xregion_lower, xregion_upper], dim='dim2')

        regions.extend([region13, region14, region15])

    # TODO what's the condition for there to be a second x-point?
    if True:
        # TODO fix this to properly cover the upper X-point
        # X-point regions
        corner1 = da[ix2 - 1, j12]
        corner2 = da[ix2, j12]
        corner3 = da[ix2, j12 + 1]
        corner4 = da[ix2 - 1, j12 + 1]

        xregion_lower = xr.concat([corner1, corner2, corner3, corner4],
                                  dim='dim1')

        corner5 = da[ix2 - 1, j21 + 1]
        corner6 = da[ix2, j21 + 1]
        corner7 = da[ix2, j21]
        corner8 = da[ix2 - 1, j21]

        xregion_upper = xr.concat([corner5, corner6, corner7, corner8],
                                  dim='dim1')

        region16 = xr.concat([xregion_lower, xregion_upper], dim='dim2')

        regions.append(region16)

    return regions


def plot_separatrices(da, ax):
    """Plot separatrices"""

    j11, j12, j21, j22, ix1, ix2, nin, nx, ny = _get_seps(da)

    R = da.coords['R'].transpose('x', 'theta')
    Z = da.coords['Z'].transpose('x', 'theta')

    if j22 + 1 < ny:
        # Lower X-point location
        Rx = 0.125 * (R[ix1 - 1, j11]     + R[ix1, j11]
                    + R[ix1, j11 + 1]     + R[ix1 - 1, j11 + 1]
                    + R[ix1 - 1, j22 + 1] + R[ix1, j22 + 1]
                    + R[ix1, j22]         + R[ix1 - 1, j22])
        Zx = 0.125 * (Z[ix1 - 1, j11]     + Z[ix1, j11]
                    + Z[ix1, j11 + 1]     + Z[ix1 - 1, j11 + 1]
                    + Z[ix1 - 1, j22 + 1] + Z[ix1, j22 + 1]
                    + Z[ix1, j22]         + Z[ix1 - 1, j22])
    else:
        Rx, Zx = None, None

    # Lower inner leg
    lower_inner_R = concatenate(
        (0.5 * (R[ix1 - 1, 0:(j11 + 1)] + R[ix1, 0:(j11 + 1)]), [Rx]))
    lower_inner_Z = concatenate(
        (0.5 * (Z[ix1 - 1, 0:(j11 + 1)] + Z[ix1, 0:(j11 + 1)]), [Zx]))

    # Lower outer leg
    lower_outer_R = concatenate(
        ([Rx], 0.5 * (R[ix1 - 1, (j22 + 1):] + R[ix1, (j22 + 1):])))
    lower_outer_Z = concatenate(
        ([Zx], 0.5 * (Z[ix1 - 1, (j22 + 1):] + Z[ix1, (j22 + 1):])))

    # Core
    core_R1 = 0.5 * (R[ix1 - 1, (j11 + 1):(j21 + 1)]
                   + R[ix1, (j11 + 1):(j21 + 1)])
    core_R2 = 0.5 * (R[ix1 - 1, (j12 + 1):(j22 + 1)]
                   + R[ix1, (j12 + 1):(j22 + 1)])
    core_R = concatenate(([Rx], core_R1, core_R2, [Rx]))

    core_Z1 = 0.5 * (Z[ix1 - 1, (j11 + 1):(j21 + 1)]
                   + Z[ix1, (j11 + 1):(j21 + 1)])
    core_Z2 = 0.5 * (Z[ix1 - 1, (j12 + 1):(j22 + 1)]
                   + Z[ix1, (j12 + 1):(j22 + 1)])
    core_Z = concatenate(([Zx], core_Z1, core_Z2, [Zx]))

    ax.plot(lower_inner_R, lower_inner_Z, 'k--')
    ax.plot(lower_outer_R, lower_outer_Z, 'k--')
    ax.plot(core_R, core_Z, 'k--')

    # TODO plot second separatrix
    if ix2 != ix1:
        if ix2 < ix1:
            raise ValueError("Inner separatrix must be the at the bottom")

        # TODO correct check that there is actually a second X-point?
        if j21 < nin:
            # Upper X-point location
            Rx = 0.125 * (R[ix2 - 1, j12] + R[ix2, j12]
                          + R[ix2, j12 + 1] + R[ix2 - 1, j12 + 1]
                          + R[ix2 - 1, j21 + 1] + R[ix2, j21 + 1]
                          + R[ix2, j21] + R[ix2 - 1, j21])
            Zx = 0.125 * (Z[ix2 - 1, j12] + Z[ix2, j12]
                          + Z[ix2, j12 + 1] + Z[ix2 - 1, j12 + 1]
                          + Z[ix2 - 1, j21 + 1] + Z[ix2, j21 + 1]
                          + Z[ix2, j21] + Z[ix2 - 1, j21])
        else:
            Rx, Zx = None, None

        lower_inner_R = 0.5 * (R[ix2 - 1, 0:(j11 + 1)] + R[ix2, 0:(j11 + 1)])
        lower_inner_Z = 0.5 * (Z[ix2 - 1, 0:(j11 + 1)] + Z[ix2, 0:(j11 + 1)])

        upper_outer_R = 0.5 * (R[ix2 - 1, nin:(j12+1)] + R[ix2, nin:(j12+1)])
        upper_outer_Z = 0.5 * (Z[ix2 - 1, nin:(j12+1)] + Z[ix2, nin:(j12+1)])

        lower_outer_R = 0.5 * (R[ix2 - 1, (j22 + 1):] + R[ix2, (j22 + 1):])
        lower_outer_Z = 0.5 * (Z[ix2 - 1, (j22 + 1):] + Z[ix2, (j22 + 1):])

        upper_inner_R = 0.5 * (R[ix2 - 1, (j21+1):nin] + R[ix2, (j21+1):nin])
        upper_inner_Z = 0.5 * (Z[ix2 - 1, (j21+1):nin] + Z[ix2, (j21+1):nin])

        # Core
        core_inner_R = 0.5 * (R[ix2 - 1, (j11 + 1):(j21 + 1)]
                         + R[ix2, (j11 + 1):(j21 + 1)])
        core_outer_R = 0.5 * (R[ix2 - 1, (j12 + 1):(j22 + 1)]
                         + R[ix2, (j12 + 1):(j22 + 1)])

        core_inner_Z = 0.5 * (Z[ix2 - 1, (j11 + 1):(j21 + 1)]
                         + Z[ix2, (j11 + 1):(j21 + 1)])
        core_outer_Z = 0.5 * (Z[ix2 - 1, (j12 + 1):(j22 + 1)]
                         + Z[ix2, (j12 + 1):(j22 + 1)])

        # TODO join these up (inc x-point) properly
        ax.plot(lower_inner_R, lower_inner_Z, 'r--')
        ax.plot(lower_outer_R, lower_outer_Z, 'r--')

        ax.plot(upper_inner_R, upper_inner_Z, 'r--')
        ax.plot(upper_outer_R, upper_outer_Z, 'r--')

        ax.plot(core_inner_R, core_inner_Z, 'r--')
        ax.plot(core_outer_R, core_outer_Z, 'r--')


def plot_targets(da, ax):
    """Plot divertor and limiter target plates"""

    j11, j12, j21, j22, ix1, ix2, nin, nx, ny = _get_seps(da)

    R = da.coords['R'].transpose('x', 'theta')
    Z = da.coords['Z'].transpose('x', 'theta')

    if j22 + 1 < ny:
        # lower PFR exists
        xin = 0
    else:
        xin = ix2

    inner_lower_target_R = R[xin:, 0]
    inner_lower_target_Z = Z[xin:, 0]
    ax.plot(inner_lower_target_R, inner_lower_target_Z, 'k-', linewidth=2)

    outer_lower_target_R = R[xin:, ny-1]
    outer_lower_target_Z = Z[xin:, ny-1]
    ax.plot(outer_lower_target_R, outer_lower_target_Z, 'k-', linewidth=2)

    if j21 < nin:
        # upper PFR exists
        xin = 0
    else:
        xin = ix2

    if j21 < nin:
        inner_upper_target_R = R[xin:, nin-1]
        inner_upper_target_Z = Z[xin:, nin-1]
        ax.plot(inner_upper_target_R, inner_upper_target_Z, 'k-', linewidth=2)

    if j12 > nin+1:
        outer_upper_target_R = R[xin:, nin]
        outer_upper_target_Z = Z[xin:, nin]
        ax.plot(outer_upper_target_R, outer_upper_target_Z, 'k-', linewidth=2)


def _get_seps(da):
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

    return j11, j12, j21, j22, ix1, ix2, nin, nx, ny