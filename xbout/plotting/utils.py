import warnings

import numpy as np
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

    # TODO is this correct check that there is actually a second X-point?
    if j21 < nin:
        # X-point regions
        corner1 = da[ix1-1, j12]
        corner2 = da[ix1, j12]
        corner3 = da[ix1, j12+1]
        corner4 = da[ix1-1, j12+1]

        xregion_lower = xr.concat([corner1, corner2, corner3, corner4],
                                  dim='dim1')

        corner5 = da[ix1 - 1, j21 + 1]
        corner6 = da[ix1, j21+1]
        corner7 = da[ix1, j21]
        corner8 = da[ix1 - 1, j21]

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
    lower_inner_R = np.concatenate(
        (0.5 * (R[ix1 - 1, 0:(j11 + 1)] + R[ix1, 0:(j11 + 1)]), [Rx]))
    lower_inner_Z = np.concatenate(
        (0.5 * (Z[ix1 - 1, 0:(j11 + 1)] + Z[ix1, 0:(j11 + 1)]), [Zx]))

    # Lower outer leg
    lower_outer_R = np.concatenate(
        ([Rx], 0.5 * (R[ix1 - 1, (j22 + 1):] + R[ix1, (j22 + 1):])))
    lower_outer_Z = np.concatenate(
        ([Zx], 0.5 * (Z[ix1 - 1, (j22 + 1):] + Z[ix1, (j22 + 1):])))

    # Core
    core_R1 = 0.5 * (R[ix1 - 1, (j11 + 1):(j21 + 1)]
                   + R[ix1, (j11 + 1):(j21 + 1)])
    core_R2 = 0.5 * (R[ix1 - 1, (j12 + 1):(j22 + 1)]
                   + R[ix1, (j12 + 1):(j22 + 1)])
    core_R = np.concatenate(([Rx], core_R1, core_R2, [Rx]))

    core_Z1 = 0.5 * (Z[ix1 - 1, (j11 + 1):(j21 + 1)]
                   + Z[ix1, (j11 + 1):(j21 + 1)])
    core_Z2 = 0.5 * (Z[ix1 - 1, (j12 + 1):(j22 + 1)]
                   + Z[ix1, (j12 + 1):(j22 + 1)])
    core_Z = np.concatenate(([Zx], core_Z1, core_Z2, [Zx]))

    ax.plot(lower_inner_R, lower_inner_Z, 'k--')
    ax.plot(lower_outer_R, lower_outer_Z, 'k--')
    ax.plot(core_R, core_Z, 'k--')

    # Plot second separatrix
    if ix2 != ix1:
        if ix2 < ix1:
            raise ValueError("Inner separatrix must be the at the bottom")

        # TODO is this correct check that there is actually a second X-point?
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

        inner_R = np.concatenate((lower_inner_R, core_inner_R, [Rx],
                                  np.flip(upper_outer_R)))
        inner_Z = np.concatenate((lower_inner_Z, core_inner_Z, [Zx],
                                  np.flip(upper_outer_Z)))
        ax.plot(inner_R, inner_Z, 'k--')

        outer_R = np.concatenate((np.flip(lower_outer_R),
                                  np.flip(core_outer_R), [Rx], upper_inner_R))
        outer_Z = np.concatenate((np.flip(lower_outer_Z),
                                  np.flip(core_outer_Z), [Zx], upper_inner_Z))
        ax.plot(outer_R, outer_Z, 'k--')


def plot_targets(da, ax, hatching=True):
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
    [line1] = ax.plot(inner_lower_target_R, inner_lower_target_Z, 'k-',
                      linewidth=2)
    if hatching:
        _add_hatching(line1, ax)

    outer_lower_target_R = R[xin:, ny-1]
    outer_lower_target_Z = Z[xin:, ny-1]
    [line2] = ax.plot(outer_lower_target_R, outer_lower_target_Z, 'k-',
                      linewidth=2)
    if hatching:
        _add_hatching(line2, ax, reversed=True)

    if j21 < nin:
        # upper PFR exists
        xin = 0
    else:
        xin = ix2

    if j21 < nin:
        inner_upper_target_R = R[xin:, nin-1]
        inner_upper_target_Z = Z[xin:, nin-1]
        [line3] = ax.plot(inner_upper_target_R, inner_upper_target_Z, 'k-',
                          linewidth=2)
        if hatching:
            _add_hatching(line3, ax, reversed=True)

    if j12 > nin+1:
        outer_upper_target_R = R[xin:, nin]
        outer_upper_target_Z = Z[xin:, nin]
        [line4] = ax.plot(outer_upper_target_R, outer_upper_target_Z, 'k-',
                          linewidth=2)
        if hatching:
            _add_hatching(line4, ax)


def _add_hatching(line, ax, reversed=False):
    """
    Adds a series of angled ticks to target plate lines to give a hatching
    effect, indicative of a solid surface
    """

    x = line.get_xdata()
    y = line.get_ydata()

    if reversed:
        x = np.flip(x)
        y = np.flip(y)

    # TODO redo this to evenly space ticks by physical distance along line
    num_hatchings = 5
    step = len(x) // num_hatchings
    hatch_inds = np.arange(0, len(x), step)

    vx, vy = x.max() - x.min(), y.max() - y.min()
    limiter_line_length = np.linalg.norm((vx, vy))
    hatch_line_length = (limiter_line_length / num_hatchings) / 1.5

    # For each hatching
    for ind in hatch_inds:
        # Compute local perpendicular vector
        dx, dy = _get_perp_vec((x[ind], y[ind]), (x[ind+1], y[ind+1]),
                               magnitude=hatch_line_length)

        # Rotate by 60 degrees
        t = -np.pi/3
        new_dx = dx * np.cos(t) - dy * np.sin(t)
        new_dy = dx * np.sin(t) + dy * np.cos(t)

        # Draw
        ax.plot([x[ind], x[ind]+new_dx], [y[ind], y[ind]+new_dy], 'k-')


def _get_perp_vec(u1, u2, magnitude=0.04):
    """Return the vector perpendicular to the vector u2-u1."""

    x1, y1 = u1
    x2, y2 = u2
    vx, vy = x2-x1, y2-y1
    v = np.linalg.norm((vx, vy))
    wx, wy = -vy/v * magnitude, vx/v * magnitude
    return wx, wy


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