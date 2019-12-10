import warnings

import matplotlib as mpl
import numpy as np
import xarray as xr


def _create_norm(logscale, norm, vmin, vmax):
    if logscale:
        if norm is not None:
            raise ValueError("norm and logscale cannot both be passed at the same "
                             "time.")
        if vmin*vmax > 0:
            # vmin and vmax have the same sign, so can use standard log-scale
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            # vmin and vmax have opposite signs, so use symmetrical logarithmic scale
            if not isinstance(logscale, bool):
                linear_scale = logscale
            else:
                linear_scale = 1.e-5
            linear_threshold = min(abs(vmin), abs(vmax)) * linear_scale
            norm = mpl.colors.SymLogNorm(linear_threshold, vmin=vmin, vmax=vmax)
    elif norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return norm


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

    j11, j12, j21, j22, ix1, ix2, nin, _, ny, y_boundary_guards = _get_seps(da)
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

    if j21 + 1 > ystart:
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

    if j22 + 1 > ystart:
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

    if j21 > j11 and j12 > j21 and j22 > j12:
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


def _is_core_only(da):

    _, _, _, _, ix1, ix2, _, nx, _, _ = _get_seps(da)

    return (ix1 >= nx and ix2 >= nx)


def plot_separatrices(da, ax):
    """Plot separatrices"""

    j11, j12, j21, j22, ix1, ix2, nin, nx, ny, y_boundary_guards = _get_seps(da)

    R = da.coords['R'].transpose(da.metadata['bout_xdim'],
                                 da.metadata['bout_ydim']).values
    Z = da.coords['Z'].transpose(da.metadata['bout_xdim'],
                                 da.metadata['bout_ydim']).values

    if j22 + 1 < ny:
        # Lower X-point location
        Rx = 0.125 * (R[ix1 - 1, j11] + R[ix1, j11]
                      + R[ix1, j11 + 1] + R[ix1 - 1, j11 + 1]
                      + R[ix1 - 1, j22 + 1] + R[ix1, j22 + 1]
                      + R[ix1, j22] + R[ix1 - 1, j22])
        Zx = 0.125 * (Z[ix1 - 1, j11] + Z[ix1, j11]
                      + Z[ix1, j11 + 1] + Z[ix1 - 1, j11 + 1]
                      + Z[ix1 - 1, j22 + 1] + Z[ix1, j22 + 1]
                      + Z[ix1, j22] + Z[ix1 - 1, j22])
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
    if j12 > j21:
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

    if ix2 != ix1:
        if ix2 < ix1:
            raise ValueError("Inner separatrix must be the at the bottom")

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
    elif j12 > j21:
        # Connected double-null - plot separatrices in upper legs
        upper_outer_R = np.concatenate(
                (0.5 * (R[ix2 - 1, nin:(j12+1)] + R[ix2, nin:(j12+1)]), [Rx]))
        upper_outer_Z = np.concatenate(
                (0.5 * (Z[ix2 - 1, nin:(j12+1)] + Z[ix2, nin:(j12+1)]), [Zx]))

        upper_inner_R = np.concatenate(
                ([Rx], 0.5 * (R[ix2 - 1, (j21+1):nin] + R[ix2, (j21+1):nin])))
        upper_inner_Z = np.concatenate(
                ([Zx], 0.5 * (Z[ix2 - 1, (j21+1):nin] + Z[ix2, (j21+1):nin])))

        ax.plot(upper_inner_R, upper_inner_Z, 'k--')
        ax.plot(upper_outer_R, upper_outer_Z, 'k--')


def plot_targets(da, ax, hatching=True):
    """Plot divertor and limiter target plates"""

    j11, j12, j21, j22, ix1, ix2, nin, nx, ny, y_boundary_guards = _get_seps(da)

    R = da.coords['R'].transpose(da.metadata['bout_xdim'],
                                 da.metadata['bout_ydim']).values
    Z = da.coords['Z'].transpose(da.metadata['bout_xdim'],
                                 da.metadata['bout_ydim']).values

    if j22 + 1 < ny:
        # lower PFR exists
        xin = 0
    else:
        xin = ix2

    inner_lower_target_R = R[xin:, y_boundary_guards]
    inner_lower_target_Z = Z[xin:, y_boundary_guards]
    [line1] = ax.plot(inner_lower_target_R, inner_lower_target_Z, 'k-',
                      linewidth=2)
    if hatching:
        _add_hatching(line1, ax)

    outer_lower_target_R = R[xin:, ny - 1 - y_boundary_guards]
    outer_lower_target_Z = Z[xin:, ny - 1 - y_boundary_guards]
    [line2] = ax.plot(outer_lower_target_R, outer_lower_target_Z, 'k-',
                      linewidth=2)
    if hatching:
        _add_hatching(line2, ax, reversed=True)

    if j21 < nin:
        # upper PFR exists
        xin = 0
    else:
        xin = ix2

    if j12 > j21:
        inner_upper_target_R = R[xin:, nin - 1 - y_boundary_guards]
        inner_upper_target_Z = Z[xin:, nin - 1 - y_boundary_guards]
        [line3] = ax.plot(inner_upper_target_R, inner_upper_target_Z, 'k-',
                          linewidth=2)
        if hatching:
            _add_hatching(line3, ax, reversed=True)

        outer_upper_target_R = R[xin:, nin + y_boundary_guards]
        outer_upper_target_Z = Z[xin:, nin + y_boundary_guards]
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
    for ind in hatch_inds[:-1]:
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

    nx = da.metadata['nx']
    ix1 = da.metadata['ixseps1']
    ix2 = da.metadata['ixseps2']

    if not da.metadata['keep_xboundaries']:
        # remove x-boundary cell count from ix1 and ix2
        x_boundary_guards = da.metadata['MXG']
        ix1 -= x_boundary_guards
        ix2 -= x_boundary_guards

    ny = da.metadata['ny']
    j11 = da.metadata['jyseps1_1']
    j12 = da.metadata['jyseps1_2']
    j21 = da.metadata['jyseps2_1']
    j22 = da.metadata['jyseps2_2']
    nin = da.metadata.get('ny_inner', j12)

    ny_array = len(da['theta'])

    if da.metadata['keep_yboundaries']:
        y_boundary_guards = da.metadata['MYG']
    else:
        y_boundary_guards = 0

    if ny_array == ny:
        # No y-boundary cells, or keep_yboundaries is False
        if y_boundary_guards > 0 and da.metadata['keep_yboundaries']:
            raise ValueError('keep_yboundaries is True and y_boundary_guards={}, which '
                             'is greater than 0, but data does not havy y-boundary '
                             'cells.')
        y_boundary_guards = 0
    elif j12 == j21 and ny_array == ny + 2*y_boundary_guards:
        # single-null with guard cells
        pass
    elif j12 > j21 and ny_array == ny + 4*y_boundary_guards:
        # double-null with guard cells
        pass
    else:
        print('j21={}, j12={}, ny_array={}, ny={}'.format(j21, j12, ny_array, ny))
        raise ValueError("Unrecognized combination of ny/jyseps")

    # translate topology indices - ones from BOUT++ do not include boundary cells
    if j21 == j12:
        upper_y_boundary_guards = 0
    else:
        upper_y_boundary_guards = y_boundary_guards
    j11 += y_boundary_guards
    j21 += y_boundary_guards
    nin += y_boundary_guards + upper_y_boundary_guards
    j12 += y_boundary_guards + 2*upper_y_boundary_guards
    j22 += y_boundary_guards + 2*upper_y_boundary_guards
    ny += 2*y_boundary_guards + 2*upper_y_boundary_guards

    return j11, j12, j21, j22, ix1, ix2, nin, nx, ny, y_boundary_guards
