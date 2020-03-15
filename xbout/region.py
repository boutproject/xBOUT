from .utils import _set_attrs_on_all_vars

import numpy as np


class Region:
    """
    Contains the global indices bounding a single topological region, i.e. a region with
    logically rectangular contiguous data.

    Also stores the names of any neighbouring regions.
    """
    def __init__(self, name, xinner, xouter, ylower, yupper, *, connect_inner=None,
                 connect_outer=None, connect_lower=None, connect_upper=None):
        self.name = name
        self.xinner = xinner
        self.xouter = xouter
        self.ylower = ylower
        self.yupper = yupper
        self.connection_inner = connect_inner
        self.connection_outer = connect_outer
        self.connection_lower = connect_lower
        self.connection_upper = connect_upper

    def getSlices(self, mxg=0, myg=0):
        """
        Return x- and y-dimension slices that select this region from the global
        DataArray.

        Returns
        -------
        xslice, yslice : slice, slice
        """
        xi = self.xinner
        if self.connection_inner is not None:
            xi -= mxg

        xo = self.xouter
        if self.connection_outer is not None:
            xi += mxg

        yl = self.ylower
        if self.connection_lower is not None:
            yl -= myg

        yu = self.yupper
        if self.connection_upper is not None:
            yu += myg

        return slice(xi, xo), slice(yl, yu)


def _in_range(val, lower, upper):
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val


def _order_vars(lower, upper):
    if upper < lower:
        return upper, lower
    else:
        return lower, upper


def _get_topology(ds):
    jys11 = ds.metadata['jyseps1_1']
    jys21 = ds.metadata['jyseps2_1']
    nyinner = ds.metadata['ny_inner']
    jys12 = ds.metadata['jyseps1_2']
    jys22 = ds.metadata['jyseps2_2']
    ny = ds.metadata['ny']
    ixs1 = ds.metadata['ixseps1']
    ixs2 = ds.metadata['ixseps2']
    nx = ds.metadata['nx']
    if jys21 == jys12:
        # No upper X-point
        if jys11 <= 0 and jys22 >= ny - 1:
            ix = min(ixs1, ixs2)
            if ix >= nx - 1:
                return 'core'
            elif ix <= 0:
                return 'sol'
            else:
                return 'limiter'

        return 'single-null'

    if jys11 == jys21 and jys12 == jys22:
        return 'xpoint'

    if ixs1 == ixs2:
        return 'connected-double-null'

    return 'disconnected-double-null'


def _create_connection_x(regions, inner, outer):
    regions[inner].connection_outer = outer
    regions[outer].connection_inner = inner


def _create_connection_y(regions, lower, upper):
    regions[lower].connection_upper = upper
    regions[upper].connection_lower = lower


def _create_regions_toroidal(ds):
    topology = _get_topology(ds)

    coordinates = {'t': ds.metadata.get('bout_tdim', None),
                   'x': ds.metadata.get('bout_xdim', None),
                   'y': ds.metadata.get('bout_ydim', None),
                   'z': ds.metadata.get('bout_zdim', None)}

    ixs1 = ds.metadata['ixseps1']
    ixs2 = ds.metadata['ixseps2']
    nx = ds.metadata['nx']

    jys11 = ds.metadata['jyseps1_1']
    jys21 = ds.metadata['jyseps2_1']
    nyinner = ds.metadata['ny_inner']
    jys12 = ds.metadata['jyseps1_2']
    jys22 = ds.metadata['jyseps2_2']
    ny = ds.metadata['ny']

    mxg = ds.metadata['MXG']
    myg = ds.metadata['MYG']
    # keep_yboundaries is 1 if there are y-boundaries and 0 if there are not
    ybndry = ds.metadata['keep_yboundaries']*myg

    # Make sure all sizes are sensible
    ixs1 = _in_range(ixs1, 0, nx)
    ixs2 = _in_range(ixs2, 0, nx)
    ixs1, ixs2 = _order_vars(ixs1, ixs2)
    jys11 = _in_range(jys11, 0, ny - 1)
    jys21 = _in_range(jys21, 0, ny - 1)
    jys12 = _in_range(jys12, 0, ny - 1)
    jys21, jys12 = _order_vars(jys21, jys12)
    nyinner = _in_range(nyinner, jys21 + 1, jys12 + 1)
    jys22 = _in_range(jys22, 0, ny - 1)

    # Adjust for boundary cells
    # keep_xboundaries is 1 if there are x-boundaries and 0 if there are not
    if not ds.metadata['keep_xboundaries']:
        ixs1 -= mxg
        ixs2 -= mxg
        nx -= 2*mxg
    jys11 += ybndry
    jys21 += ybndry
    nyinner += 2*ybndry
    jys12 += 3*ybndry
    jys22 += 3*ybndry
    ny += 4*ybndry

    # Note, include guard cells in the created regions, fill them later
    regions = {}
    if topology == 'disconnected-double-null':
        regions['lower_inner_PFR'] = Region(
                'lower_inner_PFR', 0, ixs1, 0, jys11 + 1)
        regions['lower_inner_intersep'] = Region(
                'lower_inner_intersep', ixs1, ixs2, 0, jys11 + 1)
        regions['lower_inner_SOL'] = Region(
                'lower_inner_SOL', ixs2, nx, 0, jys11 + 1)
        regions['inner_core'] = Region(
                'inner_core', 0, ixs1, jys11 + 1, jys21 + 1)
        regions['inner_intersep'] = Region(
                'inner_intersep', ixs1, ixs2, jys11 + 1, jys21 + 1)
        regions['inner_SOL'] = Region(
                'inner_SOL', ixs2, nx, jys11 + 1, jys21 + 1)
        regions['upper_inner_PFR'] = Region(
                'upper_inner_PFR', 0, ixs1, jys21 + 1, nyinner)
        regions['upper_inner_intersep'] = Region(
                'upper_inner_intersep', ixs1, ixs2, jys21 + 1, nyinner)
        regions['upper_inner_SOL'] = Region(
                'upper_inner_SOL', ixs2, nx, jys21 + 1, nyinner)
        regions['upper_outer_PFR'] = Region(
                'upper_outer_PFR', 0, ixs1, nyinner, jys12 + 1)
        regions['upper_outer_intersep'] = Region(
                'upper_outer_intersep', ixs1, ixs2, nyinner, jys12 + 1)
        regions['upper_outer_SOL'] = Region(
                'upper_outer_SOL', ixs2, nx, nyinner, jys12 + 1)
        regions['outer_core'] = Region(
                'outer_core', 0, ixs1, jys12 + 1, jys22 + 1)
        regions['outer_intersep'] = Region(
                'outer_intersep', ixs1, ixs2, jys12 + 1, jys22 + 1)
        regions['outer_SOL'] = Region(
                'outer_SOL', ixs2, nx, jys12 + 1, jys22 + 1)
        regions['lower_outer_PFR'] = Region(
                'lower_outer_PFR', 0, ixs1, jys22 + 1, ny)
        regions['lower_outer_intersep'] = Region(
                'lower_outer_intersep', ixs1, ixs2, jys22 + 1, ny)
        regions['lower_outer_SOL'] = Region(
                'lower_outer_SOL', ixs2, nx, jys22 + 1, ny)
        _create_connection_x(regions, 'lower_inner_PFR', 'lower_inner_intersep')
        _create_connection_x(regions, 'lower_inner_intersep', 'lower_inner_SOL')
        _create_connection_x(regions, 'inner_core', 'inner_intersep')
        _create_connection_x(regions, 'inner_intersep', 'inner_SOL')
        _create_connection_x(regions, 'upper_inner_PFR', 'upper_inner_intersep')
        _create_connection_x(regions, 'upper_inner_intersep', 'upper_inner_SOL')
        _create_connection_x(regions, 'upper_outer_PFR', 'upper_outer_intersep')
        _create_connection_x(regions, 'upper_outer_intersep', 'upper_outer_SOL')
        _create_connection_x(regions, 'outer_core', 'outer_intersep')
        _create_connection_x(regions, 'outer_intersep', 'outer_SOL')
        _create_connection_x(regions, 'lower_outer_PFR', 'lower_outer_intersep')
        _create_connection_x(regions, 'lower_outer_intersep', 'lower_outer_SOL')
        _create_connection_y(regions, 'lower_inner_PFR', 'lower_outer_PFR')
        _create_connection_y(regions, 'lower_inner_intersep', 'inner_intersep')
        _create_connection_y(regions, 'lower_inner_SOL', 'inner_SOL')
        _create_connection_y(regions, 'inner_core', 'outer_core')
        _create_connection_y(regions, 'outer_core', 'inner_core')
        _create_connection_y(regions, 'inner_intersep', 'outer_intersep')
        _create_connection_y(regions, 'inner_SOL', 'upper_inner_SOL')
        _create_connection_y(regions, 'upper_outer_intersep', 'upper_inner_intersep')
        _create_connection_y(regions, 'upper_outer_PFR', 'upper_inner_PFR')
        _create_connection_y(regions, 'upper_outer_SOL', 'outer_SOL')
        _create_connection_y(regions, 'outer_intersep', 'lower_outer_intersep')
        _create_connection_y(regions, 'outer_SOL', 'lower_outer_SOL')
    elif topology == 'connected-double-null':
        regions['lower_inner_PFR'] = Region(
                'lower_inner_PFR', 0, ixs1, 0, jys11 + 1)
        regions['lower_inner_SOL'] = Region(
                'lower_inner_SOL', ixs2, nx, 0, jys11 + 1)
        regions['inner_core'] = Region(
                'inner_core', 0, ixs1, jys11 + 1, jys21 + 1)
        regions['inner_SOL'] = Region(
                'inner_SOL', ixs2, nx, jys11 + 1, jys21 + 1)
        regions['upper_inner_PFR'] = Region(
                'upper_inner_PFR', 0, ixs1, jys21 + 1, nyinner)
        regions['upper_inner_SOL'] = Region(
                'upper_inner_SOL', ixs2, nx, jys21 + 1, nyinner)
        regions['upper_outer_PFR'] = Region(
                'upper_outer_PFR', 0, ixs1, nyinner, jys12 + 1)
        regions['upper_outer_SOL'] = Region(
                'upper_outer_SOL', ixs2, nx, nyinner, jys12 + 1)
        regions['outer_core'] = Region(
                'outer_core', 0, ixs1, jys12 + 1, jys22 + 1)
        regions['outer_SOL'] = Region(
                'outer_SOL', ixs2, nx, jys12 + 1, jys22 + 1)
        regions['lower_outer_PFR'] = Region(
                'lower_outer_PFR', 0, ixs1, jys22 + 1, ny)
        regions['lower_outer_SOL'] = Region(
                'lower_outer_SOL', ixs2, nx, jys22 + 1, ny)
        _create_connection_x(regions, 'lower_inner_PFR', 'lower_inner_SOL')
        _create_connection_x(regions, 'inner_core', 'inner_SOL')
        _create_connection_x(regions, 'upper_inner_PFR', 'upper_inner_SOL')
        _create_connection_x(regions, 'upper_outer_PFR', 'upper_outer_SOL')
        _create_connection_x(regions, 'outer_core', 'outer_SOL')
        _create_connection_x(regions, 'lower_outer_PFR', 'lower_outer_SOL')
        _create_connection_y(regions, 'lower_inner_PFR', 'lower_outer_PFR')
        _create_connection_y(regions, 'lower_inner_SOL', 'inner_SOL')
        _create_connection_y(regions, 'inner_core', 'outer_core')
        _create_connection_y(regions, 'outer_core', 'inner_core')
        _create_connection_y(regions, 'inner_SOL', 'upper_inner_SOL')
        _create_connection_y(regions, 'upper_outer_PFR', 'upper_inner_PFR')
        _create_connection_y(regions, 'upper_outer_SOL', 'outer_SOL')
        _create_connection_y(regions, 'outer_SOL', 'lower_outer_SOL')
    elif topology == 'single-null':
        regions['inner_PFR'] = Region(
                'inner_PFR', 0, ixs1, 0, jys11 + 1)
        regions['inner_SOL'] = Region(
                'inner_SOL', ixs1, nx, 0, jys11 + 1)
        regions['core'] = Region(
                'core', 0, ixs1, jys11 + 1, jys22 + 1)
        regions['SOL'] = Region(
                'SOL', ixs2, nx, jys11 + 1, jys22 + 1)
        regions['outer_PFR'] = Region(
                'lower_PFR', 0, ixs1, jys22 + 1, ny)
        regions['outer_SOL'] = Region(
                'lower_SOL', ixs1, nx, jys22 + 1, ny)
        _create_connection_x(regions, 'inner_PFR', 'inner_SOL')
        _create_connection_x(regions, 'core', 'SOL')
        _create_connection_x(regions, 'outer_PFR', 'outer_SOL')
        _create_connection_y(regions, 'inner_PFR', 'outer_PFR')
        _create_connection_y(regions, 'inner_SOL', 'SOL')
        _create_connection_y(regions, 'core', 'core')
        _create_connection_y(regions, 'SOL', 'outer_SOL')
    elif topology == 'limiter':
        regions['core'] = Region(
                'core', 0, ixs1, 0, ny)
        regions['SOL'] = Region(
                'SOL', ixs1, nx, 0, ny)
        _create_connection_x(regions, 'core', 'SOL')
        _create_connection_y(regions, 'core', 'core')
    elif topology == 'core':
        regions['core'] = Region(
                'core', 0, nx, 0, ny)
        _create_connection_y(regions, 'core', 'core')
    elif topology == 'sol':
        regions['sol'] = Region(
                'sol', 0, nx, 0, ny)
    elif topology == 'xpoint':
        regions['lower_inner_PFR'] = Region(
                'lower_inner_PFR', 0, ixs1, 0, jys11 + 1)
        regions['lower_inner_SOL'] = Region(
                'lower_inner_SOL', ixs1, nx, 0, jys11 + 1)
        regions['upper_inner_PFR'] = Region(
                'upper_inner_PFR', 0, ixs1, jys11 + 1, nyinner)
        regions['upper_inner_SOL'] = Region(
                'upper_inner_SOL', ixs1, nx, jys11 + 1, nyinner)
        regions['upper_outer_PFR'] = Region(
                'upper_outer_PFR', 0, ixs1, nyinner, jys22 + 1)
        regions['upper_outer_SOL'] = Region(
                'upper_outer_SOL', ixs1, nx, nyinner, jys22 + 1)
        regions['lower_outer_PFR'] = Region(
                'lower_outer_PFR', 0, ixs1, jys22 + 1, ny)
        regions['lower_outer_SOL'] = Region(
                'lower_outer_SOL', ixs1, nx, jys22 + 1, ny)
        _create_connection_x(regions, 'lower_inner_PFR', 'lower_inner_SOL')
        _create_connection_x(regions, 'upper_inner_PFR', 'upper_inner_SOL')
        _create_connection_x(regions, 'upper_outer_PFR', 'upper_outer_SOL')
        _create_connection_x(regions, 'lower_outer_PFR', 'lower_outer_SOL')
        _create_connection_y(regions, 'lower_inner_PFR', 'lower_outer_PFR')
        _create_connection_y(regions, 'lower_inner_SOL', 'upper_inner_SOL')
        _create_connection_y(regions, 'upper_outer_PFR', 'upper_inner_PFR')
        _create_connection_y(regions, 'upper_outer_SOL', 'lower_outer_SOL')
    else:
        raise NotImplementedError("Topology '" + topology + "' is not implemented")

    ds = _set_attrs_on_all_vars(ds, 'regions', regions)

    return ds
