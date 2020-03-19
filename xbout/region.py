from collections import OrderedDict

import numpy as np

from .utils import _set_attrs_on_all_vars


class Region:
    """
    Contains the global indices bounding a single topological region, i.e. a region with
    logically rectangular contiguous data.

    Also stores the names of any neighbouring regions.
    """
    def __init__(self, *, name, ds=None, xinner_ind=None, xouter_ind=None,
                 ylower_ind=None, yupper_ind=None, connect_inner=None, connect_outer=None,
                 connect_lower=None, connect_upper=None):
        """
        Parameters
        ----------
        name : str
            Name of the region
        ds : BoutDataset, optional
            Dataset to get variables to calculate coordinates from
        xinner_ind : int, optional
            Global x-index of the inner points of this region
        xouter_ind : int, optional
            Global x-index of the points just beyond the outer edge of this region
        ylower_ind : int, optional
            Global y-index of the lower points of this region
        yupper_ind : int, optional
            Global y-index of the points just beyond the upper edge of this region
        connect_inner : str, optional
            The region inside this one in the x-direction
        connect_outer : str, optional
            The region outside this one in the x-direction
        connect_lower : str, optional
            The region below this one in the y-direction
        connect_upper : str, optional
            The region above this one in the y-direction
        """
        self.name = name
        self.xinner_ind = xinner_ind
        self.xouter_ind = xouter_ind
        if xouter_ind is not None and xinner_ind is not None:
            self.nx = xouter_ind - xinner_ind
        self.ylower_ind = ylower_ind
        self.yupper_ind = yupper_ind
        if yupper_ind is not None and ylower_ind is not None:
            self.ny = yupper_ind - ylower_ind
        self.connection_inner = connect_inner
        self.connection_outer = connect_outer
        self.connection_lower = connect_lower
        self.connection_upper = connect_upper

        if ds is not None:
            # calculate start and end coordinates
            #####################################
            xcoord = ds.metadata['bout_xdim']
            ycoord = ds.metadata['bout_ydim']

            # dx is constant in any particular region in the y-direction, so convert to a 1d
            # array
            dx = ds['dx'].isel(**{ycoord: self.ylower_ind})
            dx_cumsum = dx.cumsum()
            self.xinner = dx_cumsum[xinner_ind] - dx[xinner_ind]/2.
            self.xouter = dx_cumsum[xouter_ind - 1] + dx[xouter_ind - 1]/2.

            # dy is constant in the x-direction, so convert to a 1d array
            dy = ds['dy'].isel(**{xcoord: self.xinner_ind})
            dy_cumsum = dy.cumsum()
            self.ylower = dy_cumsum[ylower_ind] - dy[ylower_ind]/2.
            self.yupper = dy_cumsum[yupper_ind - 1] + dy[yupper_ind - 1]/2.

    def __repr__(self):
        result = "<xbout.region.Region>\n"
        for attr, val in vars(self).items():
            result += "\t" + attr + "\t" + str(val) + "\n"
        return result

    def getSlices(self, mxg=0, myg=0):
        """
        Return x- and y-dimension slices that select this region from the global
        DataArray.

        Returns
        -------
        xslice, yslice : slice, slice
        """
        xi = self.xinner_ind
        if self.connection_inner is not None:
            xi -= mxg

        xo = self.xouter_ind
        if self.connection_outer is not None:
            xi += mxg

        yl = self.ylower_ind
        if self.connection_lower is not None:
            yl -= myg

        yu = self.yupper_ind
        if self.connection_upper is not None:
            yu += myg

        return slice(xi, xo), slice(yl, yu)

    def getInnerGuardsSlices(self, *, mxg, myg=0):
        """
        Return x- and y-dimension slices that select mxg guard cells on the inner-x side
        of this region from the global DataArray.

        Parameters
        ----------
        mxg : int
            Number of guard cells
        myg : int, optional
            Number of y-guard cells to include at the corners
        """
        ylower = self.ylower_ind
        if self.connection_lower is not None:
            ylower -= myg
        yupper = self.yupper_ind
        if self.connection_upper is not None:
            yupper += myg
        return slice(self.xinner_ind - mxg, self.xinner_ind), slice(ylower, yupper)

    def getOuterGuardsSlices(self, *, mxg, myg=0):
        """
        Return x- and y-dimension slices that select mxg guard cells on the outer-x side
        of this region from the global DataArray.

        Parameters
        ----------
        mxg : int
            Number of guard cells
        myg : int, optional
            Number of y-guard cells to include at the corners
        """
        ylower = self.ylower_ind
        if self.connection_lower is not None:
            ylower -= myg
        yupper = self.yupper_ind
        if self.connection_upper is not None:
            yupper += myg
        return slice(self.xouter_ind, self.xouter_ind + mxg), slice(ylower, yupper)

    def getLowerGuardsSlices(self, *, myg, mxg=0):
        """
        Return x- and y-dimension slices that select myg guard cells on the lower-y side
        of this region from the global DataArray.

        Parameters
        ----------
        myg : int
            Number of guard cells
        mxg : int, optional
            Number of x-guard cells to include at the corners
        """
        xinner = self.xinner_ind
        if self.connection_inner is not None:
            xinner -= mxg
        xouter = self.xouter_ind
        if self.connection_outer is not None:
            xouter += mxg
        return slice(xinner, xouter), slice(self.ylower_ind - myg, self.ylower_ind)

    def getUpperGuardsSlices(self, *, myg, mxg=0):
        """
        Return x- and y-dimension slices that select myg guard cells on the upper-y side
        of this region from the global DataArray.

        Parameters
        ----------
        myg : int
            Number of guard cells
        mxg : int, optional
            Number of x-guard cells to include at the corners
        """
        xinner = self.xinner_ind
        if self.connection_inner is not None:
            xinner -= mxg
        xouter = self.xouter_ind
        if self.connection_outer is not None:
            xouter += mxg
        return slice(xinner, xouter), slice(self.yupper_ind, self.yupper_ind + myg)


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
        if jys11 < nyinner -1 and jys22 > nyinner:
            return 'xpoint'
        else:
            raise ValueError('Currently unsupported topology')

    if ixs1 == ixs2:
        if jys21 < nyinner -1 and jys12 > nyinner:
            return 'connected-double-null'
        else:
            raise ValueError('Currently unsupported topology')

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
    if jys21 == jys12:
        # No upper targets
        ybndry_upper = 0
    else:
        ybndry_upper = ybndry

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
    nyinner += ybndry + ybndry_upper
    jys12 += ybndry + 2*ybndry_upper
    jys22 += ybndry + 2*ybndry_upper
    ny += 2*ybndry + 2*ybndry_upper

    # Note, include guard cells in the created regions, fill them later
    regions = OrderedDict()
    if topology == 'disconnected-double-null':
        regions['lower_inner_PFR'] = Region(
                name='lower_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=0, yupper_ind=jys11 + 1)
        regions['lower_inner_intersep'] = Region(
                name='lower_inner_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
                ylower_ind=0, yupper_ind=jys11 + 1)
        regions['lower_inner_SOL'] = Region(
                name='lower_inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=0, yupper_ind=jys11 + 1)
        regions['inner_core'] = Region(
                name='inner_core', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys11 + 1, yupper_ind=jys21 + 1)
        regions['inner_intersep'] = Region(
                name='inner_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
                ylower_ind=jys11 + 1, yupper_ind=jys21 + 1)
        regions['inner_SOL'] = Region(
                name='inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=jys11 + 1, yupper_ind=jys21 + 1)
        regions['upper_inner_PFR'] = Region(
                name='upper_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys21 + 1, yupper_ind=nyinner)
        regions['upper_inner_intersep'] = Region(
                name='upper_inner_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
                ylower_ind=jys21 + 1, yupper_ind=nyinner)
        regions['upper_inner_SOL'] = Region(
                name='upper_inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=jys21 + 1, yupper_ind=nyinner)
        regions['upper_outer_PFR'] = Region(
                name='upper_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=nyinner, yupper_ind=jys12 + 1)
        regions['upper_outer_intersep'] = Region(
                name='upper_outer_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
                ylower_ind=nyinner, yupper_ind=jys12 + 1)
        regions['upper_outer_SOL'] = Region(
                name='upper_outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=nyinner, yupper_ind=jys12 + 1)
        regions['outer_core'] = Region(
                name='outer_core', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys12 + 1, yupper_ind=jys22 + 1)
        regions['outer_intersep'] = Region(
                name='outer_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
                ylower_ind=jys12 + 1, yupper_ind=jys22 + 1)
        regions['outer_SOL'] = Region(
                name='outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=jys12 + 1, yupper_ind=jys22 + 1)
        regions['lower_outer_PFR'] = Region(
                name='lower_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys22 + 1, yupper_ind=ny)
        regions['lower_outer_intersep'] = Region(
                name='lower_outer_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
                ylower_ind=jys22 + 1, yupper_ind=ny)
        regions['lower_outer_SOL'] = Region(
                name='lower_outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=jys22 + 1, yupper_ind=ny)
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
                name='lower_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=0, yupper_ind=jys11 + 1)
        regions['lower_inner_SOL'] = Region(
                name='lower_inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=0, yupper_ind=jys11 + 1)
        regions['inner_core'] = Region(
                name='inner_core', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys11 + 1, yupper_ind=jys21 + 1)
        regions['inner_SOL'] = Region(
                name='inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=jys11 + 1, yupper_ind=jys21 + 1)
        regions['upper_inner_PFR'] = Region(
                name='upper_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys21 + 1, yupper_ind=nyinner)
        regions['upper_inner_SOL'] = Region(
                name='upper_inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=jys21 + 1, yupper_ind=nyinner)
        regions['upper_outer_PFR'] = Region(
                name='upper_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=nyinner, yupper_ind=jys12 + 1)
        regions['upper_outer_SOL'] = Region(
                name='upper_outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=nyinner, yupper_ind=jys12 + 1)
        regions['outer_core'] = Region(
                name='outer_core', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys12 + 1, yupper_ind=jys22 + 1)
        regions['outer_SOL'] = Region(
                name='outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=jys12 + 1, yupper_ind=jys22 + 1)
        regions['lower_outer_PFR'] = Region(
                name='lower_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys22 + 1, yupper_ind=ny)
        regions['lower_outer_SOL'] = Region(
                name='lower_outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
                ylower_ind=jys22 + 1, yupper_ind=ny)
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
                name='inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1, ylower_ind=0,
                yupper_ind=jys11 + 1)
        regions['inner_SOL'] = Region(
                name='inner_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx, ylower_ind=0,
                yupper_ind=jys11 + 1)
        regions['core'] = Region(
                name='core', ds=ds, xinner_ind=0, xouter_ind=ixs1, ylower_ind=jys11 + 1,
                yupper_ind=jys22 + 1)
        regions['SOL'] = Region(
                name='SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx, ylower_ind=jys11 + 1,
                yupper_ind=jys22 + 1)
        regions['outer_PFR'] = Region(
                name='lower_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys22 + 1, yupper_ind=ny)
        regions['outer_SOL'] = Region(
                name='lower_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
                ylower_ind=jys22 + 1, yupper_ind=ny)
        _create_connection_x(regions, 'inner_PFR', 'inner_SOL')
        _create_connection_x(regions, 'core', 'SOL')
        _create_connection_x(regions, 'outer_PFR', 'outer_SOL')
        _create_connection_y(regions, 'inner_PFR', 'outer_PFR')
        _create_connection_y(regions, 'inner_SOL', 'SOL')
        _create_connection_y(regions, 'core', 'core')
        _create_connection_y(regions, 'SOL', 'outer_SOL')
    elif topology == 'limiter':
        regions['core'] = Region(
                name='core', ds=ds, xinner_ind=0, xouter_ind=ixs1, ylower_ind=ybndry,
                yupper_ind=ny - ybndry)
        regions['SOL'] = Region(
                name='SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx, ylower_ind=0,
                yupper_ind=ny)
        _create_connection_x(regions, 'core', 'SOL')
        _create_connection_y(regions, 'core', 'core')
    elif topology == 'core':
        regions['core'] = Region(
                name='core', ds=ds, xinner_ind=0, xouter_ind=nx, ylower_ind=ybndry,
                yupper_ind=ny - ybndry)
        _create_connection_y(regions, 'core', 'core')
    elif topology == 'sol':
        regions['SOL'] = Region(
                name='SOL', ds=ds, xinner_ind=0, xouter_ind=nx, ylower_ind=0,
                yupper_ind=ny)
    elif topology == 'xpoint':
        regions['lower_inner_PFR'] = Region(
                name='lower_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=0, yupper_ind=jys11 + 1)
        regions['lower_inner_SOL'] = Region(
                name='lower_inner_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
                ylower_ind=0, yupper_ind=jys11 + 1)
        regions['upper_inner_PFR'] = Region(
                name='upper_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys11 + 1, yupper_ind=nyinner)
        regions['upper_inner_SOL'] = Region(
                name='upper_inner_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
                ylower_ind=jys11 + 1, yupper_ind=nyinner)
        regions['upper_outer_PFR'] = Region(
                name='upper_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=nyinner, yupper_ind=jys22 + 1)
        regions['upper_outer_SOL'] = Region(
                name='upper_outer_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
                ylower_ind=nyinner, yupper_ind=jys22 + 1)
        regions['lower_outer_PFR'] = Region(
                name='lower_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
                ylower_ind=jys22 + 1, yupper_ind=ny)
        regions['lower_outer_SOL'] = Region(
                name='lower_outer_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
                ylower_ind=jys22 + 1, yupper_ind=ny)
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
