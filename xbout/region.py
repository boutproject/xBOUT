from .utils import _set_attrs_on_all_vars


class Region:
    """
    Contains the global indices bounding a single topological region, i.e. a region with
    logically rectangular contiguous data.

    Also stores the names of any neighbouring regions.
    """
    def __init__(self, *, name, ds=None, xinner_ind=None, xouter_ind=None,
                 ylower_ind=None, yupper_ind=None, connect_inner=None,
                 connect_outer=None, connect_lower=None, connect_upper=None):
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
            # self.nx, self.ny should not include boundary points.
            # self.xinner, self.xouter, self.ylower, self.yupper
            if ds.metadata['keep_xboundaries']:
                xbndry = ds.metadata['MXG']
                if self.connection_inner is None:
                    self.nx -= xbndry

                    # used to calculate x-coordinate of inner side (self.xinner)
                    xinner_ind += xbndry

                if self.connection_outer is None:
                    self.nx -= xbndry

                    # used to calculate x-coordinate of outer side (self.xouter)
                    xouter_ind -= xbndry

            if ds.metadata['keep_yboundaries']:
                ybndry = ds.metadata['MYG']
                if self.connection_lower is None:
                    self.ny -= ybndry
                    print('check ny 2', self.ny, self.connection_lower, connect_lower)

                    # used to calculate y-coordinate of lower side (self.ylower)
                    ylower_ind += ybndry

                if self.connection_upper is None:
                    self.ny -= ybndry
                    print('check ny 3', self.ny, self.connection_upper, connect_upper)

                    # used to calculate y-coordinate of upper side (self.yupper)
                    yupper_ind -= ybndry

            # calculate start and end coordinates
            #####################################
            xcoord = ds.metadata['bout_xdim']
            ycoord = ds.metadata['bout_ydim']

            # Note the global coordinates used here are defined so that they are zero at
            # the boundaries of the grid (where the grid includes all boundary cells),
            # not necessarily the physical boundaries because constant offsets do not
            # matter, as long as these bounds are consistent with the global coordinates
            # defined in apply_geometry (we will only use these coordinates for
            # interpolation) and it is simplest to calculate them with cumsum().

            # dx is constant in any particular region in the y-direction, so convert to a
            # 1d array
            # Note that this is not the same coordinate as the 'x' coordinate that is
            # created by default from the x-index, as these values are set only for
            # particular regions, so do not need to be consistent between different
            # regions (e.g. core and PFR), so we are not forced to use just the index
            # value here.
            dx = ds['dx'].isel(**{ycoord: self.ylower_ind})
            dx_cumsum = dx.cumsum()
            self.xinner = dx_cumsum[xinner_ind] - dx[xinner_ind]
            self.xouter = dx_cumsum[xouter_ind - 1] + dx[xouter_ind - 1]

            # dy is constant in the x-direction, so convert to a 1d array
            dy = ds['dy'].isel(**{xcoord: self.xinner_ind})
            dy_cumsum = dy.cumsum()
            self.ylower = dy_cumsum[ylower_ind] - dy[ylower_ind]
            self.yupper = dy_cumsum[yupper_ind - 1]

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
    ny_inner = ds.metadata['ny_inner']
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
        if jys11 < ny_inner - 1 and jys22 > ny_inner:
            return 'xpoint'
        else:
            raise ValueError('Currently unsupported topology')

    if ixs1 == ixs2:
        if jys21 < ny_inner - 1 and jys12 > ny_inner:
            return 'connected-double-null'
        else:
            raise ValueError('Currently unsupported topology')

    return 'disconnected-double-null'


def _check_connections(regions):
    for region in regions.values():
        if region.connection_inner is not None:
            if regions[region.connection_inner].connection_outer != region.name:
                raise ValueError(
                        f'Inner connection of {region.name} is '
                        f'{region.connection_inner}, but outer connection of '
                        f'{region.connection_inner} is '
                        f'{regions[region.connection_inner].connection_outer}')
        if region.connection_outer is not None:
            if regions[region.connection_outer].connection_inner != region.name:
                raise ValueError(
                        f'Inner connection of {region.name} is '
                        f'{region.connection_outer}, but inner connection of '
                        f'{region.connection_outer} is '
                        f'{regions[region.connection_outer].connection_inner}')
        if region.connection_lower is not None:
            if regions[region.connection_lower].connection_upper != region.name:
                raise ValueError(
                        f'Inner connection of {region.name} is '
                        f'{region.connection_lower}, but upper connection of '
                        f'{region.connection_lower} is '
                        f'{regions[region.connection_lower].connection_upper}')
        if region.connection_upper is not None:
            if regions[region.connection_upper].connection_lower != region.name:
                raise ValueError(
                        f'Inner connection of {region.name} is '
                        f'{region.connection_upper}, but lower connection of '
                        f'{region.connection_upper} is '
                        f'{regions[region.connection_upper].connection_lower}')


topologies = {}


def topology_disconnected_double_null(*, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner,
                                      jys12, jys22, ny, ybndry):
    regions = {}
    regions['lower_inner_PFR'] = Region(
            name='lower_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=0, yupper_ind=jys11 + 1, connect_outer='lower_inner_intersep',
            connect_upper='lower_outer_PFR')
    regions['lower_inner_intersep'] = Region(
            name='lower_inner_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
            ylower_ind=0, yupper_ind=jys11 + 1, connect_inner='lower_inner_PFR',
            connect_outer='lower_inner_SOL', connect_upper='inner_intersep')
    regions['lower_inner_SOL'] = Region(
            name='lower_inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=0, yupper_ind=jys11 + 1, connect_inner='lower_inner_intersep',
            connect_upper='inner_SOL')
    regions['inner_core'] = Region(
            name='inner_core', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys11 + 1, yupper_ind=jys21 + 1,
            connect_outer='inner_intersep', connect_lower='outer_core',
            connect_upper='outer_core')
    regions['inner_intersep'] = Region(
            name='inner_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
            ylower_ind=jys11 + 1, yupper_ind=jys21 + 1, connect_inner='inner_core',
            connect_outer='inner_SOL', connect_lower='lower_inner_intersep',
            connect_upper='outer_intersep')
    regions['inner_SOL'] = Region(
            name='inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=jys11 + 1, yupper_ind=jys21 + 1,
            connect_inner='inner_intersep', connect_lower='lower_inner_SOL',
            connect_upper='upper_inner_SOL')
    regions['upper_inner_PFR'] = Region(
            name='upper_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys21 + 1, yupper_ind=ny_inner,
            connect_outer='upper_inner_intersep', connect_lower='upper_outer_PFR')
    regions['upper_inner_intersep'] = Region(
            name='upper_inner_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
            ylower_ind=jys21 + 1, yupper_ind=ny_inner,
            connect_inner='upper_inner_PFR', connect_outer='upper_inner_SOL',
            connect_lower='upper_outer_intersep')
    regions['upper_inner_SOL'] = Region(
            name='upper_inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=jys21 + 1, yupper_ind=ny_inner,
            connect_inner='upper_inner_intersep', connect_lower='inner_SOL')
    regions['upper_outer_PFR'] = Region(
            name='upper_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=ny_inner, yupper_ind=jys12 + 1,
            connect_outer='upper_outer_intersep', connect_upper='upper_inner_PFR')
    regions['upper_outer_intersep'] = Region(
            name='upper_outer_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
            ylower_ind=ny_inner, yupper_ind=jys12 + 1,
            connect_inner='upper_outer_PFR', connect_outer='upper_outer_SOL',
            connect_upper='upper_inner_intersep')
    regions['upper_outer_SOL'] = Region(
            name='upper_outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=ny_inner, yupper_ind=jys12 + 1,
            connect_inner='upper_outer_intersep', connect_upper='outer_SOL')
    regions['outer_core'] = Region(
            name='outer_core', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys12 + 1, yupper_ind=jys22 + 1,
            connect_outer='outer_intersep', connect_lower='inner_core',
            connect_upper='inner_core')
    regions['outer_intersep'] = Region(
            name='outer_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
            ylower_ind=jys12 + 1, yupper_ind=jys22 + 1, connect_inner='outer_core',
            connect_outer='outer_SOL', connect_lower='inner_intersep',
            connect_upper='lower_outer_intersep')
    regions['outer_SOL'] = Region(
            name='outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=jys12 + 1, yupper_ind=jys22 + 1,
            connect_inner='outer_intersep', connect_lower='upper_outer_SOL',
            connect_upper='lower_outer_SOL')
    regions['lower_outer_PFR'] = Region(
            name='lower_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys22 + 1, yupper_ind=ny,
            connect_outer='lower_outer_intersep', connect_lower='lower_inner_PFR')
    regions['lower_outer_intersep'] = Region(
            name='lower_outer_intersep', ds=ds, xinner_ind=ixs1, xouter_ind=ixs2,
            ylower_ind=jys22 + 1, yupper_ind=ny, connect_inner='lower_outer_PFR',
            connect_outer='lower_outer_SOL', connect_lower='outer_intersep')
    regions['lower_outer_SOL'] = Region(
            name='lower_outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=jys22 + 1, yupper_ind=ny,
            connect_inner='lower_outer_intersep', connect_lower='outer_SOL')
    return regions


topologies['disconnected-double-null'] = topology_disconnected_double_null


def topology_connected_double_null(*, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12,
                                   jys22, ny, ybndry):
    regions = {}
    regions['lower_inner_PFR'] = Region(
            name='lower_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=0, yupper_ind=jys11 + 1, connect_outer='lower_inner_SOL',
            connect_upper='lower_outer_PFR')
    regions['lower_inner_SOL'] = Region(
            name='lower_inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=0, yupper_ind=jys11 + 1, connect_inner='lower_inner_PFR',
            connect_upper='inner_SOL')
    regions['inner_core'] = Region(
            name='inner_core', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys11 + 1, yupper_ind=jys21 + 1, connect_outer='inner_SOL',
            connect_lower='outer_core', connect_upper='outer_core')
    regions['inner_SOL'] = Region(
            name='inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=jys11 + 1, yupper_ind=jys21 + 1, connect_inner='inner_core',
            connect_lower='lower_inner_SOL', connect_upper='upper_inner_SOL')
    regions['upper_inner_PFR'] = Region(
            name='upper_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys21 + 1, yupper_ind=ny_inner,
            connect_outer='upper_inner_SOL', connect_lower='upper_outer_PFR')
    regions['upper_inner_SOL'] = Region(
            name='upper_inner_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=jys21 + 1, yupper_ind=ny_inner,
            connect_inner='upper_inner_PFR', connect_lower='inner_SOL')
    regions['upper_outer_PFR'] = Region(
            name='upper_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=ny_inner, yupper_ind=jys12 + 1,
            connect_outer='upper_outer_SOL', connect_upper='upper_inner_PFR')
    regions['upper_outer_SOL'] = Region(
            name='upper_outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=ny_inner, yupper_ind=jys12 + 1,
            connect_inner='upper_outer_PFR', connect_upper='outer_SOL')
    regions['outer_core'] = Region(
            name='outer_core', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys12 + 1, yupper_ind=jys22 + 1, connect_outer='outer_SOL',
            connect_lower='inner_core', connect_upper='inner_core')
    regions['outer_SOL'] = Region(
            name='outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=jys12 + 1, yupper_ind=jys22 + 1, connect_inner='outer_core',
            connect_lower='upper_outer_SOL', connect_upper='lower_outer_SOL')
    regions['lower_outer_PFR'] = Region(
            name='lower_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys22 + 1, yupper_ind=ny, connect_outer='lower_outer_SOL',
            connect_lower='lower_inner_PFR')
    regions['lower_outer_SOL'] = Region(
            name='lower_outer_SOL', ds=ds, xinner_ind=ixs2, xouter_ind=nx,
            ylower_ind=jys22 + 1, yupper_ind=ny, connect_inner='lower_outer_PFR',
            connect_lower='outer_SOL')
    return regions


topologies['connected-double-null'] = topology_connected_double_null


def topology_single_null(*, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22,
                         ny, ybndry):
    regions = {}
    regions['inner_PFR'] = Region(
            name='inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1, ylower_ind=0,
            yupper_ind=jys11 + 1, connect_outer='inner_SOL',
            connect_upper='outer_PFR')
    regions['inner_SOL'] = Region(
            name='inner_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx, ylower_ind=0,
            yupper_ind=jys11 + 1, connect_inner='inner_PFR', connect_upper='SOL')
    regions['core'] = Region(
            name='core', ds=ds, xinner_ind=0, xouter_ind=ixs1, ylower_ind=jys11 + 1,
            yupper_ind=jys22 + 1, connect_outer='SOL', connect_lower='core',
            connect_upper='core')
    regions['SOL'] = Region(
            name='SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx, ylower_ind=jys11 + 1,
            yupper_ind=jys22 + 1, connect_inner='core', connect_lower='inner_SOL',
            connect_upper='outer_SOL')
    regions['outer_PFR'] = Region(
            name='outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys22 + 1, yupper_ind=ny, connect_outer='outer_SOL',
            connect_lower='inner_PFR')
    regions['outer_SOL'] = Region(
            name='outer_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
            ylower_ind=jys22 + 1, yupper_ind=ny, connect_inner='outer_PFR',
            connect_lower='SOL')
    return regions


topologies['single-null'] = topology_single_null


def topology_limiter(*, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny,
                     ybndry):
    regions = {}
    regions['core'] = Region(
            name='core', ds=ds, xinner_ind=0, xouter_ind=ixs1, ylower_ind=ybndry,
            yupper_ind=ny - ybndry, connect_outer='SOL', connect_lower='core',
            connect_upper='core')
    regions['SOL'] = Region(
            name='SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx, ylower_ind=0,
            yupper_ind=ny, connect_inner='core')
    return regions


topologies['limiter'] = topology_limiter


def topology_core(*, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny,
                  ybndry):
    regions = {}
    regions['core'] = Region(
            name='core', ds=ds, xinner_ind=0, xouter_ind=nx, ylower_ind=ybndry,
            yupper_ind=ny - ybndry, connect_lower='core', connect_upper='core')
    return regions


topologies['core'] = topology_core


def topology_sol(*, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny,
                 ybndry):
    regions = {}
    regions['SOL'] = Region(
            name='SOL', ds=ds, xinner_ind=0, xouter_ind=nx, ylower_ind=0,
            yupper_ind=ny)
    return regions


topologies['sol'] = topology_sol


def topology_xpoint(*, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny,
                    ybndry):
    regions = {}
    regions['lower_inner_PFR'] = Region(
            name='lower_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=0, yupper_ind=jys11 + 1, connect_outer='lower_inner_SOL',
            connect_upper='lower_outer_PFR')
    regions['lower_inner_SOL'] = Region(
            name='lower_inner_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
            ylower_ind=0, yupper_ind=jys11 + 1, connect_inner='lower_inner_PFR',
            connect_upper='upper_inner_SOL')
    regions['upper_inner_PFR'] = Region(
            name='upper_inner_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys11 + 1, yupper_ind=ny_inner,
            connect_outer='upper_inner_SOL', connect_lower='upper_outer_PFR')
    regions['upper_inner_SOL'] = Region(
            name='upper_inner_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
            ylower_ind=jys11 + 1, yupper_ind=ny_inner,
            connect_inner='upper_inner_PFR', connect_lower='lower_inner_SOL')
    regions['upper_outer_PFR'] = Region(
            name='upper_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=ny_inner, yupper_ind=jys22 + 1,
            connect_outer='upper_outer_SOL', connect_upper='upper_inner_PFR')
    regions['upper_outer_SOL'] = Region(
            name='upper_outer_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
            ylower_ind=ny_inner, yupper_ind=jys22 + 1,
            connect_inner='upper_outer_PFR', connect_upper='lower_outer_SOL')
    regions['lower_outer_PFR'] = Region(
            name='lower_outer_PFR', ds=ds, xinner_ind=0, xouter_ind=ixs1,
            ylower_ind=jys22 + 1, yupper_ind=ny, connect_outer='lower_outer_SOL',
            connect_lower='lower_inner_PFR')
    regions['lower_outer_SOL'] = Region(
            name='lower_outer_SOL', ds=ds, xinner_ind=ixs1, xouter_ind=nx,
            ylower_ind=jys22 + 1, yupper_ind=ny, connect_inner='lower_outer_PFR',
            connect_lower='upper_outer_SOL')
    return regions


topologies['xpoint'] = topology_xpoint


def _create_regions_toroidal(ds):
    topology = _get_topology(ds)

    ixs1 = ds.metadata['ixseps1']
    ixs2 = ds.metadata['ixseps2']
    nx = ds.metadata['nx']

    jys11 = ds.metadata['jyseps1_1']
    jys21 = ds.metadata['jyseps2_1']
    ny_inner = ds.metadata['ny_inner']
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
    ny_inner = _in_range(ny_inner, jys21 + 1, jys12 + 1)
    jys22 = _in_range(jys22, 0, ny - 1)

    # Adjust for boundary cells
    # keep_xboundaries is 1 if there are x-boundaries and 0 if there are not
    if not ds.metadata['keep_xboundaries']:
        ixs1 -= mxg
        ixs2 -= mxg
        nx -= 2*mxg
    jys11 += ybndry
    jys21 += ybndry
    ny_inner += ybndry + ybndry_upper
    jys12 += ybndry + 2*ybndry_upper
    jys22 += ybndry + 2*ybndry_upper
    ny += 2*ybndry + 2*ybndry_upper

    # Note, include guard cells in the created regions, fill them later
    try:
        regions = topologies[topology](ds=ds, ixs1=ixs1, ixs2=ixs2, nx=nx, jys11=jys11,
                                       jys21=jys21, ny_inner=ny_inner, jys12=jys12,
                                       jys22=jys22, ny=ny, ybndry=ybndry)
    except KeyError:
        raise NotImplementedError(f"Topology '{topology}' is not implemented")

    _check_connections(regions)

    ds = _set_attrs_on_all_vars(ds, 'regions', regions)

    return ds
