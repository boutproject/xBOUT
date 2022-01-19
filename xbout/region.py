from copy import copy, deepcopy
import numpy as np
import xarray as xr
from .utils import _set_attrs_on_all_vars


class Region:
    """
    Contains the global indices bounding a single topological region, i.e. a region with
    logically rectangular contiguous data.

    Also stores the names of any neighbouring regions.
    """

    def __init__(
        self,
        *,
        name,
        ds=None,
        xinner_ind=None,
        xouter_ind=None,
        ylower_ind=None,
        yupper_ind=None,
        connection_inner_x=None,
        connection_outer_x=None,
        connection_lower_y=None,
        connection_upper_y=None,
    ):
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
        connection_inner_x : str, optional
            The region inside this one in the x-direction
        connection_outer_x : str, optional
            The region outside this one in the x-direction
        connection_lower_y : str, optional
            The region below this one in the y-direction
        connection_upper_y : str, optional
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
        self.connection_inner_x = connection_inner_x
        self.connection_outer_x = connection_outer_x
        self.connection_lower_y = connection_lower_y
        self.connection_upper_y = connection_upper_y
        self.global_nx = ds.sizes[ds.metadata["bout_xdim"]]
        self.global_ny = ds.sizes[ds.metadata["bout_ydim"]]

        if ds is not None:
            # self.nx, self.ny should not include boundary points.
            # self.xinner, self.xouter, self.ylower, self.yupper
            if ds.metadata["keep_xboundaries"]:
                xbndry = ds.metadata["MXG"]
                if self.connection_inner_x is None:
                    self.nx -= xbndry

                    # used to calculate x-coordinate of inner side (self.xinner)
                    xinner_ind += xbndry

                if self.connection_outer_x is None:
                    self.nx -= xbndry

                    # used to calculate x-coordinate of outer side (self.xouter)
                    xouter_ind -= xbndry

            if ds.metadata["keep_yboundaries"]:
                ybndry = ds.metadata["MYG"]
                if self.connection_lower_y is None:
                    self.ny -= ybndry

                    # used to calculate y-coordinate of lower side (self.ylower)
                    ylower_ind += ybndry

                if self.connection_upper_y is None:
                    self.ny -= ybndry

                    # used to calculate y-coordinate of upper side (self.yupper)
                    yupper_ind -= ybndry

            # calculate start and end coordinates
            #####################################
            self.xcoord = ds.metadata["bout_xdim"]
            self.ycoord = ds.metadata["bout_ydim"]

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
            # Define ref_yind so that we avoid using values from the corner cells, which
            # might be NaN.
            if self.connection_lower_y is None and ds.metadata["keep_yboundaries"] == 1:
                ref_yind = ylower_ind + ds.metadata["MYG"]
            else:
                ref_yind = ylower_ind
            dx = ds["dx"].isel({self.ycoord: ref_yind})
            dx_cumsum = dx.cumsum()
            self.xinner = dx_cumsum[xinner_ind] - dx[xinner_ind]
            self.xouter = dx_cumsum[xouter_ind - 1] + dx[xouter_ind - 1]

            # dy is constant in the x-direction, so convert to a 1d array
            # Define ref_xind so that we avoid using values from the corner cells, which
            # might be NaN.
            if self.connection_inner_x is None and ds.metadata["keep_xboundaries"] == 1:
                ref_xind = xinner_ind + ds.metadata["MXG"]
            else:
                ref_xind = xinner_ind
            dy = ds["dy"].isel(**{self.xcoord: ref_xind})
            dy_cumsum = dy.cumsum()
            self.ylower = dy_cumsum[ylower_ind] - dy[ylower_ind]
            self.yupper = dy_cumsum[yupper_ind - 1]

    def __repr__(self):
        result = "<xbout.region.Region>\n"
        for attr, val in vars(self).items():
            result += f"\t{attr}\t{val}\n"
        return result

    def __eq__(self, other):
        return vars(self) == vars(other)

    def get_slices(self, mxg=0, myg=0):
        """
        Return x- and y-dimension slices that select this region from the global
        DataArray.

        Returns
        -------
        xslice, yslice : slice, slice
        """
        xi = self.xinner_ind
        if self.connection_inner_x is not None and xi is not None:
            xi -= mxg

        xo = self.xouter_ind
        if self.connection_outer_x is not None and xo is not None:
            xi += mxg

        yl = self.ylower_ind
        if self.connection_lower_y is not None and yl is not None:
            yl -= myg

        yu = self.yupper_ind
        if self.connection_upper_y is not None and yu is not None:
            yu += myg

        return {self.xcoord: slice(xi, xo), self.ycoord: slice(yl, yu)}

    def get_inner_guards_slices(self, *, mxg, myg=0):
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
        if self.connection_lower_y is not None:
            ylower -= myg
        yupper = self.yupper_ind
        if self.connection_upper_y is not None:
            yupper += myg
        return {
            self.xcoord: slice(self.xinner_ind - mxg, self.xinner_ind),
            self.ycoord: slice(ylower, yupper),
        }

    def get_outer_guards_slices(self, *, mxg, myg=0):
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
        if self.connection_lower_y is not None:
            ylower -= myg
        yupper = self.yupper_ind
        if self.connection_upper_y is not None:
            yupper += myg
        return {
            self.xcoord: slice(self.xouter_ind, self.xouter_ind + mxg),
            self.ycoord: slice(ylower, yupper),
        }

    def get_lower_guards_slices(self, *, myg, mxg=0):
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
        if self.connection_inner_x is not None:
            xinner -= mxg
        xouter = self.xouter_ind
        if self.connection_outer_x is not None:
            xouter += mxg

        ystart = self.ylower_ind - myg
        ystop = self.ylower_ind
        if ystart < 0:
            # Make sure negative indices wrap around correctly
            if ystop == 0:
                ystop = None
            elif ystop > 0:
                raise ValueError(
                    f"'start={ystart}' of lower guards slice is negative, but "
                    f"'stop={ystop}' is positive. Should not be possible."
                )
        return {
            self.xcoord: slice(xinner, xouter),
            self.ycoord: slice(ystart, ystop),
        }

    def get_upper_guards_slices(self, *, myg, mxg=0):
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
        if self.connection_inner_x is not None:
            xinner -= mxg
        xouter = self.xouter_ind
        if self.connection_outer_x is not None:
            xouter += mxg

        # wrap around to the beginning of the grid if necessary
        ystart = self.yupper_ind
        ystop = self.yupper_ind + myg
        if ystart >= self.global_ny:
            # wrap around to beginning of global array
            ystart = ystart - self.global_ny
            ystop = ystop - self.global_ny
        return {
            self.xcoord: slice(xinner, xouter),
            self.ycoord: slice(ystart, ystop),
        }

    def __eq__(self, other):
        if not isinstance(other, Region):
            return NotImplemented
        return vars(self) == vars(other)


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
    jys11 = ds.metadata["jyseps1_1"]
    jys21 = ds.metadata["jyseps2_1"]
    ny_inner = ds.metadata["ny_inner"]
    jys12 = ds.metadata["jyseps1_2"]
    jys22 = ds.metadata["jyseps2_2"]
    ny = ds.metadata["ny"]
    ixs1 = ds.metadata["ixseps1"]
    ixs2 = ds.metadata["ixseps2"]
    nx = ds.metadata["nx"]
    if jys21 == jys12:
        # No upper X-point
        if jys11 <= 0 and jys22 >= ny - 1:
            ix = min(ixs1, ixs2)
            if ix >= nx - 1:
                return "core"
            elif ix <= 0:
                return "sol"
            else:
                return "limiter"

        return "single-null"

    if jys11 == jys21 and jys12 == jys22:
        if jys11 < ny_inner - 1 and jys22 > ny_inner:
            return "xpoint"
        else:
            raise ValueError("Currently unsupported topology")

    if ixs1 == ixs2:
        if jys21 < ny_inner - 1 and jys12 > ny_inner:
            return "connected-double-null"
        else:
            raise ValueError("Currently unsupported topology")
    elif ixs1 < ixs2:
        return "lower-disconnected-double-null"
    else:
        # ixs1 > ixs2
        return "upper-disconnected-double-null"


def _check_connections(regions):
    for region in regions.values():
        if region.connection_inner_x is not None:
            if regions[region.connection_inner_x].connection_outer_x != region.name:
                raise ValueError(
                    f"Inner-x connection of {region.name} is "
                    f"{region.connection_inner_x}, but outer-x connection of "
                    f"{region.connection_inner_x} is "
                    f"{regions[region.connection_inner_x].connection_outer_x}"
                )
        if region.connection_outer_x is not None:
            if regions[region.connection_outer_x].connection_inner_x != region.name:
                raise ValueError(
                    f"Inner-x connection of {region.name} is "
                    f"{region.connection_outer_x}, but inner-x connection of "
                    f"{region.connection_outer_x} is "
                    f"{regions[region.connection_outer_x].connection_inner_x}"
                )
        if region.connection_lower_y is not None:
            if regions[region.connection_lower_y].connection_upper_y != region.name:
                raise ValueError(
                    f"Lower-y connection of {region.name} is "
                    f"{region.connection_lower_y}, but upper-y connection of "
                    f"{region.connection_lower_y} is "
                    f"{regions[region.connection_lower_y].connection_upper_y}"
                )
        if region.connection_upper_y is not None:
            if regions[region.connection_upper_y].connection_lower_y != region.name:
                raise ValueError(
                    f"Upper-y connection of {region.name} is "
                    f"{region.connection_upper_y}, but lower-y connection of "
                    f"{region.connection_upper_y} is "
                    f"{regions[region.connection_upper_y].connection_lower_y}"
                )


topologies = {}


def topology_lower_disconnected_double_null(
    *, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny, ybndry
):
    regions = {}
    regions["lower_inner_PFR"] = Region(
        name="lower_inner_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_outer_x="lower_inner_intersep",
        connection_upper_y="lower_outer_PFR",
    )
    regions["lower_inner_intersep"] = Region(
        name="lower_inner_intersep",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=ixs2,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_inner_x="lower_inner_PFR",
        connection_outer_x="lower_inner_SOL",
        connection_upper_y="inner_intersep",
    )
    regions["lower_inner_SOL"] = Region(
        name="lower_inner_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_inner_x="lower_inner_intersep",
        connection_upper_y="inner_SOL",
    )
    regions["inner_core"] = Region(
        name="inner_core",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys11 + 1,
        yupper_ind=jys21 + 1,
        connection_outer_x="inner_intersep",
        connection_lower_y="outer_core",
        connection_upper_y="outer_core",
    )
    regions["inner_intersep"] = Region(
        name="inner_intersep",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=ixs2,
        ylower_ind=jys11 + 1,
        yupper_ind=jys21 + 1,
        connection_inner_x="inner_core",
        connection_outer_x="inner_SOL",
        connection_lower_y="lower_inner_intersep",
        connection_upper_y="outer_intersep",
    )
    regions["inner_SOL"] = Region(
        name="inner_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=jys11 + 1,
        yupper_ind=jys21 + 1,
        connection_inner_x="inner_intersep",
        connection_lower_y="lower_inner_SOL",
        connection_upper_y="upper_inner_SOL",
    )
    regions["upper_inner_PFR"] = Region(
        name="upper_inner_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys21 + 1,
        yupper_ind=ny_inner,
        connection_outer_x="upper_inner_intersep",
        connection_lower_y="upper_outer_PFR",
    )
    regions["upper_inner_intersep"] = Region(
        name="upper_inner_intersep",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=ixs2,
        ylower_ind=jys21 + 1,
        yupper_ind=ny_inner,
        connection_inner_x="upper_inner_PFR",
        connection_outer_x="upper_inner_SOL",
        connection_lower_y="upper_outer_intersep",
    )
    regions["upper_inner_SOL"] = Region(
        name="upper_inner_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=jys21 + 1,
        yupper_ind=ny_inner,
        connection_inner_x="upper_inner_intersep",
        connection_lower_y="inner_SOL",
    )
    regions["upper_outer_PFR"] = Region(
        name="upper_outer_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=ny_inner,
        yupper_ind=jys12 + 1,
        connection_outer_x="upper_outer_intersep",
        connection_upper_y="upper_inner_PFR",
    )
    regions["upper_outer_intersep"] = Region(
        name="upper_outer_intersep",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=ixs2,
        ylower_ind=ny_inner,
        yupper_ind=jys12 + 1,
        connection_inner_x="upper_outer_PFR",
        connection_outer_x="upper_outer_SOL",
        connection_upper_y="upper_inner_intersep",
    )
    regions["upper_outer_SOL"] = Region(
        name="upper_outer_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=ny_inner,
        yupper_ind=jys12 + 1,
        connection_inner_x="upper_outer_intersep",
        connection_upper_y="outer_SOL",
    )
    regions["outer_core"] = Region(
        name="outer_core",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys12 + 1,
        yupper_ind=jys22 + 1,
        connection_outer_x="outer_intersep",
        connection_lower_y="inner_core",
        connection_upper_y="inner_core",
    )
    regions["outer_intersep"] = Region(
        name="outer_intersep",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=ixs2,
        ylower_ind=jys12 + 1,
        yupper_ind=jys22 + 1,
        connection_inner_x="outer_core",
        connection_outer_x="outer_SOL",
        connection_lower_y="inner_intersep",
        connection_upper_y="lower_outer_intersep",
    )
    regions["outer_SOL"] = Region(
        name="outer_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=jys12 + 1,
        yupper_ind=jys22 + 1,
        connection_inner_x="outer_intersep",
        connection_lower_y="upper_outer_SOL",
        connection_upper_y="lower_outer_SOL",
    )
    regions["lower_outer_PFR"] = Region(
        name="lower_outer_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_outer_x="lower_outer_intersep",
        connection_lower_y="lower_inner_PFR",
    )
    regions["lower_outer_intersep"] = Region(
        name="lower_outer_intersep",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=ixs2,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_inner_x="lower_outer_PFR",
        connection_outer_x="lower_outer_SOL",
        connection_lower_y="outer_intersep",
    )
    regions["lower_outer_SOL"] = Region(
        name="lower_outer_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_inner_x="lower_outer_intersep",
        connection_lower_y="outer_SOL",
    )
    return regions


topologies["lower-disconnected-double-null"] = topology_lower_disconnected_double_null


def topology_upper_disconnected_double_null(
    *, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny, ybndry
):
    regions = {}
    regions["lower_inner_PFR"] = Region(
        name="lower_inner_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs2,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_outer_x="lower_inner_intersep",
        connection_upper_y="lower_outer_PFR",
    )
    regions["lower_inner_intersep"] = Region(
        name="lower_inner_intersep",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=ixs1,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_inner_x="lower_inner_PFR",
        connection_outer_x="lower_inner_SOL",
        connection_upper_y="lower_outer_intersep",
    )
    regions["lower_inner_SOL"] = Region(
        name="lower_inner_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_inner_x="lower_inner_intersep",
        connection_upper_y="inner_SOL",
    )
    regions["inner_core"] = Region(
        name="inner_core",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs2,
        ylower_ind=jys11 + 1,
        yupper_ind=jys21 + 1,
        connection_outer_x="inner_intersep",
        connection_lower_y="outer_core",
        connection_upper_y="outer_core",
    )
    regions["inner_intersep"] = Region(
        name="inner_intersep",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=ixs1,
        ylower_ind=jys11 + 1,
        yupper_ind=jys21 + 1,
        connection_inner_x="inner_core",
        connection_outer_x="inner_SOL",
        connection_lower_y="outer_intersep",
        connection_upper_y="upper_inner_intersep",
    )
    regions["inner_SOL"] = Region(
        name="inner_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=jys11 + 1,
        yupper_ind=jys21 + 1,
        connection_inner_x="inner_intersep",
        connection_lower_y="lower_inner_SOL",
        connection_upper_y="upper_inner_SOL",
    )
    regions["upper_inner_PFR"] = Region(
        name="upper_inner_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs2,
        ylower_ind=jys21 + 1,
        yupper_ind=ny_inner,
        connection_outer_x="upper_inner_intersep",
        connection_lower_y="upper_outer_PFR",
    )
    regions["upper_inner_intersep"] = Region(
        name="upper_inner_intersep",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=ixs1,
        ylower_ind=jys21 + 1,
        yupper_ind=ny_inner,
        connection_inner_x="upper_inner_PFR",
        connection_outer_x="upper_inner_SOL",
        connection_lower_y="inner_intersep",
    )
    regions["upper_inner_SOL"] = Region(
        name="upper_inner_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=jys21 + 1,
        yupper_ind=ny_inner,
        connection_inner_x="upper_inner_intersep",
        connection_lower_y="inner_SOL",
    )
    regions["upper_outer_PFR"] = Region(
        name="upper_outer_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs2,
        ylower_ind=ny_inner,
        yupper_ind=jys12 + 1,
        connection_outer_x="upper_outer_intersep",
        connection_upper_y="upper_inner_PFR",
    )
    regions["upper_outer_intersep"] = Region(
        name="upper_outer_intersep",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=ixs1,
        ylower_ind=ny_inner,
        yupper_ind=jys12 + 1,
        connection_inner_x="upper_outer_PFR",
        connection_outer_x="upper_outer_SOL",
        connection_upper_y="outer_intersep",
    )
    regions["upper_outer_SOL"] = Region(
        name="upper_outer_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=ny_inner,
        yupper_ind=jys12 + 1,
        connection_inner_x="upper_outer_intersep",
        connection_upper_y="outer_SOL",
    )
    regions["outer_core"] = Region(
        name="outer_core",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs2,
        ylower_ind=jys12 + 1,
        yupper_ind=jys22 + 1,
        connection_outer_x="outer_intersep",
        connection_lower_y="inner_core",
        connection_upper_y="inner_core",
    )
    regions["outer_intersep"] = Region(
        name="outer_intersep",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=ixs1,
        ylower_ind=jys12 + 1,
        yupper_ind=jys22 + 1,
        connection_inner_x="outer_core",
        connection_outer_x="outer_SOL",
        connection_lower_y="upper_outer_intersep",
        connection_upper_y="inner_intersep",
    )
    regions["outer_SOL"] = Region(
        name="outer_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=jys12 + 1,
        yupper_ind=jys22 + 1,
        connection_inner_x="outer_intersep",
        connection_lower_y="upper_outer_SOL",
        connection_upper_y="lower_outer_SOL",
    )
    regions["lower_outer_PFR"] = Region(
        name="lower_outer_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs2,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_outer_x="lower_outer_intersep",
        connection_lower_y="lower_inner_PFR",
    )
    regions["lower_outer_intersep"] = Region(
        name="lower_outer_intersep",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=ixs1,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_inner_x="lower_outer_PFR",
        connection_outer_x="lower_outer_SOL",
        connection_lower_y="lower_inner_intersep",
    )
    regions["lower_outer_SOL"] = Region(
        name="lower_outer_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_inner_x="lower_outer_intersep",
        connection_lower_y="outer_SOL",
    )
    return regions


topologies["upper-disconnected-double-null"] = topology_upper_disconnected_double_null


def topology_connected_double_null(
    *, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny, ybndry
):
    regions = {}
    regions["lower_inner_PFR"] = Region(
        name="lower_inner_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_outer_x="lower_inner_SOL",
        connection_upper_y="lower_outer_PFR",
    )
    regions["lower_inner_SOL"] = Region(
        name="lower_inner_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_inner_x="lower_inner_PFR",
        connection_upper_y="inner_SOL",
    )
    regions["inner_core"] = Region(
        name="inner_core",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys11 + 1,
        yupper_ind=jys21 + 1,
        connection_outer_x="inner_SOL",
        connection_lower_y="outer_core",
        connection_upper_y="outer_core",
    )
    regions["inner_SOL"] = Region(
        name="inner_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=jys11 + 1,
        yupper_ind=jys21 + 1,
        connection_inner_x="inner_core",
        connection_lower_y="lower_inner_SOL",
        connection_upper_y="upper_inner_SOL",
    )
    regions["upper_inner_PFR"] = Region(
        name="upper_inner_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys21 + 1,
        yupper_ind=ny_inner,
        connection_outer_x="upper_inner_SOL",
        connection_lower_y="upper_outer_PFR",
    )
    regions["upper_inner_SOL"] = Region(
        name="upper_inner_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=jys21 + 1,
        yupper_ind=ny_inner,
        connection_inner_x="upper_inner_PFR",
        connection_lower_y="inner_SOL",
    )
    regions["upper_outer_PFR"] = Region(
        name="upper_outer_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=ny_inner,
        yupper_ind=jys12 + 1,
        connection_outer_x="upper_outer_SOL",
        connection_upper_y="upper_inner_PFR",
    )
    regions["upper_outer_SOL"] = Region(
        name="upper_outer_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=ny_inner,
        yupper_ind=jys12 + 1,
        connection_inner_x="upper_outer_PFR",
        connection_upper_y="outer_SOL",
    )
    regions["outer_core"] = Region(
        name="outer_core",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys12 + 1,
        yupper_ind=jys22 + 1,
        connection_outer_x="outer_SOL",
        connection_lower_y="inner_core",
        connection_upper_y="inner_core",
    )
    regions["outer_SOL"] = Region(
        name="outer_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=jys12 + 1,
        yupper_ind=jys22 + 1,
        connection_inner_x="outer_core",
        connection_lower_y="upper_outer_SOL",
        connection_upper_y="lower_outer_SOL",
    )
    regions["lower_outer_PFR"] = Region(
        name="lower_outer_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_outer_x="lower_outer_SOL",
        connection_lower_y="lower_inner_PFR",
    )
    regions["lower_outer_SOL"] = Region(
        name="lower_outer_SOL",
        ds=ds,
        xinner_ind=ixs2,
        xouter_ind=nx,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_inner_x="lower_outer_PFR",
        connection_lower_y="outer_SOL",
    )
    return regions


topologies["connected-double-null"] = topology_connected_double_null


def topology_single_null(
    *, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny, ybndry
):
    regions = {}
    regions["inner_PFR"] = Region(
        name="inner_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_outer_x="inner_SOL",
        connection_upper_y="outer_PFR",
    )
    regions["inner_SOL"] = Region(
        name="inner_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_inner_x="inner_PFR",
        connection_upper_y="SOL",
    )
    regions["core"] = Region(
        name="core",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys11 + 1,
        yupper_ind=jys22 + 1,
        connection_outer_x="SOL",
        connection_lower_y="core",
        connection_upper_y="core",
    )
    regions["SOL"] = Region(
        name="SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=jys11 + 1,
        yupper_ind=jys22 + 1,
        connection_inner_x="core",
        connection_lower_y="inner_SOL",
        connection_upper_y="outer_SOL",
    )
    regions["outer_PFR"] = Region(
        name="outer_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_outer_x="outer_SOL",
        connection_lower_y="inner_PFR",
    )
    regions["outer_SOL"] = Region(
        name="outer_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_inner_x="outer_PFR",
        connection_lower_y="SOL",
    )
    return regions


topologies["single-null"] = topology_single_null


def topology_limiter(
    *, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny, ybndry
):
    regions = {}
    regions["core"] = Region(
        name="core",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=ybndry,
        yupper_ind=ny - ybndry,
        connection_outer_x="SOL",
        connection_lower_y="core",
        connection_upper_y="core",
    )
    regions["SOL"] = Region(
        name="SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=0,
        yupper_ind=ny,
        connection_inner_x="core",
    )
    return regions


topologies["limiter"] = topology_limiter


def topology_core(
    *, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny, ybndry
):
    regions = {}
    regions["core"] = Region(
        name="core",
        ds=ds,
        xinner_ind=0,
        xouter_ind=nx,
        ylower_ind=ybndry,
        yupper_ind=ny - ybndry,
        connection_lower_y="core",
        connection_upper_y="core",
    )
    return regions


topologies["core"] = topology_core


def topology_sol(
    *, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny, ybndry
):
    regions = {}
    regions["SOL"] = Region(
        name="SOL", ds=ds, xinner_ind=0, xouter_ind=nx, ylower_ind=0, yupper_ind=ny
    )
    return regions


topologies["sol"] = topology_sol


def topology_xpoint(
    *, ds, ixs1, ixs2, nx, jys11, jys21, ny_inner, jys12, jys22, ny, ybndry
):
    regions = {}
    regions["lower_inner_PFR"] = Region(
        name="lower_inner_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_outer_x="lower_inner_SOL",
        connection_upper_y="lower_outer_PFR",
    )
    regions["lower_inner_SOL"] = Region(
        name="lower_inner_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=0,
        yupper_ind=jys11 + 1,
        connection_inner_x="lower_inner_PFR",
        connection_upper_y="upper_inner_SOL",
    )
    regions["upper_inner_PFR"] = Region(
        name="upper_inner_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys11 + 1,
        yupper_ind=ny_inner,
        connection_outer_x="upper_inner_SOL",
        connection_lower_y="upper_outer_PFR",
    )
    regions["upper_inner_SOL"] = Region(
        name="upper_inner_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=jys11 + 1,
        yupper_ind=ny_inner,
        connection_inner_x="upper_inner_PFR",
        connection_lower_y="lower_inner_SOL",
    )
    regions["upper_outer_PFR"] = Region(
        name="upper_outer_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=ny_inner,
        yupper_ind=jys22 + 1,
        connection_outer_x="upper_outer_SOL",
        connection_upper_y="upper_inner_PFR",
    )
    regions["upper_outer_SOL"] = Region(
        name="upper_outer_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=ny_inner,
        yupper_ind=jys22 + 1,
        connection_inner_x="upper_outer_PFR",
        connection_upper_y="lower_outer_SOL",
    )
    regions["lower_outer_PFR"] = Region(
        name="lower_outer_PFR",
        ds=ds,
        xinner_ind=0,
        xouter_ind=ixs1,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_outer_x="lower_outer_SOL",
        connection_lower_y="lower_inner_PFR",
    )
    regions["lower_outer_SOL"] = Region(
        name="lower_outer_SOL",
        ds=ds,
        xinner_ind=ixs1,
        xouter_ind=nx,
        ylower_ind=jys22 + 1,
        yupper_ind=ny,
        connection_inner_x="lower_outer_PFR",
        connection_lower_y="upper_outer_SOL",
    )
    return regions


topologies["xpoint"] = topology_xpoint


def _create_regions_toroidal(ds):
    topology = _get_topology(ds)

    ixs1 = ds.metadata["ixseps1"]
    ixs2 = ds.metadata["ixseps2"]
    nx = ds.metadata["nx"]

    jys11 = ds.metadata["jyseps1_1"]
    jys21 = ds.metadata["jyseps2_1"]
    ny_inner = ds.metadata["ny_inner"]
    jys12 = ds.metadata["jyseps1_2"]
    jys22 = ds.metadata["jyseps2_2"]
    ny = ds.metadata["ny"]

    mxg = ds.metadata["MXG"]
    myg = ds.metadata["MYG"]
    # keep_yboundaries is 1 if there are y-boundaries and 0 if there are not
    ybndry = ds.metadata["keep_yboundaries"] * myg
    if jys21 == jys12:
        # No upper targets
        ybndry_upper = 0
    else:
        ybndry_upper = ybndry

    # Make sure all sizes are sensible
    ixs1 = _in_range(ixs1, 0, nx)
    ixs2 = _in_range(ixs2, 0, nx)
    jys11 = _in_range(jys11, 0, ny - 1)
    jys21 = _in_range(jys21, 0, ny - 1)
    jys12 = _in_range(jys12, 0, ny - 1)
    jys21, jys12 = _order_vars(jys21, jys12)
    ny_inner = _in_range(ny_inner, jys21 + 1, jys12 + 1)
    jys22 = _in_range(jys22, 0, ny - 1)

    # Adjust for boundary cells
    # keep_xboundaries is 1 if there are x-boundaries and 0 if there are not
    if not ds.metadata["keep_xboundaries"]:
        ixs1 -= mxg
        ixs2 -= mxg
        nx -= 2 * mxg
    jys11 += ybndry
    jys21 += ybndry
    ny_inner += ybndry + ybndry_upper
    jys12 += ybndry + 2 * ybndry_upper
    jys22 += ybndry + 2 * ybndry_upper
    ny += 2 * ybndry + 2 * ybndry_upper

    # Note, include guard cells in the created regions, fill them later
    if topology in topologies:
        regions = topologies[topology](
            ds=ds,
            ixs1=ixs1,
            ixs2=ixs2,
            nx=nx,
            jys11=jys11,
            jys21=jys21,
            ny_inner=ny_inner,
            jys12=jys12,
            jys22=jys22,
            ny=ny,
            ybndry=ybndry,
        )
    else:
        raise NotImplementedError(f"Topology '{topology}' is not implemented")

    _check_connections(regions)

    ds = _set_attrs_on_all_vars(ds, "regions", regions)

    return ds


def _create_single_region(ds, periodic_y=True):
    nx = ds.metadata["nx"]
    ny = ds.metadata["ny"]

    mxg = ds.metadata["MXG"]
    myg = ds.metadata["MYG"]
    # keep_yboundaries is 1 if there are y-boundaries and 0 if there are not
    ybndry = ds.metadata["keep_yboundaries"] * myg

    connection = "all" if periodic_y else None

    regions = {
        "all": Region(
            name="all",
            ds=ds,
            xouter_ind=nx,
            xinner_ind=0,
            ylower_ind=0,
            yupper_ind=ny,
            connection_lower_y=connection,
            connection_upper_y=connection,
        )
    }

    _check_connections(regions)

    ds = _set_attrs_on_all_vars(ds, "regions", regions)

    return ds


def _concat_inner_guards(da, da_global, mxg):
    """
    Concatenate inner x-guard cells to da, which is in a single region, getting the guard
    cells from the global array
    """

    if mxg <= 0:
        return da

    if len(da.bout._regions) > 1:
        raise ValueError("da passed should have only one region")
    region = list(da.bout._regions.values())[0]

    if region.connection_inner_x not in da_global.bout._regions:
        # No connection, or plotting restricted set of regions not including this
        # connection
        return da

    myg_da = da_global.metadata["MYG"]
    keep_yboundaries = da_global.metadata["keep_yboundaries"]
    xcoord = da_global.metadata["bout_xdim"]
    ycoord = da_global.metadata["bout_ydim"]

    da_inner = da_global.bout.from_region(region.connection_inner_x, with_guards=0)
    if len(da_inner.bout._regions) > 1:
        raise ValueError("da_inner should have only one region")
    region_inner = list(da_inner.bout._regions.values())[0]

    if (
        myg_da > 0
        and keep_yboundaries
        and region.connection_lower_y is not None
        and region_inner.connection_lower_y is None
    ):
        # da_inner may have more points in the y-direction, because it has an actual
        # boundary, not a connection. Need to remove any extra points
        da_inner = da_inner.isel(**{ycoord: slice(myg_da, None)})
    if (
        myg_da > 0
        and keep_yboundaries
        and region.connection_upper_y is not None
        and region_inner.connection_upper_y is None
    ):
        # da_inner may have more points in the y-direction, because it has an actual
        # boundary, not a connection. Need to remove any extra points
        da_inner = da_inner.isel(**{ycoord: slice(None, -myg_da)})

    # select just the points we need to fill the guard cells of da
    da_inner = da_inner.isel(**{xcoord: slice(-mxg, None)})

    if (
        myg_da > 0
        and keep_yboundaries
        and region.connection_lower_y is None
        and region_inner.connection_lower_y is not None
    ):
        # da_inner may have fewer points in the y-direction, because it has a connection,
        # not an actual boundary. Need to get the extra points from its connection
        da_inner_lower = da_global.bout.from_region(
            region_inner.connection_lower_y, with_guards=0
        )
        da_inner_lower = da_inner_lower.isel(
            **{xcoord: slice(-mxg, None), ycoord: slice(-myg_da, None)}
        )
        save_regions = da_inner.bout._regions
        da_inner = xr.concat((da_inner_lower, da_inner), ycoord, join="exact")
        # xr.concat takes attributes from the first variable, but we need da_inner's
        # regions
        da_inner.attrs["regions"] = save_regions
    if (
        myg_da > 0
        and keep_yboundaries
        and region.connection_upper_y is None
        and region_inner.connection_upper_y is not None
    ):
        # da_inner may have fewer points in the y-direction, because it has a connection,
        # not an actual boundary. Need to get the extra points from its connection
        da_inner_upper = da_global.bout.from_region(
            region_inner.connection_upper_y, with_guards=0
        )
        da_inner_upper = da_inner_upper.isel(
            **{xcoord: slice(-mxg, None), ycoord: slice(myg_da)}
        )
        da_inner = xr.concat((da_inner, da_inner_upper), ycoord, join="exact")
        # xr.concat takes attributes from the first variable, so regions are OK

    if xcoord in da.coords:
        # Some coordinates may not be single-valued, so use local coordinates for
        # neighbouring region, not communicated ones. Ensures the coordinates are
        # continuous so that interpolation works correctly near the boundaries.
        slices = region.get_inner_guards_slices(mxg=mxg)
        new_xcoord = da_global[xcoord].isel(**{xcoord: slices[xcoord]})
        new_ycoord = da_global[ycoord].isel(**{ycoord: slices[ycoord]})

        # can't use commented out version, uncommented one works around xarray bug
        # removing attrs
        # https://github.com/pydata/xarray/issues/4415
        # https://github.com/pydata/xarray/issues/4393
        # da_inner = da_inner.assign_coords(**{xcoord: new_xcoord, ycoord: new_ycoord})
        da_inner[xcoord].data[...] = new_xcoord.data
        da_inner[ycoord].data[...] = new_ycoord.data

    save_regions = da.bout._regions
    da = xr.concat((da_inner, da), xcoord, join="exact")
    # xr.concat takes attributes from the first variable (for xarray>=0.15.0, keeps attrs
    # that are the same in all objects for xarray<0.15.0)
    da.attrs["regions"] = save_regions

    return da


def _concat_outer_guards(da, da_global, mxg):
    """
    Concatenate outer x-guard cells to da, which is in a single region, getting the guard
    cells from the global array
    """

    if mxg <= 0:
        return da

    if len(da.bout._regions) > 1:
        raise ValueError("da passed should have only one region")
    region = list(da.bout._regions.values())[0]

    if region.connection_outer_x not in da_global.bout._regions:
        # No connection, or plotting restricted set of regions not including this
        # connection
        return da

    myg_da = da_global.metadata["MYG"]
    keep_yboundaries = da_global.metadata["keep_yboundaries"]
    xcoord = da_global.metadata["bout_xdim"]
    ycoord = da_global.metadata["bout_ydim"]

    da_outer = da_global.bout.from_region(region.connection_outer_x, with_guards=0)
    if len(da_outer.bout._regions) > 1:
        raise ValueError("da_outer should have only one region")
    region_outer = list(da_outer.bout._regions.values())[0]

    if (
        myg_da > 0
        and keep_yboundaries
        and region.connection_lower_y is not None
        and region_outer.connection_lower_y is None
    ):
        # da_outer may have more points in the y-direction, because it has an actual
        # boundary, not a connection. Need to remove any extra points
        da_outer = da_outer.isel(**{ycoord: slice(myg_da, None)})
    if (
        myg_da > 0
        and keep_yboundaries
        and region.connection_upper_y is not None
        and region_outer.connection_upper_y is None
    ):
        # da_outer may have more points in the y-direction, because it has an actual
        # boundary, not a connection. Need to remove any extra points
        da_outer = da_outer.isel(**{ycoord: slice(None, -myg_da)})

    # select just the points we need to fill the guard cells of da
    da_outer = da_outer.isel(**{xcoord: slice(mxg)})

    if (
        myg_da > 0
        and keep_yboundaries
        and region.connection_lower_y is None
        and region_outer.connection_lower_y is not None
    ):
        # da_outer may have fewer points in the y-direction, because it has a connection,
        # not an actual boundary. Need to get the extra points from its connection
        da_outer_lower = da_global.bout.from_region(
            region_outer.connection_lower_y, with_guards=0
        )
        da_outer_lower = da_outer_lower.isel(
            **{xcoord: slice(-mxg, None), ycoord: slice(-myg_da, None)}
        )
        save_regions = da_outer.bout._regions
        da_outer = xr.concat((da_outer_lower, da_outer), ycoord, join="exact")
        # xr.concat takes attributes from the first variable, but we need da_outer's
        # regions
        da_outer.attrs["regions"] = save_regions
    if (
        myg_da > 0
        and keep_yboundaries
        and region.connection_upper_y is None
        and region_outer.connection_upper_y is not None
    ):
        # da_outer may have fewer points in the y-direction, because it has a connection,
        # not an actual boundary. Need to get the extra points from its connection
        da_outer_upper = da_global.bout.from_region(
            region_outer.connection_upper_y, with_guards=0
        )
        da_outer_upper = da_outer_upper.isel(
            **{xcoord: slice(-mxg, None), ycoord: slice(myg_da)}
        )
        da_outer = xr.concat((da_outer, da_outer_upper), ycoord, join="exact")
        # xr.concat takes attributes from the first variable, so regions are OK

    if xcoord in da.coords:
        # Some coordinates may not be single-valued, so use local coordinates for
        # neighbouring region, not communicated ones. Ensures the coordinates are
        # continuous so that interpolation works correctly near the boundaries.
        slices = region.get_outer_guards_slices(mxg=mxg)
        new_xcoord = da_global[xcoord].isel(**{xcoord: slices[xcoord]})
        new_ycoord = da_global[ycoord].isel(**{ycoord: slices[ycoord]})

        # can't use commented out version, uncommented one works around xarray bug
        # removing attrs
        # https://github.com/pydata/xarray/issues/4415
        # https://github.com/pydata/xarray/issues/4393
        # da_outer = da_outer.assign_coords(**{xcoord: new_xcoord, ycoord: new_ycoord})
        da_outer[xcoord].data[...] = new_xcoord.data
        da_outer[ycoord].data[...] = new_ycoord.data

    save_regions = da.bout._regions
    da = xr.concat((da, da_outer), xcoord, join="exact")
    # xarray<0.15.0 only keeps attrs that are the same on all variables passed to concat
    da.attrs["regions"] = save_regions

    return da


def _concat_lower_guards(da, da_global, mxg, myg):
    """
    Concatenate lower y-guard cells to da, which is in a single region, getting the guard
    cells from the global array
    """

    if myg <= 0:
        return da

    if len(da.bout._regions) > 1:
        raise ValueError("da passed should have only one region")
    region = list(da.bout._regions.values())[0]

    if region.connection_lower_y not in da_global.bout._regions:
        # No connection, or plotting restricted set of regions not including this
        # connection
        return da

    xcoord = da_global.metadata["bout_xdim"]
    ycoord = da_global.metadata["bout_ydim"]

    da_lower = da_global.bout.from_region(
        region.connection_lower_y, with_guards={xcoord: mxg, ycoord: 0}
    )

    # select just the points we need to fill the guard cells of da
    da_lower = da_lower.isel(**{ycoord: slice(-myg, None)})

    if ycoord in da.coords:
        # Some coordinates may not be single-valued, so use local coordinates for
        # neighbouring region, not communicated ones. Ensures the coordinates are
        # continuous so that interpolation works correctly near the boundaries.
        slices = region.get_lower_guards_slices(mxg=mxg, myg=myg)

        new_xcoord = da_global[xcoord].isel(**{xcoord: slices[xcoord]})
        new_ycoord = da_global[ycoord].isel(**{ycoord: slices[ycoord]})

        # new_ycoord might not join smoothly onto da[ycoord] for limiter or core-only
        # geometries where the guard cells have to come from the opposite side of a
        # branch cut. Need to work around this, or it breaks parallel interpolation.
        ycoord_increasing = np.any(
            da[ycoord].isel(**{ycoord: 1}) > da[ycoord].isel(**{ycoord: 0})
        )

        if ycoord_increasing and np.any(
            new_ycoord.isel(**{ycoord: -1}) > da[ycoord].isel(**{ycoord: 0})
        ):
            if not np.all(
                new_ycoord.isel(**{ycoord: -1}) > da[ycoord].isel(**{ycoord: 0})
            ):
                raise ValueError(
                    f"Inconsistent {ycoord} - should be either increasing everywhere "
                    f"or decreasing everywhere"
                )
            # Estimate what coordinate values should be by extrapolating linearly from
            # interior of region. This is exact if the grid spacing is constant, which
            # it usually is in 'y'
            offset = (
                2.0 * da[ycoord].isel(**{ycoord: 0})
                - da[ycoord].isel(**{ycoord: 1})
                - new_ycoord.isel(**{ycoord: -1})
            )
            new_ycoord = new_ycoord + offset

        elif (not ycoord_increasing) and np.any(
            new_ycoord.isel(**{ycoord: -1}) < da[ycoord].isel(**{ycoord: 0})
        ):
            if not np.all(
                new_ycoord.isel(**{ycoord: -1}) < da[ycoord].isel(**{ycoord: 0})
            ):
                raise ValueError(
                    f"Inconsistent {ycoord} - should be either increasing everywhere "
                    f"or decreasing everywhere"
                )
            # Estimate what coordinate values should be by extrapolating linearly from
            # interior of region. This is exact if the grid spacing is constant, which
            # it usually is in 'y'
            offset = (
                2.0 * da[ycoord].isel(**{ycoord: 0})
                - da[ycoord].isel(**{ycoord: 1})
                - new_ycoord.isel(**{ycoord: -1})
            )
            new_ycoord = new_ycoord + offset

        # can't use commented out version, uncommented one works around xarray bug
        # removing attrs
        # https://github.com/pydata/xarray/issues/4415
        # https://github.com/pydata/xarray/issues/4393
        # da_lower = da_lower.assign_coords(**{xcoord: new_xcoord, ycoord: new_ycoord})
        da_lower[xcoord].data[...] = new_xcoord.data
        da_lower[ycoord].data[...] = new_ycoord.data

    if "poloidal_distance" in da.coords and myg > 0:
        # Special handling for core regions to deal with branch cut
        if "core" in region.name:
            # import pdb; pdb.set_trace()
            # Try to detect whether there is branch cut at lower boundary: if there is
            # poloidal_distance_ylow should be zero at the boundary of this region
            poloidal_distance_bottom = da["poloidal_distance_ylow"].isel({ycoord: 0})
            if all(abs(poloidal_distance_bottom) < 1.0e-16):
                # Offset so that the poloidal_distance in da_lower is continuous from
                # the poloidal_distance in this region.
                # Expect there to be y-boundary cells in the Dataset, this will probably
                # fail if there are not.
                total_poloidal_distance = da["total_poloidal_distance"]
                da_lower["poloidal_distance"] -= total_poloidal_distance
                da_lower["poloidal_distance_ylow"] -= total_poloidal_distance

    save_regions = da.bout._regions
    da = xr.concat((da_lower, da), ycoord, join="exact")
    # xr.concat takes attributes from the first variable (for xarray>=0.15.0, keeps attrs
    # that are the same in all objects for xarray<0.15.0)
    da.attrs["regions"] = save_regions

    return da


def _concat_upper_guards(da, da_global, mxg, myg):
    """
    Concatenate upper y-guard cells to da, which is in a single region, getting the guard
    cells from the global array
    """

    if myg <= 0:
        return da

    if len(da.bout._regions) > 1:
        raise ValueError("da passed should have only one region")
    region = list(da.bout._regions.values())[0]

    if region.connection_upper_y not in da_global.bout._regions:
        # No connection, or plotting restricted set of regions not including this
        # connection
        return da

    xcoord = da_global.metadata["bout_xdim"]
    ycoord = da_global.metadata["bout_ydim"]

    da_upper = da_global.bout.from_region(
        region.connection_upper_y, with_guards={xcoord: mxg, ycoord: 0}
    )
    # select just the points we need to fill the guard cells of da
    da_upper = da_upper.isel(**{ycoord: slice(myg)})

    if ycoord in da.coords:
        # Some coordinates may not be single-valued, so use local coordinates for
        # neighbouring region, not communicated ones. Ensures the coordinates are
        # continuous so that interpolation works correctly near the boundaries.
        slices = region.get_upper_guards_slices(mxg=mxg, myg=myg)

        new_xcoord = da_global[xcoord].isel(**{xcoord: slices[xcoord]})
        new_ycoord = da_global[ycoord].isel(**{ycoord: slices[ycoord]})

        # new_ycoord might not join smoothly onto da[ycoord] for limiter or core-only
        # geometries where the guard cells have to come from the opposite side of a
        # branch cut. Need to work around this, or it breaks parallel interpolation.
        ycoord_increasing = np.any(
            da[ycoord].isel(**{ycoord: -1}) > da[ycoord].isel(**{ycoord: -2})
        )

        if ycoord_increasing and np.any(
            new_ycoord.isel(**{ycoord: 0}) < da[ycoord].isel(**{ycoord: -1})
        ):
            if not np.all(
                new_ycoord.isel(**{ycoord: 0}) < da[ycoord].isel(**{ycoord: -1})
            ):
                raise ValueError(
                    f"Inconsistent {ycoord} - should be either increasing everywhere "
                    f"or decreasing everywhere"
                )
            # Estimate what coordinate values should be by extrapolating linearly from
            # interior of region. This is exact if the grid spacing is constant, which
            # it usually is in 'y'
            offset = (
                2.0 * da[ycoord].isel(**{ycoord: -1})
                - da[ycoord].isel(**{ycoord: -2})
                - new_ycoord.isel(**{ycoord: 0})
            )
            new_ycoord = new_ycoord + offset

        elif (not ycoord_increasing) and np.any(
            new_ycoord.isel(**{ycoord: 0}) > da[ycoord].isel(**{ycoord: -1})
        ):
            if not np.all(
                new_ycoord.isel(**{ycoord: 0}) > da[ycoord].isel(**{ycoord: -1})
            ):
                raise ValueError(
                    f"Inconsistent {ycoord} - should be either increasing everywhere "
                    f"or decreasing everywhere"
                )
            # Estimate what coordinate values should be by extrapolating linearly from
            # interior of region. This is exact if the grid spacing is constant, which
            # it usually is in 'y'
            offset = (
                2.0 * da[ycoord].isel(**{ycoord: -1})
                - da[ycoord].isel(**{ycoord: -2})
                - new_ycoord.isel(**{ycoord: 0})
            )
            new_ycoord = new_ycoord + offset

        # can't use commented out version, uncommented one works around xarray bug
        # removing attrs
        # https://github.com/pydata/xarray/issues/4415
        # https://github.com/pydata/xarray/issues/4393
        # da_upper = da_upper.assign_coords(**{xcoord: new_xcoord, ycoord: new_ycoord})
        da_upper[xcoord].data[...] = new_xcoord.data
        da_upper[ycoord].data[...] = new_ycoord.data

    if "poloidal_distance" in da.coords and myg > 0:
        # Special handling for core regions to deal with branch cut
        if "core" in region.name:
            # import pdb; pdb.set_trace()
            # Try to detect whether there is branch cut at upper boundary: if there is
            # poloidal_distance_ylow should be zero at the boundary of da_upper
            poloidal_distance_bottom = da_upper["poloidal_distance_ylow"].isel(
                {ycoord: 0}
            )
            if all(abs(poloidal_distance_bottom) < 1.0e-16):
                # Offset so that the poloidal_distance in da_upper is continuous from
                # the poloidal_distance in this region.
                # Expect there to be y-boundary cells in the Dataset, this will probably
                # fail if there are not.
                total_poloidal_distance = da["total_poloidal_distance"]
                da_upper["poloidal_distance"] += total_poloidal_distance
                da_upper["poloidal_distance_ylow"] += total_poloidal_distance

    save_regions = da.bout._regions
    da = xr.concat((da, da_upper), ycoord, join="exact")
    # xarray<0.15.0 only keeps attrs that are the same on all variables passed to concat
    da.attrs["regions"] = save_regions

    return da


def _from_region(ds_or_da, name, with_guards):
    # ensure we do not modify the input
    ds_or_da = ds_or_da.copy(deep=True)

    region = ds_or_da.bout._regions[name]
    xcoord = ds_or_da.metadata["bout_xdim"]
    ycoord = ds_or_da.metadata["bout_ydim"]

    if with_guards is None:
        mxg = ds_or_da.metadata["MXG"]
        myg = ds_or_da.metadata["MYG"]
    else:
        try:
            mxg = with_guards.get(xcoord, ds_or_da.metadata["MXG"])
            myg = with_guards.get(ycoord, ds_or_da.metadata["MYG"])
        except AttributeError:
            mxg = with_guards
            myg = with_guards

    result = ds_or_da.isel(region.get_slices()).copy()

    # The returned result has only one region
    single_region = deepcopy(region)
    result.attrs["regions"] = {name: single_region}

    # get inner x-guard cells for result from the global array
    result = _concat_inner_guards(result, ds_or_da, mxg)
    # get outer x-guard cells for result from the global array
    result = _concat_outer_guards(result, ds_or_da, mxg)
    # get lower y-guard cells from the global array
    result = _concat_lower_guards(result, ds_or_da, mxg, myg)
    # get upper y-guard cells from the global array
    result = _concat_upper_guards(result, ds_or_da, mxg, myg)

    # If the result (which only has a single region) is passed to from_region a
    # second time, don't want to slice anything.
    single_region = list(result.bout._regions.values())[0]
    single_region.xinner_ind = None
    single_region.xouter_ind = None
    single_region.ylower_ind = None
    single_region.yupper_ind = None

    return result
