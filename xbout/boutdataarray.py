from copy import deepcopy
from pprint import pformat as prettyformat
from functools import partial

import dask.array
import numpy as np

import xarray as xr
from xarray import register_dataarray_accessor

from .plotting.animate import animate_poloidal, animate_pcolormesh, animate_line
from .plotting import plotfuncs
from .plotting.utils import _create_norm
from .region import (Region, _concat_inner_guards, _concat_outer_guards,
                     _concat_lower_guards, _concat_upper_guards)
from .utils import _update_metadata_increased_resolution


@register_dataarray_accessor('bout')
class BoutDataArrayAccessor:
    """
    Contains BOUT-specific methods to use on BOUT++ dataarrays opened by
    selecting a variable from a BOUT++ dataset.

    These BOUT-specific methods and attributes are accessed via the bout
    accessor, e.g. `da.bout.options` returns a `BoutOptionsFile` instance.
    """

    def __init__(self, da):
        self.data = da
        self.metadata = da.attrs.get('metadata')  # None if just grid file
        self.options = da.attrs.get('options')  # None if no inp file

    def __str__(self):
        """
        String representation of the BoutDataArray.

        Accessed by print(da.bout)
        """

        styled = partial(prettyformat, indent=4, compact=True)
        text = "<xbout.BoutDataset>\n" + \
               "Contains:\n{}\n".format(str(self.data)) + \
               "Metadata:\n{}\n".format(styled(self.metadata))
        if self.options:
            text += "Options:\n{}".format(styled(self.options))
        return text

    def to_dataset(self):
        """
        Convert a DataArray to a Dataset, copying the attributes from the DataArray to
        the Dataset, and dropping attributes that only make sense for a DataArray
        """
        da = self.data
        ds = da.to_dataset()

        ds.attrs = deepcopy(da.attrs)

        def dropIfExists(ds, name):
            if name in ds.attrs:
                del ds.attrs[name]

        dropIfExists(ds, 'direction_y')
        dropIfExists(ds, 'direction_z')

        return ds

    def _shiftZ(self, zShift):
        """
        Shift a DataArray in the periodic, toroidal direction using FFTs.

        Parameters
        ----------
        zShift : DataArray
            The angle to shift by
        """
        # Would be nice to use the xrft package for this, but xrft does not currently
        # implement inverse Fourier transforms (although there is an open PR
        # https://github.com/xgcm/xrft/pull/81 to add this).

        # Use dask.array.fft if self.data.data is a dask array
        if isinstance(self.data.data, dask.array.Array):
            fft = dask.array.fft
        else:
            fft = np.fft

        nz = self.data.metadata['nz']
        zlength = nz*self.data.metadata['dz']
        nmodes = nz // 2 + 1

        # Get axis position of dimension to transform
        axis = self.data.dims.index(self.data.metadata['bout_zdim'])

        # Create list the dimensions of FFT'd array
        fft_dims = list(deepcopy(self.data.dims))
        fft_dims[axis] = 'kz'

        # Fourier transform to get the DataArray in k-space
        data_fft = fft.rfft(self.data.data, axis=axis)

        # Complex phase for rotation by angle zShift
        kz = 2.*np.pi*xr.DataArray(np.arange(0, nmodes), dims='kz')/zlength
        phase = 1.j*kz*zShift

        # Ensure dimensions are in correct order for numpy broadcasting
        extra_dims = deepcopy(fft_dims)
        for dim in phase.dims:
            extra_dims.remove(dim)
        phase = phase.expand_dims(extra_dims)
        phase = phase.transpose(*fft_dims, transpose_coords=True)

        data_shifted_fft = data_fft * np.exp(phase.data)

        data_shifted = fft.irfft(data_shifted_fft, n=nz)

        # Return a DataArray with the same attributes as self, but values from
        # data_shifted
        return self.data.copy(data=data_shifted)

    def toFieldAligned(self):
        """
        Transform DataArray to field-aligned coordinates, which are shifted with respect
        to the base coordinates by an angle zShift
        """
        if self.data.direction_y != "Standard":
            raise ValueError("Cannot shift a " + self.direction_y + " type field to "
                             + "field-aligned coordinates")
        result = self._shiftZ(self.data['zShift'])
        result["direction_y"] = "Aligned"
        return result

    def fromFieldAligned(self):
        """
        Transform DataArray from field-aligned coordinates, which are shifted with
        respect to the base coordinates by an angle zShift
        """
        if self.data.direction_y != "Aligned":
            raise ValueError("Cannot shift a " + self.direction_y + " type field to "
                             + "field-aligned coordinates")
        result = self._shiftZ(-self.data['zShift'])
        result["direction_y"] = "Standard"
        return result

    def from_region(self, name, with_guards=None):
        """
        Get a logically-rectangular section of data from a certain region.
        Includes guard cells from neighbouring regions.

        Parameters
        ----------
        name : str
            Region to get data for
        with_guards : int or dict of int, optional
            Number of guard cells to include, by default use MXG and MYG from BOUT++.
            Pass a dict to set different numbers for different coordinates.
        """

        region = self.data.regions[name]
        xcoord = self.data.metadata['bout_xdim']
        ycoord = self.data.metadata['bout_ydim']

        if with_guards is None:
            mxg = self.data.metadata['MXG']
            myg = self.data.metadata['MYG']
        else:
            try:
                mxg = with_guards.get(xcoord, self.data.metadata['MXG'])
                myg = with_guards.get(ycoord, self.data.metadata['MYG'])
            except AttributeError:
                mxg = with_guards
                myg = with_guards

        da = self.data.isel(region.get_slices())
        da.attrs['region'] = region

        if region.connection_inner_x is not None:
            # get inner x-guard cells for da from the global array
            da = _concat_inner_guards(da, self.data, mxg)
        if region.connection_outer_x is not None:
            # get outer x-guard cells for da from the global array
            da = _concat_outer_guards(da, self.data, mxg)
        if region.connection_lower_y is not None:
            # get lower y-guard cells from the global array
            da = _concat_lower_guards(da, self.data, mxg, myg)
        if region.connection_upper_y is not None:
            # get upper y-guard cells from the global array
            da = _concat_upper_guards(da, self.data, mxg, myg)

        return da

    @property
    def fine_interpolation_factor(self):
        """
        The default factor to increase resolution when doing parallel interpolation
        """
        return self.data.metadata['fine_interpolation_factor']

    @fine_interpolation_factor.setter
    def fine_interpolation_factor(self, n):
        """
        Set the default factor to increase resolution when doing parallel interpolation.

        Parameters
        -----------
        n : int
            Factor to increase parallel resolution by
        """
        self.data.metadata['fine_interpolation_factor'] = n

    def interpolate_parallel(self, region=None, *, n=None, toroidal_points=None,
                             method='cubic', return_dataset=False):
        """
        Interpolate in the parallel direction to get a higher resolution version of the
        variable.

        Parameters
        ----------
        region : str, optional
            By default, return a result with all regions interpolated separately and then
            combined. If an explicit region argument is passed, then return the variable
            from only that region.
        n : int, optional
            The factor to increase the resolution by. Defaults to the value set by
            BoutDataset.setupParallelInterp(), or 10 if that has not been called.
        toroidal_points : int or sequence of int, optional
            If int, number of toroidal points to output, applies a stride to toroidal
            direction to save memory usage. If sequence of int, the indexes of toroidal
            points for the output.
        method : str, optional
            The interpolation method to use. Options from xarray.DataArray.interp(),
            currently: linear, nearest, zero, slinear, quadratic, cubic. Default is
            'cubic'.
        return_dataset : bool, optional
            If this is set to True, return a Dataset containing this variable as a member
            (by default returns a DataArray). Only used when region=None.

        Returns
        -------
        A new DataArray containing a high-resolution version of the variable. (If
        return_dataset=True, instead returns a Dataset containing the DataArray.)
        """

        if region is None:
            # Call the single-region version of this method for each region, and combine
            # the results together
            parts = [
                self.interpolate_parallel(region, n=n, toroidal_points=toroidal_points,
                                          method=method).bout.to_dataset()
                for region in self.data.regions]

            result = xr.combine_by_coords(parts)
            result.attrs = parts[0].attrs
            # xr.combine_by_coords does not keep attrs at the moment. See
            # https://github.com/pydata/xarray/issues/3865 For now just copy the attrs
            # from the first region. Can remove this workaround when the xarray issue is
            # fixed. Should be able to use instead of the above just:
            # result = xr.combine_by_coords(
            #    [self.interpolate_parallel(region, n=n, toroidal_points=toroidal_points,
            #                                method=method).bout.to_dataset()]
            # )

            # result has all regions, so should not have a region attribute
            if 'region' in result.attrs:
                del result.attrs['region']
            if 'region' in result[self.data.name].attrs:
                del result[self.data.name].attrs['region']

            if return_dataset:
                return result
            else:
                # Extract the DataArray to return
                return result[self.data.name]

        # Select a particular 'region' and interpolate to higher parallel resolution
        da = self.data
        region = da.regions[region]
        tcoord = da.metadata['bout_tdim']
        xcoord = da.metadata['bout_xdim']
        ycoord = da.metadata['bout_ydim']
        zcoord = da.metadata['bout_zdim']

        if zcoord in da.dims and da.direction_y != 'Aligned':
            aligned_input = False
            da = da.bout.toFieldAligned()
        else:
            aligned_input = True

        if n is None:
            n = self.fine_interpolation_factor

        da = da.bout.from_region(region.name, with_guards={xcoord: 0, ycoord: 2})
        da = da.chunk({ycoord: None})

        ny_fine = n*region.ny
        dy = (region.yupper - region.ylower)/ny_fine

        myg = da.metadata['MYG']
        if da.metadata['keep_yboundaries'] and region.connection_lower_y is None:
            ybndry_lower = myg
        else:
            ybndry_lower = 0
        if da.metadata['keep_yboundaries'] and region.connection_upper_y is None:
            ybndry_upper = myg
        else:
            ybndry_upper = 0

        y_fine = np.linspace(region.ylower - (ybndry_lower - 0.5)*dy,
                             region.yupper + (ybndry_upper - 0.5)*dy,
                             ny_fine + ybndry_lower + ybndry_upper)

        da = da.interp({ycoord: y_fine.data}, assume_sorted=True, method=method,
                       kwargs={'fill_value': 'extrapolate'})

        da = _update_metadata_increased_resolution(da, n)

        # Add dy to da as a coordinate. This will only be temporary, once we have
        # combined the regions together, we will demote dy to a regular variable
        dy_array = xr.DataArray(np.full([da.sizes[xcoord], da.sizes[ycoord]], dy),
                                dims=[xcoord, ycoord])
        # need a view of da with only x- and y-dimensions, unfortunately no neat way to
        # do this with isel
        da_2d = da
        if tcoord in da.sizes:
            da_2d = da_2d.isel(**{tcoord: 0}, drop=True)
        if zcoord in da.sizes:
            da_2d = da_2d.isel(**{zcoord: 0}, drop=True)
        dy_array = da_2d.copy(data=dy_array)
        da = da.assign_coords(dy=dy_array)

        # Remove regions which have incorrect information for the high-resolution grid.
        # New regions will be generated when creating a new Dataset in
        # BoutDataset.getHighParallelResVars
        del da.attrs['regions']

        if not aligned_input:
            # Want output in non-aligned coordinates
            da = da.bout.fromFieldAligned()

        if toroidal_points is not None and zcoord in da.sizes:
            if isinstance(toroidal_points, int):
                nz = len(da[zcoord])
                zstride = (nz + toroidal_points - 1)//toroidal_points
                da = da.isel(**{zcoord: slice(None, None, zstride)})
            else:
                da = da.isel(**{zcoord: toroidal_points})

        return da

    def animate2D(self, animate_over='t', x=None, y=None, animate=True, fps=10,
                  save_as=None, ax=None, poloidal_plot=False, logscale=None, **kwargs):
        """
        Plots a color plot which is animated with time over the specified
        coordinate.

        Currently only supports 2D+1 data, which it plots with animatplot's
        wrapping of matplotlib's pcolormesh.

        Parameters
        ----------
        animate_over : str, optional
            Dimension over which to animate
        x : str, optional
            Dimension to use on the x axis, default is None - then use the first spatial
            dimension of the data
        y : str, optional
            Dimension to use on the y axis, default is None - then use the second spatial
            dimension of the data
        animate : bool, optional
            If set to false, do not create the animation, just return the block or blocks
        fps : int, optional
            Frames per second of resulting gif
        save_as : True or str, optional
            If str is passed, save the animation as save_as+'.gif'.
            If True is passed, save the animation with a default name,
            '<variable name>_over_<animate_over>.gif'
        ax : matplotlib.pyplot.axes object, optional
            Axis on which to plot the gif
        poloidal_plot : bool, optional
            Use animate_poloidal to make a plot in R-Z coordinates (input field must be
            (t,x,y))
        logscale : bool or float, optional
            If True, default to a logarithmic color scale instead of a linear one.
            If a non-bool type is passed it is treated as a float used to set the linear
            threshold of a symmetric logarithmic scale as
            linthresh=min(abs(vmin),abs(vmax))*logscale, defaults to 1e-5 if True is
            passed.
        kwargs : dict, optional
            Additional keyword arguments are passed on to the plotting function
            (animatplot.blocks.Pcolormesh).
        """

        data = self.data
        variable = data.name
        n_dims = len(data.dims)

        if n_dims == 3:
            vmin = kwargs['vmin'] if 'vmin' in kwargs else data.min().values
            vmax = kwargs['vmax'] if 'vmax' in kwargs else data.max().values
            kwargs['norm'] = _create_norm(logscale, kwargs.get('norm', None), vmin, vmax)

            if poloidal_plot:
                print("{} data passed has {} dimensions - making poloidal plot with "
                      "animate_poloidal()".format(variable, str(n_dims)))
                if x is not None:
                    kwargs['x'] = x
                if y is not None:
                    kwargs['y'] = y
                poloidal_blocks = animate_poloidal(data, animate_over=animate_over,
                                                   animate=animate, fps=fps,
                                                   save_as=save_as, ax=ax, **kwargs)
                return poloidal_blocks
            else:
                print("{} data passed has {} dimensions - will use "
                      "animatplot.blocks.Pcolormesh()".format(variable, str(n_dims)))
                pcolormesh_block = animate_pcolormesh(data=data,
                                                      animate_over=animate_over, x=x,
                                                      y=y, animate=animate, fps=fps,
                                                      save_as=save_as, ax=ax, **kwargs)
                return pcolormesh_block
        else:
            raise ValueError(
                "Data passed has an unsupported number of dimensions "
                "({})".format(str(n_dims)))

    def animate1D(self, animate_over='t', animate=True, fps=10, save_as=None,
                  sep_pos=None, ax=None, **kwargs):
        """
        Plots a line plot which is animated over time over the specified coordinate.

        Currently only supports 1D+1 data, which it plots with animatplot's wrapping of
        matplotlib's plot.

        Parameters
        ----------
        animate_over : str, optional
            Dimension over which to animate
        fps : int, optional
            Frames per second of resulting gif
        save_as : True or str, optional
            If str is passed, save the animation as save_as+'.gif'.
            If True is passed, save the animation with a default name,
            '<variable name>_over_<animate_over>.gif'
        sep_pos : int, optional
            Radial position at which to plot the separatrix
        ax : Axes, optional
            A matplotlib axes instance to plot to. If None, create a new
            figure and axes, and plot to that
        kwargs : dict, optional
            Additional keyword arguments are passed on to the plotting function
            (animatplot.blocks.Line).
        """

        data = self.data
        variable = data.name
        n_dims = len(data.dims)

        if n_dims == 2:
            print("{} data passed has {} dimensions - will use "
                  "animatplot.blocks.Line()".format(variable, str(n_dims)))
            line_block = animate_line(data=data, animate_over=animate_over,
                                      sep_pos=sep_pos, animate=animate, fps=fps,
                                      save_as=save_as, ax=ax, **kwargs)
            return line_block

    # BOUT-specific plotting functionality: methods that plot on a poloidal (R-Z) plane
    def contour(self, ax=None, **kwargs):
        return plotfuncs.plot2d_wrapper(self.data, xr.plot.contour, ax=ax, **kwargs)

    def contourf(self, ax=None, **kwargs):
        return plotfuncs.plot2d_wrapper(self.data, xr.plot.contourf, ax=ax, **kwargs)

    def pcolormesh(self, ax=None, **kwargs):
        return plotfuncs.plot2d_wrapper(self.data, xr.plot.pcolormesh, ax=ax, **kwargs)

    def regions(self, ax=None, **kwargs):
        return plotfuncs.regions(self.data, ax=ax, **kwargs)
