from copy import deepcopy
from pprint import pformat as prettyformat
from functools import partial

import numpy as np

import xarray as xr
from xarray import register_dataarray_accessor

from .plotting.animate import animate_poloidal, animate_pcolormesh, animate_line
from .plotting import plotfuncs
from .plotting.utils import _create_norm
from .region import Region


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
        the Dataset.
        """
        da = self.data
        ds = da.to_dataset()

        ds.attrs = da.attrs

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

        nz = self.data.metadata['nz']

        # Get axis position of dimension to transform
        axis = self.data.dims.index(self.data.metadata['bout_zdim'])

        # Create list the dimensions of FFT'd array
        fft_dims = list(deepcopy(self.data.dims))
        fft_dims[axis] = 'kz'

        # Fourier transform to get the DataArray in k-space
        data_fft = np.fft.fft(self.data.values, axis=axis)

        # Complex phase for rotation by angle zShift
        zperiod = 1./(self.data.metadata['ZMAX'] - self.data.metadata['ZMIN'])
        kz = xr.DataArray(np.arange(0, nz*zperiod, zperiod), dims='kz')
        phases = 1.j * zShift * kz

        # Ensure dimensions are in correct order for numpy broadcasting
        extra_dims = deepcopy(fft_dims)
        for dim in phases.dims:
            extra_dims.remove(dim)
        phases = phases.expand_dims(extra_dims)
        phases = phases.transpose(*fft_dims)

        data_shifted_fft = data_fft * np.exp(phases.values)

        if(nz % 2 == 0):
            nfft = nz // 2
            data_shifted_fft[:, :, :, nfft] = data_shifted_fft[:, :, :, nfft].real
            data_shifted_fft[:, :, :, nfft+1:] = np.conj(
                    data_shifted_fft[:, :, :, nfft-1:0:-1])
        else:
            nfft = (nz - 1)//2
            data_shifted_fft[:, :, :, nfft+1:] = np.conj(
                    data_shifted_fft[:, :, :, nfft:0:-1])

        data_shifted = np.fft.ifft(data_shifted_fft).real

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

    def fromRegion(self, region, with_guards=None):
        """
        Get a logically-rectangular section of data from a certain region.
        Includes guard cells from neighbouring regions.

        Parameters
        ----------
        region : str
            Region to get data for
        with_guards : int or dict of int, optional
            Number of guard cells to include, by default use MXG and MYG from BOUT++. Pass
            a dict to set different numbers for different coordinates.
        """

        region = self.data.regions[region]
        xcoord = self.data.metadata['bout_xdim']
        ycoord = self.data.metadata['bout_ydim']

        if with_guards is None:
            mxg = self.data.metadata['MXG']
            myg = self.data.metadata['MYG']
        else:
            try:
                try:
                    mxg = with_guards[xcoord]
                except KeyError:
                    mxg = self.data.metadata['MXG']
                try:
                    myg = with_guards[ycoord]
                except KeyError:
                    myg = self.data.metadata['MYG']
            except TypeError:
                mxg = with_guards
                myg = with_guards

        xslice, yslice = region.getSlices()
        da = self.data.isel(**{xcoord: xslice, ycoord: yslice})

        if mxg > 0:
            # get guard cells from x-neighbour regions
            if region.connection_inner is not None:
                da_inner = self.data.bout.fromRegion(region.connection_inner,
                                                     with_guards={xcoord: 0, ycoord:0})

                # select just the points we need to fill the guard cells of da
                da_inner = da_inner.isel(**{xcoord: slice(-mxg, None)})

                if xcoord in da.coords:
                    # Use local coordinates for neighbouring region, not communicated ones
                    # Note, at the moment this should do nothing, because all neigbours in the
                    # x-direction are contiguous in the global array anyway, but included to
                    # be future-proof
                    xslice, yslice = region.getInnerGuardsSlices(mxg)
                    new_xcoord = self.data[xcoord].isel(**{xcoord: xslice})
                    da_inner = da_inner.assign_coords(**{xcoord: new_xcoord})

                da = xr.concat((da_inner, da), xcoord)
            if region.connection_outer is not None:
                da_outer = self.data.bout.fromRegion(region.connection_outer,
                                                     with_guards={xcoord: 0, ycoord:0})

                # select just the points we need to fill the guard cells of da
                da_outer = da_outer.isel(**{xcoord: slice(mxg)})

                if xcoord in da.coords:
                    # Use local coordinates for neighbouring region, not communicated ones
                    # Note, at the moment this should do nothing, because all neigbours in the
                    # x-direction are contiguous in the global array anyway, but included to
                    # be future-proof
                    xslice, yslice = region.getOuterGuardsSlices(mxg)
                    new_xcoord = self.data[xcoord].isel(**{xcoord: xslice})
                    da_outer = da_outer.assign_coords(**{xcoord: new_xcoord})

                da = xr.concat((da, da_outer), xcoord)

        if myg > 0:
            # get guard cells from y-neighbour regions
            if region.connection_lower is not None:
                da_lower = self.data.bout.fromRegion(region.connection_lower,
                                                     with_guards={xcoord: mxg, ycoord:0})

                # select just the points we need to fill the guard cells of da
                da_lower = da_lower.isel(**{ycoord: slice(-myg, None)})

                if ycoord in da.coords:
                    # Use local coordinates for neighbouring region, not communicated ones
                    xslice, yslice = region.getLowerGuardsSlices(myg)
                    new_ycoord = self.data[ycoord].isel(**{ycoord:yslice})
                    da_lower = da_lower.assign_coords(**{ycoord: new_ycoord})

                da = xr.concat((da_lower, da), ycoord)
            if region.connection_upper is not None:
                da_upper = self.data.bout.fromRegion(region.connection_upper,
                                                     with_guards={xcoord: mxg, ycoord:0})

                # select just the points we need to fill the guard cells of da
                da_upper = da_upper.isel(**{ycoord: slice(myg)})

                if ycoord in da.coords:
                    # Use local coordinates for neighbouring region, not communicated ones
                    xslice, yslice = region.getUpperGuardsSlices(myg)
                    new_ycoord = self.data[ycoord].isel(**{ycoord:yslice})
                    da_upper = da_upper.assign_coords(**{ycoord: new_ycoord})

                da = xr.concat((da, da_upper), ycoord)

        da.attrs['region'] = region

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
