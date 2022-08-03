from copy import copy, deepcopy
from pprint import pformat as prettyformat
from functools import partial

import dask.array
import matplotlib.path
import numpy as np
from scipy.interpolate import griddata as scipy_griddata

import xarray as xr
from xarray import register_dataarray_accessor

from .geometries import apply_geometry
from .plotting.animate import animate_poloidal, animate_pcolormesh, animate_line
from .plotting import plotfuncs
from .plotting.utils import _create_norm
from .region import _from_region
from .utils import (
    _add_cartesian_coordinates,
    _update_metadata_increased_resolution,
    _get_bounding_surfaces,
)


@register_dataarray_accessor("bout")
class BoutDataArrayAccessor:
    """
    Contains BOUT-specific methods to use on BOUT++ dataarrays opened by
    selecting a variable from a BOUT++ dataset.

    These BOUT-specific methods and attributes are accessed via the bout
    accessor, e.g. `da.bout.options` returns a `BoutOptionsFile` instance.
    """

    def __init__(self, da):
        self.data = da
        self.metadata = da.attrs.get("metadata")  # None if just grid file
        self.options = da.attrs.get("options")  # None if no inp file

    def __str__(self):
        """
        String representation of the BoutDataArray.

        Accessed by print(da.bout)
        """

        styled = partial(prettyformat, indent=4, compact=True)
        text = (
            "<xbout.BoutDataset>\n"
            + "Contains:\n{}\n".format(str(self.data))
            + "Metadata:\n{}\n".format(styled(self.metadata))
        )
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

        dropIfExists(ds, "direction_y")
        dropIfExists(ds, "direction_z")
        dropIfExists(ds, "cell_location")

        return ds

    def _shift_z(self, zShift):
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

        nz = self.data.metadata["nz"]
        # Assume dz is constant here - using FFTs doesn't make much sense if z isn't a
        # toroidal angle coordinate.
        zlength = nz * self.data["dz"].values.flatten()[0]
        nmodes = nz // 2 + 1

        # Get axis position of dimension to transform
        axis = self.data.dims.index(self.data.metadata["bout_zdim"])

        # Create list the dimensions of FFT'd array
        fft_dims = list(deepcopy(self.data.dims))
        fft_dims[axis] = "kz"

        # Fourier transform to get the DataArray in k-space
        data_fft = fft.rfft(self.data.data, axis=axis)

        # Complex phase for rotation by angle zShift
        kz = 2.0 * np.pi * xr.DataArray(np.arange(0, nmodes), dims="kz") / zlength
        phase = 1.0j * kz * zShift

        # Ensure dimensions are in correct order for numpy broadcasting
        extra_dims = deepcopy(fft_dims)
        for dim in phase.dims:
            extra_dims.remove(dim)
        phase = phase.expand_dims(extra_dims)
        phase = phase.transpose(*fft_dims, transpose_coords=True)

        data_shifted_fft = data_fft * np.exp(phase.data)

        data_shifted = fft.irfft(data_shifted_fft, n=nz, axis=axis)

        # Return a DataArray with the same attributes as self, but values from
        # data_shifted
        return self.data.copy(data=data_shifted)

    def to_field_aligned(self):
        """
        Transform DataArray to field-aligned coordinates, which are shifted with respect
        to the base coordinates by an angle zShift
        """
        if (
            self.data.cell_location == "CELL_CENTRE"
            or self.data.cell_location == "CELL_ZLOW"
        ):
            zShift_coord = "zShift"
        else:
            zShift_coord = "zShift_" + self.data.cell_location

        if self.data.direction_y != "Standard":
            raise ValueError(
                f"Cannot shift a {self.data.direction_y} type field to "
                "field-aligned coordinates"
            )

        if zShift_coord not in self.data.coords:
            raise ValueError(
                f"{zShift_coord} missing, cannot shift "
                f"{self.data.cell_location} field {self.data.name} to "
                f"field-aligned coordinates. Setting toroidal geometry is necessary to "
                f'use to_field_aligned() - did you pass the `geometry="toroidal"` '
                f"argument to open_boutdataset()?"
            )

        result = self._shift_z(self.data[zShift_coord])
        result.attrs["direction_y"] = "Aligned"
        return result

    def from_field_aligned(self):
        """
        Transform DataArray from field-aligned coordinates, which are shifted with
        respect to the base coordinates by an angle zShift
        """
        if (
            self.data.cell_location == "CELL_CENTRE"
            or self.data.cell_location == "CELL_ZLOW"
        ):
            zShift_coord = "zShift"
        else:
            zShift_coord = "zShift_" + self.data.cell_location

        if self.data.direction_y != "Aligned":
            raise ValueError(
                f"Cannot shift a {self.data.direction_y} type field from "
                "field-aligned coordinates"
            )

        if zShift_coord not in self.data.coords:
            raise ValueError(
                f"{zShift_coord} missing, cannot shift "
                f"{self.data.cell_location} field {self.data.name} from "
                f"field-aligned coordinates. Setting toroidal geometry is necessary to "
                f'use from_field_aligned() - did you pass the `geometry="toroidal"` '
                f"argument to open_boutdataset()?"
            )

        result = self._shift_z(-self.data[zShift_coord])
        result.attrs["direction_y"] = "Standard"
        return result

    @property
    def _regions(self):
        if "regions" not in self.data.attrs:
            raise ValueError(
                "Called a method requiring regions, but these have not been created. "
                "Please set the 'geometry' option when calling open_boutdataset() to "
                "create regions."
            )
        return self.data.attrs["regions"]

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
        return _from_region(self.data, name, with_guards)

    @property
    def fine_interpolation_factor(self):
        """
        The default factor to increase resolution when doing parallel interpolation
        """
        return self.data.metadata["fine_interpolation_factor"]

    @fine_interpolation_factor.setter
    def fine_interpolation_factor(self, n):
        """
        Set the default factor to increase resolution when doing parallel interpolation.

        Parameters
        -----------
        n : int
            Factor to increase parallel resolution by
        """
        self.data.metadata["fine_interpolation_factor"] = n

    def interpolate_parallel(
        self,
        region=None,
        *,
        n=None,
        toroidal_points=None,
        method="cubic",
        return_dataset=False,
    ):
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
                self.interpolate_parallel(
                    region, n=n, toroidal_points=toroidal_points, method=method
                ).bout.to_dataset()
                for region in self._regions
            ]

            # 'region' is not the same for all parts, and should not exist in the result,
            # so delete before merging
            for part in parts:
                if "region" in part.attrs:
                    del part.attrs["region"]
                if "region" in part[self.data.name].attrs:
                    del part[self.data.name].attrs["region"]

            result = xr.combine_by_coords(parts, combine_attrs="drop_conflicts")

            if return_dataset:
                return result
            else:
                # Extract the DataArray to return
                result = apply_geometry(result, self.data.geometry)
                return result[self.data.name]

        # Select a particular 'region' and interpolate to higher parallel resolution
        da = self.data
        region = da.bout._regions[region]
        tcoord = da.metadata["bout_tdim"]
        xcoord = da.metadata["bout_xdim"]
        ycoord = da.metadata["bout_ydim"]
        zcoord = da.metadata["bout_zdim"]

        da = da.bout.from_region(region.name, with_guards={xcoord: 0, ycoord: 2})

        if zcoord in da.dims and da.direction_y != "Aligned":
            aligned_input = False
            da = da.bout.to_field_aligned()
        else:
            aligned_input = True

        if n is None:
            n = self.fine_interpolation_factor

        da = da.chunk({ycoord: None})

        ny_fine = n * region.ny
        dy = (region.yupper - region.ylower) / ny_fine

        myg = da.metadata["MYG"]
        if da.metadata["keep_yboundaries"] and region.connection_lower_y is None:
            ybndry_lower = myg
        else:
            ybndry_lower = 0
        if da.metadata["keep_yboundaries"] and region.connection_upper_y is None:
            ybndry_upper = myg
        else:
            ybndry_upper = 0

        y_fine = np.linspace(
            region.ylower - (ybndry_lower - 0.5) * dy,
            region.yupper + (ybndry_upper - 0.5) * dy,
            ny_fine + ybndry_lower + ybndry_upper,
        )

        # This prevents da.interp() from being very slow.
        # Apparently large attrs (i.e. regions) on a coordinate which is passed as an
        # argument to dask.array.map_blocks() slow things down, maybe because coordinates
        # are numpy arrays, not dask arrays?
        # Slow-down was introduced in d062fa9e75c02fbfdd46e5d1104b9b12f034448f when
        # _add_attrs_to_var(updated_ds, ycoord) was added in geometries.py
        da[ycoord].attrs = {}

        da = da.interp(
            {ycoord: y_fine.data},
            assume_sorted=True,
            method=method,
            kwargs={"fill_value": "extrapolate"},
        )

        da = _update_metadata_increased_resolution(da, n)

        # Modify dy to be consistent with the higher resolution grid
        dy_array = xr.DataArray(
            np.full([da.sizes[xcoord], da.sizes[ycoord]], dy), dims=[xcoord, ycoord]
        )
        da["dy"] = da["dy"].copy(data=dy_array)

        # Remove regions which have incorrect information for the high-resolution grid.
        # New regions will be generated when creating a new Dataset in
        # BoutDataset.getHighParallelResVars
        del da.attrs["regions"]

        if not aligned_input:
            # Want output in non-aligned coordinates
            da = da.bout.from_field_aligned()

        if toroidal_points is not None and zcoord in da.sizes:
            if isinstance(toroidal_points, int):
                nz = len(da[zcoord])
                zstride = (nz + toroidal_points - 1) // toroidal_points
                da = da.isel(**{zcoord: slice(None, None, zstride)})
            else:
                da = da.isel(**{zcoord: toroidal_points})

        return da

    def add_cartesian_coordinates(self):
        """
        Add Cartesian (X,Y,Z) coordinates.

        Returns
        -------
        DataArray with new coordinates added, which are named 'X_cartesian',
        'Y_cartesian', and 'Z_cartesian'
        """
        return _add_cartesian_coordinates(self.data)

    def remove_yboundaries(self, return_dataset=False, remove_extra_upper=False):
        """
        Remove y-boundary points, if present, from the DataArray

        Parameters
        ----------
        return_dataset : bool, default False
            Return the result as a Dataset containing the new DataArray.
        """

        myg = self.data.metadata["MYG"]

        if (
            self.metadata["keep_yboundaries"] == 0 or myg == 0
        ) and not remove_extra_upper:
            # Ensure we do not modify any other references to metadata
            self.data.attrs["metadata"] = deepcopy(self.data.metadata)
            self.data.metadata["keep_yboundaries"] = 0
            # no y-boundary points to remove
            if return_dataset:
                return self.to_dataset()
            else:
                return self.data
        if self.metadata["keep_yboundaries"] == 0:
            myg = 0

        ycoord = self.data.metadata["bout_ydim"]
        parts = []
        for region in self._regions:
            part = self.data.bout.from_region(region, with_guards=0)
            part_region = list(part.bout._regions.values())[0]
            if part_region.connection_lower_y is None:
                part = part.isel({ycoord: slice(myg, None)})
            if part_region.connection_upper_y is None:
                part = part.isel(
                    {ycoord: slice(-myg if not remove_extra_upper else -myg - 1)}
                )
            del part.attrs["regions"]
            parts.append(part.bout.to_dataset())

        result = xr.combine_by_coords(parts)
        # Ensure we do not modify any other references to metadata
        result.attrs = copy(parts[0].attrs)
        result.attrs["metadata"] = deepcopy(self.data.metadata)
        result[self.data.name].attrs["metadata"] = deepcopy(self.data.metadata)

        # result is as if we had not kept y-boundaries when loading
        result.metadata["keep_yboundaries"] = 0
        result[self.data.name].metadata["keep_yboundaries"] = 0

        if remove_extra_upper:
            # modify jyseps*, ny_inner, ny so that sliced variable gets consistent
            # regions
            if result.metadata["jyseps1_2"] == result.metadata["jyseps2_1"]:
                # single-null
                result.metadata["ny"] -= 1
            else:
                # double-null
                result.metadata["ny_inner"] -= 1
                result.metadata["jyseps1_2"] -= 1
                result.metadata["jyseps2_2"] -= 1
                result.metadata["ny"] -= 2

        if return_dataset:
            return result
        else:
            # Extract the DataArray to return
            return result[self.data.name]

    def ddx(self):
        """
        Special method for calculating a derivative in the "bout_xdim"
        direction (radial, x-direction), needed because the 1d coordinate in this
        direction is index number, not coordinate values (because psi can be different
        in core and PFR regions).

        This method uses a second-order accurate central finite difference method to
        calculate the derivative.

        Note values at the boundaries will be NaN - if Dataset was loaded with
        keep_xboundaries=True, these should only ever be in boundary cells.
        """
        da = self.data
        xcoord = da.metadata["bout_xdim"]

        if da.cell_location == "CELL_CENTRE":
            dx = da["dx"]
        elif da.cell_location == "CELL_XLOW":
            dx = da["dx_CELL_XLOW"]
        elif da.cell_location == "CELL_YLOW":
            dx = da["dx_CELL_YLOW"]
        elif da.cell_location == "CELL_ZLOW":
            dx = da["dx_CELL_ZLOW"]
        else:
            raise ValueError(f'Unrecognised cell location "{da.cell_location}"')

        result = (da.shift({xcoord: -1}) - da.shift({xcoord: 1})) / (2.0 * dx)

        result.name = f"d({da.name})/dx"
        if "standard_name" in result.attrs:
            result.attrs["standard_name"] = f"d({result.attrs['standard_name']})/dx"
        if "long_name" in result.attrs:
            result.attrs["long_name"] = f"x-derivative of {result.attrs['long_name']}"
        if "units" in result.attrs:
            result.attrs["units"] = ""

        return result

    def ddy(self, region=None):
        """
        Special method for calculating a derivative in the "bout_ydim"
        direction (parallel, y-direction), needed because we need to (a) do the
        calculation region-by-region to take account of the branch cuts in the
        y-direction and (b) transform to a field-aligned grid to take parallel
        derivatives.

        This method uses a second-order accurate central finite difference
        method to calculate the derivative. It transforms to a globally field-aligned
        grid to calculate the derivative - there is currently no option to use the same
        method as the BOUT++ approach using 'yup' and 'ydown' fields.

        Note values at the boundaries will be NaN - if Dataset was loaded with
        keep_yboundaries=True, these should only ever be in boundary cells.

        Warnings
        --------
        Depending on how parallel boundary conditions were applied in the BOUT++
        PhysicsModel, the y-boundary cells may not contain valid data, in which case the
        result may be incorrect in the grid cell closest to the boundary.

        Parameters
        ----------
        region : str, optional
            By default, return a result with the derivative calculated in all regions
            separately and then combined. If an explicit region argument is passed, then
            return the result from only that region.

        Returns
        -------
        A new DataArray containing the y-derivative of the variable.
        """
        if region is None:
            # Call the single-region version of this method for each region, and combine
            # the results together
            parts = [self.ddy(r).to_dataset() for r in self._regions]

            # 'region' is not the same for all parts, and should not exist in the
            # result, so delete before merging
            for part in parts:
                if "region" in part.attrs:
                    del part.attrs["region"]

            name = self.data.name
            result = xr.combine_by_coords(parts)[f"d({name})/dy"]

            # regions get mixed up during the split and combine_by_coords, so reset them
            result.attrs["regions"] = self._regions

            return result

        da = self.data
        xcoord = da.metadata["bout_xdim"]
        ycoord = da.metadata["bout_ydim"]
        zcoord = da.metadata["bout_zdim"]

        da = self.data.bout.from_region(region, with_guards={xcoord: 0, ycoord: 1})

        if zcoord in da.dims and da.direction_y != "Aligned":
            aligned_input = False
            da = da.bout.to_field_aligned()
        else:
            aligned_input = True

        if da.cell_location == "CELL_CENTRE":
            dy = da["dy"]
        elif da.cell_location == "CELL_XLOW":
            dy = da["dy_CELL_XLOW"]
        elif da.cell_location == "CELL_YLOW":
            dy = da["dy_CELL_YLOW"]
        elif da.cell_location == "CELL_ZLOW":
            dy = da["dy_CELL_ZLOW"]
        else:
            raise ValueError(f'Unrecognised cell location "{da.cell_location}"')

        result = (da.shift({ycoord: -1}) - da.shift({ycoord: 1})) / (2.0 * dy)

        # Remove any y-guard cells
        region_object = da.bout._regions[region]
        if region_object.connection_lower_y is None:
            ylower = None
        else:
            ylower = 1
        if region_object.connection_upper_y is None:
            yupper = None
        else:
            yupper = -1
        result = result.isel({ycoord: slice(ylower, yupper)})

        if not aligned_input:
            # Want output in non-aligned coordinates
            result = result.bout.from_field_aligned()

        if "regions" in result.attrs:
            del result.attrs["regions"]

        result.name = f"d({da.name})/dy"
        if "standard_name" in result.attrs:
            result.attrs["standard_name"] = f"d({result.attrs['standard_name']})/dy"
        if "long_name" in result.attrs:
            result.attrs["long_name"] = f"y-derivative of {result.attrs['long_name']}"
        if "units" in result.attrs:
            result.attrs["units"] = ""

        return result

    def ddz(self):
        """
        Special method for calculating a derivative in the "bout_zdim"
        direction (toroidal, z-direction), needed because xarray's
        differentiate method doesn't have an option to handle a periodic
        dimension (as of xarray-0.17.0).

        This method uses a second-order accurate central finite difference method to
        calculate the derivative.
        """
        da = self.data
        zcoord = da.metadata["bout_zdim"]
        result = (
            da.roll({zcoord: -1}, roll_coords=False)
            - da.roll({zcoord: 1}, roll_coords=False)
        ) / (2.0 * da["dz"])

        result.name = f"d({da.name})/dz"
        if "standard_name" in result.attrs:
            result.attrs["standard_name"] = f"d({result.attrs['standard_name']})/dz"
        if "long_name" in result.attrs:
            result.attrs["long_name"] = f"z-derivative of {result.attrs['long_name']}"
        if "units" in result.attrs:
            result.attrs["units"] = ""

        return result

    def get_bounding_surfaces(self, coords=("R", "Z")):
        """
        Get bounding surfaces.
        Surfaces are returned as arrays of points describing a polygon, assuming the
        third spatial dimension is a symmetry direction.

        Parameters
        ----------
        coords : (str, str), default ("R", "Z")
            Pair of names of coordinates whose values are used to give the positions of
            the points in the result

        Returns
        -------
        result : list of DataArrays
            Each DataArray in the list contains points on a boundary, with size
            (<number of points in the bounding polygon>, 2). Points wind clockwise around
            the outside domain, and anti-clockwise around the inside (if there is an
            inner boundary).
        """
        return _get_bounding_surfaces(self.data, coords)

    def animate2D(
        self,
        animate_over=None,
        x=None,
        y=None,
        animate=True,
        axis_coords=None,
        fps=10,
        save_as=None,
        ax=None,
        poloidal_plot=False,
        logscale=None,
        **kwargs,
    ):
        """
        Plots a color plot which is animated with time over the specified
        coordinate.

        Currently only supports 2D+1 data, which it plots with animatplot's
        wrapping of matplotlib's pcolormesh.

        Parameters
        ----------
        animate_over : str, optional
            Dimension over which to animate, defaults to the time dimension
        x : str, optional
            Dimension to use on the x axis, default is None - then use the first spatial
            dimension of the data
        y : str, optional
            Dimension to use on the y axis, default is None - then use the second spatial
            dimension of the data
        animate : bool, optional
            If set to false, do not create the animation, just return the block or blocks
        axis_coords : None, str, dict
            Coordinates to use for axis labelling.
            - None: Use the dimension coordinate for each axis, if it exists.
            - "index": Use the integer index values.
            - dict: keys are dimension names, values set axis_coords for each axis
              separately. Values can be: None, "index", the name of a 1d variable or
              coordinate (which must have the dimension given by 'key'), or a 1d
              numpy array, dask array or DataArray whose length matches the length of
              the dimension given by 'key'.
            Only affects time coordinate for plots with poloidal_plot=True.
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
        aspect : str or None, optional
            Argument to set_aspect(). Defaults to "equal" for poloidal plots and "auto"
            for others.
        kwargs : dict, optional
            Additional keyword arguments are passed on to the plotting function
            (animatplot.blocks.Pcolormesh).

        Returns
        -------
        animation or blocks
            If animate==True, returns an animatplot.Animation object, otherwise
            returns a list of animatplot.blocks.Pcolormesh instances.
        """

        data = self.data
        variable = data.name
        n_dims = len(data.dims)

        if n_dims == 3:
            vmin = kwargs.pop("vmin") if "vmin" in kwargs else data.min().values
            vmax = kwargs.pop("vmax") if "vmax" in kwargs else data.max().values
            kwargs["norm"] = _create_norm(
                logscale, kwargs.get("norm", None), vmin, vmax
            )

            if poloidal_plot:
                print(
                    "{} data passed has {} dimensions - making poloidal plot with "
                    "animate_poloidal()".format(variable, str(n_dims))
                )
                if x is not None:
                    kwargs["x"] = x
                if y is not None:
                    kwargs["y"] = y
                poloidal_blocks = animate_poloidal(
                    data,
                    animate_over=animate_over,
                    animate=animate,
                    axis_coords=axis_coords,
                    fps=fps,
                    save_as=save_as,
                    ax=ax,
                    **kwargs,
                )
                return poloidal_blocks
            else:
                print(
                    "{} data passed has {} dimensions - will use "
                    "animatplot.blocks.Pcolormesh()".format(variable, str(n_dims))
                )
                pcolormesh_block = animate_pcolormesh(
                    data=data,
                    animate_over=animate_over,
                    x=x,
                    y=y,
                    animate=animate,
                    axis_coords=axis_coords,
                    fps=fps,
                    save_as=save_as,
                    ax=ax,
                    **kwargs,
                )
                return pcolormesh_block
        else:
            raise ValueError(
                "Data passed has an unsupported number of dimensions "
                "({})".format(str(n_dims))
            )

    def animate1D(
        self,
        animate_over=None,
        animate=True,
        axis_coords=None,
        fps=10,
        save_as=None,
        sep_pos=None,
        ax=None,
        **kwargs,
    ):
        """
        Plots a line plot which is animated over time over the specified coordinate.

        Currently only supports 1D+1 data, which it plots with animatplot's wrapping of
        matplotlib's plot.

        Parameters
        ----------
        animate_over : str, optional
            Dimension over which to animate, defaults to the time dimension
        axis_coords : None, str, dict
            Coordinates to use for axis labelling.
            - None: Use the dimension coordinate for each axis, if it exists.
            - "index": Use the integer index values.
            - dict: keys are dimension names, values set axis_coords for each axis
              separately. Values can be: None, "index", the name of a 1d variable or
              coordinate (which must have the dimension given by 'key'), or a 1d
              numpy array, dask array or DataArray whose length matches the length of
              the dimension given by 'key'.
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
        aspect : str or None, optional
            Argument to set_aspect(), defaults to "auto"
        kwargs : dict, optional
            Additional keyword arguments are passed on to the plotting function
            (animatplot.blocks.Line).

        Returns
        -------
        animation or block
            If animate==True, returns an animatplot.Animation object, otherwise
            returns an animatplot.blocks.Line instance.
        """

        data = self.data
        variable = data.name
        n_dims = len(data.dims)

        if n_dims == 2:
            print(
                "{} data passed has {} dimensions - will use "
                "animatplot.blocks.Line()".format(variable, str(n_dims))
            )
            line_block = animate_line(
                data=data,
                animate_over=animate_over,
                axis_coords=axis_coords,
                sep_pos=sep_pos,
                animate=animate,
                fps=fps,
                save_as=save_as,
                ax=ax,
                **kwargs,
            )
            return line_block

    def interpolate_from_unstructured(
        self,
        *,
        fill_value=np.nan,
        structured_output=True,
        unstructured_dim_name="unstructured_dim",
        **kwargs,
    ):
        """Interpolate DataArray onto new grids of some existing coordinates

        Parameters
        ----------
        **kwargs : (str, array)
            Each keyword is the name of a coordinate in the DataArray, the argument is a
            1d array giving the values of that coordinate on the output grid
        fill_value : float, default np.nan
            fill_value passed through to scipy.interpolation.griddata
        structured_output : bool, default True
            If True, treat output coordinates values as a structured grid.
            If False, output coordinate values must all have the same length and are not
            broadcast together.
        unstructured_dim_name : str, default "unstructured_dim"
            Name used for the dimension in the output that replaces the dimensions of
            the interpolated coordinates. Only used if structured_output=False.

        Returns
        -------
        DataArray
            Data interpolated onto a new, structured grid
        """

        da = self.data

        if structured_output:
            new_coords = {
                name: xr.DataArray(values, dims=name) for name, values in kwargs.items()
            }

            coord_arrays = tuple(
                np.meshgrid(*[values for values in kwargs.values()], indexing="ij")
            )

            new_output_dims = [d for d in kwargs]
        else:
            new_coords = {
                name: xr.DataArray(values, dims=unstructured_dim_name)
                for name, values in kwargs.items()
            }

            coord_arrays = tuple(kwargs.values())

            lengths = [len(c) for c in coord_arrays]
            if np.any([x != lengths[0] for x in lengths[1:]]):
                raise ValueError(
                    f"When structured_output=False, all the arrays of output coordinate"
                    f"values must have the same length. Got lengths "
                    f"{dict((name, len(coord)) for name, coord in kwargs.items())}"
                )

            new_output_dims = [unstructured_dim_name]

        # Figure out number of dimensions in the coordinates to be interpolated
        dims = set()
        for coord in kwargs:
            dims = dims.union(da[coord].dims)
        dims = tuple(dims)
        ndim = len(dims)

        # dimensions that are not being interpolated
        remaining_dims = tuple(d for d in da.dims if d not in dims)

        # Select interpolation method
        if ndim <= 2:
            # "cubic" only available for 1d or 2d interpolation
            method = "cubic"
        else:
            method = "linear"

        # extend input coordinates to cover all dims, so we can flatten them
        input_coords = []
        for coord in kwargs:
            data = da[coord]
            missing_dims = tuple(set(dims) - set(data.dims))
            expand = {dim: da.sizes[dim] for dim in missing_dims}
            expand_positions = tuple(dims.index(d) for d in missing_dims)
            da[coord] = data.expand_dims(expand, axis=expand_positions)

        # scipy.interpolate.griddata requires the axis being interpolated to be the first
        # one, so stack together 'dims', and then transpose so the resulting stacked
        # dimension is the first
        dims_name_list = [d for d in da.dims if d in dims]
        stacked_dim_name = "stacked_" + "_".join(dims_name_list)
        stacked = da.stack({stacked_dim_name: dims_name_list})
        stacked = stacked.transpose(
            *((stacked_dim_name,) + remaining_dims), transpose_coords=True
        )

        result = scipy_griddata(
            tuple(stacked[coord] for coord in kwargs),
            stacked,
            coord_arrays,
            method=method,
            fill_value=fill_value,
        )

        # griddata only sets points outside the 'convex hull' to fill_value
        # Nicer to set all points outside the grid boundaries to fill_value
        ###################################################################
        boundaries = self.get_bounding_surfaces(coords=[c for c in kwargs])
        points = np.stack(coord_arrays, axis=-1)

        # boundaries[0] is the outer boundary
        path = matplotlib.path.Path(boundaries[0], closed=True, readonly=True)
        is_contained = path.contains_points(points.reshape([-1, 2]))
        is_contained = is_contained.reshape(
            coord_arrays[0].shape + (1,) * len(remaining_dims)
        )
        result = np.where(is_contained, result, fill_value)

        # boundaries[1] is the inner boundary if it exists
        if len(boundaries) > 1:
            path = matplotlib.path.Path(boundaries[1], closed=True, readonly=True)
            is_contained = path.contains_points(points.reshape([-1, 2]))
            is_contained = is_contained.reshape(
                coord_arrays[0].shape + (1,) * len(remaining_dims)
            )
            result = np.where(is_contained, fill_value, result)

        if len(boundaries) > 2:
            raise ValueError(f"Found {len(boundaries)} boundaries, expected at most 2")

        # Create DataArray to return, with as much metadata as possible retained
        ########################################################################
        new_coords.update(
            {
                name: array
                for name, array in stacked.coords.items()
                if stacked_dim_name not in array.dims
            }
        )

        result = xr.DataArray(
            result,
            dims=new_output_dims + list(remaining_dims),
            coords=new_coords,
            name=da.name,
            attrs=da.attrs,
        )

        return result

    def interpolate_to_cartesian(self, *args, **kwargs):
        """
        Interpolate the DataArray to a regular Cartesian grid.

        This method is intended to be used to produce data for visualisation, which
        normally does not require double-precision values, so by default the data is
        converted to `np.float32`. Pass `use_float32=False` to retain the original
        precision.

        Parameters
        ----------
        nX : int (default 300)
            Number of grid points in the X direction
        nY : int (default 300)
            Number of grid points in the Y direction
        nZ : int (default 100)
            Number of grid points in the Z direction
        use_float32 : bool (default True)
            Downgrade precision to `np.float32`?
        fill_value : float (default np.nan)
            Value to use for points outside the interpolation domain (passed to
            `scipy.RegularGridInterpolator`)

        See Also
        --------
        BoutDataset.interpolate_to_cartesian
        """
        da = self.data
        name = da.name
        ds = da.to_dataset()
        # Dataset needs geometry and metadata attributes, but these are not copied from
        # the DataArray by default
        ds.attrs["geometry"] = da.geometry
        ds.attrs["metadata"] = da.metadata
        return ds.bout.interpolate_to_cartesian(*args, **kwargs)[name]

    # BOUT-specific plotting functionality: methods that plot on a poloidal (R-Z) plane
    def contour(self, ax=None, **kwargs):
        """
        Contour-plot a radial-poloidal slice on the R-Z plane
        """
        return plotfuncs.plot2d_wrapper(self.data, xr.plot.contour, ax=ax, **kwargs)

    def contourf(self, ax=None, **kwargs):
        """
        Filled-contour-plot a radial-poloidal slice on the R-Z plane
        """
        return plotfuncs.plot2d_wrapper(self.data, xr.plot.contourf, ax=ax, **kwargs)

    def pcolormesh(self, ax=None, **kwargs):
        """
        Colour-plot a radial-poloidal slice on the R-Z plane
        """
        return plotfuncs.plot2d_wrapper(self.data, xr.plot.pcolormesh, ax=ax, **kwargs)

    def plot_regions(self, ax=None, **kwargs):
        """
        Plot the regions into which xBOUT splits radial-poloidal arrays to handle
        tokamak topology.
        """
        return plotfuncs.plot_regions(self.data, ax=ax, **kwargs)
