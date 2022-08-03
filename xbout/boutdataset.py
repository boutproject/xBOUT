import collections
from copy import copy
from pprint import pformat as prettyformat
from functools import partial
from itertools import chain
from pathlib import Path
import warnings
import gc

import xarray as xr
import animatplot as amp
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter

from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from dask.diagnostics import ProgressBar

from .geometries import apply_geometry
from .plotting.animate import (
    animate_poloidal,
    animate_pcolormesh,
    animate_line,
    _add_controls,
    _normalise_time_coord,
    _parse_coord_option,
)
from .region import _from_region
from .utils import (
    _add_cartesian_coordinates,
    _get_bounding_surfaces,
    _split_into_restarts,
)


@xr.register_dataset_accessor("bout")
class BoutDatasetAccessor:
    """
    Contains BOUT-specific methods to use on BOUT++ datasets opened using
    `open_boutdataset()`.

    These BOUT-specific methods and attributes are accessed via the bout
    accessor, e.g. `ds.bout.options` returns a `BoutOptionsFile` instance.
    """

    def __init__(self, ds):
        self.data = ds
        self.metadata = ds.attrs.get("metadata")  # None if just grid file
        self.options = ds.attrs.get("options")  # None if no inp file

    def __str__(self):
        """
        String representation of the BoutDataset.

        Accessed by print(ds.bout)
        """

        styled = partial(prettyformat, indent=4, compact=True)
        text = (
            "<xbout.BoutDataset>\n"
            + "Contains:\n{}\n".format(str(self.data))
            + "Metadata:\n{}\n".format(styled(self.metadata))
        )
        if self.options:
            text += "Options:\n{}".format(self.options)
        return text

    # def __repr__(self):
    #    return 'boutdata.BoutDataset(', {}, ',', {}, ')'.format(self.datapath,
    #  self.prefix)

    def get_field_aligned(self, name, caching=True):
        """
        Get a field-aligned version of a variable, calculating (and caching in the
        Dataset) if necessary

        Parameters
        ----------
        name : str
            Name of the variable to get field-aligned version of
        caching : bool, optional
            Save the field-aligned variable in the Dataset (default: True)
        """
        aligned_name = name + "_aligned"
        try:
            result = self.data[aligned_name]
            if result.direction_y != "Aligned":
                raise ValueError(
                    aligned_name + " exists, but is not field-aligned, it "
                    "has direction_y=" + result.direction_y
                )
            return result
        except KeyError:
            if caching:
                self.data[aligned_name] = self.data[name].bout.to_field_aligned()
            return self.data[aligned_name]

    def to_field_aligned(self):
        """
        Create a new Dataset with all 3d variables transformed to field-aligned
        coordinates, which are shifted with respect to the base coordinates by an angle
        zShift
        """
        result = self.data.copy()

        for v in chain(result, result.coords):
            da = result[v]
            # Need to transform any z-dependent variables or coordinates
            if (result.metadata["bout_zdim"] in da.dims) and (
                da.attrs.get("direction_y", None) == "Standard"
            ):
                result[v] = da.bout.to_field_aligned()

        return result

    def from_field_aligned(self):
        """
        Create a new Dataset with all 3d variables transformed to non-field-aligned
        coordinates
        """
        result = self.data.copy()

        for v in chain(result, result.coords):
            da = result[v]
            # Need to transform any 3d variables or coordinates
            if (result.metadata["bout_zdim"] in da.dims) and (
                da.attrs.get("direction_y", None) == "Aligned"
            ):
                result[v] = da.bout.from_field_aligned()

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
        return _from_region(self.data, name, with_guards)

    @property
    def _regions(self):
        if "regions" not in self.data.attrs:
            raise ValueError(
                "Called a method requiring regions, but these have not been created. "
                "Please set the 'geometry' option when calling open_boutdataset() to "
                "create regions."
            )
        return self.data.attrs["regions"]

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
        ds = self.data
        ds.metadata["fine_interpolation_factor"] = n
        for da in ds.data_vars.values():
            da.metadata["fine_interpolation_factor"] = n

    def interpolate_parallel(self, variables, **kwargs):
        """
        Interpolate in the parallel direction to get a higher resolution version of a
        subset of variables.

        Note that the high-resolution variables are all loaded into memory, so most
        likely it is necessary to select only a small number. The toroidal_points
        argument can also be used to reduce the memory demand.

        Parameters
        ----------
        variables : str or sequence of str or ...
            The names of the variables to interpolate. If 'variables=...' is passed
            explicitly, then interpolate all variables in the Dataset.
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

        Returns
        -------
        A new Dataset containing a high-resolution versions of the variables. The new
        Dataset is a valid BoutDataset, although containing only the specified variables.
        """

        if variables is ...:
            variables = [v for v in self.data]

        if isinstance(variables, str):
            variables = [variables]
        if isinstance(variables, tuple):
            variables = list(variables)

        # Need to start with a Dataset with attrs as merge() drops the attrs of the
        # passed-in argument.
        # Make sure the first variable has all dimensions so we don't lose any
        # coordinates
        def find_with_dims(first_var, dims):
            if first_var is None:
                dims = set(dims)
                for v in variables:
                    if set(self.data[v].dims) == dims:
                        first_var = v
                        break
            return first_var

        tcoord = self.data.metadata.get("bout_tdim", "t")
        zcoord = self.data.metadata.get("bout_zdim", "z")
        first_var = find_with_dims(None, self.data.dims)
        first_var = find_with_dims(first_var, set(self.data.dims) - set(tcoord))
        first_var = find_with_dims(first_var, set(self.data.dims) - set(zcoord))
        first_var = find_with_dims(
            first_var, set(self.data.dims) - set([tcoord, zcoord])
        )
        if first_var is None:
            raise ValueError(
                f"Could not find variable to interpolate with both "
                f"{self.data.metadata.get('bout_xdim', 'x')} and "
                f"{self.data.metadata.get('bout_ydim', 'y')} dimensions"
            )
        variables.remove(first_var)
        ds = self.data[first_var].bout.interpolate_parallel(
            return_dataset=True, **kwargs
        )
        xcoord = ds.metadata.get("bout_xdim", "x")
        ycoord = ds.metadata.get("bout_ydim", "y")
        for var in variables:
            da = self.data[var]
            if xcoord in da.dims and ycoord in da.dims:
                ds = ds.merge(
                    da.bout.interpolate_parallel(return_dataset=True, **kwargs)
                )
            elif ycoord not in da.dims:
                ds[var] = da
            # Can't interpolate a variable that depends on y but not x, so just skip

        # Apply geometry
        ds = apply_geometry(ds, ds.geometry)

        return ds

    def integrate_midpoints(self, variable, *, dims=None, cumulative_t=False):
        """
        Integrate using the midpoint rule for spatial dimensions, and trapezium rule for
        time.

        The quantity being integrated is assumed to be a scalar variable.

        When doing a 1d integral in the 'y' dimension, the integral is calculated as a
        poloidal integral if the variable is on the standard grid ('direction_y'
        attribute is "Standard"), or as a parallel-to-B integral if the variable is on
        the field-aligned grid ('direction_y' attribute is "Aligned").

        When doing a 2d integral over 'x' and 'y' dimensions, the integral will be over
        poloidal cross-sections if the variable is not field-aligned (direction_y ==
        "Standard") and over field-aligned surfaces if the variable is field-aligned
        (direction_ == "Aligned"). The latter seems unlikely to be useful as the
        surfaces depend on the arbitrary origin used for zShift.

        Is a method of BoutDataset accessor rather than of BoutDataArray so we can use
        other variables like `J`, `g11`, `g_22` for the integration.

        Note the xarray.DataArray.integrate() method uses the trapezium rule, which is
        not consistent with the way BOUT++ defines grid spacings as cell widths. Also,
        this way for example::

            inner = da.isel(x=slice(i)).bout.integrate_midpoints()
            outer = da.isel(x=slice(i, None).bout.integrate_midpoints()
            total = da.bout.integrate_midpoints()
            inner + outer == total

        while with the trapezium rule you would have to select ``radial=slice(i+1)`` for
        inner to get a similar relation to be true.

        Parameters
        ----------
        variable : str or DataArray
            Name of the variable to integrate, or the variable itself as a DataArray.
        dims : str, list of str or ...
            Dimensions to integrate over. Can be any combination of of the dimensions of
            the Dataset. Defaults to integration over all spatial dimensions. If `...`
            is passed, integrate over all dimensions including time.
        cumulative_t : bool, default False
            If integrating in time, return the cumulative integral (integral from the
            beginning up to each point in the time dimension) instead of the definite
            integral.
        """
        ds = self.data

        if isinstance(variable, str):
            variable = ds[variable]

        location = variable.cell_location
        suffix = "" if location == "CELL_CENTRE" else f"_{location}"

        tcoord = ds.metadata["bout_tdim"]
        xcoord = ds.metadata["bout_xdim"]
        ycoord = ds.metadata["bout_ydim"]
        zcoord = ds.metadata["bout_zdim"]

        if dims is None:
            dims = []
            if xcoord in ds.dims:
                dims.append(xcoord)
            if ycoord in ds.dims:
                dims.append(ycoord)
            if zcoord in ds.dims:
                dims.append(zcoord)
        elif dims is ...:
            dims = []
            if tcoord in ds.dims:
                dims.append(tcoord)
            if xcoord in ds.dims:
                dims.append(xcoord)
            if ycoord in ds.dims:
                dims.append(ycoord)
            if zcoord in ds.dims:
                dims.append(zcoord)
        elif isinstance(dims, str):
            dims = [dims]

        dx = ds[f"dx{suffix}"]
        dy = ds[f"dy{suffix}"]
        if ds.metadata["BOUT_VERSION"] >= 5.0:
            dz = ds[f"dz{suffix}"]
        else:
            dz = ds["dz"]

        # Work out the spatial volume element
        if xcoord in dims and ycoord in dims and zcoord in dims:
            # Volume integral, use the 3d Jacobian "J"
            spatial_volume_element = ds[f"J{suffix}"] * dx * dy * dz
        elif xcoord in dims and ycoord in dims:
            # 2d integral on poloidal planes
            if variable.direction_y == "Standard":
                # Need to use a metric constructed from basis vectors within the
                # poloidal plane, so use 'reciprocal basis vectors' Grad(x^i)
                # J = 1/sqrt(det(g_2d))
                # det(g_2d) = g11*g22 - g12**2
                g = ds[f"g11{suffix}"] * ds[f"g22{suffix}"] - ds[f"g12{suffix}"] ** 2
                J = 1.0 / np.sqrt(g)
            elif variable.direction_y == "Aligned":
                # Need to work out area element from metric coefficients. See book by
                # D'haeseleer, Hitchon, Callen and Shohet eq. (2.5.51).
                # Need to use a metric constructed from basis vectors within the
                # field-aligned x-y plane, so use 'tangent basis vectors' e_i
                # J = sqrt(g_11*g_22 - g_12**2)
                J = np.sqrt(
                    ds[f"g_11{suffix}"] * ds[f"g_22{suffix}"] - ds[f"g_12{suffix}"] ** 2
                )
            spatial_volume_element = J * dx * dy
        elif xcoord in dims and zcoord in dims:
            # 2d integral on toroidal planes
            # Need to work out area element from metric coefficients. See book by
            # D'haeseleer, Hitchon, Callen and Shohet eq. (2.5.51)
            # J = sqrt(g_11*g_33 - g_13**2)
            J = np.sqrt(
                ds[f"g_11{suffix}"] * ds[f"g_33{suffix}"] - ds[f"g_13{suffix}"] ** 2
            )
            spatial_volume_element = J * dx * dz
        elif ycoord in dims and zcoord in dims:
            # 2d integral on flux-surfaces
            # Need to work out area element from metric coefficients. See book by
            # D'haeseleer, Hitchon, Callen and Shohet eq. (2.5.51)
            # J = sqrt(g_22*g_33 - g_23**2)
            J = np.sqrt(
                ds[f"g_22{suffix}"] * ds[f"g_33{suffix}"] - ds[f"g_23{suffix}"] ** 2
            )
            spatial_volume_element = J * dy * dz
        elif xcoord in dims:
            if variable.direction_y == "Aligned":
                raise ValueError(
                    "Variable is field-aligned, but radial integral along coordinate "
                    "line in globally field-aligned coordinates not supported"
                )
            # 1d radial integral, line element is sqrt(g_11)*dx
            spatial_volume_element = np.sqrt(ds[f"g_11{suffix}"]) * dx
        elif ycoord in dims:
            if variable.direction_y == "Standard":
                # Poloidal integral, line element is e_y projected onto a unit vector in
                # the poloidal direction. e_z is in the toroidal direction and Grad(x)
                # is orthogonal to flux surfaces, so their cross product is in the
                # poloidal direction (within flux surfaces). e_z and Grad(x) are also
                # always orthogonal, so the magnitude of their cross product is the
                # product of their magnitudes. Therefore
                #   e_y.hat{e}_pol = e_y.(e_z x Grad(x))/|Grad(x)||e_z|
                #   e_y.hat{e}_pol = e_y.(e_z x Grad(x))/sqrt(g11*g_33)
                # and using eqs. (2.3.12) and (2.5.22a) from D'haeseleer
                #   e_y.hat{e}_pol = e_y.(e_z x (e_y x e_z / J))/sqrt(g11*g_33)
                #   e_y.hat{e}_pol = e_y.(e_z x (e_y x e_z))/ (J*sqrt(g11*g_33))
                # The double cross product identity is A x (B x C) = (A.C)B - (A.B)C.
                #   e_y.hat{e}_pol = e_y.((e_z.e_z)*e_y - (e_z.e_y)*e_z)/(J*sqrt(g11*g_33))
                #   e_y.hat{e}_pol = e_y.(g_33*e_y - g_23*e_z)/(J*sqrt(g11*g_33))
                #   e_y.hat{e}_pol = (g_33*g_22 - g_23*g_23)/(J*sqrt(g11*g_33))
                # For 'orthogonal' coordinates (radial and poloidal directions are
                # orthogonal) this is equal to 1/sqrt(g22)
                spatial_volume_element = (
                    (
                        ds[f"g_22{suffix}"] * ds[f"g_33{suffix}"]
                        - ds[f"g_23{suffix}"] ** 2
                    )
                    / (
                        ds[f"J{suffix}"]
                        * np.sqrt(ds[f"g11{suffix}"] * ds[f"g_33{suffix}"])
                    )
                    * dy
                )
            elif variable.direction_y == "Aligned":
                # Parallel integral, line element is sqrt(g_22)*dy
                spatial_volume_element = np.sqrt(ds[f"g_22{suffix}"]) * dy
        elif zcoord in dims:
            # Toroidal integral, line element is sqrt(g_33)*dz
            spatial_volume_element = np.sqrt(ds[f"g_33{suffix}"]) * dz
        else:
            # No spatial integral
            spatial_volume_element = 1.0

        spatial_dims = set(dims) - set([tcoord])

        integrand = variable * spatial_volume_element

        # Need to check if the variable being integrated is a Field2D, which does not
        # have a z-dimension to sum over. Other variables are OK because metric
        # coefficients, dx and dy all have both x- and y-dimensions so variable would be
        # broadcast to include them if necessary
        missing_z_sum = zcoord in dims and zcoord not in variable.dims

        # If integrand is a Field2D, need to multiply by nz if integrating over z
        if missing_z_sum:
            spatial_dims -= set(zcoord)
            integral = integrand.sum(dim=spatial_dims)
            integral = integral * ds.sizes[zcoord]
        else:
            integral = integrand.sum(dim=spatial_dims)

        if tcoord in dims:
            if cumulative_t:
                integral = integral.cumulative_integrate(coord=tcoord)
            else:
                integral = integral.integrate(coord=tcoord)

        return integral

    def interpolate_from_unstructured(
        self,
        variables,
        *,
        fill_value=np.nan,
        structured_output=True,
        unstructured_dim_name="unstructured_dim",
        **kwargs,
    ):
        """Interpolate Dataset onto new grids of some existing coordinates

        Parameters
        ----------
        variables : str or sequence of str or ...
            The names of the variables to interpolate. If 'variables=...' is passed
            explicitly, then interpolate all variables in the Dataset.
        **kwargs : (str, array)
            Each keyword is the name of a coordinate in the DataArray, the argument is a
            1d array giving the values of that coordinate on the output grid
        fill_value : float
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
        Dataset
            Dataset interpolated onto a new, structured grid
        """

        if variables is ...:
            variables = [v for v in self.data]
            explicit_variables_arg = False
        else:
            explicit_variables_arg = True

        if isinstance(variables, str):
            variables = [variables]
        if isinstance(variables, tuple):
            variables = list(variables)

        coords_to_interpolate = []
        for coord in self.data.coords:
            if coord not in variables and coord not in kwargs:
                coords_to_interpolate.append(coord)

        ds = xr.Dataset()

        for v in variables + coords_to_interpolate:
            if np.all([c in self.data[v].coords for c in kwargs]):
                ds = ds.merge(
                    self.data[v]
                    .bout.interpolate_from_unstructured(
                        fill_value=fill_value,
                        structured_output=structured_output,
                        unstructured_dim_name=unstructured_dim_name,
                        **kwargs,
                    )
                    .to_dataset()
                )
            elif explicit_variables_arg and v in variables:
                # User explicitly requested v to be interpolated
                raise ValueError(
                    f"Could not interpolate {v} because it does not depend on all "
                    f"coordinates {[c for c in kwargs]}"
                )
            elif v in coords_to_interpolate:
                coords_to_interpolate.remove(v)

        ds = ds.set_coords(coords_to_interpolate)

        return ds

    def interpolate_to_cartesian(
        self, nX=300, nY=300, nZ=100, *, use_float32=True, fill_value=np.nan
    ):
        """
        Interpolate the Dataset to a regular Cartesian grid.

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
        BoutDataArray.interpolate_to_cartesian
        """
        ds = self.data
        ds = ds.bout.add_cartesian_coordinates()

        if not isinstance(use_float32, bool):
            raise ValueError(f"use_float32 must be a bool, got '{use_float32}'")
        if use_float32:
            float_type = np.float32
            ds = ds.astype(float_type)
            for coord in ds.coords:
                # Coordinates are not converted by Dataset.astype, so convert explicitly
                ds[coord] = ds[coord].astype(float_type)
            fill_value = float_type(fill_value)
        else:
            float_type = ds[ds.data_vars[0]].dtype

        tdim = ds.metadata["bout_tdim"]
        zdim = ds.metadata["bout_zdim"]
        if tdim in ds.dims:
            nt = ds.sizes[tdim]
        n_toroidal = ds.sizes[zdim]

        # Create Cartesian grid to interpolate to
        Xmin = ds["X_cartesian"].min()
        Xmax = ds["X_cartesian"].max()
        Ymin = ds["Y_cartesian"].min()
        Ymax = ds["Y_cartesian"].max()
        Zmin = ds["Z_cartesian"].min()
        Zmax = ds["Z_cartesian"].max()
        newX_1d = xr.DataArray(np.linspace(Xmin, Xmax, nX), dims="X")
        newX = newX_1d.expand_dims({"Y": nY, "Z": nZ}, axis=[1, 2])
        newY_1d = xr.DataArray(np.linspace(Ymin, Ymax, nY), dims="Y")
        newY = newY_1d.expand_dims({"X": nX, "Z": nZ}, axis=[0, 2])
        newZ_1d = xr.DataArray(np.linspace(Zmin, Zmax, nZ), dims="Z")
        newZ = newZ_1d.expand_dims({"X": nX, "Y": nY}, axis=[0, 1])
        newR = np.sqrt(newX**2 + newY**2)
        newzeta = np.arctan2(newY, newX)
        # Define newzeta in range 0->2*pi
        newzeta = np.where(newzeta < 0.0, newzeta + 2.0 * np.pi, newzeta)

        from scipy.interpolate import (
            RegularGridInterpolator,
            griddata,
        )

        # Create Cylindrical coordinates for intermediate grid
        Rcyl_min = float_type(ds["R"].min())
        Rcyl_max = float_type(ds["R"].max())
        Zcyl_min = float_type(ds["Z"].min())
        Zcyl_max = float_type(ds["Z"].max())
        n_Rcyl = int(round(nZ * (Rcyl_max - Rcyl_min) / (Zcyl_max - Zcyl_min)))
        Rcyl = xr.DataArray(np.linspace(Rcyl_min, Rcyl_max, 2 * n_Rcyl), dims="r")
        Zcyl = xr.DataArray(np.linspace(Zcyl_min, Zcyl_max, 2 * nZ), dims="z")

        # Create Dataset for result
        result = xr.Dataset()
        result.attrs["metadata"] = ds.metadata

        # Interpolate in two stages for efficiency. Unstructured 3d interpolation is
        # very slow. Unstructured 2d interpolation onto Cartesian (R, Z) grids, followed
        # by structured 3d interpolation onto the (X, Y, Z) grid, is much faster.
        # Structured 3d interpolation straight from (psi, theta, zeta) to (X, Y, Z)
        # leaves artifacts in the output, because theta does not vary continuously
        # everywhere (has branch cuts).

        zeta_out = np.zeros(n_toroidal + 1)
        zeta_out[:-1] = ds[zdim].values
        zeta_out[-1] = zeta_out[-2] + ds["dz"].mean()

        def interp_single_time(da):
            print("    interpolate poloidal planes")

            da_cyl = da.bout.interpolate_from_unstructured(R=Rcyl, Z=Zcyl).transpose(
                "R", "Z", zdim, missing_dims="ignore"
            )

            if zdim not in da_cyl.dims:
                da_cyl = da_cyl.expand_dims({zdim: n_toroidal + 1}, axis=-1)
            else:
                # Impose toroidal periodicity by appending zdim=0 to end of array
                da_cyl = xr.concat((da_cyl, da_cyl.isel({zdim: 0})), zdim)

            print("    build 3d interpolator")
            interp = RegularGridInterpolator(
                (Rcyl.values, Zcyl.values, zeta_out),
                da_cyl.values,
                bounds_error=False,
                fill_value=fill_value,
            )

            print("    do 3d interpolation")
            return interp(
                (newR, newZ, newzeta),
                method="linear",
            )

        for name, da in ds.data_vars.items():
            print(f"\ninterpolating {name}")
            # order of dimensions does not really matter here - output only depends on
            # shape of newR, newZ, newzeta. Possibly more efficient to assign the 2d
            # results in the loop to the last two dimensions, so put zeta first.  Can't
            # just use da.min().item() here (to get a scalar value instead of a
            # zero-size array) because .item() doesn't work for dask arrays (yet!).

            datamin = float_type(da.min().values)
            datamax = float_type(da.max().values)

            if tdim in da.dims:
                data_cartesian = np.zeros((nt, nX, nY, nZ), dtype=float_type)
                for tind in range(nt):
                    print(f"  tind={tind}")
                    data_cartesian[tind, :, :, :] = interp_single_time(
                        da.isel({tdim: tind})
                    )
                result[name] = xr.DataArray(data_cartesian, dims=[tdim, "X", "Y", "Z"])
            else:
                data_cartesian = interp_single_time(da)
                result[name] = xr.DataArray(data_cartesian, dims=["X", "Y", "Z"])

            # Copy metadata to data variables, in case it is needed
            result[name].attrs["metadata"] = ds.metadata

        result = result.assign_coords(X=newX_1d, Y=newY_1d, Z=newZ_1d)

        return result

    def add_cartesian_coordinates(self):
        """
        Add Cartesian (X,Y,Z) coordinates.

        Returns
        -------
        Dataset with new coordinates added, which are named 'X_cartesian',
        'Y_cartesian', and 'Z_cartesian'
        """
        return _add_cartesian_coordinates(self.data)

    def remove_yboundaries(self, **kwargs):
        """
        Remove y-boundary points, if present, from the Dataset
        """

        variables = []
        xcoord = self.data.metadata["bout_xdim"]
        ycoord = self.data.metadata["bout_ydim"]
        new_metadata = None
        for v in self.data:
            if xcoord in self.data[v].dims and ycoord in self.data[v].dims:
                variables.append(
                    self.data[v].bout.remove_yboundaries(return_dataset=True, **kwargs)
                )
                new_metadata = variables[-1].metadata
            elif ycoord in self.data[v].dims:
                raise ValueError(
                    f"{v} only has a {ycoord}-dimension so cannot split "
                    f"into regions."
                )
            else:
                variable = self.data[v]
                if "keep_yboundaries" in variable.metadata:
                    variable.attrs["metadata"] = copy(variable.metadata)
                    variable.metadata["keep_yboundaries"] = 0
                variables.append(variable.bout.to_dataset())
        if new_metadata is None:
            # were no 2d or 3d variables so do not have updated jyseps*, ny_inner but
            # does not matter because missing metadata is only useful for 2d or 3d
            # variables
            new_metadata = variables[0].metadata

        result = xr.merge(variables)

        result.attrs = copy(self.data.attrs)

        # Copy metadata to get possibly modified jyseps*, ny_inner, ny
        result.attrs["metadata"] = new_metadata

        if "regions" in result.attrs:
            # regions are not correct for modified BoutDataset
            del result.attrs["regions"]

        # call to re-create regions
        result = apply_geometry(result, self.data.geometry)

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

    def save(
        self,
        savepath="./boutdata.nc",
        filetype="NETCDF4",
        variables=None,
        save_dtype=None,
        separate_vars=False,
        pre_load=False,
    ):
        """
        Save data variables to a netCDF file.

        Parameters
        ----------
        savepath : str, optional
        filetype : str, optional
        variables : list of str, optional
            Variables from the dataset to save. Default is to save all of them.
        separate_vars: bool, optional
            If this is true then every variable which depends on time (but not
            solely on time) will be saved into a different output file.
            The files are labelled by the name of the variable. Variables which
            don't meet this criterion will be present in every output file.
        pre_load : bool, optional
            When saving separate variables, will load each variable into memory
            before saving to file, which can be considerably faster.

        Examples
        --------
        If `separate_vars=True`, then multiple files will be created. These can
        all be opened and merged in one go using a call of the form:

        ds = xr.open_mfdataset('boutdata_*.nc', combine='nested', concat_dim=None)
        """

        if variables is None:
            # Save all variables
            to_save = self.data
        else:
            to_save = self.data[variables]

        if savepath == "./boutdata.nc":
            print(
                "Will save data into the current working directory, named as"
                " boutdata_[var].nc"
            )
        if savepath is None:
            raise ValueError("Must provide a path to which to save the data.")

        # make shallow copy of Dataset, so we do not modify the attributes of the data
        # when we change things to save
        to_save = to_save.copy()

        options = to_save.attrs.pop("options")
        if options:
            # TODO Convert Ben's options class to a (flattened) nested
            # dictionary then store it in ds.attrs?
            warnings.warn(
                "Haven't decided how to write options file back out yet - deleting "
                "options for now. To re-load this Dataset, pass the same inputfilepath "
                "to open_boutdataset when re-loading."
            )
        # Delete placeholders for options on each variable and coordinate
        for var in chain(to_save.data_vars, to_save.coords):
            try:
                del to_save[var].attrs["options"]
            except KeyError:
                pass

        # Store the metadata as individual attributes instead because
        # netCDF can't handle storing arbitrary objects in attrs
        def dict_to_attrs(obj, section):
            for key, value in obj.attrs.pop(section).items():
                obj.attrs[section + ":" + key] = value

        dict_to_attrs(to_save, "metadata")
        # Must do this for all variables and coordinates in dataset too
        for varname, da in chain(to_save.data_vars.items(), to_save.coords.items()):
            try:
                dict_to_attrs(da, "metadata")
            except KeyError:
                pass

        if "regions" in to_save.attrs:
            # Do not need to save regions as these can be reconstructed from the metadata
            try:
                del to_save.attrs["regions"]
            except KeyError:
                pass
            for var in chain(to_save.data_vars, to_save.coords):
                try:
                    del to_save[var].attrs["regions"]
                except KeyError:
                    pass

        if save_dtype is not None:
            encoding = {v: {"dtype": save_dtype} for v in to_save}
        else:
            encoding = None

        if separate_vars:
            # Save each major variable to a different netCDF file

            # Determine which variables are "major"
            # Defined as time-dependent, but not solely time-dependent
            major_vars, minor_vars = _find_major_vars(to_save)

            print("Will save the variables {} separately".format(str(major_vars)))

            # Save each one to separate file
            # TODO perform the save in parallel with save_mfdataset?
            for major_var in major_vars:
                # Group variables so that there is only one time-dependent
                # variable saved in each file
                minor_data = [to_save[minor_var] for minor_var in minor_vars]
                single_var_ds = xr.merge([to_save[major_var], *minor_data])

                # Add the attrs back on
                single_var_ds.attrs = to_save.attrs

                if pre_load:
                    single_var_ds.load()

                # Include the name of the variable in the name of the saved
                # file
                path = Path(savepath)
                var_savepath = (
                    str(path.parent / path.stem) + "_" + str(major_var) + path.suffix
                )
                if encoding is not None:
                    var_encoding = {major_var: encoding[major_var]}
                else:
                    var_encoding = None
                print("Saving " + major_var + " data...")
                with ProgressBar():
                    single_var_ds.to_netcdf(
                        path=str(var_savepath),
                        format=filetype,
                        compute=True,
                        encoding=var_encoding,
                    )

                # Force memory deallocation to limit RAM usage
                single_var_ds.close()
                del single_var_ds
                gc.collect()
        else:
            # Save data to a single file
            print("Saving data...")
            with ProgressBar():
                to_save.to_netcdf(
                    path=savepath, format=filetype, compute=True, encoding=encoding
                )

        return

    def to_restart(
        self,
        variables=None,
        *,
        savepath=".",
        nxpe=None,
        nype=None,
        tind=-1,
        prefix="BOUT.restart",
        overwrite=False,
    ):
        """
        Write out a timestep as a set of netCDF BOUT.restart files.

        If processor decomposition is not specified then data will be saved
        using the decomposition it had when loaded.

        Parameters
        ----------
        variables : str or sequence of str, optional
            The evolving variables needed in the restart files. If not given explicitly,
            all time-evolving variables in the Dataset will be used, which may result in
            larger restart files than necessary. If there is no time-dimension in the
            Dataset (e.g. if it was loaded from restart files), then all variables will
            be added if this argument is not given explicitly.
        savepath : str, default '.'
            Directory to save the created restart files under
        nxpe : int, optional
            Number of processors in the x-direction. If not given, keep the number used
            for the original simulation
        nype : int, optional
            Number of processors in the y-direction. If not given, keep the number used
            for the original simulation
        tind : int, default -1
            Time-index of the slice to write to the restart files. Note, when creating
            restart files from 'dump' files it is recommended to open the Dataset using
            the full time range and use the `tind` argument here, rather than selecting
            a time point manually, so that the calculation of `hist_hi` in the output
            can be correct (which requires knowing the existing value of `hist_hi`
            (output step count at the end of the simulation), `tind` and the total
            number of time points in the current output data).
        prefix : str, default "BOUT.restart"
            Prefix to use for names of restart files
        overwrite : bool, default False
            By default, raises if restart file already exists. Set to True to overwrite
            existing files
        """

        if isinstance(variables, str):
            variables = [variables]

        # Set processor decomposition if not given
        if nxpe is None:
            nxpe = self.metadata["NXPE"]
        if nype is None:
            nype = self.metadata["NYPE"]

        # Is this even possible without saving the guard cells?
        # Can they be recreated?
        restart_datasets, paths = _split_into_restarts(
            self.data,
            variables,
            savepath,
            nxpe,
            nype,
            tind,
            prefix,
            overwrite,
        )

        with ProgressBar():
            xr.save_mfdataset(restart_datasets, paths, compute=True)

    def animate_list(
        self,
        variables,
        animate_over=None,
        save_as=None,
        show=False,
        fps=10,
        nrows=None,
        ncols=None,
        poloidal_plot=False,
        axis_coords=None,
        subplots_adjust=None,
        vmin=None,
        vmax=None,
        logscale=None,
        titles=None,
        aspect=None,
        extend=None,
        controls="both",
        tight_layout=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        variables : list of str or BoutDataArray
            The variables to plot. For any string passed, the corresponding
            variable in this DataSet is used - then the calling DataSet must
            have only 3 dimensions. It is possible to pass BoutDataArrays to
            allow more flexible plots, e.g. with different variables being
            plotted against different axes.
        animate_over : str, optional
            Dimension over which to animate, defaults to the time dimension
        save_as : str, optional
            If passed, a gif is created with this filename
        show : bool, optional
            Call pyplot.show() to display the animation
        fps : float, optional
            Indicates the number of frames per second to play
        nrows : int, optional
            Specify the number of rows of plots
        ncols : int, optional
            Specify the number of columns of plots
        poloidal_plot : bool or sequence of bool, optional
            If set to True, make all 2D animations in the poloidal plane instead of using
            grid coordinates, per variable if sequence is given
        axis_coords : None, str, dict or list of None, str or dict
            Coordinates to use for axis labelling.
            - None: Use the dimension coordinate for each axis, if it exists.
            - "index": Use the integer index values.
            - dict: keys are dimension names, values set axis_coords for each axis
              separately. Values can be: None, "index", the name of a 1d variable or
              coordinate (which must have the dimension given by 'key'), or a 1d
              numpy array, dask array or DataArray whose length matches the length of
              the dimension given by 'key'.
            Only affects time coordinate for plots with poloidal_plot=True.
            If a list is passed, it must have the same length as 'variables' and gives
            the axis_coords setting for each plot individually.
            The setting to use for the 'animate_over' coordinate can be passed in one or
            more dict values, but must be the same in all dicts if given more than once.
        subplots_adjust : dict, optional
            Arguments passed to fig.subplots_adjust()()
        vmin : float or sequence of floats
            Minimum value for color scale, per variable if a sequence is given
        vmax : float or sequence of floats
            Maximum value for color scale, per variable if a sequence is given
        logscale : bool or float, sequence of bool or float, optional
            If True, default to a logarithmic color scale instead of a linear one.
            If a non-bool type is passed it is treated as a float used to set the linear
            threshold of a symmetric logarithmic scale as
            linthresh=min(abs(vmin),abs(vmax))*logscale, defaults to 1e-5 if True is
            passed.
            Per variable if sequence is given.
        titles : sequence of str or None, optional
            Custom titles for each plot. Pass None in the sequence to use the default for
            a certain variable
        aspect : str or None, or sequence of str or None, optional
            Argument to set_aspect() for each plot. Defaults to "equal" for poloidal
            plots and "auto" for others.
        extend : str or None, optional
            Passed to fig.colorbar()
        controls : string or None, default "both"
            By default, add both the timeline and play/pause toggle to the animation. If
            "timeline" is passed add only the timeline, if "toggle" is passed add only
            the play/pause toggle. If None or an empty string is passed, add neither.
        tight_layout : bool or dict, optional
            If set to False, don't call tight_layout() on the figure.
            If a dict is passed, the dict entries are passed as arguments to
            tight_layout()
        **kwargs : dict, optional
            Additional keyword arguments are passed on to each animation function, per
            variable if a sequence is given.

        Returns
        -------
        animation
            An animatplot.Animation object.
        """

        if animate_over is None:
            animate_over = self.metadata.get("bout_tdim", "t")

        nvars = len(variables)

        if nrows is None and ncols is None:
            ncols = int(np.ceil(np.sqrt(nvars)))
            nrows = int(np.ceil(nvars / ncols))
        elif nrows is None:
            nrows = int(np.ceil(nvars / ncols))
        elif ncols is None:
            ncols = int(np.ceil(nvars / nrows))
        else:
            if nrows * ncols < nvars:
                raise ValueError("Not enough rows*columns to fit all variables")

        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
        axes = axes.flatten()

        ncells = nrows * ncols

        if nvars < ncells:
            for index in range(ncells - nvars):
                fig.delaxes(axes[ncells - index - 1])

        if subplots_adjust is not None:
            fig.subplots_adjust(**subplots_adjust)

        def _expand_list_arg(arg, arg_name):
            if isinstance(arg, collections.abc.Sequence) and not isinstance(arg, str):
                if len(arg) != len(variables):
                    raise ValueError(
                        "if %s is a sequence, it must have the same "
                        'number of elements as "variables"' % arg_name
                    )
            else:
                arg = [arg] * len(variables)
            return arg

        poloidal_plot = _expand_list_arg(poloidal_plot, "poloidal_plot")
        vmin = _expand_list_arg(vmin, "vmin")
        vmax = _expand_list_arg(vmax, "vmax")
        logscale = _expand_list_arg(logscale, "logscale")
        titles = _expand_list_arg(titles, "titles")
        aspect = _expand_list_arg(aspect, "aspect")
        extend = _expand_list_arg(extend, "extend")
        axis_coords = _expand_list_arg(axis_coords, "axis_coords")
        for k in kwargs:
            kwargs[k] = _expand_list_arg(kwargs[k], k)

        blocks = []

        def is_list(variable):
            return (
                isinstance(variable, list)
                or isinstance(variable, tuple)
                or isinstance(variable, set)
            )

        for i, subplot_args in enumerate(
            zip(
                variables,
                axes,
                poloidal_plot,
                axis_coords,
                vmin,
                vmax,
                logscale,
                titles,
                aspect,
                extend,
            )
        ):

            (
                v,
                ax,
                this_poloidal_plot,
                this_axis_coords,
                this_vmin,
                this_vmax,
                this_logscale,
                this_title,
                this_aspect,
                this_extend,
            ) = subplot_args

            this_kwargs = {k: v[i] for k, v in kwargs.items()}

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            if is_list(v):
                for i in range(len(v)):
                    if isinstance(v[i], str):
                        v[i] = self.data[v[i]]
                # list of variables for one subplot only supported for line plots with 1
                # dimension plus time
                ndims = 2
                dims = v[0].dims
                if len(dims) != 2:
                    raise ValueError(
                        "Variables in sublist must be 2d - can only overlay line plots"
                    )
                for w in v:
                    if not w.dims == dims:
                        raise ValueError(
                            f"All variables in sub-list must have same dimensions."
                            f"{v[0].name} had {v[0].dims} but {w.name} had {w.dims}."
                        )
            else:
                if isinstance(v, str):
                    v = self.data[v]

                data = v.bout.data
                ndims = len(data.dims)
                ax.set_title(data.name)

            if ndims == 2:
                if not is_list(v):
                    blocks.append(
                        animate_line(
                            data=data,
                            ax=ax,
                            animate_over=animate_over,
                            animate=False,
                            axis_coords=this_axis_coords,
                            aspect=this_aspect,
                            vmin=this_vmin,
                            vmax=this_vmax,
                            **this_kwargs,
                        )
                    )
                else:
                    for w in v:
                        blocks.append(
                            animate_line(
                                data=w,
                                ax=ax,
                                animate_over=animate_over,
                                animate=False,
                                axis_coords=this_axis_coords,
                                aspect=this_aspect,
                                vmin=this_vmin,
                                vmax=this_vmax,
                                label=w.name,
                                **this_kwargs,
                            )
                        )
                    legend = ax.legend()
                    legend.set_draggable(True)
                    # set 'v' to use for the timeline below
                    v = v[0]
            elif ndims == 3:
                if this_poloidal_plot:
                    var_blocks = animate_poloidal(
                        data,
                        ax=ax,
                        cax=cax,
                        animate_over=animate_over,
                        animate=False,
                        axis_coords=this_axis_coords,
                        vmin=this_vmin,
                        vmax=this_vmax,
                        logscale=this_logscale,
                        aspect=this_aspect,
                        extend=this_extend,
                        **this_kwargs,
                    )
                    for block in var_blocks:
                        blocks.append(block)
                else:
                    blocks.append(
                        animate_pcolormesh(
                            data=data,
                            ax=ax,
                            cax=cax,
                            animate_over=animate_over,
                            animate=False,
                            axis_coords=this_axis_coords,
                            vmin=this_vmin,
                            vmax=this_vmax,
                            logscale=this_logscale,
                            aspect=this_aspect,
                            extend=this_extend,
                            **this_kwargs,
                        )
                    )
            else:
                raise ValueError(
                    "Unsupported number of dimensions "
                    + str(ndims)
                    + ". Dims are "
                    + str(v.dims)
                )

            if this_title is not None:
                # Replace default title with user-specified one
                ax.set_title(this_title)

        if np.all([a == "index" for a in axis_coords]):
            time_opt = "index"
        elif np.any([isinstance(a, dict) and animate_over in a for a in axis_coords]):
            given_values = [
                a[animate_over]
                for a in axis_coords
                if isinstance(a, dict) and animate_over in a
            ]
            time_opt = given_values[0]
            if len(given_values) > 1 and not np.all(
                [v == time_opt for v in given_values[1:]]
            ):
                raise ValueError(
                    f"Inconsistent axis_coords values given for animate_over "
                    f"coordinate ({animate_over}). Got {given_values}."
                )
        else:
            time_opt = None
        time_values, time_label = _parse_coord_option(animate_over, time_opt, self.data)
        time_values, time_suffix = _normalise_time_coord(time_values)

        timeline = amp.Timeline(time_values, fps=fps, units=time_suffix)
        anim = amp.Animation(blocks, timeline)

        if tight_layout:
            if subplots_adjust is not None:
                warnings.warn(
                    "tight_layout argument to animate_list() is True, but "
                    "subplots_adjust argument is not None. subplots_adjust "
                    "is being ignored."
                )
            if not isinstance(tight_layout, dict):
                tight_layout = {}
            fig.tight_layout(**tight_layout)

        _add_controls(anim, controls, time_label)

        if save_as is not None:
            anim.save(save_as + ".gif", writer=PillowWriter(fps=fps))

        if show:
            plt.show()

        return anim


def _find_major_vars(data):
    """
    Splits data into those variables likely to require a lot of storage space
    (defined as those which depend on time and at least one other dimension).
    These are normally the variables of physical interest.
    """

    # TODO Use an Ordered Set instead to preserve order of variables in files?
    tcoord = data.attrs.get("metadata:bout_tdim", "t")
    major_vars = set(
        var
        for var in data.data_vars
        if (tcoord in data[var].dims) and data[var].dims != (tcoord,)
    )
    minor_vars = set(data.data_vars) - set(major_vars)
    return list(major_vars), list(minor_vars)
