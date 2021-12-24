from warnings import warn
from textwrap import dedent

import xarray as xr
import numpy as np

from .region import Region, _create_regions_toroidal, _create_single_region
from .utils import (
    _add_attrs_to_var,
    _set_attrs_on_all_vars,
    _set_as_coord,
    _1d_coord_from_spacing,
)

REGISTERED_GEOMETRIES = {}


class UnregisteredGeometryError(Exception):
    """Error for unregistered geometry type"""


# TODO remove coordinates argument completely. All the functionality should
# instead be implemented by defining a particular geometry that does what is
# desired


def apply_geometry(ds, geometry_name, *, coordinates=None, grid=None):
    """

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset (from
    geometry_name : str
        Name under which the desired geometry function was registered
    coordinates : dict of str, optional
        Names to give the physical coordinates corresponding to 'x', 'y' and 'z'; values
        corresponding to 'x', 'y' and 'z' keys in the passed dict are used as the names
        of the dimensions. Any not passed are given default values. If not specified,
        default names are chosen.
    grid : Dataset, optional
        Dataset containing extra geometrical information not stored in the dump files
        that is needed to add coordinates for the geometry being applied. For example,
        should contain 2d arrays Rxy, Zxy and psixy for toroidal geometry.

    Returns
    -------
    updated_ds

    Raises
    ------
    UnregisteredGeometryError
    """

    if geometry_name is None or geometry_name == "":
        ds = _set_attrs_on_all_vars(ds, "geometry", "")
        updated_ds = ds
    else:
        try:
            add_geometry_coords = REGISTERED_GEOMETRIES[geometry_name]
        except KeyError:
            message = dedent(
                """{} is not a registered geometry. Inspect the global
                             variable REGISTERED_GEOMETRIES to see which geometries
                             have been registered.""".format(
                    geometry_name
                )
            )
            raise UnregisteredGeometryError(message)

        # User-registered functions may accept 'coordinates' and 'grid' arguments, but
        # do not have to as long as they are not used
        if coordinates is not None and grid is not None:
            updated_ds = add_geometry_coords(ds, coordinates=coordinates, grid=grid)
        elif coordinates is not None:
            updated_ds = add_geometry_coords(ds, coordinates=coordinates)
        elif grid is not None:
            updated_ds = add_geometry_coords(ds, grid=grid)
        else:
            updated_ds = add_geometry_coords(ds)

        # Set "geometry" attribute after adding coordinates, so that functions in
        # `REGISTERED_GEOMETRIES` can check if ds.attrs["geometry"] is already defined
        # to see if they are being applied for the first time or re-applied.
        updated_ds = _set_attrs_on_all_vars(updated_ds, "geometry", geometry_name)

    del ds

    # Set dimension names if they were not set by add_geometry_coords(). Dimensions
    # should not have been renamed without having set their names already.
    if "bout_tdim" not in updated_ds.metadata:
        if "t" in updated_ds.dims:
            updated_ds.metadata["bout_tdim"] = "t"
        else:
            raise ValueError(
                '"t" dimension was renamed, but metadata["bout_tdim"] was not set'
            )
    if "bout_xdim" not in updated_ds.metadata:
        if "x" in updated_ds.dims:
            updated_ds.metadata["bout_xdim"] = "x"
        else:
            raise ValueError(
                '"x" dimension was renamed, but metadata["bout_xdim"] was not set'
            )
    if "bout_ydim" not in updated_ds.metadata:
        if "y" in updated_ds.dims:
            updated_ds.metadata["bout_ydim"] = "y"
        else:
            raise ValueError(
                '"y" dimension was renamed, but metadata["bout_ydim"] was not set'
            )
    if "bout_zdim" not in updated_ds.metadata:
        if "z" in updated_ds.dims:
            updated_ds.metadata["bout_zdim"] = "z"
        else:
            raise ValueError(
                '"z" dimension was renamed, but metadata["bout_zdim"] was not set'
            )

    # Add global 1D coordinates
    # ######################
    # Note the global coordinates used here are defined so that they are zero at
    # the boundaries of the grid (where the grid includes all boundary cells), not
    # necessarily the physical boundaries, because constant offsets do not matter, as
    # long as these bounds are consistent with the global coordinates defined in
    # Region.__init__() (we will only use these coordinates for interpolation) and it is
    # simplest to calculate them with cumsum().
    tcoord = updated_ds.metadata["bout_tdim"]
    xcoord = updated_ds.metadata["bout_xdim"]
    ycoord = updated_ds.metadata["bout_ydim"]
    zcoord = updated_ds.metadata["bout_zdim"]

    if (tcoord not in updated_ds.coords) and (tcoord in updated_ds.dims):
        # Create the time coordinate from t_array
        # Slightly odd looking way to create coordinate ensures 'index variable' is
        # created, which using set_coords() does not (possible xarray bug?
        # https://github.com/pydata/xarray/issues/4417
        updated_ds[tcoord] = updated_ds["t_array"]
        updated_ds = updated_ds.drop_vars("t_array")

    if xcoord not in updated_ds.coords:
        # Make index 'x' a coordinate, useful for handling global indexing
        # Note we have to use the index value, not the value calculated from 'dx' because
        # 'dx' may not be consistent between different regions (e.g. core and PFR).
        # For some geometries xcoord may have already been created by
        # add_geometry_coords, in which case we do not need this.
        nx = updated_ds.dims[xcoord]

        # can't use commented out version, uncommented one works around xarray bug
        # removing attrs
        # https://github.com/pydata/xarray/issues/4415
        # https://github.com/pydata/xarray/issues/4393
        # updated_ds = updated_ds.assign_coords(**{xcoord: np.arange(nx)})
        updated_ds[xcoord] = (xcoord, np.arange(nx))

        _add_attrs_to_var(updated_ds, xcoord)

    if ycoord not in updated_ds.coords:
        ny = updated_ds.dims[ycoord]
        # dy should always be constant in x, so it is safe to convert to a 1d
        # coordinate.  [The y-coordinate has to be a 1d coordinate that labels x-z
        # slices of the grid (similarly x-coordinate is 1d coordinate that labels y-z
        # slices and z-coordinate is a 1d coordinate that labels x-y slices). A
        # coordinate might have different values in disconnected regions, but there are
        # no branch-cuts allowed in the x-direction in BOUT++ (at least for the
        # momement), so the y-coordinate has to be 1d and single-valued. Therefore
        # similarly dy has to be 1d and single-valued.]

        # calculate ycoord at the centre of each cell
        y = _1d_coord_from_spacing(updated_ds["dy"], ycoord)

        # can't use commented out version, uncommented one works around xarray bug
        # removing attrs as _1d_coord_from_spacing returns an xr.Variable
        # https://github.com/pydata/xarray/issues/4415
        # https://github.com/pydata/xarray/issues/4393
        # updated_ds = updated_ds.assign_coords(**{ycoord: y.values})
        updated_ds[ycoord] = y

        _add_attrs_to_var(updated_ds, ycoord)

    # If full data (not just grid file) then toroidal dim will be present
    if zcoord in updated_ds.dims and zcoord not in updated_ds.coords:
        # Generates a coordinate whose value is 0 on the first grid point, not dz/2, to
        # match how BOUT++ generates fields from input file expressions.
        nz = updated_ds.dims[zcoord]

        # In BOUT++ v5, dz is either a Field2D or Field3D.
        # We can use it as a 1D coordinate if it's a Field3D, _or_ if nz == 1
        bout_version = updated_ds.metadata.get("BOUT_VERSION", 4.3)
        bout_v5 = bout_version > 5.0 or (
            bout_version == 5.0 and updated_ds["dz"].ndim >= 2
        )
        use_metric_3d = updated_ds.metadata.get("use_metric_3d", False)
        can_use_1d_z_coord = (nz == 1) or use_metric_3d

        if can_use_1d_z_coord:
            z = _1d_coord_from_spacing(updated_ds["dz"], zcoord, updated_ds)
        else:
            if bout_v5:
                if not np.all(updated_ds["dz"].min() == updated_ds["dz"].max()):
                    raise ValueError(
                        f"Spacing is not constant. Cannot create z coordinate"
                    )

                dz = updated_ds["dz"].min()
            else:
                dz = updated_ds["dz"]

            z0 = 2 * np.pi * updated_ds.metadata["ZMIN"]
            z1 = z0 + nz * dz
            if not np.all(
                np.isclose(
                    z1,
                    2.0 * np.pi * updated_ds.metadata["ZMAX"],
                    rtol=1.0e-15,
                    atol=0.0,
                )
            ):
                warn(
                    f"Size of toroidal domain as calculated from nz*dz ({str(z1 - z0)}"
                    f" is not the same as 2pi*(ZMAX - ZMIN) ("
                    f"{2.*np.pi*updated_ds.metadata['ZMAX'] - z0}): using value from dz"
                )
            z = xr.DataArray(
                np.linspace(start=z0, stop=z1, num=nz, endpoint=False), dims=zcoord
            )

        # can't use commented out version, uncommented one works around xarray bug
        # removing attrs
        # https://github.com/pydata/xarray/issues/4415
        # https://github.com/pydata/xarray/issues/4393
        # updated_ds = updated_ds.assign_coords(**{zcoord: z})
        updated_ds[zcoord] = (zcoord, z.values)

        _add_attrs_to_var(updated_ds, zcoord)

    # Add dx, dy and dz as coordinates, so that they are available with BoutDataArrays
    updated_ds = _set_as_coord(updated_ds, "dx")
    updated_ds = _set_as_coord(updated_ds, "dy")
    updated_ds = _set_as_coord(updated_ds, "dz")

    return updated_ds


def register_geometry(name):
    """
    Register a new geometry type.

    Used as a decorator on a user-defined function which accepts a dataset and
    returns a new dataset, The returned dataset should have dimensions changed
    and coordinates added as appropriate for this new geometry type.

    Parameters
    ----------
    name: str
        Name under which this geometry should be registered.

    """

    # TODO if anything more needs to be done with geometries than just adding
    # coordinates once, they should be reimplemented as class inheritance

    if not isinstance(name, str):
        raise ValueError("The name of the new geometry type must be a string")

    def wrapper(add_geometry_coords):
        if name in REGISTERED_GEOMETRIES:
            warn(
                "There is aready a geometry type registered with the "
                "name {}, it will be overrided".format(name)
            )

        REGISTERED_GEOMETRIES[name] = add_geometry_coords

        return add_geometry_coords

    return wrapper


def _set_default_toroidal_coordinates(coordinates, ds):
    if coordinates is None:
        coordinates = {}

    # Replace any values that have not been passed in with defaults
    if ds.metadata["is_restart"] == 0:
        # Don't need "t" coordinate for restart files which have no time dimension, and
        # adding it breaks the check for reloading in open_boutdataset
        coordinates["t"] = coordinates.get("t", ds.metadata["bout_tdim"])

    default_x = (
        ds.metadata["bout_xdim"] if ds.metadata["bout_xdim"] != "x" else "psi_poloidal"
    )
    coordinates["x"] = coordinates.get("x", default_x)

    default_y = ds.metadata["bout_ydim"] if ds.metadata["bout_ydim"] != "y" else "theta"
    coordinates["y"] = coordinates.get("y", default_y)

    default_z = ds.metadata["bout_zdim"] if ds.metadata["bout_zdim"] != "z" else "zeta"
    coordinates["z"] = coordinates.get("z", default_z)

    return coordinates


def _add_vars_from_grid(ds, grid, variables, *, optional_variables=None):
    # Get extra geometry information from grid file if it's not in the dump files
    for v in variables:
        if v not in ds:
            if grid is None:
                raise ValueError(
                    f"Grid file is required to provide {v}. Pass the grid "
                    f"file name as the 'gridfilepath' argument to "
                    f"open_boutdataset()."
                )
            # ds[v] = grid[v]
            # Work around issue where xarray drops attributes on coordinates when a new
            # DataArray is assigned to the Dataset, see
            # https://github.com/pydata/xarray/issues/4415
            # https://github.com/pydata/xarray/issues/4393
            # This way adds as a 'Variable' instead of as a 'DataArray'
            ds[v] = (grid[v].dims, grid[v].values)

            _add_attrs_to_var(ds, v)

    if optional_variables is not None:
        for v in optional_variables:
            if v not in ds:
                if grid is None:
                    continue
                if v in grid:
                    # ds[v] = grid[v]
                    # Work around issue where xarray drops attributes on
                    # coordinates when a new DataArray is assigned to the
                    # Dataset, see https://github.com/pydata/xarray/issues/4415
                    # https://github.com/pydata/xarray/issues/4393
                    # This way adds as a 'Variable' instead of as a 'DataArray'
                    ds[v] = (grid[v].dims, grid[v].values)

                    _add_attrs_to_var(ds, v)

    return ds


@register_geometry("toroidal")
def add_toroidal_geometry_coords(ds, *, coordinates=None, grid=None):

    coordinates = _set_default_toroidal_coordinates(coordinates, ds)

    if ds.attrs.get("geometry", None) == "toroidal":
        # Loading a Dataset which already had the coordinates created for it
        ds = _create_regions_toroidal(ds)
        return ds

    # Check whether coordinates names conflict with variables in ds
    bad_names = [
        name for name in coordinates.values() if name in ds and name not in ds.coords
    ]
    if bad_names:
        raise ValueError(
            "Coordinate names {} clash with variables in the dataset. "
            "Register a different geometry to provide alternative names. "
            "It may be useful to use the 'coordinates' argument to "
            "add_toroidal_geometry_coords() for this.".format(bad_names)
        )

    # Get extra geometry information from grid file if it's not in the dump files
    ds = _add_vars_from_grid(
        ds,
        grid,
        ["psixy", "Rxy", "Zxy"],
        optional_variables=[
            "Bpxy",
            "Brxy",
            "Bzxy",
            "poloidal_distance",
            "poloidal_distance_ylow",
            "total_poloidal_distance",
            "zShift",
            "zShift_ylow",
        ],
    )

    if "t" in ds.dims:
        # Rename 't' if user requested it
        ds = ds.rename(t=coordinates["t"])

    # Change names of dimensions to Orthogonal Toroidal ones
    ds = ds.rename(y=coordinates["y"])

    # TODO automatically make this coordinate 1D in simplified cases?
    ds = ds.rename(psixy=coordinates["x"])
    ds = ds.set_coords(coordinates["x"])
    ds[coordinates["x"]].attrs["units"] = "Wb"

    # Record which dimensions 't', 'x', and 'y' were renamed to.
    if ds.metadata["is_restart"] == 0:
        ds.metadata["bout_tdim"] = coordinates["t"]
    # x dimension not renamed, so this is still 'x'
    ds.metadata["bout_xdim"] = "x"
    ds.metadata["bout_ydim"] = coordinates["y"]

    # If full data (not just grid file) then toroidal dim will be present
    if "z" in ds.dims:
        ds = ds.rename(z=coordinates["z"])

        # Record which dimension 'z' was renamed to.
        ds.metadata["bout_zdim"] = coordinates["z"]

    # Ensure metadata is the same on all variables
    ds = _set_attrs_on_all_vars(ds, "metadata", ds.metadata)

    # Add 2D Cylindrical coordinates
    if ("R" not in ds) and ("Z" not in ds):
        ds = ds.rename(Rxy="R", Zxy="Z")
        ds = ds.set_coords(("R", "Z"))
    else:
        ds = ds.set_coords(("Rxy", "Zxy"))

    # Rename zShift_ylow if it was added from grid file, to be consistent with name if
    # it was added from dump file
    if "zShift_CELL_YLOW" in ds and "zShift_ylow" in ds:
        # Remove redundant copy
        del ds["zShift_ylow"]
    elif "zShift_ylow" in ds:
        ds = ds.rename(zShift_ylow="zShift_CELL_YLOW")

    if "poloidal_distance" in ds:
        ds = ds.set_coords(
            ["poloidal_distance", "poloidal_distance_ylow", "total_poloidal_distance"]
        )

    # Add zShift as a coordinate, so that it gets interpolated along with a variable
    ds = _set_as_coord(ds, "zShift")
    if "zShift_CELL_YLOW" in ds:
        ds = _set_as_coord(ds, "zShift_CELL_YLOW")

    ds = _create_regions_toroidal(ds)

    return ds


@register_geometry("s-alpha")
def add_s_alpha_geometry_coords(ds, *, coordinates=None, grid=None):

    coordinates = _set_default_toroidal_coordinates(coordinates, ds)

    if set(coordinates.values()).issubset(set(ds.coords).union(ds.dims)):
        # Loading a Dataset which already had the coordinates created for it
        ds = _create_regions_toroidal(ds)
        return ds

    ds = add_toroidal_geometry_coords(ds, coordinates=coordinates, grid=grid)

    # Get extra geometry information from grid file if it's not in the dump files
    # Add 'hthe' from grid file, needed below for radial coordinate
    if "hthe" not in ds:
        hthe_from_grid = True
        ycoord = "y"
        if grid is None:
            raise ValueError(
                "Grid file is required to provide %s. Pass the grid "
                "file name as the 'gridfilepath' argument to "
                "open_boutdataset()."
            )
        ds["hthe"] = grid["hthe"]
        _add_attrs_to_var(ds, "hthe")
    else:
        hthe_from_grid = False
        ycoord = coordinates["y"]

    # Add 1D radial coordinate
    if "r" in ds:
        raise ValueError(
            "Cannot have variable 'r' in dataset when using " "geometry='s-alpha'"
        )
    ds["r"] = ds["hthe"].isel({ycoord: 0}).squeeze(drop=True)
    ds["r"].attrs["units"] = "m"
    ds = ds.set_coords("r")
    ds = ds.rename(x="r")
    ds.metadata["bout_xdim"] = "r"

    if hthe_from_grid:
        # remove hthe because it does not have correct metadata
        del ds["hthe"]

    return ds


@register_geometry("fci")
def add_fci_geometry_coords(ds, *, coordinates=None, grid=None):
    assert coordinates is None, "Not implemented"
    ds = _add_vars_from_grid(ds, grid, ["R", "Z"])
    ds = ds.set_coords(("R", "Z"))
    ds = _create_single_region(ds, periodic_y=True)
    return ds
