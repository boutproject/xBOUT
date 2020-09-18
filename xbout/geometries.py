from warnings import warn
from textwrap import dedent

import xarray as xr
import numpy as np

from .region import Region, _create_regions_toroidal
from .utils import _add_attrs_to_var, _set_attrs_on_all_vars, _1d_coord_from_spacing

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

    if geometry_name is None:
        updated_ds = ds
    else:
        ds = _set_attrs_on_all_vars(ds, "geometry", geometry_name)

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

    del ds

    # Add global 1D coordinates
    # ######################
    # Note the global coordinates used here are defined so that they are zero at
    # the boundaries of the grid (where the grid includes all boundary cells), not
    # necessarily the physical boundaries, because constant offsets do not matter, as
    # long as these bounds are consistent with the global coordinates defined in
    # Region.__init__() (we will only use these coordinates for interpolation) and it is
    # simplest to calculate them with cumsum().
    tcoord = updated_ds.metadata.get("bout_tdim", "t")
    xcoord = updated_ds.metadata.get("bout_xdim", "x")
    ycoord = updated_ds.metadata.get("bout_ydim", "y")
    zcoord = updated_ds.metadata.get("bout_zdim", "z")

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
        z0 = 2 * np.pi * updated_ds.metadata["ZMIN"]
        z1 = z0 + nz * updated_ds.metadata["dz"]
        if not np.isclose(
            z1, 2.0 * np.pi * updated_ds.metadata["ZMAX"], rtol=1.0e-15, atol=0.0
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
    coordinates["t"] = coordinates.get("t", ds.metadata.get("bout_tdim", "t"))
    coordinates["x"] = coordinates.get(
        "x", ds.metadata.get("bout_xdim", "psi_poloidal")
    )
    coordinates["y"] = coordinates.get("y", ds.metadata.get("bout_ydim", "theta"))
    coordinates["z"] = coordinates.get("z", ds.metadata.get("bout_zdim", "zeta"))

    return coordinates


@register_geometry("toroidal")
def add_toroidal_geometry_coords(ds, *, coordinates=None, grid=None):

    coordinates = _set_default_toroidal_coordinates(coordinates, ds)

    if set(coordinates.values()).issubset(set(ds.coords).union(ds.dims)):
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
    needed_variables = ["psixy", "Rxy", "Zxy"]
    for v in needed_variables:
        if v not in ds:
            if grid is None:
                raise ValueError(
                    "Grid file is required to provide %s. Pass the grid "
                    "file name as the 'gridfilepath' argument to "
                    "open_boutdataset()."
                )
            # ds[v] = grid[v]
            # Work around issue where xarray drops attributes on coordinates when a new
            # DataArray is assigned to the Dataset, see
            # https://github.com/pydata/xarray/issues/4415
            # https://github.com/pydata/xarray/issues/4393
            # This way adds as a 'Variable' instead of as a 'DataArray'
            ds[v] = (grid[v].dims, grid[v].values)

            _add_attrs_to_var(ds, v)

    # Rename 't' if user requested it
    ds = ds.rename(t=coordinates["t"])

    # Change names of dimensions to Orthogonal Toroidal ones
    ds = ds.rename(y=coordinates["y"])

    # TODO automatically make this coordinate 1D in simplified cases?
    ds = ds.rename(psixy=coordinates["x"])
    ds = ds.set_coords(coordinates["x"])
    ds[coordinates["x"]].attrs["units"] = "Wb"

    # Record which dimensions 't', 'x', and 'y' were renamed to.
    ds.metadata["bout_tdim"] = coordinates["t"]
    # x dimension not renamed, so this is still 'x'
    ds.metadata["bout_xdim"] = "x"
    ds.metadata["bout_ydim"] = coordinates["y"]

    # If full data (not just grid file) then toroidal dim will be present
    if "z" in ds.dims:
        ds = ds.rename(z=coordinates["z"])

        # Record which dimension 'z' was renamed to.
        ds.metadata["bout_zdim"] = coordinates["z"]

    # Add 2D Cylindrical coordinates
    if ("R" not in ds) and ("Z" not in ds):
        ds = ds.rename(Rxy="R", Zxy="Z")
        ds = ds.set_coords(("R", "Z"))
    else:
        ds = ds.set_coords(("Rxy", "Zxy"))

    # Add zShift as a coordinate, so that it gets interpolated along with a variable
    try:
        ds = ds.set_coords("zShift")
    except ValueError:
        pass
    try:
        ds = ds.set_coords("zShift_CELL_XLOW")
    except ValueError:
        pass
    try:
        ds = ds.set_coords("zShift_CELL_YLOW")
    except ValueError:
        pass
    try:
        ds = ds.set_coords("zShift_CELL_ZLOW")
    except ValueError:
        pass

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
