from warnings import warn
from textwrap import dedent

import xarray as xr
import numpy as np

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
        Dataset containing extra geometrical information not stored in the dump files that
        is needed to add coordinates for the geometry being applied. For example, should
        contain 2d arrays Rxy, Zxy and psixy for toroidal geometry.

    Returns
    -------
    updated_ds

    Raises
    ------
    UnregisteredGeometryError
    """

    try:
        add_geometry_coords = REGISTERED_GEOMETRIES[geometry_name]
    except KeyError:
        message = dedent("""{} is not a registered geometry. Inspect the global
                         variable REGISTERED_GEOMETRIES to see which geometries
                         have been registered.""".format(geometry_name))
        raise UnregisteredGeometryError(message)

    # User-registered functions may accept 'coordinates' and 'grid' arguments, but do not
    # have to as long as they are not used
    if coordinates is not None and grid is not None:
        updated_ds = add_geometry_coords(ds, coordinates=coordinates, grid=grid)
    elif coordinates is not None:
        updated_ds = add_geometry_coords(ds, coordinates=coordinates)
    elif grid is not None:
        updated_ds = add_geometry_coords(ds, grid=grid)
    else:
        updated_ds = add_geometry_coords(ds)
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
            warn("There is aready a geometry type registered with the "
                 "name {}, it will be overrided".format(name))

        REGISTERED_GEOMETRIES[name] = add_geometry_coords

        return add_geometry_coords
    return wrapper


def _set_default_toroidal_coordinates(coordinates):
    if coordinates is None:
        coordinates = {}

    # Replace any values that have not been passed in with defaults
    coordinates['x'] = coordinates.get('x', 'psi_poloidal')
    coordinates['y'] = coordinates.get('y', 'theta')
    coordinates['z'] = coordinates.get('z', 'zeta')

    return coordinates


@register_geometry('toroidal')
def add_toroidal_geometry_coords(ds, *, coordinates=None, grid=None):

    coordinates = _set_default_toroidal_coordinates(coordinates)

    # Check whether coordinates names conflict with variables in ds
    bad_names = [name for name in coordinates.values() if name in ds]
    if bad_names:
        raise ValueError("Coordinate names {} clash with variables in the dataset. "
                         "Register a different geometry to provide alternative names. "
                         "It may be useful to use the 'coordinates' argument to "
                         "add_toroidal_geometry_coords() for this.".format(bad_names))

    # Get extra geometry information from grid file if it's not in the dump files
    needed_variables = ['psixy', 'Rxy', 'Zxy']
    for v in needed_variables:
        if v not in ds:
            if grid is None:
                raise ValueError("Grid file is required to provide %s. Pass the grid "
                                 "file name as the 'gridfilepath' argument to "
                                 "open_boutdataset().")
            ds[v] = grid[v]

    # Change names of dimensions to Orthogonal Toroidal ones
    ds = ds.rename(y=coordinates['y'])

    # Add 1D Orthogonal Toroidal coordinates
    ny = ds.dims[coordinates['y']]
    theta = xr.DataArray(np.linspace(start=0, stop=2 * np.pi, num=ny),
                         dims=coordinates['y'])
    ds = ds.assign_coords(**{coordinates['y']: theta})

    # TODO automatically make this coordinate 1D in simplified cases?
    ds = ds.rename(psixy=coordinates['x'])
    ds = ds.set_coords(coordinates['x'])
    ds[coordinates['x']].attrs['units'] = 'Wb'

    # If full data (not just grid file) then toroidal dim will be present
    if 'z' in ds.dims:
        ds = ds.rename(z=coordinates['z'])
        nz = ds.dims[coordinates['z']]
        phi = xr.DataArray(np.linspace(start=0, stop=2 * np.pi, num=nz),
                           dims=coordinates['z'])
        ds = ds.assign_coords(**{coordinates['z']: phi})

    # Add 2D Cylindrical coordinates
    if ('R' not in ds) and ('Z' not in ds):
        ds = ds.rename(Rxy='R', Zxy='Z')
        ds = ds.set_coords(('R', 'Z'))
    else:
        ds.set_coords(('Rxy', 'Zxy'))

    return ds


@register_geometry('s-alpha')
def add_s_alpha_geometry_coords(ds, *, coordinates=None, grid=None):

    coordinates = _set_default_toroidal_coordinates(coordinates)

    # Add 'hthe' from grid file, needed below for radial coordinate
    if 'hthe' not in ds:
        hthe_from_grid = True
        if grid is None:
            raise ValueError("Grid file is required to provide %s. Pass the grid "
                             "file name as the 'gridfilepath' argument to "
                             "open_boutdataset().")
        ds['hthe'] = grid['hthe']
    else:
        hthe_from_grid = False

    ds = add_toroidal_geometry_coords(ds, coordinates=coordinates, grid=grid)

    # Add 1D radial coordinate
    if 'r' in ds:
        raise ValueError("Cannot have variable 'r' in dataset when using "
                         "geometry='s-alpha'")
    ds['r'] = ds['hthe'].isel({coordinates['y']: 0}).squeeze(drop=True)
    ds['r'].attrs['units'] = 'm'
    ds = ds.set_coords('r')
    ds = ds.rename(x='r')

    if hthe_from_grid:
        # remove hthe because it does not have correct metadata
        del ds['hthe']

    return ds
