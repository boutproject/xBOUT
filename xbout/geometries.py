from warnings import warn
from textwrap import dedent

import xarray as xr
import numpy as np

REGISTERED_GEOMETRIES = {}


class UnregisteredGeometryError(Exception):
    """Error for unregistered geometry type"""


def apply_geometry(ds, geometry_name, coordinates=None):
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

    updated_ds = add_geometry_coords(ds, coordinates=coordinates)
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


@register_geometry('toroidal')
def add_toroidal_geometry_coords(ds, coordinates=None):

    if coordinates is None:
        coordinates = {}

    # Replace any values that have not been passed in with defaults
    coordinates['x'] = coordinates.get('x', 'psi')
    coordinates['y'] = coordinates.get('y', 'theta')
    coordinates['z'] = coordinates.get('z', 'phi')

    # Check whether coordinates names conflict with variables in ds
    bad_names = []
    for name in coordinates.values():
        if name in ds:
            bad_names.append(name)
    if bad_names:
        raise ValueError('Coordinate names {} clash with variables in the dataset. '
                         "Use the 'coordinates' argument of open_boutdataset to provide "
                         "alternative names".format(bad_names))

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
def add_s_alpha_geometry_coords(ds, coordinates=None):

    ds = add_toroidal_geometry_coords(ds, coordinates=coordinates)

    # Add 1D radial coordinate
    if 'r' in ds:
        raise ValueError("Cannot have variable 'r' in dataset when using "
                         "geometry='s-alpha'")
    ds['r'] = ds['hthe'].isel({coordinates[1]: 0}).squeeze(drop=True)
    ds['r'].attrs['units'] = 'm'
    ds = ds.set_coords('r')
    ds = ds.rename(x='r')

    # Simplify psi to be radially-varying only
    ds[coordinates[0]] = ds[coordinates[0]].isel({coordinates[1]: 0}).squeeze(drop=True)

    return ds
