from warnings import warn

import xarray as xr
import numpy as np

REGISTERED_GEOMETRIES = {}


class UnregisteredGeometryError(Exception):
    """Error for unregistered geometry type"""


def apply_geometry(ds, geometry_name):
    """

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset (from
    geometry_name : str
        Name under which the desired geometry function was registered

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
        raise UnregisteredGeometryError

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


@register_geometry('toroidal')
def add_toroidal_geometry_coords(ds):

    # Change names of dimensions to Orthogonal Toroidal ones
    ds = ds.rename(y='theta', inplace=True)

    # Add 1D Orthogonal Toroidal coordinates
    ny = ds.dims['theta']
    theta = xr.DataArray(np.linspace(start=0, stop=2 * np.pi, num=ny),
                         dims='theta')
    ds = ds.assign_coords(theta=theta)

    # TODO automatically make this coordinate 1D in simplified cases?
    ds = ds.rename(psixy='psi', inplace=True)
    ds = ds.set_coords('psi')
    ds['psi'].attrs['units'] = 'Wb'

    # If full data (not just grid file) then toroidal dim will be present
    if 'z' in ds.dims:
        ds = ds.rename(z='phi', inplace=True)
        nz = ds.dims['phi']
        phi = xr.DataArray(np.linspace(start=0, stop=2 * np.pi, num=nz),
                           dims='phi')
        ds = ds.assign_coords(phi=phi)

    # Add 2D Cylindrical coordinates
    ds = ds.rename(Rxy='R', Zxy='Z', inplace=True)
    ds = ds.set_coords(['R', 'Z'])

    return ds


@register_geometry('s-alpha')
def add_s_alpha_geometry_coords(ds):

    ds = add_toroidal_geometry_coords(ds)

    # Add 1D radial coordinate
    ds['r'] = ds['hthe'].isel(theta=0).squeeze(drop=True)
    ds['r'].attrs['units'] = 'm'
    ds = ds.set_coords('r')
    ds = ds.rename(x='r', inplace=True)

    # Simplify psi to be radially-varying only
    ds['psi'] = ds['psi'].isel(theta=0).squeeze(drop=True)

    return ds
