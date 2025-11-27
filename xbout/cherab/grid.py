import numpy as np
import xarray as xr

from .triangulate import Triangulate


def da_with_cherab_grid(da):
    """
    Convert an BOUT++ DataArray to a format that Cherab can use:
    - A 'cell_number' coordinate
    - A 'cherab_grid` attribute

    The cell_number coordinate enables the DataArray to be sliced
    before input to Cherab.

    Parameters
    ----------
    ds : xarray.DataArray

    Returns
    -------
    updated_da
    """
    if "cherab_grid" in da.attrs:
        # Already has required attribute
        return da

    if da.attrs["geometry"] != "toroidal":
        raise ValueError("xhermes.plotting.cherab: Geometry must be toroidal")

    if da.sizes["zeta"] != 1:
        raise ValueError("xhermes.plotting.cherab: Zeta index must have size 1")

    # Cell corners
    rm = np.stack(
        (
            da.coords["Rxy_upper_right_corners"],
            da.coords["Rxy_upper_left_corners"],
            da.coords["Rxy_lower_right_corners"],
            da.coords["Rxy_lower_left_corners"],
        ),
        axis=-1,
    )
    zm = np.stack(
        (
            da.coords["Zxy_upper_right_corners"],
            da.coords["Zxy_upper_left_corners"],
            da.coords["Zxy_lower_right_corners"],
            da.coords["Zxy_lower_left_corners"],
        ),
        axis=-1,
    )

    grid = Triangulate(rm, zm)

    # Store the cell number as a coordinate.
    # This allows slicing of arrays before passing to Cherab

    # Create a DataArray for the vertices and triangles
    cell_number = xr.DataArray(
        grid.cell_number, dims=["x", "theta"], name="cell_number"
    )

    result = da.assign_coords(cell_number=cell_number)
    result.attrs.update(cherab_grid=grid)
    return result


def ds_with_cherab_grid(ds):
    """
    Create an xarray DataSet with a Cherab grid

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    updated_ds
    """

    # The same operation works on a DataSet
    ds = da_with_cherab_grid(ds)

    # Add the Cherab grid as an attribute to all variables
    grid = ds.attrs["cherab_grid"]
    for var in ds.data_vars:
        ds[var].attrs.update(cherab_grid=grid)

    return ds
