import numpy as np
import xarray as xr


# TODO write a decorator to add functions as accessors to dataarrays?


def _rms_gufunc(x):
    """Generalized ufunc to calculate root mean squared value of an array."""
    squares = np.square(x)
    return np.sqrt(np.mean(squares, axis=-1))


def rms(da, dim=None, dask="parallelized", keep_attrs=True):
    """
    Reduces a dataarray by calculating the root mean square along the dimension
    dim.
    """

    # TODO If dim is None then take the root mean square along all dimensions?
    if dim is None:
        raise ValueError("Must supply a dimension along which to calculate rms")

    rms = xr.apply_ufunc(
        _rms_gufunc,
        da,
        input_core_dims=[[dim]],
        dask=dask,
        output_dtypes=[da.dtype],
        keep_attrs=keep_attrs,
    )

    # Return the name of the da as variable_rms
    rms.name = str(da.name) + "_rms"

    return rms
