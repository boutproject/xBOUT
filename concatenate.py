import xarray as xr
import numpy as np


def concat_nd(obj_grid, concat_dims=None, **kwargs):
    """
    Concatenates a structured ndarray of xarray Datasets along multiple dimensions.

    Parameters
    ----------
    obj_grid : numpy array of Dataset and DataArray objects
        N-dimensional numpy object array containing xarray objects in the shape they
        are to be concatenated. Each object is expected to
        consist of variables and coordinates with matching shapes except for
        along the concatenated dimension.
    concat_dims : list of str or DataArray or pandas.Index
        Names of the dimensions to concatenate along. Each dimension in this argument
        is passed on to :py:func:`xarray.concat` along with the dataset objects.
        Should therefore be a list of valid dimension arguments to xarray.concat().
        Each element in concat_dims can either be a new dimension name, in which case
        it is added along axis=0, or an existing dimension name, in which case the
        location of the dimension is unchanged. If dimension is provided as a DataArray
        or Index, its name is used as the dimension to concatenate along and the values
        are added as a coordinate.
    **kwargs : optional
        Additional arguments passed on to xarray.concat().

    Returns
    -------
    combined : xarray.Dataset

    See Also
    --------
    xarray.concat
    xarray.auto_combine
    xarray.open_mfdataset
    """

    print(obj_grid)

    # Check inputs
    if any(obj_grid.shape) > 1:
        raise xr.MergeError('The logical grid of datasets should not have any unit-length '
                            'dimensions, but is of shape ' + str(obj_grid.shape))
    if any([not (isinstance(ds, xr.Dataset) or isinstance(ds, xr.Dataset)) for ds in obj_grid.flat]):
        raise xr.MergeError('obj_grid contains at least one element that is neither a Dataset or Dataarray.')
    if len(concat_dims) != obj_grid.ndim:
        raise xr.MergeError('List of dimensions along which to concatenate is incompatible '
                            'with the given array of dataset objects. Lengths: '
                            + str(len(concat_dims)) + ', ' + str(obj_grid.ndim))

    # Combine datasets along one dimension at a time
    # Start with last axis and finish with axis=0
    # TODO check that python actually does loops in this way (i.e. overwrites dataset_grid)
    for axis in reversed(range(obj_grid.ndim)):
        obj_grid = np.apply_along_axis(xr.concat, axis, arr=obj_grid,
                                       dim=concat_dims[axis], **kwargs)

    # Grid should now only contain one xarray object
    if obj_grid.shape == ():
        return obj_grid.item
    else:
        raise xr.MergeError('Multidimensional concatenation failed: '
                            'result has shape ' + str(obj_grid.shape))
