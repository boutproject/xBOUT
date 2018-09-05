import xarray as xr
import numpy as np


def _concat_nd(obj_grid, concat_dims=None, data_vars=None, **kwargs):
    """
    Concatenates a structured ndarray of xarray Datasets along multiple dimensions.

    The supplied ndarray must have each element obj_grid[i][j] contained in the form {'key': obj},
    so that numpy does not see their __len__ or __getitem__ properties.

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
    data_vars : list of {'minimal', 'different', 'all' or list of str}, optional
        These data variables will be concatenated together:
          * 'minimal': Only data variables in which the dimension already
            appears are included.
          * 'different': Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * 'all': All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the 'minimal' data variables.
        If objects are DataArrays, data_vars must be 'all'.
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

    # Check inputs
    if any(obj_grid.shape) > 1:
        raise xr.MergeError('The logical grid of datasets should not have any unit-length '
                            'dimensions, but is of shape ' + str(obj_grid.shape))
    if any([not (isinstance(ds['key'], xr.Dataset) or isinstance(ds['key'], xr.DataArray)) for ds in obj_grid.flat]):
        raise TypeError('obj_grid contains at least one element that is neither a Dataset or Dataarray.')
    if len(concat_dims) != obj_grid.ndim:
        raise xr.MergeError('List of dimensions along which to concatenate is incompatible '
                            'with the given array of dataset objects. Respective lengths: '
                            + str(len(concat_dims)) + ', ' + str(obj_grid.ndim))
    if data_vars is None:
        data_vars = ['all']*len(concat_dims)
    if len(data_vars) != obj_grid.ndim:
        raise xr.MergeError('List of methods with which to concatenate is incompatible '
                            'with the given array of dataset objects. Respective lengths: '
                            + str(len(data_vars)) + ', ' + str(obj_grid.ndim))

    # Combine datasets along one dimension at a time,
    # Have to start with last axis and finish with axis=0 otherwise axes will disappear as they are looped over
    for axis in reversed(range(obj_grid.ndim)):
        obj_grid = np.apply_along_axis(_concat_dicts, axis, arr=obj_grid,
                                       dim=concat_dims[axis], data_vars=data_vars[axis], **kwargs)

    # Grid should now only contain one xarray object
    return obj_grid.item()['key']


def _concat_dicts(dict_objs, dim, **kwargs):
    """
    This is required instead of just using xr.concat because numpy attempts to turn bare Datasets into arrays
    See https://github.com/pydata/xarray/issues/2159#issuecomment-417802225
    and https://stackoverflow.com/questions/7667799/numpy-object-arrays
    """
    objs = [dict_obj['key'] for dict_obj in dict_objs]
    return {'key': xr.concat(objs, dim, **kwargs)}

