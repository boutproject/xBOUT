# Custom implementation of xarray.open_mfdataset()

import xarray as xr


def concat_outer(dss, operation=None):
    """
    Concatenate nested lists along their outer dimension

    # Example

    >>> m = [[1,2,3],
             [2,3,3],
             [5,4,3]]

    >>> concat_outer(m)

    [[1, 2, 5], [2, 3, 4], [3, 3, 3]]

    >>> concat_outer(m, operation=sum)

    [8, 9, 9]
    
    """
    if not isinstance(dss[0], list):
        # Input is a 1D list
        if operation is not None:
            return operation(dss)
        return dss

    # Two or more dimensions
    # Swap first and second indices then concatenate inner
    if len(dss[0]) == 1:
        return concat_outer([dss[j][0] for j in range(len(dss))], operation=operation)

    return [
        concat_outer([dss[j][i] for j in range(len(dss))], operation=operation)
        for i in range(len(dss[0]))
    ]


def mfdataset(paths, chunks=None, concat_dim=None, preprocess=None):
    if chunks is None:
        chunks = {}

    if not isinstance(concat_dim, list):
        concat_dim = [concat_dim]

    if isinstance(paths, list):
        # Read nested dataset

        dss = [
            mfdataset(
                path, chunks=chunks, concat_dim=concat_dim[1:], preprocess=preprocess
            )
            for path in paths
        ]

        # The dimension to concatenate along
        if concat_dim[0] is None:
            # Not concatenating
            if len(dss) == 1:
                return dss[0]
            return dss

        # Concatenating along the top-level dimension
        return concat_outer(
            dss,
            operation=lambda ds: xr.concat(
                ds,
                concat_dim[0],
                data_vars="minimal",  # Only data variables in which the dimension already appears are concatenated.
                coords="minimal",  # Only coordinates in which the dimension already appears are concatenated.
                compat="override",  # Duplicate data taken from first dataset
                join="exact",  # Don't align. Raise ValueError when indexes to be aligned are not equal
                combine_attrs="override",  # Duplicate attributes taken from first dataset
                create_index_for_new_dim=False,
            ),
        )
    # A single path
    ds = xr.open_dataset(paths, chunks=chunks)
    if preprocess is not None:
        ds = preprocess(ds)
    return ds
