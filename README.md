# collect
Collects data spread across multiple NetCDF files into one contiguous
xarray dataset.

Intended to be used to expand xarray's functionality
by addressing [this issue.](https://github.com/pydata/xarray/issues/2159)

Current approach is falling foul of issues with storing xarray objects
inside numpy arrays. Might have to rewrite using lists of lists instead.