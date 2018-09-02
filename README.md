# xcollect

Collects data spread across multiple NetCDF files into one contiguous
xarray dataset.

Intended to be used to expand xarray's functionality by addressing
[this issue.](https://github.com/pydata/xarray/issues/2159)

Current approach is falling foul of issues with storing xarray objects
inside numpy arrays. Might have to rewrite using lists of lists instead.


## BoutDataset

This repo also contains a prototype `BoutDataset` class, a general
structure for storing all of the output of a
[BOUT++](https://boutproject.github.io/) simulation, including data,
input options and log file output. `BoutDataset` uses `xcollect` to
replace the current `boutdata.collect`, which is slow and scales poorly.

`BoutDataset` is intended to be subclassed for specific BOUT++ modules.
There is included an example of a `StormDataset` which
contains all the data from a [STORM](https://github.com/boutproject/STORM) simulation, as well as extra
calculated quantities which are specific to the STORM module.
