Loading your data
=================

The function `open_boutdataset` uses xarray & dask to collect BOUT++
data spread across multiple NetCDF files into one contiguous xarray
dataset.

The data from a BOUT++ run can be loaded with just::

  bd = open_boutdataset('./run_dir*/BOUT.dmp.*.nc', inputfilepath='./BOUT.inp')

`open_boutdataset` returns an instance of an `xarray.Dataset` which
contains BOUT-specific information in the `attrs`, so represents a
general structure for storing all of the output of a simulation,
including data, input options and (soon) grid data.
