Loading your data
=================

The function `open_boutdataset` uses xarray & dask to collect BOUT++
data spread across multiple NetCDF files into one contiguous xarray
dataset.

The data from a BOUT++ run can be loaded with just::

  bd = open_boutdataset('./run_dir*/BOUT.dmp.*.nc', inputfilepath='./BOUT.inp')

`open_boutdataset` returns an instance of an `xarray.Dataset` which
contains BOUT-specific information in the ``attrs``, so represents a
general structure for storing all of the output of a simulation,
including data, input options and (soon) grid data.

Datasets that were saved by xBOUT (with `BoutDataset.save`) can be
reloaded by passing the path to the saved file(s). If such a Dataset
has been converted to Zarr, it can be reloaded directly from the Zarr
store, either as a path or URL ending in ``.zarr`` (or prefixed with
``zarr://``) or as a zarr store object, for example an Icechunk
session store. Keyword arguments are passed on to `xarray.open_zarr`,
so ``group`` selects a group within the store::

  bd = open_boutdataset('./boutdata.zarr')
  bd = open_boutdataset(icechunk_session.store, group='shot_21846', geometry='toroidal')

Only Datasets saved by xBOUT can be opened from Zarr; raw BOUT++ dump
files in Zarr are not supported.
