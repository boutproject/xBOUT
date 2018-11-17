# xBOUT

Provides an interface for collecting the output data from a
[BOUT++](https://boutproject.github.io/) simulation into an xarray
dataset in an efficient and scalable way, as well as accessor methods
for manipulating and plotting the data.

### open_boutdataset

The function `open_boutdataset()` uses xarray + dask to collect BOUT++
data spread across multiple NetCDF files into one contiguous xarray
dataset.

The data from a BOUT++ run can be loaded with just

```python
bd = open_boutdataset('./run_dir*/BOUT.dmp.*.nc', inputfilepath='./BOUT.inp')

print(bd)
print(bd.bout.options)
```


It relies on two upstream additions to xarray
([first](https://github.com/pydata/xarray/pull/2482) &
[second](https://github.com/pydata/xarray/pull/2553) pull requests),
so the most recent version of xarray must be used.


### BoutDataset Accessor Methods

`BoutDataset` class, a general
structure for storing all of the output of a simulation, including data,
input options and log file output. `BoutDataset` uses `xcollect` to
replace the current `boutdata.collect`, which is slow and scales poorly.


The accessor class `BoutAccessor` is intended to be subclassed for
specific BOUT++ modules. There is included an example of a
`StormDataset` which contains all the data from a
[STORM](https://github.com/boutproject/STORM) simulation, as well as
extra calculated quantities which are specific to the STORM module.
