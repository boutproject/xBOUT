# xBOUT

Provides an interface for collecting the output data from a
[BOUT++](https://boutproject.github.io/) simulation into an xarray
dataset in an efficient and scalable way, as well as accessor methods
for common BOUT++ analysis and plotting tasks.

Currently only in alpha (until 1.0 released) so please report any bugs,
and feel free to raise issues asking questions or making suggestions.



### Loading your data

The function `open_boutdataset()` uses xarray + dask to collect BOUT++
data spread across multiple NetCDF files into one contiguous xarray
dataset.

The data from a BOUT++ run can be loaded with just

```python
bd = open_boutdataset('./run_dir*/BOUT.dmp.*.nc', inputfilepath='./BOUT.inp')
```

# PUT OUTPUT HERE

`open_boutdataset()` returns an instance of an `xarray.Dataset` which
contains BOUT-specific information in the `attrs`, so represents a
general structure for storing all of the output of a simulation,
including data, input options and (soon) log file output.



### BoutDataset Accessor Methods

xBOUT defines a set of accessor methods on the loaded Datasets and
DataArrays, which are accessed as `ds.bout.method()`.

This is where BOUT-specific data manipulation, analysis and plotting
functionality is stored, for example

```python
ds.bout.animate2D(animate_over='t', x='x', y='z')
```

# PUT GIF HERE

### Extending xBOUT for your specific BOUT module

The accessor classes `BoutDatasetAccessor` and `BoutDataArrayAccessor`
intended to be subclassed for specific BOUT++ modules.

For example to add an extra method specific to the `STORM` BOUT++
module:

```python
from xarray import register_dataset_accessor
from xbout.boutdataset import BoutDatasetAccessor

@register_dataset_accessor('storm')
class StormAccessor(BoutAccessor):
    """
    Class specifically for holding data from a simulation using the STORM module for BOUT++.

    Implements methods for normalising the data with STORM-specific normalisation constants.
    """

    def __init__(self, ds_object):
        super().__init__(ds_object)

    def storm_specific_method(self):
        print("Do something only STORM users would want to do")
```

There is included an example of a
`StormDataset` which contains all the data from a
[STORM](https://github.com/boutproject/STORM) simulation, as well as
extra calculated quantities which are specific to the STORM module.



### Installation

Currently not on pip or conda, so you will need to clone this repo and
install using `python setup.py`
You can run the tests by navigating to the `/xBOUT/` directory and
entering `pytest`.


It relies on two upstream additions to xarray
([first](https://github.com/pydata/xarray/pull/2482) &
[second](https://github.com/pydata/xarray/pull/2553) pull requests).
The first is merged, but the second isn't, so for now you need to clone
the branch of xarray containing the PR
[here](https://github.com/TomNicholas/xarray/tree/feature/nd_combine).

You will also need to install [dask](https://dask.org/),
as described in the xarray documentation
[here](http://xarray.pydata.org/en/stable/installing.html#for-parallel-computing).



### Contributing

Feel free to raise issues about anything, or submit pull requests,
though I would encourage you to submit an issue before writing a pull
request.
For a general guide on how to contribute to an open-source python
project see
[xarray's guide for contributors](http://xarray.pydata.org/en/stable/contributing.html).

The existing code was written using Test-Driven Development, and I would
like to continue this, so please include `pytest` tests with any pull
requests.

If you write a new accessor, then this should really live with the code
for your BOUT module, but it could potentially be added as an example to
this repository too.


### Development plan

Things which definitely need to be included (see the 1.0 milestone):

- [ ] More tests, both with
 and against the
`boutdata.collect()`
- [ ] Speed test against old collect

Things which would be nice and I plan to do:

- [ ] Infer concatenation order from global indexes (see
[issue](https://github.com/TomNicholas/xBOUT/issues/3))
- [ ] Real-space coordinates
- [ ] Variable names and units (following CF conventions)
- [ ] Variable normalisations

Things which would require a lot of effort by another developer but
could be very powerful:

- [ ] Using real-space coordinates to create tokamak-shaped plots
- [ ] Support for staggered grids using xgcm
- [ ] Support for encoding topology using xgcm
- [ ] API for applying BoutCore operations (hopefully using `xr.apply_ufunc`)
