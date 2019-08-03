# xBOUT

[![Build Status](https://travis-ci.org/boutproject/xBOUT.svg?branch=master)](https://travis-ci.org/boutproject/xBOUT.svg?branch=master)
[![codecov](https://codecov.io/gh/boutproject/xBOUT/branch/master/graph/badge.svg)](https://codecov.io/gh/boutproject/xBOUT)

Documentation: https://xbout.readthedocs.io

xBOUT provides an interface for collecting the output data from a
[BOUT++](https://boutproject.github.io/) simulation into an
[xarray](http://xarray.pydata.org/en/stable/index.html)
dataset in an efficient and scalable way, as well as accessor methods
for common BOUT++ analysis and plotting tasks.

Currently only in alpha (until 1.0 released) so please report any bugs,
and feel free to raise issues asking questions or making suggestions.

### Loading your data

The function `open_boutdataset()` uses xarray & dask to collect BOUT++
data spread across multiple NetCDF files into one contiguous xarray
dataset.

The data from a BOUT++ run can be loaded with just

```python
bd = open_boutdataset('./run_dir*/BOUT.dmp.*.nc', inputfilepath='./BOUT.inp')
```

`open_boutdataset()` returns an instance of an `xarray.Dataset` which
contains BOUT-specific information in the `attrs`, so represents a
general structure for storing all of the output of a simulation,
including data, input options and (soon) grid data.



### BoutDataset Accessor Methods

xBOUT defines a set of
[accessor](http://xarray.pydata.org/en/stable/internals.html#extending-xarray)
methods on the loaded Datasets and DataArrays, which are called by
`ds.bout.method()`.

This is where BOUT-specific data manipulation, analysis and plotting
functionality is stored, for example

```python
ds['n'].bout.animate2D(animate_over='t', x='x', y='z')
```

![density](doc/images/n_over_t.gif)

or

```python
ds.bout.create_restarts(savepath='.', nxpe=4, nype=4)
```


### Extending xBOUT for your BOUT module

The accessor classes `BoutDatasetAccessor` and `BoutDataArrayAccessor`
are intended to be subclassed for specific BOUT++ modules. The subclass
accessor will then inherit all the `.bout` accessor methods, but you
will also be able to override these and define your own methods within
your new accessor.


For example to add an extra method specific to the `STORM` BOUT++
module:

```python
from xarray import register_dataset_accessor
from xbout.boutdataset import BoutDatasetAccessor

@register_dataset_accessor('storm')
class StormAccessor(BoutAccessor):
    def __init__(self, ds_object):
        super().__init__(ds_object)

    def special_method(self):
        print("Do something only STORM users would want to do")

ds.storm.special_method()
```
```
Out [1]: Do something only STORM users would want to do
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

Requires xarray v0.12.2 or later.

You will also need to install [dask](https://dask.org/),
as described in the xarray documentation
[here](http://xarray.pydata.org/en/stable/installing.html#for-parallel-computing),
as well as [natsort](https://github.com/SethMMorton/natsort)
and [animatplot](https://github.com/t-makaro/animatplot).



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
 and against the original
`boutdata.collect()`
- [ ] Speed test against old collect

Things which would be nice and I plan to do:

- [ ] Infer concatenation order from global indexes (see
[issue](https://github.com/TomNicholas/xBOUT/issues/3))
- [ ] Real-space coordinates
- [ ] Variable names and units (following CF conventions)
- [ ] Variable normalisations

Things which might require a fair amount of effort by another developer but
could be very powerful:

- [ ] Using real-space coordinates to create tokamak-shaped plots
- [ ] Support for staggered grids using xgcm
- [ ] Support for encoding topology using xgcm
- [ ] API for applying BoutCore operations (hopefully using `xr.apply_ufunc`)
