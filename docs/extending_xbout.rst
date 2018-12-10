Extending xBOUT for your BOUT module
====================================

The accessor classes `BoutDatasetAccessor` and `BoutDataArrayAccessor`
are intended to be subclassed for specific BOUT++ modules. The
subclass accessor will then inherit all the `.bout` accessor methods,
but you will also be able to override these and define your own
methods within your new accessor.


For example to add an extra method specific to the `STORM` BOUT++
module::

  from xarray import register_dataset_accessor
  from xbout.boutdataset import BoutDatasetAccessor
  
  @register_dataset_accessor('storm')
  class StormAccessor(BoutAccessor):
      def __init__(self, ds_object):
          super().__init__(ds_object)
  
      def special_method(self):
          print("Do something only STORM users would want to do")


  >>> ds.storm.special_method()
  Do something only STORM users would want to do

There is included an example of a `StormDataset` which contains all
the data from a STORM_ simulation, as well as extra calculated
quantities which are specific to the STORM module.

.. _STORM: https://github.com/boutproject/STORM
