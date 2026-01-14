BoutDataset Accessor Methods
============================

xBOUT defines a set of accessor_ methods on the loaded `xarray.Dataset` and
`xarray.DataArray`, which are called by ``ds.bout.<method>``.

This is where BOUT-specific data manipulation, analysis and plotting
functionality is stored, for example::

  ds['n'].bout.animate2D(animate_over='t', x='x', y='z')


.. image:: images/n_over_t.gif
   :alt: density

or::

  ds.bout.create_restarts(savepath='.', nxpe=4, nype=4)

See `BoutDatasetAccessor` and `BoutDataArrayAccessor` for details on
the available methods.

.. _accessor: https://docs.xarray.dev/en/stable/internals/extending-xarray.html
