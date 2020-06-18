BoutDataset Accessor Methods
============================

xBOUT defines a set of accessor_ methods on the loaded Datasets and
DataArrays, which are called by `ds.bout.method`.

This is where BOUT-specific data manipulation, analysis and plotting
functionality is stored, for example::

  ds['n'].bout.animate2D(animate_over='t', x='x', y='z')


.. image:: images/n_over_t.gif
   :alt: density

or::

  ds.bout.create_restarts(savepath='.', nxpe=4, nype=4)

.. _accessor: https://xarray.pydata.org/en/stable/internals.html#extending-xarray
