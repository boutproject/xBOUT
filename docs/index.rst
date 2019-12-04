.. xBOUT documentation master file, created by
   sphinx-quickstart on Mon Dec 10 14:17:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xBOUT's documentation!
=================================

:py:mod:`xbout` provides an interface for collecting the output data
from a `BOUT++`_ simulation into an xarray_ dataset in an efficient
and scalable way, as well as accessor methods for common BOUT++
analysis and plotting tasks.

Currently only in alpha (until 1.0 released) so please report any
bugs, and feel free to raise issues asking questions or making
suggestions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   loading_data
   accessor_methods
   extending_xbout
   xbout

Requirements
------------

- :py:mod:`xbout` requires Python 3

- The following modules are also needed:
  - ``xarray >= v0.12.2``
  - ``dask[array] >= 1.0.0``
  - ``natsort >= 5.5.0``
  - ``matplotlib >= 2.2``
  - ``animatplot >= 0.3``
  - ``netcdf4 >= 1.4.0``
- All dependencies can be installed by running:

.. code-block:: bash

  pip3 install --user -r requirements.txt

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _BOUT++: https://boutproject.github.io/
.. _xarray: http://xarray.pydata.org/en/stable/index.html
.. _this fork: https://github.com/TomNicholas/xarray/tree/feature/nd_combine
