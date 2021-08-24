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

Installation
------------

With `pip`:

.. code-block:: bash

  pip install --user xbout

or `conda`:

.. code-block:: bash

  conda install -c conda-forge xbout

You can run the tests by running `pytest --pyargs xbout`.

xBOUT will install the required python packages when you run one of
the above install commands if they are not already installed on your
system.

Examples
--------

You can find some example notebooks demonstrating various features of
:py:mod:`xbout` here: https://github.com/boutproject/xBOUT-examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _BOUT++: https://boutproject.github.io/
.. _xarray: https://xarray.pydata.org/en/stable/index.html
.. _this fork: https://github.com/TomNicholas/xarray/tree/feature/nd_combine
