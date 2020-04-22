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
  - ``xarray >= v0.13.0``
  - ``dask[array] >= 1.0.0``
  - ``natsort >= 5.5.0``
  - ``matplotlib >= 2.2``
  - ``animatplot >= 0.4.0``
  - ``netcdf4 >= 1.4.0``
- All dependencies can be installed by running:

.. code-block:: bash

  pip3 install --user -r requirements.txt


Installation
------------

- :py:mod:`xbout` is not currently on pip or conda. Therefore to install xbout on
  your system you must first clone the repository using:

.. code-block:: bash

  git clone git@github.com:boutproject/xBOUT.git

or

.. code-block:: bash

  git clone https://github.com/boutproject/xBOUT.git


Once cloned navigate to the `xBOUT` directory and run the following command:

.. code-block:: bash

  pip3 install --user ./

or

.. code-block:: bash

  python3 setup.py install


You can run the tests by navigating to the `/xBOUT/` directory and
entering `pytest`. You can also test your installation of `xBOUT` by
running `pytest --pyargs xbout`.

xBOUT will install the required python packages when you run one of
the above install commands if they are not already installed on your
system.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _BOUT++: https://boutproject.github.io/
.. _xarray: https://xarray.pydata.org/en/stable/index.html
.. _this fork: https://github.com/TomNicholas/xarray/tree/feature/nd_combine
