Cherab interface
================

Cherab (https://www.cherab.info/) is a python library for forward
modelling diagnostics based on spectroscopic plasma emission.  It is
based on the Raysect (http://www.raysect.org/) scientific ray-tracing
framework.

Triangulation
-------------

Before Cherab can be used, a triangulated mesh must be generated. This
mesh is stored in an attribute ``cherab_grid``. This can be attached to
a Dataset or DataArray by calling the ``bout.with_cherab_grid()`` accessor::

  ds = ds.bout.with_cherab_grid()

Following operations will generate a grid if needed, but if this
attribute is already present then the grid is not
recalculated. Calling this method on a dataset before performing
Cherab operations improves efficiency.


Wall heat fluxes
----------------

To calculate radiation heat fluxes to walls, first create an ``xbout.AxisymmetricWall``
object that represents a 2D (R, Z) axisymmetric wall. This can be done by
reading the wall coordinates from a GEQDSK equilibrium file::
   
   wall = xbout.wall.read_geqdsk("geqdsk")

Triangulate a grid and attach to the dataset::

  ds = ds.bout.with_cherab_grid()

Extract a data array. This must be 2D (x, y) so select a single time
slice and toroidal angle e.g. excitation radiation::

  da = -bd['Rd+_ex'].isel(t=1, zeta=0)

Remove potentially invalid data in the guard cells::
  
  da[:2,:] = 0.0
  da[-2:,:] = 0.0

Attach data to the Cherab triangulated mesh, returning an
``xbout.cherab.TriangularData`` object that Cherab can work with::

  data = da.bout.as_cherab_data()

Calculate the wall heat fluxes::
  
  result = data.wall_flux(cat_wall, pixel_samples=1000)

This result is a list of dicts, one for each wall element. Those dicts
contain coordinates "rz1" and "rz2", results "power_density" and
"total_power".

