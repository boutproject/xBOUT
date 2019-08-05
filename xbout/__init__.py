from .load import open_boutdataset

from . import geometries
from .geometries import register_geometry, REGISTERED_GEOMETRIES

from .boutdataset import BoutDatasetAccessor
from .boutdataarray import BoutDataArrayAccessor

from .plotting.animate import animate_pcolormesh
from .plotting.utils import plot_separatrix

from ._version import __version__
