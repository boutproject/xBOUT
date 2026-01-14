from .load import open_boutdataset, collect

from . import geometries
from .geometries import register_geometry, REGISTERED_GEOMETRIES

from .boutdataset import BoutDatasetAccessor
from .boutdataarray import BoutDataArrayAccessor

from .plotting.animate import animate_pcolormesh, animate_poloidal
from .plotting.utils import plot_separatrix

from .fastoutput import open_fastoutput

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

__all__ = [
    "open_boutdataset",
    "collect",
    "geometries",
    "register_geometry",
    "REGISTERED_GEOMETRIES",
    "BoutDataArrayAccessor",
    "BoutDatasetAccessor",
    "animate_pcolormesh",
    "animate_poloidal",
    "plot_separatrix",
    "open_fastoutput",
]
