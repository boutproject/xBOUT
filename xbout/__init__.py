from .load import open_boutdataset, collect

from . import geometries
from .geometries import register_geometry, REGISTERED_GEOMETRIES

from .boutdataset import BoutDatasetAccessor
from .boutdataarray import BoutDataArrayAccessor

from .plotting.animate import animate_pcolormesh, animate_poloidal
from .plotting.utils import plot_separatrix

from .fastoutput import open_fastoutput

try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)
