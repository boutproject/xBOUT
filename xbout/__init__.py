"""
xbout package public API.

This module is intentionally lightweight: many xbout features depend on optional
third-party packages (e.g. matplotlib, dask, boutdata). Import those lazily so
that submodules like ``xbout.adioswriter`` can be used without pulling in the
full dependency set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from . import geometries
    from .boutdataarray import BoutDataArrayAccessor
    from .boutdataset import BoutDatasetAccessor
    from .fastoutput import open_fastoutput
    from .geometries import REGISTERED_GEOMETRIES, register_geometry
    from .lazyload import lazy_open_boutdataset
    from .load import collect, open_boutdataset
    from .plotting.animate import animate_pcolormesh, animate_poloidal
    from .plotting.utils import plot_separatrix

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except Exception:  # pragma: no cover
        __version__ = "0+unknown"

__all__ = [
    "open_boutdataset",
    "lazy_open_boutdataset",
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


def __getattr__(name: str):
    if name in {"geometries", "register_geometry", "REGISTERED_GEOMETRIES"}:
        try:
            from . import geometries
            from .geometries import REGISTERED_GEOMETRIES, register_geometry
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "xbout geometries require optional dependencies (e.g. xarray)."
            ) from e
        return {
            "geometries": geometries,
            "register_geometry": register_geometry,
            "REGISTERED_GEOMETRIES": REGISTERED_GEOMETRIES,
        }[name]

    if name in {"open_boutdataset", "collect"}:
        try:
            from .load import collect, open_boutdataset
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "xbout.load requires optional dependencies (e.g. 'boutdata'). "
                "Install the full xbout extras, or import submodules that do not "
                "require boutdata."
            ) from e
        return {"open_boutdataset": open_boutdataset, "collect": collect}[name]

    if name == "lazy_open_boutdataset":
        try:
            from .lazyload import lazy_open_boutdataset
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "xbout.lazyload requires optional dependencies (e.g. dask, h5py, xarray)."
            ) from e
        return lazy_open_boutdataset

    if name in {"BoutDatasetAccessor", "BoutDataArrayAccessor"}:
        try:
            from .boutdataarray import BoutDataArrayAccessor
            from .boutdataset import BoutDatasetAccessor
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "xbout accessors require optional dependencies (e.g. xarray)."
            ) from e
        return {
            "BoutDatasetAccessor": BoutDatasetAccessor,
            "BoutDataArrayAccessor": BoutDataArrayAccessor,
        }[name]

    if name in {"animate_pcolormesh", "animate_poloidal"}:
        try:
            from .plotting.animate import animate_pcolormesh, animate_poloidal
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "xbout plotting requires optional dependencies (e.g. matplotlib)."
            ) from e
        return {
            "animate_pcolormesh": animate_pcolormesh,
            "animate_poloidal": animate_poloidal,
        }[name]

    if name == "plot_separatrix":
        try:
            from .plotting.utils import plot_separatrix
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "xbout plotting requires optional dependencies (e.g. matplotlib)."
            ) from e
        return plot_separatrix

    if name == "open_fastoutput":
        try:
            from .fastoutput import open_fastoutput
        except ImportError as e:  # pragma: no cover
            raise ImportError("xbout.fastoutput requires optional dependencies.") from e
        return open_fastoutput

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
