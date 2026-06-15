"""
xarray backend for reading ADIOS2 ``.bp`` files.

This backend provides an xarray ``BackendEntrypoint`` that can open ADIOS2
datasets via the ``adios2`` Python package. Variables are represented as
``LazilyIndexedArray`` objects backed by ADIOS2 selections.

On-disk conventions used by this backend:
- Per-variable dimension names are stored as attributes
  ``{varname}/__xarray_dimensions__``.
- Dataset-level attributes are stored under ``__xarray_dataset_attrs__/{key}``.
"""

from __future__ import annotations

import os

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ItemsView

import numpy as np

from xarray import Dataset, Variable
from xarray.backends.common import (
    BackendArray,
    BackendEntrypoint,
    _normalize_path,
)

from xarray.core import indexing

if TYPE_CHECKING:  # pragma: no cover
    from adios2 import FileReader
    from io import BufferedIOBase
    from xarray.backends.common import AbstractDataStore


DATASET_ATTR_PREFIX = "__xarray_dataset_attrs__/"
XARRAY_DIMS_ATTR = "__xarray_dimensions__"
XARRAY_ORIGINAL_DTYPE_ATTR = "__xarray_original_dtype__"

adios_to_numpy_type = {
    "char": np.char,
    "int8_t": np.int8,
    "int16_t": np.int16,
    "int32_t": np.int32,
    "int64_t": np.int64,
    "uint8_t": np.uint8,
    "uint16_t": np.uint16,
    "uint32_t": np.uint32,
    "uint64_t": np.uint64,
    "float": float,
    "double": np.double,
    "long double": np.longdouble,
    "float complex": np.complex64,
    "double complex": np.complex128,
    "string": np.char,
}


class BoutADIOSBackendArray(BackendArray):
    """
    Lazily indexed array backed by an ADIOS2 Variable.

    xarray calls ``__getitem__`` with an ``ExplicitIndexer``; this class maps the
    indexer into ADIOS2 ``set_step_selection`` (for time/steps) and
    ``set_selection`` (for spatial dimensions), then reads the selection.
    """

    def __init__(
        self,
        shape: list,
        dtype: np.dtype,
        lock,
        adiosfile: FileReader,
        varname: str,
        *,
        cast_dtype: np.dtype | None = None,
    ):
        """
        Parameters
        ----------
        shape
            Full xarray-visible shape. If the ADIOS2 variable has steps, the first
            dimension is the synthetic xarray time dimension.
        dtype
            Numpy dtype used by ADIOS2 for the stored variable.
        lock
            Optional lock for thread-safety. ADIOS2 reads are not thread-safe.
        adiosfile
            Open ADIOS2 ``FileReader`` handle.
        varname
            Name of the ADIOS2 variable to read.
        cast_dtype
            Optional dtype to cast to after reading (used to round-trip types that
            are stored differently on disk, e.g. ``bool`` stored as ``uint8``).
        """
        self.shape = shape
        self.dtype = dtype
        self.lock = lock
        self.fh = adiosfile
        self.varname = varname
        self.cast_dtype = cast_dtype
        self.adiosvar = self.fh.inquire_variable(varname)
        self.steps = self.adiosvar.steps()

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        """Read a selection defined by an xarray ``ExplicitIndexer``."""
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        """
        Convert xarray basic indexing into ADIOS2 selection calls and read.

        Notes
        -----
        - ADIOS2 does not support stepped slicing (``slice.step != 1``).
        - ADIOS2 time is represented as "steps". If an ADIOS2 variable has steps,
          xarray's first dimension is treated as time and mapped via
          ``set_step_selection``.
        """
        start = []
        count = []
        dimid = 0
        first_sl = True
        for sl in key:
            if isinstance(sl, slice):
                if sl.start is None:
                    st = 0
                else:
                    st = sl.start

                if sl.stop is None:
                    ct = self.shape[dimid] - st
                else:
                    ct = sl.stop - st

                if sl.step != 1 and sl.step is not None:
                    msg = (
                        "The indexing operation with step != 1 you are attempting to perform "
                        "is not valid on ADIOS2.Variable object. "
                    )
                    raise IndexError(msg)
            else:
                st = sl - 1
                ct = 1

            if self.steps > 1 and first_sl:  # key[0] is the step selection
                # print(f"    data step selection start = {st}  count = {ct}")
                self.adiosvar.set_step_selection([st, ct])
                # Advance past the implicit steps dimension in self.shape
                dimid += 1
            else:
                start.append(st)
                count.append(ct)
                dimid += 1
            first_sl = False
        self.adiosvar.set_selection([start, count])

        data = self.fh.read(self.adiosvar)
        if self.steps > 1:
            # ADIOS does not have time dimension. Read returns n-dim array
            # with the steps included in the first dimension
            dim0 = int(data.shape[0] / self.steps)
            if data.shape[0] % self.steps != 0:
                print(
                    f"ERROR in BoutADIOSBackendArray: first dimension problem "
                    f"with handling steps. Variable name={self.varname} "
                    f"shape={data.shape}, steps={self.steps}"
                )
            data = data.reshape((self.steps, dim0) + data.shape[1:])
        if self.cast_dtype is not None:
            data = np.asarray(data).astype(self.cast_dtype, copy=False)
        return data


def attrs_of_var(varname: str, items: ItemsView, separator: str = "/"):
    """Return (name, info) pairs for attributes scoped to a variable."""
    return [(key, value) for key, value in items if key.startswith(varname + separator)]


# pylint: disable=R0902   # Too many instance attributes
# pylint: disable=R0912   # Too many branches
# pylint: disable=E1121   # too-many-function-args
class BoutAdiosBackendEntrypoint(BackendEntrypoint):
    """
    xarray backend entrypoint for ADIOS2 ``.bp`` datasets.

    For more information about the underlying library, visit:
    https://adios2.readthedocs.io/en/stable

    See Also
    --------
    backends.AdiosStore
    """

    description = "Open ADIOS2 files/folders (.bp) using adios2 in Xarray"
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.ZarrBackendEntrypoint.html"

    def __init__(self):
        self._fh = None

    def close(self) -> None:
        """Close the underlying ADIOS2 file handle, if one is open."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    ) -> bool:
        """Return True if this backend can open the provided filename."""
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            return ext in {".bp"}

        return False

    def open_dataset(  # type: ignore[override]  # allow LSP violation, not supporting **kwargs
        self,
        filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
    ) -> Dataset:
        """
        Open an ADIOS2 ``.bp`` file/folder as an xarray Dataset.

        Parameters
        ----------
        filename_or_obj
            Path to a ``.bp`` dataset (directory or file, depending on ADIOS2 engine).
        drop_variables
            Optional variable name or iterable of variable names to exclude.
        """
        from adios2 import FileReader

        filename_or_obj = _normalize_path(filename_or_obj)

        self._fh = FileReader(filename_or_obj)
        vars = self._fh.available_variables()
        attrs = self._fh.available_attributes()
        attr_items = attrs.items()
        xvars = {}

        for varname, varinfo in vars.items():
            if drop_variables is not None and varname in drop_variables:
                continue
            shape_str = varinfo["Shape"].split(", ")
            if shape_str[0]:
                shape_list = list(map(int, shape_str))
            else:
                shape_list = []
                shape_str = []
            steps = int(varinfo["AvailableStepsCount"])
            varattrs = attrs_of_var(varname, attr_items, "/")
            dims = None
            vlen = len(varname) + 1  # include /
            xattrs = {}
            original_dtype: np.dtype | None = None
            for aname, ainfo in varattrs:
                attr_value = self._fh.read_attribute(aname)
                if aname == varname + "/" + XARRAY_DIMS_ATTR:
                    dims = attr_value
                elif aname == varname + "/" + XARRAY_ORIGINAL_DTYPE_ATTR:
                    try:
                        original_dtype = np.dtype(str(attr_value))
                    except TypeError:
                        original_dtype = None
                else:
                    xattrs[aname[vlen:]] = attr_value
                attrs.pop(aname)

            # Create the xarray variable
            if dims is None:
                dims = shape_str
            if shape_list != []:
                if steps > 1:
                    shape_list.insert(0, steps)
                    dims.insert(0, "t")
                nptype = np.dtype(adios_to_numpy_type[varinfo["Type"]])
                cast_dtype = (
                    original_dtype
                    if original_dtype is not None and original_dtype != nptype
                    else None
                )
                xdata = indexing.LazilyIndexedArray(
                    BoutADIOSBackendArray(
                        shape_list,
                        nptype,
                        None,
                        self._fh,
                        varname,
                        cast_dtype=cast_dtype,
                    )
                )
                # print(f"\tDefine VARIABLE {varname} with dims {dims}")
                xvar = Variable(
                    dims,
                    xdata,
                    attrs=xattrs,
                    encoding={"dtype": (original_dtype or nptype)},
                )
            else:
                if steps > 1:
                    avar = self._fh.inquire_variable(varname)
                    avar.set_step_selection([0, avar.steps()])
                    data = self._fh.read(avar)
                    if original_dtype is not None and data.dtype != original_dtype:
                        data = np.asarray(data).astype(original_dtype, copy=False)
                    xvar = Variable(
                        "t", data, attrs=xattrs, encoding={"dtype": data.dtype}
                    )
                else:
                    data = self._fh.read(varname)
                    if (
                        original_dtype is not None
                        and np.asarray(data).dtype != original_dtype
                    ):
                        data = np.asarray(data).astype(original_dtype, copy=False)
                    if varinfo["Type"] == "string":
                        xvar = Variable([], data, attrs=xattrs, encoding=None)
                    else:
                        xvar = Variable([], data, attrs=xattrs, encoding=None)
            xvars[varname] = xvar

        ds_attrs = {}
        for attname in list(attrs.keys()):
            attr_value = self._fh.read_attribute(attname)
            if isinstance(attname, str) and attname.startswith(DATASET_ATTR_PREFIX):
                ds_attrs[attname[len(DATASET_ATTR_PREFIX) :]] = attr_value
            else:
                ds_attrs[attname] = attr_value

        ds = Dataset(xvars, None, ds_attrs)
        ds.set_close(self.close)
        return ds
