from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

DATASET_ATTR_PREFIX = "__xarray_dataset_attrs__/"
VAR_ATTR_PREFIX_DIMENSIONS = "__xarray_dimensions__"
VAR_ATTR_PREFIX_ORIGINAL_DTYPE = "__xarray_original_dtype__"


class Adios2NotInstalledError(ImportError):
    pass


def _normalize_attr_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, (str, bytes, int, float, bool, np.number)):
        return value
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (str, bytes, int, float, bool, np.number)) for v in value):
            return list(value)
        return json.dumps(value, default=str)
    if isinstance(value, Mapping):
        return json.dumps(value, default=str)
    return json.dumps(value, default=str)


def _numpy_for_write(data: Any) -> np.ndarray:
    arr = np.asarray(data)

    if arr.dtype == np.dtype("bool"):
        return arr.astype(np.uint8)

    if arr.dtype.kind in {"M", "m"}:
        raise TypeError(
            "datetime64/timedelta64 not supported for ADIOS2 writer yet; "
            "encode to int64 with units attrs first"
        )

    if arr.dtype == np.dtype("O"):
        # Conservative: only allow scalar string-like objects.
        if arr.ndim == 0 and isinstance(arr.item(), (str, bytes)):
            return np.asarray(arr.item())
        raise TypeError(
            "object dtype not supported for ADIOS2 writer (except scalar strings)"
        )

    return arr


def _attr_to_numpy(value: Any) -> Any:
    norm = _normalize_attr_value(value)
    if isinstance(norm, np.ndarray):
        if norm.ndim == 0:
            return norm.item()
        return norm
    if isinstance(norm, (str, bytes, int, float, bool, np.number)):
        return norm
    return np.asarray(norm)


def write_dataset_bp(
    ds,
    path: str,
    *,
    time_dim: str = "t",
    engine: str = "BP4",
    parameters: Mapping[str, str] | None = None,
    overwrite: bool = True,
    variables: Iterable[str] | None = None,
) -> None:
    """
    Write an xarray Dataset to an ADIOS2 .bp output.

    Design choices:
    - Map ``time_dim`` to ADIOS2 steps (variables containing time_dim are written
      one slice per step).
    - Store per-variable dimension names as attribute ``{var}/__xarray_dimensions__``
      excluding ``time_dim`` when writing steps.
    - Store dataset attributes under ``__xarray_dataset_attrs__/{key}``.
    """
    try:
        import adios2  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise Adios2NotInstalledError(
            "adios2 is required to write .bp files; install adios2-python"
        ) from e
    if engine != "BP4":
        raise NotImplementedError(
            "Only BP4 is supported by the current writer implementation"
        )
    if parameters:
        raise NotImplementedError(
            "ADIOS2 engine parameters are not supported by the current writer implementation"
        )

    if variables is None:
        names_to_write = list(ds.variables)
    else:
        names_to_write = list(variables)

    # Determine step count across all time-dependent variables.
    step_count: int | None = None
    for name in names_to_write:
        var = ds.variables[name]
        if time_dim in var.dims:
            if var.dims[0] != time_dim:
                raise ValueError(
                    f"Only supports {time_dim!r} as the first dimension for {name!r}; "
                    f"got dims={var.dims!r}"
                )
            nsteps = int(var.sizes[time_dim])
            step_count = nsteps if step_count is None else max(step_count, nsteps)
    if step_count is None:
        step_count = 1

    mode = "w" if overwrite else "a"
    stream = adios2.Stream(path, mode)
    try:
        # Dataset attrs.
        for k, v in ds.attrs.items():
            stream.write_attribute(DATASET_ATTR_PREFIX + str(k), _attr_to_numpy(v))

        # Define all variables up-front to avoid Stream.write(name, ndarray) inference,
        # which is unreliable on some adios2-python builds.
        io = stream.io
        adios_vars: dict[str, Any] = {}
        sample_buffers: dict[str, np.ndarray] = {}

        for name in names_to_write:
            var = ds.variables[name]
            write_dtype = _numpy_for_write(var.data).dtype

            if time_dim in var.dims:
                shape = [int(var.sizes[d]) for d in var.dims if d != time_dim]
            else:
                shape = [int(s) for s in np.asarray(var.data).shape]

            start = [0] * len(shape)
            count = shape[:]

            if write_dtype.kind in {"U", "S"}:
                sample = np.asarray([""], dtype=write_dtype)
            else:
                sample = np.array([0], dtype=write_dtype)

            # Keep a reference to the sample buffer alive for the lifetime of the
            # stream. Some adios2-python builds appear to keep a view/pointer to the
            # provided numpy object when defining variables.
            sample_buffers[name] = sample

            adios_var = io.define_variable(name, sample, shape, start, count)
            if shape:
                adios_var.set_shape(shape)
                adios_var.set_selection([start, count])
            adios_vars[name] = adios_var

        # Per-variable attrs/dims (now that variables exist).
        for name in names_to_write:
            var = ds.variables[name]
            dims_wo_time = [d for d in var.dims if d != time_dim]
            stream.write_attribute(
                VAR_ATTR_PREFIX_DIMENSIONS,
                _attr_to_numpy(dims_wo_time),
                variable_name=name,
                separator="/",
            )
            for ak, av in var.attrs.items():
                stream.write_attribute(
                    str(ak),
                    _attr_to_numpy(av),
                    variable_name=name,
                    separator="/",
                )

            original_dtype = np.asarray(var.data).dtype
            write_arr = _numpy_for_write(var.data)
            if write_arr.dtype != original_dtype:
                stream.write_attribute(
                    VAR_ATTR_PREFIX_ORIGINAL_DTYPE,
                    _attr_to_numpy(str(original_dtype)),
                    variable_name=name,
                    separator="/",
                )

        for step in range(step_count):
            stream.begin_step()
            try:
                for name in names_to_write:
                    var = ds.variables[name]
                    adios_var = adios_vars[name]
                    if time_dim in var.dims:
                        if step >= int(var.sizes[time_dim]):
                            continue
                        data = _numpy_for_write(var.data)[step]
                    elif step == 0:
                        data = _numpy_for_write(var.data)
                    else:
                        continue

                    data_arr = np.asarray(data)
                    if data_arr.ndim:
                        # Use an owned, contiguous buffer for ADIOS2 Put().
                        # Some adios2-python builds appear to segfault when passed
                        # non-owned views.
                        data_arr = np.array(data_arr, copy=True, order="C")
                        # Ensure selection matches the provided buffer.
                        adios_var.set_selection(
                            [[0] * data_arr.ndim, list(data_arr.shape)]
                        )
                    else:
                        # Ensure a stable 0-d buffer with the intended dtype.
                        data_arr = np.asarray(
                            data_arr.item(), dtype=data_arr.dtype
                        ).reshape(())
                    stream.write(adios_var, data_arr)
            finally:
                stream.end_step()
    finally:
        stream.close()
