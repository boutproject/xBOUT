# SPDX-FileCopyrightText: 2023,2024 David Bold <dave@ipp.mpg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .tracing import setup_mesh, rz_to_ab, OutOfDomainError
import xarray as xr
from itertools import product
from ..boutdataset import BoutDatasetAccessor

import time

try:
    from tqdm import tqdm
except ImportError:

    class tqdm:
        def __init__(self, arg, **kw):
            self.arg = arg

        def __iter__(self):
            return self.arg.__iter__()

        def __enter__(self, *k):
            return self.arg.__enter__(*k)

        def __exit__(self, *k):
            return self.arg.__exit__(*k)


class timeit:
    def __init__(self, info="%f"):
        self.info = info

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *args):
        print(self.info % (time.time() - self.t0))


def weights(rz, mesh):
    ab, ij = rz_to_ab(rz, mesh)
    a, b = ab
    weigth = np.array(
        [
            [(1 - a) * (1 - b), (1 - b) * a],
            [(1 - a) * b, a * b],
        ]
    )
    # print(np.sum(weigth))
    nz = mesh.shape[1]
    ij = ij // nz, ij % nz
    return weigth, ij


def _evalat(ds, r, phi, z, key, delta_phi, fill_value, progress):
    r0, phi0, z0 = r, phi, z
    dims, shape, coords, (r, phi, z) = get_out_shape(r, phi, z)
    plus = tuple([f"delta_{x}" for x in "xyz"])
    dimsplus = *dims, *plus
    weights = np.empty((*shape, 2, 2, 2))
    ids = np.empty((*shape, 2, 2, 2, 3), dtype=int)
    missing = np.zeros(shape, dtype=bool)
    for ijk in tqdm(
        product(*[range(x) for x in shape]), total=np.prod(shape), disable=not progress
    ):
        try:
            w, idc = _evaluate_get_single(ds, r[ijk], phi[ijk], z[ijk], delta_phi)
        except OutOfDomainError:
            weights[ijk] = 0
            ids[ijk] = -1
            missing[ijk] = True
        else:
            weights[ijk] = w
            for io, (i, j, k) in enumerate(idc):
                # print(io, i, j, k)
                for jo in range(2):
                    for ko in range(2):
                        ids[(*ijk, io, jo, ko)] = i + jo, j, k + ko
    for i, c in enumerate([len(ds[k]) for k in "xyz"]):
        if i != 0:  # y and z is periodic
            ids[..., i] %= c
        ids[..., i][ids[..., i] == -1] = c - 1

    def add_dims(out):
        for k, v in zip(("R", "phi", "Z"), (r, phi, z)):
            k = f"dim_{k}"
            if k not in out:
                out[k] = dims, v
        return out

    if key is None:
        return add_dims(
            xr.Dataset(
                dict(
                    x=(dimsplus, ids[..., 0]),
                    y=(dimsplus, ids[..., 1]),
                    z=(dimsplus, ids[..., 2]),
                    weights=(dimsplus, weights),
                    missing=(dims, missing),
                )
            )
        )
    if isinstance(key, str):
        key = (key,)
    slc = {x: xr.DataArray(ids[..., i], dims=dimsplus) for i, x in enumerate("xyz")}
    weights = xr.DataArray(weights, dims=dimsplus)
    out = xr.Dataset()
    for k in key:
        theisel = ds[k].isel(**slc, missing_dims="ignore")
        try:
            theisel = theisel.compute()
        except AttributeError:
            pass
        out[k] = (theisel * weights).sum(dim=plus)
        if np.any(missing):
            # out[k].isel(
            assert (
                tuple(dims) == out[k].dims[-len(dims) :]
            ), f"{tuple(dims)} != {out[k].dims}"
            out[k].values[..., missing] = _fill_value(fill_value, out[k].dtype)
            # raise NotImplementedError("Missing data")
    return add_dims(out)


def _fill_value(fill_value, dtype):
    if fill_value is None:
        if isinstance(dtype.type, float):
            return np.nan
        return -1
    return fill_value


def _startswith(hay, needle):
    if len(hay) < len(needle):
        return False
    return hay[: len(needle)] == needle


def evaluate_at_keys(ds, keys, key, fill_value=np.nan, slow=False):
    """
    ds : xr.Dataset
         the bout dataset to evaluate at certain spatial points
    keys : xr.Dataset
        the mapping returned by evaluate_at with key=None
    key : tuple of str
        the variables to return
    """
    if key is None:
        return keys

    if isinstance(key, str):
        key = (key,)
    slc = {x: keys[x] for x in "xyz"}
    weights = keys["weights"]
    missing = keys["missing"]
    out = xr.Dataset()
    for k in key:
        if np.any(missing):
            for x in "xyz":
                _startswith(slc[x].dims, missing.dims)
                slc[x].values[missing] = 0
                print(x, np.min(slc[x]), np.max(slc[x]))
        # Fix periodic indexing
        for x in "yz":
            slc[x] %= len(ds[x])
        if slow:
            theisel = ds[k].isel(**slc, missing_dims="ignore")
        else:
            slcp = [slc[d] if d in slc else slice(None) for d in ds[k].dims]
            theisel = ds[k].values[tuple(slcp)]
        out[k] = (theisel * weights).sum(dim=tuple([f"delta_{x}" for x in "xyz"]))
        if np.any(missing):
            outk = out[k].transpose(*missing.dims, ...)
            outk.values[missing.values] = _fill_value(fill_value, outk.dtype)
    return out


# todo: caching
def _evaluate_get_single(ds, r, phi, z, delta_phi):
    # def _evalat_single(ds, r, z, p):
    try:
        period = ds.periodicity
    except AttributeError:
        period = int(round(float(2 * np.pi / ds.dy[0, 0, 0].values / len(ds.y))))

    assert period
    phi %= 2 * np.pi / period

    j = int(round(float((phi - delta_phi / 2) / delta_phi))) if delta_phi else -1

    try:
        cache = ds.bout._evaluate_at_cache
    except AttributeError:
        cache = {}
        setattr(ds.bout, "_evaluate_at_cache", cache)

    try:
        mshs, fs, nexti = cache[j]
        lasti = nexti - 1
    except KeyError:
        nexti = np.where(ds.y > phi)[0]
        if len(nexti):
            nexti = nexti[0]
        else:
            nexti = 0
        lasti = nexti - 1
        dsl = ds.isel(y=lasti)
        dsn = ds.isel(y=nexti)
        dy = float(dsl.dy[0, 0].data)
        cdy = dsn.y - dsl.y if lasti >= 0 else dsn.y - dsl.y + 2 * np.pi / period
        assert np.isclose(cdy, dy), f"{float(cdy)} is not close to {dy}"
        f1 = float(dsn.y.data - phi) / dy
        f2 = float(phi - dsl.y.data) / dy
        if f1 < 0:
            f1 += len(ds.y)
        if f2 < 0:
            f2 += len(ds.y)
        # print(f1, f2, dy)
        fs = [f1, f2]
        mshs = []
        for dsc, pre1, pre2 in ((dsl, "", "forward_"), (dsn, "backward_", "")):
            RZ = [dsc[pre1 + v] * f1 + dsc[pre2 + v] * f2 for v in "RZ"]
            # print([x.dims for x in RZ])
            # RZ = [np.concatenate((x, x[:, :1]), axis=1) for x in RZ]
            msh = setup_mesh(*RZ)
            mshs.append(msh)
        if delta_phi:
            cache[j] = (mshs, fs, nexti)
        # print(fs)
        # print(np.array([r, z]).shape)
    ijs = [weights(np.array([r, z]).ravel(), msh) for msh in mshs]
    # ws = [(w * f) for (w, (i, k)), f, j in zip(ijs, fs, (lasti, nexti))]
    ijs = [(w * f, (i, j, k)) for (w, (i, k)), f, j in zip(ijs, fs, (lasti, nexti))]
    a, b = ijs
    ijs = (a[0], b[0]), (a[1], b[1])
    return ijs


# Copied from xemc3
def get_out_shape(*data):
    """
    Convert xarray arrays and plain arrays to same shape and return
    dims, shape and the raw arrays
    """
    if any([isinstance(x, xr.DataArray) for x in data]):
        dims = []
        shape = []
        out = []
        coords = {}
        for d in data:
            if isinstance(d, xr.DataArray):
                for dim in d.dims:
                    if dim in dims:
                        assert len(d[dim]) == shape[dims.index(dim)]
                    else:
                        dims.append(dim)
                        shape.append(len(d[dim]))
                        coords[dim] = d.coords[dim]
            else:
                assert (
                    np.prod(np.shape(d)) == 1
                ), "Cannot mix `xr.DataArray`s and `np.ndarray`s"
        outzero = xr.DataArray(np.zeros(shape), dims=dims, coords=coords)
        out = [outzero + d for d in data]
        # for o in out:
        #    assert all(o.dims == dims) if len(o.dims) > 1 else o.dims == dims
        out = [d.data for d in out]
        return dims, shape, coords, out
    outzero = np.zeros(1)
    for d in data:
        outzero = outzero * d
    out = [outzero + d for d in data]
    dims = [f"dim_{d}" for d in range(len(outzero.shape))]
    shape = outzero.shape
    return dims, shape, None, out


def evaluate_at_xyz(self, x, y, z, *args, **kwargs):
    """
    See evaluate_at_rpz for options. Unlike evaluate_at_rpz the
    coordinates are given here in cartesian coordinates.
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return evaluate_at_rpz(self, r, phi, z, *args, **kwargs)


def evaluate_at_rpz(
    self,
    r,
    phi,
    z,
    key=None,
    delta_phi: float = None,
    fill_value=np.nan,
    lazy=False,
    progress=False,
):
    """
    Evaluate the field key in the dataset at the positions given by
    the array r, phi, z.  If key is None, return the indices to access
    the 3D field and get the appropriate values.

    Parameters
    ----------
    r : array-like
        The (major) radial coordinates to evaluate
    phi : array-like
        The toroidal coordinate
    z : array-like
        The z component
    key : None or str or sequence of str
        If None return the index-coordinates otherwise evaluate the
        specified field in the dataset
    delta_phi : None or float
        If not None, delta_phi gives the accuracy of the precision at
        which phi is evaluated. Giving a float enables caching of the
        phi slices, and can speed up computation. Note that it should
        be smaller then the toroidal resolution. For a grid with 1°
        resolution, delta_phi=2 * np.pi / 360 would be the upper
        bound.  None disables caching.
    fill_value : None or any
        If fill_value is None, missing data is initialised as
        np.nan, or as -1 for non-float datatypes. Otherwise
        fill_value is used to set missing data.
    lazy : bool
        Force the loading of the data for key. Defaults to False.
        Can significantly decrease performance, but can decrease
        memory usage.
    progress : bool
        Show the progress of the mapping. Defaults to False.
    """

    return _evalat(self, r, phi, z, key, delta_phi, fill_value, progress)


setattr(evaluate_at_rpz, "evaluate_at_rpz", BoutDatasetAccessor)
setattr(evaluate_at_xyz, "evaluate_at_xyz", BoutDatasetAccessor)
