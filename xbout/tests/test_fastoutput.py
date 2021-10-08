import numpy as np
import random

import pytest

from xarray import DataArray, Dataset
import xarray.testing as xrt

from xbout.fastoutput import open_fastoutput


class TestFastOutput:
    def test_open_fastoutput(self, tmp_path):
        datapath_list, xinds, yinds, zinds = make_fastoutput_set(tmp_path, 4)

        ds = open_fastoutput(tmp_path.joinpath("BOUT.fast.*.nc"))

        expected_x = list(set(xinds))
        expected_x.sort()
        expected_y = list(set(yinds))
        expected_y.sort()
        expected_z = list(set(zinds))
        expected_z.sort()

        assert ds.sizes["t"] == 3

        assert all(ds["x"] == expected_x)
        assert all(ds["y"] == expected_y)
        assert all(ds["z"] == expected_z)

        for v in ("n", "T"):
            assert v in ds
            assert "t" in ds[v].dims
            assert "x" in ds[v].dims
            assert "y" in ds[v].dims
            assert "z" in ds[v].dims


def make_fastoutput_set(path, n):
    """
    Create a set of FastOutput files/Datasets, with random data and location indices.

    Parameters
    ----------
    path : pathlib.Path
        Directory to save the files to.
    n : int
        Number of output files/Datasets to create

    Returns
    -------
    result_list : list of pathlib.Path or Dataset
        Output files/Datasets in format duplicating files produced by FastOutput
    xinds : list of int
        x-indices where data exists
    yinds : list of int
        y-indices where data exists
    zinds : list of int
        z-indices where data exists
    """
    # Choose a fixed seed to ensure reproducibility
    random.seed(83)

    procinds = random.sample(range(10 * n), n)
    npoints = [random.randint(1, 4) for _ in procinds]
    total_points = sum(npoints)
    xinds = random.sample(range(max(100, total_points)), total_points)
    yinds = random.sample(range(max(100, total_points)), total_points)
    zinds = random.sample(range(max(100, total_points)), total_points)

    # Add some points which share an x-, y- or z-index (but not all three) with other
    # points
    if total_points < 5:
        raise ValueError(f"Not enough points for n={n}. Increase n.")
    xinds[1] = xinds[0]
    yinds[1] = yinds[0]
    zinds[4] = zinds[3]

    result_list = []

    counter = 0
    for i, procind in enumerate(procinds):
        locations = {}
        for _ in range(npoints[i]):
            locations[counter] = (xinds[counter], yinds[counter], zinds[counter])
            counter += 1
        result_list.append(make_fastoutput(path, procind, locations))

    return result_list, xinds, yinds, zinds


def make_fastoutput(path, i, locations):
    """
    Create a single file in the format produced by
    [`FastOutput`](https://github.com/johnomotani/BoutFastOutput)

    Parameters
    ----------
    path : pathlib.Path
        Directory to save the file to.
    i : int
        Processor index of the file.
    locations : dict of {int: tuple of int}
        Keys are the FastOutput indices of the variables to be saved.
        Values are 3-tuples of ints giving x, y, and z indices associated with the key.
    """

    nt = 3

    # Each variable contains unique random noise
    np.random.seed(seed=i)

    ds = Dataset()

    time = DataArray(np.linspace(0.0, 0.5, nt), dims="t")
    ds["time"] = time

    for j, (xind, yind, zind) in locations.items():
        n = DataArray(np.random.randn(nt), dims="t")
        T = DataArray(np.random.randn(nt), dims="t")

        for v in [n, T]:
            v.attrs["ix"] = xind
            v.attrs["iy"] = yind
            v.attrs["iz"] = zind

        ds[f"n{j}"] = n
        ds[f"T{j}"] = T

    filepath = path.joinpath(f"BOUT.fast.{i}.nc")
    ds.to_netcdf(filepath)
