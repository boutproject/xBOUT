import operator
import re
from copy import deepcopy
from functools import reduce
from pathlib import Path

import numpy as np
import pytest
from xarray import DataArray

from xbout.tests.utils_for_tests import (
    _get_kwargs,
    create_bout_ds_list,
    create_bout_grid_ds,
)


@pytest.fixture(scope="session")
def bout_xyt_example_files(tmp_path_factory):
    return _bout_xyt_example_files


_bout_xyt_example_files_cache = {}


def _bout_xyt_example_files(
    tmp_path_factory,
    prefix="BOUT.dmp",
    lengths=(6, 2, 4, 7),
    nxpe=4,
    nype=2,
    nt=1,
    guards=None,
    syn_data_type="random",
    grid=None,
    squashed=False,
    topology="core",
    write_to_disk=False,
    bout_v5=False,
    metric_3D=False,
):
    """
    Mocks up a set of BOUT-like Datasets

    Either returns list of Datasets (if write_to_disk=False)
    or writes Datasets to netCDF files and returns the temporary test directory
    containing them, deleting the temporary directory once that test is done (if
    write_to_disk=True).
    """
    call_args = _get_kwargs(ignore="tmp_path_factory")

    try:
        # Has been called with the same signature before, just return the cached result
        return deepcopy(_bout_xyt_example_files_cache[call_args])
    except KeyError:
        pass

    if guards is None:
        guards = {}

    if squashed:
        # create a single data-file, but alter the 'nxpe' and 'nype' variables, as if the
        # file had been created by combining a set of BOUT.dmp.*.nc files
        this_lengths = (
            lengths[0],
            lengths[1] * nxpe,
            lengths[2] * nype,
            lengths[3],
        )
        ds_list, file_list = create_bout_ds_list(
            prefix=prefix,
            lengths=this_lengths,
            nxpe=1,
            nype=1,
            nt=nt,
            guards=guards,
            topology=topology,
            syn_data_type=syn_data_type,
            squashed=True,
            bout_v5=bout_v5,
            metric_3D=metric_3D,
        )
        ds_list[0]["nxpe"] = nxpe
        ds_list[0]["nype"] = nype
    else:
        ds_list, file_list = create_bout_ds_list(
            prefix=prefix,
            lengths=lengths,
            nxpe=nxpe,
            nype=nype,
            nt=nt,
            guards=guards,
            topology=topology,
            syn_data_type=syn_data_type,
            bout_v5=bout_v5,
            metric_3D=metric_3D,
        )

    if grid is not None:
        xsize = lengths[1] * nxpe
        ysize = lengths[2] * nype
        grid_ds = create_bout_grid_ds(
            xsize=xsize,
            ysize=ysize,
            guards=guards,
            topology=topology,
            ny_inner=3 * lengths[2],
        )

    if not write_to_disk:
        if grid is None:
            _bout_xyt_example_files_cache[call_args] = ds_list
            return deepcopy(ds_list)
        else:
            _bout_xyt_example_files_cache[call_args] = ds_list, grid_ds
            return deepcopy((ds_list, grid_ds))
        raise ValueError("tmp_path_factory required when write_to_disk=True")

    save_dir = tmp_path_factory.mktemp("data")

    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(save_dir.joinpath(file_name))

    if grid is not None:
        grid_ds.to_netcdf(save_dir.joinpath(grid + ".nc"))

    # Return a glob-like path to all files created, which has all file numbers replaced
    # with a single asterix
    path = str(save_dir.joinpath(file_list[-1]))

    count = 1
    if nt > 1:
        count += 1
    # We have to reverse the path before limiting the number of numbers replaced so that the
    # tests don't get confused by pytest's persistent temporary directories (which are also designated
    # by different numbers)
    glob_pattern = Path((re.sub(r"\d+", "*", path[::-1], count=count))[::-1])
    _bout_xyt_example_files_cache[call_args] = glob_pattern
    return glob_pattern


fci_shape = (2, 2, 3, 4)
fci_guards = (2, 2, 0)


@pytest.fixture
def create_example_grid_file_fci(tmp_path_factory):
    """
    Mocks up a FCI-like netCDF file, and return the temporary test
    directory containing them.

    Deletes the temporary directory once that test is done.
    """

    # Create grid dataset
    shape = (fci_shape[1] + 2 * fci_guards[0], *fci_shape[2:])
    arr = np.arange(reduce(operator.mul, shape, 1)).reshape(shape)
    grid = DataArray(data=arr, name="R", dims=["x", "y", "z"]).to_dataset()
    grid["Z"] = DataArray(np.random.random(shape), dims=["x", "y", "z"])
    grid["dy"] = DataArray(np.ones(shape), dims=["x", "y", "z"])
    grid = grid.set_coords(["dy"])

    # Create temporary directory
    save_dir = tmp_path_factory.mktemp("griddata")

    # Save
    filepath = save_dir.joinpath("fci.nc")
    grid.to_netcdf(filepath, engine="netcdf4")

    return filepath


@pytest.fixture
def create_example_files_fci(tmp_path_factory):
    return _bout_xyt_example_files(
        tmp_path_factory,
        lengths=fci_shape,
        nxpe=1,
        nype=1,
        # nt=1,
        guards={a: b for a, b in zip("xyz", fci_guards)},
        syn_data_type="random",
        grid=None,
        squashed=False,
        # topology="core",
        write_to_disk=False,
        bout_v5=True,
        metric_3D=True,
    )
