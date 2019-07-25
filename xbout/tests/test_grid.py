from pathlib import Path

from xarray import Dataset, DataArray, open_dataset, merge
from xarray.testing import assert_equal
import pytest
import numpy as np

from xbout.grid import open_grid
from xbout.geometries import register_geometry, REGISTERED_GEOMETRIES


@pytest.fixture
def create_example_grid_file(tmpdir_factory):
    """
    Mocks up a set of BOUT-like netCDF files, and return the temporary test directory containing them.

    Deletes the temporary directory once that test is done.
    """

    # Create grid dataset
    arr = np.arange(6).reshape(2, 3)
    grid = DataArray(data=arr, dims=['x', 'y'])

    # Create temporary directory
    save_dir = tmpdir_factory.mktemp("griddata")

    # Save
    filepath = save_dir + '/grid.nc'
    grid.to_netcdf(filepath)

    return Path(filepath)


class TestOpenGrid:
    def test_open_grid(self, create_example_grid_file):
        example_grid = create_example_grid_file
        result = open_grid(gridfilepath=example_grid)
        assert_equal(result, open_dataset(example_grid))
        result.close()

    @pytest.mark.xfail(reason="unsolved bug in test")
    def test_open_grid_extra_dims(self):
        example_grid_path = create_example_grid_file
        # TODO this throws an error and I have no idea why
        # this doesn't also affect the previous test
        example_grid = open_dataset(example_grid_path)

        new_var = DataArray([[1, 2], [8, 9]], dims=['x', 'w'])
        # TODO this should be handled by pytest's tmpdir factory too
        dodgy_grid_path = 'dodgy_grid'
        merge([example_grid, new_var]).to_netcdf(dodgy_grid_path)

        with pytest.warns(Warning, match="Will drop all variables containing"
                                         " the dimensions w"):
            result = open_grid(gridfilepath=dodgy_grid_path)
        assert_equal(result, example_grid)
        result.close()

    @pytest.mark.skip
    def test_open_grid_merge_ds(self):
        ...

    def test_open_grid_apply_geometry(self, create_example_grid_file):
        @register_geometry(name="Schwarzschild")
        def add_schwarzschild_coords(ds):
            ds['event_horizon'] = 4.0
            return ds

        example_grid = create_example_grid_file

        result = open_grid(gridfilepath=example_grid, geometry="Schwarzschild")
        assert_equal(result['event_horizon'], DataArray(4.0))

        # clean up
        del REGISTERED_GEOMETRIES["Schwarzschild"]
        result.close()
