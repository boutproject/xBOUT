from pathlib import Path

from xarray import Dataset, DataArray, open_dataset, merge
from xarray.testing import assert_equal
import pytest
import numpy as np

from xbout.load import open_boutdataset
from xbout.geometries import register_geometry, REGISTERED_GEOMETRIES


@pytest.fixture
def create_example_grid_file(tmpdir_factory):
    """
    Mocks up a set of BOUT-like netCDF files, and return the temporary test
    directory containing them.

    Deletes the temporary directory once that test is done.
    """

    # Create grid dataset
    arr = np.arange(6).reshape(2, 3)
    grid = DataArray(data=arr, dims=['x', 'y'])

    # Create temporary directory
    save_dir = tmpdir_factory.mktemp("griddata")

    # Save
    # Convert to str because tmpdir_factory returns a 'LocalPath' object, while xarray's
    # to_netcdf method expects a str, and 'Path' is compatible with 'LocalPath' in
    # Python-3.6 but not in Python-3.5
    filepath = str(save_dir + '/grid.nc')
    grid.to_netcdf(filepath, engine='netcdf4')

    return Path(filepath)


class TestOpenGrid:
    def test_open_grid(self, create_example_grid_file):
        example_grid = create_example_grid_file
        result = open_boutdataset(datapath=example_grid)
        assert_equal(result, open_dataset(example_grid))
        result.close()

    def test_open_grid_extra_dims(self, create_example_grid_file):
        example_grid = open_dataset(create_example_grid_file)

        new_var = DataArray(name='new', data=[[1, 2], [8, 9]], dims=['x', 'w'])
        # TODO this should be handled by pytest's tmpdir factory too
        dodgy_grid_path = 'dodgy_grid.nc'
        merge([example_grid, new_var]).to_netcdf(dodgy_grid_path,
                                                 engine='netcdf4')

        with pytest.warns(UserWarning, match="drop all variables containing "
                                             "the dimensions 'w'"):
            result = open_boutdataset(datapath=dodgy_grid_path)
        assert_equal(result, example_grid)
        result.close()

    def test_open_grid_apply_geometry(self, create_example_grid_file):
        @register_geometry(name="Schwarzschild")
        def add_schwarzschild_coords(ds, coordinates=None):
            ds['event_horizon'] = 4.0
            return ds

        example_grid = create_example_grid_file

        result = result = open_boutdataset(datapath=example_grid,
                                           geometry="Schwarzschild")
        assert_equal(result['event_horizon'], DataArray(4.0))

        # clean up
        del REGISTERED_GEOMETRIES["Schwarzschild"]
        result.close()
