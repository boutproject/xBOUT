from xarray import Dataset, DataArray, open_dataset, merge
from xarray.testing import assert_equal
import pytest
import numpy as np

from xbout.load import open_boutdataset
from xbout.geometries import register_geometry, REGISTERED_GEOMETRIES


@pytest.fixture
def create_example_grid_file(tmp_path_factory):
    """
    Mocks up a set of BOUT-like netCDF files, and return the temporary test
    directory containing them.

    Deletes the temporary directory once that test is done.
    """

    # Create grid dataset
    arr = np.arange(15).reshape(5, 3)
    grid = DataArray(data=arr, name="arr", dims=["x", "y"]).to_dataset()
    grid["dy"] = DataArray(np.ones((5, 3)), dims=["x", "y"])
    grid = grid.set_coords(["dy"])

    # Create temporary directory
    save_dir = tmp_path_factory.mktemp("griddata")

    # Save
    filepath = save_dir.joinpath("grid.nc")
    grid.to_netcdf(filepath, engine="netcdf4")

    return filepath


class TestOpenGrid:
    def test_open_grid(self, create_example_grid_file):
        example_grid = create_example_grid_file
        with pytest.warns(UserWarning):
            result = open_boutdataset(datapath=example_grid)
        result = result.drop_vars(["x", "y"])
        assert_equal(result, open_dataset(example_grid))
        result.close()

    def test_open_grid_extra_dims(self, create_example_grid_file, tmp_path_factory):
        example_grid = open_dataset(create_example_grid_file)

        new_var = DataArray(
            name="new",
            data=[[1, 2], [8, 9], [16, 17], [27, 28], [37, 38]],
            dims=["x", "w"],
        )

        dodgy_grid_directory = tmp_path_factory.mktemp("dodgy_grid")
        dodgy_grid_path = dodgy_grid_directory.joinpath("dodgy_grid.nc")
        merge([example_grid, new_var]).to_netcdf(dodgy_grid_path, engine="netcdf4")

        with pytest.warns(
            UserWarning, match="drop all variables containing " "the dimensions 'w'"
        ):
            result = open_boutdataset(datapath=dodgy_grid_path)
        result = result.drop_vars(["x", "y"])
        assert_equal(result, example_grid)
        result.close()

    def test_open_grid_apply_geometry(self, create_example_grid_file):
        @register_geometry(name="Schwarzschild")
        def add_schwarzschild_coords(ds, coordinates=None):
            ds["event_horizon"] = 4.0
            ds["event_horizon"].attrs = ds.attrs.copy()
            return ds

        example_grid = create_example_grid_file

        result = result = open_boutdataset(
            datapath=example_grid, geometry="Schwarzschild"
        )
        assert_equal(result["event_horizon"], DataArray(4.0))

        # clean up
        del REGISTERED_GEOMETRIES["Schwarzschild"]
        result.close()

    def test_open_grid_chunks(self, create_example_grid_file):
        example_grid = create_example_grid_file
        with pytest.warns(UserWarning):
            result = open_boutdataset(datapath=example_grid, chunks={"x": 4, "y": 5})
        result = result.drop_vars(["x", "y"])
        assert_equal(result, open_dataset(example_grid))
        result.close()

    def test_open_grid_chunks_not_in_grid(self, create_example_grid_file):
        example_grid = create_example_grid_file
        with pytest.warns(UserWarning):
            result = open_boutdataset(
                datapath=example_grid, chunks={"anonexistantdimension": 5}
            )
        result = result.drop_vars(["x", "y"])
        assert_equal(result, open_dataset(example_grid))
        result.close()
