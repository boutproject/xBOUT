import pytest

from xarray import Dataset, DataArray, concat, open_dataset, open_mfdataset
import xarray.testing as xrt
import numpy as np
from pathlib import Path

from xbout.tests.test_load import bout_xyt_example_files, create_bout_ds
from xbout import BoutDatasetAccessor, open_boutdataset
from xbout.geometries import apply_geometry
from xbout.utils import _set_attrs_on_all_vars


EXAMPLE_OPTIONS_FILE_PATH = './xbout/tests/data/options/BOUT.inp'


class TestBoutDatasetIsXarrayDataset:
    """
    Set of tests to check that BoutDatasets behave similarly to xarray Datasets.
    (With the accessor approach these should pass trivially now.)
    """

    def test_concat(self, tmpdir_factory, bout_xyt_example_files):
        path1 = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        bd1 = open_boutdataset(datapath=path1, inputfilepath=None,
                               keep_xboundaries=False)
        path2 = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        bd2 = open_boutdataset(datapath=path2, inputfilepath=None,
                               keep_xboundaries=False)
        result = concat([bd1, bd2], dim='run')
        assert result.dims == {**bd1.dims, 'run': 2}

    def test_isel(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        bd = open_boutdataset(datapath=path, inputfilepath=None,
                              keep_xboundaries=False)
        actual = bd.isel(x=slice(None,None,2))
        expected = bd.bout.data.isel(x=slice(None,None,2))
        xrt.assert_equal(actual, expected)


class TestBoutDatasetMethods:
    @pytest.mark.skip
    def test_test_method(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None)
        #ds = collect(path=path)
        #bd = BoutAccessor(ds)
        print(ds)
        #ds.bout.test_method()
        #print(ds.bout.options)
        #print(ds.bout.metadata)
        print(ds.isel(t=-1))

        #ds.bout.set_extra_data('stored')
        ds.bout.extra_data = 'stored'

        print(ds.bout.extra_data)

    def test_getFieldAligned(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=3, nype=4, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=None, keep_xboundaries=False)

        ds['psixy'] = ds['x']
        ds['Rxy'] = ds['x']
        ds['Zxy'] = ds['y']

        ds = apply_geometry(ds, 'toroidal')

        n = ds['n']
        n.attrs['direction_y'] = 'Standard'
        n_aligned_from_array = n.bout.toFieldAligned()

        # check n_aligned does not exist yet
        assert 'n_aligned' not in ds

        n_aligned_from_ds = ds.bout.getFieldAligned('n')
        xrt.assert_allclose(n_aligned_from_ds, n_aligned_from_array)
        xrt.assert_allclose(ds['n_aligned'], n_aligned_from_array)

        # check getting the cached version
        ds['n_aligned'] = ds['T']
        xrt.assert_allclose(ds.bout.getFieldAligned('n'), ds['T'])

    def test_resetParallelInterpFactor(self):
        ds = Dataset()
        ds['a'] = DataArray()
        ds = _set_attrs_on_all_vars(ds, 'metadata', {})

        with pytest.raises(KeyError):
            ds.metadata['fine_interpolation_factor']
        with pytest.raises(KeyError):
            ds['a'].metadata['fine_interpolation_factor']

        ds.bout.resetParallelInterpFactor(42)

        assert ds.metadata['fine_interpolation_factor'] == 42
        assert ds['a'].metadata['fine_interpolation_factor'] == 42


class TestLoadInputFile:
    @pytest.mark.skip
    def test_load_options(self):
        from boutdata.data import BoutOptionsFile, BoutOptions
        options = BoutOptionsFile(EXAMPLE_OPTIONS_FILE_PATH)
        assert isinstance(options, BoutOptions)
        # TODO Check it contains the same text

    @pytest.mark.skip
    def test_load_options_in_dataset(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)
        ds = open_boutdataset(datapath=path, inputfilepath=EXAMPLE_OPTIONS_FILE_PATH)
        assert isinstance(ds.options, BoutOptions)


@pytest.mark.skip(reason="Not yet implemented")
class TestLoadLogFile:
    pass

@pytest.mark.skip(reason="Need to sort out issue with saving metadata")
class TestSave:
    @pytest.mark.parametrize("options", [False, True])
    def test_save_all(self, tmpdir_factory, bout_xyt_example_files, options):
        # Create data
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=5, nt=1)

        # Load it as a boutdataset
        original = open_boutdataset(datapath=path, inputfilepath=None)
        if not options:
            original.attrs['options'] = {}

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + 'temp_boutdata.nc'
        original.bout.save(savepath=savepath)

        # Load it again using bare xarray
        recovered = open_dataset(savepath)

        # Compare
        xrt.assert_equal(original, recovered)

    @pytest.mark.parametrize("save_dtype", [np.float64, np.float32])
    def test_save_dtype(self, tmpdir_factory, bout_xyt_example_files, save_dtype):

        # Create data
        path = bout_xyt_example_files(tmpdir_factory, nxpe=1, nype=1, nt=1)

        # Load it as a boutdataset
        original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + 'temp_boutdata.nc'
        original.bout.save(savepath=savepath, save_dtype=np.dtype(save_dtype))

        # Load it again using bare xarray
        recovered = open_dataset(savepath)

        assert recovered['n'].values.dtype == np.dtype(save_dtype)

    def test_save_separate_variables(self, tmpdir_factory, bout_xyt_example_files):
        path = bout_xyt_example_files(tmpdir_factory, nxpe=4, nype=1, nt=1)

        # Load it as a boutdataset
        original = open_boutdataset(datapath=path, inputfilepath=None)

        # Save it to a netCDF file
        savepath = str(Path(path).parent) + '/temp_boutdata.nc'
        original.bout.save(savepath=savepath, separate_vars=True)

        for var in ['n', 'T']:
            # Load it again using bare xarray
            savepath = str(Path(path).parent) + '/temp_boutdata_' + var + '.nc'
            recovered = open_dataset(savepath)

            # Compare
            xrt.assert_equal(recovered[var], original[var])


class TestSaveRestart:
    pass
