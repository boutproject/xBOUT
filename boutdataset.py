from xarray import open_mfdataset, save_mfdataset
from dask.diagnostics import ProgressBar
from numpy import asscalar
from pprint import pprint

from boutdata.data import BoutOptionsFile

from xcollect.collect import collect


class BoutDataset:
    """
    Contains the BOUT output variables in the form of an xarray.Dataset, and optionally the input file.
    """

    def __init__(self, datapath='.', chunks={}, input_file=False, run_name=None, log_file=False, info=True):

        # Load data variables
        # Should we just load whole dataset here?
        self.datapath = datapath
        self.chunks = {}
        ds = open_mfdataset(datapath, chunks)
        # self.data is an xarray Dataset, self.metadata is a dict
        self.data, self.metadata = _strip_metadata(ds)
        if info:
            print('Read in data:\n')
            print(self.data)
            print('Read in metadata:\n')
            print(self.metadata)

        self.run_name = run_name

        if input_file is True:
            # Load options from input file using Ben's classes
            self.options = BoutOptionsFile(datapath + 'BOUT.inp')
            if info:
                print('Read in options:\n')
                pprint(self.options.as_dict())

        if log_file is True:
            # Read in important info from the log file
            # Especially the git commit hashes of the executables used
            log, bout_version, bout_git_commit_hash, module_git_commit_hash = read_log_file(datapath)
            self.log = log
            self.bout_version = bout_version
            self.bout_git_commit = bout_git_commit_hash
            self.module_git_commit = module_git_commit_hash

        # This is where you would load the grid file as a separate object
        # (Ideally using xgcm but could also just store the grid.nc file as another dataset)

    def data(self):
        """Return the xarray.Dataset containing all data variables."""
        # Somehow find out all the variables to read from the datafiles
        # Then load them all before returning entire dataset
        return self.data

    def metadata(self):
        """Return dictionary containing simulation metadata (non-data information like nype)."""
        return self.metadata

    def options(self):
        """Return boutdata.BoutOptionsFile instance containing all options specified in the BOUT.inp file."""
        return self.options

    def name(self):
        return self.run_name

    def __getitem__(self, var):
        if var not in self.data.vars:
            # Collect and return the variable as a DataArray
            var_da = collect(var, path=self.datapath, chunks=self.chunks, lazy=True)
            self.data[var] = var_da
            return var_da

    def __setitem__(self, key, value):
        self.data[key] = value

    def __str__(self):
        text = 'BoutDataset ' + self.run_name + '\n'
        text += 'Contains:\n'
        text += self.data.__str__
        text += 'With options:\n'
        text += self.options.__str__
        text += self.log
        return text

    def __repr__(self):
        return 'boutdata.BoutDataset(', {}, ',', {}, ')'.format(self.datapath, )

    def save(self, savepath='.', filetype='netcdf4', variables=None, save_dtype=None):
        """
        Save data variables.

        Parameters
        ----------
        savepath : str
        filetype : str
        variables : list of str
        """

        if variables is None:
            # Save all variables
            to_save = self.data
        else:
            to_save = self.data[variables]

        if save_dtype is not None:
            to_save = to_save.astype(save_dtype)

        # Store the metadata in the attributes dictionary rather than as data variables in the dataset
        to_save.attrs['metadata'] = self.metadata

        # TODO How should I store other data? In the attributes dict?
        # TODO Convert Ben's options class to a (flattened) nested dictionary then store it in ds.attrs?

        # Save data to a single file
        # TODO option to save each variable to a different netCDF file?
        with ProgressBar():
            # Should give it a descriptive filename (using the run name?)
            if variables is None:
                # Save all variables
                to_save.to_netcdf(path=savepath, engine=filetype, compute=True)
            else:
                to_save.data[variables].to_netcdf(path=savepath, engine=filetype, compute=True)

        return

    def save_restart(self, savepath='.', nxpe=None, nype=None, original_decomp=False):
        """
        Write out final timestep as a set of netCDF BOUT.restart files.

        If processor decomposition is not specified then data will be saved used the decomposition it had when loaded.

        Parameters
        ----------
        savepath : str
        nxpe : int
        nype : int
        """

        # Set processor decomposition if not given
        if original_decomp:
            if any([nxpe, nype]):
                raise ValueError('Inconsistent choices for domain decomposition.')
            else:
                nxpe, nype = self.metadata['NXPE'], self.metadata['NYPE']

        # Is this even possible without saving the ghost cells? Can they be recreated?
        restart_datasets, paths = _split_into_restarts(self.data, savepath, nxpe, nype)
        with ProgressBar():
            save_mfdataset(restart_datasets, paths, compute=True)
        return


def _strip_metadata(ds):
    """
    Extract the metadata (nxpe, myg etc.) from the Dataset.

    Assumes that all scalar variables are metadata, not physical data!
    """

    # Find only the scalar variables
    variables = list(ds.variables)
    scalar_vars = [var for var in variables if not any(dim in ['t', 'x', 'y', 'z'] for dim in ds[var].dims)]

    # Save metadata as a dictionary
    metadata_vals = [asscalar(ds[var].values) for var in scalar_vars]
    metadata = dict(zip(scalar_vars, metadata_vals))

    return ds.drop(scalar_vars), metadata
