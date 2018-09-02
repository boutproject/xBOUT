from xarray import Dataset

from boutdata.data import BoutOptionsFile

from xcollect.collect import collect


class BoutDataset:
    """
    Contains the BOUT output variables in the form of an xarray.Dataset, and optionally the input file.
    """

    def __init__(self, datapath='.', chunks={}, input_file=False, run_name=None, log_file=False):
        self.datapath = datapath
        self.chunks = {}
        # Should we just load whole dataset here?
        self.ds = None

        self.run_name = run_name

        if input_file is True:
            # Load options from input file using Ben's classes
            self.options = BoutOptionsFile(datapath + 'BOUT.inp')

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

    def name(self):
        return self.run_name

    def __str__(self):
        text = 'BoutDataset ' + self.run_name + '\n'
        text += 'Contains:\n'
        text += self.ds.__str__
        text += 'With options:\n'
        text += self.options.__str__
        text += self.log
        return text

    def __getitem__(self, var):
        if var not in self.ds.vars:
            # Collect and return the variable as a DataArray
            var_da = collect(var, path=self.datapath, chunks=self.chunks, lazy=True)
            self.ds[var] = var_da
            return var_da

    def __setitem__(self, key, value):
        self.ds[key] = value

    def all_data(self):
        # Somehow find out all the variables to read from the datafiles
        # Then load them all before returning entire dataset
        return self.ds

    def options(self):
        return self.options

    def save(self, savepath='.', filetype='netcdf4'):
        # Save data variables
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            # Should give it a descriptive filename (using the run name?)
            self.ds.to_netcdf(path=savepath, engine=filetype, compute=True)

        # How do I store other data? In the attributes dict?
        # Convert Ben's options class to a (flattened) nested dictionary then store it in ds.attrs?
        return

    def create_restart(self, savepath='.'):
        # Write out final timestep as a set of netCDF BOUT.restart files
        # Would need to specify the processor division
        # Is this even possible without saving the ghost cells? Can they be recreated?
        return
