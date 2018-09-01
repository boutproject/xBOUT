from xarray import Dataset

from xcollect.collect import collect


class BoutDataset:
    """
    Contains the BOUT output variables in the form of an xarray.Dataset, and optionally the input file as a dictionary.
    """

    def __init__(self, datapath='.', chunks={}, input_file=False):
        self.datapath = datapath
        self.chunks = {}
        # Should we just load whole dataset here?
        self.ds = None

        if input_file is True:
            # Load that using Ben's classes
            self.options = get_options_file(datapath)

        # This is where you would load the grid file as a separate object (using xgcm or otherwise)

    def __getitem__(self, var):
        if var == 'all':
            # Return entire dataset
            # Somehow find out all the variables to read from the datafiles
            # Then load them all before
            return self.ds

        if var not in self.ds.vars:
            # Collect and return the variable as a DataArray
            var_da = collect(var, path=self.datapath, chunks=self.chunks, lazy=True)
            self.ds[var] = var_da
            return var_da

    def __setitem__(self, key, value):
        self.ds[key] = value

    def options(self):
        return self.options

    def save(self, savepath='.', filetype='netcdf4'):
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            self.ds.to_netcdf(path=savepath, engine=filetype, compute=True)
        return
