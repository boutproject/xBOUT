from xarray import register_dataset_accessor, save_mfdataset, set_options, merge

from dask.diagnostics import ProgressBar

from pprint import pprint
from pathlib import Path

from boutdata.data import BoutOptionsFile

from .load import _auto_open_mfboutdataset


# This code should run whenever any function from this module is imported
# Set all attrs to survive all mathematical operations
# (see https://github.com/pydata/xarray/pull/2482)
try:
    set_options(keep_attrs=True)
except ValueError:
    raise ImportError("For dataset attributes to be permanent you need to be "
                      "using the development version of xarray - found at "
                      "https://github.com/pydata/xarray/")
try:
    set_options(file_cache_maxsize=256)
except ValueError:
    raise ImportError("For open and closing of netCDF files correctly you need"
                      " to be using the development version of xarray - found"
                      " at https://github.com/pydata/xarray/")

# TODO somehow check that we have access to the latest version of auto_combine


def open_boutdataset(datapath='./BOUT.dmp.*.nc', chunks={},
                     inputfilepath=None, gridfilepath=None,
                     run_name=None, info=True):
    """
    Load a dataset from a set of BOUT output files, including the input options file.

    Parameters
    ----------
    datapath : str, optional
    prefix : str, optional
    slices : slice object, optional
    chunks : dict, optional
    inputfilepath : str, optional
    gridfilepath : str, optional
    run_name : str, optional
    info : bool, optional

    Returns
    -------
    ds : xarray.Dataset
    """

    # TODO handle possibility that we are loading a previously saved (and trimmed dataset)

    # Gather pointers to all numerical data from BOUT++ output files
    ds, metadata = _auto_open_mfboutdataset(datapath=datapath, chunks=chunks, info=info)
    ds.attrs['metadata'] = metadata

    if inputfilepath:
        # Use Ben's options class to store all input file options
        options = BoutOptionsFile(inputfilepath)
    else:
        options = None
    ds.attrs['options'] = options

    # TODO This is where you would load the grid file as a separate object
    # (Ideally using xgcm but could also just store the grid.nc file as another dataset)

    # TODO read and store git commit hashes from output files

    if run_name:
        ds.name = run_name

    if info:
        print('Read in BOUT data:')
        print(ds)
        print('Read in BOUT metadata:')
        print(metadata)
        print('Read in BOUT options:')
        print(options)

    return ds


@register_dataset_accessor('bout')
class BoutAccessor(object):
    """
    Contains BOUT-specific methods to use on BOUT++ datasets opened using open_boutdataset.

    These BOUT-specific methods and attributes are accessed via the bout accessor, e.g. ds.bout.options returns a
    BoutOptionsFile instance.
    """

    # TODO a BoutDataarray class which uses the register_dataarray_accessor??

    def __init__(self, ds):

        # # Load data variables
        # # Should we just load whole dataset here?
        # self.datapath = datapath
        # self.prefix = prefix

        self.data = ds
        self.metadata = ds.attrs['metadata']
        self.options = ds.attrs['options']

    def __str__(self):
        """
        String represenation of the BoutDataset.

        Accessed by print(ds.bout)
        """

        text = 'BoutDataset\n'
        text += 'Contains:\n'
        text += self.data.__str__()
        text += 'with metadata'
        text += self.metadata.__str__()
        if self.options:
            text += 'and options:\n'
            text += self.options.__str__()
        return text

    #def __repr__(self):
    #    return 'boutdata.BoutDataset(', {}, ',', {}, ')'.format(self.datapath, self.prefix)

    def save(self, savepath='./boutdata.nc', filetype='NETCDF4', variables=None, save_dtype=None, separate_vars=False):
        """
        Save data variables to a netCDF file.

        Parameters
        ----------
        savepath : str, optional
        filetype : str, optional
        variables : list of str, optional
            Variables from the dataset to save. Default is to save all of them.
        separate_vars: bool, optional
            If this is true then every variable which depends on time will be saved into a different output file.
            The files are labelled by the name of the variable. Variables which don't depend on time will be present in
            every output file.
        """

        if variables is None:
            # Save all variables
            to_save = self.data
        else:
            to_save = self.data[variables]

        if savepath is './boutdata.nc':
            print('Will save data into the current working directory, named as boutdata_[var].nc')
        if savepath is None:
            raise ValueError('Must provide a path to which to save the data.')

        if save_dtype is not None:
            to_save = to_save.astype(save_dtype)

        # TODO How should I store other data? In the attributes dict?
        # TODO Convert Ben's options class to a (flattened) nested dictionary then store it in ds.attrs?
        if self.options is None:
            to_save.attrs = {}
        else:
            # Store the metadata in the attributes dictionary rather than as data variables in the dataset
            to_save.attrs['metadata'] = self.metadata

        if separate_vars:
            # Save each time-dependent variable to a different netCDF file

            # Determine which variables are time-dependent
            time_dependent_vars, time_independent_vars = _find_time_dependent_vars(to_save)

            print('Will save the variables ' + str(time_dependent_vars) + ' separately')

            # Save each one to separate file
            for var in time_dependent_vars:
                # Group variables so that there is only one time-dependent variable saved in each file
                time_independent_data = [to_save[time_ind_var] for time_ind_var in time_independent_vars]
                single_var_ds = merge([to_save[var], *time_independent_data])

                # Include the name of the variable in the name of the saved file
                path = Path(savepath)
                var_savepath = str(path.parent / path.stem) + '_' + str(var) + path.suffix
                print('Saving ' + var + ' data...')
                with ProgressBar():
                    single_var_ds.to_netcdf(path=str(var_savepath), format=filetype, compute=True)
        else:
            # Save data to a single file
            print('Saving data...')
            with ProgressBar():
                to_save.to_netcdf(path=savepath, format=filetype, compute=True)

        return

    def to_restart(self, savepath='.', nxpe=None, nype=None, original_decomp=False):
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

    # TODO BOUT-specific plotting functionality would be implemented as methods here, e.g. ds.bout.plot_poloidal
    # TODO Could trial a 2D surface plotting method here

    def animate(self, var, animate_over='t', x='x', y='y', sep_pos=None,
                aspect='auto', fps=10, save_as=None, writer='imagemagick',
                **kwargs):
        """
        Plots a color plot which is animated with time over the specified
        coordinate.

        Currently only supports 2D+1 data, which it plots with xarray's
        wrapping of matplotlib's imshow.

        Parameters
        ----------
        var : str
            Variable to plot
        animate_over : str, optional
            Coordinate over which to animate
        x : str, optional
        y : str, optional
        sep_x : int, optional
            Radial position of the separatrix as an integer
        save_as: str, optional
            Filename to give to the resulting gif
        fps : int, optional
        aspect : {'auto', 'equal'} or int, optional
            Method for choosing aspect ratio of 2D plot. 'auto' uses the number
             of grid points, 'equal' uses
        kwargs : dict, optional
            Additional keyword arguments are passed on to the plotting function
            (e.g. imshow for 2D plots).
        """


        from .animate import _animated_plot

        _animated_plot(self.data, var, animate_over=animate_over,
                       x=x, y=y, sep_pos=sep_pos, aspect=aspect, fps=fps,
                       save_as=save_as, writer=writer, **kwargs)

    def animatplot(self, var, animate_over='t', x='x', y='y', animate=True,
                   fps=10, save_as=None, sep_pos=None, **kwargs):

        import animatplot as amp
        import numpy as np

        # TODO implement this as a method on a BOUT DataArray accessor instead
        # so that the var argument is not needed
        data = self.data[var]

        # TODO add a colorbar
        # TODO add a title

        variable = data.name
        n_dims = len(data.dims)
        if n_dims == 3:
            print("{} data passed has {} dimensions - will use "
                  "xarray.plot.imshow()".format(variable, str(n_dims)))
        else:
            raise ValueError(
                "Data passed has an unsupported number of dimensions "
                "({})".format(str(n_dims)))

        # Check plot is the right orientation
        y_read, x_read = data.isel(**{animate_over: 0}).dims
        if (x_read is x) & (y_read is y):
            transpose = False
        elif (x_read is y) & (y_read is x):
            transpose = True
        else:
            raise ValueError

        # Construct data to plot
        images = []
        for i in np.arange(data.sizes[animate_over]):
            frame_data = data.isel(**{animate_over: i})
            if transpose:
                frame_data = frame_data.T

            # Load values eagerly otherwise for some reason the plotting takes
            # 100's of times longer - for some reason animatplot does not deal
            # well with dask arrays!
            images.append(frame_data.values)

        if not save_as:
            save_as = "{}_over_{}".format(variable, animate_over)

        block = amp.blocks.Imshow(images, **kwargs)

        if animate:
            anim = amp.Animation([block])

            anim.controls()
            anim.save_gif(save_as)

        return block


def _find_time_dependent_vars(data):
    evolving_vars = set(var for var in data.data_vars if 't' in data[var].dims)
    time_independent_vars = set(data.data_vars) - set(evolving_vars)
    return list(evolving_vars), list(time_independent_vars)
