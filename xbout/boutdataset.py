import collections
from pprint import pformat as prettyformat
from functools import partial
from itertools import chain
from pathlib import Path
import warnings
import gc

import xarray as xr
import animatplot as amp
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter

from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from dask.diagnostics import ProgressBar

from .plotting.animate import animate_poloidal, animate_pcolormesh, animate_line
from .plotting.utils import _create_norm


@xr.register_dataset_accessor('bout')
class BoutDatasetAccessor:
    """
    Contains BOUT-specific methods to use on BOUT++ datasets opened using
    `open_boutdataset()`.

    These BOUT-specific methods and attributes are accessed via the bout
    accessor, e.g. `ds.bout.options` returns a `BoutOptionsFile` instance.
    """

    def __init__(self, ds):
        self.data = ds
        self.metadata = ds.attrs.get('metadata')  # None if just grid file
        self.options = ds.attrs.get('options')  # None if no inp file

    def __str__(self):
        """
        String representation of the BoutDataset.

        Accessed by print(ds.bout)
        """

        styled = partial(prettyformat, indent=4, compact=True)
        text = "<xbout.BoutDataset>\n" + \
               "Contains:\n{}\n".format(str(self.data)) + \
               "Metadata:\n{}\n".format(styled(self.metadata))
        if self.options:
            text += "Options:\n{}".format(styled(self.options))
        return text

    #def __repr__(self):
    #    return 'boutdata.BoutDataset(', {}, ',', {}, ')'.format(self.datapath,
    #  self.prefix)

    def getFieldAligned(self, name, caching=True):
        """
        Get a field-aligned version of a variable, calculating (and caching in the
        Dataset) if necessary

        Parameters
        ----------
        name : str
            Name of the variable to get field-aligned version of
        caching : bool, optional
            Save the field-aligned variable in the Dataset (default: True)
        """
        aligned_name = name + '_aligned'
        try:
            result = self.data[aligned_name]
            if result.direction_y != 'Aligned':
                raise ValueError(aligned_name + " exists, but is not field-aligned, it "
                                 "has direction_y=" + result.direction_y)
            return result
        except KeyError:
            if caching:
                self.data[aligned_name] = self.data[name].bout.toFieldAligned()
            return self.data[aligned_name]

    def save(self, savepath='./boutdata.nc', filetype='NETCDF4',
             variables=None, save_dtype=None, separate_vars=False, pre_load=False):
        """
        Save data variables to a netCDF file.

        Parameters
        ----------
        savepath : str, optional
        filetype : str, optional
        variables : list of str, optional
            Variables from the dataset to save. Default is to save all of them.
        separate_vars: bool, optional
            If this is true then every variable which depends on time (but not
            solely on time) will be saved into a different output file.
            The files are labelled by the name of the variable. Variables which
            don't meet this criterion will be present in every output file.
        pre_load : bool, optional
            When saving separate variables, will load each variable into memory
            before saving to file, which can be considerably faster.

        Examples
        --------
        If `separate_vars=True`, then multiple files will be created. These can
        all be opened and merged in one go using a call of the form:
        
        ds = xr.open_mfdataset('boutdata_*.nc', combine='nested', concat_dim=None)
        """

        if variables is None:
            # Save all variables
            to_save = self.data
        else:
            to_save = self.data[variables]

        if savepath == './boutdata.nc':
            print("Will save data into the current working directory, named as"
                  " boutdata_[var].nc")
        if savepath is None:
            raise ValueError('Must provide a path to which to save the data.')

        if save_dtype is not None:
            # Workaround to keep attributes while calling astype. See
            # https://github.com/pydata/xarray/issues/2049
            # https://github.com/pydata/xarray/pull/2070
            for da in chain(to_save.values(), to_save.coords.values()):
                da.data = da.data.astype(save_dtype)

        # make shallow copy of Dataset, so we do not modify the attributes of the data
        # when we change things to save
        to_save = to_save.copy()

        options = to_save.attrs.pop('options')
        if options:
            # TODO Convert Ben's options class to a (flattened) nested
            # dictionary then store it in ds.attrs?
            warnings.warn(
                "Haven't decided how to write options file back out yet - deleting "
                "options for now. To re-load this Dataset, pass the same inputfilepath "
                "to open_boutdataset when re-loading."
            )
        # Delete placeholders for options on each variable and coordinate
        for var in chain(to_save.data_vars, to_save.coords):
            try:
                del to_save[var].attrs['options']
            except KeyError:
                pass

        # Store the metadata as individual attributes instead because
        # netCDF can't handle storing arbitrary objects in attrs
        def dict_to_attrs(obj, section):
            for key, value in obj.attrs.pop(section).items():
                obj.attrs[section + ":" + key] = value
        dict_to_attrs(to_save, 'metadata')
        # Must do this for all variables and coordinates in dataset too
        for varname, da in chain(to_save.data_vars.items(), to_save.coords.items()):
            try:
                dict_to_attrs(da, 'metadata')
            except KeyError:
                pass

        if 'regions' in to_save.attrs:
            # Do not need to save regions as these can be reconstructed from the metadata
            del to_save.attrs['regions']
            for var in chain(to_save.data_vars, to_save.coords):
                try:
                    del to_save[var].attrs['regions']
                except KeyError:
                    pass

        if separate_vars:
            # Save each major variable to a different netCDF file

            # Determine which variables are "major"
            # Defined as time-dependent, but not solely time-dependent
            major_vars, minor_vars = _find_major_vars(to_save)

            print("Will save the variables {} separately"
                  .format(str(major_vars)))

            # Save each one to separate file
            # TODO perform the save in parallel with save_mfdataset?
            for major_var in major_vars:
                # Group variables so that there is only one time-dependent
                # variable saved in each file
                minor_data = [to_save[minor_var] for minor_var in minor_vars]
                single_var_ds = xr.merge([to_save[major_var], *minor_data])

                # Add the attrs back on
                single_var_ds.attrs = to_save.attrs

                if pre_load:
                    single_var_ds.load()

                # Include the name of the variable in the name of the saved
                # file
                path = Path(savepath)
                var_savepath = str(path.parent / path.stem) + '_' \
                               + str(major_var) + path.suffix
                print('Saving ' + major_var + ' data...')
                with ProgressBar():
                    single_var_ds.to_netcdf(path=str(var_savepath),
                                            format=filetype, compute=True)

                # Force memory deallocation to limit RAM usage
                single_var_ds.close()
                del single_var_ds
                gc.collect()
        else:
            # Save data to a single file
            print('Saving data...')
            with ProgressBar():
                to_save.to_netcdf(path=savepath, format=filetype, compute=True)

        return

    def to_restart(self, savepath='.', nxpe=None, nype=None,
                   original_splitting=False):
        """
        Write out final timestep as a set of netCDF BOUT.restart files.

        If processor decomposition is not specified then data will be saved
        using the decomposition it had when loaded.

        Parameters
        ----------
        savepath : str
        nxpe : int
        nype : int
        """

        # Set processor decomposition if not given
        if original_splitting:
            if any([nxpe, nype]):
                raise ValueError('Inconsistent choices for domain '
                                 'decomposition.')
            else:
                nxpe, nype = self.metadata['NXPE'], self.metadata['NYPE']

        # Is this even possible without saving the guard cells?
        # Can they be recreated?
        restart_datasets, paths = _split_into_restarts(self.data, savepath,
                                                       nxpe, nype)
        with ProgressBar():
            xr.save_mfdataset(restart_datasets, paths, compute=True)
        return

    def animate_list(self, variables, animate_over='t', save_as=None, show=False, fps=10,
                     nrows=None, ncols=None, poloidal_plot=False, subplots_adjust=None,
                     vmin=None, vmax=None, logscale=None, titles=None, aspect='equal',
                     controls=True, tight_layout=True, **kwargs):
        """
        Parameters
        ----------
        variables : list of str or BoutDataArray
            The variables to plot. For any string passed, the corresponding
            variable in this DataSet is used - then the calling DataSet must
            have only 3 dimensions. It is possible to pass BoutDataArrays to
            allow more flexible plots, e.g. with different variables being
            plotted against different axes.
        animate_over : str, optional
            Dimension over which to animate
        save_as : str, optional
            If passed, a gif is created with this filename
        show : bool, optional
            Call pyplot.show() to display the animation
        fps : float, optional
            Indicates the number of frames per second to play
        nrows : int, optional
            Specify the number of rows of plots
        ncols : int, optional
            Specify the number of columns of plots
        poloidal_plot : bool or sequence of bool, optional
            If set to True, make all 2D animations in the poloidal plane instead of using
            grid coordinates, per variable if sequence is given
        subplots_adjust : dict, optional
            Arguments passed to fig.subplots_adjust()()
        vmin : float or sequence of floats
            Minimum value for color scale, per variable if a sequence is given
        vmax : float or sequence of floats
            Maximum value for color scale, per variable if a sequence is given
        logscale : bool or float, sequence of bool or float, optional
            If True, default to a logarithmic color scale instead of a linear one.
            If a non-bool type is passed it is treated as a float used to set the linear
            threshold of a symmetric logarithmic scale as
            linthresh=min(abs(vmin),abs(vmax))*logscale, defaults to 1e-5 if True is
            passed.
            Per variable if sequence is given.
        titles : sequence of str or None, optional
            Custom titles for each plot. Pass None in the sequence to use the default for
            a certain variable
        aspect : str or None, or sequence of str or None, optional
            Argument to set_aspect() for each plot
        controls : bool, optional
            If set to False, do not show the time-slider or pause button
        tight_layout : bool or dict, optional
            If set to False, don't call tight_layout() on the figure.
            If a dict is passed, the dict entries are passed as arguments to
            tight_layout()
        **kwargs : dict, optional
            Additional keyword arguments are passed on to each animation function
        """

        nvars = len(variables)

        if nrows is None and ncols is None:
            ncols = int(np.ceil(np.sqrt(nvars)))
            nrows = int(np.ceil(nvars/ncols))
        elif nrows is None:
            nrows = int(np.ceil(nvars/ncols))
        elif ncols is None:
            ncols = int(np.ceil(nvars/nrows))
        else:
            if nrows*ncols < nvars:
                raise ValueError('Not enough rows*columns to fit all variables')

        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
        axes = axes.flatten()

        ncells = nrows*ncols

        if nvars < ncells:
            for index in range(ncells-nvars):
                fig.delaxes(axes[ncells - index - 1])

        if subplots_adjust is not None:
            fig.subplots_adjust(**subplots_adjust)

        def _expand_list_arg(arg, arg_name):
            if isinstance(arg, collections.Sequence) and not isinstance(arg, str):
                if len(arg) != len(variables):
                    raise ValueError('if %s is a sequence, it must have the same '
                                     'number of elements as "variables"' % arg_name)
            else:
                arg = [arg] * len(variables)
            return arg
        poloidal_plot = _expand_list_arg(poloidal_plot, 'poloidal_plot')
        vmin = _expand_list_arg(vmin, 'vmin')
        vmax = _expand_list_arg(vmax, 'vmax')
        logscale = _expand_list_arg(logscale, 'logscale')
        titles = _expand_list_arg(titles, 'titles')
        aspect = _expand_list_arg(aspect, 'aspect')

        blocks = []
        for subplot_args in zip(variables, axes, poloidal_plot, vmin, vmax,
                                logscale, titles, aspect):

            (v, ax, this_poloidal_plot, this_vmin, this_vmax, this_logscale,
             this_title, this_aspect) = subplot_args

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            ax.set_aspect(this_aspect)

            if isinstance(v, str):
                v = self.data[v]

            data = v.bout.data
            ndims = len(data.dims)
            ax.set_title(data.name)

            if ndims == 2:
                blocks.append(animate_line(data=data, ax=ax, animate_over=animate_over,
                                           animate=False, **kwargs))
            elif ndims == 3:
                if this_vmin is None:
                    this_vmin = data.min().values
                if this_vmax is None:
                    this_vmax = data.max().values

                norm = _create_norm(this_logscale, kwargs.get('norm', None), this_vmin,
                                    this_vmax)

                if this_poloidal_plot:
                    var_blocks = animate_poloidal(data, ax=ax, cax=cax,
                                                  animate_over=animate_over,
                                                  animate=False, vmin=this_vmin,
                                                  vmax=this_vmax, norm=norm,
                                                  aspect=this_aspect, **kwargs)
                    for block in var_blocks:
                        blocks.append(block)
                else:
                    blocks.append(animate_pcolormesh(data=data, ax=ax, cax=cax,
                                                     animate_over=animate_over,
                                                     animate=False, vmin=this_vmin,
                                                     vmax=this_vmax, norm=norm,
                                                     **kwargs))
            else:
                raise ValueError("Unsupported number of dimensions "
                                 + str(ndims) + ". Dims are " + str(v.dims))

            if this_title is not None:
                # Replace default title with user-specified one
                ax.set_title(this_title)

        timeline = amp.Timeline(np.arange(v.sizes[animate_over]), fps=fps)
        anim = amp.Animation(blocks, timeline)

        if tight_layout:
            if subplots_adjust is not None:
                warnings.warn('tight_layout argument to animate_list() is True, but '
                              'subplots_adjust argument is not None. subplots_adjust '
                              'is being ignored.')
            if not isinstance(tight_layout, dict):
                tight_layout = {}
            fig.tight_layout(**tight_layout)

        if controls:
            anim.controls(timeline_slider_args={'text': animate_over})

        if save_as is not None:
            anim.save(save_as + '.gif', writer=PillowWriter(fps=fps))

        if show:
            plt.show()

        return anim


def _find_major_vars(data):
    """
    Splits data into those variables likely to require a lot of storage space
    (defined as those which depend on time and at least one other dimension).
    These are normally the variables of physical interest.
    """

    # TODO Use an Ordered Set instead to preserve order of variables in files?
    major_vars = set(var for var in data.data_vars
                     if ('t' in data[var].dims) and data[var].dims != ('t', ))
    minor_vars = set(data.data_vars) - set(major_vars)
    return list(major_vars), list(minor_vars)
