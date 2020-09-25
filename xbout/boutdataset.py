import collections
from copy import copy
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

from .geometries import apply_geometry
from .plotting.animate import animate_poloidal, animate_pcolormesh, animate_line
from .plotting.utils import _create_norm
from .region import _from_region
from .utils import _get_bounding_surfaces, _split_into_restarts


@xr.register_dataset_accessor("bout")
class BoutDatasetAccessor:
    """
    Contains BOUT-specific methods to use on BOUT++ datasets opened using
    `open_boutdataset()`.

    These BOUT-specific methods and attributes are accessed via the bout
    accessor, e.g. `ds.bout.options` returns a `BoutOptionsFile` instance.
    """

    def __init__(self, ds):
        self.data = ds
        self.metadata = ds.attrs.get("metadata")  # None if just grid file
        self.options = ds.attrs.get("options")  # None if no inp file

    def __str__(self):
        """
        String representation of the BoutDataset.

        Accessed by print(ds.bout)
        """

        styled = partial(prettyformat, indent=4, compact=True)
        text = (
            "<xbout.BoutDataset>\n"
            + "Contains:\n{}\n".format(str(self.data))
            + "Metadata:\n{}\n".format(styled(self.metadata))
        )
        if self.options:
            text += "Options:\n{}".format(styled(self.options))
        return text

    # def __repr__(self):
    #    return 'boutdata.BoutDataset(', {}, ',', {}, ')'.format(self.datapath,
    #  self.prefix)

    def get_field_aligned(self, name, caching=True):
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
        aligned_name = name + "_aligned"
        try:
            result = self.data[aligned_name]
            if result.direction_y != "Aligned":
                raise ValueError(
                    aligned_name + " exists, but is not field-aligned, it "
                    "has direction_y=" + result.direction_y
                )
            return result
        except KeyError:
            if caching:
                self.data[aligned_name] = self.data[name].bout.to_field_aligned()
            return self.data[aligned_name]

    def from_region(self, name, with_guards=None):
        """
        Get a logically-rectangular section of data from a certain region.
        Includes guard cells from neighbouring regions.

        Parameters
        ----------
        name : str
            Region to get data for
        with_guards : int or dict of int, optional
            Number of guard cells to include, by default use MXG and MYG from BOUT++.
            Pass a dict to set different numbers for different coordinates.
        """
        return _from_region(self.data, name, with_guards)

    @property
    def regions(self):
        if "regions" not in self.data.attrs:
            raise ValueError(
                "Called a method requiring regions, but these have not been created. "
                "Please set the 'geometry' option when calling open_boutdataset() to "
                "create regions."
            )
        return self.data.attrs["regions"]

    @property
    def fine_interpolation_factor(self):
        """
        The default factor to increase resolution when doing parallel interpolation
        """
        return self.data.metadata["fine_interpolation_factor"]

    @fine_interpolation_factor.setter
    def fine_interpolation_factor(self, n):
        """
        Set the default factor to increase resolution when doing parallel interpolation.

        Parameters
        -----------
        n : int
            Factor to increase parallel resolution by
        """
        ds = self.data
        ds.metadata["fine_interpolation_factor"] = n
        for da in ds.data_vars.values():
            da.metadata["fine_interpolation_factor"] = n

    def interpolate_parallel(self, variables, **kwargs):
        """
        Interpolate in the parallel direction to get a higher resolution version of a
        subset of variables.

        Note that the high-resolution variables are all loaded into memory, so most
        likely it is necessary to select only a small number. The toroidal_points
        argument can also be used to reduce the memory demand.

        Parameters
        ----------
        variables : str or sequence of str or ...
            The names of the variables to interpolate. If 'variables=...' is passed
            explicitly, then interpolate all variables in the Dataset.
        n : int, optional
            The factor to increase the resolution by. Defaults to the value set by
            BoutDataset.setupParallelInterp(), or 10 if that has not been called.
        toroidal_points : int or sequence of int, optional
            If int, number of toroidal points to output, applies a stride to toroidal
            direction to save memory usage. If sequence of int, the indexes of toroidal
            points for the output.
        method : str, optional
            The interpolation method to use. Options from xarray.DataArray.interp(),
            currently: linear, nearest, zero, slinear, quadratic, cubic. Default is
            'cubic'.

        Returns
        -------
        A new Dataset containing a high-resolution versions of the variables. The new
        Dataset is a valid BoutDataset, although containing only the specified variables.
        """

        if variables is ...:
            variables = [v for v in self.data]

        if isinstance(variables, str):
            variables = [variables]
        if isinstance(variables, tuple):
            variables = list(variables)

        if "dy" in variables:
            # dy is treated specially, as it is converted to a coordinate, and then
            # converted back again below, so must not call
            # interpolate_parallel('dy').
            variables.remove("dy")

        # Add extra variables needed to make this a valid Dataset
        if "dx" not in variables:
            variables.append("dx")

        # Need to start with a Dataset with attrs as merge() drops the attrs of the
        # passed-in argument.
        # Make sure the first variable has all dimensions so we don't lose any
        # coordinates
        def find_with_dims(first_var, dims):
            if first_var is None:
                dims = set(dims)
                for v in variables:
                    if set(self.data[v].dims) == dims:
                        first_var = v
                        break
            return first_var

        tcoord = self.data.metadata.get("bout_tdim", "t")
        zcoord = self.data.metadata.get("bout_zdim", "z")
        first_var = find_with_dims(None, self.data.dims)
        first_var = find_with_dims(first_var, set(self.data.dims) - set(tcoord))
        first_var = find_with_dims(first_var, set(self.data.dims) - set(zcoord))
        first_var = find_with_dims(
            first_var, set(self.data.dims) - set([tcoord, zcoord])
        )
        if first_var is None:
            raise ValueError(
                f"Could not find variable to interpolate with both "
                f"{ds.metadata.get('bout_xdim', 'x')} and "
                f"{ds.metadata.get('bout_ydim', 'y')} dimensions"
            )
        variables.remove(first_var)
        ds = self.data[first_var].bout.interpolate_parallel(
            return_dataset=True, **kwargs
        )
        xcoord = ds.metadata.get("bout_xdim", "x")
        ycoord = ds.metadata.get("bout_ydim", "y")
        for var in variables:
            da = self.data[var]
            if xcoord in da.dims and ycoord in da.dims:
                ds = ds.merge(
                    da.bout.interpolate_parallel(return_dataset=True, **kwargs)
                )
            elif ycoord not in da.dims:
                ds[var] = da
            # Can't interpolate a variable that depends on y but not x, so just skip

        # dy needs to be compatible with the new poloidal coordinate
        # dy was created as a coordinate in BoutDataArray.interpolate_parallel, here just
        # need to demote back to a regular variable.
        ds = ds.reset_coords("dy")

        # Apply geometry
        if hasattr(ds, "geometry"):
            ds = apply_geometry(ds, ds.geometry)
        # if no geometry was originally applied, then ds has no geometry attribute and we
        # can continue without applying geometry here

        return ds

    def interpolate_from_unstructured(
        self,
        variables,
        *,
        fill_value=np.nan,
        structured_output=True,
        unstructured_dim_name="unstructured_dim",
        **kwargs,
    ):
        """Interpolate Dataset onto new grids of some existing coordinates

        Parameters
        ----------
        variables : str or sequence of str or ...
            The names of the variables to interpolate. If 'variables=...' is passed
            explicitly, then interpolate all variables in the Dataset.
        **kwargs : (str, array)
            Each keyword is the name of a coordinate in the DataArray, the argument is a
            1d array giving the values of that coordinate on the output grid
        fill_value : float
            fill_value passed through to scipy.interpolation.griddata
        structured_output : bool, default True
            If True, treat output coordinates values as a structured grid.
            If False, output coordinate values must all have the same length and are not
            broadcast together.
        unstructured_dim_name : str, default "unstructured_dim"
            Name used for the dimension in the output that replaces the dimensions of
            the interpolated coordinates. Only used if structured_output=False.

        Returns
        -------
        Dataset
            Dataset interpolated onto a new, structured grid
        """

        if variables is ...:
            variables = [v for v in self.data]
            explicit_variables_arg = False
        else:
            explicit_variables_arg = True

        if isinstance(variables, str):
            variables = [variables]
        if isinstance(variables, tuple):
            variables = list(variables)

        coords_to_interpolate = []
        for coord in self.data.coords:
            if coord not in variables and coord not in kwargs:
                coords_to_interpolate.append(coord)

        ds = xr.Dataset()

        for v in variables + coords_to_interpolate:
            if np.all([c in self.data[v].coords for c in kwargs]):
                ds = ds.merge(
                    self.data[v]
                    .bout.interpolate_from_unstructured(
                        fill_value=fill_value,
                        structured_output=structured_output,
                        unstructured_dim_name=unstructured_dim_name,
                        **kwargs,
                    )
                    .to_dataset()
                )
            elif explicit_variables_arg and v in variables:
                # User explicitly requested v to be interpolated
                raise ValueError(
                    f"Could not interpolate {v} because it does not depend on all "
                    f"coordinates {[c for c in kwargs]}"
                )
            elif v in coords_to_interpolate:
                coords_to_interpolate.remove(v)

        ds = ds.set_coords(coords_to_interpolate)

        return ds

    def remove_yboundaries(self, **kwargs):
        """
        Remove y-boundary points, if present, from the Dataset
        """

        variables = []
        xcoord = self.data.metadata["bout_xdim"]
        ycoord = self.data.metadata["bout_ydim"]
        new_metadata = None
        for v in self.data:
            if xcoord in self.data[v].dims and ycoord in self.data[v].dims:
                variables.append(
                    self.data[v].bout.remove_yboundaries(return_dataset=True, **kwargs)
                )
                new_metadata = variables[-1].metadata
            elif ycoord in self.data[v].dims:
                raise ValueError(
                    f"{v} only has a {ycoord}-dimension so cannot split "
                    f"into regions."
                )
            else:
                variable = self.data[v]
                if "keep_yboundaries" in variable.metadata:
                    variable.attrs["metadata"] = copy(variable.metadata)
                    variable.metadata["keep_yboundaries"] = 0
                variables.append(variable.bout.to_dataset())
        if new_metadata is None:
            # were no 2d or 3d variables so do not have updated jyseps*, ny_inner but
            # does not matter because missing metadata is only useful for 2d or 3d
            # variables
            new_metadata = variables[0].metadata

        result = xr.merge(variables)

        result.attrs = copy(self.data.attrs)

        # Copy metadata to get possibly modified jyseps*, ny_inner, ny
        result.attrs["metadata"] = new_metadata

        if "regions" in result.attrs:
            # regions are not correct for modified BoutDataset
            del result.attrs["regions"]

        # call to re-create regions
        result = apply_geometry(result, self.data.geometry)

        return result

    def get_bounding_surfaces(self, coords=("R", "Z")):
        """
        Get bounding surfaces.
        Surfaces are returned as arrays of points describing a polygon, assuming the
        third spatial dimension is a symmetry direction.

        Parameters
        ----------
        coords : (str, str), default ("R", "Z")
            Pair of names of coordinates whose values are used to give the positions of
            the points in the result

        Returns
        -------
        result : list of DataArrays
            Each DataArray in the list contains points on a boundary, with size
            (<number of points in the bounding polygon>, 2). Points wind clockwise around
            the outside domain, and anti-clockwise around the inside (if there is an
            inner boundary).
        """
        return _get_bounding_surfaces(self.data, coords)

    def save(
        self,
        savepath="./boutdata.nc",
        filetype="NETCDF4",
        variables=None,
        save_dtype=None,
        separate_vars=False,
        pre_load=False,
    ):
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

        if savepath == "./boutdata.nc":
            print(
                "Will save data into the current working directory, named as"
                " boutdata_[var].nc"
            )
        if savepath is None:
            raise ValueError("Must provide a path to which to save the data.")

        # make shallow copy of Dataset, so we do not modify the attributes of the data
        # when we change things to save
        to_save = to_save.copy()

        options = to_save.attrs.pop("options")
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
                del to_save[var].attrs["options"]
            except KeyError:
                pass

        # Store the metadata as individual attributes instead because
        # netCDF can't handle storing arbitrary objects in attrs
        def dict_to_attrs(obj, section):
            for key, value in obj.attrs.pop(section).items():
                obj.attrs[section + ":" + key] = value

        dict_to_attrs(to_save, "metadata")
        # Must do this for all variables and coordinates in dataset too
        for varname, da in chain(to_save.data_vars.items(), to_save.coords.items()):
            try:
                dict_to_attrs(da, "metadata")
            except KeyError:
                pass

        if "regions" in to_save.attrs:
            # Do not need to save regions as these can be reconstructed from the metadata
            try:
                del to_save.attrs["regions"]
            except KeyError:
                pass
            for var in chain(to_save.data_vars, to_save.coords):
                try:
                    del to_save[var].attrs["regions"]
                except KeyError:
                    pass

        if save_dtype is not None:
            encoding = {v: {"dtype": save_dtype} for v in to_save}
        else:
            encoding = None

        if separate_vars:
            # Save each major variable to a different netCDF file

            # Determine which variables are "major"
            # Defined as time-dependent, but not solely time-dependent
            major_vars, minor_vars = _find_major_vars(to_save)

            print("Will save the variables {} separately".format(str(major_vars)))

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
                var_savepath = (
                    str(path.parent / path.stem) + "_" + str(major_var) + path.suffix
                )
                if encoding is not None:
                    var_encoding = {major_var: encoding[major_var]}
                else:
                    var_encoding = None
                print("Saving " + major_var + " data...")
                with ProgressBar():
                    single_var_ds.to_netcdf(
                        path=str(var_savepath),
                        format=filetype,
                        compute=True,
                        encoding=var_encoding,
                    )

                # Force memory deallocation to limit RAM usage
                single_var_ds.close()
                del single_var_ds
                gc.collect()
        else:
            # Save data to a single file
            print("Saving data...")
            with ProgressBar():
                to_save.to_netcdf(
                    path=savepath, format=filetype, compute=True, encoding=encoding
                )

        return

    def to_restart(
        self,
        variables=None,
        *,
        savepath=".",
        nxpe=None,
        nype=None,
        tind=-1,
        prefix="BOUT.restart",
        overwrite=False,
    ):
        """
        Write out a timestep as a set of netCDF BOUT.restart files.

        If processor decomposition is not specified then data will be saved
        using the decomposition it had when loaded.

        Parameters
        ----------
        variables : str or sequence of str, optional
            The evolving variables needed in the restart files. If not given explicitly,
            all time-evolving variables in the Dataset will be used, which may result in
            larger restart files than necessary.
        savepath : str, default '.'
            Directory to save the created restart files under
        nxpe : int, optional
            Number of processors in the x-direction. If not given, keep the number used
            for the original simulation
        nype : int, optional
            Number of processors in the y-direction. If not given, keep the number used
            for the original simulation
        tind : int, default -1
            Time-index of the slice to write to the restart files
        prefix : str, default "BOUT.restart"
            Prefix to use for names of restart files
        overwrite : bool, default False
            By default, raises if restart file already exists. Set to True to overwrite
            existing files
        """

        if isinstance(variables, str):
            variables = [variables]

        # Set processor decomposition if not given
        if nxpe is None:
            nxpe = self.metadata["NXPE"]
        if nype is None:
            nype = self.metadata["NYPE"]

        # Is this even possible without saving the guard cells?
        # Can they be recreated?
        restart_datasets, paths = _split_into_restarts(
            self.data,
            variables,
            savepath,
            nxpe,
            nype,
            tind,
            prefix,
            overwrite,
        )

        with ProgressBar():
            xr.save_mfdataset(restart_datasets, paths, compute=True)

    def animate_list(
        self,
        variables,
        animate_over="t",
        save_as=None,
        show=False,
        fps=10,
        nrows=None,
        ncols=None,
        poloidal_plot=False,
        subplots_adjust=None,
        vmin=None,
        vmax=None,
        logscale=None,
        titles=None,
        aspect="equal",
        extend=None,
        controls=True,
        tight_layout=True,
        **kwargs,
    ):
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
        extend : str or None, optional
            Passed to fig.colorbar()
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
            nrows = int(np.ceil(nvars / ncols))
        elif nrows is None:
            nrows = int(np.ceil(nvars / ncols))
        elif ncols is None:
            ncols = int(np.ceil(nvars / nrows))
        else:
            if nrows * ncols < nvars:
                raise ValueError("Not enough rows*columns to fit all variables")

        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
        axes = axes.flatten()

        ncells = nrows * ncols

        if nvars < ncells:
            for index in range(ncells - nvars):
                fig.delaxes(axes[ncells - index - 1])

        if subplots_adjust is not None:
            fig.subplots_adjust(**subplots_adjust)

        def _expand_list_arg(arg, arg_name):
            if isinstance(arg, collections.abc.Sequence) and not isinstance(arg, str):
                if len(arg) != len(variables):
                    raise ValueError(
                        "if %s is a sequence, it must have the same "
                        'number of elements as "variables"' % arg_name
                    )
            else:
                arg = [arg] * len(variables)
            return arg

        poloidal_plot = _expand_list_arg(poloidal_plot, "poloidal_plot")
        vmin = _expand_list_arg(vmin, "vmin")
        vmax = _expand_list_arg(vmax, "vmax")
        logscale = _expand_list_arg(logscale, "logscale")
        titles = _expand_list_arg(titles, "titles")
        aspect = _expand_list_arg(aspect, "aspect")
        extend = _expand_list_arg(extend, "extend")

        blocks = []

        def is_list(variable):
            return (
                isinstance(variable, list)
                or isinstance(variable, tuple)
                or isinstance(variable, set)
            )

        for subplot_args in zip(
            variables, axes, poloidal_plot, vmin, vmax, logscale, titles, aspect, extend
        ):

            (
                v,
                ax,
                this_poloidal_plot,
                this_vmin,
                this_vmax,
                this_logscale,
                this_title,
                this_aspect,
                this_extend,
            ) = subplot_args

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            ax.set_aspect(this_aspect)

            if is_list(v):
                for i in range(len(v)):
                    if isinstance(v[i], str):
                        v[i] = self.data[v[i]]
                # list of variables for one subplot only supported for line plots with 1
                # dimension plus time
                ndims = 2
                dims = v[0].dims
                if len(dims) != 2:
                    raise ValueError(
                        "Variables in sublist must be 2d - can only overlay line plots"
                    )
                for w in v:
                    if not w.dims == dims:
                        raise ValueError(
                            f"All variables in sub-list must have same dimensions."
                            f"{v[0].name} had {v[0].dims} but {w.name} had {w.dims}."
                        )
            else:
                if isinstance(v, str):
                    v = self.data[v]

                data = v.bout.data
                ndims = len(data.dims)
                ax.set_title(data.name)

            if ndims == 2:
                if not is_list(v):
                    blocks.append(
                        animate_line(
                            data=data,
                            ax=ax,
                            animate_over=animate_over,
                            animate=False,
                            **kwargs,
                        )
                    )
                else:
                    for w in v:
                        blocks.append(
                            animate_line(
                                data=w,
                                ax=ax,
                                animate_over=animate_over,
                                animate=False,
                                label=w.name,
                                **kwargs,
                            )
                        )
                    legend = ax.legend()
                    legend.set_draggable(True)
                    # set 'v' to use for the timeline below
                    v = v[0]
            elif ndims == 3:
                if this_vmin is None:
                    this_vmin = data.min().values
                if this_vmax is None:
                    this_vmax = data.max().values

                norm = _create_norm(
                    this_logscale, kwargs.get("norm", None), this_vmin, this_vmax
                )

                if this_poloidal_plot:
                    var_blocks = animate_poloidal(
                        data,
                        ax=ax,
                        cax=cax,
                        animate_over=animate_over,
                        animate=False,
                        vmin=this_vmin,
                        vmax=this_vmax,
                        norm=norm,
                        aspect=this_aspect,
                        extend=this_extend,
                        **kwargs,
                    )
                    for block in var_blocks:
                        blocks.append(block)
                else:
                    blocks.append(
                        animate_pcolormesh(
                            data=data,
                            ax=ax,
                            cax=cax,
                            animate_over=animate_over,
                            animate=False,
                            vmin=this_vmin,
                            vmax=this_vmax,
                            norm=norm,
                            extend=this_extend,
                            **kwargs,
                        )
                    )
            else:
                raise ValueError(
                    "Unsupported number of dimensions "
                    + str(ndims)
                    + ". Dims are "
                    + str(v.dims)
                )

            if this_title is not None:
                # Replace default title with user-specified one
                ax.set_title(this_title)

        timeline = amp.Timeline(np.arange(v.sizes[animate_over]), fps=fps)
        anim = amp.Animation(blocks, timeline)

        if tight_layout:
            if subplots_adjust is not None:
                warnings.warn(
                    "tight_layout argument to animate_list() is True, but "
                    "subplots_adjust argument is not None. subplots_adjust "
                    "is being ignored."
                )
            if not isinstance(tight_layout, dict):
                tight_layout = {}
            fig.tight_layout(**tight_layout)

        if controls:
            anim.controls(timeline_slider_args={"text": animate_over})

        if save_as is not None:
            anim.save(save_as + ".gif", writer=PillowWriter(fps=fps))

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
    tcoord = data.attrs.get("metadata:bout_tdim", "t")
    major_vars = set(
        var
        for var in data.data_vars
        if (tcoord in data[var].dims) and data[var].dims != (tcoord,)
    )
    minor_vars = set(data.data_vars) - set(major_vars)
    return list(major_vars), list(minor_vars)
