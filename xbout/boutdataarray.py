from pprint import pformat

from xarray import register_dataarray_accessor

from .plotting.animate import animate_imshow, animate_line


@register_dataarray_accessor('bout')
class BoutDataArrayAccessor:
    """
    Contains BOUT-specific methods to use on BOUT++ dataarrays opened by
    selecting a variable from a BOUT++ dataset.

    These BOUT-specific methods and attributes are accessed via the bout
    accessor, e.g. `da.bout.options` returns a `BoutOptionsFile` instance.
    """

    def __init__(self, da):

        # # Load data variables
        # # Should we just load whole dataset here?
        # self.datapath = datapath
        # self.prefix = prefix

        self.data = da
        self.metadata = da.attrs['metadata']
        self.options = da.attrs['options']

    def __str__(self):
        """
        String representation of the BoutDataArray.

        Accessed by print(da.bout)
        """

        text = "<xbout.BoutDataArray>\n" + \
               "Contains:\n{}\n".format(str(self.data)) + \
               "Metadata:\n{}".format(pformat(self.metadata,
                                              indent=4, compact=True))
        if self.options:
            text += "Options:\n{}".format(pformat(self.options.as_dict(),
                                                  indent=4, compact=True))
        return text

    def animate2D(self, animate_over='t', x=None, y=None, animate=True,
                  fps=10, save_as=None, sep_pos=None, ax=None, **kwargs):
        """
        Plots a color plot which is animated with time over the specified
        coordinate.

        Currently only supports 2D+1 data, which it plots with xarray's
        wrapping of matplotlib's imshow.

        Parameters
        ----------
        animate_over : str, optional
            Dimension over which to animate
        x : str, optional
            Dimension to use on the x axis, default is None - then use the first spatial
            dimension of the data
        y : str, optional
            Dimension to use on the y axis, default is None - then use the second spatial
            dimension of the data
        sep_pos : int, optional
            Radial position at which to plot the separatrix
        fps : int, optional
            Frames per second of resulting gif
        save_as : str, optional
            Filename to give to the resulting gif
        sep_pos : int, optional
            Position along the 'x' dimension to plot the separatrix
        ax : matplotlib.pyplot.axes object, optional
            Axis on which to plot the gif
        kwargs : dict, optional
            Additional keyword arguments are passed on to the plotting function
            (e.g. imshow for 2D plots).
        """

        data = self.data
        variable = data.name
        n_dims = len(data.dims)
        if n_dims == 3:
            print("{} data passed has {} dimensions - will use "
                  "animatplot.blocks.Imshow()".format(variable, str(n_dims)))
            imshow_block = animate_imshow(data=data, animate_over=animate_over,
                                          x=x, y=y, sep_pos=sep_pos,
                                          animate=animate, fps=fps,
                                          save_as=save_as, ax=ax, **kwargs)
            return imshow_block
        else:
            raise ValueError(
                "Data passed has an unsupported number of dimensions "
                "({})".format(str(n_dims)))

    def animate1D(self, animate_over='t', x='x', y='y', animate=True,
                 fps=10, save_as=None, sep_pos=None, ax=None, **kwargs):
        data = self.data
        variable = data.name
        n_dims = len(data.dims)

        if n_dims == 2:
            print("{} data passed has {} dimensions - will use "
                  "animatplot.blocks.Line()".format(variable, str(n_dims)))
            line_block = animate_line(data=data, animate_over=animate_over,
                                      x=x, y=y, sep_pos=sep_pos,
                                      animate=animate, fps=fps,
                                      save_as=save_as, ax=ax, **kwargs)
            return line_block

    # TODO BOUT-specific plotting functionality would be implemented as methods here, e.g. ds.bout.plot_poloidal
    # TODO Could trial a 2D surface plotting method here
