from xarray import register_dataset_accessor

from xcollect.boutdataset import BoutAccessor

from wake import add_plasma_params, add_normalisations
from units import convert_units


@register_dataset_accessor("storm")
class StormAccessor(BoutAccessor):
    """
    Class specifically for holding data from a simulation using the STORM module for BOUT++.

    Implements methods for normalising the data with STORM-specific normalisation constants.
    """

    def __init__(self, ds_object):
        super().__init__(ds_object)

        # Calculate plasma parameters from options file
        # self.params = self._calc_params()

        # Calculate normalisations of data variables from parameters
        #        self._norms = self._calc_norms()

        # Set default normalisation state of data
        self.normalisation = "computational"

    def print_options(self):
        print(self.extra_data)

    # TODO as BOUT v4.2 can now write attributes to the output NetCDF files
    # would it be better to calculate all normalisations within the BOUT++ module itself?

    def _calc_params(self):
        """
        Calculates physical parameters (e.g. collisionality) from the options in the input file.

        Returns
        -------
        params : dict
        """

        params = add_plasma_params(self.options)
        return params

    def _calc_norms(self):
        """
        Calculates normalisations of physical quantities in the dataset from the options in the input file.

        Returns
        -------
        normed_ds : xarray.Dataset
        """

        normed_ds = add_normalisations(self.ds, self.options)
        return normed_ds

    def renormalise(self, desired_normalisation):
        self.data = convert_units(self.data, desired_normalisation)
        return self

    # This is where module-specific methods would go
    # For example maybe elm-pb would have a .elm_growth_rate() method?
