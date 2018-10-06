from xcollect.boutdataset import BoutDataset

from wake import add_plasma_params, add_normalisations
from units import convert_units


class StormDataset(BoutDataset):
    def __init__(self, datapath='.', chunks={}, run_name=None, log_file=False):
        super().__init__(datapath='.', chunks={}, input_file=True, run_name=None, log_file=False)

        # Calculate plasma parameters from options file
        self.params = self._calc_params()

        # Calculate normalisations of data variables from parameters
        self._norms = self._calc_norms()

        # Set default normalisation state of data
        self._normalisation = 'computational'

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
