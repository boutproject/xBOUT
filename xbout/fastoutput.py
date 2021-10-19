import re
import xarray as xr
from xbout.load import _expand_filepaths


def open_fastoutput(datapath="BOUT.fast.*.nc"):
    """
    Opens fast output data and combines into a single dataset.

    """

    # Get list of all files
    filepaths, filetype = _expand_filepaths(datapath)

    # Iterate over all files, extracting DataArrays ready for combining
    fo_data = []
    for i, filepath in enumerate(filepaths):

        fo = xr.open_dataset(filepath)

        if i == 0:
            # Get time coordinate from first file
            time = fo["time"]

        # Time is global, and we already extracted it
        fo = fo.drop_vars("time", errors="ignore")

        # There might be no virtual probe in this region
        if len(fo.data_vars) > 0:

            for name, da in fo.items():

                # Save the physical position (in index units)
                da = da.expand_dims(x=1, y=1, z=1)
                da = da.assign_coords(
                    x=xr.DataArray([da.attrs["ix"]], dims=["x"]),
                    y=xr.DataArray([da.attrs["iy"]], dims=["y"]),
                    z=xr.DataArray([da.attrs["iz"]], dims=["z"]),
                )

                # Re-attach the time coordinate
                da = da.assign_coords(time=time)

                # We saved the position, so don't care what number the variable was
                # Only need it's name (i.e. n, T, etc.)
                regex = re.compile(r"(\D+)([0-9]+)")
                match = regex.match(name)
                if match is None:
                    raise ValueError(f"Regex could not parse the variable named {name}")
                var, num = match.groups()
                da.name = var

                # Must promote DataArrays to Datasets until we require xarray-0.19.0
                # where xarray GH #3248 is fixed
                ds = xr.Dataset({var: da})
                fo_data.append(ds)

        fo.close()

    # This will merge different variables, and arrange by physical position
    full_fo = xr.combine_by_coords(fo_data, combine_attrs="drop_conflicts")

    return full_fo
