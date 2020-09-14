from xarray import concat
from xcollect.boutdataset import BoutDataset

# Parameter which was scanned over
p = [0.5, 1.0, 1.5]
# The directories to find the results in
datapaths = ["./" + str(pressure) for pressure in p]

# Load the results of each run as a separate dataset
boutdatasets = [BoutDataset(datapath=path) for path in datapaths]

# We  can now merge these into a single dataset along a new dimension
ds = concat(datasets, "p", p)

# Plot results over the scanned parameter
