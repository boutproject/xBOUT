from xcollect.boutdataset import open_boutdataset


ds = open_boutdataset(datapath=".")

# To access just the desired numpy array values directly, slice then use .values
n = ds["n"].isel(t=slice(10, None, 5)).values

# The dataset can also be loaded without using dask, though this will be far more memory-intensive
ds = open_boutdataset(datapath=".", chunks=None)
