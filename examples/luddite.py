from xcollect.boutdataset import BoutDataset


ds = BoutDataset(datapath='.')

# To access just the desired numpy array values directly, slice then use .values
n = ds['n'].isel(t=slice(10,None,5)).values

# The dataset can also be loaded without using dask, though this will be far more memory-intensive
ds = BoutDataset(datapath='.', chunks=None)
