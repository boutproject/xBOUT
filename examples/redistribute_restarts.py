from xcollect.boutdataset import BoutDataset


bd = BoutDataset(datapath=".")

# Multiply density and temperature fluctuations by 2
for var in ["n", "T"]:
    bd.data["var"] = bd.data["var"] * 2

# Downsample the data by interpolating onto a coarser x coordinate vector
new_x = bd.data["x"].slice(None, None, 3)
bd.data = bd.data.interp(x=new_x)

# Save new restart files with a different parallelisation
bd.save_restart(savepath=".", nxpe=4, nype=8)
