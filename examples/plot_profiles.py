import matplotlib.pyplot as plt

from xcollect.boutdataset import BoutDataset


ds = BoutDataset(datapath=".")

ds["n_profile"] = ds["n"].mean(dim=("t", "z"))

ds["n_profile"].plot()

plt.show()
