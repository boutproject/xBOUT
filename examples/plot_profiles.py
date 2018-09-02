import matplotlib.pyplot as plt

from xcollect.boutdataset import BoutDataset


bd = BoutDataset(datapath='.')

bd['n_profile'] = bd['n'].mean(dim=('t', 'z'))

bd['n_profile'].plot()

plt.show()
