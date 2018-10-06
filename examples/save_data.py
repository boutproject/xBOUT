from xcollect.boutdataset import BoutDataset


ds = BoutDataset(datapath='.')

ds.bout.save(savepath='.', variables=['n', 'phi', 'T'])
