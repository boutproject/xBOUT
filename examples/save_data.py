from xcollect.boutdataset import BoutDataset


bd = BoutDataset(datapath='.')

bd.save(savepath='.', variables=['n', 'phi', 'T'])
