from xcollect.boutdataset import open_boutdataset
from combine import combine_boutdata

ds = open_boutdataset(datapath="./BOUT.dmp.*.nc")

ds.bout.save(savepath=".")
