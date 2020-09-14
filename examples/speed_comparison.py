"""
This example is to compare the speed of using dask + xarray vs the old collect function.

It just operates on the bout files in the local directory.
It first opens all the data then saves it again at a lower resolution.
"""

from timeit import Timer

from xcollect.boutdataset import open_boutdataset
from combine import combine_boutdata


def new_collect():
    ds = open_boutdataset(datapath="./BOUT.dmp.*.nc")
    ds.bout.save(save_dtype="float32", separate_vars=True)


def old_collect():
    combine_boutdata(
        datadir=".",
        prefix="/BOUT.dmp.",
        inpfile="BOUT.inp",
        output_file_prefix="boutdata_old",
        save_fields_separately=True,
        ignore_fields=["logn", "logT", "uE"],
        only_save_fields=None,
        save_grid=False,
        save_dtype="float32",
    )


t_new = Timer(new_collect).timeit(number=1)
t_old = Timer(old_collect).timeit(number=1)

print("New collect took " + t_new)
print("Old collect took " + t_old)
