from matplotlib import pyplot as plt

from xbout import open_boutdataset


# We do not distribute binary files with xBOUT, so you need to provide your own gridfile,
# for example one created using Hypnotoad.
gridfilepath = "grid.nc"
grid = open_boutdataset(gridfilepath, geometry="toroidal")

grid["psi_poloidal"].bout.contourf()
grid["psi_poloidal"].bout.contour()
grid["psi_poloidal"].bout.pcolormesh()
grid["psi_poloidal"].bout.pcolormesh(shading="gouraud")

grid["psi_poloidal"].bout.regions()

plt.show()
