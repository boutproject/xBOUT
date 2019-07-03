import matplotlib.pyplot as plt

from xbout import open_grid


file = '/home/tegn500/Documents/Work/Code/xBOUT/examples/data/disconnected-double-null.grd.nc'
salpha = '/home/tegn500/Documents/Work/Code/Scripts/analyse_STORM/pylib/tomnicholas/salpha/grid.nc'
grid = open_grid(file, geometry='toroidal')

fig, ax = plt.subplots()

grid['psi'].bout.contourf(ax=ax)

#x = grid['R'].values.flat
#y = grid['Z'].values.flat
#z = grid['psi'].values.flat
#tri = ax.tricontourf(x, y, z)

#ax.separatrices()
#ax.limiters()
#ax.branch_cuts()

plt.show()
