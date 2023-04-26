import numpy as np
import eudist

from warnings import warn


class OutOfDomainError(ValueError):
    pass


def _rz_to_ab(rz, mesh):
    """
    This functions finds the position (R-z) in the mesh and
    returns the index and the relative coordinates [0,1] in that cell.
    It uses a newton iteration.
    """
    ij = mesh.find_cell(rz)
    if ij < 0:
        raise OutOfDomainError()
    _, nz = mesh.shape
    i, j = ij // nz, ij % nz
    ABCD = mesh.grid[i : i + 2, j : j + 2]
    A = ABCD[0, 0]
    a = ABCD[0, 1] - A
    b = ABCD[1, 0] - A
    c = ABCD[1, 1] - A - a - b
    rz0 = rz - A

    def fun(albe):
        "The forwards function"
        al, be = albe
        return rz0 - a * al - b * be - c * al * be

    def J(albe):
        "The jacobian"
        al, be = albe
        return np.array([-a - c * be, -b - c * al])

    tol = 1e-13
    albe = np.ones(2) / 2
    while True:
        res = np.sum(fun(albe) ** 2)
        albe = albe - np.linalg.inv(J(albe).T) @ fun(albe)
        if res < tol:
            return albe, ij


def _ab_to_rz(ab, ij, mesh):
    """
    Calculate the position in cartesian R-z coordinates
    given the relative position in the grid cell (ab) and
    the grid indices ij.
    """
    _, nz = mesh.shape
    i, j = ij // nz, ij % nz
    A = mesh.grid[i, j]
    a = mesh.grid[i, j + 1] - A
    b = mesh.grid[i + 1, j] - A
    c = mesh.grid[i + 1, j + 1] - A - a - b
    al, be = ab
    return A + al * a + be * b + al * be * c


def _setup_mesh(x, y):
    """
    Setup the mesh and store some additional info.

    The fci-mesh is assumed to periodic in z - but eudist does not
    handle this so we need to copy the first corners around.
    """

    def make_periodic(d):
        "The grid should be periodic in z"
        return np.concatenate((d, d[:, :1]), axis=1)

    assert x.dims == y.dims
    assert x.dims == ("x", "z")
    x = make_periodic(x.data)
    y = make_periodic(y.data)
    return _MyMesh(x, y)


class _MyMesh(eudist.PolyMesh):
    """
    Like the PolyMesh but with extra data
    """

    def __init__(self, x, y):
        super().__init__()
        self.r = x
        self.z = y
        self.grid = np.array([x, y]).transpose(1, 2, 0)
        self.shape = tuple([x - 1 for x in x.shape])


class Tracer:
    """
    Use an EMC3-like tracing. This relies on the grid containing a
    tracing to the next slice. The current position in RZ coordinates
    is converted to the relative position in the grid, and then the
    reverse is done for the end of the "flux tube" defined by corners
    of the cells.
    """

    def __init__(self, ds, direction="forward"):
        """
        ds: xr.Dataset
            a dataset with the needed FCI data from zoidberg.

        direction: str
            "forward" or "backward"
        """
        meshes = []
        for yi in range(len(ds.y)):
            dsi = ds.isel(y=yi)
            meshes.append(
                [
                    _setup_mesh(dsi[f"{pre}R"], dsi[f"{pre}Z"])
                    for pre in ["", f"{direction}_"]
                ]
                + [yi]
            )
        self.meshes = meshes

    def poincare(self, rz, yind=0, num=100, early_exit="warn"):
        """
        rz : array-like with 2 values
            The RZ position where to start tracing
        yind : int
            The y-index of the slice where to start tracing.
        num : int
            Number of rounds to trace for
        early_exit : str
            How to handle if we leave the domain before doing `num` rounds.
            The possible values are:
            "ignore" : do nothing
            "warn": raise a warning
            "plot" : try to plot the grid and where we leave
            "raise" : Raise the exception for handling by the caller

        Returns
        -------
        np.array
        An array of shape (num, 2) or less then num if the tracing leaves the domain.
        It contains the r-z coordinates at the y-index where tracing started.
        """
        rz = np.array(rz)
        assert rz.shape == (2,)
        thismeshes = self.meshes[yind:] + self.meshes[:yind]
        out = np.empty((num, 2))
        out[0] = rz
        last = None
        for i in range(1, num):
            for d, meshes in enumerate(thismeshes):
                try:
                    abij = _rz_to_ab(rz, meshes[0])
                except AssertionError as e:
                    if early_exit == "warn":
                        warn(f"early exit in iteration {i} because `{e}`")
                    elif early_exit == "plot":
                        m = meshes[0]
                        import matplotlib.pyplot as plt

                        plt.plot(m.r, m.z)
                        if last:
                            plt.plot(last[1].r.T, last[1].z.T)

                        plt.plot(*rz, "o")
                        plt.show()
                    elif early_exit == "raise":
                        raise
                    else:
                        assert early_exit == "ignore", (
                            "early_exit needs to be one of "
                            + '["warn", "plot", "raise", ignore"] '
                            + f"but got `{early_exit}`"
                        )
                    return out[:i]
                rz = _ab_to_rz(*abij, meshes[1])
                last = meshes
            out[i] = rz
        return out
