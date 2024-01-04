import numpy as np
import eudist

from warnings import warn


class OutOfDomainError(ValueError):
    pass


def rz_to_ab(rz, mesh):
    ij = mesh.find_cell(rz)
    if ij < 0:
        raise OutOfDomainError("We left the domain")
    _, nz = mesh.shape
    i, j = ij // nz, ij % nz
    ABCD = mesh.grid[i : i + 2, j : j + 2]
    A = ABCD[0, 0]
    a = ABCD[0, 1] - A
    b = ABCD[1, 0] - A
    c = ABCD[1, 1] - A - a - b
    rz0 = rz - A

    def fun(albe):
        al, be = albe
        return rz0 - a * al - b * be - c * al * be

    def J(albe):
        al, be = albe
        return np.array([-a - c * be, -b - c * al])

    tol = 1e-13
    albe = np.ones(2) / 2
    while True:
        res = np.sum(fun(albe) ** 2)
        albe = albe - np.linalg.inv(J(albe).T) @ fun(albe)
        if res < tol:
            return albe, ij


def ab_to_rz(ab, ij, mesh):
    _, nz = mesh.shape
    i, j = ij // nz, ij % nz
    A = mesh.grid[i, j]
    a = mesh.grid[i, j + 1] - A
    b = mesh.grid[i + 1, j] - A
    c = mesh.grid[i + 1, j + 1] - A - a - b
    al, be = ab
    return A + al * a + be * b + al * be * c


def setup_mesh(x, y):
    def per(d):
        return np.concatenate((d, d[:, :1]), axis=1)

    assert x.dims == y.dims
    assert x.dims == ("x", "z")
    x = per(x.data)
    y = per(y.data)
    return mymesh(x, y)


class mymesh(eudist.PolyMesh):
    def __init__(self, x, y):
        super().__init__()
        self.r = x
        self.z = y
        self.grid = np.array([x, y]).transpose(1, 2, 0)
        self.shape = tuple([x - 1 for x in x.shape])
        self.ij = -1

    def find_cell(self, rz, guess=None):
        if guess is None:
            if self.tree:
                _, guess = self.tree.query(rz)
                guess //= self.tree_prec
            else:
                guess = self.ij

        self.ij = super().find_cell(rz, guess=guess)
        return self.ij


class Tracer:
    def __init__(self, ds, direction="forward"):
        meshes = []
        for yi in range(len(ds.y)):
            dsi = ds.isel(y=yi)
            meshes.append(
                [
                    setup_mesh(dsi[f"{pre}R"], dsi[f"{pre}Z"])
                    for pre in ["", f"{direction}_"]
                ]
                + [yi]
            )
        self.meshes = meshes

    def poincare(self, rz, yind=0, num=100, early_exit="warn"):
        rz = np.array(rz)
        assert rz.shape == (2,)
        thismeshes = self.meshes[yind:] + self.meshes[:yind]
        out = np.empty((num, 2))
        out[0] = rz
        last = None
        for i in range(1, num):
            for d, meshes in enumerate(thismeshes):
                try:
                    abij = rz_to_ab(rz, meshes[0])
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
                        assert (
                            early_exit == "ignore"
                        ), f'early_exit needs to be one of ["warn", "plot", "raise", ignore"] but got `{early_exit}`'
                    return out[:i]
                rz = ab_to_rz(*abij, meshes[1])
                last = meshes
            out[i] = rz
        return out
