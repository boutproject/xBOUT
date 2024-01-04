import xarray as xr
import numpy as np


class rotating_circle:
    def __init__(self, num, **kw):
        print(num, kw)
        self.num = num
        self.kw = kw
        self.R0 = 5
        self.r0 = 1
        self.r1 = 2

    def gends(self, shape):
        nx, ny, nz = shape
        ds = xr.Dataset()

        x, y, z = np.meshgrid(
            np.linspace(self.r0, self.r1, nx, endpoint=True),
            np.linspace(0, np.pi * 2 / self.num, ny, endpoint=False),
            np.linspace(0, 2 * np.pi, nz, endpoint=False),
            sparse=False,
            indexing="ij",
        )
        dims = "x", "y", "z"
        ds["R"] = dims, x * np.sin(z) + self.R0
        ds["Z"] = dims, x * np.cos(z)
        ds["y"] = y[0, :, 0]
        ds["dy"] = dims, x * 0 + np.pi * 2 / self.num / ny
        shift = x / np.max(x) * 2 * np.pi / self.num / ny
        ds["forward_R"] = dims, x * np.sin(z + shift) + self.R0
        ds["forward_Z"] = dims, x * np.cos(z + shift)
        ds["backward_R"] = dims, x * np.sin(z - shift) + self.R0
        ds["backward_Z"] = dims, x * np.cos(z - shift)

        return ds

    def rand_rpt(self, i):
        slc = [slice(None)]
        if isinstance(i, np.ndarray):
            slc += [None] * len(i.shape)
        else:
            slc += [None]
        tmp = (
            np.random.random((3, i))
            * np.array([self.geom.r * 0.9, np.pi * 2, np.pi * 2])[tuple(slc)]
        )
        r, p, t = tmp
        return r, p, t

    # def rand(self, i):
    #     return self.geom.rpt_to_rpz(*self.rand_rpt(i))

    # def phi_test_value(self, a, b):
    #     dphi = self.dphi
    #     phi = np.linspace(dphi / 2, 2 * np.pi / 5 - dphi / 2, self.shape[2])
    #     self.ds["var"] = self.dims, np.zeros(self.shape) + phi
    #     for r, p, z in [self.rand(a) for _ in range(b)]:
    #         exp = (np.round((p - dphi / 2) / dphi) % self.shape[2]) * dphi + dphi / 2
    #         got = self.ds.emc3.evaluate_at_rpz(r, p, z, "var", updownsym=self.geom.sym)[
    #             "var"
    #         ]
    #         assert np.allclose(
    #             exp,
    #             got,
    #         )

    # def rpt_to_rpz
