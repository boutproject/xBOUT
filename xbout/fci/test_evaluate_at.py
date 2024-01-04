# SPDX-FileCopyrightText: 2023 David Bold <dave@ipp.mpg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import warnings
from timeit import timeit

import numpy as np
import pytest
import xarray as xr

# from ..core import dataset
from . import gen_ds

import xbout.fci.evaluate_at

try:
    import matplotlib  # type: ignore
except ImportError:
    matplotlib = None  # type: ignore


class Test_eval_at_rpz(object):
    def setitup(self, shape=None, **kwargs):
        if not isinstance(shape, tuple):
            shape = None
        self.shape = shape or (2, 4, 20)
        self.geom = gen_ds.rotating_circle(5, **kwargs)
        self.ds = self.geom.gends(self.shape)
        self.dims = self.ds["R"].dims
        self.dphi = 2 * np.pi / self.shape[2] / 5

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

    def rand(self, i):
        return self.geom.rpt_to_rpz(*self.rand_rpt(i))

    def do_test_value(self, key, func=None, shape=None, **tol):
        a, b = 2, 4
        self.setitup(shape=shape)
        keytoindex = {"R": 0, "Z": 2, "y": 1}
        dx = (self.geom.r1 - self.geom.r0) * 0.9
        print(dx)
        for _ in range(a):
            Rpz = (
                np.random.random((3, b)) * np.array([dx, np.pi * 2, np.pi * 2])[:, None]
                + np.array([self.geom.R0, 0, 0])[:, None]
            )
            xyz = (
                np.random.random((3, b)) * np.array([dx, np.pi * 2, np.pi * 2])[:, None]
                + np.array([self.geom.r0, 0, 0])[:, None]
            )
            R = np.cos(xyz[2]) * xyz[0] + self.geom.R0
            Z = np.sin(xyz[2]) * xyz[0]
            Rpz = R, xyz[1], Z
            expect = Rpz[keytoindex[key]]
            key0 = key
            if func:
                expect = func(expect)
                self.ds["tmp"] = func(self.ds[key])
                key0 = "tmp"
            ret = xbout.fci.evaluate_at.evaluate_at_rpz(self.ds, *Rpz, key0)
            print(np.max(np.abs(ret[key0] - expect)))
            assert np.allclose(ret[key0], expect, **tol)

    def test_R(self):
        self.do_test_value("R", None, atol=3e-3)

    def test_phi(self):
        self.do_test_value(
            "y", func=lambda x: np.sin(x * 5), atol=2e-2, shape=(3, 20, 20)
        )

    def test_z(self):
        self.do_test_value("Z", atol=3e-3)

    def _test_debug(self):
        func = None
        for shape in (2, 4, 20), (20, 4, 20), (2, 40, 20), (2, 4, 200):
            a, b = 30, 40
            self.setitup(shape=shape)
            tol = {}
            R = 6.3
            y = np.linspace(0, 2 * np.pi, 1000)
            z = 0
            key = "R"
            r = xbout.fci.evaluate_at.evaluate_at_rpz(self.ds, R, y, z, key)[key]
            import matplotlib.pyplot as plt

            plt.plot(y, r)
            plt.savefig(f"debug_{shape}.png")
            plt.cla()
            plt.show()
        raise


if __name__ == "__main__":
    Test_eval_at_rpz().test_nan_value()
