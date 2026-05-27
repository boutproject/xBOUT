import numpy as np
import pytest


adios2 = pytest.importorskip("adios2")
xr = pytest.importorskip("xarray")


def test_adios2_roundtrip_dataset_attrs_and_vars(tmp_path):
    from xbout.adioswriter import write_dataset_bp
    from xbout.xarraybackend import BoutAdiosBackendEntrypoint

    ds = xr.Dataset(
        data_vars={
            "a": (("t", "x"), np.arange(6, dtype=np.float32).reshape(3, 2)),
            "flag": (("t",), np.array([True, False, True], dtype=bool)),
            "scalar": ((), np.int64(7)),
        },
        coords={
            "t": ("t", np.array([0.0, 0.5, 1.0], dtype=np.float64)),
            "x": ("x", np.array([10, 20], dtype=np.int32)),
        },
        attrs={"title": "roundtrip", "answer": 42},
    )
    ds["a"].attrs["units"] = "arb"

    path = tmp_path / "roundtrip.bp"
    write_dataset_bp(ds, str(path), time_dim="t", overwrite=True)

    ds2 = BoutAdiosBackendEntrypoint().open_dataset(str(path))
    try:
        assert ds2.attrs["title"] == "roundtrip"
        assert int(ds2.attrs["answer"]) == 42

        np.testing.assert_allclose(ds2["t"].values, ds["t"].values)
        np.testing.assert_allclose(ds2["x"].values, ds["x"].values)
        np.testing.assert_allclose(ds2["a"].values, ds["a"].values)
        assert ds2["a"].attrs["units"] == "arb"

        assert ds2["flag"].dtype == bool
        assert ds2["flag"].dims == ("t",)
        np.testing.assert_array_equal(ds2["flag"].values, ds["flag"].values)

        assert int(ds2["scalar"].values) == 7
    finally:
        ds2.close()

