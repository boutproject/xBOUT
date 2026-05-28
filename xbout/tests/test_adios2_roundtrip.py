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


def test_adios2_write_ints_as_int32_on_disk(tmp_path):
    from xbout.adioswriter import write_dataset_bp

    ds = xr.Dataset(
        data_vars={
            "i64": (("t",), np.array([1, 2, 3], dtype=np.int64)),
            "u64_small": (("t",), np.array([0, 7, 42], dtype=np.uint64)),
            "f32": (("t",), np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        },
        coords={"t": ("t", np.array([0, 1, 2], dtype=np.int32))},
    )

    path = tmp_path / "ints_as_int32.bp"
    write_dataset_bp(
        ds, str(path), time_dim="t", overwrite=True, write_ints_as_int32=True
    )

    fh = adios2.FileReader(str(path))
    try:
        vars = fh.available_variables()
        assert vars["i64"]["Type"] == "int32_t"
        assert vars["u64_small"]["Type"] == "int32_t"
        assert vars["t"]["Type"] == "int32_t"
        assert vars["f32"]["Type"] == "float"
    finally:
        fh.close()


def test_adios2_write_ints_as_int32_overflow_raises(tmp_path):
    from xbout.adioswriter import write_dataset_bp

    ds = xr.Dataset(
        data_vars={
            "too_big": (("t",), np.array([np.iinfo(np.int32).max + 1], dtype=np.int64)),
        },
        coords={"t": ("t", np.array([0], dtype=np.int32))},
    )

    path = tmp_path / "overflow.bp"
    with pytest.raises(ValueError, match=r"Cannot safely cast"):
        write_dataset_bp(
            ds, str(path), time_dim="t", overwrite=True, write_ints_as_int32=True
        )
