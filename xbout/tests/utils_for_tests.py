from boutdata.data import BoutOptionsFile
import numpy as np
from pathlib import Path


def load_example_input(name, *, nx=None, ny=None, nz=None):
    # Find input file in the 'inputs' subdirectory
    path = Path(__file__).parent.joinpath("inputs", name)

    return BoutOptionsFile(path, nx=nx, ny=ny, nz=nz)


def set_geometry_from_input_file(ds, name):
    ds = ds.copy()  # do not modify existing ds

    options = load_example_input(
        name, nx=ds.metadata["nx"], ny=ds.metadata["ny"], nz=ds.metadata["nz"]
    )

    # Set nx and ny in the options in case these are needed for expressions, as this is
    # not done automatically by BoutOptionsFile
    options["mesh:nx"] = ds.metadata["nx"]
    options["mesh:ny"] = ds.metadata["ny"]

    if not ds.metadata["MXG"] == options._keys.get("MXG", 2):
        raise ValueError(
            f"MXG does not match, {ds.metadata['MXG']} in ds and "
            f"{options._keys.get('MXG', 2)} in options"
        )
    if not ds.metadata["MYG"] == options._keys.get("MYG", 2):
        raise ValueError(
            f"MYG does not match, {ds.metadata['MYG']} in ds and "
            f"{options._keys.get('MYG', 2)} in options"
        )

    if ds.metadata["keep_xboundaries"] or ds.metadata["MXG"] == 0:
        slicex = slice(None)
    else:
        mxg = ds.metadata["MXG"]
        slicex = slice(mxg, -mxg)

    if ds.metadata["keep_yboundaries"] or ds.metadata["MYG"] == 0:
        slicey = slice(None)
    else:
        myg = ds.metadata["MYG"]
        slicey = slice(myg, -myg)

    slices = (slicex, slicey)

    # Replace values of fields already created by bout_xyt_example_files
    shape_2d = ds["g11"].shape
    for v in [
        "g11",
        "g22",
        "g33",
        "g12",
        "g13",
        "g23",
        "g_11",
        "g_22",
        "g_33",
        "g_12",
        "g_13",
        "g_23",
        "J",
        "Bxy",
        "zShift",
        "dx",
        "dy",
    ]:
        for location in ["CELL_CENTRE", "CELL_XLOW", "CELL_YLOW", "CELL_ZLOW"]:
            suffix = "" if location == "CELL_CENTRE" else f"_{location}"
            # Need all arrays returned from options.evaluate() to be the right shape.
            # Recommend adding '0*x' or '0*y' in the input file expressions if the
            # expression would be 1d otherwise.
            ds[v + suffix] = ds[v].copy(
                data=np.broadcast_to(
                    options.evaluate(f"mesh:{v}", location=location).squeeze(axis=2)[
                        slices
                    ],
                    shape_2d,
                )
            )

    # Set dz as it would be calculated by BOUT++ (but don't support zmin, zmax or
    # zperiod here)
    if "dz" in options["mesh"]:
        ds.metadata["dz"] = options.evaluate_scalar("mesh:dz")
    else:
        ds.metadata["dz"] = 2.0 * np.pi / ds.metadata["nz"]

    # Add extra fields needed by "toroidal" geometry
    for v in ["Rxy", "Zxy", "psixy"]:
        for location in ["CELL_CENTRE", "CELL_XLOW", "CELL_YLOW", "CELL_ZLOW"]:
            suffix = "" if location == "CELL_CENTRE" else f"_{location}"
            # Need all arrays returned from options.evaluate() to be the right shape.
            # Recommend adding '0*x' or '0*y' in the input file expressions if the
            # expression would be 1d otherwise.
            ds[v + suffix] = ds["g11"].copy(
                data=np.broadcast_to(
                    options.evaluate(f"mesh:{v}", location=location).squeeze(axis=2)[
                        slices
                    ],
                    shape_2d,
                )
            )

    # Set fields that don't have to be in input files to NaN
    for v in ["G1", "G2", "G3"]:
        if v in ds:
            ds[v] = ds[v].copy(data=np.broadcast_to(float("nan"), ds[v].shape))

    return ds, options
