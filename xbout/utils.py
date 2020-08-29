from copy import deepcopy
from itertools import chain
from pathlib import Path

import numpy as np
import xarray as xr


def _set_attrs_on_all_vars(ds, key, attr_data, copy=False):
    ds.attrs[key] = attr_data
    if copy:
        for v in chain(ds.data_vars, ds.coords):
            ds[v].attrs[key] = deepcopy(attr_data)
    else:
        for v in chain(ds.data_vars, ds.coords):
            ds[v].attrs[key] = attr_data
    return ds


def _add_attrs_to_var(ds, varname, copy=False):
    if copy:
        for attr in ["metadata", "options", "geometry", "regions"]:
            if attr in ds.attrs and attr not in ds[varname].attrs:
                ds[varname].attrs[attr] = deepcopy(ds.attrs[attr])
    else:
        for attr in ["metadata", "options", "geometry", "regions"]:
            if attr in ds.attrs and attr not in ds[varname].attrs:
                ds[varname].attrs[attr] = ds.attrs[attr]


def _check_filetype(path):
    if path.suffix == '.nc':
        filetype = 'netcdf4'
    elif path.suffix == '.h5netcdf':
        filetype = 'h5netcdf'
    else:
        raise IOError("Do not know how to read file extension {}"
                      .format(path.suffix))
    return filetype


def _separate_metadata(ds):
    """
    Extract the metadata (nxpe, myg etc.) from the Dataset.

    Assumes that all scalar variables are metadata, not physical data!
    """

    # Find only the scalar variables
    variables = list(ds.variables)
    scalar_vars = [var for var in variables
                   if not any(dim in ['t', 'x', 'y', 'z'] for dim in ds[var].dims)]

    # Save metadata as a dictionary
    metadata_vals = [ds[var].values.item() for var in scalar_vars]
    metadata = dict(zip(scalar_vars, metadata_vals))

    return ds.drop(scalar_vars), metadata


def _update_metadata_increased_resolution(da, n):
    """
    Update the metadata variables to account for a y-direction resolution increased by a
    factor n.

    Parameters
    ----------
    da : DataArray
        The variable to update
    n : int
        The factor to increase the y-resolution by
    """

    # Take deepcopy to ensure we do not alter metadata of other variables
    da.attrs['metadata'] = deepcopy(da.metadata)

    def update_jyseps(name):
        # If any jyseps<=0, need to leave as is
        if da.metadata[name] > 0:
            da.metadata[name] = n*(da.metadata[name] + 1) - 1
    update_jyseps('jyseps1_1')
    update_jyseps('jyseps2_1')
    update_jyseps('jyseps1_2')
    update_jyseps('jyseps2_2')

    def update_ny(name):
        da.metadata[name] = n*da.metadata[name]
    update_ny('ny')
    update_ny('ny_inner')
    update_ny('MYSUB')

    # Update attrs of coordinates to be consistent with da
    for coord in da.coords:
        da[coord].attrs = {}
        _add_attrs_to_var(da, coord)

    return da


def _check_new_nxpe(ds, nxpe):
    # Check nxpe is valid
    if (ds.metadata["nx"] - 2*ds.metadata["MXG"]) % nxpe != 0:
        raise ValueError(
            f"nxpe={nxpe} must divide total number of points "
            f"nx-2*MXG={ds.metadata['nx'] - 2*ds.metadata['MXG']}"
        )
    if (
        ds.metadata["ixseps1"] >= 0
        and ds.metadata["ixseps1"] <= ds.metadata["nx"]
        and (ds.metadata["ixseps1"] - ds.metadata["MXG"]) % nxpe != 0
    ):
        raise ValueError(
            f"nxpe={nxpe} must divide number of points inside first separatrix "
            f"ixseps1-MXG={ds.metadata['ixseps1'] - ds.metadata['MXG']}"
        )
    if (
        ds.metadata["ixseps2"] >= 0
        and ds.metadata["ixseps2"] <= ds.metadata["nx"]
        and (ds.metadata["ixseps2"] - ds.metadata["MXG"]) % nxpe != 0
    ):
        raise ValueError(
            f"nxpe={nxpe} must divide number of points inside second separatrix "
            f"ixseps2-MXG={ds.metadata['ixseps2'] - ds.metadata['MXG']}"
        )


def _check_new_nype(ds, nype):
    # Check nype is valid

    ny = ds.metadata["ny"]

    if ny % nype != 0:
        raise ValueError(
            f"nype={nype} must divide total number of points ny={ds.metadata['ny']}"
        )
    if (
        ds.metadata["jyseps1_1"] >= 0
        and ds.metadata["jyseps1_1"] <= ny
        and (ds.metadata["jyseps1_1"] + 1) % nype != 0
    ):
        raise ValueError(
            f"nype={nype} must divide jyseps1_1+1={ds.metadata['jyseps1_1'] + 1}"
        )
    if (
        ds.metadata["jyseps2_1"] >= 0
        and ds.metadata["jyseps2_1"] <= ny
        and ds.metadata["jyseps1_2"] != ds.metadata["jyseps2_1"]
        and (ds.metadata["jyseps2_1"] + 1) % nype != 0
    ):
        raise ValueError(
            f"nype={nype} must divide jyseps2_1+1={ds.metadata['jyseps2_1'] + 1}"
        )
    if (
        ds.metadata["jyseps1_2"] != ds.metadata["jyseps2_1"]
        and ds.metadata["ny_inner"] % nype != 0
    ):
        raise ValueError(
            f"nype={nype} must divide ny_inner={ds.metadata['ny_inner']}"
        )
    if (
        ds.metadata["jyseps1_2"] >= 0
        and ds.metadata["jyseps1_2"] <= ny
        and ds.metadata["jyseps1_2"] != ds.metadata["jyseps2_1"]
        and (ds.metadata["jyseps1_2"] + 1) % nype != 0
    ):
        raise ValueError(
            f"nype={nype} must divide jyseps1_2+1={ds.metadata['jyseps1_2'] + 1}"
        )
    if (
        ds.metadata["jyseps2_2"] >= 0
        and ds.metadata["jyseps2_2"] <= ny
        and (ds.metadata["jyseps2_2"] + 1) % nype != 0
    ):
        raise ValueError(
            f"nype={nype} must divide jyseps2_2+1={ds.metadata['jyseps2_2'] + 1}"
        )


def _pad_x_boundaries(ds):
    # pad Dataset if boundaries are not already present

    mxg = ds.metadata["MXG"]
    xcoord = ds.metadata.get("bout_xdim", "x")

    if not ds.metadata["keep_xboundaries"] and mxg > 0:
        boundary_pad = ds.isel({xcoord: slice(mxg)}).copy(deep=True).load()
        for v in boundary_pad:
            if xcoord in boundary_pad[v].dims:
                boundary_pad[v].values[...] = np.nan
        ds = xr.concat(
            [boundary_pad, ds.load, boundary_pad], dim=xcoord, data_vars="minimal"
        )

    return ds


def _pad_y_boundaries(ds):
    # pad Dataset if boundaries are not already present

    myg = ds.metadata["MYG"]
    ycoord = ds.metadata.get("bout_ydim", "y")
    has_second_divertor = (ds.metadata["jyseps2_1"] != ds.metadata["jyseps1_2"])

    if not ds.metadata["keep_yboundaries"] and myg > 0:
        boundary_pad = ds.isel({ycoord: slice(myg)}).copy(deep=True).load()
        for v in boundary_pad:
            if ycoord in boundary_pad[v].dims:
                boundary_pad[v].values[...] = np.nan
        if not has_second_divertor:
            ds = xr.concat(
                [boundary_pad, ds.load(), boundary_pad], dim=ycoord, data_vars="minimal"
            )
        else:
            # Include second divertor
            ny_inner = ds.metadata["ny_inner"]
            ds = xr.concat(
                [
                    boundary_pad,
                    ds.isel({ycoord: slice(ny_inner)}),
                    boundary_pad,
                    boundary_pad,
                    ds.isel({ycoord: slice(ny_inner, None)}),
                    boundary_pad,
                ],
                dim=ycoord,
                data_vars="minimal",
            )

    return ds


def _split_into_restarts(ds, variables, savepath, nxpe, nype, tind, prefix, overwrite):

    _check_new_nxpe(ds, nxpe)
    _check_new_nype(ds, nype)

    # Create paths for the files. Do this first so we can check if files exist before
    # spending time creating restart Datasets
    paths = []
    for xproc in range(nxpe):
        for yproc in range(nype):
            # Global processor number
            i = yproc*nxpe + xproc

            paths.append(Path(savepath).joinpath(f"{prefix}.{i}.nc"))
            if not overwrite and paths[-1].exists():
                raise ValueError(
                    f"File {paths[-1]} already exists. Pass overwrite=True to overwrite."
                )

    mxg = ds.metadata["MXG"]
    myg = ds.metadata["MYG"]

    ny_inner = ds.metadata["ny_inner"]

    tcoord = ds.metadata.get("bout_tdim", "t")
    xcoord = ds.metadata.get("bout_xdim", "x")
    ycoord = ds.metadata.get("bout_ydim", "y")

    # These variables need to be saved to restart files in addition to evolving ones
    restart_metadata_vars = [
        "zperiod",
        "MZSUB",
        "MXG",
        "MYG",
        "MZG",
        "nx",
        "ny",
        "nz",
        "MZ",
        "NZPE",
        "ixseps1",
        "ixseps2",
        "jyseps1_1",
        "jyseps2_1",
        "jyseps1_2",
        "jyseps2_2",
        "ny_inner",
        "ZMAX",
        "ZMIN",
        "dz",
        "BOUT_VERSION",
    ]

    if variables is None:
        # If variables to be saved were not specified, add all time-evolving variables
        variables = [v for v in ds if tcoord in ds[v].dims]

    # Add extra variables always needed
    for v in [
        "dx",
        "dy",
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
    ]:
        if v not in variables:
            variables.append(v)

    # number of points in the domain on each processor, not including guard or boundary
    # points
    mxsub = (ds.metadata["nx"] - 2*mxg) // nxpe
    mysub = ds.metadata["ny"] // nype

    # hist_hi represents the number of iterations before the restart. Attempt to
    # reconstruct here
    iteration = ds.metadata.get("iteration", -1)
    nt = ds.sizes[tcoord]
    hist_hi = iteration - (nt - tind)
    if hist_hi < 0:
        hist_hi = -1

    has_second_divertor = (ds.metadata["jyseps2_1"] != ds.metadata["jyseps1_2"])

    # select desired time-index for the restart files
    ds = ds.isel({tcoord: tind}).persist()

    ds = _pad_x_boundaries(ds)
    ds = _pad_y_boundaries(ds)

    restart_datasets = []
    for xproc in range(nxpe):
        xslice = slice(xproc*mxsub, (xproc + 1)*mxsub + 2*mxg)
        for yproc in range(nype):
            print(f"creating {xproc*nype + yproc + 1}/{nxpe*nype}", end="\r")
            if (
                has_second_divertor
                and yproc*mysub >= ny_inner
            ):
                yslice = slice(yproc*mysub + 2*myg, (yproc + 1)*mysub + 4*myg)
            else:
                yslice = slice(yproc*mysub, (yproc + 1)*mysub + 2*myg)

            ds_slice = ds.isel({xcoord: xslice, ycoord: yslice})

            restart_ds = xr.Dataset()
            for v in variables:
                data_variable = ds_slice[v].variable

                # delete attrs so we don't try to save metadata dict to restart files
                data_variable.attrs = {}

                restart_ds[v] = data_variable
            for v in restart_metadata_vars:
                restart_ds[v] = ds.metadata[v]
            restart_ds["MXSUB"] = mxsub
            restart_ds["MYSUB"] = mysub
            restart_ds["NXPE"] = nxpe
            restart_ds["NYPE"] = nype
            restart_ds["hist_hi"] = hist_hi

            # tt is the simulation time where the restart happens
            # t_array variable may have been assigned as a coordinate and renamed as
            # tcoord
            t_array = ds.get("t_array", tcoord)
            restart_ds["tt"] = t_array.values.flatten()[0]

            restart_datasets.append(restart_ds)

    return restart_datasets, paths
