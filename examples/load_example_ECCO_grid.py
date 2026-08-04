"""Load the ECCOv4r4 native lat-lon-cap (LLC90) grid and 2010 budget diagnostics
as an ``xgcm.Grid`` ready for an ``xwmb`` water-mass budget.

The data is a curated subset of NASA's ECCO V4r4 state estimate redistributed on
Zenodo (record ``21479854``, concept DOI ``10.5281/zenodo.21479854``) so the example
runs without a NASA Earthdata Login. Each required file is downloaded to ``data_dir``
if missing. Please cite the original NASA/ECCO sources (see the Zenodo README).

The grid is genuinely multi-tile: the 13 LLC90 faces are stitched with
``face_connections`` so sections (and region boundaries) can be traced across tile
seams. The native MITgcm 'left' staggering (vorticity on the SW corner, ``XG/YG``)
is consumed directly by ``sectionate``/``regionate``.
"""

import os
import urllib.request

import numpy as np
import xeos
import xarray as xr
import xgcm

from ecco_budget_diagnostics import (
    eccov4r4_geothermal_heat_flux_tendency,
    eccov4r4_nonpenetrative_heat_flux_tendency,
    eccov4r4_penetrative_heat_flux_tendency,
)

ZENODO_RECORD = "21479854"
RHO0 = 1029.0  # ECCO reference density (kg/m^3), matches the xbudget ECCO preset

# ECCO example dataset files (record 21479854).
GEOMETRY = "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"
TS_MEAN = "OCEAN_TEMPERATURE_SALINITY_mon_mean_2010_ECCO_V4r4_native_llc0090.nc"
VOLUME_FLUX = "OCEAN_3D_VOLUME_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc"
TEMP_FLUX = "OCEAN_3D_TEMPERATURE_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc"
SALT_FLUX = "OCEAN_3D_SALINITY_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc"
SURFACE_HEAT = "OCEAN_AND_ICE_SURFACE_HEAT_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc"
SURFACE_FW = "OCEAN_AND_ICE_SURFACE_FW_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc"
BOLUS = "OCEAN_BOLUS_VELOCITY_mon_mean_2010_ECCO_V4r4_native_llc0090.nc"
TS_SNAP = "OCEAN_TEMPERATURE_SALINITY_snap_2010_ECCO_V4r4_native_llc0090.nc"
SSH_SNAP = "SEA_SURFACE_HEIGHT_snap_2010_ECCO_V4r4_native_llc0090.nc"
GEOTHERMAL = "GEOTHERMAL_FLUX_ECCO_V4r4_native_llc0090.nc"

# Canonical xgcm face_connections for the 13-tile LLC90 grid (xgcm ECCOv4 example).
LLC90_FACE_CONNECTIONS = {"tile": {
    0:  {"X": ((12, "Y", False), (3, "X", False)), "Y": (None,            (1, "Y", False))},
    1:  {"X": ((11, "Y", False), (4, "X", False)), "Y": ((0, "Y", False), (2, "Y", False))},
    2:  {"X": ((10, "Y", False), (5, "X", False)), "Y": ((1, "Y", False), (6, "X", False))},
    3:  {"X": ((0,  "X", False), (9, "Y", False)), "Y": (None,            (4, "Y", False))},
    4:  {"X": ((1,  "X", False), (8, "Y", False)), "Y": ((3, "Y", False), (5, "Y", False))},
    5:  {"X": ((2,  "X", False), (7, "Y", False)), "Y": ((4, "Y", False), (6, "Y", False))},
    6:  {"X": ((2,  "Y", False), (7, "X", False)), "Y": ((5, "Y", False), (10, "X", False))},
    7:  {"X": ((6,  "X", False), (8, "X", False)), "Y": ((5, "X", False), (10, "Y", False))},
    8:  {"X": ((7,  "X", False), (9, "X", False)), "Y": ((4, "X", False), (11, "Y", False))},
    9:  {"X": ((8,  "X", False), None),            "Y": ((3, "X", False), (12, "Y", False))},
    10: {"X": ((6,  "Y", False), (11, "X", False)), "Y": ((7, "Y", False), (2, "X", False))},
    11: {"X": ((10, "X", False), (12, "X", False)), "Y": ((8, "Y", False), (1, "X", False))},
    12: {"X": ((11, "X", False), None),            "Y": ((9, "Y", False), (0, "X", False))},
}}


def download_ECCO_file(filename, data_dir="../data/ecco"):
    """Return the local path to ``filename``, downloading it from Zenodo if missing."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{filename}?download=1"
        print(f"Downloading {filename} ...", flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def _open(filename, data_dir):
    return xr.open_dataset(download_ECCO_file(filename, data_dir=data_dir))


def _annual_mean(ds):
    """Annual (2010) mean of a monthly-mean dataset (time reduced away).

    The mean fields are kept *time-less*: xgcm routes vertical cumsum/interp on a
    ``face_connections`` grid through its face-padding path, which mishandles an extra
    (time) dimension, so the mean quantities must not carry one. The storage term
    instead lives on the independent ``time_bounds`` snapshot axis.
    """
    return ds.mean("time", keep_attrs=True)


def _time_bounds_snapshots(snap, varnames, data_dir):
    """Year-endpoint snapshots (first and last) renamed to ``{var}_bounds`` on a
    ``time_bounds`` outer axis, plus the ``time_bounds`` coordinate itself."""
    endpoints = snap.isel(time=[0, -1])
    out = xr.Dataset()
    for var in varnames:
        da = endpoints[var].rename({"time": "time_bounds"})
        out[f"{var}_bounds"] = da
    out = out.assign_coords(time_bounds=("time_bounds", endpoints["time"].values))
    return out


def assemble_ECCO_dataset(data_dir="../data/ecco"):
    """Merge the ECCO subset into one annual-mean-plus-snapshots budget dataset."""
    geom = _open(GEOMETRY, data_dir)

    # Annual means of the monthly-mean 3D fluxes and forcing.
    ts = _annual_mean(_open(TS_MEAN, data_dir)[["THETA", "SALT"]])
    vol = _annual_mean(_open(VOLUME_FLUX, data_dir)[["UVELMASS", "VVELMASS", "WVELMASS"]])
    tflux = _annual_mean(_open(TEMP_FLUX, data_dir)[
        ["ADVx_TH", "ADVy_TH", "ADVr_TH", "DFxE_TH", "DFyE_TH", "DFrE_TH", "DFrI_TH"]
    ])
    sflux = _annual_mean(_open(SALT_FLUX, data_dir)[
        ["ADVx_SLT", "ADVy_SLT", "ADVr_SLT", "DFxE_SLT", "DFyE_SLT", "DFrE_SLT",
         "DFrI_SLT", "oceSPtnd"]
    ])
    shf = _annual_mean(_open(SURFACE_HEAT, data_dir)[["TFLUX", "oceQsw"]])
    fwf = _annual_mean(_open(SURFACE_FW, data_dir)[["SFLUX", "oceFWflx"]])

    parts = [geom, ts, vol, tflux, sflux, shf, fwf]

    # Optional: bolus (GM) velocities for exact mass closure; geothermal for heat.
    try:
        bolus = _annual_mean(_open(BOLUS, data_dir)[["UVELSTAR", "VVELSTAR", "WVELSTAR"]])
        parts.append(bolus)
    except (FileNotFoundError, KeyError, OSError):
        print("BOLUS velocities unavailable; mass bolus convergence skipped.")
    try:
        geo = _open(GEOTHERMAL, data_dir)
        geoname = "geothermalFlux" if "geothermalFlux" in geo else list(geo.data_vars)[0]
        parts.append(geo[[geoname]].rename({geoname: "geothermalFlux"}))
    except (FileNotFoundError, OSError):
        print("GEOTHERMAL flux unavailable; bottom heat flux skipped.")

    # Year-endpoint snapshots -> *_bounds on the time_bounds axis (for the LHS storage).
    ts_snap = _open(TS_SNAP, data_dir)
    ssh_snap = _open(SSH_SNAP, data_dir)
    ssh_var = "ETAN" if "ETAN" in ssh_snap else list(ssh_snap.data_vars)[0]
    bounds = xr.merge([
        _time_bounds_snapshots(ts_snap, ["THETA", "SALT"], data_dir),
        _time_bounds_snapshots(ssh_snap.rename({ssh_var: "ETAN"}), ["ETAN"], data_dir),
    ])
    parts.append(bounds)

    ds = xr.merge(parts, compat="override")

    # Rename native ECCO coords to the geolon/geolat convention used by
    # sectionate/regionate, and expose lat/lon (needed by xwmt's TEOS-10 density).
    ds = ds.rename({"XC": "geolon", "YC": "geolat", "XG": "geolon_c", "YG": "geolat_c"})
    ds = ds.assign_coords(lat=ds["geolat"], lon=ds["geolon"])
    return ds


def add_derived_budget_terms(ds):
    """Add the ECCO-preset derived terms that are not raw diagnostics (in place)."""
    # dt (s) from the snapshot spacing, and cell volume.
    dt = ds["time_bounds"].diff("time_bounds").rename({"time_bounds": "time"})
    ds = ds.assign_coords(
        dt=("time", (dt / np.timedelta64(1, "s")).values.astype("float64")),
        volcello=(ds["drF"] * ds["hFacC"]) * ds["rA"],
    )

    # Surface/bottom heat forcing (penetrative + non-penetrative + geothermal).
    ds["pen_boundary_forcing_heat_tendency"] = eccov4r4_penetrative_heat_flux_tendency(ds)
    ds["nonpen_boundary_forcing_heat_tendency"] = eccov4r4_nonpenetrative_heat_flux_tendency(ds)
    ds["boundary_forcing_heat_tendency"] = (
        ds["pen_boundary_forcing_heat_tendency"]
        + ds["nonpen_boundary_forcing_heat_tendency"]
    )
    if "geothermalFlux" in ds:
        ds["geothermal_heat_flux_convergence"] = eccov4r4_geothermal_heat_flux_tendency(ds)

    # Salt surface forcing: 2D SFLUX into k=0 plus the 3D salt-plume oceSPtnd.
    SFLUX = ds["SFLUX"].assign_coords(k=0).expand_dims(dim="k", axis=1)
    ds["boundary_forcing_salt_tendency"] = xr.concat(
        [SFLUX + ds["oceSPtnd"], ds["oceSPtnd"].isel(k=slice(1, None))], dim="k"
    )

    # Mass: interior vertical transport (top interface zeroed) and FW volume forcing.
    ds["WVELMASS_interior"] = xr.where(ds["k_l"] != ds["k_l"].isel(k_l=0), ds["WVELMASS"], 0.0)
    k = ds["k"]
    ds["boundary_forcing_volume_tendency"] = xr.where(
        k == k.isel(k=0), ds["oceFWflx"].expand_dims({"k": k}), 0.0
    )

    # Snapshot water-mass fields (for the storage/tendency term): potential density
    # sigma2 and stretched thickness at the two year-endpoint snapshots. sigma2 is
    # computed from the snapshot potential temperature / practical salinity with the
    # MITgcm equation of state (JMD95, ECCOv4r4's EOS), via xeos -- the same EOS used
    # for the annual-mean density below (`eos="jmd95"`), referenced to 2000 dbar.
    ds["sigma2_bounds"] = xr.apply_ufunc(
        lambda t, s: xeos.rho(t, s, 2000.0, eos="jmd95") - 1000.0,
        ds["THETA_bounds"], ds["SALT_bounds"], dask="parallelized",
        output_dtypes=[float],
    )
    ds["thkcello_bounds"] = (ds["drF"] * ds["hFacC"]) * (
        1.0 + ds["ETAN_bounds"] / ds["Depth"]
    )

    # Cell-center thickness (m) used by xwmt as the layer metric.
    ds["thkcello"] = (ds["drF"] * ds["hFacC"]).transpose("k", "tile", "j", "i")

    # Face mass transports (kg/s) for sectionate's convergent transport.
    ds["umo"] = (ds["UVELMASS"] * ds["dyG"] * ds["drF"] * RHO0).transpose(
        "k", "tile", "j", "i_g"
    )
    ds["vmo"] = (ds["VVELMASS"] * ds["dxG"] * ds["drF"] * RHO0).transpose(
        "k", "tile", "j_g", "i"
    )

    # Uniform dask chunks (all tiles in one chunk so the LLC divergence can stitch
    # across seams) so xbudget's diffs/divergences run without chunk conflicts.
    chunks = {
        "tile": 13, "i": 90, "j": 90, "i_g": 90, "j_g": 90,
        "k": 50, "k_l": 50, "k_u": 50, "k_p1": 51, "time": 1, "time_bounds": 2,
    }
    ds = ds.fillna(0.0)
    return ds.chunk({d: n for d, n in chunks.items() if d in ds.dims})


def construct_budget_grid(ds):
    """The native ('left'-staggered) LLC90 grid used to *collect* the budget.

    Z is center=``k`` / left=``k_l`` so xbudget's vertical-flux differences land on
    ``k``; the time axis is center=``time`` / outer=``time_bounds`` for the LHS
    storage tendencies.
    """
    coords = {
        "X": {"center": "i", "left": "i_g"},
        "Y": {"center": "j", "left": "j_g"},
        "T": {"center": "time", "outer": "time_bounds"},
        "Z": {"center": "k", "left": "k_l"},
    }
    metrics = {
        ("X",): ["dxG"],
        ("Y",): ["dyG"],
        ("Z",): ["drF"],
        ("X", "Y"): ["rA", "rAw", "rAs"],
    }
    padding = {"X": "fill", "Y": "fill", "Z": "fill", "T": "fill"}
    return xgcm.Grid(
        ds,
        coords=coords,
        metrics=metrics,
        padding=padding,
        fill_value={"Z": 0.0},
        face_connections=LLC90_FACE_CONNECTIONS,
        autoparse_metadata=False,
    )


def construct_wmb_grid(ds):
    """The LLC90 grid used by ``xwmb``: Z is center=``z_l`` / outer=``z_i`` (so tracers
    can be interpolated to interfaces and remapped conservatively into sigma space).

    The native integer level dims ``k``/``k_p1`` are renamed to ``z_l``/``z_i`` and
    given physical-depth coordinates. sectionate requires the layer/interface names to
    follow the ``_l``/``_i`` convention (``layer.replace("l","i") == interface``), which
    ``z_l``/``z_i`` satisfy; the physical depths make the conservative vertical remap's
    target monotonic.
    """
    ds = ds.rename({"k": "z_l", "k_p1": "z_i"})
    ds = ds.assign_coords(z_l=("z_l", ds["Z"].values), z_i=("z_i", ds["Zp1"].values))
    coords = {
        "X": {"center": "i", "left": "i_g"},
        "Y": {"center": "j", "left": "j_g"},
        "Z": {"center": "z_l", "outer": "z_i"},
    }
    metrics = {("X", "Y"): ["rA"]}
    padding = {"X": "fill", "Y": "fill", "Z": "extend"}
    return xgcm.Grid(
        ds,
        coords=coords,
        metrics=metrics,
        padding=padding,
        face_connections=LLC90_FACE_CONNECTIONS,
        autoparse_metadata=False,
    )
