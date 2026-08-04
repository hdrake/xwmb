"""Grid and recipe builders for the xwmb test suite.

The synthetic grids are tiny, data-free, and constructed to exercise each grid
topology xwmb claims to support: a plain single-tile grid, a single-tile grid with
a bipolar north fold (``padding={"Y": {"fold": ...}}``), and a genuinely multi-tile
grid joined by ``face_connections``. They run everywhere, including CI, which is
the point: the topology support is the headline feature of this release, and it
would otherwise be exercised only on a machine that happens to have a downloaded
netCDF file lying around.

Also here: the loader for the coarsened MOM6 example, for the tests that validate
against real ocean-model output when the file is present.

These live in a module rather than in ``conftest.py`` so that the test modules can
import the builders directly; ``conftest.py`` wraps them as pytest fixtures.
"""

import copy
import os

import numpy as np
import xarray as xr
import xgcm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MOM6_FILE = os.path.join(DATA_DIR, "MOM6_global_example_sigma2_budgets_v0_0_6.nc")

#: A minimal recipe: a mass budget whose lateral advection is umo/vmo, and a
#: tracer budget whose only right-hand-side process is a prescribed tendency.
#: ``units`` is declared on both, so the terms xbudget materializes -- and
#: everything xwmt and xwmb derive from them -- come back with units.
SYNTHETIC_RECIPE = {
    "mass": {
        "units": "kg s-1",
        "thickness": "thkcello",
        "rhs": {
            "sum": {
                "advection": {
                    "sum": {
                        "lateral": {
                            "sum": {
                                "zonal_convergence": {
                                    "product": {
                                        "zonal_divergence": {
                                            "difference": {
                                                "zonal_mass_transport": "umo"
                                            }
                                        },
                                        "sign": -1.0,
                                    }
                                },
                                "meridional_convergence": {
                                    "product": {
                                        "meridional_divergence": {
                                            "difference": {
                                                "meridional_mass_transport": "vmo"
                                            }
                                        },
                                        "sign": -1.0,
                                    }
                                },
                            }
                        }
                    }
                }
            }
        },
    },
    "tracer": {
        "units": "kg s-1",
        "lambda": "sigma",
        "rhs": {"sum": {"tendency": {"var": "tend"}}},
    },
}


def synthetic_recipe():
    """A fresh copy of :data:`SYNTHETIC_RECIPE` (a recipe is mutable shared state)."""
    return copy.deepcopy(SYNTHETIC_RECIPE)


# ---------------------------------------------------------------------------
# Single tile
# ---------------------------------------------------------------------------


def _single_tile_dataset():
    """A 1x2 horizontal box, two lambda layers, one time step.

    ``vmo`` is nonzero only on the interior face and varies with lambda, so a
    regional budget has a nontrivial, layer-dependent boundary transport.
    """
    x_f = np.array([-0.5, 0.5])
    y_f = np.array([0.5, 1.5, 2.5])
    sigma_f = np.array([0.0, 1.0, 2.0])
    t_f = np.array([0.0, 1.0])

    ds = xr.Dataset(
        coords={
            "x_c": 0.5 * (x_f[:-1] + x_f[1:]),
            "x_f": x_f,
            "y_c": 0.5 * (y_f[:-1] + y_f[1:]),
            "y_f": y_f,
            "sigma_c": 0.5 * (sigma_f[:-1] + sigma_f[1:]),
            "sigma_f": sigma_f,
            "t_c": 0.5 * (t_f[:-1] + t_f[1:]),
            "t_f": t_f,
        }
    )
    ds = ds.assign_coords(
        {
            "geolon": xr.broadcast(ds.x_c, ds.y_c)[0],
            "geolat": xr.broadcast(ds.y_c, ds.x_c)[0],
            "geolon_c": xr.broadcast(ds.x_f, ds.y_f)[0],
            "geolat_c": xr.broadcast(ds.y_f, ds.x_f)[0],
        }
    )

    ds["areacello"] = xr.ones_like(xr.broadcast(ds.x_c, ds.y_c)[0])
    ds["areacello"].attrs["units"] = "m2"

    ds["sigma"] = ds.sigma_c * xr.ones_like(
        xr.broadcast(ds.t_c, ds.sigma_c, ds.y_c, ds.x_c)[0]
    )
    ds["sigma"].attrs["units"] = "1"
    ds["thkcello"] = xr.ones_like(ds.sigma)
    ds["thkcello"].attrs["units"] = "m"

    ds["sigma_bounds"] = ds.sigma_c * xr.ones_like(
        xr.broadcast(ds.t_f, ds.sigma_c, ds.y_c, ds.x_c)[0]
    )
    ds["sigma_bounds"].attrs["units"] = "1"
    ds["thkcello_bounds"] = xr.ones_like(ds.sigma_bounds)
    ds["thkcello_bounds"].attrs["units"] = "m"

    ds["umo"] = xr.zeros_like(xr.broadcast(ds.t_c, ds.sigma_c, ds.y_c, ds.x_f)[0])
    ds["umo"].attrs["units"] = "kg s-1"
    ds["vmo"] = xr.where(
        ds.y_f == 1.5,
        ds.sigma_c - 1.0,
        xr.zeros_like(xr.broadcast(ds.t_c, ds.sigma_c, ds.y_f, ds.x_c)[0]),
    )
    ds["vmo"].attrs["units"] = "kg s-1"

    ds["tend"] = xr.where(
        ds.y_c == 2.0,
        1.0,
        xr.zeros_like(xr.broadcast(ds.t_c, ds.sigma_c, ds.y_c, ds.x_c)[0]),
    )
    ds["tend"].attrs["units"] = "kg s-1"
    return ds.rename({"t_c": "time", "t_f": "time_bounds"})


def make_single_tile_grid():
    return xgcm.Grid(
        _single_tile_dataset(),
        coords={
            "X": {"center": "x_c", "outer": "x_f"},
            "Y": {"center": "y_c", "outer": "y_f"},
            "Z": {"center": "sigma_c", "outer": "sigma_f"},
            "T": {"center": "time", "outer": "time_bounds"},
        },
        padding={"X": "extend", "Y": "extend", "Z": "extend", "T": "extend"},
        metrics={("X", "Y"): "areacello"},
        autoparse_metadata=False,
    )


# ---------------------------------------------------------------------------
# Bipolar north fold (tripolar)
# ---------------------------------------------------------------------------


def _fold_coords(Nx, Ny):
    """Corner and centre lon/lat for a fold grid, after regionate's own fixture.

    The bipolar seam is a *line* between two poles: seam corner ``(Ny, i)`` sits at
    a point parameterized by ``s(i) = min(i mod Nx, -i mod Nx)``, so only genuine
    mirror pairs coincide. A fold-straddling region can therefore only be stitched
    into one boundary loop through the grid topology, never through an accident of
    coordinates.
    """
    xq = np.arange(Nx + 1)
    yq = np.arange(Ny + 1)
    s = np.minimum(xq % Nx, (-xq) % Nx)
    t = (yq / Ny)[:, None]
    LONc = (1 - t) * (xq * 60.0)[None, :] + t * (s * 30.0)[None, :]
    LATc = 60.0 + t * (25.0 + s[None, :])
    LON = np.broadcast_to((np.arange(Nx) + 0.5) * 60.0, (Ny, Nx))
    LAT = np.broadcast_to(
        ((np.arange(Ny) + 0.5) * (30.0 / Ny) + 61.0)[:, None], (Ny, Nx)
    )
    return LONc, LATc, np.asarray(LON), np.asarray(LAT)


def make_fold_dataset(Nx=6, Ny=4, nlam=2, seed=0):
    """A fold grid's dataset: a prebinned lambda axis and fold-consistent flows.

    The fold's vector sign constraint on the seam row, ``V[Ny, i] = -V[Ny, Nx-1-i]``,
    is imposed explicitly. Without it the transport field is not one this grid could
    have produced, and no boundary tracing could reproduce its convergence.
    """
    rng = np.random.default_rng(seed)
    LONc, LATc, LON, LAT = _fold_coords(Nx, Ny)
    sigma_i = np.arange(nlam + 1, dtype=float)
    sigma_l = 0.5 * (sigma_i[:-1] + sigma_i[1:])

    U = rng.standard_normal((nlam, Ny, Nx + 1))
    V = rng.standard_normal((nlam, Ny + 1, Nx))
    for i in range(Nx):
        mirror = Nx - 1 - i
        if i < mirror:
            half = 0.5 * (V[:, Ny, i] - V[:, Ny, mirror])
            V[:, Ny, i], V[:, Ny, mirror] = half, -half

    ds = xr.Dataset(
        coords={
            "xh": ("xh", np.arange(Nx) + 0.5),
            "xq": ("xq", np.arange(Nx + 1).astype(float)),
            "yh": ("yh", np.arange(Ny) + 0.5),
            "yq": ("yq", np.arange(Ny + 1).astype(float)),
            "sigma_l": ("sigma_l", sigma_l),
            "sigma_i": ("sigma_i", sigma_i),
            "geolon_c": (("yq", "xq"), LONc),
            "geolat_c": (("yq", "xq"), LATc),
            "geolon": (("yh", "xh"), LON),
            "geolat": (("yh", "xh"), LAT),
        }
    )
    _add_common_fields(ds, ("sigma_l", "yh", "xh"), sigma_l, U, V, ("sigma_l", "yh", "xq"), ("sigma_l", "yq", "xh"), (Ny, Nx))
    return ds


def _add_common_fields(ds, center_dims, sigma_l, U, V, udims, vdims, hshape):
    """Attach the fields every synthetic budget needs, with units on all of them."""
    shape = tuple(ds.sizes[d] for d in center_dims)
    ds["areacello"] = (center_dims[1:], np.ones(hshape))
    ds["areacello"].attrs["units"] = "m2"
    ds["umo"] = (udims, U)
    ds["umo"].attrs["units"] = "kg s-1"
    ds["vmo"] = (vdims, V)
    ds["vmo"].attrs["units"] = "kg s-1"
    ds["thkcello"] = (center_dims, np.ones(shape))
    ds["thkcello"].attrs["units"] = "m"
    lam_field = np.broadcast_to(
        sigma_l.reshape([-1 if d == "sigma_l" else 1 for d in center_dims]), shape
    )
    ds["sigma"] = (center_dims, lam_field.copy())
    ds["sigma"].attrs["units"] = "1"
    ds["tend"] = (center_dims, np.zeros(shape))
    ds["tend"].attrs["units"] = "kg s-1"


def make_fold_grid(**kwargs):
    return xgcm.Grid(
        make_fold_dataset(**kwargs),
        coords={
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
            "Z": {"center": "sigma_l", "outer": "sigma_i"},
        },
        padding={"X": "periodic", "Y": {"fold": "corner"}, "Z": "extend"},
        metrics={("X", "Y"): "areacello"},
        autoparse_metadata=False,
    )


# ---------------------------------------------------------------------------
# Multi-tile (face_connections)
# ---------------------------------------------------------------------------

FACE_CONNECTIONS = {
    "face": {
        0: {"X": (None, (1, "X", False))},
        1: {"X": ((0, "X", False), None)},
    }
}


def make_tiled_dataset(Nc=6, nlam=2, seed=0):
    """Two faces side by side in longitude, joined at a single X seam.

    This synthetic grid uses symmetric ('outer') staggering, so the shared seam
    U-face is stored on *both* tiles; it is made single-valued by hand, exactly as
    regionate's own multi-tile fixture does. On a native 'left' grid (real ECCO) the
    seam face is stored once and no such fixup is needed.
    """
    rng = np.random.default_rng(seed)
    ng = Nc + 1
    yq = np.linspace(-45.0, 45.0, ng)
    yh = 0.5 * (yq[:-1] + yq[1:])
    lonq = [np.linspace(0.0, 90.0, ng), np.linspace(90.0, 180.0, ng)]
    lonh = [0.5 * (edges[:-1] + edges[1:]) for edges in lonq]

    LONc = np.stack([np.broadcast_to(lonq[f], (ng, ng)) for f in range(2)])
    LATc = np.stack([np.broadcast_to(yq[:, None], (ng, ng)) for f in range(2)])
    LON = np.stack([np.broadcast_to(lonh[f], (Nc, Nc)) for f in range(2)])
    LAT = np.stack([np.broadcast_to(yh[:, None], (Nc, Nc)) for f in range(2)])

    sigma_i = np.arange(nlam + 1, dtype=float)
    sigma_l = 0.5 * (sigma_i[:-1] + sigma_i[1:])

    U = rng.standard_normal((2, nlam, Nc, ng))
    V = rng.standard_normal((2, nlam, ng, Nc))
    # The seam is one face of the grid stored twice: face 0's xq=Nc *is* face 1's
    # xq=0. Give the two representations one value.
    U[1, :, :, 0] = U[0, :, :, Nc]

    ds = xr.Dataset(
        coords={
            "xq": ("xq", np.arange(ng)),
            "yq": ("yq", np.arange(ng)),
            "xh": ("xh", np.arange(Nc)),
            "yh": ("yh", np.arange(Nc)),
            "face": ("face", [0, 1]),
            "sigma_l": ("sigma_l", sigma_l),
            "sigma_i": ("sigma_i", sigma_i),
            "geolon_c": (("face", "yq", "xq"), LONc),
            "geolat_c": (("face", "yq", "xq"), LATc),
            "geolon": (("face", "yh", "xh"), LON),
            "geolat": (("face", "yh", "xh"), LAT),
        }
    )
    _add_common_fields(
        ds,
        ("face", "sigma_l", "yh", "xh"),
        sigma_l,
        U,
        V,
        ("face", "sigma_l", "yh", "xq"),
        ("face", "sigma_l", "yq", "xh"),
        (2, Nc, Nc),
    )
    return ds


def make_tiled_grid(**kwargs):
    return xgcm.Grid(
        make_tiled_dataset(**kwargs),
        coords={
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
            "Z": {"center": "sigma_l", "outer": "sigma_i"},
        },
        padding={"X": "fill", "Y": "fill", "Z": "extend"},
        fill_value=np.nan,
        face_connections=FACE_CONNECTIONS,
        metrics={("X", "Y"): "areacello"},
        autoparse_metadata=False,
    )


def mask_from_cells(grid, cells):
    """A boolean centre mask from ``[(j, i), ...]`` or ``{face: [(j, i), ...]}``."""
    template = grid._ds["geolon"]
    arr = np.zeros(template.shape, dtype=bool)
    if isinstance(cells, dict):
        for face, entries in cells.items():
            for j, i in entries:
                arr[face, j, i] = True
    else:
        for j, i in cells:
            arr[j, i] = True
    return xr.DataArray(arr, dims=template.dims, coords=template.coords)


# ---------------------------------------------------------------------------
# Real model output
# ---------------------------------------------------------------------------


def _construct_mom6_grid(ds):
    return xgcm.Grid(
        ds,
        coords={
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
            "Z": {"center": "sigma2_l", "outer": "sigma2_i"},
        },
        metrics={("X", "Y"): "areacello"},
        padding={"X": "periodic", "Y": "extend", "Z": "extend"},
        autoparse_metadata=False,
    )
