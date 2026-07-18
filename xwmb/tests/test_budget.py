import xgcm
import xwmb
import xbudget
import pytest
from xwmb.budget import _resolve_recipe
import xarray as xr
import numpy as np

def synthetic_dataset():
    x_f = np.array([-0.5, 0.5])
    x_c = 0.5*(x_f[:-1] + x_f[1:])

    y_f = np.array([0.5, 1.5, 2.5])
    y_c = 0.5*(y_f[:-1] + y_f[1:])

    lam_f = np.array([0, 1, 2])
    lam_c = 0.5*(lam_f[:-1] + lam_f[1:])

    t_f = np.array([0, 1])
    t_c = 0.5*(t_f[:-1] + t_f[1:])

    coords = {
        "x_c": x_c,
        "x_f": x_f,
        "y_c": y_c,
        "y_f": y_f,
        "lam_c": lam_c,
        "lam_f": lam_f,
        "t_c": t_c,
        "t_f": t_f
    }
    ds = xr.Dataset(coords=coords)
    ds = ds.assign_coords({
        "geolon": xr.broadcast(ds.x_c, ds.y_c)[0],
        "geolat": xr.broadcast(ds.x_c, ds.y_c)[0],
        "geolon_c": xr.broadcast(ds.x_f, ds.y_f)[0],
        "geolat_c": xr.broadcast(ds.x_f, ds.y_f)[1],
    })

    # Grid cell area
    ds["area"] = xr.ones_like(xr.broadcast(ds.x_c, ds.y_c)[0])

    # Time-averaged size and contours of water mass
    ds["lam"] = ds.lam_c * xr.ones_like(xr.broadcast(ds.t_c, ds.lam_c, ds.y_c, ds.x_c)[0])
    ds["thickness"] = xr.ones_like(ds.lam)

    # Bounding snapshots of size and contours of water mass
    ds["lam_bounds"] = ds.lam_c * xr.ones_like(xr.broadcast(ds.t_f, ds.lam_c, ds.y_c, ds.x_c)[0])
    ds["thickness_bounds"] = xr.ones_like(ds.lam_bounds)

    # Lateral mass transport
    ds["umo"] = xr.zeros_like(xr.broadcast(ds.t_c, ds.lam_c, ds.y_c, ds.x_f)[0])
    ds["vmo"] = xr.where(
        ds.y_f == 1.5,
        ds.lam_c - 1.,
        xr.zeros_like(xr.broadcast(ds.t_c, ds.lam_c, ds.y_f, ds.x_c)[0])
    )

    # Volume-integrated tendency
    ds["tend"] = xr.where(
        ds.y_c == 2.0,
        1.,
        xr.zeros_like(xr.broadcast(ds.t_c, ds.lam_c, ds.y_c, ds.x_c)[0])
    )
    return ds

def synthetic_grid():

    ds = synthetic_dataset()

    # Placeholder until https://github.com/hdrake/xbudget/issues/21
    ds = ds.rename({"t_c":"time", "t_f":"time_bounds"})

    coords = {
        "X": {"center":"x_c", "outer":"x_f"},
        "Y": {"center":"y_c", "outer":"y_f"},
        "Z": {"center":"lam_c", "outer":"lam_f"},
        "T": {"center":"time", "outer":"time_bounds"}
    }
    grid = xgcm.Grid(
        ds,
        coords = coords,
        padding = {"X": "extend", "Y":"extend", "Z":"extend", "T":"extend"},
        metrics = {("X","Y"): "area"},
        autoparse_metadata=False
    )

    return grid

def _minimal_recipe():
    return {
        "mass": {
            "thickness": "thickness",
            "rhs": {"sum": {"advection": {"sum": {"lateral": {"sum": {
                "zonal_convergence": {"product": {"zonal_divergence": {"difference": {"zonal_mass_transport": "umo"}}}},
                "meridional_convergence": {"product": {"meridional_divergence": {"difference": {"meridional_mass_transport": "vmo"}}}}
            }}}}}}
        },
        "tracer": {"lambda": "lam", "rhs": {"sum": {"tendency": {"var": "tend"}}}}
    }


def _collected(grid=None):
    """A grid plus a legacy-filled recipe, ready for WaterMassBudget."""
    grid = grid if grid is not None else synthetic_grid()
    recipe = _minimal_recipe()
    xbudget.collect_budgets(grid, recipe, name_scheme="legacy")
    return grid, recipe


# -- `xbudget_dict` -> `recipe` rename: the deprecation shim -----------------
#
# These exercise the shim itself, deliberately WITHOUT constructing a
# WaterMassBudget. Construction pulls in regionate/sectionate, whose xgcm 0.10
# migration is still in flux; coupling the shim tests to it made them fail for
# environment reasons and silently mask real shim regressions.


def _bare():
    """A WaterMassBudget instance with no __init__ run (attributes only)."""
    return object.__new__(xwmb.WaterMassBudget)


def test_resolve_recipe_positional():
    assert _resolve_recipe({"mass": {}}, None, "f") == {"mass": {}}


def test_resolve_recipe_deprecated_kwarg_warns():
    with pytest.warns(FutureWarning, match="xbudget_dict"):
        assert _resolve_recipe(None, {"mass": {}}, "f") == {"mass": {}}


def test_resolve_recipe_both_raises():
    with pytest.raises(TypeError, match="both"):
        _resolve_recipe({"mass": {}}, {"mass": {}}, "f")


def test_resolve_recipe_neither_raises():
    with pytest.raises(TypeError, match="recipe"):
        _resolve_recipe(None, None, "f")


def test_resolve_recipe_empty_recipe_is_not_missing():
    """An empty dict is a value, not an absence: presence is tested with `is None`."""
    assert _resolve_recipe({}, None, "f") == {}
    with pytest.warns(FutureWarning, match="xbudget_dict"):
        assert _resolve_recipe(None, {}, "f") == {}


def test_deprecated_full_xbudget_dict_property():
    wmb = _bare()
    wmb.full_recipe = {"mass": {"thickness": "thickness"}}
    with pytest.warns(FutureWarning, match="full_xbudget_dict"):
        assert wmb.full_xbudget_dict is wmb.full_recipe
    with pytest.warns(FutureWarning, match="full_xbudget_dict"):
        wmb.full_xbudget_dict = {"mass": {}}
    assert wmb.full_recipe == {"mass": {}}


def test_deprecated_boundary_property():
    """xgcm 0.10 renamed boundary->padding; the old attribute must still work."""
    wmb = _bare()
    wmb.padding = {"X": "extend"}
    with pytest.warns(FutureWarning, match="boundary"):
        assert wmb.boundary == {"X": "extend"}
    with pytest.warns(FutureWarning, match="boundary"):
        wmb.boundary = {"X": "periodic"}
    assert wmb.padding == {"X": "periodic"}


def test_recipe_positional_still_works():
    grid, recipe = _collected()
    wmb = xwmb.WaterMassBudget(grid, recipe, rho_ref=1.)
    assert wmb.full_recipe is recipe


def test_unfilled_recipe_raises_actionable_error():
    """A default (v1) collect leaves the recipe unfilled; say what to do."""
    grid = synthetic_grid()
    with pytest.raises(ValueError, match="name_scheme='legacy'"):
        xwmb.WaterMassBudget(grid, _minimal_recipe(), rho_ref=1.)


def test_mass_budget():
    grid = synthetic_grid()

    recipe = {
        "mass": {
            "thickness": "thickness",
            "rhs": {"sum": {"advection": {"sum": {"lateral": {"sum": {
                "zonal_convergence": {"product": {"zonal_divergence": {"difference": {"zonal_mass_transport": "umo"}}}},
                "meridional_convergence": {"product": {"meridional_divergence": {"difference": {"meridional_mass_transport": "vmo"}}}}
            }}}}}}
        },
        "tracer": {"lambda": "lam", "rhs": {"sum": {"tendency": {"var": "tend"}}}}
    }

    # xbudget >= 0.7 defaults to name_scheme="v1"; WaterMassBudget still uses the
    # deprecated aggregate() path, which needs the recipe filled in by a legacy run.
    xbudget.collect_budgets(grid, recipe, name_scheme="legacy")

    wmb = xwmb.WaterMassBudget(
        grid,
        recipe,
        rebin=False,
        rho_ref = 1.
    )

    wmb.mass_budget("tracer", bins=grid._ds.lam_f).compute()