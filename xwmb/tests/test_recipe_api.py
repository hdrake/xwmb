"""The xbudget 0.8.0 recipe API."""

import numpy as np
import pytest
import xbudget

import xwmb
from xwmb.mass import MASS_SOURCE_PATH, mass_source_varname
from xwmb.transport import transport_varnames

from .synthetic import synthetic_recipe


# ---------------------------------------------------------------------------
# The renamed arguments are renamed, not aliased
# ---------------------------------------------------------------------------


def test_recipe_is_required(single_tile_grid):
    with pytest.raises(TypeError):
        xwmb.WaterMassBudget(single_tile_grid)


def test_the_old_spellings_are_gone(single_tile_grid):
    """v0.7.0 renames without aliasing: the whole stack breaks compatibility here,
    so carrying a shim would only have let a caller *think* they had migrated."""
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    with pytest.raises(TypeError):
        xwmb.WaterMassBudget(single_tile_grid, xbudget_dict=recipe)
    with pytest.raises(TypeError):
        xwmb.WaterMassBudget(single_tile_grid, recipe, teos10=True)
    wmb = xwmb.WaterMassBudget(single_tile_grid, recipe, rho_ref=1.0)
    assert not hasattr(wmb, "full_xbudget_dict")


def test_default_bins_is_spelled_on_bins_now(single_tile_grid):
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    with pytest.raises(TypeError):
        xwmb.WaterMassBudget(single_tile_grid, recipe, rho_ref=1.0).mass_budget(
            "tracer", default_bins=True
        )


def test_bins_default_builds_a_target_grid(single_tile_grid):
    """`bins="default"` replaces the capability that only `default_bins=True` had.

    Exercised at the coordinate layer: the default bin edges are only defined for
    the density/heat/salt lambdas, and the synthetic recipe's tracer is none of
    those.
    """
    from xwmb.coordinates import resolve_target_coords

    grid, target = resolve_target_coords(
        single_tile_grid, "sigma", "sigma2", bins="default"
    )
    assert "Z_target" in grid.axes
    edges = grid._ds[target["outer"]].values
    assert edges[0] == 0.0 and np.isclose(edges[1] - edges[0], 0.05)


def test_bins_rejects_an_unknown_string(single_tile_grid):
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    wmb = xwmb.WaterMassBudget(single_tile_grid, recipe, rho_ref=1.0)
    with pytest.raises(ValueError, match="only string accepted"):
        wmb.mass_budget("tracer", bins="defaults")


# ---------------------------------------------------------------------------
# Variable names come from the query, not from hardcoded strings
# ---------------------------------------------------------------------------


def test_transport_names_resolved_from_the_recipe():
    recipe = xbudget.load_preset_budget(model="MOM6")
    assert transport_varnames(xbudget.BudgetQuery(None, recipe)) == {
        "utr": "umo",
        "vtr": "vmo",
    }


def test_mass_source_name_is_the_0_8_name_not_the_legacy_one():
    """xbudget 0.8.0 dropped the operator infixes from derived-variable names.

    The previously hardcoded `"mass_rhs_sum_surface_exchange_flux"` matches nothing
    under 0.8.0, and because the lookup was guarded by `if ... in grid._ds` the mass
    source was silently dropped from every budget rather than raising.
    """
    recipe = xbudget.load_preset_budget(model="MOM6")
    name = mass_source_varname(xbudget.BudgetQuery(None, recipe))
    assert name == "mass_rhs_surface_exchange_flux"
    assert "_sum_" not in name


def test_missing_terms_resolve_to_none_rather_than_raising():
    """A recipe with no surface mass exchange has no mass source -- not an error."""
    query = xbudget.BudgetQuery(None, synthetic_recipe())
    assert mass_source_varname(query) is None


def test_transport_names_none_without_lateral_advection():
    recipe = synthetic_recipe()
    del recipe["mass"]["rhs"]["sum"]["advection"]
    assert transport_varnames(xbudget.BudgetQuery(None, recipe)) is None


def test_query_is_available_on_the_instance(single_tile_grid):
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    wmb = xwmb.WaterMassBudget(single_tile_grid, recipe, rho_ref=1.0)
    assert isinstance(wmb.query, xbudget.BudgetQuery)
    assert wmb.query.var(("mass", "rhs")) == "mass_rhs"
    assert MASS_SOURCE_PATH == ("mass", "rhs", "surface_exchange_flux")
