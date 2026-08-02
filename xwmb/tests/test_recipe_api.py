"""The xbudget 0.8.0 recipe API, and the deprecation of the old spelling."""

import pytest
import xbudget

import xwmb
from xwmb.budget import _resolve_recipe
from xwmb.mass import MASS_SOURCE_PATH, mass_source_varname
from xwmb.transport import transport_varnames

from .synthetic import synthetic_recipe


# ---------------------------------------------------------------------------
# `xbudget_dict` -> `recipe`
# ---------------------------------------------------------------------------


def test_recipe_passed_positionally():
    recipe = {"mass": {}}
    assert _resolve_recipe(recipe, None, "f") is recipe


def test_deprecated_keyword_still_works_but_warns():
    recipe = {"mass": {}}
    with pytest.warns(FutureWarning, match="xbudget_dict"):
        assert _resolve_recipe(None, recipe, "f") is recipe


def test_passing_both_is_an_error():
    with pytest.raises(TypeError, match="both"):
        _resolve_recipe({"a": 1}, {"b": 2}, "f")


def test_passing_neither_is_an_error():
    with pytest.raises(TypeError, match="missing required argument"):
        _resolve_recipe(None, None, "f")


def test_falsy_recipe_is_still_a_recipe():
    """Presence must be tested with `is None`, not truthiness: an empty recipe is
    a legitimate (if useless) argument, and silently rejecting it would report a
    missing argument for one that was passed."""
    with pytest.warns(FutureWarning):
        assert _resolve_recipe(None, {}, "f") == {}


def test_deprecated_attribute_reads_and_writes(single_tile_grid):
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    wmb = xwmb.WaterMassBudget(single_tile_grid, recipe, rho_ref=1.0)

    with pytest.warns(FutureWarning, match="full_xbudget_dict"):
        assert wmb.full_xbudget_dict is recipe
    replacement = {"mass": {}}
    with pytest.warns(FutureWarning, match="full_xbudget_dict"):
        wmb.full_xbudget_dict = replacement
    assert wmb.full_recipe is replacement


def test_deprecated_constructor_keyword(single_tile_grid):
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    with pytest.warns(FutureWarning, match="xbudget_dict"):
        wmb = xwmb.WaterMassBudget(
            single_tile_grid, xbudget_dict=recipe, rho_ref=1.0
        )
    assert wmb.full_recipe is recipe


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
