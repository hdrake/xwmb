"""Closing the budget, and refusing to name a residual that has not earned it."""

import numpy as np
import pytest
import xbudget

import xwmb
from xwmb.completeness import CompletenessReport, budget_completeness

from .synthetic import synthetic_recipe


def _run(grid, recipe):
    wmb = xwmb.WaterMassBudget(grid, recipe, rho_ref=1.0)
    wmt = wmb.mass_budget("tracer", bins=grid._ds.sigma_f).compute()
    return wmb, wmt


def test_closed_budget_names_its_residual(single_tile_grid):
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    wmb, wmt = _run(single_tile_grid, recipe)

    assert wmb.completeness.is_complete
    assert "spurious_numerical_mixing" in wmt
    np.testing.assert_allclose(
        wmt.spurious_numerical_mixing.values, wmt.residual.values
    )
    assert "xwmb_unaccounted_terms" not in wmt.residual.attrs


def test_a_declared_but_unsupplied_term_blocks_the_attribution(single_tile_grid):
    """The dataset supplies no `wfo`, so the surface mass source never materializes.

    Its absence would otherwise land silently in the residual, which would then be
    reported as spurious numerical mixing with the wrong magnitude.
    """
    recipe = synthetic_recipe()
    recipe["mass"]["rhs"]["sum"]["surface_exchange_flux"] = {
        "product": {"flux": "wfo", "area": "areacello"}
    }
    xbudget.collect_budgets(single_tile_grid, recipe)

    with pytest.warns(UserWarning, match="not closed"):
        wmb, wmt = _run(single_tile_grid, recipe)

    assert not wmb.completeness.is_complete
    assert "spurious_numerical_mixing" not in wmt
    assert "residual" in wmt
    unaccounted = wmt.residual.attrs["xwmb_unaccounted_terms"]
    assert "S (surface mass source)" in unaccounted
    assert "wfo" in unaccounted


def test_the_warning_names_the_input_to_go_and_find(single_tile_grid):
    recipe = synthetic_recipe()
    recipe["mass"]["rhs"]["sum"]["surface_exchange_flux"] = {
        "product": {"flux": "wfo", "area": "areacello"}
    }
    xbudget.collect_budgets(single_tile_grid, recipe)
    with pytest.warns(UserWarning) as record:
        _run(single_tile_grid, recipe)
    message = next(str(w.message) for w in record if "not closed" in str(w.message))
    assert "wfo" in message
    # Only the root cause: not every ancestor term that inherited the gap.
    assert message.count("wfo") == 1


def test_a_legitimately_zero_term_is_not_a_gap(single_tile_grid):
    """A full-domain region has no boundary for Psi to cross, and this recipe
    declares no surface mass exchange. Neither absence is a defect."""
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    wmb, _ = _run(single_tile_grid, recipe)
    assert set(wmb.completeness.zero_terms) == {
        "convergent_mass_transport",
        "mass_source",
    }
    assert wmb.completeness.absent_terms == []


def test_absent_terms_are_recorded_rather_than_aborting(single_tile_grid):
    """Terms xwmb treats as zero are named on the output, not left to be inferred."""
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    _, wmt = _run(single_tile_grid, recipe)
    assert "mass_source" in wmt.realized_transformation.attrs["xwmb_assumed_zero"]


def test_report_reduces_to_root_causes():
    report = CompletenessReport(
        missing_inputs={
            ("mass", "rhs"): ["surface_exchange_flux"],
            ("mass", "rhs", "surface_exchange_flux"): ["wfo"],
        }
    )
    assert list(report.root_causes()) == [("mass", "rhs", "surface_exchange_flux")]
    assert report.unaccounted() == [
        "mass/rhs/surface_exchange_flux (missing input(s): wfo)"
    ]
    assert not report.is_complete


def test_empty_report_is_complete():
    report = CompletenessReport()
    assert report.is_complete
    assert report.describe() == "the budget is fully accounted for"
