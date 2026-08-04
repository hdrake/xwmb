"""Metadata on the variables xwmb derives."""

import numpy as np
import pytest
import xarray as xr
import xbudget

import xwmb
from xwmb import attrs as A

from .synthetic import synthetic_recipe


@pytest.fixture
def budget(single_tile_grid):
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    wmb = xwmb.WaterMassBudget(single_tile_grid, recipe, rho_ref=1.0)
    return wmb.mass_budget("tracer", bins=single_tile_grid._ds.sigma_f).compute()


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------


def test_units_are_composed_not_copied():
    assert A.product_units("kg m-3", "m", "m2") == "kg"
    assert A.quotient_units("kg", "s") == "kg s-1"


def test_unknown_operand_makes_the_result_unknown():
    assert A.product_units("kg m-3", None) is None
    assert A.quotient_units(None, "s") is None


def test_summing_disagreeing_units_warns_and_yields_nothing():
    with pytest.warns(UserWarning, match="different units"):
        assert A.common_units(["kg s-1", "W"], term="boundary_fluxes") is None


def test_annotate_does_not_let_a_source_description_leak_through():
    """A derived field is a *different* quantity, so nothing carries over."""
    da = xr.DataArray(
        np.arange(3.0),
        attrs={"long_name": "sea water potential temperature", "units": "degC",
               "valid_range": [-2.0, 40.0]},
    )
    A.annotate(da, "layer_mass", units="kg")
    assert da.attrs["units"] == "kg"
    assert da.attrs["long_name"] == A.LONG_NAMES["layer_mass"]
    assert "valid_range" not in da.attrs


def test_annotate_omits_unknown_units_rather_than_guessing():
    da = xr.DataArray(np.arange(3.0))
    A.annotate(da, "mass_source", units=None)
    assert "units" not in da.attrs
    assert da.attrs["xwmb_units_source"] == "unknown"


def test_annotate_never_writes_the_string_none():
    da = xr.DataArray(np.arange(3.0))
    A.annotate(da, "residual")
    assert "None" not in da.attrs.values()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

EXPECTED_UNITS = {
    "layer_mass": "kg",
    "mass_bounds": "kg",
    "mass_tendency": "kg s-1",
    "dt": "s",
    "convergent_mass_transport": "kg s-1",
    "realized_transformation": "kg s-1",
    "residual": "kg s-1",
    "spurious_numerical_mixing": "kg s-1",
}


@pytest.mark.parametrize("name,units", sorted(EXPECTED_UNITS.items()))
def test_derived_variables_carry_derived_units(budget, name, units):
    assert budget[name].attrs["units"] == units


@pytest.mark.parametrize("name", sorted(EXPECTED_UNITS))
def test_derived_variables_are_described(budget, name):
    attrs = budget[name].attrs
    assert attrs["long_name"]
    assert attrs["xwmb_term"] == name
    assert attrs["xwmb_version"] == xwmb.__version__


def test_unlabelled_area_metric_is_reported_not_silently_absorbed(single_tile_grid):
    """xbudget infers a term's units from its operands, and multiplies the cell
    area into nearly all of them, so an unlabelled area costs the units of the
    whole budget rather than one attribute."""
    del single_tile_grid._ds["areacello"].attrs["units"]
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    with pytest.warns(UserWarning, match="areacello"):
        xwmb.WaterMassBudget(single_tile_grid, recipe, rho_ref=1.0)


def test_greater_than_sign_flip_keeps_the_upstream_metadata(single_tile_grid):
    """Negating a rate does not change what it is; xwmt's units must survive."""
    recipe = synthetic_recipe()
    xbudget.collect_budgets(single_tile_grid, recipe)
    wmb = xwmb.WaterMassBudget(single_tile_grid, recipe, rho_ref=1.0)
    wmt = wmb.mass_budget(
        "tracer", bins=single_tile_grid._ds.sigma_f, greater_than=True
    )
    assert wmt.material_transformation.attrs.get("units") == "kg s-1"
