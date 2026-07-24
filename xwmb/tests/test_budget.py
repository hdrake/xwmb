"""Single-tile end-to-end budget tests on the coarsened MOM6 example."""

import numpy as np
import pytest

import xwmb


@pytest.fixture(scope="module")
def collected(mom6_grid, mom6_xbudget):
    import xbudget

    xbudget.collect_budgets(mom6_grid, mom6_xbudget)
    return mom6_grid, mom6_xbudget


def test_global_budget_runs_and_is_finite(collected):
    grid, xbudget_dict = collected
    wmb = xwmb.WaterMassBudget(grid, xbudget_dict)
    wmt = wmb.mass_budget("sigma2", greater_than=True).squeeze().load()
    # The full-domain net boundary transport must vanish.
    assert float(np.abs(wmt.convergent_mass_transport).max()) == 0.0
    for term in ["mass_tendency", "boundary_fluxes", "spurious_numerical_mixing"]:
        assert np.isfinite(wmt[term]).all()


def test_regional_transport_methods_agree(collected):
    """The along-section and grid-cell divergence transports must agree."""
    grid, xbudget_dict = collected
    mask = grid._ds.geolat < -30.0

    along = (
        xwmb.WaterMassBudget(grid, xbudget_dict, mask)
        .mass_budget("sigma2", greater_than=True, along_section=True)
        .squeeze()
        .load()
    )
    diverg = (
        xwmb.WaterMassBudget(grid, xbudget_dict, mask)
        .mass_budget("sigma2", greater_than=True, along_section=False)
        .squeeze()
        .load()
    )
    a = along.convergent_mass_transport
    d = diverg.convergent_mass_transport.reindex_like(a)
    scale = float(np.abs(a).max())
    assert scale > 0.0
    # Agreement to a small fraction of the peak overturning.
    np.testing.assert_allclose(a.values, d.values, atol=1e-3 * scale)


def test_greater_than_tuple_region(collected):
    grid, xbudget_dict = collected
    lons = np.array([-70.0, -40.0, 30.0, 5.0])
    lats = np.array([50.0, 75.0, 60.0, 44.0])
    wmt = (
        xwmb.WaterMassBudget(grid, xbudget_dict, (lons, lats))
        .mass_budget("sigma2", greater_than=True)
        .squeeze()
        .load()
    )
    assert np.isfinite(wmt.convergent_mass_transport).all()
