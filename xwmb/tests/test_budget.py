"""End-to-end budgets on each supported grid topology.

The synthetic tests run everywhere; the MOM6 ones skip until the example file has
been downloaded.

The central assertion for a region's boundary transport is the **discrete
divergence theorem**: the transport xwmb integrates along the traced region
boundary must equal, to round-off, the flux convergence summed over the region's
cells. That is the property the whole regional budget rests on, and it is exactly
what a topology bug breaks -- a boundary loop that misses a fold seam or a tile
seam still produces a perfectly plausible number.
"""

import numpy as np
import pytest
import xbudget

import xwmb

from .synthetic import mask_from_cells, synthetic_recipe


def _collect(grid):
    recipe = synthetic_recipe()
    xbudget.collect_budgets(grid, recipe)
    return recipe


def _layer_convergence(wmb):
    """xwmb's per-lambda-layer boundary convergence, summed over the region."""
    layer = wmb.grid._ds["convergent_mass_transport_layer"].compute()
    keep = [d for d in layer.dims if d.endswith("_l_target")]
    return layer.sum([d for d in layer.dims if d not in keep]).values


def _divergence_theorem(grid, cells):
    """The same quantity from the raw transports: minus the cells' divergence."""
    u = grid._ds.umo.values
    v = grid._ds.vmo.values
    entries = (
        [(f, j, i) for f, lst in cells.items() for j, i in lst]
        if isinstance(cells, dict)
        else [(None, j, i) for j, i in cells]
    )
    total = 0.0
    for face, j, i in entries:
        if face is None:
            total -= (u[:, j, i + 1] - u[:, j, i]) + (v[:, j + 1, i] - v[:, j, i])
        else:
            total -= (u[face, :, j, i + 1] - u[face, :, j, i]) + (
                v[face, :, j + 1, i] - v[face, :, j, i]
            )
    return total


# ---------------------------------------------------------------------------
# Single tile
# ---------------------------------------------------------------------------


def test_full_domain_budget_closes(single_tile_grid):
    """A full-domain budget has no boundary, so Psi vanishes and the budget closes."""
    recipe = _collect(single_tile_grid)
    wmb = xwmb.WaterMassBudget(single_tile_grid, recipe, rho_ref=1.0)
    wmt = wmb.mass_budget("tracer", bins=single_tile_grid._ds.sigma_f).compute()

    assert float(np.abs(wmt.convergent_mass_transport).max()) == 0.0
    assert wmb.completeness.is_complete
    assert "spurious_numerical_mixing" in wmt
    # The residual and the named mixing are the same numbers under two names.
    np.testing.assert_allclose(
        wmt.residual.values, wmt.spurious_numerical_mixing.values
    )
    for term in ("mass_tendency", "layer_mass", "realized_transformation"):
        assert np.isfinite(wmt[term]).all()


def test_recipe_accepted_as_a_keyword(single_tile_grid):
    recipe = _collect(single_tile_grid)
    wmb = xwmb.WaterMassBudget(single_tile_grid, recipe=recipe, rho_ref=1.0)
    assert wmb.full_recipe is recipe


# ---------------------------------------------------------------------------
# Bipolar north fold
# ---------------------------------------------------------------------------

#: A top-row cell and its across-fold mirror: adjacent only *through* the fold.
FOLD_CELLS = [(3, 1), (3, 4)]


def test_fold_straddling_region_is_one_boundary_loop(fold_grid):
    recipe = _collect(fold_grid)
    wmb = xwmb.WaterMassBudget(
        fold_grid, recipe, mask_from_cells(fold_grid, FOLD_CELLS), rho_ref=1.0
    )
    assert len(wmb.region.boundaries) == 1
    assert wmb.region.boundaries[0].f_c is None  # single tile: no face index
    assert not wmb.region.is_multitile


def test_fold_boundary_transport_obeys_divergence_theorem(fold_grid):
    recipe = _collect(fold_grid)
    wmb = xwmb.WaterMassBudget(
        fold_grid, recipe, mask_from_cells(fold_grid, FOLD_CELLS), rho_ref=1.0
    )
    wmb.mass_budget("tracer", along_section=True)
    np.testing.assert_allclose(
        _layer_convergence(wmb), _divergence_theorem(fold_grid, FOLD_CELLS), atol=1e-10
    )


def test_fold_transport_methods_agree(fold_grid):
    """The along-boundary and grid-cell-divergence methods must give the same Psi."""
    along = xwmb.WaterMassBudget(
        fold_grid,
        _collect(fold_grid),
        mask_from_cells(fold_grid, FOLD_CELLS),
        rho_ref=1.0,
    )
    along.mass_budget("tracer", along_section=True)

    diverg = xwmb.WaterMassBudget(
        fold_grid,
        _collect(fold_grid),
        mask_from_cells(fold_grid, FOLD_CELLS),
        rho_ref=1.0,
    )
    diverg.mass_budget("tracer", along_section=False)

    np.testing.assert_allclose(
        _layer_convergence(along), _layer_convergence(diverg), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Multi-tile (face_connections)
# ---------------------------------------------------------------------------

#: A region straddling the tile seam: the two rightmost columns of face 0 and the
#: two leftmost of face 1.
TILED_CELLS = {
    0: [(2, 4), (2, 5), (3, 4), (3, 5)],
    1: [(2, 0), (2, 1), (3, 0), (3, 1)],
}


def test_tiled_region_stitches_across_the_seam(tiled_grid):
    recipe = _collect(tiled_grid)
    wmb = xwmb.WaterMassBudget(
        tiled_grid, recipe, mask_from_cells(tiled_grid, TILED_CELLS), rho_ref=1.0
    )
    assert len(wmb.region.boundaries) == 1
    assert wmb.region.is_multitile
    assert set(np.asarray(wmb.region.boundaries[0].f_c).tolist()) == {0, 1}


def test_tiled_boundary_transport_obeys_divergence_theorem(tiled_grid):
    recipe = _collect(tiled_grid)
    wmb = xwmb.WaterMassBudget(
        tiled_grid, recipe, mask_from_cells(tiled_grid, TILED_CELLS), rho_ref=1.0
    )
    wmb.mass_budget("tracer", along_section=True)
    np.testing.assert_allclose(
        _layer_convergence(wmb),
        _divergence_theorem(tiled_grid, TILED_CELLS),
        atol=1e-10,
    )


def test_tiled_divergence_method_refuses_rather_than_guesses(tiled_grid):
    """The grid-cell divergence method is single-tile only, and must say so."""
    recipe = _collect(tiled_grid)
    wmb = xwmb.WaterMassBudget(
        tiled_grid, recipe, mask_from_cells(tiled_grid, TILED_CELLS), rho_ref=1.0
    )
    with pytest.raises(NotImplementedError, match="along_section=True"):
        wmb.mass_budget("tracer", along_section=False)


# ---------------------------------------------------------------------------
# Real model output
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def collected_mom6(mom6_grid, mom6_recipe):
    xbudget.collect_budgets(mom6_grid, mom6_recipe)
    return mom6_grid, mom6_recipe


def test_mom6_global_budget_runs_and_is_finite(collected_mom6):
    grid, recipe = collected_mom6
    wmb = xwmb.WaterMassBudget(grid, recipe)
    wmt = wmb.mass_budget("sigma2", greater_than=True).squeeze().load()
    assert float(np.abs(wmt.convergent_mass_transport).max()) == 0.0
    for term in ["mass_tendency", "boundary_fluxes", "residual"]:
        assert np.isfinite(wmt[term]).all()


def test_mom6_regional_transport_methods_agree(collected_mom6):
    grid, recipe = collected_mom6
    mask = grid._ds.geolat < -30.0

    along = (
        xwmb.WaterMassBudget(grid, recipe, mask)
        .mass_budget("sigma2", greater_than=True, along_section=True)
        .squeeze()
        .load()
    )
    diverg = (
        xwmb.WaterMassBudget(grid, recipe, mask)
        .mass_budget("sigma2", greater_than=True, along_section=False)
        .squeeze()
        .load()
    )
    a = along.convergent_mass_transport
    d = diverg.convergent_mass_transport.reindex_like(a)
    scale = float(np.abs(a).max())
    assert scale > 0.0
    np.testing.assert_allclose(a.values, d.values, atol=1e-3 * scale)


def test_mom6_greater_than_tuple_region(collected_mom6):
    grid, recipe = collected_mom6
    lons = np.array([-70.0, -40.0, 30.0, 5.0])
    lats = np.array([50.0, 75.0, 60.0, 44.0])
    wmt = (
        xwmb.WaterMassBudget(grid, recipe, (lons, lats))
        .mass_budget("sigma2", greater_than=True)
        .squeeze()
        .load()
    )
    assert np.isfinite(wmt.convergent_mass_transport).all()
