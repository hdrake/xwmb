"""Pytest fixtures, wrapping the builders in :mod:`xwmb.tests.synthetic`."""

import os

import pytest
import xarray as xr

from .synthetic import (
    MOM6_FILE,
    _construct_mom6_grid,
    make_fold_grid,
    make_single_tile_grid,
    make_tiled_grid,
)


@pytest.fixture
def single_tile_grid():
    """A plain single-tile grid with no exotic topology."""
    return make_single_tile_grid()


@pytest.fixture
def fold_grid():
    """A single-tile grid with a bipolar (tripolar) north fold, corner pivot."""
    return make_fold_grid()


@pytest.fixture
def tiled_grid():
    """A two-tile grid joined by ``face_connections``."""
    return make_tiled_grid()


@pytest.fixture(scope="module")
def mom6_grid():
    """The coarsened, prebinned-sigma2 global MOM6 example grid (single-tile)."""
    if not os.path.exists(MOM6_FILE):
        pytest.skip(f"MOM6 example data not found at {MOM6_FILE}")
    ds = xr.open_dataset(MOM6_FILE, chunks=-1).fillna(0.0)
    # The published file ships `areacello` unlabelled. xbudget multiplies it into
    # every term it materializes and infers units from its operands, so without
    # this the entire budget comes back undescribed.
    ds["areacello"].attrs.setdefault("units", "m2")
    return _construct_mom6_grid(ds)


@pytest.fixture(scope="module")
def mom6_recipe():
    xbudget = pytest.importorskip("xbudget")
    return xbudget.load_preset_budget(model="MOM6")
