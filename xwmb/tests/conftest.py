import os

import numpy as np
import pytest
import xarray as xr
import xgcm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MOM6_FILE = os.path.join(DATA_DIR, "MOM6_global_example_sigma2_budgets_v0_0_6.nc")


def _construct_mom6_grid(ds):
    coords = {
        "X": {"center": "xh", "outer": "xq"},
        "Y": {"center": "yh", "outer": "yq"},
        "Z": {"center": "sigma2_l", "outer": "sigma2_i"},
    }
    padding = {"X": "periodic", "Y": "extend", "Z": "extend"}
    return xgcm.Grid(
        ds,
        coords=coords,
        metrics={("X", "Y"): "areacello"},
        padding=padding,
        autoparse_metadata=False,
    )


@pytest.fixture(scope="module")
def mom6_grid():
    """The coarsened, prebinned-sigma2 global MOM6 example grid (single-tile)."""
    if not os.path.exists(MOM6_FILE):
        pytest.skip(f"MOM6 example data not found at {MOM6_FILE}")
    ds = xr.open_dataset(MOM6_FILE, chunks=-1).fillna(0.0)
    return _construct_mom6_grid(ds)


@pytest.fixture(scope="module")
def mom6_xbudget():
    xbudget = pytest.importorskip("xbudget")
    return xbudget.load_preset_budget(model="MOM6")
