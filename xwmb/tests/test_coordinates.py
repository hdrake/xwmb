"""Unit tests for the lambda-coordinate helpers (no data download)."""

import numpy as np
import xarray as xr
import xgcm

from xwmb.coordinates import accumulate_in_lambda, horizontal_grid


def _lambda_grid(n=12):
    ds = xr.Dataset(
        coords={
            "lam_l_target": ("lam_l_target", np.linspace(20.0, 38.0, n)),
            "lam_i_target": ("lam_i_target", np.linspace(19.5, 38.5, n + 1)),
        }
    )
    ds["dens"] = ("lam_l_target", np.arange(1.0, n + 1.0))
    tc = {"center": "lam_l_target", "outer": "lam_i_target"}
    grid = xgcm.Grid(
        ds, coords={"lam": tc}, padding={"lam": "extend"}, autoparse_metadata=False
    )
    return grid, ds, tc


def test_accumulate_forward_matches_numpy_cumsum():
    grid, ds, tc = _lambda_grid()
    out = accumulate_in_lambda(grid, ds["dens"], tc, greater_than=False).values
    expected = np.concatenate([[0.0], np.cumsum(ds["dens"].values)])
    np.testing.assert_allclose(out, expected)


def test_accumulate_reverse_matches_numpy_reverse_cumsum():
    """`reverse=True` must reproduce the old isel-reverse / cumsum / isel-reverse dance."""
    grid, ds, tc = _lambda_grid()
    out = accumulate_in_lambda(grid, ds["dens"], tc, greater_than=True).values
    x = ds["dens"].values
    expected = np.concatenate([np.cumsum(x[::-1])[::-1], [0.0]])
    np.testing.assert_allclose(out, expected)


def test_horizontal_grid_drops_vertical_axis():
    ds = xr.Dataset(
        coords={
            "xh": ("xh", np.arange(4.0)),
            "xq": ("xq", np.arange(5.0)),
            "yh": ("yh", np.arange(3.0)),
            "yq": ("yq", np.arange(4.0)),
            "zl": ("zl", np.arange(2.0)),
            "zi": ("zi", np.arange(3.0)),
        }
    )
    grid = xgcm.Grid(
        ds,
        coords={
            "X": {"center": "xh", "outer": "xq"},
            "Y": {"center": "yh", "outer": "yq"},
            "Z": {"center": "zl", "outer": "zi"},
        },
        padding={"X": "periodic", "Y": "extend", "Z": "extend"},
        autoparse_metadata=False,
    )
    hgrid = horizontal_grid(grid)
    assert set(hgrid.axes) == {"X", "Y"}
    assert hgrid.axes["X"].padding == "periodic"
    assert hgrid.axes["Y"].padding == "extend"
