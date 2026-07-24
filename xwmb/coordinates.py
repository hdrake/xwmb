"""Target-coordinate (lambda-bin) setup and cumulative-in-lambda accumulation.

A water mass budget is expressed as a function of a vertical tracer coordinate
``lambda`` (e.g. ``sigma2``). Extensive terms (convergent transport, mass source,
mass) are *cumulatively integrated* in lambda so that, for a water mass defined by
``lambda >= lambda*`` (``greater_than=True``), the value at ``lambda*`` is the
integral over all denser layers.

xgcm >= 0.10 provides ``grid.cumsum(..., reverse=True)``, which performs that
high-to-low accumulation (and the associated grid-position bookkeeping) internally.
This replaces the four hand-rolled ``isel[::-1]`` / ``cumsum`` / ``isel[::-1]``
blocks in the previous monolithic implementation.
"""

import warnings

import numpy as np
import xarray as xr
from xgcm import Grid
from xwmt.wm import add_gridcoords

__all__ = [
    "lambda_grid",
    "horizontal_grid",
    "vertical_grid",
    "resolve_target_coords",
    "add_bins_gridcoords",
    "add_default_gridcoords",
    "accumulate_in_lambda",
    "interp_to_center",
    "interp_to_interfaces",
    "transform_to_lambda",
]


def horizontal_grid(grid):
    """A view of ``grid`` registering only the horizontal (X, Y) axes.

    sectionate/regionate build neighbor maps by padding corner-index arrays over
    *every* registered axis; a vertical (Z) axis those 2-D arrays don't span makes
    xgcm's ``pad`` raise. All sectionate/regionate example grids therefore register
    only X and Y — this reconstructs that horizontal-only grid (sharing the same
    ``_ds`` and any ``face_connections``) from a full budget grid.
    """
    coords = {ax: dict(grid.axes[ax].coords) for ax in ("X", "Y") if ax in grid.axes}
    padding = {ax: grid.axes[ax].padding for ax in coords}
    face_connections = getattr(grid, "_face_connections", None)
    extra = {"face_connections": face_connections} if face_connections else {}
    return Grid(
        grid._ds,
        coords=coords,
        padding=padding,
        autoparse_metadata=False,
        **extra,
    )


def lambda_grid(grid, target_coords):
    """A one-axis ``xgcm.Grid`` over the lambda target coordinate.

    Used for cumulative sums and interpolations along the target tracer axis.
    """
    return Grid(
        grid._ds,
        coords={"lam": target_coords},
        padding={"lam": "extend"},
        autoparse_metadata=False,
    )


def vertical_grid(grid, axis="Z"):
    """A view of ``grid`` with only the vertical ``axis`` and no ``face_connections``.

    Used to pad/interpolate *section* data (whose tile dimension has been collapsed
    along the section) vertically: on a ``face_connections`` grid, xgcm routes every
    pad through the face-connection path, which needs a face dimension the section
    data no longer has.
    """
    return Grid(
        grid._ds,
        coords={axis: dict(grid.axes[axis].coords)},
        padding={axis: grid.axes[axis].padding},
        autoparse_metadata=False,
    )


def resolve_target_coords(
    grid, lambda_var, lambda_name=None, bins=None, default_bins=None
):
    """Ensure ``grid`` carries a ``Z_target`` axis for the lambda target bins.

    Returns ``(grid, target_coords)`` where ``target_coords`` is
    ``{"center": f"{lambda_var}_l_target", "outer": f"{lambda_var}_i_target"}``.

    The target coordinates may be given explicitly as ``bins`` (a 1D array of bin
    edges), supplied by the caller as coordinates already present in ``grid._ds``,
    or derived from the pre-existing lambda coordinates. ``default_bins`` is the
    deprecated spelling of ``bins``.
    """
    target_coords = {
        "center": f"{lambda_var}_l_target",
        "outer": f"{lambda_var}_i_target",
    }

    if default_bins is not None:
        warnings.warn(
            "`default_bins` is deprecated and will be removed in a future version. "
            "Use `bins` instead. "
            "Note: The behavior has changed - `bins=None` is now the default, "
            "and you should pass an array to `bins` to specify the edges of custom bins.",
            DeprecationWarning,
            stacklevel=3,
        )
        if default_bins is True:
            if "Z_target" in grid.axes:
                raise ValueError(
                    "Cannot pass `default_bins=True` when `Z_target` in "
                    "`WaterMassBudget.grid.axes`."
                )
            return add_default_gridcoords(grid, lambda_var, lambda_name), target_coords
        if default_bins is False:
            bins = None
        elif len(default_bins) == 3 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in default_bins
        ):
            bins = np.arange(*default_bins)
        else:
            raise TypeError(
                f"Boolean or list of 3 numbers expected, got "
                f"{type(default_bins).__name__}"
            )

    if isinstance(bins, xr.DataArray):
        bins = bins.values
    if bins is not None:
        if not isinstance(bins, np.ndarray):
            raise TypeError(f"None or array expected, got {type(bins).__name__}")
        return add_bins_gridcoords(grid, lambda_var, bins), target_coords

    if "Z_target" in grid.axes:
        return grid, target_coords

    avail_target = [c in grid._ds for c in target_coords.values()]
    avail_lambda = [
        c.replace("_target", "") in grid._ds for c in target_coords.values()
    ]
    if not all(avail_lambda):
        raise ValueError(
            f"To specify the target grid, either pass a 1D array of bin edges to "
            f"`bins` or include {target_coords['center']} and "
            f"{target_coords['outer']} in `WaterMassBudget.grid._ds`."
        )

    if not all(avail_target):
        grid._ds = grid._ds.assign_coords(
            {
                target_coords["center"]: xr.DataArray(
                    grid._ds[target_coords["center"].replace("_target", "")].values,
                    dims=(target_coords["center"],),
                ),
                target_coords["outer"]: xr.DataArray(
                    grid._ds[target_coords["outer"].replace("_target", "")].values,
                    dims=(target_coords["outer"],),
                ),
            }
        )
    grid = add_gridcoords(grid, {"Z_target": target_coords}, {"Z_target": "extend"})
    return grid, target_coords


def add_bins_gridcoords(grid, lambda_var, bin_edges):
    """Register a ``Z_target`` axis from an explicit 1D array of lambda bin edges."""
    grid._ds = grid._ds.assign_coords(
        {
            f"{lambda_var}_l_target": 0.5 * (bin_edges[1:] + bin_edges[:-1]),
            f"{lambda_var}_i_target": bin_edges,
        }
    )
    return add_gridcoords(
        grid,
        {
            "Z_target": {
                "outer": f"{lambda_var}_i_target",
                "center": f"{lambda_var}_l_target",
            }
        },
        {"Z_target": "extend"},
    )


def add_default_gridcoords(grid, lambda_var, lambda_name):
    """Assign a default, finely-spaced target grid for ``lambda_name`` and register it."""
    if lambda_name is not None and "sigma" in lambda_name:
        bin_edges = np.arange(0.0, 50.0 + 0.05, 0.05)
    elif lambda_name == "heat":
        bin_edges = np.arange(-4.0, 40.0 + 0.05, 0.05)
    elif lambda_name == "salt":
        bin_edges = np.arange(-1.0, 40.0 + 0.05, 0.05)
    else:
        raise ValueError(
            f"No default target bins are defined for lambda_name={lambda_name!r}; "
            f"provide `{lambda_var}_l_target`/`{lambda_var}_i_target` explicitly."
        )
    return add_bins_gridcoords(grid, lambda_var, bin_edges)


def accumulate_in_lambda(grid, da, target_coords, greater_than=False, name=None):
    """Cumulatively integrate a per-layer density (on target *centers*) in lambda.

    Returns the cumulative integral on the target *outer* (interface) coordinate.
    For ``greater_than=True`` the accumulation runs from dense to light
    (``reverse=True``): the value at each interface is the integral over all denser
    layers.
    """
    accumulated = lambda_grid(grid, target_coords).cumsum(
        da, "lam", padding="fill", fill_value=0.0, reverse=greater_than
    )
    accumulated = accumulated.chunk({target_coords["outer"]: -1}).assign_coords(
        {target_coords["outer"]: grid._ds[target_coords["outer"]]}
    )
    if name is not None:
        accumulated = accumulated.rename(name)
    return accumulated


def interp_to_center(grid, da, target_coords):
    """Interpolate a lambda term from target *outer* (interfaces) to *centers*."""
    return lambda_grid(grid, target_coords).interp(da, "lam").assign_coords(
        {target_coords["center"]: grid._ds[target_coords["center"]]}
    )


def interp_to_interfaces(grid, da, axis="Z"):
    """Interpolate a tracer from ``axis`` centers to interfaces (outer position).

    Vertical interpolation never needs horizontal face padding, so on a
    ``face_connections`` grid we operate through a vertical-only view (xgcm otherwise
    routes the Z pad through its face-connection path, which mishandles a time axis /
    tile-collapsed section data).
    """
    outer = grid.axes[axis].coords["outer"]
    if getattr(grid, "_face_connections", None) is not None:
        grid = vertical_grid(grid, axis)
    return grid.interp(da, axis, padding="extend").chunk({outer: -1})


def transform_to_lambda(grid, da, target_coords, target_data, axis="Z"):
    """Conservatively remap a per-``axis`` density into lambda layers (target centers).

    ``da`` is a vertically-extensive quantity on ``axis`` centers; ``target_data`` is
    the lambda value at ``axis`` interfaces. The result lives on the target *center*
    (layer) coordinate.
    """
    return (
        grid.transform(
            da.fillna(0.0),
            axis,
            target=grid._ds[target_coords["outer"]],
            target_data=target_data,
            method="conservative",
        )
        .rename({target_coords["outer"]: target_coords["center"]})
        .assign_coords({target_coords["center"]: grid._ds[target_coords["center"]]})
    )
