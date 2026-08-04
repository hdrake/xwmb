"""Mass terms of the budget: layer mass, surface mass source, and mass tendency.

* ``layer_mass_term`` — the (non-cumulative) mass per lambda layer in the region.
* ``mass_source_term`` — mass added directly to the water mass by the surface mass
  (freshwater) flux, cumulatively integrated in lambda.
* ``mass_bounds_term`` — snapshots of the cumulative water-mass mass at the time
  bounds, differenced by ``mass_tendency`` to give the storage term ∂ₜM.

All horizontal integrals reduce over ``wmb._horizontal_dims`` (which includes the
face dimension on multi-tile grids).
"""

import warnings

import numpy as np
from xwmt import units as _units

from . import attrs as _attrs
from .coordinates import (
    accumulate_in_lambda,
    interp_to_center,
    interp_to_interfaces,
    transform_to_lambda,
)

__all__ = [
    "MASS_SOURCE_PATH",
    "mass_source_varname",
    "layer_mass_term",
    "mass_source_term",
    "mass_bounds_term",
    "mass_tendency",
]

#: Recipe path to the surface mass (freshwater) exchange flux.
MASS_SOURCE_PATH = ("mass", "rhs", "surface_exchange_flux")


def mass_source_varname(query):
    """The dataset variable holding the surface mass flux, or ``None``.

    Resolved through ``xbudget.BudgetQuery`` rather than hardcoded. The name this
    used to be spelled with, ``"mass_rhs_sum_surface_exchange_flux"``, was the
    xbudget 0.6.x one; 0.8.0 dropped the operator infixes from derived-variable
    names, so the hardcoded string silently matched nothing and the mass source
    was quietly dropped from every budget.

    Returns ``None`` both when the recipe has no such term and when it has one
    that did not materialize -- in either case there is no mass source to add.
    """
    try:
        return query.var(MASS_SOURCE_PATH)
    except KeyError:
        return None


def _horizontal_dims(wmb, da):
    return [d for d in wmb._horizontal_dims if d in da.dims]


def _cell_methods(wmb, integrate):
    """CF ``cell_methods`` for a term integrated (or not) over the region."""
    if not integrate:
        return None
    return " ".join(f"{d}: sum" for d in wmb._horizontal_dims)


def _area_units(wmb):
    """Units of the horizontal area metric, or ``None`` if it is unlabelled."""
    try:
        area = wmb.grid.get_metric(wmb.grid._ds[wmb.h_name], ("X", "Y"))
    except Exception:  # pragma: no cover - no metric registered
        return None
    return _attrs.units_of(area)


def _mass_density_units(wmb):
    """Units of ``rho_ref * h``: a mass per unit area."""
    return _attrs.product_units(
        _units.RHO_REF_UNITS, _attrs.units_of(wmb.grid._ds.get(wmb.h_name))
    )


def _lambda_interfaces(wmb, lambda_var, prebinned, axis="Z", suffix=""):
    """Lambda value at ``axis`` interfaces (or the prebinned interface coordinate)."""
    if prebinned:
        return wmb.grid._ds[f"{lambda_var}_i"]
    interfaces = interp_to_interfaces(
        wmb.grid, wmb.grid._ds[f"{lambda_var}{suffix}"], axis
    ).rename(f"{lambda_var}_i{suffix}")
    wmb.grid._ds[f"{lambda_var}_i{suffix}"] = interfaces
    return interfaces


def layer_mass_term(wmb, lambda_name, target_coords, region, *, integrate=True, prebinned=False):
    """Mass per lambda layer within the region (not cumulatively integrated)."""
    grid = wmb.grid
    lambda_var = wmb.get_lambda_var(lambda_name)
    target_data = _lambda_interfaces(wmb, lambda_var, prebinned, "Z")

    mass_density = (
        transform_to_lambda(
            grid, wmb.rho_ref * grid._ds[wmb.h_name], target_coords, target_data
        )
        * region.mask
    )
    density_units = _mass_density_units(wmb)
    _attrs.annotate(
        mass_density,
        "mass_density",
        units=density_units,
        lambda_name=lambda_name,
        lambda_var=lambda_var,
        sources={wmb.h_name: grid._ds.get(wmb.h_name)},
    )
    grid._ds["mass_density"] = mass_density

    layer_mass = mass_density * grid.get_metric(mass_density, ("X", "Y"))
    if integrate:
        layer_mass = layer_mass.sum(_horizontal_dims(wmb, layer_mass))
    return _attrs.annotate(
        layer_mass,
        "layer_mass",
        units=_attrs.product_units(density_units, _area_units(wmb)),
        lambda_name=lambda_name,
        lambda_var=lambda_var,
        cell_methods=_cell_methods(wmb, integrate),
        sources={wmb.h_name: grid._ds.get(wmb.h_name)},
    )


def mass_source_term(
    wmb,
    lambda_name,
    target_coords,
    region,
    *,
    greater_than=False,
    integrate=True,
    prebinned=False,
    mass_source_var=None,
):
    """Surface mass (freshwater) source, cumulatively integrated in lambda.

    ``mass_source_var`` names the surface mass-flux density variable in
    ``grid._ds``; by default it is resolved from the recipe. Returns ``None`` when
    the recipe declares no such term, or when it declares one that the dataset did
    not supply the inputs for.
    """
    grid = wmb.grid
    mass_source_var = mass_source_var or mass_source_varname(wmb.query)
    if mass_source_var is None or mass_source_var not in grid._ds:
        return None
    lambda_var = wmb.get_lambda_var(lambda_name)
    suffix = "greater_than" if greater_than else "less_than"
    target_data = _lambda_interfaces(wmb, lambda_var, prebinned, "Z")

    density = (
        transform_to_lambda(
            grid, grid._ds[mass_source_var], target_coords, target_data
        )
        * region.mask
    )
    grid._ds["mass_source_density"] = density

    accumulated = accumulate_in_lambda(
        grid, density, target_coords, greater_than, name=f"mass_source_density_{suffix}"
    )
    grid._ds[accumulated.name] = accumulated

    if integrate:
        accumulated = accumulated.sum(_horizontal_dims(wmb, accumulated))
    grid._ds[f"mass_source_{suffix}"] = accumulated

    # The remap, the cumulative sum in lambda, the regional sum and the interp to
    # layer centres are all unit-preserving, so the surface flux's own units carry
    # straight through. xbudget stamps them when it materializes the term.
    source = grid._ds[mass_source_var]
    return _attrs.annotate(
        interp_to_center(grid, accumulated, target_coords),
        "mass_source",
        units=_attrs.units_of(source),
        units_source="source" if _attrs.units_of(source) else None,
        lambda_name=lambda_name,
        lambda_var=lambda_var,
        cell_methods=_cell_methods(wmb, integrate),
        sources={mass_source_var: source},
    )


def mass_bounds_term(
    wmb, lambda_name, target_coords, region, *, greater_than=False, integrate=True, prebinned=False
):
    """Snapshots of the cumulative water-mass mass at the time bounds.

    Returns ``None`` unless the dataset carries a ``time_bounds`` dimension.
    """
    grid = wmb.grid
    if "time_bounds" not in grid._ds.dims:
        return None
    lambda_var = wmb.get_lambda_var(lambda_name)
    ax_bounds = wmb.ax_bounds
    suffix = "greater_than" if greater_than else "less_than"

    if prebinned:
        target_data = grid._ds[f"{lambda_var}_i"]
    else:
        target_data = _lambda_interfaces(
            wmb, lambda_var, prebinned, ax_bounds, suffix="_bounds"
        )

    density = (
        transform_to_lambda(
            grid,
            wmb.rho_ref * grid._ds[f"{wmb.h_name}_bounds"],
            target_coords,
            target_data,
            axis=ax_bounds,
        )
        * region.mask
    )
    grid._ds["mass_density_bounds"] = density

    accumulated = accumulate_in_lambda(
        grid, density, target_coords, greater_than, name=f"mass_density_bounds_{suffix}"
    )
    grid._ds[accumulated.name] = accumulated

    mass_bounds_density = accumulated * grid.get_metric(accumulated, ("X", "Y"))
    grid._ds[f"mass_bounds_{suffix}"] = mass_bounds_density

    mass_bounds = interp_to_center(grid, mass_bounds_density, target_coords)
    if integrate:
        mass_bounds = mass_bounds.sum(_horizontal_dims(wmb, mass_bounds))
    return _attrs.annotate(
        mass_bounds,
        "mass_bounds",
        units=_attrs.product_units(_mass_density_units(wmb), _area_units(wmb)),
        lambda_name=lambda_name,
        lambda_var=lambda_var,
        cell_methods=_cell_methods(wmb, integrate),
        sources={f"{wmb.h_name}_bounds": grid._ds.get(f"{wmb.h_name}_bounds")},
    )


def mass_tendency(ds):
    """Time-mean mass tendency by finite-differencing the water-mass snapshots."""
    if not all(v in ds for v in ["time_bounds", "mass_bounds"]):
        raise ValueError("Needs both `time_bounds` and `mass_bounds` variables")
    dt = ds.time_bounds.diff("time_bounds")
    # Convert timedelta to float seconds.
    if str(dt.dtype).startswith("timedelta64"):
        dt = dt / np.timedelta64(1, "s")
    elif dt.dtype == "<m8[ns]":
        dt = dt.astype("float") * 1.0e-9
    if ds.time_bounds.size == ds.time.size:
        time_target = ds.time[1:]
        warnings.warn(
            "There are as many time bounds as time steps, so the mass snapshots "
            "bracket only the intervals after the first: the first value of "
            "`mass_tendency` may be NaN.",
            stacklevel=2,
        )
    elif ds.time_bounds.size == (ds.time.size + 1):
        time_target = ds.time
    else:
        raise ValueError("time_bounds inconsistent with time")

    mass_units = _attrs.units_of(ds.mass_bounds)
    tendency = (
        (ds.mass_bounds.diff("time_bounds") / dt)
        .rename({"time_bounds": "time"})
        .assign_coords({"time": time_target})
    )
    ds["mass_tendency"] = _attrs.annotate(
        tendency,
        "mass_tendency",
        units=_attrs.quotient_units(mass_units, "s"),
        lambda_name=ds.attrs.get("xwmt_lambda"),
        cell_methods="time: difference",
        sources={"mass_bounds": ds.mass_bounds},
    )
    ds["dt"] = _attrs.annotate(
        dt.rename({"time_bounds": "time"}).assign_coords({"time": time_target}),
        "dt",
        units="s",
        units_source="derived",
    )
