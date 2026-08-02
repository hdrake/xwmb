"""Convergent mass transport across the region boundary (the diascalar overturning).

Two equivalent methods for the net transport into the region, per lambda layer:

* **along-section** (``along_section=True``): integrate the signed normal transport
  along the region boundary with :func:`sectionate.convergent_transport`, threading
  the per-corner face index ``f_c`` so it is correct on multi-tile
  ``face_connections`` grids. This is the recommended path and preserves the
  along-boundary (streamfunction) structure.

* **grid-cell divergence** (``along_section=False``): accumulate the per-cell
  transport divergence inside the mask. Single-tile only; on multi-tile grids it
  raises and directs the caller to ``along_section=True``.

Both produce a per-lambda-layer convergence density, which is then cumulatively
integrated in lambda (dense-to-light for ``greater_than``) by
:func:`xwmb.coordinates.accumulate_in_lambda`.
"""

import xarray as xr
import sectionate

from . import attrs as _attrs
from .coordinates import (
    accumulate_in_lambda,
    horizontal_grid,
    interp_to_center,
    interp_to_interfaces,
    transform_to_lambda,
    vertical_grid,
)

__all__ = ["transport_varnames", "convergent_transport_term"]


#: Recipe path to the differenced face transport, per horizontal direction.
_TRANSPORT_PATHS = {
    "utr": (
        "mass",
        "rhs",
        "advection",
        "lateral",
        "zonal_convergence",
        "zonal_divergence",
    ),
    "vtr": (
        "mass",
        "rhs",
        "advection",
        "lateral",
        "meridional_convergence",
        "meridional_divergence",
    ),
}


def transport_varnames(query):
    """Extract the (zonal, meridional) mass-transport variable names from the recipe.

    Returns ``{"utr": ..., "vtr": ...}`` or ``None`` when the recipe carries no
    lateral advective transport terms.

    Resolution goes through ``xbudget.BudgetQuery`` rather than walking the recipe
    dict by hand. Hand-walking is no longer safe under xbudget 0.8.0: recipes no
    longer carry ``var: null`` placeholders to read, and a string operand may be a
    reference into the recipe's top-level ``constants:`` table rather than the name
    of a dataset variable.
    """
    names = {}
    for short, path in _TRANSPORT_PATHS.items():
        try:
            operands = query.get_vars(path)
        except KeyError:
            return None
        difference = operands.get("difference")
        if not difference:
            return None
        names[short] = difference[0]
    return names


def convergent_transport_term(
    wmb,
    lambda_name,
    target_coords,
    region,
    *,
    greater_than=False,
    integrate=True,
    along_section=False,
    prebinned=False,
    utr=None,
    vtr=None,
):
    """Compute the convergent mass transport term of the budget.

    ``utr``/``vtr`` name the (zonal, meridional) face mass-transport variables in
    ``grid._ds``. If omitted they are resolved from the recipe (which must follow
    the MOM6 convention's ``zonal_convergence``/``meridional_convergence`` term
    names); other conventions should pass them explicitly. Stores per-layer
    intermediates on ``wmb.grid._ds`` and returns the term (integrated over the
    region when ``integrate=True``).
    """
    grid = wmb.grid
    lambda_var = wmb.get_lambda_var(lambda_name)
    suffix = "greater_than" if greater_than else "less_than"

    if region.assert_zero_transport:
        return _annotate_transport(
            wmb, xr.DataArray(0.0), lambda_name, lambda_var, None, None,
            integrate=integrate,
            comment=(
                "Zero by assertion: the region has no boundary across which mass "
                "can be transported (`assert_zero_transport`)."
            ),
        )

    if utr is None or vtr is None:
        names = transport_varnames(wmb.query)
        if names is None:
            return _annotate_transport(
                wmb, xr.DataArray(0.0), lambda_name, lambda_var, None, None,
                integrate=integrate,
                comment=(
                    "Zero: the recipe declares no lateral advective mass "
                    "transport terms to compute a boundary transport from."
                ),
            )
        utr = utr or names["utr"]
        vtr = vtr or names["vtr"]
    if not all(v in grid._ds for v in (utr, vtr)):
        raise ValueError(
            f"Lateral transports {utr!r}/{vtr!r} are not available in `grid._ds`!"
        )

    if along_section:
        layer_conv = _convergence_along_section(
            wmb, region, lambda_var, target_coords, utr, vtr, prebinned
        )
    else:
        layer_conv = _convergence_from_divergence(
            wmb, region, lambda_var, target_coords, utr, vtr, prebinned
        )
    grid._ds["convergent_mass_transport_layer"] = layer_conv

    accumulated = accumulate_in_lambda(
        grid,
        layer_conv,
        target_coords,
        greater_than,
        name=f"convergent_mass_transport_{suffix}",
    )
    grid._ds[accumulated.name] = accumulated

    conv = interp_to_center(grid, accumulated, target_coords)
    if integrate:
        if "sect" in conv.dims:
            grid._ds["convergent_mass_transport_along"] = conv
            conv = conv.sum("sect")
        else:
            area_dims = [d for d in wmb._horizontal_dims if d in conv.dims]
            conv = conv.sum(area_dims)
    return _annotate_transport(
        wmb, conv, lambda_name, lambda_var, utr, vtr,
        integrate=integrate,
        along_section=along_section,
    )


def _annotate_transport(
    wmb, da, lambda_name, lambda_var, utr, vtr, *,
    integrate=True, along_section=False, comment=None,
):
    """Describe the convergent transport term.

    Every step from the face transports to this term -- the conservative remap
    into lambda layers, the cumulative sum in lambda, the boundary or divergence
    sum, and the interpolation to layer centres -- preserves units, so the face
    transports' own units carry through. They must agree with each other: a
    zonal transport in kg s-1 and a meridional one in m3 s-1 do not sum to
    anything.
    """
    ds = wmb.grid._ds
    sources = {name: ds.get(name) for name in (utr, vtr) if name is not None}
    if sources:
        units = _attrs.common_units(
            [_attrs.units_of(v) for v in sources.values()],
            term="convergent_mass_transport",
        )
        units_source = "source" if units else None
    else:
        # An identically-zero transport still has units -- those of the mass budget
        # it belongs to, as the recipe declares them. Leaving it undescribed would
        # poison the units of every sum it later takes part in.
        units = _budget_units(wmb)
        units_source = "recipe" if units else None

    if integrate and along_section:
        cell_methods = "sect: sum"
    elif integrate:
        cell_methods = " ".join(f"{d}: sum" for d in wmb._horizontal_dims)
    else:
        cell_methods = None

    return _attrs.annotate(
        da,
        "convergent_mass_transport",
        units=units,
        units_source=units_source,
        lambda_name=lambda_name,
        lambda_var=lambda_var,
        cell_methods=cell_methods,
        sources=sources,
        extra={"comment": comment} if comment else None,
    )


def _budget_units(wmb):
    """The units the recipe declares for the mass budget, if it declares any."""
    try:
        return wmb.query.budget_units("mass")
    except (KeyError, AttributeError):  # pragma: no cover - recipe without `mass`
        return None


def _convergence_along_section(
    wmb, region, lambda_var, target_coords, utr, vtr, prebinned
):
    grid = wmb.grid
    zc = grid.axes["Z"].coords["center"]
    zi = grid.axes["Z"].coords["outer"]
    # sectionate needs a horizontal-only grid (a Z axis breaks its corner padding);
    # `_ds` is shared, so transports/tracers still resolve against the full dataset.
    hgrid = horizontal_grid(grid)

    convs, tracers = [], []
    for loop in region.boundaries:
        conv = sectionate.convergent_transport(
            hgrid,
            loop.i_c,
            loop.j_c,
            f_c=loop.f_c,
            utr=utr,
            vtr=vtr,
            layer=zc,
            interface=zi,
            geometry="spherical",
            positive_in=region.mask,
        )
        conv = conv.rename({"lon": "lon_sect", "lat": "lat_sect"})[
            "conv_mass_transport"
        ]
        convs.append(conv)
        if not prebinned:
            tracers.append(
                sectionate.extract_tracer(
                    lambda_var, hgrid, loop.i_c, loop.j_c, f_c=loop.f_c
                )
            )

    conv = xr.concat(convs, dim="sect") if len(convs) > 1 else convs[0]
    grid._ds["convergent_mass_transport_original"] = conv

    if prebinned:
        target_data = grid._ds[f"{lambda_var}_i"]
    else:
        tracer = xr.concat(tracers, dim="sect") if len(tracers) > 1 else tracers[0]
        grid._ds[f"{lambda_var}_sect"] = tracer
        # The section tracer has no face dimension, so interpolate it to interfaces
        # with a vertical-only grid (avoids xgcm's face-connection padding path).
        target_data = interp_to_interfaces(vertical_grid(grid, "Z"), tracer, "Z").rename(
            f"{lambda_var}_i_sect"
        )
        grid._ds[f"{lambda_var}_i_sect"] = target_data

    return transform_to_lambda(grid, conv, target_coords, target_data)


def _convergence_from_divergence(
    wmb, region, lambda_var, target_coords, utr, vtr, prebinned
):
    grid = wmb.grid
    if region.is_multitile or getattr(grid, "_facedim", None) is not None:
        raise NotImplementedError(
            "The grid-cell divergence method is not implemented for multi-tile "
            "(face_connections) grids. Use `along_section=True`, which computes the "
            "boundary transport with sectionate (face-index aware)."
        )
    zc = grid.axes["Z"].coords["center"]
    xo = grid.axes["X"].coords["outer"]
    yo = grid.axes["Y"].coords["outer"]

    if prebinned:
        lam_XZ = grid._ds[f"{lambda_var}_i"]
        lam_YZ = grid._ds[f"{lambda_var}_i"]
    else:
        lam_XZ = interp_to_interfaces(
            grid, grid.interp(grid._ds[lambda_var], "X"), "Z"
        )
        lam_YZ = interp_to_interfaces(
            grid, grid.interp(grid._ds[lambda_var], "Y"), "Z"
        )

    divergence_X = grid.diff(
        transform_to_lambda(
            grid, grid._ds[utr].chunk({zc: -1}), target_coords, lam_XZ
        )
        .fillna(0.0)
        .chunk({xo: -1}),
        "X",
    )
    divergence_Y = grid.diff(
        transform_to_lambda(
            grid, grid._ds[vtr].chunk({zc: -1}), target_coords, lam_YZ
        )
        .fillna(0.0)
        .chunk({yo: -1}),
        "Y",
    )
    return -(divergence_X + divergence_Y) * region.mask
