"""Metadata for the variables xwmb derives.

Every term xwmb adds to the budget -- the convergent transport, the mass source,
the layer mass and its snapshots, the tendency, and the residual -- used to come
back as a bare array: no units, no name, and whatever attributes xarray happened
to carry over from whichever operand was on the left of the arithmetic. This
module gives each of them a description composed from its inputs.

Two rules, adopted verbatim from ``xwmt.attrs`` so that reasoning transfers
between the packages:

* Units strings are UDUNITS-2 parseable, as the CF conventions require:
  ``"kg s-1"``, never ``"kg/s"``. The algebra is ``xwmt.units``, a thin wrapper
  over ``cf_units`` -- xwmb does not reimplement it.
* When the units of an input cannot be determined, the ``units`` key is **omitted**
  rather than guessed, and ``xwmb_units_source`` records which authority answered.

The generic attribute plumbing (``set_default_attrs``, ``strip_inherited_attrs``,
``netcdf_safe``, ``prettify``, ``collect_source_attrs``) is imported from
``xwmt.attrs`` rather than copied: each carries a non-obvious rationale in its
docstring -- the allowlist in ``strip_inherited_attrs``, the flattening of
xbudget's mixed-type ``provenance`` lists in ``netcdf_safe`` -- that a second
implementation would drift away from.
"""

import warnings

from xwmt import units as _units
from xwmt.attrs import (
    collect_source_attrs,
    netcdf_safe,
    prettify,
    set_default_attrs,
    strip_inherited_attrs,
)

from .version import __version__

__all__ = [
    "annotate",
    "units_of",
    "product_units",
    "quotient_units",
    "common_units",
    "collect_source_attrs",
    "netcdf_safe",
    "prettify",
    "set_default_attrs",
    "strip_inherited_attrs",
]

#: Attributes that ``xbudget`` stamps onto the terms it materializes. They
#: describe where a quantity came from, so they follow it downstream.
XBUDGET_PASSTHROUGH = ("provenance", "xbudget_path", "xbudget_op")

#: Prose for each derived term. Anything not listed falls back to
#: :func:`prettify` of the term key, so an unlisted term is still described.
LONG_NAMES = {
    "mass_density": "mass per unit area of the water mass in each lambda layer",
    "layer_mass": "mass of the water mass in each lambda layer",
    "mass_bounds": "mass of the water mass at the time-average bounds",
    "mass_tendency": "tendency of the mass of the water mass",
    "dt": "length of the averaging interval",
    "convergent_mass_transport": (
        "convergent mass transport across the region boundary"
    ),
    "mass_source": "mass source of the water mass from the surface mass flux",
    "boundary_fluxes": (
        "water mass transformation by surface and boundary fluxes"
    ),
    "realized_transformation": (
        "realized water mass transformation, inferred from the mass budget"
    ),
    "residual": "residual of the water mass budget",
    "spurious_numerical_mixing": (
        "water mass transformation by spurious numerical mixing"
    ),
    "advection_plus_BC": (
        "water mass transformation by resolved advection, including its "
        "boundary contribution"
    ),
    "diabatic_advection": (
        "diabatic water mass transformation by advection, resolved and spurious"
    ),
}


def units_of(da):
    """The ``units`` attribute of ``da``, or ``None`` if it has none."""
    if da is None:
        return None
    return getattr(da, "attrs", {}).get("units")


def product_units(*specs):
    """UDUNITS string for a product of quantities, or ``None`` if any is unknown."""
    if any(s is None for s in specs):
        return None
    return _units.format_units(_units.multiply(*[_units.parse(s) for s in specs]))


def quotient_units(numerator, denominator):
    """UDUNITS string for a quotient, or ``None`` if either side is unknown."""
    if numerator is None or denominator is None:
        return None
    return _units.format_units(
        _units.divide(_units.parse(numerator), _units.parse(denominator))
    )


def common_units(specs, term=None):
    """The shared units of a set of summands, or ``None`` if they disagree.

    A sum of terms in different units is not a quantity, so rather than adopting
    whichever operand happened to be first, the result is left undescribed and
    the disagreement is reported. Mirrors ``xwmt.wmt._sum_terms``.
    """
    known = [s for s in specs if s is not None]
    if not known or len(known) != len(list(specs)):
        return None
    if not _units.same_units(known):
        warnings.warn(
            f"Combining {'terms' if term is None else repr(term)} with different "
            f"units {sorted(set(known))}; the result is left without a 'units' "
            f"attribute rather than adopting one of them.",
            stacklevel=3,
        )
        return None
    return known[0]


def annotate(
    da,
    term,
    *,
    units=None,
    units_source=None,
    lambda_name=None,
    lambda_var=None,
    cell_methods=None,
    sources=None,
    extra=None,
):
    """Describe a derived budget variable, in place, and return it.

    Parameters
    ----------
    da : xr.DataArray
        The derived quantity. Whatever attributes xarray carried over from its
        operands are dropped first: a derived field is a *different* quantity, so
        an inherited ``long_name`` or ``units`` describes something else.
    term : str
        The budget term key, e.g. ``"convergent_mass_transport"``. Names the
        variable (``xwmb_term``) and, via :data:`LONG_NAMES`, describes it.
    units : str, optional
        UDUNITS-2 units string. Omitted from the output when ``None``.
    units_source : str, optional
        Which authority supplied ``units`` -- ``"source"`` (an input variable's
        own attribute), ``"recipe"`` (the budget's declared units), ``"derived"``
        (composed from inputs by unit algebra), or ``"unknown"``. Defaults to
        ``"derived"`` when units are known and ``"unknown"`` when they are not.
    lambda_name, lambda_var : str, optional
        The water-mass coordinate and the dataset variable holding it.
    cell_methods : str, optional
        CF ``cell_methods`` for the reductions applied.
    sources : dict, optional
        ``{variable name: xr.DataArray or None}`` for the inputs this term was
        built from. Their xbudget provenance is passed through.
    extra : dict, optional
        Additional attributes, e.g. a completeness flag.
    """
    strip_inherited_attrs(da)

    if units_source is None:
        units_source = "derived" if units is not None else "unknown"

    attrs = {
        "units": units,
        "xwmb_units_source": units_source,
        "long_name": LONG_NAMES.get(term, prettify(term)),
        "cell_methods": cell_methods,
        "xwmb_term": term,
        "xwmb_lambda": lambda_name,
        "xwmb_lambda_variable": lambda_var,
        "xwmb_version": __version__,
    }
    if sources:
        present = {k: v for k, v in sources.items() if v is not None}
        attrs.update(collect_source_attrs(present))
        if present:
            attrs["xwmb_source_variables"] = ", ".join(sorted(present))
    if extra:
        attrs.update(extra)

    # Drop the unset keys before coercing: `netcdf_safe` would render a None as
    # the string "None", and an attribute claiming the units are "None" is worse
    # than no attribute at all.
    set_default_attrs(
        da, {k: netcdf_safe(v) for k, v in attrs.items() if v is not None}
    )
    return da
