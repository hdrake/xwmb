"""Close the water-mass budget and, when earned, name its residual.

``realized_transformation`` is what the mass budget says the transformation across
each lambda surface was: the storage tendency, less what the surface mass source
and the boundary transport put there. ``residual`` is the part of it that the
model's own material processes do not account for.

Whether that residual may be called *spurious numerical mixing* is a separate
question, answered by :mod:`xwmb.completeness`: only a budget with nothing else
missing has a residual that means what the name says.
"""

import xarray as xr

from . import attrs as _attrs
from .completeness import warn_incomplete

__all__ = ["close_budget", "REALIZED_TERMS"]

#: The signed terms of the realized transformation: dM/dt - S - Psi.
REALIZED_TERMS = (
    ("mass_tendency", 1.0),
    ("mass_source", -1.0),
    ("convergent_mass_transport", -1.0),
)


def close_budget(ds, report=None, lambda_name=None):
    """Compute the budget residual, and name it only if the budget is closed.

    Parameters
    ----------
    ds : xr.Dataset
        The budget terms, modified in place.
    report : CompletenessReport, optional
        The audit from :func:`xwmb.completeness.budget_completeness`. Without one
        the budget is assumed complete, and the residual is named as before.
    lambda_name : str, optional
        Used in the warning text and in the derived variables' metadata.

    Notes
    -----
    Absent terms are treated as zero *and recorded* in ``xwmb_assumed_zero``,
    rather than aborting the calculation. The previous behavior -- returning
    silently, having computed nothing at all, whenever any one of the three terms
    was missing -- left the user with no residual and no explanation for why.
    """
    lambda_name = lambda_name or ds.attrs.get("xwmt_lambda")
    lambda_var = ds.attrs.get("xwmt_lambda_variable")

    present = [(name, sign) for name, sign in REALIZED_TERMS if name in ds]
    if not present:
        return
    absent = [name for name, _ in REALIZED_TERMS if name not in ds]

    realized = sum(sign * ds[name] for name, sign in present)
    units = _attrs.common_units(
        [_attrs.units_of(ds[name]) for name, _ in present],
        term="realized_transformation",
    )
    _attrs.annotate(
        realized,
        "realized_transformation",
        units=units,
        units_source="source" if units else None,
        lambda_name=lambda_name,
        lambda_var=lambda_var,
        extra={"xwmb_assumed_zero": ", ".join(absent)} if absent else None,
    )
    ds["realized_transformation"] = realized

    # By construction kinematic == material transformation; use whichever is present.
    if "material_transformation" in ds:
        parameterized = ds.material_transformation
    elif "kinematic_transformation" in ds:
        parameterized = ds.kinematic_transformation
    else:
        parameterized = None

    if parameterized is not None:
        residual = realized - parameterized
        residual_units = _attrs.common_units(
            [units, _attrs.units_of(parameterized)], term="residual"
        )
        complete = report is None or report.is_complete
        _attrs.annotate(
            residual,
            "residual",
            units=residual_units,
            units_source="source" if residual_units else None,
            lambda_name=lambda_name,
            lambda_var=lambda_var,
            extra=(
                None
                if complete
                else {
                    "xwmb_unaccounted_terms": "; ".join(report.unaccounted()),
                    "comment": (
                        "The budget is not closed, so this residual is the budget "
                        "imbalance and is NOT an estimate of spurious numerical "
                        "mixing. See `xwmb_unaccounted_terms`."
                    ),
                }
            ),
        )
        ds["residual"] = residual

        if complete:
            # Numerically the same array; the second name records the physical
            # interpretation that the completeness audit licenses.
            spurious = residual.copy()
            spurious.attrs = {}
            _attrs.annotate(
                spurious,
                "spurious_numerical_mixing",
                units=residual_units,
                units_source="source" if residual_units else None,
                lambda_name=lambda_name,
                lambda_var=lambda_var,
            )
            ds["spurious_numerical_mixing"] = spurious
        else:
            warn_incomplete(report, lambda_name)

    if "advection" in ds:
        if "surface_ocean_flux_advective_negative_lhs" in ds:
            total = ds.advection + ds.surface_ocean_flux_advective_negative_lhs
            summed = ["advection", "surface_ocean_flux_advective_negative_lhs"]
        else:
            total = ds.advection.copy()
            summed = ["advection"]
        _attrs.annotate(
            total,
            "advection_plus_BC",
            units=_attrs.common_units(
                [_attrs.units_of(ds[t]) for t in summed], term="advection_plus_BC"
            ),
            units_source="source",
            lambda_name=lambda_name,
            lambda_var=lambda_var,
            extra={"xwmb_summed_terms": ", ".join(summed)},
        )
        ds["advection_plus_BC"] = total

    if ("spurious_numerical_mixing" in ds) and ("advection_plus_BC" in ds):
        diabatic = ds.advection_plus_BC + ds.spurious_numerical_mixing
        _attrs.annotate(
            diabatic,
            "diabatic_advection",
            units=_attrs.common_units(
                [
                    _attrs.units_of(ds.advection_plus_BC),
                    _attrs.units_of(ds.spurious_numerical_mixing),
                ],
                term="diabatic_advection",
            ),
            units_source="source",
            lambda_name=lambda_name,
            lambda_var=lambda_var,
            extra={
                "xwmb_summed_terms": "advection_plus_BC, spurious_numerical_mixing"
            },
        )
        ds["diabatic_advection"] = diabatic
