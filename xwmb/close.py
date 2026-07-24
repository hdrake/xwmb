"""Close the water-mass budget: infer spurious numerical mixing from the residual."""

__all__ = ["close_budget"]


def close_budget(ds):
    """Close the budget by identifying the residual as spurious numerical mixing.

    The realized (diagnosed) transformation is the residual of the storage, source,
    and convergent-transport terms; the part of it not explained by the parameterized
    material transformation is attributed to spurious numerical mixing.
    """
    realized_transformation_terms = [
        "mass_tendency",
        "mass_source",
        "convergent_mass_transport",
    ]
    if not all(term in ds for term in realized_transformation_terms):
        return

    ds["realized_transformation"] = (
        ds.mass_tendency - ds.mass_source - ds.convergent_mass_transport
    )
    # By construction kinematic == material transformation; use whichever is present.
    if "material_transformation" in ds:
        ds["spurious_numerical_mixing"] = (
            ds.realized_transformation - ds.material_transformation
        )
    elif "kinematic_transformation" in ds:
        ds["spurious_numerical_mixing"] = (
            ds.realized_transformation - ds.kinematic_transformation
        )

    if "advection" in ds:
        if "surface_ocean_flux_advective_negative_lhs" in ds:
            ds["advection_plus_BC"] = (
                ds.advection + ds.surface_ocean_flux_advective_negative_lhs
            )
        else:
            ds["advection_plus_BC"] = ds.advection

    if ("spurious_numerical_mixing" in ds) and ("advection_plus_BC" in ds):
        ds["diabatic_advection"] = (
            ds.advection_plus_BC + ds.spurious_numerical_mixing
        )
