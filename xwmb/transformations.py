"""Water-mass transformation terms (the diascalar material-process rates).

Thin wrapper over ``xwmt``'s ``integrate_transformations`` / ``map_transformations``
that (1) restricts the calculation to the region mask, (2) flips the sign for
``greater_than`` water masses (the unit normal to the lambda isosurface reverses),
and (3) groups the surface boundary-flux processes into a single ``boundary_fluxes``
term.

Multi-tile grids are handled entirely inside ``xwmt``: its integration paths reduce
over ``self._horizontal_dims``, which includes the face dimension when the grid
carries ``face_connections``.
"""

__all__ = ["compute_transformations", "BOUNDARY_FLUX_TERMS"]

#: Surface/boundary processes summed into a single ``boundary_fluxes`` term.
BOUNDARY_FLUX_TERMS = [
    "surface_exchange_flux",
    "surface_ocean_flux_advective_negative_rhs",
    "bottom_flux",
    "frazil_ice",
]


def compute_transformations(
    wmt_obj, lambda_name, target_coords, mask, greater_than=False, integrate=True
):
    """Return the transformation terms as an ``xr.Dataset`` on the target coordinate."""
    kwargs = {
        "bins": wmt_obj.grid._ds[target_coords["outer"]],
        "mask": mask,
        "group_processes": True,
        "sum_components": True,
    }
    method = (
        wmt_obj.integrate_transformations if integrate else wmt_obj.map_transformations
    )
    wmt = method(lambda_name, **kwargs)

    wmt = wmt.assign_coords(
        {
            target_coords["center"]: wmt_obj.grid._ds[target_coords["center"]],
            target_coords["outer"]: wmt_obj.grid._ds[target_coords["outer"]],
        }
    )

    # For a "greater than" water mass the normal to the isosurface points the other
    # way, so every transformation rate flips sign.
    if greater_than:
        for v in wmt.data_vars:
            wmt[v] = wmt[v] * -1

    present = [t for t in BOUNDARY_FLUX_TERMS if t in wmt]
    if present:
        wmt["boundary_fluxes"] = sum(wmt[t] for t in present)

    return wmt
