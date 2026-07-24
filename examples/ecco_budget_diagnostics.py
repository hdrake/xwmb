"""ECCOv4r4 derived surface/bottom heat-flux tendencies.

These build the volume-integrated potential-temperature tendencies that the
`xbudget` ``ECCOV4r4_native`` preset expects but that are not raw ECCO diagnostics:
the geothermal bottom flux, and the penetrative (shortwave) and non-penetrative
parts of the surface heat flux. They are adapted verbatim from the `xbudget`
package's ``examples/eccov4r4_budget_diagnostics.py`` (same authors), and expect the
merged ECCO dataset ``ds = grid._ds`` with native metrics (``drF``, ``hFacC``,
``rA``, ``Z``, ``Zp1``) and the raw flux fields (``geothermalFlux``, ``oceQsw``,
``TFLUX``). All return ``degree_C m3 s-1``.
"""

import numpy as np
import xarray as xr

RHO0 = 1029.0  # reference seawater density (kg/m^3)
C_P = 3994.0   # heat capacity (J/kg/K)


def eccov4r4_geothermal_heat_flux_tendency(ds):
    """Geothermal bottom heat flux distributed into the deepest wet cell of each column."""
    geothermal_flux = ds["geothermalFlux"].copy(deep=True)
    cell_volume = ds["drF"] * ds["hFacC"] * ds["rA"]

    cell_mask = xr.where(ds["hFacC"] > 0, 1.0, 0.0)
    shifted_cell_mask = cell_mask.shift(k=-1).fillna(0.0)
    geothermal_flux_3d = geothermal_flux * (cell_mask - shifted_cell_mask)
    geothermal_flux_3d = geothermal_flux_3d.transpose("k", "tile", "j", "i")

    geothermal_forcing = (geothermal_flux_3d / (RHO0 * C_P)) / (ds["hFacC"] * ds["drF"])
    tend = geothermal_forcing * cell_volume
    tend.attrs = {
        "standard_name": "geothermal_heat_flux_convergence",
        "long_name": "Geothermal heat flux convergence",
        "units": "degree_C m3 s-1",
    }
    return tend.fillna(0.0)


def eccov4r4_penetrative_heat_flux_tendency(ds):
    """Penetrative (shortwave) heating via the ECCO double-exponential attenuation."""
    cell_volume = ((ds["drF"] * ds["hFacC"]) * ds["rA"]).fillna(0.0)

    shortwave_fraction = 0.62
    zeta1, zeta2 = 0.6, 20.0
    z_cutoff = -200

    cell_center_depth = ds["Z"].compute()
    cell_interface_depth = ds["Zp1"].compute()
    interface_depth = np.concatenate([cell_interface_depth.values[:-1], [np.nan]])

    def decay(z):
        return shortwave_fraction * np.exp(z / zeta1) + (
            1.0 - shortwave_fraction
        ) * np.exp(z / zeta2)

    upper_decay = xr.DataArray(
        decay(interface_depth[:-1]), coords=[cell_center_depth.k], dims=["k"]
    )
    lower_decay = xr.DataArray(
        decay(interface_depth[1:]), coords=[cell_center_depth.k], dims=["k"]
    )

    cutoff_index = np.where(cell_center_depth < z_cutoff)[0][0]
    upper_decay.values[cutoff_index:] = 0
    lower_decay.values[cutoff_index - 1 :] = 0

    cell_mask = xr.where(ds["hFacC"] > 0, 1.0, 0.0)
    lower_cell_mask = xr.where(cell_mask.shift(k=-1) == 1.0, 1.0, 0.0)

    interior = (upper_decay * cell_mask - lower_decay * lower_cell_mask) * ds["oceQsw"]
    surface = ((upper_decay[0] - lower_decay[0]) * ds["oceQsw"]) * cell_mask.isel(k=0)
    surface = surface.expand_dims(k=[ds["k"].isel(k=0).item()])
    convergence = xr.concat(
        [surface, interior.isel(k=slice(1, None))], dim="k"
    ).fillna(0.0)

    tend = (convergence * cell_volume / (RHO0 * C_P)) / (ds["hFacC"] * ds["drF"])
    tend.attrs = {
        "long_name": "penetrative (shortwave) potential-temperature tendency",
        "units": "degree_C m3 s-1",
    }
    return tend


def eccov4r4_nonpenetrative_heat_flux_tendency(ds):
    """Non-penetrative surface heat flux (TFLUX - oceQsw) absorbed in the top layer."""
    cell_volume = ((ds["drF"] * ds["hFacC"]) * ds["rA"]).fillna(0.0)

    cell_mask = xr.where(ds["hFacC"] > 0, 1.0, 0.0)
    surface_cell_mask = cell_mask * xr.where(ds["k"] == 0, 1.0, np.nan)
    surface_forcing = ((ds["TFLUX"] - ds["oceQsw"]) * surface_cell_mask).fillna(0.0)

    nonshortwave = (surface_forcing / (RHO0 * C_P)) / (ds["hFacC"] * ds["drF"])
    tend = (nonshortwave * cell_volume).fillna(0.0)
    tend.attrs = {
        "long_name": "non-penetrative surface potential-temperature tendency",
        "units": "degree_C m3 s-1",
    }
    return tend
