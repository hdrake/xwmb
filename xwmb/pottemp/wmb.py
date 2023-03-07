import xarray as xr
import numpy as np
import warnings

from .wmt import *
from .dMdt import *
from .psi import *

def calc_wmb_theta(
        ds, grid, snap, grid_snap, ocean_grid,
        mask, i, j,
        theta_min = -2., theta_max = 36., Δtheta = 0.2,
        rho0=1035., Cp=3992.
    ):
    
    # Transform budget to theta coordinates
    theta_i_bins = np.arange(theta_min - Δtheta*0.5, theta_max + Δtheta*0.5, Δtheta)
    theta_l_levs = np.arange(theta_min, theta_max, Δtheta)
    ds_theta = transform_to_theta(ds, grid, theta_i_bins, theta_l_levs)
    
    wmb = xr.Dataset()
    calc_wmt_theta(wmb, ds_theta, grid, ocean_grid, Δtheta, mask=mask, rho0=rho0, Cp=Cp)
    calc_dMdt_theta(wmb, ds, snap, grid_snap, ocean_grid, wmb.basin_mask, theta_i_bins, rho0=rho0)
    calc_psi_theta(wmb, ds, grid, i, j, theta_i_bins)
    
    return wmb

def transform_to_theta(ds, grid, theta_i_bins, theta_l_levs):
    
    ds['thetao_i'] = grid.transform(ds.thetao, 'Z', ds.z_i, target_data=ds.z_l, method="linear")
    
    budget_vars = [
        "opottempdiff", "opottemppmdiff",
        "frazil_heat_tendency", "internal_heat_heat_tendency", "boundary_forcing_heat_tendency",
        "T_advection_xy", "opottemptend", "Th_tendency_vert_remap",
        "boundary_forcing_h_tendency", "dynamics_h_tendency"
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        ds_theta = xr.Dataset(
            {v:grid.transform(ds[v], 'Z', theta_i_bins, target_data=ds.thetao_i, method="conservative") for v in budget_vars}
        ) # note: this binning effectively already calculates the difference between the integral below Θ+ΔΘ and the integral below Θ

        ds_theta = ds_theta.assign_coords({"thetao_l": theta_l_levs})
        
    return ds_theta