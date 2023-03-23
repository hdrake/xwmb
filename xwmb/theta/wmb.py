import xarray as xr
import numpy as np
import warnings

import matplotlib.pyplot as plt

from .wmt import *
from .dMdt import *
from .psi import *

def calc_wmb_theta(
        ds, grid, snap, grid_snap, ocean_grid,
        mask, i, j,
        theta_min = -4, theta_max = 40., Δtheta = 0.1,
        rho0=1035., Cp=3992.
    ):
    
    # Transform budget to theta coordinates
    theta_i_bins = np.arange(theta_min - Δtheta*0.5, theta_max + Δtheta*0.5, Δtheta)
    theta_l_levs = np.arange(theta_min, theta_max, Δtheta)
    ds_theta = transform_to_theta(ds, grid, theta_i_bins, theta_l_levs)
    
    wmb = xr.Dataset()
    calc_wmt_theta(wmb, ds_theta, grid, ocean_grid, Δtheta, mask=mask, rho0=rho0, Cp=Cp)
    calc_dMdt_theta(wmb, ds, snap, grid_snap, ocean_grid, wmb.basin_mask, theta_i_bins, rho0=rho0)
    calc_psi_theta(wmb, ds, grid, i, j, theta_i_bins, ocean_grid['geolon'].shape==ocean_grid['geolon_c'].shape)
    
    # Discretization error/residual terms
    wmb['N_A'] =   wmb.G_adv  + wmb.Psi
    wmb['N_D'] = - wmb.G_tend - wmb.dMdt
    wmb['N'] = wmb['N_A'] + wmb['N_D']
        
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
            {v:grid.transform(ds[v], 'Z', theta_i_bins, target_data=ds.thetao_i, method="conservative") for v in budget_vars if v in ds.data_vars}
        ) # note: this binning effectively already calculates the difference between the integral below Θ+ΔΘ and the integral below Θ

        ds_theta = ds_theta.assign_coords({"thetao_l": theta_l_levs})
        
    return ds_theta

def plot_wmb(wmb, ylim=[-3, 36], rho0=1035.):
    plt.figure(figsize=(12,5))

    plt.subplot(1,3,1)
    (-wmb.dMdt.mean('time')  / rho0*1e-6 ).plot(y="thetao_i", label=r"$-$d$M/$d$t$, from diff of $h(\Theta)$ snapshots")
    ((wmb.G_tend).mean('time') / rho0*1e-6 ).plot(y="thetao_i", label=r"$\mathcal{G}_{tend}$ from $\dot{\Theta}$")
    plt.title("")
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.15)
    plt.ylabel("Potential temperature [$\degree$C]")
    plt.xlabel("Watermass volume tendency [Sv]")
    plt.ylim(ylim)

    plt.subplot(1,3,2)
    (wmb.Psi.mean('time')  / rho0*1e-6 ).plot(y="thetao_i", label="$\Psi(\Theta)$ from $\mathbf{u}(\Theta)$")
    ((-wmb.G_adv).mean('time') / rho0*1e-6 ).plot(y="thetao_i", label="$-\mathcal{G}_{adv}$ from $\dot{\Theta}$")
    plt.title("")
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.15)
    plt.ylabel("Potential temperature [$\degree$C]")
    plt.xlabel("Watermass volume tendency [Sv]")
    plt.ylim(ylim)
    
    plt.subplot(1,3,3)
    plt.fill_betweenx(
        wmb.thetao_i,
        wmb.G_NC.mean('time')/rho0*1e-6, (-wmb.dMdt + wmb.Psi).mean('time')/rho0*1e-6, where=wmb.N.mean('time')>=0, 
        alpha=0.25, color="r", label=r"$\mathcal{N} = \mathcal{N}_{A} + \mathcal{N}_{D} = -$d$M/$d$t + \Psi - \mathcal{G}_{NC}$"
    )
    plt.fill_betweenx(
        wmb.thetao_i,
        wmb.G_NC.mean('time')/rho0*1e-6, (-wmb.dMdt + wmb.Psi).mean('time')/rho0*1e-6, where=wmb.N.mean('time')<0, 
        alpha=0.25, color="b"
    )
    ((-wmb.dMdt + wmb.Psi)/rho0*1e-6).mean('time').plot(y="thetao_i", label=r"$-$d$M/$d$t + \Psi$")
    ((wmb.G_tend - wmb.G_adv)/rho0*1e-6).mean('time').plot(y="thetao_i", label=r"$\mathcal{G}_{tend} - \mathcal{G}_{adv}$")
    ((wmb.G_NC) / rho0*1e-6).mean('time').plot(y="thetao_i", label=r"$\mathcal{G}_{NC}$", linestyle="--", alpha=0.8, lw=1.)
    plt.title("")
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.15)
    plt.ylabel("Potential temperature [$\degree$C]")
    plt.xlabel("Watermass volume tendency [Sv]")
    plt.ylim(ylim)
    plt.tight_layout()
    
def plot_wmb_decomposed(wmb, ylim=[-3, 36], rho0=1035.):
    plt.figure(figsize=(14, 6))

    plt.subplot(1,2,1)
    (wmb['dMdt'].mean('time')/rho0*1e-6).plot(ls="-", y="thetao_i", label=r"d$M(\Theta)/$d$t$ (Mass Tendency)")
    (-wmb['Psi'].mean('time')/rho0*1e-6).plot(ls="-", y="thetao_i", label=r"$-\Psi(\Theta)$ (Convergent Transport)")
    (wmb['G_NC'].mean('time')/rho0*1e-6).plot(ls="-", y="thetao_i", label=r"$\mathcal{G}_{NC}(\Theta)$ (Non-Conservative Forcing)")
    (wmb['N'].mean('time')/rho0*1e-6).plot(ls="-", y="thetao_i", label=r"$\mathcal{N}(\Theta)$ (Numerical Mixing)")

    plt.title("Time-mean Water Mass Budget")
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.15)
    plt.ylabel("Potential temperature [$\degree$C]")
    plt.xlabel("Watermass volume tendency [Sv]")
    plt.ylim(ylim)

    plt.subplot(1,2,2)
    (wmb['G_NC'].mean('time')/rho0*1e-6).plot(color="k", ls="-", y="thetao_i", label=r"$\mathcal{G}_{NC}(\Theta)$ (Non-Conservative Forcing)")
    (wmb['G_surf'].mean('time')/rho0*1e-6).plot(ls="-", y="thetao_i", label=r"Surface Fluxes")
    l = (wmb['G_mix'].mean('time')/rho0*1e-6).plot(ls="-", y="thetao_i", label=r"Mixing")
    (wmb['G_ice'].mean('time')/rho0*1e-6).plot(ls="-", y="thetao_i", label=r"Frazil Ice")
    (wmb['G_geo'].mean('time')/rho0*1e-6).plot(ls="-", y="thetao_i", label=r"Geothermal")
    (wmb['G_iso'].mean('time')/rho0*1e-6).plot(ls="-", y="thetao_i", label=r"Stirring")
    (wmb['N'].mean('time')/rho0*1e-6).plot(color=l[0].get_c(), alpha=0.5, ls="--", y="thetao_i", label=r"$\mathcal{N}(\Theta)$ (Numerical Mixing)")


    plt.title("Decomposition of time-mean Non-Conservative WMT")
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.15)
    plt.ylabel("Potential temperature [$\degree$C]")
    plt.xlabel("Watermass volume tendency [Sv]")
    plt.ylim(ylim)
    plt.tight_layout()