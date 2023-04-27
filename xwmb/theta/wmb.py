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
        rho0=1035., Cp=3992.,
    ):
    
    # Transform budget to theta coordinates
    theta_i_bins = np.arange(theta_min - Δtheta*0.5, theta_max + Δtheta*0.5, Δtheta)
    theta_l_bins = np.arange(theta_min, theta_max, Δtheta)
    ds_theta = transform_to_theta(ds, grid, theta_l_bins, theta_i_bins)
    
    wmb = xr.Dataset()
    calc_wmt_theta(wmb, ds, ocean_grid, theta_l_bins, mask=mask, rho_ref=rho0, Cp=Cp)
    calc_dMdt_theta(wmb, ds, snap, grid_snap, ocean_grid, wmb.basin_mask, theta_i_bins, rho0=rho0)
    calc_psi_theta(wmb, ds, grid, i, j, theta_i_bins, ocean_grid['geolon'].shape!=ocean_grid['geolon_c'].shape)
    
    # Discretization error/residual terms
    wmb['numerical_mixing'] =   wmb.advection  + wmb.psi
    wmb['volume_discretization_error'] = - wmb.Eulerian_tendency - wmb.dMdt
    wmb['numerical_errors'] = wmb['numerical_mixing'] + wmb['volume_discretization_error']
        
    return wmb

def transform_to_theta(ds, grid, theta_l_bins, theta_i_bins):
    
    budget_vars = [
        "opottempdiff", "opottemppmdiff",
        "frazil_heat_tendency", "internal_heat_heat_tendency", "boundary_forcing_heat_tendency",
        "T_advection_xy", "opottemptend", "Th_tendency_vert_remap",
        "boundary_forcing_h_tendency", "dynamics_h_tendency",
        "speed"
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        ds_theta = xr.Dataset(
            {v:grid.transform(ds[v], 'Z', target=theta_l_bins, target_data=ds.thetao, method="conservative") for v in budget_vars if v in ds.data_vars}
        ) # note: this binning effectively already calculates the difference between the integral below Θ+ΔΘ and the integral below Θ

        ds_theta = ds_theta.rename({'thetao':'thetao_i'}).assign_coords({"thetao_i": theta_i_bins[1:-1]})
        
    return ds_theta

def plot_wmb(wmb, ylim=[-3, 36], rho0=1035.):
    fig, axes = plt.subplots(1,3,figsize=(12,5))

    ax=axes[0]
    (-wmb.dMdt.mean('time')  / rho0*1e-6 ).plot(ax=ax, y="thetao_i", label=r"$-$d$M/$d$t$, from diff of $h(\Theta)$ snapshots")
    ((wmb.Eulerian_tendency).mean('time') / rho0*1e-6 ).plot(ax=ax, y="thetao_i", label=r"$\mathcal{G}_{tend}$ from $\dot{\Theta}$")
    ax.set_title("")
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.15)
    ax.set_ylabel("Potential temperature [$\degree$C]")
    ax.set_xlabel("Watermass volume tendency [Sv]")
    ax.set_ylim(ylim)

    ax=axes[1]
    (wmb.psi.mean('time')  / rho0*1e-6 ).plot(ax=ax, y="thetao_i", label="$\Psi(\Theta)$ from $\mathbf{u}(\Theta)$")
    ((-wmb.advection).mean('time') / rho0*1e-6 ).plot(ax=ax, y="thetao_i", label="$-\mathcal{G}_{adv}$ from $\dot{\Theta}$")
    ax.set_title("")
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.15)
    ax.set_ylabel("Potential temperature [$\degree$C]")
    ax.set_xlabel("Watermass volume tendency [Sv]")
    ax.set_ylim(ylim)
    
    ax=axes[2]
    ax.fill_betweenx(
        wmb.thetao_i,
        wmb.diabatic_forcing.mean('time')/rho0*1e-6, (-wmb.dMdt + wmb.psi).mean('time')/rho0*1e-6, where=wmb.numerical_errors.mean('time')>=0, 
        alpha=0.25, color="r", label=r"$\mathcal{N} = \mathcal{N}_{A} + \mathcal{N}_{D} = -$d$M/$d$t + \Psi - \mathcal{G}_{NC}$"
    )
    ax.fill_betweenx(
        wmb.thetao_i,
        wmb.diabatic_forcing.mean('time')/rho0*1e-6, (-wmb.dMdt + wmb.psi).mean('time')/rho0*1e-6, where=wmb.numerical_errors.mean('time')<0, 
        alpha=0.25, color="b"
    )
    ((-wmb.dMdt + wmb.psi)/rho0*1e-6).mean('time').plot(ax=ax, y="thetao_i", label=r"$-$d$M/$d$t + \Psi$")
    ((wmb.Eulerian_tendency - wmb.advection)/rho0*1e-6).mean('time').plot(ax=ax, y="thetao_i", label=r"$\mathcal{G}_{tend} - \mathcal{G}_{adv}$")
    ((wmb.diabatic_forcing) / rho0*1e-6).mean('time').plot(ax=ax, y="thetao_i", label=r"$\mathcal{G}_{NC}$", linestyle="--", alpha=0.8, lw=1.)
    ax.set_title("")
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.15)
    ax.set_ylabel("Potential temperature [$\degree$C]")
    ax.set_xlabel("Watermass volume tendency [Sv]")
    ax.set_ylim(ylim)
    fig.tight_layout()
    
    return fig, axes
    
def plot_wmb_decomposed(wmb, ylim=[-3, 36], rho0=1035.):
    fig, axes = plt.subplots(1,2,figsize=(14, 6))

    ax = axes[0]
    (wmb['dMdt'].mean('time')/rho0*1e-6).plot(ax=ax, ls="-", y="thetao_i", label=r"d$M(\Theta)/$d$t$ (Mass Tendency)")
    (-wmb['psi'].mean('time')/rho0*1e-6).plot(ax=ax, ls="-", y="thetao_i", label=r"$-\Psi(\Theta)$ (Convergent Transport)")
    (wmb['diabatic_forcing'].mean('time')/rho0*1e-6).plot(ax=ax, ls="-", y="thetao_i", label=r"$\mathcal{G}_{NC}(\Theta)$ (Non-Conservative Forcing)")
    (wmb['numerical_mixing'].mean('time')/rho0*1e-6).plot(ax=ax, ls="-", y="thetao_i", label=r"$\mathcal{N}(\Theta)$ (Numerical Mixing)")

    ax.set_title("Time-mean Water Mass Budget")
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.15)
    ax.set_ylabel("Potential temperature [$\degree$C]")
    ax.set_xlabel("Watermass volume tendency [Sv]")
    ax.set_ylim(ylim)

    ax = axes[1]
    (wmb['diabatic_forcing'].mean('time')/rho0*1e-6).plot(ax=ax, color="k", ls="-", y="thetao_i", label=r"$\mathcal{G}_{NC}(\Theta)$ (Non-Conservative Forcing)")
    (wmb['boundary_forcing'].mean('time')/rho0*1e-6).plot(ax=ax, ls="-", y="thetao_i", label=r"Surface Fluxes")
    l = (wmb['vertical_diffusion'].mean('time')/rho0*1e-6).plot(ax=ax, ls="-", y="thetao_i", label=r"Mixing")
    (wmb['frazil_ice'].mean('time')/rho0*1e-6).plot(ax=ax, ls="-", y="thetao_i", label=r"Frazil Ice")
    (wmb['geothermal'].mean('time')/rho0*1e-6).plot(ax=ax, ls="-", y="thetao_i", label=r"Geothermal")
    (wmb['neutral_diffusion'].mean('time')/rho0*1e-6).plot(ax=ax, ls="-", y="thetao_i", label=r"Stirring")
    (wmb['numerical_errors'].mean('time')/rho0*1e-6).plot(ax=ax, color=l[0].get_c(), alpha=0.5, ls="--", y="thetao_i", label=r"$\mathcal{N}(\Theta)$ (Numerical Errors)")


    ax.set_title("Decomposition of time-mean Non-Conservative WMT")
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.15)
    ax.set_ylabel("Potential temperature [$\degree$C]")
    ax.set_xlabel("Watermass volume tendency [Sv]")
    ax.set_ylim(ylim)
    fig.tight_layout()
    
    return fig, axes