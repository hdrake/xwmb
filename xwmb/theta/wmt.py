import xarray as xr

def calc_wmt_theta(wmb, ds, grid, ocean_grid, Δtheta, mask=None, rho0=1035., Cp=3992.):
    
    if mask is None:
        wmb['basin_mask'] = xr.DataArray(ocean_grid.wet.data, coords=(ds.yh, ds.xh,))
    else:
        wmb['basin_mask'] = xr.DataArray(mask.data, coords=(ds.yh, ds.xh,))

    ds["net_tendencies"] = (
        ds.opottempdiff +
        ds.opottemppmdiff +
        ds.internal_heat_heat_tendency +
        ds.frazil_heat_tendency +
        ds.boundary_forcing_heat_tendency
    )

    wmb["G_NC"] = ((rho0*ds.net_tendencies.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()

    wmb["G_mix"] = ((rho0*ds.opottempdiff.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["G_geo"] = ((rho0*ds.internal_heat_heat_tendency.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["G_iso"] = ((rho0*ds.opottemppmdiff.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["G_ice"] = ((rho0*ds.frazil_heat_tendency.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["G_surf"] = ((rho0*ds.boundary_forcing_heat_tendency.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()

    wmb["G_adv_h"] = ((rho0*ds.T_advection_xy.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["G_adv_v"] = ((rho0*ds.Th_tendency_vert_remap.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["G_adv"] = wmb.G_adv_h + wmb.G_adv_v

    wmb["G_tend"] = ((rho0*ds.opottemptend.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    
    if 'boundary_forcing_h_tendency' in ds.data_vars:
        wmb["W"] = (rho0*ds.boundary_forcing_h_tendency.where(wmb.basin_mask)*ocean_grid.areacello).sum(['yh', 'xh']).cumsum("thetao_i").compute()
    if 'dynamics_h_tendency' in ds.data_vars:
        wmb["G_adv_htend"] =  -(rho0*ds.dynamics_h_tendency.where(wmb.basin_mask)*ocean_grid.areacello).sum(['yh', 'xh']).cumsum("thetao_i").compute()