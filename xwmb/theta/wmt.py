import xarray as xr
from xwmt.wmt import wmt

def calc_wmt_theta_old(wmb, ds, ocean_grid, Δtheta, mask=None, rho0=1035., Cp=3992.):
    
    if mask is None:
        wmb['basin_mask'] = xr.DataArray(ocean_grid.wet.data, coords=(ds.yh, ds.xh,))
    else:
        wmb['basin_mask'] = xr.DataArray(mask.data, coords=(ds.yh, ds.xh,))

    ds["total_tendency"] = (
        ds.opottempdiff +
        ds.opottemppmdiff +
        ds.internal_heat_heat_tendency +
        ds.frazil_heat_tendency +
        ds.boundary_forcing_heat_tendency
    )

    wmb["diabatic_forcing"] = ((rho0*ds.total_tendency.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()

    wmb["vertical_diffusion"] = ((rho0*ds.opottempdiff.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["geothermal"] = ((rho0*ds.internal_heat_heat_tendency.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["neutral_diffusion"] = ((rho0*ds.opottemppmdiff.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["frazil_ice"] = ((rho0*ds.frazil_heat_tendency.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["boundary_forcing"] = ((rho0*ds.boundary_forcing_heat_tendency.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()

    wmb["horizontal_advection"] = ((rho0*ds.T_advection_xy.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["vertical_advection"] = ((rho0*ds.Th_tendency_vert_remap.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    wmb["advection"] = wmb.horizontal_advection + wmb.vertical_advection

    wmb["Eulerian_tendency"] = ((rho0*ds.opottemptend.where(wmb.basin_mask)/(Cp*rho0)*ocean_grid.areacello).sum(['yh', 'xh'])/Δtheta).compute()
    
    if 'boundary_forcing_h_tendency' in ds.data_vars:
        wmb["W"] = (rho0*ds.boundary_forcing_h_tendency.where(wmb.basin_mask)*ocean_grid.areacello).sum(['yh', 'xh']).cumsum("thetao_i").compute()
    if 'dynamics_h_tendency' in ds.data_vars:
        wmb["horizontal_advection_h_tendency"] =  -(rho0*ds.dynamics_h_tendency.where(wmb.basin_mask)*ocean_grid.areacello).sum(['yh', 'xh']).cumsum("thetao_i").compute()
        
def calc_wmt_theta(wmb, ds, ocean_grid, bins, mask=None, rho_ref=1035., Cp=3992.):

    if mask is None:
        wmb['basin_mask'] = xr.DataArray(ocean_grid.wet.data, coords=(ds.yh, ds.xh,))
    else:
        wmb['basin_mask'] = xr.DataArray(mask.data, coords=(ds.yh, ds.xh,))
    
    ds_mod = ds.copy().assign_coords({
        'geolon':xr.DataArray(ocean_grid['geolon'].values, dims=('yh', 'xh',)),
        'geolat':xr.DataArray(ocean_grid['geolat'].values, dims=('yh', 'xh',))
    })
    ds_mod['areacello'] = ocean_grid['areacello']
    ds_mod = ds_mod.rename({'rho2_l': 'lev',
         'rho2_i': 'lev_outer',
         'xh': 'x',
         'yh': 'y',
         'geolat': 'lat',
         'geolon': 'lon'
    }).drop_dims(['xq', 'yq', 'nv'])
    mask_mod = wmb['basin_mask'].rename({'xh':'x', 'yh':'y'})
    G = wmt(ds_mod.where(mask_mod), Cp=Cp, rho_ref=rho_ref).G(
        "theta",
        bins=bins,
        sum_components=True,
        group_processes=True
    ).rename({'thetao':'thetao_i'})
    for k,v in G.items():
        wmb[k] = v*rho_ref