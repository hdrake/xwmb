import xarray as xr
from xwmt.wmt import wmt
        
def calc_wmt_theta(wmb, ds, ocean_grid, bins, mask=None, rho_ref=1035., Cp=3992.):

    if mask is None:
        wmb['basin_mask'] = xr.DataArray(ocean_grid.wet.data, coords=(ds.yh, ds.xh,))
    else:
        wmb['basin_mask'] = xr.DataArray(mask.data, coords=(ds.yh, ds.xh,))
    
    # To do: refactor so that this pre-processing is not necessary!
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