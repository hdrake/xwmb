import numpy as np
import xarray as xr
import xgcm

z_suffixes = {
    "zstr": "z",
    "rho2": "rho2"
}

def load_CM4p25(lambda_name, lambda_lims, z_coord="zstr", Nlam=100):
    realm = "ocean"
    frequency = "annual"
    diag_path = "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20221223/CM4_piControl_c192_OM4p25_v8/gfdl.ncrc4-intel18-prod-openmp/pp/"
    
    suffix = z_suffixes[z_coord]
    ds = xr.open_mfdataset(f"{diag_path}/{realm}_{frequency}_{suffix}/ts/{frequency}/10yr/*.0341*.nc", chunks={'time':1}, decode_times=False).isel(time=[0])
    
    # Add lambda target grid coordinates
    dlam = (lambda_lims[-1]-lambda_lims[0])/Nlam
    ds = ds.assign_coords({
        f"{lambda_name}_l" : np.linspace(lambda_lims[0], lambda_lims[-1], Nlam),
        f"{lambda_name}_i" : np.linspace(lambda_lims[0]-dlam/2., lambda_lims[-1]+dlam/2, Nlam+1)
    })
    
    og = xr.open_dataset(f"{diag_path}/{realm}_{frequency}_{suffix}/{realm}_{frequency}_{suffix}.static.nc")
    sg = xr.open_dataset("/archive/Raphael.Dussin/datasets/OM4p25/c192_OM4_025_grid_No_mg_drag_v20160808_unpacked/ocean_hgrid.nc")
    
    ds = fix_grid_coords(ds, og, sg)
    return ds_to_grid(ds, lambda_name)
    
    
def fix_grid_coords(ds, og, sg):
    og['deptho'] = (
        og['deptho'].where(~np.isnan(og['deptho']), 0.)
    )
    og = og.assign_coords({
        'geolon'  : xr.DataArray(sg['x'][1::2,1::2].data, dims=["yh", "xh"]),
        'geolat'  : xr.DataArray(sg['y'][1::2,1::2].data, dims=["yh", "xh"]),
        'geolon_u': xr.DataArray(sg['x'][1::2,0::2].data, dims=["yh", "xq"]),
        'geolat_u': xr.DataArray(sg['y'][1::2,0::2].data, dims=["yh", "xq"]),
        'geolon_v': xr.DataArray(sg['x'][0::2,1::2].data, dims=["yq", "xh"]),
        'geolat_v': xr.DataArray(sg['y'][0::2,1::2].data, dims=["yq", "xh"]),
        'geolon_c': xr.DataArray(sg['x'][0::2,0::2].data, dims=["yq", "xq"]),
        'geolat_c': xr.DataArray(sg['y'][0::2,0::2].data, dims=["yq", "xq"])
    })
    
    ds = ds.assign_coords({
        'dxCv': xr.DataArray(
            og['dxCv'].transpose('xh', 'yq').values, dims=('xh', 'yq',)
        ),
        'dyCu': xr.DataArray(
            og['dyCu'].transpose('xq', 'yh').values, dims=('xq', 'yh',)
        )
    }) # add velocity face widths to calculate distances along the section
    
    ds = ds.assign_coords({
        'areacello':xr.DataArray(og['areacello'].values, dims=("yh", "xh")),
        'geolon':   xr.DataArray(og['geolon'].values, dims=("yh", "xh")),
        'geolat':   xr.DataArray(og['geolat'].values, dims=("yh", "xh")),
        'geolon_u': xr.DataArray(og['geolon_u'].values, dims=("yh", "xq",)),
        'geolat_u': xr.DataArray(og['geolat_u'].values, dims=("yh", "xq",)),
        'geolon_v': xr.DataArray(og['geolon_v'].values, dims=("yq", "xh",)),
        'geolat_v': xr.DataArray(og['geolat_v'].values, dims=("yq", "xh",)),
        'geolon_c': xr.DataArray(og['geolon_c'].values, dims=("yq", "xq",)),
        'geolat_c': xr.DataArray(og['geolat_c'].values, dims=("yq", "xq",)),
        'deptho':   xr.DataArray(og['deptho'].values, dims=("yh", "xh",)),
    })
    ds['lat'] = ds['geolat']
    ds['lon'] = ds['geolon']
    ds['thkcello'] = ds['volcello']/ds['areacello']
    
    return ds

def ds_to_grid(ds, lambda_name, z_coord="zstr"):
    coords={
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'},
        'lam': {'center': f'{lambda_name}_l', 'outer': f'{lambda_name}_i'}
    }
    if "z_l" in ds.dims:
        coords = {
            **coords,
            **{'Z': {'center': 'z_l', 'outer': 'z_i'}}
        }
    elif "rho2_l" in ds.dims:
        coords = {
            **coords,
            **{'Z': {'center': 'rho2_l', 'outer': 'rho2_i'}}
        }
        
    metrics = {
        ('X','Y'): "areacello",
    }
    
    return ds, xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=["X"])