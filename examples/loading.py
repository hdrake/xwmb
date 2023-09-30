import xarray as xr
import numpy as np
from xgcm import Grid

sim = "wmt_incsurffluxes.natv_rho2_zstr.monthly_daily_hourly.13months"
rootdir = f"/archive/Graeme.Macgilchrist/MOM6-examples/ice_ocean_SIS2/Baltic_OM4_025/{sim}/Baltic_OM4_025/"

def load_baltic(gridname, dt):
    """
    Arguments:
    - gridname: str
      Choose from ["natv", "zstr", "rho2"]
    - dt: str
      Choose from ["monthly", "daily", "hourly"]
    """
    
    prefix = '19000101.ocean_'+dt+'_' 
    time = "190*"

    # Diagnostics were saved into different files
    suffixs = ['surf','thck','heat','salt','xtra']
    Zprefixes = {'rho2':'sigma2_', 'zstr':'z_', 'natv':'z'}
    Zprefix = Zprefixes[gridname]
    ds = xr.Dataset()
    for suffix in suffixs:
        if suffix == "surf":
            filename = prefix+suffix+'_'+time+'.nc'
        else:
            filename = prefix+gridname+'_'+suffix+'_'+time+'.nc'
        dsnow = xr.open_mfdataset(rootdir+filename, decode_times=False)
        ds = xr.merge([ds,dsnow])

    # Load snapshot data (for mass tendency term)
    suffix = 'snap'
    filename = prefix+gridname+'_'+suffix+'_'+time+'.nc'
    snap = xr.open_mfdataset(rootdir+filename, decode_times=False)

    # Align N+1 snapshots so they bound N averages, and select year-long subset (from Feb 1 to Feb 1)
    ds = ds.sel(time=slice(snap.time[0], snap.time[-1]))
    ds_decoded = xr.decode_cf(ds.assign_coords({'i': xr.DataArray(np.arange(ds.time.size), coords=(ds.time,))}))
    snap_decoded = xr.decode_cf(snap.assign_coords({'i': xr.DataArray(np.arange(snap.time.size), coords=(snap.time,))}))
    ds = ds.sel(time=ds['time'].values[ds_decoded['time'].sel(time=slice('1900-02-01 00', '1901-02-01 00'))['i'].values])
    snap = snap.sel(time=snap['time'].values[snap_decoded['time'].sel(time=slice('1900-02-01 00', '1901-02-01 00'))['i'].values])

    #  Load grid
    oceangridname = '19000101.ocean_static.nc'
    ocean_grid = xr.open_dataset(rootdir+oceangridname).squeeze()

    # Some renaming to match standard MOM6 conventions (see `xbudget`)
    ocean_grid = ocean_grid.rename({'depth_ocean':'deptho'})
    ds = ds.rename({'temp':'thetao', "salt":'so'})
    snap = snap.rename({'temp':'thetao', "salt":'so'})

    # Merge snapshots with time-averages
    snap = snap.rename({
        **{'time':'time_bounds'},
        **{v:f"{v}_bounds" for v in snap.data_vars}
    })
    ds = xr.merge([ds, snap])

    # Add core coordinates of ocean_grid to ds
    ds = ds.assign_coords({
        "wet": xr.DataArray(ocean_grid["wet"].values, dims=('yh', 'xh',)),
        "areacello": xr.DataArray(ocean_grid["areacello"].values, dims=('yh', 'xh',)),
        'xq': xr.DataArray(ocean_grid['xq'].values, dims=('xq',)),
        'yq': xr.DataArray(ocean_grid['yq'].values, dims=('yq',)),
        'geolon': xr.DataArray(ocean_grid['geolon'].values, dims=('yh','xh')),
        'geolat': xr.DataArray(ocean_grid['geolat'].values, dims=('yh','xh')),
        'geolon_u': xr.DataArray(ocean_grid['geolon_u'].values, dims=('yh','xq')),
        'geolat_u': xr.DataArray(ocean_grid['geolat_u'].values, dims=('yh','xq')),
        'geolon_v': xr.DataArray(ocean_grid['geolon_v'].values, dims=('yq','xh')),
        'geolat_v': xr.DataArray(ocean_grid['geolat_v'].values, dims=('yq','xh')),
        'geolon_c': xr.DataArray(ocean_grid['geolon_c'].values, dims=('yq','xq')),
        'geolat_c': xr.DataArray(ocean_grid['geolat_c'].values, dims=('yq','xq')),
        'dxt': xr.DataArray(ocean_grid['dxt'].values, dims=('yh', 'xh',)),
        'dyt': xr.DataArray(ocean_grid['dyt'].values, dims=('yh', 'xh',)),
        'dxCv': xr.DataArray(ocean_grid['dxCv'].values, dims=('yq', 'xh',)),
        'dyCu': xr.DataArray(ocean_grid['dyCu'].values, dims=('yh', 'xq',)),
        'dxCu': xr.DataArray(ocean_grid['dxCu'].values, dims=('yh', 'xq',)),
        'dyCv': xr.DataArray(ocean_grid['dyCv'].values, dims=('yq', 'xh',)),
        'deptho': xr.DataArray(ocean_grid['deptho'].values, dims=('yh', 'xh',)),
    })
    # lon, lat variables required by gsw package for sea water equation thermodynamics
    ds['lon'] = ds.geolon
    ds['lat'] = ds.geolat

    ds['sigma2_bounds'] = ds['rhopot2_bounds'] - 1000.
    ds['sigma2'] = ds['rhopot2'] - 1000.

    if gridname=="rho2":
        ds = ds.assign_coords({
            "sigma2_l": ds['rho2_l'] - 1000.,
            "sigma2_i": ds['rho2_i'] - 1000.
        }).swap_dims({'rho2_l':'sigma2_l', 'rho2_i':'sigma2_i'})
        
    ds = xr.decode_cf(ds)

    # z-coordinate dataset containing basic state variables
    coords = {
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'},
        'Z': {'center': f'{Zprefix}l', 'outer': f'{Zprefix}i'},
    }
    metrics = {
        ('X','Y'): "areacello",
    }
    grid = Grid(ds, coords=coords, metrics=metrics, boundary={"X":"extend", "Y":"extend", "Z":"extend"}, autoparse_metadata=False)
    
    return grid