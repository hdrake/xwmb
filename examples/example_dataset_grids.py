import numpy as np
import xarray as xr
import xgcm

z_suffixes = {
    "natv": "",
    "zstr": "_z",
    "rho2": "_rho2"
}
chunks = {'time':1, 'xh':400, 'yh':400}

def load_OM4p5(z_coord="zstr"):
    realm = "ocean"
    frequency = "monthly"
    diag_path = "/archive/Jan-erik.Tesdal/FMS2021.02_mom6_20210630/OM4p5_JRA55do1.4_0netfw_cycle1/gfdl.ncrc4-intel18-prod/pp/"

    chunks = {'time':1, 'xh':-1, 'yh':-1}
    
    suffix = z_suffixes[z_coord]
    ds = xr.open_mfdataset(
        f"{diag_path}/{realm}_{frequency}{suffix}/ts/{frequency}/5yr/*2013*.nc", chunks=chunks
    ).isel(time=slice(12,None))
    if z_coord != "natv":
        ds_surf = xr.open_mfdataset(
            f"{diag_path}/{realm}_{frequency}/ts/{frequency}/5yr/*2013*os.nc", chunks=chunks
        ).isel(time=slice(12,None))
        ds = xr.merge([ds, ds_surf])
    ds['sigma2'] = ds['rhopot2'] - 1000.
    ds = make_symmetric(ds)

    if (z_coord in ["zstr", "rho2"]) and (frequency=="monthly"):
        frequency_mod = "month"
    else:
        frequency_mod = frequency
    snap = xr.open_mfdataset(
        f"{diag_path}/{realm}_{frequency_mod}{suffix}_snap/ts/{frequency}/5yr/*2013*.nc", chunks=chunks
    ).isel(time=slice(11,None))

    ### !!! COMMENT OUT FOR PRODUCTION !!! 
    #ds = ds.isel(time=slice(0,1))
    #snap = snap.isel(time=slice(0,2))
    ds = ds.isel(time=slice(0,12))
    snap = snap.isel(time=slice(0,13))
    ### !!! COMMENT OUT FOR PRODUCTION !!!
    
    # Merge snapshots with time-averages
    snap['sigma2'] = snap['rhopot2'] - 1000.
    snap = snap.rename({
        **{'time':'time_bounds'},
        **{v:f"{v}_bounds" for v in snap.data_vars}
    })
    ds = xr.merge([ds, snap])
    
    og = xr.open_dataset(f"{diag_path}/{realm}_{frequency}{suffix}/{realm}_{frequency}{suffix}.static.nc")
    og = make_symmetric(og)
    og['geolon_c'] = xr.where(
        (og.xq==og.xq[0]) & (og.yq!=og.yq[-1]),
        og['geolon_c'].isel(xq=-1)-360.,
        og['geolon_c']
    )
    og['geolon_c'] = xr.where(
        og.yq==og.yq[0],
        og['geolon_c'].isel(yq=1),
        og['geolon_c']
    )
    og['geolat_c'] = xr.where(
        og.yq==og.yq[0],
        og['geolat_c'].isel(yq=1)+(og.yq[0]-og.yq[1]),
        og['geolat_c']
    )
    og['geolat_v'] = xr.where(
        og.yq==og.yq[0],
        og['geolat_v'].isel(yq=1)+(og.yq[0]-og.yq[1]),
        og['geolat_v']
    )
    
    ds = add_grid_coords(ds, og)
    return ds_to_grid(ds)

def load_CM4p25(z_coord="zstr", Nlam=100):
    realm = "ocean"
    frequency = "annual"
    diag_path = "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20221223/CM4_piControl_c192_OM4p25_v8/gfdl.ncrc4-intel18-prod-openmp/pp/"
    
    suffix = z_suffixes[z_coord]
    ds = xr.open_mfdataset(
        f"{diag_path}/{realm}_{frequency}{suffix}/ts/{frequency}/10yr/*.0341*.nc",
        chunks=chunks,
        decode_times=False
    ).isel(time=[0])
    
    og = xr.open_dataset(f"{diag_path}/{realm}_{frequency}_{suffix}/{realm}_{frequency}_{suffix}.static.nc")
    sg = xr.open_dataset("/archive/Raphael.Dussin/datasets/OM4p25/c192_OM4_025_grid_No_mg_drag_v20160808_unpacked/ocean_hgrid.nc")
    
    og = fix_geo_coords(og, sg)
    ds = add_grid_coords(ds, og)
    return ds_to_grid(ds)
    

def fix_geo_coords(og, sg):
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
    return og
    
def add_grid_coords(ds, og):
    og['deptho'] = (
        og['deptho'].where(~np.isnan(og['deptho']), 0.)
    )
    
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

def ds_to_grid(ds, z_coord="zstr"):
    coords={
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'},
    }
    if "z_l" in ds.dims:
        coords = {
            **coords,
            **{'Z': {'center': 'z_l', 'outer': 'z_i'}}
        }
    elif "zl" in ds.dims:
        coords = {
            **coords,
            **{'Z': {'center': 'zl', 'outer': 'zi'}}
        }
    elif "rho2_l" in ds.dims:
        coords = {
            **coords,
            **{'Z': {'center': 'rho2_l', 'outer': 'rho2_i'}}
        }
        
    metrics = {('X','Y'): "areacello"}
    boundary = {"X":"periodic", "Y":"extend", "Z":"extend"}
    return xgcm.Grid(ds, coords=coords, metrics=metrics, boundary=boundary, autoparse_metadata=False)

def make_symmetric(ds):
    for dim, offset in zip(["xq", "yq"], [-360., -ds.yq[-1]+ds.yq[0]+(ds.yq[0]-ds.yq[1])]):
        ds = xr.concat(
            [
                (
                    ds.isel({dim:-1})
                    .assign_coords({dim:(ds[dim].isel({dim:-1}) + offset).values})
                ),
                ds
            ],
            dim=dim,
            data_vars="minimal"
        )
    
    ds = ds.transpose("yq", "yh", "xq", "xh", ...).chunk({'yq':-1, 'xq':-1})
    return ds