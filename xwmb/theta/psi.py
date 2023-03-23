import xarray as xr
import numpy as np

import sectionate as sec

def calc_psi_theta(wmb, ds, grid, i, j, theta_i_bins, symmetric):
    extend_bottom_theta(ds, grid)
    
    uvmo_conv_sec = sec.MOM6_convergent_transport(ds, i, j, symmetric=symmetric)['uvnormal'].compute()
    uvmo_conv_sec = uvmo_conv_sec.assign_coords({'sect': np.arange(0, uvmo_conv_sec.sect.size)})
    thetao_sec = sec.MOM6_extract_hydro(ds.thetao_i_extended, i, j).compute()
    
    wmb['uvmo_conv'] = grid.transform(
        uvmo_conv_sec.fillna(0.), 'Z', theta_i_bins, target_data=thetao_sec.rename('thetao_i'), method="conservative"
    ).cumsum('thetao_i').compute()
    wmb['Psi'] = wmb['uvmo_conv'].sum("sect").compute()
    
def extend_bottom_theta(ds, grid):
    # Interpolate theta to from cell center to cell interfaces (and extend to sea surface)
    ds['thetao_i_extended'] = grid.transform(ds.thetao, 'Z', ds.z_i, target_data=ds.z_l, method="linear", mask_edges=False)
    
    # Estimate temperature at seafloor interface (extend bottom value by; equivalent to no-flux vertical bottom boundary condition)
    def zsel(da, idx): return da[idx]

    def zsel_above(da, idx): return da[idx-1]
    
    ds['bottom_interface_depth'] = xr.apply_ufunc(
        zsel, ds.z_i, np.isnan(ds.thetao_i_extended).argmax('z_i'),
        input_core_dims=[['z_i'], []], dask='parallelized', vectorize=True, output_dtypes=[float]
    )
    ds['bottom_thetao'] = xr.apply_ufunc(
        zsel_above, ds.thetao, np.isnan(ds.thetao_i_extended).argmax('z_i'),
        input_core_dims=[['z_l'], []], dask='parallelized', vectorize=True, output_dtypes=[float]
    )
    ds['thetao_i_extended'] = xr.where(ds.z_i==ds.bottom_interface_depth, ds.bottom_thetao, ds.thetao_i_extended)