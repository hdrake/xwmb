import xarray as xr
import numpy as np

import sectionate as sec

def calc_psi_theta(wmb, ds, grid, i, j, theta_i_bins, symmetric):
    kwargs = {"layer": grid.axes["Z"].coords['center'], 'interface':grid.axes["Z"].coords['outer']}
    conv = sec.convergent_transport(ds, i, j, symmetric, **kwargs)['conv_mass_transport'].compute()
    thetao_sec = sec.extract_tracer(ds.thetao, i, j, symmetric).compute()
    
    # Compute the convergent horizontal transport integrated below each of the temperature interfaces (theta_i_bins),
    # but omit the value for the last interface because there is no equivalent watermass transformation term
    wmb['conv_transport'] = grid.transform(
        conv.fillna(0.), 'Z', theta_i_bins, target_data=thetao_sec, method="conservative"
    ).cumsum('thetao').rename({'thetao':'thetao_i'}).assign_coords({'thetao_i':theta_i_bins[1:]}).isel(thetao_i=slice(0, -1)).compute()
    wmb['psi'] = wmb['conv_transport'].sum("sect").compute()