
def calc_dMdt_theta(wmb, ds, snap, grid_snap, ocean_grid, mask, theta_i_bins, rho0=1035.):
        
    # Transform snapshot mass onto snapshot temperature grid
    m_on_temp = grid_snap.transform(
        rho0*snap.thkcello,
        'Z',
        target=theta_i_bins,
        target_data=snap.thetao,
        method='conservative'
    )

    # Integrate mass in each temperature layer over the horizontal surface area
    M_on_temp = (m_on_temp*ocean_grid.areacello).where(mask).sum(['xh', 'yh']).compute()
    wmb['M0'] = M_on_temp.isel(time=slice(0, -1)).assign_coords({'time':ds.time})

    # Compute the cumulative integral below each of the temperature interfaces (theta_i_bins),
    # but omit the value for the last interface because there is no equivalent watermass transformation term
    M_below_temp = M_on_temp.cumsum('thetao').rename({'thetao':'thetao_i'}).assign_coords({'thetao_i':theta_i_bins[1:]}).isel(thetao_i=slice(0, -1))

    # Finite difference over time_bnds of the time-averaged tendencies
    dMdt = M_below_temp.diff('time')/(M_below_temp.time.diff('time').astype('float')*1.e-9)

    # Assign the coordinates to align with G (centre of the averaging period)
    wmb["dMdt"] = dMdt.assign_coords({'time':ds.time}).compute()