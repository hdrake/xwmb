
def calc_dMdt_theta(wmb, ds, snap, grid_snap, ocean_grid, mask, theta_i_bins, rho0=1035.):
    
    snap['thetao_i'] = grid_snap.transform(snap.thetao, 'Z', snap.z_i, target_data=snap.z_l, method="linear")
    
    # Transform snapshot mass onto snapshot temperature grid
    m_on_temp = grid_snap.transform(
        rho0*snap.thkcello,
        'Z',
        target=theta_i_bins,
        target_data=snap.thetao_i.chunk(dict(z_i=-1)),
        method='conservative'
    )

    # Integrate over surface area
    M_on_temp = (m_on_temp*ocean_grid.areacello).where(mask).sum(['xh', 'yh']).compute()

    # Integrate 
    M_below_temp = M_on_temp.cumsum('thetao_i')

    # Finite difference over time_bnds of monthly-averaged tendencies
    dMdt = M_below_temp.diff('time')/(M_below_temp.time.diff('time').astype('float')*1.e-9)

    # Assign the coordinates to align with G (centre of the averaging period)
    wmb["dMdt"] = dMdt.assign_coords({'time':ds.time}).compute()