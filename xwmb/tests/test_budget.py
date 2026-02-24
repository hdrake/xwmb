import xgcm
import xwmb
import xarray as xr
import numpy as np

def synthetic_dataset():
    x_f = np.array([-0.5, 0.5])
    x_c = 0.5*(x_f[:-1] + x_f[1:])

    y_f = np.array([0.5, 1.5, 2.5])
    y_c = 0.5*(y_f[:-1] + y_f[1:])

    lam_f = np.array([0, 1, 2])
    lam_c = 0.5*(lam_f[:-1] + lam_f[1:])

    t_f = np.array([0, 1])
    t_c = 0.5*(t_f[:-1] + t_f[1:])

    coords = {
        "x_c": x_c,
        "x_f": x_f,
        "y_c": y_c,
        "y_f": y_f,
        "lam_c": lam_c,
        "lam_f": lam_f,
        "t_c": t_c,
        "t_f": t_f
    }
    ds = xr.Dataset(coords=coords)
    ds = ds.assign_coords({
        "geolon": xr.broadcast(ds.x_c, ds.y_c)[0],
        "geolat": xr.broadcast(ds.x_c, ds.y_c)[0],
        "geolon_c": xr.broadcast(ds.x_f, ds.y_f)[0],
        "geolat_c": xr.broadcast(ds.x_f, ds.y_f)[1],
    })

    # Grid cell area
    ds["area"] = xr.ones_like(xr.broadcast(ds.x_c, ds.y_c)[0])

    # Time-averaged size and contours of water mass
    ds["lam"] = ds.lam_c * xr.ones_like(xr.broadcast(ds.t_c, ds.lam_c, ds.y_c, ds.x_c)[0])
    ds["thickness"] = xr.ones_like(ds.lam)

    # Bounding snapshots of size and contours of water mass
    ds["lam_bounds"] = ds.lam_c * xr.ones_like(xr.broadcast(ds.t_f, ds.lam_c, ds.y_c, ds.x_c)[0])
    ds["thickness_bounds"] = xr.ones_like(ds.lam_bounds)

    # Lateral mass transport
    ds["umo"] = xr.zeros_like(xr.broadcast(ds.t_c, ds.lam_c, ds.y_c, ds.x_f)[0])
    ds["vmo"] = xr.where(
        ds.y_f == 1.5,
        ds.lam_c - 1.,
        xr.zeros_like(xr.broadcast(ds.t_c, ds.lam_c, ds.y_f, ds.x_c)[0])
    )

    # Volume-integrated tendency
    ds["tend"] = xr.where(
        ds.y_c == 2.0,
        1.,
        xr.zeros_like(xr.broadcast(ds.t_c, ds.lam_c, ds.y_c, ds.x_c)[0])
    )
    return ds

def synthetic_grid():

    ds = synthetic_dataset()

    # Placeholder until https://github.com/hdrake/xbudget/issues/21
    ds = ds.rename({"t_c":"time", "t_f":"time_bounds"})

    coords = {
        "X": {"center":"x_c", "outer":"x_f"},
        "Y": {"center":"y_c", "outer":"y_f"},
        "Z": {"center":"lam_c", "outer":"lam_f"},
        "T": {"center":"time", "outer":"time_bounds"}
    }
    grid = xgcm.Grid(
        ds,
        coords = coords,
        boundary = {"X": "extend", "Y":"extend", "Z":"extend", "T":"extend"},
        metrics = {("X","Y"): "area"},
        autoparse_metadata=False
    )

    return grid

def test_mass_budget():
    grid = synthetic_grid()

    xbudget_dict = {
        "mass": {
            "thickness": "thickness",
            "rhs": {"sum": {"advection": {"sum": {"lateral": {"sum": {
                "zonal_convergence": {"product": {"zonal_divergence": {"difference": {"zonal_mass_transport": "umo"}}}},
                "meridional_convergence": {"product": {"meridional_divergence": {"difference": {"meridional_mass_transport": "vmo"}}}}
            }}}}}}
        },
        "tracer": {"lambda": "lam", "rhs": {"sum": {"tendency": {"var": "tend"}}}}
    }

    wmb = xwmb.WaterMassBudget(
        grid,
        xbudget_dict,
        rebin=False,
        rho_ref = 1.
    )

    wmb.mass_budget("tracer", bins=grid._ds.lam_f).compute()