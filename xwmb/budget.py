import xarray as xr
import numpy as np
import warnings

import regionate
from xwmt.wmt import WaterMassTransformations

from .transformation import *
from .mass import *
from .transport import *

class WaterMassBudget(WaterMassTransformations):
    def __init__(
        self,
        grid,
        budgets_dict,
        region,
        teos10=True,
        rho_ref=1035.,
        cp=3992.,
        ):
        super().__init__(
            grid,
            budgets_dict,
            teos10=teos10,
            cp=cp,
            rho_ref=rho_ref
        )
    
        if isinstance(region, regionate.region.GriddedRegion):
            self.region = region
        elif type(region) is tuple:
            if len(region)==2:
                lons, lats = region[0], region[1]
                self.region = regionate.region.GriddedRegion(
                    "WaterMass",
                    lons,
                    lats,
                    self.grid
                )
        elif isinstance(region, (xr.DataArray)):
            mask = region
            self.region = regionate.regions.MaskRegions(
                mask,
                self.grid
            ).regions[0]
        elif region is None:
            pass

    def diagnose_mass_budget(self):
        pass
        
    def convergent_transport(self):
        pass
    
    def mass_tendency(self, lambda_name):
        self.ds['mass_density'] = (
            self.grid.transform(
                self.rho_ref*self.ds[f"{self.h_name}_bounds"].fillna(0.),
                "Z",
                target = self.ds[f"{lambda_name}_i"].values,
                target_data = self.ds[f"{lambda_name}_bounds"],
                method="conservative"
            )
            .rename({f"{lambda_name}_bounds": f"{lambda_name}_l"})
        ) * self.region.mask
        
        self.ds['mass_density_below'] = self.grid.cumsum(
            self.ds.mass_density, "lam", boundary="fill", fill_value=0.
        )
        
        self.ds['mass_below'] = (
            self.ds.mass_density_below *
            self.grid.get_metric(self.ds[f"{lambda_name}_bounds"], ("X", "Y"))
        ).sum([
            self.grid.axes['X'].coords['center'],
            self.grid.axes['Y'].coords['center']
        ])
        
        self.ds['mass_tendency_below'] = (
            self.ds.mass_below.diff('time_bounds') /
            (self.ds.time_bounds.diff('time_bounds').astype('float')*1.e-9)
        ).rename({"time_bounds":"time"}).assign_coords({'time': self.ds.time})
        
        return self.ds.mass_tendency_below

    #calc_water_mass_transformations(wmb, ds, ocean_grid, bins, mask=mask, rho_ref=rho_ref, cp=cp)
    #calc_mass_tendency(wmb, ds, snap, grid_snap, ocean_grid, wmb.basin_mask, theta_i_bins, rho_ref=rho_ref)
    #calc_convergent_transport(wmb, ds, grid, i, j, theta_i_bins, ocean_grid['geolon'].shape!=ocean_grid['geolon_c'].shape)

    # Discretization error/residual terms
    #wmb['numerical_mixing'] =   wmb.advection  + wmb.psi
    #wmb['volume_discretization_error'] = - wmb.Eulerian_tendency - wmb.dMdt
    #wmb['numerical_errors'] = wmb['numerical_mixing'] + wmb['volume_discretization_error']