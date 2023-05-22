import xarray as xr
import numpy as np
import warnings

import regionate
import sectionate
from xwmt.wmt import WaterMassTransformations

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

    def mass_budget(self, lam):
        lambda_name = self.lambdas_dict[lam]
        self.wmt = self.transformations(lam)
        self.wmt['overturning'] = self.convergent_transport(lambda_name)
        self.wmt['mass_tendency'] = self.mass_tendency(lambda_name)
        self.numerical_mixing()
        return self.wmt
        
    def transformations(self, lam):
        return self.integrate_transformations(
            lam,
            bins=self.ds.thetao_i,
            mask=self.region.mask,
            group_processes=True,
            sum_components=True
        ).rename({f"{self.lambdas_dict[lam]}_i":f"{self.lambdas_dict[lam]}_l"})
        
    def convergent_transport(self, lambda_name):
        kwargs = {
            "utr": self.budgets_dict['mass']['transport']['X'],
            "vtr": self.budgets_dict['mass']['transport']['Y'],
            "layer": self.grid.axes["Z"].coords['center'],
            "interface": self.grid.axes["Z"].coords['outer'],
            "geometry": "spherical"
        }
        conv = sectionate.convergent_transport(
            self.grid,
            self.region.i,
            self.region.j,
            positive_in = self.region.mask,
            **kwargs
        )['conv_mass_transport']
        lambda_sect = sectionate.extract_tracer(
            lambda_name,
            self.grid,
            self.region.i,
            self.region.j,
        )
        
        self.ds['convergent_mass_transport'] = (
            self.grid.transform(
                conv.fillna(0.),
                "Z",
                target = self.ds[f"{lambda_name}_i"],
                target_data = lambda_sect,
                method="conservative"
            )
            .rename({f"{lambda_name}_i": f"{lambda_name}_l"})
        )
        
        self.ds['convergent_mass_transport_below'] = self.grid.cumsum(
            self.ds.convergent_mass_transport, "lam", boundary="fill", fill_value=0.
        ).chunk({f"{lambda_name}_i": -1})
        
        self.ds['overturning'] = self.grid.interp(
            self.ds['convergent_mass_transport_below'].sum("sect"),
            "lam"
        )
        
        return self.ds.overturning
        
    
    def mass_tendency(self, lambda_name):
        self.ds['mass_density'] = (
            self.grid.transform(
                self.rho_ref*self.ds[f"{self.h_name}_bounds"].fillna(0.),
                "Z",
                target = self.ds[f"{lambda_name}_i"],
                target_data = self.ds[f"{lambda_name}_bounds"],
                method="conservative"
            )
            .rename({f"{lambda_name}_i": f"{lambda_name}_l"})
        ) * self.region.mask
        
        self.ds['mass_density_below'] = self.grid.cumsum(
            self.ds.mass_density, "lam", boundary="fill", fill_value=0.
        ).chunk({f"{lambda_name}_i": -1})
        
        self.ds['mass_below'] = (
            self.ds.mass_density_below *
            self.grid.get_metric(self.ds[f"{lambda_name}_bounds"], ("X", "Y"))
        ).sum([
            self.grid.axes['X'].coords['center'],
            self.grid.axes['Y'].coords['center']
        ])
        
        self.ds['mass_tendency_below'] = self.grid.interp(
            (
                self.ds.mass_below.diff('time_bounds') /
                (self.ds.time_bounds.diff('time_bounds').astype('float')*1.e-9)
            ).rename({"time_bounds":"time"}).assign_coords({'time': self.ds.time}),
            "lam"
        )
        
        return self.ds.mass_tendency_below
    
    def numerical_mixing(self):
        self.wmt['numerical_mixing'] = self.wmt.advection  + self.wmt.overturning
        self.wmt['volume_discretization_error'] = (-self.wmt.mass_tendency) - self.wmt.total_tendency 
        self.wmt['numerical_errors'] = self.wmt['numerical_mixing'] + self.wmt['volume_discretization_error']
        
        return self.wmt.numerical_mixing