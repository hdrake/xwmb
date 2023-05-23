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
    
        if isinstance(region, regionate.GriddedRegion):
            self.region = region
        elif type(region) is tuple:
            if len(region)==2:
                lons, lats = region[0], region[1]
                self.region = regionate.GriddedRegion(
                    "WaterMass",
                    lons,
                    lats,
                    self.grid
                )
        elif isinstance(region, (xr.DataArray)):
            mask = region
            self.region = regionate.MaskRegions(
                mask,
                self.grid
            ).regions[0]
        elif region is None:
            pass

    def mass_budget(self, lam):
        lambda_name = self.get_lambda(lam)
        self.wmt = self.transformations(lam)
        self.convergent_transport(lambda_name)
        self.mass_tendency(lambda_name)
        self.numerical_mixing()
        return self.wmt
        
    def transformations(self, lam):
        trans = (
            self.integrate_transformations(
                lam,
                bins=self.ds[f"{self.get_lambda(lam)}_i"],
                mask=self.region.mask,
                group_processes=True,
                sum_components=True
            )
            .rename({f"{self.get_lambda(lam)}_i":f"{self.get_lambda(lam)}_l"})
            .assign_coords({f"{self.get_lambda(lam)}_l": self.ds[f"{self.get_lambda(lam)}_l"]})
        )
        return trans
        
    def convergent_transport(self, lambda_name):
        kwargs = {
            "utr": self.budgets_dict['mass']['transport']['X'],
            "vtr": self.budgets_dict['mass']['transport']['Y'],
            "layer": self.grid.axes["Z"].coords['center'],
            "interface": self.grid.axes["Z"].coords['outer'],
            "geometry": "spherical"
        }
        self.ds['conv'] = sectionate.convergent_transport(
            self.grid,
            self.region.i,
            self.region.j,
            positive_in = self.region.mask,
            **kwargs
        ).rename({"lat":"lat_sect", "lon":"lon_sect"})['conv_mass_transport']
        
        self.ds[f'{lambda_name}_sect'] = sectionate.extract_tracer(
            lambda_name,
            self.grid,
            self.region.i,
            self.region.j,
        )
        
        self.ds[f'{lambda_name}_i_sect'] = (
            self.grid.interp(self.ds[f"{lambda_name}_sect"], "Z", boundary="extend")
            .chunk({self.grid.axes['Z'].coords['outer']: -1})
            .rename(f'{lambda_name}_i_sect')
        )
        
        self.ds['convergent_mass_transport'] = (
            self.grid.transform(
                self.ds.conv.fillna(0.),
                "Z",
                target = self.ds[f"{lambda_name}_i"],
                target_data = self.ds[f'{lambda_name}_i_sect'],
                method="conservative"
            )
            .rename({f"{lambda_name}_i": f"{lambda_name}_l"})
            .assign_coords({f"{lambda_name}_l": self.ds[f"{lambda_name}_l"]})
        )
        
        self.ds['convergent_mass_transport_below'] = self.grid.cumsum(
            self.ds.convergent_mass_transport, "lam", boundary="fill", fill_value=0.
        ).chunk({f"{lambda_name}_i": -1})
        
        self.wmt['overturning'] = self.grid.interp(
            self.ds['convergent_mass_transport_below'].sum("sect"),
            "lam"
        )
        
        return self.wmt.overturning
        
    
    def mass_tendency(self, lambda_name):
        if "time_bounds" in self.ds.dims:
            self.ds[f"{lambda_name}_i_bounds"] = (
                self.grid.interp(self.ds[f"{lambda_name}_bounds"], "Z", boundary="extend")
                .chunk({self.grid.axes['Z'].coords['outer']: -1})
                .rename(f"{lambda_name}_i_bounds")
            )
            
            self.ds['mass_density'] = (
                self.grid.transform(
                    self.rho_ref*self.ds[f"{self.h_name}_bounds"].fillna(0.),
                    "Z",
                    target = self.ds[f"{lambda_name}_i"],
                    target_data = self.ds[f"{lambda_name}_i_bounds"],
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

            self.wmt['mass_tendency_below'] = self.grid.interp(
                (
                    self.ds.mass_below.diff('time_bounds') /
                    (self.ds.time_bounds.diff('time_bounds').astype('float')*1.e-9)
                ).rename({"time_bounds":"time"}).assign_coords({'time': self.ds.time}),
                "lam"
            )
            return self.wmt.mass_tendency_below

        else:
            return
        
    
    def numerical_mixing(self):
        if ("advection" in self.wmt) and ("overturning" in self.wmt):
            self.wmt['numerical_mixing'] = self.wmt.advection  + self.wmt.overturning
        if ("total_tendency" in self.wmt) and ("mass_tendency" in self.wmt):
            self.wmt['volume_discretization_error'] = (-self.wmt.mass_tendency) - self.wmt.total_tendency 
        if ("numerical_mixing" in self.wmt) and ("volume_discretization_error" in self.wmt):
            self.wmt['numerical_errors'] = self.wmt['numerical_mixing'] + self.wmt['volume_discretization_error']