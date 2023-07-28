import xarray as xr
import numpy as np
import warnings

import regionate
import sectionate

from xwmt.wmt import WaterMassTransformations
from xgcm import Grid
import xbudget

class WaterMassBudget(WaterMassTransformations):
    def __init__(
        self,
        grid,
        budgets_dict,
        region,
        decompose=[],
        teos10=True,
        rho_ref=1035.,
        cp=3992.,
        assert_zero_transport=False
        ):

        super().__init__(
            grid,
            xbudget.aggregate(budgets_dict, decompose=decompose),
            teos10=teos10,
            cp=cp,
            rho_ref=rho_ref
        )
        self.full_budgets_dict = budgets_dict
        self.assert_zero_transport = assert_zero_transport
    
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

    def mass_budget(self, lam, lam_coords=None):
        lambda_name = self.get_lambda_var(lam)
        if lam_coords is None:
            lam_coords = self.lam_coord_defaults(lam)
        self.lam_coords = lam_coords

        self.wmt = self.transformations(lam)
        if self.get_lambda_key(lam) == "density":
            for v in self.wmt.data_vars:
                self.wmt[v] *= -1
                
        self.mass_tendency(lambda_name)
        self.convergent_transport(lambda_name)
        self.numerical_mixing()
        return self.wmt
        
    def transformations(self, lam):
        lambda_name = self.get_lambda_var(lam)
        wmt = (
            self.integrate_transformations(
                lam,
                bins=self.ds[self.lam_coords["outer"]],
                mask=self.region.mask,
                group_processes=True,
                sum_components=True
            )
            .rename({self.lam_coords["outer"] : self.lam_coords["center"]})
            .assign_coords({self.lam_coords["center"]: self.ds[self.lam_coords["center"]]})
        )
        
        return wmt
        
    def convergent_transport(self, lambda_name):
        kwargs = {
            "layer": self.grid.axes["Z"].coords['center'],
            "interface": self.grid.axes["Z"].coords['outer'],
            "geometry": "spherical"
        }
        if not self.assert_zero_transport:
            kwargs = {**kwargs, **{
                "utr": self.full_budgets_dict['mass']['transport']['X'],
                "vtr": self.full_budgets_dict['mass']['transport']['Y']
            }}
        
            # Compute horizontal boundary flux term
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
                    target = self.ds[self.lam_coords["outer"]],
                    target_data = self.ds[f'{lambda_name}_i_sect'],
                    method="conservative"
                )
                .rename({self.lam_coords["outer"]: self.lam_coords["center"]})
                .assign_coords({self.lam_coords["center"]: self.ds[self.lam_coords["center"]]})
            )
        
            lam_grid = Grid(self.ds, coords={'lam': self.lam_coords}, periodic=None)
            if "sigma" not in lambda_name:
                self.ds['convergent_mass_transport_below'] = lam_grid.cumsum(
                    self.ds.convergent_mass_transport, "lam", boundary="fill", fill_value=0.
                ).chunk({self.lam_coords["outer"]: -1})
            else:
                self.ds = self.ds.sel({
                    self.lam_coords["outer"]: self.ds[self.lam_coords["outer"]][::-1],
                    self.lam_coords["center"]: self.ds[self.lam_coords["center"]][::-1],
                })
                lam_rev_grid = Grid(self.ds, coords={'lam': self.lam_coords}, periodic=None)
                self.ds['convergent_mass_transport_below'] = lam_rev_grid.cumsum(
                    self.ds.convergent_mass_transport, "lam", boundary="fill", fill_value=0.
                ).chunk({self.lam_coords["outer"]: -1})
                self.ds = self.ds.sel({
                    self.lam_coords["outer"]: self.ds[self.lam_coords["outer"]][::-1],
                    self.lam_coords["center"]: self.ds[self.lam_coords["center"]][::-1],
                })

        # Compute mass source term
        lam_i = (
            self.grid.interp(self.ds[f"{lambda_name}"], "Z", boundary="extend")
            .chunk({self.grid.axes['Z'].coords['outer']: -1})
            .rename(f"{lambda_name}_i")
        )
        
        mass_flux = xbudget.get_vars(self.full_budgets_dict, "mass_rhs_surface_exchange_flux")
        self.ds['mass_source_density'] = (
            self.grid.transform(
                self.ds[mass_flux].fillna(0.),
                "Z",
                target = self.ds[self.lam_coords["outer"]],
                target_data = lam_i,
                method="conservative"
            )
            .rename({self.lam_coords["outer"]: self.lam_coords["center"]})
            .assign_coords({self.lam_coords["center"]: self.ds[self.lam_coords["center"]]})
        ) * self.region.mask

        lam_grid = Grid(self.ds, coords={'lam': self.lam_coords}, periodic=None)
        if "sigma" not in lambda_name:
            self.ds['mass_source_density_below'] = lam_grid.cumsum(
                self.ds.mass_source_density, "lam", boundary="fill", fill_value=0.
            ).chunk({self.lam_coords["outer"]: -1})
        else:
            self.ds = self.ds.sel({
                self.lam_coords["outer"]: self.ds[self.lam_coords["outer"]][::-1],
                self.lam_coords["center"]: self.ds[self.lam_coords["center"]][::-1],
            })
            lam_rev_grid = Grid(self.ds, coords={'lam': self.lam_coords}, periodic=None)
            self.ds['mass_source_density_below'] = lam_rev_grid.cumsum(
                self.ds.mass_source_density, "lam", boundary="fill", fill_value=0.
            ).chunk({self.lam_coords["outer"]: -1})
            self.ds = self.ds.sel({
                self.lam_coords["outer"]: self.ds[self.lam_coords["outer"]][::-1],
                self.lam_coords["center"]: self.ds[self.lam_coords["center"]][::-1],
            })

        self.ds['mass_source_below'] = (
            self.ds.mass_source_density_below *
            self.grid.get_metric(self.ds[f"{lambda_name}"], ("X", "Y"))
        ).sum([
            self.grid.axes['X'].coords['center'],
            self.grid.axes['Y'].coords['center']
        ])
        
        self.wmt['mass_source'] = lam_grid.interp(
            self.ds.mass_source_below,
            "lam"
        )
        
        # interpolate terms onto tracer levels
        lam_grid = Grid(self.ds, coords={'lam': self.lam_coords}, periodic=None)
        self.wmt['mass_source'] = lam_grid.interp(
            self.ds['mass_source_below'],
            "lam",
            boundary="extend"
        )

        if not self.assert_zero_transport:
            self.wmt['overturning'] = lam_grid.interp(
                self.ds['convergent_mass_transport_below'],
                "lam",
                boundary="extend"
            ).sum("sect")
        else:
            self.wmt['overturning'] = xr.zeros_like(self.wmt['mass_source'])
        
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
                    target = self.ds[self.lam_coords["outer"]],
                    target_data = self.ds[f"{lambda_name}_i_bounds"],
                    method="conservative"
                )
                .rename({self.lam_coords["outer"]: self.lam_coords["center"]})
                .assign_coords({self.lam_coords["center"]: self.ds[self.lam_coords["center"]]})
            ) * self.region.mask

            lam_grid = Grid(self.ds, coords={'lam': self.lam_coords}, periodic=None)
            if "sigma" not in lambda_name:
                self.ds['mass_density_below'] = lam_grid.cumsum(
                    self.ds.mass_density, "lam", boundary="fill", fill_value=0.
                ).chunk({self.lam_coords["outer"]: -1})
            else:
                self.ds = self.ds.sel({
                    self.lam_coords["outer"]: self.ds[self.lam_coords["outer"]][::-1],
                    self.lam_coords["center"]: self.ds[self.lam_coords["center"]][::-1],
                })
                lam_rev_grid = Grid(self.ds, coords={'lam': self.lam_coords}, periodic=None)
                self.ds['mass_density_below'] = lam_rev_grid.cumsum(
                    self.ds.mass_density, "lam", boundary="fill", fill_value=0.
                ).chunk({self.lam_coords["outer"]: -1})
                self.ds = self.ds.sel({
                    self.lam_coords["outer"]: self.ds[self.lam_coords["outer"]][::-1],
                    self.lam_coords["center"]: self.ds[self.lam_coords["center"]][::-1],
                })

            self.ds['mass_below'] = (
                self.ds.mass_density_below *
                self.grid.get_metric(self.ds[f"{lambda_name}_bounds"], ("X", "Y"))
            ).sum([
                self.grid.axes['X'].coords['center'],
                self.grid.axes['Y'].coords['center']
            ])

            dt = self.ds.time_bounds.diff('time_bounds').astype('float')*1.e-9
            self.wmt['mass_tendency'] = lam_grid.interp(
                (self.ds.mass_below.diff('time_bounds') / dt)
                .rename({"time_bounds":"time"}).assign_coords({'time': self.ds.time}),
                "lam"
            )
            self.wmt['dt'] = dt.rename({"time_bounds":"time"}).assign_coords({'time':self.ds['time']})
            return self.wmt.mass_tendency

        else:
            return
        
    
    def numerical_mixing(self):
        numerical_mixing_requirements = [
            "mass_tendency",
            "mass_source",
            "overturning",
            "process_material_derivative"
        ]
            
        if all([term in self.wmt for term in numerical_mixing_requirements]):
            self.wmt['numerical_mixing'] = (
                -(self.wmt.mass_tendency -
                  self.wmt.mass_source -
                  self.wmt.overturning)
                - self.wmt.process_material_derivative
            )
        
    def lam_coord_defaults(self, lam):
        lambda_name = self.get_lambda_var(lam)
        if lam=="sigma2":
            lam_min, lam_max, dlam = 0., 50., 0.1
            
        elif lam=="heat":
            lam_min, lam_max, dlam = -4, 40., 0.1

        elif lam=="salt":
            lam_min, lam_max, dlam = -1., 50., 0.1

        self.ds = self.ds.assign_coords({
            f"{lambda_name}_l" : np.arange(lam_min, lam_max, dlam),
            f"{lambda_name}_i" : np.arange(lam_min-dlam/2., lam_max+dlam/2, dlam),
        })
        return {"outer": f"{lambda_name}_i", "center": f"{lambda_name}_l"}