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
        assert_zero_transport=False,
        method="default",
        rebin=False
        ):

        super().__init__(
            grid,
            xbudget.aggregate(budgets_dict, decompose=decompose),
            teos10=teos10,
            cp=cp,
            rho_ref=rho_ref,
            method=method,
            rebin=rebin
        )
        self.full_budgets_dict = budgets_dict
        self.assert_zero_transport = assert_zero_transport
        self.boundary = {ax:self.grid.axes[ax]._boundary for ax in self.grid.axes.keys()}
    
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

    def mass_budget(self, lam, greater_than=False, default_bins=False):
        lambda_name = self.get_lambda_var(lam)
        if "target_coords" not in list(vars(self)):
            self.target_coords = {"center": f"{lambda_name}_l_target", "outer": f"{lambda_name}_i_target"}
            if default_bins:
                self.target_coords = self.lam_coord_defaults(lam)
            else:
                if all([c.replace("_target", "") in self.grid._ds for c in self.target_coords.values()]):
                    self.grid._ds = self.grid._ds.assign_coords({
                        f"{lambda_name}_l_target": xr.DataArray(
                            self.grid._ds[f"{lambda_name}_l"].values, dims=(f"{lambda_name}_l_target",)
                        ),
                        f"{lambda_name}_i_target": xr.DataArray(
                            self.grid._ds[f"{lambda_name}_i"].values, dims=(f"{lambda_name}_i_target",)
                        )
                    })
                else:
                    raise ValueError(
                        """Requires one of the following:
                             1) 'f{lambda_name}_i' and 'f{lambda_name}_l' in WaterMassBudget.grid._ds,
                             2) specify WaterMassBudget.target_coords, or
                             3) set `default_bins=True`.""")

        self.wmt = self.transformations(lam, greater_than=greater_than)
        self.mass_tendency(lambda_name, greater_than=greater_than)
        self.convergent_transport(lambda_name, greater_than=greater_than)
        self.numerical_mixing()
        return self.wmt
        
    def transformations(self, lam, greater_than=False):
        lambda_name = self.get_lambda_var(lam)
        wmt = self.integrate_transformations(
            lam,
            bins=self.grid._ds[self.target_coords["outer"]],
            mask=self.region.mask,
            group_processes=True,
            sum_components=True
        ).assign_coords({
            self.target_coords["center"]: self.grid._ds[self.target_coords["center"]],
            self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]],
        })
        
        # Because vector normal to isosurface switches sign, we need to flip water mass transformation terms
        # Alternatively, I think we could have just switched the ordering of the bins so that \Delta \lambda flips.
        if greater_than:
            for v in wmt.data_vars:
                wmt[v] *= -1
                
        boundary_flux_terms = [
            'surface_exchange_flux',
            'surface_ocean_flux_advective_negative_rhs',
            'bottom_flux',
            'frazil_ice'
        ]
        if all([term in wmt for term in boundary_flux_terms]):
            wmt['boundary_fluxes'] = sum([wmt[term] for term in boundary_flux_terms if term in wmt])
        
        return wmt
        
    def convergent_transport(self, lambda_name, greater_than=False):
        kwargs = {
            "layer":     self.grid.axes["Z"].coords['center'],
            "interface": self.grid.axes["Z"].coords['outer'],
            "geometry":  "spherical"
        }
        if not self.assert_zero_transport:
            kwargs = {**kwargs, **{
                "utr": self.full_budgets_dict['mass']['transport']['X'],
                "vtr": self.full_budgets_dict['mass']['transport']['Y']
            }}
        
            # Compute horizontal boundary flux term
            self.grid._ds['conv'] = sectionate.convergent_transport(
                self.grid,
                self.region.i,
                self.region.j,
                positive_in = self.region.mask,
                **kwargs
            ).rename({"lat":"lat_sect", "lon":"lon_sect"})['conv_mass_transport']

            self.grid._ds[f'{lambda_name}_sect'] = sectionate.extract_tracer(
                lambda_name,
                self.grid,
                self.region.i,
                self.region.j,
            )

            self.grid._ds[f'{lambda_name}_i_sect'] = (
                self.grid.interp(self.grid._ds[f"{lambda_name}_sect"], "Z", boundary="extend")
                .chunk({self.grid.axes['Z'].coords['outer']: -1})
                .rename(f'{lambda_name}_i_sect')
            )

            self.grid._ds['convergent_mass_transport'] = (
                self.grid.transform(
                    self.grid._ds.conv.fillna(0.),
                    "Z",
                    target = self.grid._ds[self.target_coords["outer"]],
                    target_data = self.grid._ds[f'{lambda_name}_i_sect'],
                    method="conservative"
                )
                .rename({self.target_coords["outer"]: self.target_coords["center"]})
                .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]})
            )
            
            suffix = 'greater_than' if greater_than else 'less_than'
            lam_grid = Grid(
                self.grid._ds,
                coords={'lam': self.target_coords},
                boundary={'lam': 'extend'},
                autoparse_metadata=False
            )
            if greater_than:
                self.grid._ds = self.grid._ds.sel({
                    self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]][::-1],
                    self.target_coords["center"]: self.grid._ds[self.target_coords["center"]][::-1],
                })
                lam_rev_grid = Grid(
                    self.grid._ds,
                    coords={'lam': self.target_coords},
                    boundary={'lam': 'extend'},
                    autoparse_metadata=False
                )
                self.grid._ds[f'convergent_mass_transport_{suffix}'] = lam_rev_grid.cumsum(
                    self.grid._ds.convergent_mass_transport, "lam", boundary="fill", fill_value=0.
                ).chunk({self.target_coords["outer"]: -1})
                self.grid._ds = self.grid._ds.sel({
                    self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]][::-1],
                    self.target_coords["center"]: self.grid._ds[self.target_coords["center"]][::-1],
                })
            else:
                self.grid._ds[f'convergent_mass_transport_{suffix}'] = lam_grid.cumsum(
                    self.grid._ds.convergent_mass_transport, "lam", boundary="fill", fill_value=0.
                ).chunk({self.target_coords["outer"]: -1}).assign_coords(
                    {self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]]}
                )

        # Compute mass source term
        lam_i = (
            self.grid.interp(self.grid._ds[f"{lambda_name}"], "Z", boundary="extend")
            .chunk({self.grid.axes['Z'].coords['outer']: -1})
            .rename(f"{lambda_name}_i")
        )
        
        mass_flux_varname = "mass_rhs_sum_surface_exchange_flux"
        self.grid._ds['mass_source_density'] = (
            self.grid.transform(
                self.grid._ds[mass_flux_varname].fillna(0.),
                "Z",
                target = self.grid._ds[self.target_coords["outer"]],
                target_data = lam_i,
                method="conservative"
            )
            .rename({self.target_coords["outer"]: self.target_coords["center"]})
            .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]})
        ) * self.region.mask
        
        suffix = 'greater_than' if greater_than else 'less_than'
        lam_grid = Grid(
            self.grid._ds,
            coords={'lam': self.target_coords},
            boundary={'lam': 'extend'},
            autoparse_metadata=False
        )
        if greater_than:
            self.grid._ds = self.grid._ds.sel({
                self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]][::-1],
                self.target_coords["center"]: self.grid._ds[self.target_coords["center"]][::-1],
            })
            lam_rev_grid = Grid(
                self.grid._ds,
                coords={'lam': self.target_coords},
                boundary={'lam': 'extend'},
                autoparse_metadata=False
            )
            self.grid._ds[f'mass_source_density_{suffix}'] = lam_rev_grid.cumsum(
                self.grid._ds.mass_source_density, "lam", boundary="fill", fill_value=0.
            ).chunk({self.target_coords["outer"]: -1})
            self.grid._ds = self.grid._ds.sel({
                self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]][::-1],
                self.target_coords["center"]: self.grid._ds[self.target_coords["center"]][::-1],
            })
        else:
            self.grid._ds[f'mass_source_density_{suffix}'] = lam_grid.cumsum(
                self.grid._ds.mass_source_density, "lam", boundary="fill", fill_value=0.
            ).chunk({self.target_coords["outer"]: -1})

        self.grid._ds[f'mass_source_{suffix}'] = (
            self.grid._ds[f'mass_source_density_{suffix}']*
            self.grid.get_metric(self.grid._ds[f"{lambda_name}"], ("X", "Y"))
        ).sum([
            self.grid.axes['X'].coords['center'],
            self.grid.axes['Y'].coords['center']
        ])
        
        # Compute layer mass
        self.grid._ds['mass_density'] = (self.grid.transform(
                self.rho_ref*self.grid._ds[self.h_name].fillna(0.),
                "Z",
                target = self.grid._ds[self.target_coords["outer"]],
                target_data = lam_i,
                method="conservative"
            )
            .rename({self.target_coords["outer"]: self.target_coords["center"]})
            .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]})
        ) * self.region.mask
        
        self.wmt['layer_mass'] = (
            self.grid._ds['mass_density'] *
            self.grid.get_metric(self.grid._ds['mass_density'], ("X", "Y"))
        ).sum([
            self.grid.axes['X'].coords['center'],
            self.grid.axes['Y'].coords['center']
        ])
        
        # interpolate terms onto tracer levels
        lam_grid = Grid(
            self.grid._ds,
            coords={'lam': self.target_coords},
            boundary={'lam': 'extend'},
            autoparse_metadata=False
        )
        self.wmt['mass_source'] = lam_grid.interp(
            self.grid._ds[f'mass_source_{suffix}'],
            "lam",
            boundary="extend"
        ).assign_coords(
            {self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]}
        )

        if not self.assert_zero_transport:
            self.wmt['overturning'] = lam_grid.interp(
                self.grid._ds[f'convergent_mass_transport_{suffix}'],
                "lam",
                boundary="extend"
            ).sum("sect").assign_coords(
                {self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]}
            )
        else:
            self.wmt['overturning'] = xr.zeros_like(self.wmt['mass_source'])
        
        return self.wmt.overturning
        
    
    def mass_tendency(self, lambda_name, greater_than=False):
        ax = "Z" if "Z_bounds" not in self.grid.axes else "Z_bounds"
                
        if "time_bounds" in self.grid._ds.dims:
            self.grid._ds[f"{lambda_name}_i_bounds"] = (
                self.grid.interp(self.grid._ds[f"{lambda_name}_bounds"], ax, boundary="extend")
                .chunk({self.grid.axes[ax].coords['outer']: -1})
                .rename(f"{lambda_name}_i_bounds")
            )
            
            self.grid._ds['mass_density_bounds'] = (
                self.grid.transform(
                    self.rho_ref*self.grid._ds[f"{self.h_name}_bounds"].fillna(0.),
                    ax,
                    target = self.grid._ds[self.target_coords["outer"]],
                    target_data = self.grid._ds[f"{lambda_name}_i_bounds"],
                    method="conservative"
                )
                .rename({self.target_coords["outer"]: self.target_coords["center"]})
                .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]})
            ) * self.region.mask

            suffix = 'greater_than' if greater_than else 'less_than'
            lam_grid = Grid(
                self.grid._ds,
                coords={'lam': self.target_coords},
                boundary={'lam': 'extend'},
                autoparse_metadata=False
            )
            if greater_than:
                self.grid._ds = self.grid._ds.sel({
                    self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]][::-1],
                    self.target_coords["center"]: self.grid._ds[self.target_coords["center"]][::-1],
                })
                lam_rev_grid = Grid(
                    self.grid._ds,
                    coords={'lam': self.target_coords},
                    boundary={'lam': 'extend'},
                    autoparse_metadata=False
                )
                self.grid._ds[f'mass_density_bounds_{suffix}'] = lam_rev_grid.cumsum(
                    self.grid._ds.mass_density_bounds, "lam", boundary="fill", fill_value=0.
                ).chunk({self.target_coords["outer"]: -1})
                self.grid._ds = self.grid._ds.sel({
                    self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]][::-1],
                    self.target_coords["center"]: self.grid._ds[self.target_coords["center"]][::-1],
                })
            else:
                self.grid._ds[f'mass_density_bounds_{suffix}'] = lam_grid.cumsum(
                    self.grid._ds.mass_density_bounds, "lam", boundary="fill", fill_value=0.
                ).chunk({self.target_coords["outer"]: -1}).assign_coords(
                    {self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]]}
                )

            self.grid._ds[f'mass_bounds_{suffix}'] = (
                self.grid._ds[f'mass_density_bounds_{suffix}'] *
                self.grid.get_metric(self.grid._ds[f"{lambda_name}_bounds"], ("X", "Y"))
            ).sum([
                self.grid.axes['X'].coords['center'],
                self.grid.axes['Y'].coords['center']
            ])

            dt = self.grid._ds.time_bounds.diff('time_bounds').astype('float')*1.e-9
            self.wmt['mass_tendency'] = lam_grid.interp(
                (self.grid._ds[f'mass_bounds_{suffix}'].diff('time_bounds') / dt)
                .rename({"time_bounds":"time"}).assign_coords({'time': self.grid._ds.time}),
                "lam"
            ).assign_coords(
                {self.target_coords["center"]: self.grid._ds[self.target_coords["center"]].values}
            )
            self.wmt['dt'] = dt.rename({"time_bounds":"time"}).assign_coords({'time':self.grid._ds['time']})
            self.wmt = self.wmt.assign_coords({"time_bounds": self.grid._ds["time_bounds"]})
            return self.wmt.mass_tendency

        else:
            return
        
    
    def numerical_mixing(self):
        Leibniz_material_derivative_terms = [
            "mass_tendency",
            "mass_source",
            "overturning",
        ]   
        if all([term in self.wmt for term in Leibniz_material_derivative_terms]):
            self.wmt["Leibniz_material_derivative"] = - (
                self.wmt.mass_tendency -
                self.wmt.mass_source -
                self.wmt.overturning
            )
            # By construction, kinematic_material_derivative == process_material_derivative, so use whichever available
            if "kinematic_material_derivative" in self.wmt:
                self.wmt["spurious_numerical_mixing"] = (
                    self.wmt.Leibniz_material_derivative -
                    self.wmt.kinematic_material_derivative
                )
            elif "process_material_derivative" in self.wmt:
                self.wmt["spurious_numerical_mixing"] = (
                    self.wmt.Leibniz_material_derivative -
                    self.wmt.process_material_derivative
                )

            if "advection" in self.wmt:
                if "surface_ocean_flux_advective_negative_lhs" in self.wmt:
                    self.wmt["advection_plus_BC"] = self.wmt.advection + self.wmt.surface_ocean_flux_advective_negative_lhs
                else:
                    self.wmt["advection_plus_BC"] = self.wmt.advection

            if ("spurious_numerical_mixing" in self.wmt) and ("advection_plus_BC" in self.wmt):
                self.wmt["diabatic_advection"] = self.wmt.advection_plus_BC + self.wmt.spurious_numerical_mixing

    def lam_coord_defaults(self, lam):
        lambda_name = self.get_lambda_var(lam)
        if lam=="sigma2":
            lam_min, lam_max, dlam = 0., 50., 0.1
            
        elif lam=="heat":
            lam_min, lam_max, dlam = -4, 40., 0.1

        elif lam=="salt":
            lam_min, lam_max, dlam = -1., 50., 0.1

        self.grid._ds = self.grid._ds.assign_coords({
            f"{lambda_name}_l_target" : np.arange(lam_min, lam_max, dlam),
            f"{lambda_name}_i_target" : np.arange(lam_min-dlam/2., lam_max+dlam/2, dlam),
        })
        return {"outer": f"{lambda_name}_i_target", "center": f"{lambda_name}_l_target"}