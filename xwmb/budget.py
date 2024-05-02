import xarray as xr
import numpy as np
import warnings

import regionate
import sectionate

from xwmt.wmt import WaterMassTransformations
from xwmt.wm import add_gridcoords
from xgcm import Grid
import xbudget

class WaterMassBudget(WaterMassTransformations):
    def __init__(
        self,
        grid,
        budgets_dict,
        region=None,
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
            mask = xr.ones_like(
                self.grid._ds[self.grid.axes['Y'].coords['center']] *
                self.grid._ds[self.grid.axes['X'].coords['center']]
            )
            self.region = regionate.MaskRegions(
                mask,
                self.grid
            ).regions[0]
            self.assert_zero_transport = True

    def mass_budget(self, lam, greater_than=False, integrate=True, along_section=False, default_bins=False):
        lambda_name = self.get_lambda_var(lam)
        target_coords = {"center": f"{lambda_name}_l_target", "outer": f"{lambda_name}_i_target"}
        if default_bins:
            if "Z_target" not in self.grid.axes:
                self.add_default_gridcoords(lam)
            else:
                raise ValueError(
                    """Cannot pass `default_bins=True` when `Z_target in WaterMassBudget.grid.axes`."""
                )
        else:
            avail_target_coords = [c in self.grid._ds for c in target_coords.values()]
            avail_lambda_coords = [
                c.replace("_target", "") in self.grid._ds
                for c in target_coords.values()
            ]
            if "Z_target" not in self.grid.axes:
                if all(avail_lambda_coords):
                    if not all(avail_target_coords):
                        self.grid._ds = self.grid._ds.assign_coords({
                            target_coords["center"]: xr.DataArray(
                                self.grid._ds[target_coords["center"].replace("_target", "")].values,
                                dims=(target_coords["center"],)
                            ),
                            target_coords["outer"]: xr.DataArray(
                                self.grid._ds[target_coords["outer"].replace("_target", "")].values,
                                dims=(target_coords["outer"],)
                            )
                        })
                    self.grid = add_gridcoords(
                        self.grid,
                        {"Z_target": target_coords},
                        {"Z_target": "extend"}
                    )
                else: # `elif not all(avail_lambda_coords):`
                    raise ValueError(
                        f"""To specify target grid, either pass `default_bins=True` or
                        include {target_coords["center"]} and {target_coords["outer"]}
                        in `WaterMassBudget.grid._ds`.""")
        self.target_coords = target_coords
        self.ax_bounds = "Z" if "Z_bounds" not in self.grid.axes else "Z_bounds"
        self.prebinned = all([
            (c in self.grid.axes[self.ax_bounds].coords.values())
            for c in [f"{lambda_name}_l", f"{lambda_name}_i"]
        ])
        
        self.wmt = self.transformations(lam, integrate=integrate, greater_than=greater_than)
        self.mass_bounds(lambda_name, integrate=integrate, greater_than=greater_than)
        self.convergent_transport(lambda_name, integrate=integrate, greater_than=greater_than, along_section=along_section)
        mass_tendency(self.wmt)
        close_budget(self.wmt)
        return self.wmt
        
    def transformations(self, lam, integrate=True, greater_than=False):
        lambda_name = self.get_lambda_var(lam)
        kwargs = {
            "bins": self.grid._ds[self.target_coords["outer"]],
            "mask": self.region.mask,
            "group_processes": True,
            "sum_components": True
        }
        if integrate:
            wmt = self.integrate_transformations(lam, **kwargs)
        else:
            wmt = self.map_transformations(lam, **kwargs)
            
        wmt = wmt.assign_coords({
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
        if any([term in wmt for term in boundary_flux_terms]):
            wmt['boundary_fluxes'] = sum([wmt[term] for term in boundary_flux_terms if term in wmt])
        
        return wmt
        
    def convergent_transport(self, lambda_name, integrate=True, greater_than=False, along_section=False):
        kwargs = {
            "layer":     self.grid.axes["Z"].coords['center'],
            "interface": self.grid.axes["Z"].coords['outer'],
            "geometry":  "spherical"
        }
        if not(integrate) and along_section:
            raise ValueError("Cannot have both `integrate=False` and `along_section=True`.")
            
        if not self.assert_zero_transport:
            lateral_transports = self.full_budgets_dict['mass']['rhs']['sum']['advection']['sum']['lateral']
            if "sum" in lateral_transports:
                kwargs = {**kwargs, **{
                        f"{di_shorthand}tr":
                        lateral_transports['sum'][f'{di}_convergence']['product'][f'{di}_divergence']['difference'][f'{di}_mass_transport']
                        for (di, di_shorthand) in zip(["zonal", "meridional"], ["u", "v"])
                    }}
                surface_integral_condition = np.all(
                    [kwargs[f"{di_shorthand}tr"] in self.grid._ds
                     for di_shorthand in ["u", "v"]]
                )
                if not surface_integral_condition:
                    raise ValueError("Lateral transports are not available in `ds`!")
                    
                if along_section: # compute normal transports using sectionate
                    self.grid._ds['convergent_mass_transport_original'] = sectionate.convergent_transport(
                        self.grid,
                        self.region.i,
                        self.region.j,
                        positive_in = self.region.mask,
                        **kwargs
                    ).rename({"lat":"lat_sect", "lon":"lon_sect"})['conv_mass_transport']
    
                    if self.prebinned:
                        target_data = self.grid._ds[f'{lambda_name}_i']
                    else:
                        self.grid._ds[f'{lambda_name}_sect'] = sectionate.extract_tracer(
                            lambda_name,
                            self.grid,
                            self.region.i,
                            self.region.j,
                        )
            
                        self.grid._ds[f'{lambda_name}_i_sect'] = (
                            self.grid.interp(self.grid._ds[f'{lambda_name}_sect'], "Z", boundary="extend")
                            .chunk({self.grid.axes['Z'].coords['outer']: -1})
                            .rename(f'{lambda_name}_i_sect')
                        )
                        target_data = self.grid._ds[f'{lambda_name}_i_sect']

                    self.grid._ds['convergent_mass_transport_layer'] = (
                        self.grid.transform(
                            self.grid._ds['convergent_mass_transport_original'].fillna(0.),
                            "Z",
                            target = self.grid._ds[self.target_coords["outer"]],
                            target_data = target_data,
                            method="conservative"
                        )
                        .rename({self.target_coords["outer"]: self.target_coords["center"]})
                        .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]})
                    )

                elif not(along_section): # compute normal transports for each grid cell then sum
                    if self.prebinned:
                        lam_itpXZ = self.grid._ds[f'{lambda_name}_i']
                        lam_itpYZ = self.grid._ds[f'{lambda_name}_i']
                        
                    else:
                        lam_itpXZ = self.grid.interp(
                            self.grid.interp(self.grid._ds[lambda_name], "X"),
                            "Z",
                            boundary="extend"
                        ).chunk({self.grid.axes['Z'].coords['outer']: -1})
                        lam_itpYZ = self.grid.interp(
                            self.grid.interp(self.grid._ds[lambda_name], "Y"),
                            "Z",
                            boundary="extend"
                        ).chunk({self.grid.axes['Z'].coords['outer']: -1})

                    divergence_X = self.grid.diff(
                        self.grid.transform(
                            self.grid._ds[kwargs['utr']].fillna(0.),
                            "Z",
                            target = self.grid._ds[self.target_coords["outer"]],
                            target_data = lam_itpXZ,
                            method="conservative"
                        ).fillna(0.)
                        .rename({self.target_coords["outer"]: self.target_coords["center"]})
                        .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]}),
                        "X"
                    )
                    divergence_Y = self.grid.diff(
                        self.grid.transform(
                            self.grid._ds[kwargs['vtr']].fillna(0.),
                            "Z",
                            target = self.grid._ds[self.target_coords["outer"]],
                            target_data = lam_itpYZ,
                            method="conservative"
                        ).fillna(0.)
                        .rename({self.target_coords["outer"]: self.target_coords["center"]})
                        .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]}),
                        "Y"
                    )
                    
                    self.grid._ds['convergent_mass_transport_layer'] = -(
                        divergence_X + divergence_Y
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
                    self.target_coords["outer"]:  slice(None, None, -1),
                    self.target_coords["center"]: slice(None, None, -1)
                })
                lam_rev_grid = Grid(
                    self.grid._ds,
                    coords={'lam': self.target_coords},
                    boundary={'lam': 'extend'},
                    autoparse_metadata=False
                )
                self.grid._ds[f'convergent_mass_transport_{suffix}'] = lam_rev_grid.cumsum(
                    self.grid._ds['convergent_mass_transport_layer'], "lam", boundary="fill", fill_value=0.
                ).chunk({self.target_coords["outer"]: -1})
                self.grid._ds = self.grid._ds.isel({
                    self.target_coords["outer"]:  slice(None, None, -1),
                    self.target_coords["center"]: slice(None, None, -1),
                })
            else:
                self.grid._ds[f'convergent_mass_transport_{suffix}'] = lam_grid.cumsum(
                    self.grid._ds['convergent_mass_transport_layer'], "lam", boundary="fill", fill_value=0.
                ).chunk({self.target_coords["outer"]: -1}).assign_coords(
                    {self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]]}
                )

        # Compute mass source term
        if self.prebinned:
            target_data = self.grid._ds[f'{lambda_name}_i']
        else:
            target_data = (
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
                target_data = target_data,
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
            self.grid._ds = self.grid._ds.isel({
                self.target_coords["outer"]:  slice(None, None, -1),
                self.target_coords["center"]: slice(None, None, -1)
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
            self.grid._ds = self.grid._ds.isel({
                self.target_coords["outer"]:  slice(None, None, -1),
                self.target_coords["center"]: slice(None, None, -1)
            })
        else:
            self.grid._ds[f'mass_source_density_{suffix}'] = lam_grid.cumsum(
                self.grid._ds.mass_source_density, "lam", boundary="fill", fill_value=0.
            ).chunk({self.target_coords["outer"]: -1})

        if integrate:
            self.grid._ds[f'mass_source_{suffix}'] = (self.grid._ds[f'mass_source_density_{suffix}']).sum([
                self.grid.axes['X'].coords['center'],
                self.grid.axes['Y'].coords['center']
            ])
        else:
            self.grid._ds[f'mass_source_{suffix}'] = self.grid._ds[f'mass_source_density_{suffix}']
        
        # Compute layer mass
        self.grid._ds['mass_density'] = (self.grid.transform(
                self.rho_ref*self.grid._ds[self.h_name].fillna(0.),
                "Z",
                target = self.grid._ds[self.target_coords["outer"]],
                target_data = target_data,
                method="conservative"
            )
            .rename({self.target_coords["outer"]: self.target_coords["center"]})
            .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]})
        ) * self.region.mask
        
        self.wmt['layer_mass'] = (
            self.grid._ds['mass_density'] *
            self.grid.get_metric(self.grid._ds['mass_density'], ("X", "Y"))
        )
        if integrate:
            self.wmt['layer_mass'] = self.wmt['layer_mass'].sum([
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
            convergent_transport = lam_grid.interp(
                self.grid._ds[f'convergent_mass_transport_{suffix}'],
                "lam",
                boundary="extend"
            )
            if integrate:
                if "sect" in convergent_transport.dims:
                    self.grid._ds['convergent_mass_transport_along'] = convergent_transport
                    self.wmt['convergent_mass_transport'] = convergent_transport.sum("sect")
                else:
                    area_dims = list(self.grid.get_metric(convergent_transport, ("X", "Y")).dims)
                    self.wmt['convergent_mass_transport'] = convergent_transport.sum(area_dims)
            else:
                self.wmt['convergent_mass_transport'] = convergent_transport
            
            self.wmt = self.wmt.assign_coords(
                    {self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]}
                )
            
        else:
            self.wmt['convergent_mass_transport'] = xr.zeros_like(self.wmt['mass_source'])
        
        return self.wmt.convergent_mass_transport
        
    
    def mass_bounds(self, lambda_name, integrate=True, greater_than=False):
                
        if "time_bounds" in self.grid._ds.dims:
            if self.prebinned:
                target_data = self.grid._ds[f"{lambda_name}_i"]
            else:
                self.grid._ds[f"{lambda_name}_i_bounds"] = (
                    self.grid.interp(self.grid._ds[f"{lambda_name}_bounds"], self.ax_bounds, boundary="extend")
                    .chunk({self.grid.axes[self.ax_bounds].coords['outer']: -1})
                    .rename(f"{lambda_name}_i_bounds")
                )
                target_data = self.grid._ds[f"{lambda_name}_i_bounds"]
                
            self.grid._ds['mass_density_bounds'] = (
                self.grid.transform(
                    self.rho_ref*self.grid._ds[f"{self.h_name}_bounds"].fillna(0.),
                    self.ax_bounds,
                    target = self.grid._ds[self.target_coords["outer"]],
                    target_data = target_data,
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
                self.grid._ds = self.grid._ds.isel({
                    self.target_coords["outer"]:  slice(None, None, -1),
                    self.target_coords["center"]: slice(None, None, -1),
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
                self.grid._ds = self.grid._ds.isel({
                    self.target_coords["outer"]:  slice(None, None, -1),
                    self.target_coords["center"]: slice(None, None, -1),
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
            )
            
            if integrate:
                self.wmt['mass_bounds'] = lam_grid.interp(
                    self.grid._ds[f'mass_bounds_{suffix}'],
                    "lam"
                ).sum([
                    self.grid.axes['X'].coords['center'],
                    self.grid.axes['Y'].coords['center']
                ])
            else:
                self.wmt['mass_bounds'] = lam_grid.interp(
                    self.grid._ds[f'mass_bounds_{suffix}'],
                    "lam"
                )
            
            self.wmt['mass_bounds'] = self.wmt['mass_bounds'].assign_coords(
                {self.target_coords["center"]: self.grid._ds[self.target_coords["center"]].values}
            )

        return
        
    def add_default_gridcoords(self, lam):
        lambda_name = self.get_lambda_var(lam)
        if "sigma" in lam:
            lam_min, lam_max, dlam = 0., 50., 0.1
            
        elif lam=="heat":
            lam_min, lam_max, dlam = -4, 40., 0.1

        elif lam=="salt":
            lam_min, lam_max, dlam = -1., 50., 0.1

        self.grid._ds = self.grid._ds.assign_coords({
            f"{lambda_name}_l_target" : np.arange(lam_min, lam_max, dlam),
            f"{lambda_name}_i_target" : np.arange(lam_min-dlam/2., lam_max+dlam/2, dlam),
        })
        
        self.grid = add_gridcoords(
            self.grid,
            {"Z_target": {"outer": f"{lambda_name}_i_target", "center": f"{lambda_name}_l_target"}},
            {"Z_target": "extend"}
        )
        
def mass_tendency(ds):
    dt = ds.time_bounds.diff('time_bounds').astype('float')*1.e-9
    if ds.time_bounds.size == ds.time.size:
        time_target = ds.time[1:]
        print("Warning: first value of `mass_tendency` may be NaN!")
    elif ds.time_bounds.size == (ds.time.size + 1):
        time_target = ds.time
    else:
        raise ValueError("time_bounds inconsistent with time")
    ds['mass_tendency'] = (
        ds.mass_bounds.diff('time_bounds') / dt
    ).rename({"time_bounds":"time"}).assign_coords({'time': time_target})
    ds['dt'] = dt.rename({"time_bounds":"time"}).assign_coords({'time':time_target})
        
def close_budget(ds):
    Leibniz_material_derivative_terms = [
        "mass_tendency",
        "mass_source",
        "convergent_mass_transport",
    ]
    if all([term in ds for term in Leibniz_material_derivative_terms]):
        ds["Leibniz_material_derivative"] = - (
            ds.mass_tendency -
            ds.mass_source -
            ds.convergent_mass_transport
        )
        # By construction, kinematic_material_derivative == process_material_derivative,
        # so use whichever available
        if "kinematic_material_derivative" in ds:
            ds["spurious_numerical_mixing"] = (
                ds.Leibniz_material_derivative -
                ds.kinematic_material_derivative
            )
        elif "process_material_derivative" in ds:
            ds["spurious_numerical_mixing"] = (
                ds.Leibniz_material_derivative -
                ds.process_material_derivative
            )

        if "advection" in ds:
            if "surface_ocean_flux_advective_negative_lhs" in ds:
                ds["advection_plus_BC"] = (
                    ds.advection +
                    ds.surface_ocean_flux_advective_negative_lhs
                )
            else:
                ds["advection_plus_BC"] = ds.advection

        if ("spurious_numerical_mixing" in ds) and ("advection_plus_BC" in ds):
            ds["diabatic_advection"] = (
                ds.advection_plus_BC +
                ds.spurious_numerical_mixing
            )
    
    return
