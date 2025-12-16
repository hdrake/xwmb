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
    """An extension of the WaterMass class that includes methods for WaterMass transformation analysis."""
    def __init__(
        self,
        grid,
        xbudget_dict,
        region=None,
        teos10=True,
        cp=3992.,
        rho_ref=1035.,
        method="default",
        rebin=False,
        decompose=[],
        assert_zero_transport=False
        ):
        """
        Create a new WaterMassBudget object from an input xgcm.Grid and xbudget dictionary.

        Parameters
        ----------
        grid : xgcm.Grid
            Contains information about ocean model grid coordinates, metrics, and data variables.
        xbudget_dict : dict
            Nested dictionary containing information about lambda and tendency variable names.
            See `xwmt/conventions` for examples of how this dictionary should be structured
            or the `xbudget` package: https://github.com/hdrake/xbudget
        region : regionate.GriddedRegion, tuple, or xr.DataArray (default: None)
            If tuple: must be of length two with (lons, lats) arrays of equal length; uses
            regionate.GriddedRegion to create the region that approximates these coordinates.
            If xr.DataArray: assume bool dtype and use regionate.MaskRegions to create region
            based on this mask (pick the contiguous region boundary with the longest perimeter).
            If None, the region defaults to the full `xgcm.Grid` domain.
        teos10 : bool (default: True)
            Use Thermodynamic Equation Of Seawater - 2010 (TEOS-10). True by default.
        cp : float (default: 3992.0, the MOM6 default value)
            Value of specific heat capacity.
        rho_ref : float (default: 1035.0, the MOM6 default value)
            Value of reference potential density. Note: WaterMass is assumed to be Boussinesq.
        method : str (default: "default")
            Method used for vertical transformations.
            Supported options: "default", "xhistogram", "xgcm".
            If "default", use "xhistogram" for area-integrated calculations (`integrate=True`)
            or "xgcm" for column-wise calculations (`integrate=False`) for efficiency.
            The other options force the use of a specific method, perhaps at the cost of efficiency.
        rebin : bool (default: False)
            Set to True to force a transformation into the target coordinates, even if these
            coordinates already exist in the `grid` data structure.
        decompose : list (default: [])
            Decompose these summed xbudget terms into their constituent parts.
        assert_zero_transport : bool (default: False)
            Optionally assert that the diapycnal transport term is zero, accelerating the
            calculations for domains where it is already known that this term vanishes.
        """

        super().__init__(
            grid,
            xbudget.aggregate(xbudget_dict, decompose=decompose),
            teos10=teos10,
            cp=cp,
            rho_ref=rho_ref,
            method=method,
            rebin=rebin
        )
        self.full_xbudget_dict = xbudget_dict
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
            ).region_dict[0]
        elif region is None:
            mask = xr.ones_like(
                self.grid._ds[self.grid.axes['Y'].coords['center']] *
                self.grid._ds[self.grid.axes['X'].coords['center']]
            )
            self.region = regionate.MaskRegions(
                mask,
                self.grid
            ).region_dict[0]
            self.assert_zero_transport = True

    def mass_budget(self, lambda_name, greater_than=False, integrate=True, along_section=False, default_bins=False):
        """
        Lazily evaluates the mass budget

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'heat', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        greater_than : bool (default: False)
            Whether the budget should be for a water mass with tracer values "greater than" the threshold, or
            not ("less than" the threshold).
        integrate : bool (default: True)
            Whether to integrate the in the ("X", "Y") dimensions
        along_section : bool (default: False)
            Whether to include information about the along-section structure of convergent transports (with `sectionate`)
        default_bins : bool (default: False)
            Whether to use the default target coordinate grids for the vertical coordinate transformations

        Returns
        -------
        xr.Dataset
            Contains all terms in the full water mass transformation budget
        """
        lambda_var = self.get_lambda_var(lambda_name)
        target_coords = {"center": f"{lambda_var}_l_target", "outer": f"{lambda_var}_i_target"}
        if default_bins:
            if "Z_target" not in self.grid.axes:
                self.add_default_gridcoords(lambda_name)
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
            for c in [f"{lambda_var}_l", f"{lambda_var}_i"]
        ])
        
        self.wmt = self.transformations(lambda_name, integrate=integrate, greater_than=greater_than)
        self.mass_bounds(lambda_name, integrate=integrate, greater_than=greater_than)
        self.convergent_transport(lambda_name, integrate=integrate, greater_than=greater_than, along_section=along_section)
        mass_tendency(self.wmt)
        close_budget(self.wmt)
        return self.wmt
        
    def transformations(self, lambda_name, greater_than=False, integrate=True):
        """
        Lazily evaluates the water mass transformation terms in the budget

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'heat', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        greater_than : bool (default: False)
            Whether the budget should be for a water mass with tracer values "greater than" the threshold, or
            not ("less than" the threshold).
        integrate : bool (default: True)
            Whether to integrate the in the ("X", "Y") dimensions

        Returns
        -------
        xr.Dataset
            Contains the transformation terms in the full water mass transformation budget
        """
        kwargs = {
            "bins": self.grid._ds[self.target_coords["outer"]],
            "mask": self.region.mask,
            "group_processes": True,
            "sum_components": True
        }
        if integrate:
            wmt = self.integrate_transformations(lambda_name, **kwargs)
        else:
            wmt = self.map_transformations(lambda_name, **kwargs)
            
        wmt = wmt.assign_coords({
            self.target_coords["center"]: self.grid._ds[self.target_coords["center"]],
            self.target_coords["outer"]: self.grid._ds[self.target_coords["outer"]],
        })
        
        # Because the unit vector normal to isosurface switches directions,
        # we need to flip water mass transformation terms.
        # Alternatively, I think we could have just switched the ordering
        # of the bins so that \Delta \lambda flips.
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
        
    def convergent_transport(self, lambda_name, greater_than=False, integrate=True, along_section=False):
        """
        Lazily evaluates the convergent transport (diascalar overturning) term in the water mass budget

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'heat', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        greater_than : bool (default: False)
            Whether the budget should be for a water mass with tracer values "greater than" the threshold, or
            not ("less than" the threshold).
        integrate : bool (default: True)
            Whether to integrate the in the ("X", "Y") dimensions
        along_section : bool (default: False)
            Whether to include information about the along-section structure of convergent transports (with `sectionate`)

        Returns
        -------
        xr.DataArray
            The convergent transport (diascalar overturning) term in the full water mass transformation budget
        """
        lambda_var = self.get_lambda_var(lambda_name)
        kwargs = {
            "layer":     self.grid.axes["Z"].coords['center'],
            "interface": self.grid.axes["Z"].coords['outer'],
            "geometry":  "spherical"
        }
        if not(integrate) and along_section:
            raise ValueError("Cannot have both `integrate=False` and `along_section=True`.")
            
        if not self.assert_zero_transport:
            lateral_transports = self.full_xbudget_dict['mass']['rhs']['sum']['advection']['sum']['lateral']
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
                        self.region.i_c,
                        self.region.j_c,
                        positive_in = self.region.mask,
                        **kwargs
                    ).rename({"lat":"lat_sect", "lon":"lon_sect"})['conv_mass_transport']
    
                    if self.prebinned:
                        target_data = self.grid._ds[f'{lambda_var}_i']
                    else:
                        self.grid._ds[f'{lambda_var}_sect'] = sectionate.extract_tracer(
                            lambda_var,
                            self.grid,
                            self.region.i_c,
                            self.region.j_c,
                        )
            
                        self.grid._ds[f'{lambda_var}_i_sect'] = (
                            self.grid.interp(self.grid._ds[f'{lambda_var}_sect'], "Z", boundary="extend")
                            .chunk({self.grid.axes['Z'].coords['outer']: -1})
                            .rename(f'{lambda_var}_i_sect')
                        )
                        target_data = self.grid._ds[f'{lambda_var}_i_sect']

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
                        lam_itpXZ = self.grid._ds[f'{lambda_var}_i']
                        lam_itpYZ = self.grid._ds[f'{lambda_var}_i']
                        
                    else:
                        lam_itpXZ = self.grid.interp(
                            self.grid.interp(self.grid._ds[lambda_var], "X"),
                            "Z",
                            boundary="extend"
                        ).chunk({self.grid.axes['Z'].coords['outer']: -1})
                        lam_itpYZ = self.grid.interp(
                            self.grid.interp(self.grid._ds[lambda_var], "Y"),
                            "Z",
                            boundary="extend"
                        ).chunk({self.grid.axes['Z'].coords['outer']: -1})

                    divergence_X = self.grid.diff(
                        self.grid.transform(
                            self.grid._ds[kwargs['utr']].fillna(0.).chunk({self.grid.axes['Z'].coords['center']: -1}),
                            "Z",
                            target = self.grid._ds[self.target_coords["outer"]],
                            target_data = lam_itpXZ,
                            method="conservative"
                        ).fillna(0.)
                        .rename({self.target_coords["outer"]: self.target_coords["center"]})
                        .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]})
                        .chunk({self.grid.axes['X'].coords['outer']: -1}),
                        "X"
                    ).chunk({self.grid.axes['X'].coords['center']: 100, self.grid.axes['Y'].coords['center']: 100})
                    divergence_Y = self.grid.diff(
                        self.grid.transform(
                            self.grid._ds[kwargs['vtr']].fillna(0.).chunk({self.grid.axes['Z'].coords['center']: -1}),
                            "Z",
                            target = self.grid._ds[self.target_coords["outer"]],
                            target_data = lam_itpYZ,
                            method="conservative"
                        ).fillna(0.)
                        .rename({self.target_coords["outer"]: self.target_coords["center"]})
                        .assign_coords({self.target_coords["center"]: self.grid._ds[self.target_coords["center"]]})
                        .chunk({self.grid.axes['Y'].coords['outer']: -1}),
                        "Y"
                    ).chunk({self.grid.axes['X'].coords['center']: 100, self.grid.axes['Y'].coords['center']: 100})
                    
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
            target_data = self.grid._ds[f'{lambda_var}_i']
        else:
            target_data = (
                self.grid.interp(self.grid._ds[f"{lambda_var}"], "Z", boundary="extend")
                .chunk({self.grid.axes['Z'].coords['outer']: -1})
                .rename(f"{lambda_var}_i")
            )
        
        mass_flux_varname = "mass_rhs_sum_surface_exchange_flux"
        if mass_flux_varname in self.grid._ds:
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

        if mass_flux_varname in self.grid._ds:
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
            self.wmt['convergent_mass_transport'] = 0.
        
        return self.wmt.convergent_mass_transport
        
    
    def mass_bounds(self, lambda_name, greater_than=False, integrate=True):
        """
        Lazily evaluates snapshots (time "bounds") of the water mass' mass

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'heat', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        greater_than : bool (default: False)
            Whether the budget should be for a water mass with tracer values "greater than" the threshold, or
            not ("less than" the threshold).
        integrate : bool (default: True)
            Whether to integrate the in the ("X", "Y") dimensions
            
        Returns
        -------
        None, but adds "mass_bounds" to the xr.Dataset `self.wmt`
        """
        lambda_var = self.get_lambda_var(lambda_name)
        if "time_bounds" in self.grid._ds.dims:
            if self.prebinned:
                target_data = self.grid._ds[f"{lambda_var}_i"]
            else:
                self.grid._ds[f"{lambda_var}_i_bounds"] = (
                    self.grid.interp(self.grid._ds[f"{lambda_var}_bounds"], self.ax_bounds, boundary="extend")
                    .chunk({self.grid.axes[self.ax_bounds].coords['outer']: -1})
                    .rename(f"{lambda_var}_i_bounds")
                )
                target_data = self.grid._ds[f"{lambda_var}_i_bounds"]
                
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
                self.grid.get_metric(self.grid._ds[f"{lambda_var}_bounds"], ("X", "Y"))
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
        
    def add_default_gridcoords(self, lambda_name):
        lambda_var = self.get_lambda_var(lambda_name)
        if "sigma" in lambda_name:
            lam_min, lam_max, dlam = 0., 50., 0.1
            
        elif lambda_name=="heat":
            lam_min, lam_max, dlam = -4, 40., 0.1

        elif lambda_name=="salt":
            lam_min, lam_max, dlam = -1., 50., 0.1

        self.grid._ds = self.grid._ds.assign_coords({
            f"{lambda_var}_l_target" : np.arange(lam_min, lam_max, dlam),
            f"{lambda_var}_i_target" : np.arange(lam_min-dlam/2., lam_max+dlam/2, dlam),
        })
        
        self.grid = add_gridcoords(
            self.grid,
            {"Z_target": {"outer": f"{lambda_var}_i_target", "center": f"{lambda_var}_l_target"}},
            {"Z_target": "extend"}
        )
        
def mass_tendency(ds):
    """
    Computes the time-mean mass tendency by finite-differencing water mass snapshots
    """
    if not all([v in ds for v in ["time_bounds", "mass_bounds"]]):
        raise ValueError("Needs both `time_bounds` and `mass_bounds` variables")
    dt = ds.time_bounds.diff('time_bounds')
    if dt.dtype == "<m8[ns]":
        dt = dt.astype('float')*1.e-9
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
    """
    Close the full water mass transformation budget by identifying the residual as spurious numerical mixing.
    """
    realized_transformation_terms = [
        "mass_tendency",
        "mass_source",
        "convergent_mass_transport",
    ]
    if all([term in ds for term in realized_transformation_terms]):
        ds["realized_transformation"] = (
            ds.mass_tendency -
            ds.mass_source -
            ds.convergent_mass_transport
        )
        # By construction, kinematic_transformation == material_transformation,
        # so use whichever available
        if "material_transformation" in ds:
            ds["spurious_numerical_mixing"] = (
                ds.realized_transformation -
                ds.material_transformation
            )
        elif "kinematic_transformation" in ds:
            ds["spurious_numerical_mixing"] = (
                ds.realized_transformation -
                ds.kinematic_transformation
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
