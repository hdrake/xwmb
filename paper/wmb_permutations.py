#!/usr/bin/env python
# coding: utf-8

import sys, getopt, pathlib
import warnings
import numpy as np
import xarray as xr
import xbudget
import regionate
import xwmt
import xwmb
import xhistogram

sys.path.append("../examples/")
from loading import *

def main(argv):
    inputfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv,"hg:t:",["gridname=","dt="])
    for opt, arg in opts:
        if opt == '-h':
            print ('wmb_permutations.py -g <gridname> -t <dt>')
            sys.exit()
        elif opt in ("-g", "--grid"):
            gridname = arg
        elif opt in ("-t", "--dt"):
            dt = arg
    print(f"Computing water mass budgets for the Baltic sea on {gridname} grid and with {dt} output.")

    grid_ref = load_baltic("rho2", "monthly") # Just to get density bins for direct comparison

    for lamopt in ["heat", "salt", "sigma0", "sigma2-online", "sigma2-offline", "sigma2-online-diagbins","sigma2-offline-diagbins"]:
        online_density=False
        if "online" in lamopt: online_density=True
        grid = load_baltic(gridname, dt, online_density=online_density)

        default_bins=True
        if "diagbins" in lamopt:
            grid._ds = grid._ds.assign_coords({
                "sigma2_l_target": grid_ref._ds['sigma2_l'].rename({"sigma2_l":"sigma2_l_target"}),
                "sigma2_i_target": grid_ref._ds['sigma2_i'].rename({"sigma2_i":"sigma2_i_target"}),
            })
            grid = xwmt.add_gridcoords(
                grid,
                {"Z_target": {"center": "sigma2_l_target", "outer": "sigma2_i_target"}},
                {"Z_target": "extend"}
            )
            default_bins=False

        if "sigma2" in lamopt:
            lam = "sigma2"
        else:
            lam = lamopt

        budgets_dict = xbudget.load_preset_budget(model="MOM6_3Donly")
        xbudget.collect_budgets(grid, budgets_dict)
        simple_budget = xbudget.aggregate(budgets_dict)

        # Note: the properties of this region are quite different from the rest of the Baltic!
        name = "Baltic"
        lons = np.array([13, 10, 9.0, 10., 12, 20.,  29., 24.5, 23.5, 22.5, 17.5])
        lats = np.array([58, 57.5, 56, 54, 53.5, 53.5, 54.5,  59.,  61.,  63., 64.5])
        region = regionate.GriddedRegion(name, lons, lats, grid)

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            wmb = xwmb.WaterMassBudget(
                grid,
                budgets_dict,
                region
            )
            wmb.mass_budget(lam, default_bins=default_bins)

            wmb.get_density("sigma2")
            dm = wmb.rho_ref*wmb.grid._ds['thkcello']*wmb.grid._ds['areacello']*region.mask
            wmb.wmt['thermo_hist'] = xhistogram.xarray.histogram(
                wmb.grid._ds['ct'], wmb.grid._ds['sa'],
                bins=[np.arange(-4, 40, 0.1), np.arange(-1, 50, 0.1)],
                weights=dm,
                block_size=None,
                density=True
            )

            wmb.wmt.load()

        path = pathlib.Path("data/")
        path.mkdir(parents=True, exist_ok=True)
        wmb.wmt.to_netcdf(f"data/baltic_wmb_{lamopt}_{gridname}_{dt}.nc", mode="w")

if __name__ == "__main__":
    main(sys.argv[1:])


