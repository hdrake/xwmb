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

sys.path.append("../examples/")
from loading import *

def main(argv):
    inputfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv,"hg:t:",["gridname=","dt="])
    for opt, arg in opts:
        if opt == '-h':
            print ('wmb_permutations.py -g <gridname> -o <dt>')
            sys.exit()
        elif opt in ("-g", "--grid"):
            gridname = arg
        elif opt in ("-t", "--dt"):
            dt = arg
    print(f"Computing water mass budgets for the Baltic sea on {gridname} grid and with {dt} output.")

    grid_ref = load_baltic("rho2", "monthly") # Just to get density bins for direct comparison

    for lam in ["heat", "salt", "sigma2"]:
        grid = load_baltic(gridname, dt)
        default_bins=True
        if lam=="sigma2": # Specify the target sigma2 bins from the diagnostic rho2 grid
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

        budgets_dict = xbudget.load_preset_budget(model="MOM6_3Donly")
        xbudget.collect_budgets(grid._ds, budgets_dict)
        simple_budget = xbudget.aggregate(budgets_dict)

        # Note: the properties of this region are quite different from the rest of the Baltic!
        name = "intBaltic"
        lons = np.array([8.,   20.,  29., 24.5, 24.5, 26.1, 17.5, 11.5])
        lats = np.array([53.5, 53.5, 54.5,  59.,  61.,  63., 64.5,  62.])
        region = regionate.GriddedRegion(name, lons, lats, grid)

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            wmb = xwmb.WaterMassBudget(
                grid,
                budgets_dict,
                region
            )
            wmb.mass_budget(lam, default_bins=default_bins)
            wmb.wmt.load()

        path = pathlib.Path("data/")
        path.mkdir(parents=True, exist_ok=True)
        wmb.wmt.to_netcdf(f"data/baltic_wmb_{lam}_{gridname}_{dt}.nc", mode="w")

if __name__ == "__main__":
    main(sys.argv[1:])


