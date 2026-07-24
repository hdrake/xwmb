# xwmb

**xWMB** is a Python package that provides a efficient and lazy computation of Water Mass Budgets in arbitrary sub-domains of C-grid ocean models. Most of the heavy lifting is done by dependency packages by the same team of developers:
- [`sectionate`](https://github.com/MOM6-Community/sectionate): for computing transports normal to a section (open or closed)
- [`regionate`](https://github.com/hdrake/regionate): for converting between gridded masks and the closed sections that bound them
- [`xbudget`](https://github.com/hdrake/xbudget): for model-agnostic wrangling of multi-level tracer budgets
- [`xwmt`](https://github.com/NOAA-GFDL/xwmt): for computing bulk water mass transformations from these budgets

As of `xwmb` 0.6.0, the whole stack is **topology-aware**: budgets can be computed on
arbitrary `xgcm.Grid` topologies — single-tile periodic and bipolar/tripolar
north-fold grids, and genuinely multi-tile grids defined by `face_connections`
(e.g. ECCOv4r4 lat-lon-cap / LLC90). See `examples/ECCO_AABW_watermass_budget.ipynb`
for a full σ₂ budget of Antarctic Bottom Water on the 13-tile ECCO grid.

Documentation is not yet available, but the core API is illustrated in the example notebooks here and in each of the dependency packages. The basic usage is unchanged:

```python
import xwmb
wmb = xwmb.WaterMassBudget(grid, xbudget_dict, region=region)
wmt = wmb.mass_budget("sigma2", greater_than=True)   # a closed budget as a function of σ₂
```

`region` may be a `regionate.GriddedRegion`/`MaskRegion`, a `(lons, lats)` tuple, a
boolean `xr.DataArray` mask, or `None` (the full domain). On multi-tile grids, pass
`along_section=True` so the boundary transport is computed with `sectionate`
(face-index aware).

If you use `xwmb`, please cite the companion manuscript: Henri F. Drake, Shanice Bailey, Raphael Dussin, Stephen M. Griffies, John Krasting, Graeme MacGilchrist, Geoffrey Stanley, Jan-Erik Tesdal, Jan D. Zika. Water Mass Transformation Budgets in Finite-Volume Generalized Vertical Coordinate Ocean Models. Journal of Advances in Modeling Earth Systems. 08 March 2025. DOI: [doi.org/10.1029/2024MS004383](https://doi.org/10.1029/2024MS004383)

Quick Start Guide
-----------------

**Minimal installation within an existing environment**
```bash
pip install xwmb
```

**Installing from scratch using `conda`**

This is the recommended mode of installation for developers.
```bash
git clone git@github.com:hdrake/xwmb.git
cd xwmb
conda env create -f docs/environment.yml
conda activate docs_env_xwmb
pip install -e .
```

You can verify that the package was properly installed by confirming it passes all of the tests with:
```bash
pytest -v
```

You can launch a Jupyterlab instance using this environment with:
```bash
python -m ipykernel install --user --name docs_env_xwmb --display-name "docs_env_xwmb"
jupyter-lab
```

