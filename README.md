xwmb: xarray-enabled Water Mass transformation Budgets (WMB) using structured ocean model output
=========================

[![PyPI](https://badge.fury.io/py/xwmb.svg)](https://badge.fury.io/py/xwmb)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xwmb)](https://anaconda.org/conda-forge/xwmb)
[![Docs](https://readthedocs.org/projects/xwmb/badge/?version=latest)](https://xwmb.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/hdrake/xwmb)](https://github.com/hdrake/xwmb)


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
conda install -c conda-forge xwmb
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

Releasing
---------

**The git tag is the version.** `xwmb` has no version string checked into the
source tree: `hatch-vcs` derives it from the tag at build time and writes
`xwmb/_version.py` (gitignored, but shipped inside the sdist and wheel). To cut a
release you tag; there is no file to bump and nothing to keep in sync.

1. Make sure `main` is green and has everything you want in the release.
2. **Publish a GitHub Release** whose tag is `vX.Y.Z`, targeting the commit you
   want to ship:
   ```bash
   gh release create vX.Y.Z --target "$(git rev-parse origin/main)" \
     --title vX.Y.Z --generate-notes
   ```
   Publishing it (not merely pushing a tag) is what fires the workflow. Target
   the commit you actually want: a tag placed *before* the commit you meant to
   release builds the previous version, and PyPI rejects it as a duplicate.
3. The **Publish to PyPI** workflow builds from that tag and uploads. It checks
   out with `fetch-depth: 0` so the tag is visible to `hatch-vcs`, and asserts
   that the built version matches the tag before publishing.
4. Verify: <https://pypi.org/project/xwmb/>.
5. conda-forge builds from the PyPI sdist and lags by design; the autotick bot
   opens the version-bump PR. The feedstock recipe must list `hatch-vcs`
   alongside `hatchling` in its `host` requirements, since it builds with
   `--no-build-isolation`.

Two things follow from the tag being the version. Never add a version literal
back to the tree — `xwmb/version.py` is a shim over the generated file, and a
`0.0.0+unknown` from it means the package was imported without being built or
installed, not that a number is missing. And any checkout that installs the
package needs its tags: a shallow clone, or a fork that never fetched them,
resolves a `.devN` version instead of the release line. That is why CI checks out
with `fetch-depth: 0` and Read the Docs unshallows in `post_checkout`.

