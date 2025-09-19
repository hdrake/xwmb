# xwmb

**xWMB** is a Python package that provides a efficient and lazy computation of Water Mass Budgets in arbitrary sub-domains of C-grid ocean models. Most of the heavy lifting is done by dependency packages by the same team of developers:
- [`sectionate`](https://github.com/MOM6-Community/sectionate): for computing transports normal to a section (open or closed)
- [`regionate`](https://github.com/hdrake/regionate): for converting between gridded masks and the closed sections that bound them
- [`xbudget`](https://github.com/hdrake/xbudget): for model-agnostic wrangling of multi-level tracer budgets
- [`xwmt`](https://github.com/NOAA-GFDL/xwmt): for computing bulk water mass transformations from these budgets

Documentation is not yet available, but the core API is illustrated in the example notebooks here and in each of the dependency packages.

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

