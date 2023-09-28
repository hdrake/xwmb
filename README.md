# xwmb

**xWMB** is a Python package that provides a efficient and lazy computation of Water Mass Budgets in arbitrary sub-domains of C-grid ocean models. Most of the heavy lifting is done by dependency packages by the same team of developers:
- [`sectionate`](https://github.com/hdrake/sectionate): for computing transports normal to a section (open or closed)
- [`regionate`](https://github.com/hdrake/regionate): for converting between gridded masks and the closed sections that bound them
- [`xbudget`](https://github.com/hdrake/xbudget): for model-agnostic wrangling of multi-level tracer budgets
- [`xwmt`](https://github.com/hdrake/xwmt): for computing bulk water mass transformations from these budgets

Documentation is not yet available, but the core API is illustrated in the example notebooks here and in each of the dependency packages.

Quick Start Guide
-----------------

**Minimal installation within an existing environment**
```bash
pip install git+https://github.com/hdrake/xwmb.git@main
```

See the [`xwmt` Quick-Start guide](https://github.com/hdrake/xwmt#quick-start-guide) for instructions on configuring a conda environment from scratch.
