"""The ``WaterMassBudget`` orchestrator.

``WaterMassBudget`` extends ``xwmt.WaterMassTransformations`` with the remaining
terms of a closed water-mass budget (Drake et al. 2025): the convergent mass
transport across the region boundary, the surface mass source, the mass storage
(tendency), and the residual spurious numerical mixing. The heavy lifting for each
term lives in a dedicated module; this class only wires them together.
"""

import warnings

import xbudget

from xwmt.wmt import WaterMassTransformations

from .regions import normalize_region
from .coordinates import resolve_target_coords
from .transformations import compute_transformations
from .transport import convergent_transport_term
from .mass import (
    layer_mass_term,
    mass_source_term,
    mass_bounds_term,
    mass_tendency,
)
from .close import close_budget
from .completeness import budget_completeness

# Only the class this module defines. Re-exporting the helpers it happens to
# import would document each of them at three names (`xwmb.x`, `xwmb.budget.x`
# and `xwmb.<home>.x`), which sphinx-apidoc reports as a duplicate object
# description -- and, with `fail_on_warning`, fails the docs build over. The
# package's public surface is assembled in `xwmb/__init__.py` instead.
__all__ = ["WaterMassBudget"]


def _warn_unlabelled_area(grid):
    """Warn once if the horizontal area metric carries no ``units``.

    xbudget multiplies the area metric into essentially every term it
    materializes, and it infers a term's units from its operands'. An unlabelled
    area therefore does not cost one attribute -- it costs the units of the whole
    budget, and of everything xwmt and xwmb derive downstream. The published MOM6
    example file is one of the datasets that ships `areacello` unlabelled, so this
    is the common case rather than the exotic one.
    """
    metrics = [da for das in grid._metrics.values() for da in das]
    unlabelled = sorted(
        {da.name for da in metrics if "units" not in da.attrs and da.name}
    )
    if unlabelled:
        warnings.warn(
            f"The grid metric(s) {', '.join(repr(n) for n in unlabelled)} carry no "
            f"'units' attribute. xbudget multiplies the cell area into every term "
            f"it materializes and infers units from its operands, so the derived "
            f"budget variables will come back without units rather than with "
            f"guessed ones. Label them before collecting the budget, e.g. "
            f"`ds['areacello'].attrs['units'] = 'm2'`.",
            stacklevel=3,
        )


class WaterMassBudget(WaterMassTransformations):
    """Lazy, closed water-mass budget in a sub-domain of a C-grid ocean model."""

    def __init__(
        self,
        grid,
        recipe,
        region=None,
        eos="teos10",
        cp=3992.0,
        rho_ref=1035.0,
        t_var="conservative",
        s_var="absolute",
        method="default",
        rebin=False,
        decompose=(),
        assert_zero_transport=False,
    ):
        """Create a ``WaterMassBudget`` from an ``xgcm.Grid`` and an xbudget recipe.

        Parameters
        ----------
        grid : xgcm.Grid
            Ocean-model grid coordinates, metrics, and data variables. May carry
            ``face_connections`` for multi-tile (e.g. ECCO LLC90) grids.
        recipe : dict
            A budget recipe: the nested dictionary naming each budget's lambda,
            thickness, and tendency variables. Load one with
            ``xbudget.load_preset_budget(model=...)`` (see ``xbudget/recipes`` for
            the shipped presets) and materialize its terms into ``grid`` with
            ``xbudget.collect_budgets(grid, recipe)`` before constructing this.
        region : regionate.GriddedRegion, regionate.MaskRegion, tuple, xr.DataArray, or None
            The sub-domain. A ``(lons, lats)`` tuple builds a ``GriddedRegion``; a
            boolean ``xr.DataArray`` builds a ``MaskRegion`` (largest connected
            component); ``None`` uses the full domain (and asserts zero net boundary
            transport). See :func:`xwmb.regions.normalize_region`.
        eos : str or None (default: "teos10")
            Equation of state, forwarded to ``xwmt`` (via ``xeos``). Pass ``None`` to
            require pre-computed ``alpha``/``beta``/density in the grid dataset.
        cp, rho_ref : float
            Specific heat capacity and reference (Boussinesq) density.
        method : str (default: "default")
            Vertical-transformation method: "default", "xhistogram", or "xgcm".
        rebin : bool (default: False)
            Force transformation into the target coordinates even when they exist.
        decompose : str or iterable of str (default: ())
            Decompose these summed recipe terms into their constituent parts.
            Matching is exact (an xbudget 0.8.0 semantic, not a substring match).
        assert_zero_transport : bool (default: False)
            Assert the net boundary transport vanishes, accelerating the calculation
            for domains where it is already known to be zero.

        Example
        -------
        >>> grid = xgcm.Grid(ds, coords=coords, padding=padding)
        >>> recipe = xbudget.load_preset_budget(model="MOM6")
        >>> xbudget.collect_budgets(grid, recipe)
        >>> wmb = xwmb.WaterMassBudget(grid, recipe)
        """
        # xbudget 0.8.0 reads a recipe through a query object rather than by
        # walking the dict: `collect_budgets` no longer fills the recipe's `var`
        # fields, and the module-level `aggregate()` helper is gone. The query is
        # kept on the instance because the budget terms below resolve their
        # variable names (and units) through it too, instead of hardcoding the
        # names the evaluator happens to produce.
        self.query = xbudget.BudgetQuery(grid, recipe)

        super().__init__(
            grid,
            self.query.aggregate(decompose=decompose),
            eos=eos,
            cp=cp,
            rho_ref=rho_ref,
            t_var=t_var,
            s_var=s_var,
            method=method,
            rebin=rebin,
        )

        self.full_recipe = recipe
        self.padding = {ax: self.grid.axes[ax].padding for ax in self.grid.axes.keys()}

        # Normalize the region against the (deep-copied) grid we compute on.
        self.region = normalize_region(region, self.grid)
        if assert_zero_transport:
            self.region.assert_zero_transport = True
        self.assert_zero_transport = self.region.assert_zero_transport

        _warn_unlabelled_area(self.grid)

    def mass_budget(
        self,
        lambda_name,
        greater_than=False,
        integrate=True,
        along_section=False,
        bins=None,
        utr=None,
        vtr=None,
        mass_source_var=None,
    ):
        """Lazily evaluate the full water-mass budget as a function of ``lambda``.

        Parameters
        ----------
        lambda_name : str
            The tracer defining the water mass (e.g. "sigma2", "heat", "salt").
        greater_than : bool (default: False)
            Budget for waters with tracer values *greater than* the threshold.
        integrate : bool (default: True)
            Horizontally integrate the terms over the region.
        along_section : bool (default: False)
            Compute the convergent transport along the region boundary with
            ``sectionate`` (required for multi-tile grids; preserves the
            along-boundary streamfunction structure). Requires ``integrate=True``.
        bins : None, array, or "default" (default: None)
            If None: assume the lambda bins are already in the dataset.
            If an array: a 1D array of bin edges.
            If "default": a finely-spaced default grid for ``lambda_name``.
        utr, vtr : str, optional
            Names of the (zonal, meridional) face mass-transport variables in
            ``grid._ds``. By default they are resolved from the recipe, which must
            then use the MOM6 convention's ``zonal_convergence`` /
            ``meridional_convergence`` term names; pass them explicitly otherwise.
        mass_source_var : str, optional
            Name of the surface mass-flux variable. By default it is resolved from
            the recipe's ``("mass", "rhs", "surface_exchange_flux")`` term.

        Returns
        -------
        xr.Dataset
            All terms in the closed water-mass transformation budget (``self.wmt``).
        """
        if along_section and not integrate:
            raise ValueError("Cannot have both `integrate=False` and `along_section=True`.")

        lambda_var = self.get_lambda_var(lambda_name)
        # Ensure the lambda field itself is available (e.g. a density derived from
        # T/S), even if no transformation process references it.
        if lambda_var not in self.grid._ds:
            self.get_density(lambda_var)
        self.grid, self.target_coords = resolve_target_coords(
            self.grid, lambda_var, lambda_name, bins=bins
        )
        self.ax_bounds = "Z" if "Z_bounds" not in self.grid.axes else "Z_bounds"
        self.prebinned = all(
            c in self.grid.axes[self.ax_bounds].coords.values()
            for c in [f"{lambda_var}_l", f"{lambda_var}_i"]
        )

        term_kwargs = dict(
            greater_than=greater_than, integrate=integrate, prebinned=self.prebinned
        )

        # Transformation rates (from xwmt), grouped and signed.
        self.wmt = compute_transformations(
            self,
            lambda_name,
            self.target_coords,
            self.region.mask,
            greater_than=greater_than,
            integrate=integrate,
        )

        # Storage snapshots -> mass tendency.
        mass_bounds = mass_bounds_term(
            self, lambda_name, self.target_coords, self.region, **term_kwargs
        )
        if mass_bounds is not None:
            self.wmt["mass_bounds"] = mass_bounds

        # Convergent transport across the region boundary.
        self.wmt["convergent_mass_transport"] = convergent_transport_term(
            self,
            lambda_name,
            self.target_coords,
            self.region,
            along_section=along_section,
            utr=utr,
            vtr=vtr,
            **term_kwargs,
        )

        # Surface mass source.
        mass_source = mass_source_term(
            self,
            lambda_name,
            self.target_coords,
            self.region,
            mass_source_var=mass_source_var,
            **term_kwargs,
        )
        if mass_source is not None:
            self.wmt["mass_source"] = mass_source

        # Layer mass.
        self.wmt["layer_mass"] = layer_mass_term(
            self,
            lambda_name,
            self.target_coords,
            self.region,
            integrate=integrate,
            prebinned=self.prebinned,
        )

        if "mass_bounds" in self.wmt:
            mass_tendency(self.wmt)

        # Audit before closing: whether the residual may be *called* spurious
        # numerical mixing depends on nothing else being unaccounted for.
        self.completeness = budget_completeness(self, self.wmt, lambda_name)
        close_budget(
            self.wmt,
            report=self.completeness,
            lambda_name=lambda_name,
            lambda_var=lambda_var,
        )
        return self.wmt
