"""The ``WaterMassBudget`` orchestrator.

``WaterMassBudget`` extends ``xwmt.WaterMassTransformations`` with the remaining
terms of a closed water-mass budget (Drake et al. 2025): the convergent mass
transport across the region boundary, the surface mass source, the mass storage
(tendency), and the residual spurious numerical mixing. The heavy lifting for each
term lives in a dedicated module; this class only wires them together.
"""

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

__all__ = ["WaterMassBudget", "mass_tendency", "close_budget"]


class WaterMassBudget(WaterMassTransformations):
    """Lazy, closed water-mass budget in a sub-domain of a C-grid ocean model."""

    def __init__(
        self,
        grid,
        xbudget_dict,
        region=None,
        eos="teos10",
        cp=3992.0,
        rho_ref=1035.0,
        t_var="conservative",
        s_var="absolute",
        method="default",
        rebin=False,
        decompose=[],
        assert_zero_transport=False,
        teos10=None,
    ):
        """Create a ``WaterMassBudget`` from an ``xgcm.Grid`` and an xbudget dict.

        Parameters
        ----------
        grid : xgcm.Grid
            Ocean-model grid coordinates, metrics, and data variables. May carry
            ``face_connections`` for multi-tile (e.g. ECCO LLC90) grids.
        xbudget_dict : dict
            Nested budget dictionary (see the ``xbudget`` package) mapping lambda and
            tendency variable names.
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
        decompose : list (default: [])
            Decompose these summed xbudget terms into their constituent parts.
        assert_zero_transport : bool (default: False)
            Assert the net boundary transport vanishes, accelerating the calculation
            for domains where it is already known to be zero.
        teos10 : bool, optional
            Deprecated back-compat alias for ``eos`` (``True``→"teos10",
            ``False``→``None``). Prefer ``eos``.
        """
        super_kwargs = dict(
            cp=cp, rho_ref=rho_ref, t_var=t_var, s_var=s_var, method=method, rebin=rebin
        )
        if teos10 is not None:
            super_kwargs["teos10"] = teos10
        else:
            super_kwargs["eos"] = eos
        super().__init__(
            grid,
            xbudget.aggregate(xbudget_dict, decompose=decompose),
            **super_kwargs,
        )

        self.full_xbudget_dict = xbudget_dict
        self.padding = {ax: self.grid.axes[ax].padding for ax in self.grid.axes.keys()}

        # Normalize the region against the (deep-copied) grid we compute on.
        self.region = normalize_region(region, self.grid)
        if assert_zero_transport:
            self.region.assert_zero_transport = True
        self.assert_zero_transport = self.region.assert_zero_transport

    def mass_budget(
        self,
        lambda_name,
        greater_than=False,
        integrate=True,
        along_section=False,
        bins=None,
        default_bins=None,
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
        bins : None or array (default: None)
            If None: assume the lambda bins are already in the dataset.
            If an array: a 1D array of bin edges.
        default_bins : deprecated
            Deprecated parameter. Use `bins` instead.
            If True: generate the default bins for `lambda_name`.
            If False: corresponds to ``bins=None``.
            If a list: corresponds to ``bins=np.arange(*default_bins)``.
        utr, vtr : str, optional
            Names of the (zonal, meridional) face mass-transport variables in
            ``grid._ds``. Defaults extract them from the (MOM6-convention) budget
            dict; pass explicitly for other conventions (e.g. ECCO ``"umo"``/``"vmo"``).
        mass_source_var : str, optional
            Name of the surface mass-flux density variable (defaults to the
            MOM6-convention name).

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
            self.grid, lambda_var, lambda_name, bins=bins, default_bins=default_bins
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
        close_budget(self.wmt)
        return self.wmt
