"""Is the budget actually closed, and may its residual be called mixing?

The whole point of a water mass budget is that its residual is interpretable: what
the mass budget says the transformation *was* (``realized_transformation``), minus
what the model's parameterized processes account for (``material_transformation``),
is the transformation the numerics produced without anyone asking -- spurious
numerical mixing.

That reading is only valid if nothing else is missing. Every term that belongs in
the budget and is not there instead lands in the residual, wearing the name of
something it is not. The failure is silent and the result is plausible, which is
the worst combination: an unlabelled surface mass flux, or a heat-budget diffusion
term the dataset never supplied, comes back as a confident spurious-mixing estimate
with the wrong magnitude.

So xwmb always computes the residual -- it is a real, useful diagnostic of budget
imbalance either way -- but only *names* it ``spurious_numerical_mixing`` once
everything else is accounted for:

* every input the recipe names for the relevant budgets was supplied
  (``xbudget.BudgetQuery.missing`` / ``incomplete_terms``), and
* each of dM/dt (storage), Psi (convergent transport) and S (surface mass source)
  is either present or legitimately zero.

"Legitimately zero" is a real category, not an excuse. A full-domain budget has no
boundary to transport mass across, so Psi vanishes by assertion; a recipe that
declares no surface mass exchange (a rigid-lid or volume-conserving configuration)
has no S to be missing. Both are distinguished here from a term the recipe *does*
declare that simply failed to materialize.
"""

import warnings
from dataclasses import dataclass, field

__all__ = ["CompletenessReport", "budget_completeness"]

#: The budget terms that must be accounted for before a residual can be called
#: spurious numerical mixing, with the symbol each is known by.
REQUIRED_TERMS = {
    "mass_tendency": "dM/dt (mass storage)",
    "convergent_mass_transport": "Psi (convergent mass transport)",
    "mass_source": "S (surface mass source)",
}


@dataclass
class CompletenessReport:
    """What, if anything, is unaccounted for in a water mass budget."""

    #: ``{term path: [recipe-named inputs the dataset did not supply]}``.
    missing_inputs: dict = field(default_factory=dict)
    #: Materialized variables xbudget flagged ``xbudget_incomplete``.
    incomplete_terms: list = field(default_factory=list)
    #: Budget terms that are absent and should not be.
    absent_terms: list = field(default_factory=list)
    #: Budget terms that are absent for a legitimate reason, with that reason.
    zero_terms: dict = field(default_factory=dict)

    @property
    def is_complete(self):
        """Whether the residual may be attributed to spurious numerical mixing."""
        return not (self.missing_inputs or self.incomplete_terms or self.absent_terms)

    def unaccounted(self):
        """One short phrase per gap, for a warning message or an attribute.

        Only the *root* causes: a term whose own child is also reported missing is
        dropped, because "mass/rhs is missing surface_exchange_flux" and
        "mass/rhs/surface_exchange_flux is missing wfo" are one problem described
        twice, and only the second names the input to go and find.
        """
        gaps = [REQUIRED_TERMS.get(term, term) for term in self.absent_terms]
        for path, inputs in sorted(self.root_causes().items()):
            name = "/".join(str(p) for p in path)
            gaps.append(f"{name} (missing input(s): {', '.join(sorted(inputs))})")
        gaps += [f"{name} (incomplete)" for name in sorted(self.incomplete_terms)]
        return gaps

    def root_causes(self):
        """``missing_inputs`` with entries superseded by a deeper one removed."""
        paths = set(self.missing_inputs)
        return {
            path: inputs
            for path, inputs in self.missing_inputs.items()
            if not any(
                other != path and other[: len(path)] == path for other in paths
            )
        }

    def describe(self):
        """A human-readable summary of the gaps, or a statement that there are none."""
        gaps = self.unaccounted()
        if not gaps:
            return "the budget is fully accounted for"
        return "; ".join(gaps)


def relevant_budgets(wmb, lambda_name):
    """Which recipe budgets have to be complete for *this* lambda's budget.

    Always the mass budget, which supplies Psi and S. Plus the tracer budget
    driving the transformation: the like-named one for a tracer lambda, and both
    heat and salt for a density lambda (whose transformation is driven by the two
    through the thermal expansion and haline contraction coefficients).

    A gap in a budget that does not feed this calculation -- an incomplete salt
    budget under a heat-coordinate transformation -- is real, but it is not a
    reason to distrust *this* residual, and reporting it would train the reader to
    ignore the warning.
    """
    budgets = ["mass"]
    if wmb._is_density_lambda(lambda_name):
        budgets += ["heat", "salt"]
    else:
        budgets.append(lambda_name)
    return [b for b in budgets if b in wmb.full_recipe]


def budget_completeness(wmb, ds, lambda_name):
    """Audit a computed budget ``ds`` and return a :class:`CompletenessReport`."""
    from .mass import MASS_SOURCE_PATH, mass_source_varname
    from .transport import transport_varnames

    query = wmb.query
    budgets = relevant_budgets(wmb, lambda_name)

    missing_inputs = {
        path: inputs
        for path, inputs in query.missing().items()
        if path and path[0] in budgets
    }
    # A term is flagged `xbudget_incomplete` when *anything* beneath it dropped an
    # input, so every ancestor of a missing input shows up here too. Report only
    # the ones `missing_inputs` does not already explain -- otherwise a single
    # absent diagnostic is listed once per level of the recipe it sits under.
    explained = {query.var(path) for path in missing_inputs}
    for path in list(missing_inputs):
        for depth in range(1, len(path)):
            try:
                explained.add(query.var(path[:depth]))
            except KeyError:  # pragma: no cover - partial paths are terms
                pass
    incomplete = [
        name
        for name in query.incomplete_terms()
        if any(name.startswith(f"{b}_") for b in budgets) and name not in explained
    ]

    absent, zero = [], {}

    if "mass_tendency" not in ds:
        absent.append("mass_tendency")

    if wmb.assert_zero_transport:
        zero["convergent_mass_transport"] = (
            "the region has no boundary to transport mass across "
            "(`assert_zero_transport`)"
        )
    elif "convergent_mass_transport" not in ds:
        absent.append("convergent_mass_transport")
    elif transport_varnames(query) is None:
        absent.append("convergent_mass_transport")

    if "mass_source" in ds:
        pass
    elif mass_source_varname(query) is None and not _declares(query, MASS_SOURCE_PATH):
        zero["mass_source"] = "the recipe declares no surface mass exchange flux"
    else:
        absent.append("mass_source")

    return CompletenessReport(
        missing_inputs=missing_inputs,
        incomplete_terms=incomplete,
        absent_terms=absent,
        zero_terms=zero,
    )


def _declares(query, path):
    """Whether the recipe declares a term at ``path`` at all."""
    try:
        query.var(path)
    except KeyError:
        return False
    return True


def warn_incomplete(report, lambda_name=None):
    """Explain, once, why the residual is not being called spurious mixing."""
    subject = "budget" if lambda_name is None else f"{lambda_name} budget"
    warnings.warn(
        f"The {subject} is not closed, so its residual is not attributed to "
        f"spurious numerical mixing: {report.describe()}. `residual` is still "
        f"computed and carries `xwmb_unaccounted_terms`, but it is the budget "
        f"imbalance -- read it as a diagnostic of what is missing, not as an "
        f"estimate of mixing.",
        stacklevel=3,
    )
