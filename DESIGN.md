# xwmb v0.7.0 — design notes

This release updates `xwmb` to the topology-aware dependency stack, migrates to the
xbudget 0.8.0 recipe API, gives every derived variable a description, and stops the
budget residual from being called spurious numerical mixing when it is not. The
public API is unchanged apart from the `xbudget_dict` → `recipe` rename, which the
old spelling still satisfies with a `FutureWarning`:

```python
wmb = xwmb.WaterMassBudget(grid, recipe, region=None)
wmt = wmb.mass_budget(lambda_name, greater_than=True)
```

## Dependency updates

| package | old | new | why |
|---|---|---|---|
| xgcm | 0.9 | **>= 0.10.1** | `boundary`→`padding` rename, `periodic` dropped, `cumsum(..., reverse=True)`, north-fold + `face_connections` padding fixes |
| xbudget | (transitive) | **>= 0.8.0** | now a direct dependency. The dict-walking engine and `xbudget.aggregate()` are gone; recipes are read through `BudgetQuery`, which also carries the UDUNITS units xwmb's metadata is composed from |
| sectionate | 0.3 | **>= 0.4.0rc1** | topology-driven `grid_section`/`GriddedSection`; `f_c`-aware `convergent_transport`/`extract_tracer` |
| regionate | 0.5 | **>= 0.6.0rc1** | `GriddedRegion`/`MaskRegion(s)` whose boundaries are `sectionate.GriddedSection`s carrying `i_c/j_c/f_c` |
| xwmt | 0.1 | **>= 0.3.0rc1** | transforms broadcast across tiles via `_facedim`/`_horizontal_dims`; EOS via `xeos`; `xwmt.units` supplies the cf-units algebra |
| xeos | (transitive) | **>= 0.2.2** | the equation of state behind `eos=`, now that xwmb's own API exposes the choice |

The pre-release pins are load-bearing: naming the `rc` explicitly is what lets pip
install one at all (PEP 440 admits pre-releases only for specifiers that mention
one). Bump each to the final release as it lands.

Every dependency now supports **arbitrary `xgcm.Grid` topologies**: single-tile
periodic, bipolar/tripolar north-fold (`padding={"Y": {"fold": ...}}`), and genuinely
multi-tile `face_connections` grids (ECCOv4r4 lat-lon-cap / LLC90). All three are
exercised by data-free synthetic tests, which assert the discrete divergence theorem
for a region straddling the fold seam and one straddling a tile seam.

## What the new APIs let us delete

1. **The `greater_than` reverse-cumsum dance.** The old code wrapped every cumulative-
   in-λ accumulation in `ds.isel({outer: slice(None,None,-1), center: ...})` … `cumsum`
   … `isel(...)` (four copies). xgcm 0.10 `grid.cumsum(..., reverse=True)` does exactly
   this internally, including the grid-position bookkeeping. All four copies collapse to
   one helper, `coordinates.accumulate_in_lambda(...)`.

2. **`boundary=`/`periodic=` everywhere** → `padding=`. `grid.axes[ax]._boundary` →
   `grid.axes[ax].padding`.

3. **Hard-coded single-tile horizontal reductions.** `.sum([xc, yc])` →
   `.sum(self._horizontal_dims)` (inherited from `xwmt.WaterMass`; includes the face
   dim on multi-tile grids, so integrations broadcast across tiles automatically).

4. **The bespoke region branching.** The region normalization (tuple / mask / None /
   `GriddedRegion` / `MaskRegion`) moves to `regions.normalize_region`, which always
   yields a canonical object exposing `.mask` and `.boundaries` (a list of
   `sectionate.GriddedSection`), threading `f_c` for multi-tile grids.

## Module layout

```
xwmb/
  __init__.py         public exports
  budget.py           WaterMassBudget (thin orchestrator) + mass_budget()
  regions.py          normalize_region() -> RegionBoundary(.mask, .boundaries)
  coordinates.py      target-coord setup + accumulate_in_lambda() (cumsum reverse)
  transformations.py  xwmt integrate/map wrapper + boundary-flux grouping + sign
  transport.py        convergent transport: along-section (sectionate, multi-tile)
                      and grid-cell divergence methods; mass-source term
  mass.py             layer mass, mass-snapshot bounds, mass_tendency
  attrs.py            units and metadata for the derived variables
  completeness.py     is the budget closed? -> CompletenessReport
  close.py            close_budget(): realized -> residual -> spurious mixing
```

Each budget term (`transformations`, `convergent_transport`, `mass_source`,
`layer_mass`, `mass_bounds`, `mass_tendency`) is a function of the vertical tracer
coordinate λ, integrated over the region. `WaterMassBudget.mass_budget` orchestrates:

```
target coords -> transformations G(λ)     [transformations.py]
              -> mass bounds M(λ, t_bnds)  [mass.py]
              -> Ψ(λ), S(λ), layer mass    [transport.py, mass.py]
              -> mass_tendency ∂ₜM(λ)      [mass.py]
              -> audit completeness         [completeness.py]
              -> close budget (residual)    [close.py]
```

## Reading a recipe (xbudget 0.8.0)

`collect_budgets` no longer fills the recipe's `var` fields, and
`xbudget.aggregate()` no longer exists. `WaterMassBudget` builds a
`BudgetQuery(grid, recipe)`, keeps it as `self.query`, and feeds
`query.aggregate(decompose=…)` to `xwmt`. Every variable name xwmb needs is then
resolved through that query rather than hardcoded or dict-walked:

| what | how |
|---|---|
| umo / vmo | `query.get_vars(("mass","rhs","advection","lateral",…))["difference"][0]` |
| surface mass flux | `query.var(("mass","rhs","surface_exchange_flux"))` |
| declared budget units | `query.budget_units("mass")` |
| what did not materialize | `query.missing()`, `query.incomplete_terms()` |

Hand-walking the recipe dict is no longer safe: `var: null` placeholders are gone,
and a string operand may be a reference into the recipe's top-level `constants:`
table rather than the name of a dataset variable.

## Metadata

`attrs.py` gives every derived variable UDUNITS-2 units, a `long_name`, CF
`cell_methods`, and the xbudget provenance of its inputs. The unit algebra is
`xwmt.units` (a `cf_units` wrapper) rather than a second implementation. Units are
*composed* from the inputs — `rho_ref [kg m-3] × h [m] × areacello [m2] → kg`,
`mass_bounds / dt → kg s-1` — and are **omitted** rather than guessed when an input
is unlabelled, with `xwmb_units_source` recording which authority answered.

## Closing the budget honestly

The residual is interpretable as spurious numerical mixing only if nothing else is
missing; anything that belongs in the budget and is not there lands in the residual
wearing the name of something it is not. `completeness.py` audits the budget —
inputs the recipe names that the dataset did not supply, and whether dM/dt, Ψ and S
are present or *legitimately* zero — and `close_budget` emits
`spurious_numerical_mixing` only when the audit is clean. Otherwise it still
computes `residual`, but warns, names each gap, and stamps
`xwmb_unaccounted_terms`.

## Multi-tile (`face_connections`) support

- **Transformations**: handled entirely inside `xwmt` (reduces over `_horizontal_dims`).
- **Convergent transport (along-section)**: `sectionate.convergent_transport(..., f_c=…)`
  threads the per-corner face index; the region boundary is a list of `GriddedSection`
  loops that may wrap tile seams. This is the recommended path and what the ECCO AABW
  example uses.
- **Convergent transport (grid-cell divergence)**: single-tile only for now; raises a
  clear error on multi-tile grids, directing users to `along_section=True`.
- **Horizontal integrals** everywhere use `self._horizontal_dims`.

## Worked example

`examples/ECCO_AABW_watermass_budget.ipynb` downloads the ECCOv4r4 LLC90 data store
(Zenodo `10.5281/zenodo.21479854`) and computes the full σ₂ water-mass budget for
Antarctic Bottom Water south of 30°S — where the AABW density class is defined by the
abyssal minimum of the σ₂ meridional-overturning streamfunction at 30°S.
