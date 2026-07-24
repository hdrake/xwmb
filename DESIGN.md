# xwmb modernization — design notes

This branch (`modernize-topology-stack`) updates `xwmb` to the topology-aware
dependency stack and refactors the previously monolithic `budget.py` into small,
single-responsibility modules. The public API is unchanged:

```python
wmb = xwmb.WaterMassBudget(grid, xbudget_dict, region=None)
wmt = wmb.mass_budget(lambda_name, greater_than=True)
```

## Dependency updates

| package | old | new | why |
|---|---|---|---|
| xgcm | 0.9 | **>= 0.10.1** | `boundary`→`padding` rename, `periodic` dropped, `cumsum(..., reverse=True)`, north-fold + `face_connections` padding fixes |
| sectionate | 0.3 | **>= 0.3.4** | topology-driven `grid_section`/`GriddedSection`; `f_c`-aware `convergent_transport`/`extract_tracer` |
| regionate | 0.5 | **>= 0.6.0** | `GriddedRegion`/`MaskRegion(s)` whose boundaries are `sectionate.GriddedSection`s carrying `i_c/j_c/f_c` |
| xwmt | 0.1 | **>= 0.3.0** | transforms broadcast across tiles via `_facedim`/`_horizontal_dims`; EOS via `xeos` |
| xbudget | (transitive) | **>= 0.6.0** | now a direct dependency (`xbudget.aggregate`) |

All four dependencies now support **arbitrary `xgcm.Grid` topologies**: single-tile
periodic, bipolar/tripolar north-fold (`padding={"Y": {"fold": ...}}`), and genuinely
multi-tile `face_connections` grids (ECCOv4r4 lat-lon-cap / LLC90).

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
  close.py            close_budget(): residual -> spurious numerical mixing
```

Each budget term (`transformations`, `convergent_transport`, `mass_source`,
`layer_mass`, `mass_bounds`, `mass_tendency`) is a function of the vertical tracer
coordinate λ, integrated over the region. `WaterMassBudget.mass_budget` orchestrates:

```
target coords -> transformations G(λ)     [transformations.py]
              -> mass bounds M(λ, t_bnds)  [mass.py]
              -> Ψ(λ), S(λ), layer mass    [transport.py, mass.py]
              -> mass_tendency ∂ₜM(λ)      [mass.py]
              -> close budget (residual)   [close.py]
```

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
