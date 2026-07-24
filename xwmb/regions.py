"""Normalize the many accepted ``region`` inputs to a single canonical object.

``WaterMassBudget`` accepts a region as a ``regionate.GriddedRegion``, a
``regionate.MaskRegion``, a ``(lons, lats)`` tuple, a boolean ``xr.DataArray`` mask,
or ``None`` (the full grid domain). Downstream budget code only ever needs two
things from a region:

* ``mask`` — a boolean ``xr.DataArray`` selecting the region's cells, and
* ``boundaries`` — a list of boundary loops, each exposing ``i_c``/``j_c``/``f_c``
  vorticity-point indices to hand to :mod:`sectionate`.

With regionate >= 0.6.0 a region's boundary is a first-class
``sectionate.GriddedSection`` (or a *list* of them, one per enclosing loop, for a
``MaskRegion``), and every index array is accompanied by a per-corner face index
``f_c`` (``None`` on single-tile grids) so that transports are computed correctly on
multi-tile ``face_connections`` grids.
"""

import warnings
from dataclasses import dataclass

import xarray as xr
import regionate

from .coordinates import horizontal_grid

__all__ = ["RegionBoundary", "normalize_region"]


@dataclass
class BoundaryLoop:
    """A single closed boundary loop, as vorticity-point (corner) indices."""

    i_c: object
    j_c: object
    f_c: object = None


class RegionBoundary:
    """Canonical region: a cell ``mask`` plus a list of boundary loops."""

    def __init__(self, mask, boundaries, assert_zero_transport=False, source=None):
        self.mask = mask
        self.boundaries = list(boundaries)
        #: True for the full-domain case, where the net boundary transport vanishes.
        self.assert_zero_transport = assert_zero_transport
        #: The originating regionate object (if any), for introspection/plotting.
        self.source = source

    @property
    def is_multitile(self):
        return any(b.f_c is not None for b in self.boundaries)


def _loops_from_indices(obj):
    """Wrap an object exposing ``i_c``/``j_c``/``f_c`` (GriddedRegion/GriddedSection)."""
    return BoundaryLoop(obj.i_c, obj.j_c, getattr(obj, "f_c", None))


def _mask_region_class():
    # MaskRegion is flattened into the regionate namespace via ``import *``; guard in
    # case an older regionate is installed.
    return getattr(regionate, "MaskRegion", ())


def normalize_region(region, grid):
    """Return a :class:`RegionBoundary` for any accepted ``region`` input."""
    if region is None:
        xc = grid.axes["X"].coords["center"]
        yc = grid.axes["Y"].coords["center"]
        mask = xr.ones_like(grid._ds[xc] * grid._ds[yc])
        return RegionBoundary(mask, [], assert_zero_transport=True)

    if isinstance(region, RegionBoundary):
        return region

    if isinstance(region, regionate.GriddedRegion):
        # GriddedRegion (and BoundedRegion) expose the boundary as one index set.
        return RegionBoundary(
            region.mask, [_loops_from_indices(region)], source=region
        )

    if isinstance(region, _mask_region_class()):
        return _from_mask_region(region)

    # sectionate/regionate need a horizontal-only grid (see horizontal_grid).
    hgrid = horizontal_grid(grid)

    if isinstance(region, tuple):
        if len(region) != 2:
            raise ValueError("A tuple region must be `(lons, lats)`.")
        lons, lats = region
        gridded = regionate.GriddedRegion("WaterMass", lons, lats, hgrid)
        return RegionBoundary(
            gridded.mask, [_loops_from_indices(gridded)], source=gridded
        )

    if isinstance(region, xr.DataArray):
        return _from_mask(region, hgrid)

    raise TypeError(
        "`region` must be a regionate.GriddedRegion, regionate.MaskRegion, a "
        "(lons, lats) tuple, a boolean xr.DataArray mask, or None; got "
        f"{type(region).__name__}."
    )


def _from_mask_region(region):
    boundaries = [_loops_from_indices(loop) for loop in region.boundaries]
    return RegionBoundary(region.mask, boundaries, source=region)


def _from_mask(mask, grid):
    """Build a region from a boolean mask via topology-aware connected components."""
    components = list(regionate.MaskRegions(mask, grid).region_dict.values())
    if not components:
        raise ValueError("`mask` selects no cells.")
    if len(components) > 1:
        components = sorted(
            components, key=lambda r: int(r.mask.sum()), reverse=True
        )
        warnings.warn(
            f"Mask split into {len(components)} connected components; using the "
            f"largest. Pass a `regionate.MaskRegion` explicitly to select another."
        )
    return _from_mask_region(components[0])
