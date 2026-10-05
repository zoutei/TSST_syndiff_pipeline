"""The canonical same-projection convolved PS1 cell, defined once.

A canonical cell is the PSF-blurred, same-projection-only padded image of one
PS1 skycell (``convolved_store``, padding mode ``same_projection_only_v2``).
Before this module, three code paths built it three different ways (the
sliding-row snapshot, the per-skycell sparse path, and the investigation
replicas), and the stored pixels depended on the publishing run's row order
and mapping list (``doc/seam_neighbour_fix_plan_20260930.md``). This module is
the single reference: the row path, the sparse path, readers that compute the
expected fingerprint, and verify's spot-check all call it.

Geometry (all in one projection's "master row" frame):

- Every row of a projection is anchored at the projection's minimum grid
  column ``anchor_x``. PS1 rows are column-aligned by ``x`` (cell (r, x) and
  (r+1, x) share their vertical overlap), so cross-row padding is only correct
  when both rows use the same anchor. Before v2 each row was anchored at its
  own first ``x``; adjacent rows that start at different ``x`` (25 of 113 row
  pairs on S24 C2K2) then copied the wrong neighbour into the top/bottom pad.
- Cell ``(r, x)`` occupies master columns ``[x0, x0 + W)`` with
  ``x0 = PAD + (x - anchor_x) * (W - CELL_OVERLAP)``. Cells are written left
  to right; a cell whose left neighbour ``x - 1`` is present in the row is
  written from column ``EFFECTIVE_OVERLAP`` of its own image, so the left
  cell supplies the shared strip. A cell with no left neighbour is written in
  full.
- Cross-row padding: rows ``[0, PAD + EE)`` of row R's master come from row
  R-1, rows ``[PAD + H - EE, H + 2 PAD)`` from row R+1, column for column. A
  missing adjacent row leaves that pad NaN (it is never copied from a NaN
  buffer over the cell's own rows: that was the blanked-top-strip bug).
- NaN -> 0, Gaussian blur (``convolution_utils.apply_gaussian_convolution``,
  truncated at ``radius``), NaN restored, cell region cropped.

Because the kernel is truncated at ``radius`` <= ``PAD`` and every output
pixel of the cell depends only on inputs within ``radius``, blurring the
window ``[x0 - PAD, x0 + W + PAD)`` reproduces the full master row exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Must equal ps1_process / cross_projection_padding (asserted in tests).
PAD_SIZE = 480
CELL_OVERLAP = 480
EDGE_EXCLUSION = 10
EFFECTIVE_OVERLAP = CELL_OVERLAP - EDGE_EXCLUSION

NEIGHBOUR_INPUT_PREFIX = "nbr:"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def projection_anchor_x(metadata: Mapping) -> int:
    """Minimum grid column over every row of the projection's metadata."""
    xs = [int(x) for cells in metadata["rows"].values() for _, x in cells]
    if not xs:
        raise ValueError(f"projection {metadata.get('projection')} has no cells")
    return min(xs)


def projection_span_cells(metadata: Mapping) -> int:
    """Number of grid columns between the projection's min and max ``x`` (inclusive)."""
    xs = [int(x) for cells in metadata["rows"].values() for _, x in cells]
    return max(xs) - min(xs) + 1 if xs else 0


def cell_master_x0(x: int, anchor_x: int, cell_width: int) -> int:
    """Master-row column of cell ``x``'s first pixel."""
    return PAD_SIZE + (int(x) - int(anchor_x)) * (int(cell_width) - CELL_OVERLAP)


def cell_source_x_start(x: int, present_xs: Iterable[int]) -> int:
    """First own column a cell writes: EFFECTIVE_OVERLAP when its left neighbour is placed."""
    return EFFECTIVE_OVERLAP if (int(x) - 1) in {int(v) for v in present_xs} else 0


def master_row_width(metadata: Mapping, cell_width: int) -> int:
    """Width of a master row array that holds every cell of the projection at one anchor."""
    span = projection_span_cells(metadata)
    return PAD_SIZE + span * (int(cell_width) - CELL_OVERLAP) + CELL_OVERLAP + PAD_SIZE


def place_row_cells(
    target: np.ndarray,
    row_cells: Iterable[tuple[str, int]],
    fetch_image: Callable[[str], Optional[np.ndarray]],
    *,
    anchor_x: int,
    cell_width: int,
    col0: int = 0,
) -> list[str]:
    """Write one row's cells into ``target`` (rows ``[PAD, PAD + H)``, master
    columns ``[col0, col0 + target.shape[1])``) with the canonical rule.

    ``target`` must already be NaN-filled. Cells whose image ``fetch_image``
    returns ``None`` are not placed; the left-neighbour rule only counts placed
    cells. Returns the names placed.
    """
    images: dict[str, tuple[int, np.ndarray]] = {}
    for name, x in sorted(row_cells, key=lambda item: int(item[1])):
        img = fetch_image(name)
        if img is not None:
            images[name] = (int(x), img)
    present_xs = {x for x, _ in images.values()}
    col1 = col0 + target.shape[1]
    placed: list[str] = []
    for name, (x, img) in sorted(images.items(), key=lambda item: item[1][0]):
        h, w = img.shape
        src0 = cell_source_x_start(x, present_xs)
        t0 = cell_master_x0(x, anchor_x, cell_width) + src0
        a, b = max(t0, col0), min(t0 + (w - src0), col1)
        if b > a:
            target[PAD_SIZE:PAD_SIZE + h, a - col0:b - col0] = img[:, src0 + (a - t0):src0 + (b - t0)]
        placed.append(name)
    return placed


def apply_cross_row(
    current: np.ndarray,
    previous: Optional[np.ndarray],
    following: Optional[np.ndarray],
    cell_height: int,
) -> None:
    """Fill ``current``'s bottom pad from ``previous`` and top pad from
    ``following`` (column-aligned master arrays of rows R-1 and R+1). A ``None``
    neighbour row leaves that pad untouched (NaN)."""
    if previous is not None:
        current[:PAD_SIZE + EDGE_EXCLUSION] = previous[
            cell_height - CELL_OVERLAP:PAD_SIZE + cell_height - CELL_OVERLAP + EDGE_EXCLUSION
        ]
    if following is not None:
        current[cell_height - EDGE_EXCLUSION + PAD_SIZE:] = following[
            PAD_SIZE + CELL_OVERLAP - EDGE_EXCLUSION:2 * PAD_SIZE + CELL_OVERLAP
        ]


def _row_and_x(metadata: Mapping, cell_name: str) -> tuple[int, int]:
    for row_id, cells in metadata["rows"].items():
        for name, x in cells:
            if name == cell_name:
                return int(row_id), int(x)
    raise KeyError(f"{cell_name} not in projection {metadata.get('projection')} metadata")


def cell_window_columns(metadata: Mapping, cell_name: str) -> tuple[int, int]:
    """Master columns ``[c0, c1)`` of the cell's blur window (cell +- PAD)."""
    _, x = _row_and_x(metadata, cell_name)
    x0 = cell_master_x0(x, projection_anchor_x(metadata), metadata["cell_width"])
    return x0 - PAD_SIZE, x0 + int(metadata["cell_width"]) + PAD_SIZE


def canonical_neighbour_names(metadata: Mapping, cell_name: str) -> list[str]:
    """Cells of rows R-1, R, R+1 whose placed columns intersect the cell's blur
    window (excluding the cell itself): every combined cell whose pixels can
    reach the canonical cell. Computed from the mapping list only, so a reader
    derives exactly what the producer used."""
    row_id, _ = _row_and_x(metadata, cell_name)
    anchor = projection_anchor_x(metadata)
    w = int(metadata["cell_width"])
    c0, c1 = cell_window_columns(metadata, cell_name)
    names: set[str] = set()
    for r in (row_id - 1, row_id, row_id + 1):
        cells = metadata["rows"].get(r) or []
        xs = {int(x) for _, x in cells}
        for name, x in cells:
            if name == cell_name:
                continue
            t0 = cell_master_x0(x, anchor, w) + cell_source_x_start(x, xs)
            t1 = cell_master_x0(x, anchor, w) + w
            if t0 < c1 and t1 > c0:
                names.add(str(name))
    return sorted(names)


# ---------------------------------------------------------------------------
# Reference image
# ---------------------------------------------------------------------------


def canonical_cell_image(
    cell_name: str,
    metadata: Mapping,
    fetch_image: Callable[[str], Optional[np.ndarray]],
    psf_sigma: float,
    radius: int,
) -> Optional[np.ndarray]:
    """The canonical convolved image of ``cell_name`` (float32, cell shape), or
    ``None`` if the cell's own combined image is unavailable.

    ``fetch_image(name)`` returns a combined (post star-removal) image or
    ``None``. Neighbours that return ``None`` stay NaN, which is why the
    convolved fingerprint records the neighbour set (see
    :func:`neighbour_input_fingerprints`).
    """
    from syndiff_pipeline.template_creation.processing import convolution_utils

    if int(radius) > PAD_SIZE:
        raise ValueError(f"radius {radius} exceeds the {PAD_SIZE}-px pad")
    if fetch_image(cell_name) is None:
        return None
    row_id, _ = _row_and_x(metadata, cell_name)
    anchor = projection_anchor_x(metadata)
    w, h = int(metadata["cell_width"]), int(metadata["cell_height"])
    c0, c1 = cell_window_columns(metadata, cell_name)

    def row_window(r: int) -> Optional[np.ndarray]:
        cells = metadata["rows"].get(r)
        if not cells:
            return None
        out = np.full((h + 2 * PAD_SIZE, c1 - c0), np.nan, dtype=np.float32)
        place_row_cells(out, cells, fetch_image, anchor_x=anchor, cell_width=w, col0=c0)
        return out

    current = row_window(row_id)
    apply_cross_row(current, row_window(row_id - 1), row_window(row_id + 1), h)
    nan_mask = np.isnan(current)
    current[nan_mask] = 0.0
    convolved = convolution_utils.apply_gaussian_convolution(current, sigma=psf_sigma, radius=int(radius))
    convolved[nan_mask] = np.nan
    return np.asarray(convolved[PAD_SIZE:PAD_SIZE + h, PAD_SIZE:PAD_SIZE + w], dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _split(name: str) -> tuple[str, str]:
    parts = str(name).split(".")
    if len(parts) < 3:
        raise ValueError(f"not a skycell identity: {name}")
    return ".".join(parts[:2]), parts[2]


def neighbour_input_fingerprints(
    data_root: str | Path,
    cell_name: str,
    metadata: Mapping,
    combined_recipe: Mapping,
    *,
    neighbour_names: Optional[Iterable[str]] = None,
) -> Optional[list[str]]:
    """``["nbr:<skycell>=<combined fp>", ...]`` for the cell's canonical
    neighbours, sorted. These are the convolved cell's extra Merkle inputs
    (its own combined fingerprint is the primary input). ``None`` when any
    neighbour's combined fingerprint is undefined (its projection catalogue
    is missing)."""
    from syndiff_pipeline.template_creation.processing.combined_store import (
        expected_combined_fingerprint,
    )

    names = canonical_neighbour_names(metadata, cell_name) if neighbour_names is None else sorted(neighbour_names)
    out: list[str] = []
    for name in names:
        projection, cell = _split(name)
        fp = expected_combined_fingerprint(data_root, projection, cell, combined_recipe)
        if fp is None:
            return None
        out.append(f"{NEIGHBOUR_INPUT_PREFIX}{name}={fp}")
    return out


def expected_convolved_fingerprint(
    data_root: str | Path,
    cell_name: str,
    metadata: Mapping,
    combined_recipe: Mapping,
    convolved_recipe: Mapping,
) -> Optional[str]:
    """Fingerprint of the canonical cell a run with this mapping list (via
    ``metadata``) and these recipes must use. ``None`` if undefined."""
    from syndiff_pipeline.template_creation.processing.combined_store import (
        expected_combined_fingerprint,
    )
    from syndiff_pipeline.template_creation.processing.convolved_store import (
        convolved_fingerprint,
        convolved_recipe_id,
    )

    projection, cell = _split(cell_name)
    own = expected_combined_fingerprint(data_root, projection, cell, combined_recipe)
    if own is None:
        return None
    nbr = neighbour_input_fingerprints(data_root, cell_name, metadata, combined_recipe)
    if nbr is None:
        return None
    return convolved_fingerprint(projection, cell, convolved_recipe_id(convolved_recipe), sorted({own, *nbr}))


def resolve_canonical_convolved_fp(
    data_root: str | Path,
    cell_name: str,
    metadata: Mapping,
    combined_recipe: Mapping,
    convolved_recipe: Mapping,
) -> Optional[str]:
    """Like :func:`expected_convolved_fingerprint`, but only if that exact
    payload is published (and its upstream combined cell exists). Never falls
    back to another fingerprint."""
    from syndiff_pipeline.template_creation.processing.combined_store import (
        resolve_combined_fingerprint_for_recipe,
    )
    from syndiff_pipeline.template_creation.processing.convolved_store import (
        resolve_convolved_fingerprint_for_recipe,
    )

    projection, cell = _split(cell_name)
    own = resolve_combined_fingerprint_for_recipe(data_root, projection, cell, combined_recipe)
    if own is None:
        return None
    nbr = neighbour_input_fingerprints(data_root, cell_name, metadata, combined_recipe)
    if nbr is None:
        return None
    return resolve_convolved_fingerprint_for_recipe(
        data_root, projection, cell, convolved_recipe, own, extra_input_fingerprints=nbr,
    )


# ---------------------------------------------------------------------------
# Spot check (end of a ps1_process run)
# ---------------------------------------------------------------------------

SPOT_CHECK_RTOL = 1e-6


def spot_check_sample(cell_names: Iterable[str], metadata_by_projection: Mapping[str, Mapping],
                      *, fraction: float = 0.02, minimum: int = 5) -> list[str]:
    """Deterministic sample: every cell in a projection's top or bottom row
    (where the last-row strip and cross-row bugs lived), then every k-th of the
    rest, at least ``minimum`` cells and about ``fraction`` of them."""
    names = sorted({str(n) for n in cell_names})
    edge: list[str] = []
    for name in names:
        md = metadata_by_projection.get(name.split(".")[1])
        if md is None:
            continue
        row_id, _ = _row_and_x(md, name)
        rows = sorted(md["rows"])
        if row_id in (rows[0], rows[-1]):
            edge.append(name)
    target = min(len(names), max(int(minimum), int(np.ceil(fraction * len(names)))))
    n_edge = min(len(edge), max(1, target // 2))
    edge_pick = edge[:: max(1, len(edge) // max(1, n_edge))][:n_edge]
    rest = [n for n in names if n not in set(edge_pick)]
    n_rest = target - len(edge_pick)
    rest_pick = rest[:: max(1, len(rest) // max(1, n_rest))][:n_rest] if n_rest > 0 else []
    return sorted(edge_pick + rest_pick)


def spot_check_cells(
    data_root: str | Path,
    cell_names: Iterable[str],
    mapping_df: pd.DataFrame,
    combined_recipe: Mapping,
    convolved_recipe: Mapping,
    *,
    rtol: float = SPOT_CHECK_RTOL,
) -> list[dict]:
    """Recompute each cell with :func:`canonical_cell_image` from the combined
    store and compare with the stored canonical cell this run's recipes and
    mapping list resolve to. Returns one record per cell with
    ``status`` in ``{"ok", "mismatch", "not_published", "inputs_missing"}`` and
    ``max_rel`` = max |stored - recomputed| / max |stored|."""
    from syndiff_pipeline.template_creation.processing.combined_store import (
        seed_band_cache_from_combined_store,
    )
    from syndiff_pipeline.template_creation.processing.convolved_store import try_load_convolved_cell

    out: list[dict] = []
    for name in cell_names:
        projection, cell = _split(name)
        md = metadata_for_cell(mapping_df, name)
        fp = resolve_canonical_convolved_fp(data_root, name, md, combined_recipe, convolved_recipe)
        if fp is None:
            out.append({"cell": name, "status": "not_published"})
            continue
        stored = try_load_convolved_cell(data_root, projection, cell, fp)
        needed = [name, *canonical_neighbour_names(md, name)]
        cache = seed_band_cache_from_combined_store(data_root, needed, combined_recipe)
        if any(n not in cache for n in needed) or stored is None:
            out.append({"cell": name, "status": "inputs_missing",
                        "missing": sorted(n for n in needed if n not in cache)})
            continue
        recomputed = canonical_cell_image(
            name, md, lambda n: None if n not in cache else np.asarray(cache[n]["combined_image"], np.float32),
            float(convolved_recipe["psf_sigma"]), int(convolved_recipe["radius"]),
        )
        s = np.asarray(stored["convolved_image"], np.float64)
        r = np.asarray(recomputed, np.float64)
        same_nan = bool(np.array_equal(np.isnan(s), np.isnan(r)))
        finite = np.isfinite(s) & np.isfinite(r)
        peak = float(np.max(np.abs(s[finite]))) if finite.any() else 0.0
        max_rel = float(np.max(np.abs(s[finite] - r[finite])) / peak) if peak > 0 else 0.0
        ok = same_nan and max_rel <= rtol
        out.append({"cell": name, "status": "ok" if ok else "mismatch", "max_rel": max_rel,
                    "nan_pattern_equal": same_nan, "fingerprint": fp})
    return out


# ---------------------------------------------------------------------------
# Mapping-list metadata (consumer side)
# ---------------------------------------------------------------------------

# id(DataFrame) -> (weakref to it, {projection: metadata}). The weakref check
# stops a recycled id from serving another mapping list's metadata.
_METADATA_CACHE: dict[int, tuple[object, dict[str, dict]]] = {}


def metadata_for_cell(mapping_df: pd.DataFrame, cell_name: str) -> dict:
    """Projection metadata (``ps1_process.extract_projection_metadata``) of the
    projection holding ``cell_name``, from a consumer's own mapping list.
    Cached per (DataFrame object, projection) because readers call this once
    per cell."""
    import weakref

    from syndiff_pipeline.template_creation.processing.ps1_process import extract_projection_metadata

    projection = str(cell_name).split(".")[1]
    entry = _METADATA_CACHE.get(id(mapping_df))
    if entry is None or entry[0]() is not mapping_df:
        if len(_METADATA_CACHE) > 64:
            _METADATA_CACHE.clear()
        entry = (weakref.ref(mapping_df), {})
        _METADATA_CACHE[id(mapping_df)] = entry
    per_projection = entry[1]
    if projection not in per_projection:
        df = mapping_df.reset_index(drop=True) if "NAME" in mapping_df.columns else mapping_df.reset_index()
        if not {"projection", "x", "y", "NAME"}.issubset(df.columns):
            raise ValueError("mapping list lacks NAME/projection/x/y columns needed for the canonical neighbour set")
        per_projection[projection] = extract_projection_metadata(df, projection)
    return per_projection[projection]
