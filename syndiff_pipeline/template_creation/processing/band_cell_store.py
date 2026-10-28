"""Shared, sky-keyed store of per-band star-removed PS1 skycells (UNWEIGHTED).

Companion of ``combined_store`` (design contract:
``dev_runs/s26_ownwcs_20261027/band_cells_design.md``, revised by decision D1 of
``dev_runs/perband_production_spec_20261028/SPEC.md``). The combined cell is

    C = Z * sum_b w^rem_b F_b,

with ``Z`` the 0/1 keep mask of ``band_utils.remove_background`` and ``w^rem_b`` the removal/detection
weights of the combined recipe (default 0.238/0.344/0.283/0.135). This store keeps the four
UNWEIGHTED, flux-converted band cells ``Z * F_b`` (float32, the combined cell's NaN pattern added to
every band) under

    {data_root}/ps1_skycells_zarr/ps1_band_cells.zarr/{projection}/{skycell}/{fp}/

where ``fp`` is **the fingerprint of the combined cell of the same recipe**
(``combined_store.expected_combined_fingerprint``; it keeps the REMOVAL weights and never changes
with template weights or zpcell factors), so every lookup resolves through
``combined_store.resolve_combined_fingerprint_for_recipe`` unchanged. Layout, atomic publish
(``common.provenance.publish.try_publish_dir``), idempotence and the provenance sidecar mirror
``combined_store.publish_combined_cell``. Only the sidecar ``kind`` differs
(``KIND = "ps1_band_cells"``).

Payload: ``arrays.npz {band_r, band_i, band_z, band_y, combined_mask}``, ``headers.json``,
``removed_stars.json``; the sidecar ``meta`` records ``band_weighted: false``,
``removal_band_weights`` and ``band_dtype``. ``try_load_band_cells`` REFUSES a cell whose meta lacks
``band_weighted: false`` (weighted pilot cells must never be mixed in).

Read-time weighting
-------------------
Template weights ``w_b`` (and optional per-cell zpcell factors ``f_b``) are applied when a cell is
read: ``plane = sum_b (w_b / f_b) Z F_b`` in float64 (``perband.weighted_sum``).
``combined_from_band_cells(loaded)`` uses the recorded removal weights, sums in float64 and casts to
float32: within 4 float32 ulp of the stored combined cell (the combiner accumulates float64 terms into a
float32 accumulator, one rounding per band; we round the stored bands and the final sum). For float64
band cells the difference is at the 1e-16 level.
"""
from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from syndiff_pipeline.template_creation.processing import combined_store, perband

logger = logging.getLogger(__name__)

KIND = "ps1_band_cells"
BAND_CELLS_ZARR_BASENAME = "ps1_band_cells.zarr"
BANDS = perband.BANDS
#: dtype of the stored band cells ("float32" or "float64"; see the module docstring).
DEFAULT_BAND_DTYPE = "float32"
#: zlib level of ``arrays.npz`` (``np.savez_compressed`` hard-codes the default, 6). Measured on 6 real cells
#: (round2_README.md): level 3 = 2.4 s/cell and 27.3 MB, default 6 = 4.3 s and 24.0 MB, level 9 = 10.8 s and 23.9 MB.
BAND_CELL_ZLIB_LEVEL = 3

_ARRAYS_FILENAME = "arrays.npz"
_HEADERS_FILENAME = "headers.json"
_REMOVED_STARS_FILENAME = "removed_stars.json"
_PROVENANCE_SIDECAR_FILENAME = "_provenance.json"
_REQUIRED_MEMBERS = (_ARRAYS_FILENAME, _HEADERS_FILENAME, _REMOVED_STARS_FILENAME)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _band_cells_zarr_root(data_root: str | Path) -> Path:
    return (
        Path(data_root).expanduser()
        / combined_store.PS1_SKYCELLS_ZARR_DIRNAME
        / BAND_CELLS_ZARR_BASENAME
    )


def band_cells_root(data_root: str | Path) -> Path:
    """``{data_root}/ps1_skycells_zarr/ps1_band_cells.zarr``."""
    return _band_cells_zarr_root(data_root)


def band_cell_dir(data_root: str | Path, projection: str, skycell: str, fp: str) -> Path:
    """``{data_root}/ps1_skycells_zarr/ps1_band_cells.zarr/{projection}/{skycell}/{fp}/``."""
    return _band_cells_zarr_root(data_root) / str(projection) / str(skycell) / str(fp)


def _payload_complete(cell_dir: Path) -> bool:
    if not cell_dir.is_dir():
        return False
    return all((cell_dir / name).is_file() for name in _REQUIRED_MEMBERS)


def _is_unweighted(cell_dir: Path) -> bool:
    meta = _read_meta(cell_dir)
    return meta is not None and meta.get("band_weighted") is False


def band_cells_complete(data_root: str | Path, projection: str, skycell: str, fp: str) -> bool:
    """True when the band-cell dir of ``fp`` has every required member AND is unweighted
    (``band_weighted: false`` in the sidecar meta; weighted/legacy cells do not count)."""
    d = band_cell_dir(data_root, projection, skycell, fp)
    return _payload_complete(d) and _is_unweighted(d)


def resolve_band_cells_fingerprint_for_recipe(
    data_root: str | Path,
    projection: str,
    skycell: str,
    recipe: Mapping,
    *,
    raw_fp: str | None = None,
) -> str | None:
    """The combined-cell fingerprint of ``recipe`` if the BAND cells of that fingerprint are
    published and complete, else ``None`` (same contract as
    ``combined_store.resolve_combined_fingerprint_for_recipe``)."""
    try:
        fp = combined_store.expected_combined_fingerprint(
            data_root, projection, skycell, recipe, raw_fp=raw_fp
        )
    except Exception:
        logger.warning(
            "resolve_band_cells_fingerprint_for_recipe failed for %s/%s", projection, skycell,
            exc_info=True,
        )
        return None
    if fp is None or not band_cells_complete(data_root, projection, skycell, fp):
        return None
    return fp


def band_record_fingerprint(
    projection: str,
    skycell: str,
    recipe: Mapping,
    input_fingerprints: Iterable[str],
    fp: str,
    code_version: int = combined_store.COMBINED_RECIPE_SCHEMA_VERSION,
) -> tuple[str, str]:
    """``(record_fingerprint, record_recipe_id)`` of the PROVENANCE record of a band-cell dir.

    The directory is named by the combined cell's fingerprint ``fp`` (so lookups resolve through the
    combined store unchanged), but the ``artifacts`` table has the fingerprint as primary key and
    ingest does INSERT OR REPLACE: a record with the same fingerprint would overwrite the combined
    cell's row. The record is therefore keyed by the Merkle name of the ``ps1_band_cells`` kind,
    ``fingerprint(KIND, spatial_key, recipe_id(KIND, recipe), inputs + [fp])`` (the combined cell is
    one of its inputs).
    """
    rid = combined_store._compute_recipe_id(KIND, recipe, code_version)
    inputs = sorted(set(input_fingerprints)) + [str(fp)]
    spatial_key = combined_store._spatial_key(projection, skycell)
    return combined_store._compute_fingerprint(KIND, spatial_key, rid, inputs), rid


def band_cell_record_fingerprint(cell_dir: str | Path) -> Optional[str]:
    """The provenance record fingerprint stored in a band-cell dir's sidecar, or ``None``."""
    try:
        with open(Path(cell_dir) / _PROVENANCE_SIDECAR_FILENAME) as fh:
            return str(json.load(fh)["fingerprint"])
    except Exception:
        return None


def required_band_cells(skycell_df, cells: Iterable[str]) -> list[str]:
    """Every cell whose band cells a template over ``cells`` reads: the cells, their canonical
    same-projection neighbours and their cross-projection padding sources (f01_select's
    ``cells | same_nbrs | xproj_src``). ``skycell_df``: mapping dataframe indexed by NAME."""
    from syndiff_pipeline.template_creation.processing.padding_correction import (
        cross_projection_padding_spec,
    )
    from syndiff_pipeline.template_creation.processing.ps1_process import (
        _find_projection_neighbors,
        extract_projection_metadata,
    )

    cells = sorted(cells)
    need = set(cells)
    md_cache: dict = {}
    for name in cells:
        proj = str(skycell_df.loc[name, "projection"])
        if proj not in md_cache:
            md_cache[proj] = extract_projection_metadata(skycell_df, proj)
        nb = _find_projection_neighbors(
            md_cache[proj], name, int(skycell_df.loc[name, "y"]), int(skycell_df.loc[name, "x"])
        )
        need |= {n for n, _, _ in nb}
        need |= {s["neighbor"] for s in cross_projection_padding_spec(skycell_df.loc[name])}
    return sorted(need)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray], compresslevel: int) -> None:
    """``np.savez_compressed`` with an explicit zlib level (same ``.npz`` layout; ``np.load`` reads it)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=int(compresslevel)) as zf:
        for key, arr in arrays.items():
            with zf.open(f"{key}.npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)


# ---------------------------------------------------------------------------
# Publish / load
# ---------------------------------------------------------------------------


def publish_band_cells(
    data_root: str | Path,
    projection: str,
    skycell: str,
    *,
    band_images: Mapping[str, np.ndarray],
    combined_mask: np.ndarray,
    headers_data: Mapping[str, str],
    removed_stars: list,
    recipe: Mapping,
    input_fingerprints: Iterable[str] = (),
    code_version: int = combined_store.COMBINED_RECIPE_SCHEMA_VERSION,
    producer: str = "template_creation.processing.band_cell_store.publish_band_cells",
    band_dtype: str = DEFAULT_BAND_DTYPE,
    missing_bands: Iterable[str] = (),
    zlib_level: Optional[int] = None,
) -> dict | None:
    """Atomically publish the band cells of one skycell under the combined cell's fingerprint.

    ``band_images`` maps band name -> ``Z * F_b`` (UNWEIGHTED flux-converted; all four bands present; the
    combined NaN pattern must already be in every band). A band the PS1 cell does not have is passed as
    an all-zero array with the NaN pattern (``build_band_cells`` does that) and named in
    ``missing_bands``, which is recorded in the sidecar meta: it contributes 0, exactly as the combiner
    skips a missing band. ``recipe`` / ``input_fingerprints`` /
    ``code_version`` are those of the combined cell (the fingerprint is computed from them with
    ``combined_store``). Never raises; returns ``{fingerprint, recipe_id, dir, already_published}``
    or ``None`` on failure.
    """
    try:
        if band_dtype not in ("float32", "float64"):
            raise ValueError(f"band_dtype must be float32 or float64, not {band_dtype!r}")
        missing = [b for b in BANDS if b not in band_images]
        if missing:
            raise ValueError(f"band_images lacks bands {missing}")
        input_fps = sorted(set(input_fingerprints))
        rid = combined_store.combined_recipe_id(recipe, code_version)
        fp = combined_store.combined_fingerprint(projection, skycell, rid, input_fps)
        spatial_key = combined_store._spatial_key(projection, skycell)

        record_fp, record_rid = band_record_fingerprint(
            projection, skycell, recipe, input_fps, fp, code_version
        )
        final_dir = band_cell_dir(data_root, projection, skycell, fp)
        dest_root = final_dir.parent

        if _payload_complete(final_dir):
            if not _is_unweighted(final_dir):
                logger.error(
                    "publish_band_cells: %s holds a weighted/legacy band cell (meta lacks "
                    "band_weighted=false); not overwriting the immutable store", final_dir,
                )
                return None
            return {
                "fingerprint": fp, "recipe_id": rid, "dir": final_dir, "already_published": True,
                "record_fingerprint": record_fp,
            }

        weights = (
            perband.band_weights_from_recipe(recipe)
            if recipe.get("band_weights")
            else dict(perband.DEFAULT_BAND_WEIGHTS)
        )

        def _write_payload(tmp_dir: Path) -> None:
            arrays = {
                f"band_{b}": np.asarray(band_images[b], dtype=np.dtype(band_dtype)) for b in BANDS
            }
            _write_npz(
                tmp_dir / _ARRAYS_FILENAME,
                {**arrays, "combined_mask": np.asarray(combined_mask, dtype=np.uint16)},
                BAND_CELL_ZLIB_LEVEL if zlib_level is None else zlib_level,
            )
            with open(tmp_dir / _HEADERS_FILENAME, "w") as fh:
                json.dump(dict(headers_data), fh, default=combined_store._json_default)
            with open(tmp_dir / _REMOVED_STARS_FILENAME, "w") as fh:
                json.dump(list(removed_stars), fh, default=combined_store._json_default)

        missing = sorted(set(missing_bands))
        if any(b not in BANDS for b in missing):
            raise ValueError(f"missing_bands must be bands of {BANDS}: {missing}")
        meta = {
            "band_weighted": False, "removal_band_weights": weights, "band_dtype": band_dtype,
            "missing_bands": missing, "combined_fingerprint": fp,
        }
        if combined_store._HAVE_PROVENANCE_PUBLISH:
            published_dir = combined_store._prov_try_publish_dir(  # type: ignore[attr-defined]
                dest_root,
                fp,
                KIND,
                spatial_key,
                record_rid,
                code_version,
                input_fps + [fp],
                _write_payload,
                recipe_params=dict(recipe),
                spool_dir=combined_store._spool_dir(data_root),
                produced_by=producer,
                meta=meta,
                record_fingerprint=record_fp,
            )
            if published_dir is None:
                return None
        else:  # pragma: no cover - provenance package always present in this tree
            raise RuntimeError("common.provenance.publish is required for band cells")

        return {
            "fingerprint": fp, "recipe_id": rid, "dir": published_dir, "already_published": False,
            "record_fingerprint": record_fp,
        }
    except Exception:
        logger.warning(
            "publish_band_cells failed for projection=%s skycell=%s (best-effort, no raise)",
            projection,
            skycell,
            exc_info=True,
        )
        return None


def _read_meta(cell_dir: Path) -> Optional[dict]:
    try:
        with open(cell_dir / _PROVENANCE_SIDECAR_FILENAME) as fh:
            rec = json.load(fh)
        return dict(rec.get("meta") or {})
    except Exception:
        return None


def try_load_band_cells(
    data_root: str | Path, projection: str, skycell: str, fp: str
) -> dict | None:
    """Load a published band-cell dir, or ``None`` if absent/incomplete/corrupt/not unweighted.

    A cell whose sidecar meta lacks ``band_weighted: false`` (the weighted S26 pilot cells, or
    anything not written by this module) is REFUSED: ``None`` plus a warning, never mixed in.

    Returns ``{band_images: {b: Z F_b}, combined_mask, headers_data, removed_stars,
    removal_band_weights}``. The band arrays keep the stored dtype.
    """
    cell_dir = band_cell_dir(data_root, projection, skycell, fp)
    try:
        if not _payload_complete(cell_dir):
            return None
        meta = _read_meta(cell_dir)
        if meta is None or meta.get("band_weighted") is not False:
            logger.warning(
                "try_load_band_cells: refusing %s: sidecar meta lacks band_weighted=false "
                "(weighted/legacy band cells must not be used)", cell_dir,
            )
            return None
        rw = meta.get("removal_band_weights")
        removal = None if rw is None else perband.validate_weights(rw, "removal_band_weights")
        with np.load(cell_dir / _ARRAYS_FILENAME, allow_pickle=False) as z:
            band_images = {b: np.array(z[f"band_{b}"]) for b in BANDS}
            combined_mask = np.array(z["combined_mask"])
        with open(cell_dir / _HEADERS_FILENAME) as fh:
            headers_data = json.load(fh)
        with open(cell_dir / _REMOVED_STARS_FILENAME) as fh:
            removed_stars = json.load(fh)
        return {
            "band_images": band_images,
            "combined_mask": combined_mask,
            "headers_data": headers_data,
            "removed_stars": removed_stars,
            "removal_band_weights": removal,
            "missing_bands": list(meta.get("missing_bands") or []),
        }
    except Exception:
        logger.warning("try_load_band_cells failed for %s (best-effort)", cell_dir, exc_info=True)
        return None


def combined_from_band_cells(
    loaded: Mapping[str, Any], weights: Optional[Mapping[str, float]] = None
) -> np.ndarray:
    """``sum_b w_b Z F_b`` accumulated in float64, cast to float32.

    ``weights`` default: the removal weights recorded in the cell's meta, which reproduces the stored
    combined cell within 4 float32 ulp. No hidden global: a loaded cell without recorded weights needs
    explicit ``weights``.
    """
    if weights is None:
        weights = loaded.get("removal_band_weights")
        if weights is None:
            raise ValueError("band cells record no removal weights; pass weights explicitly")
    return perband.weighted_sum(loaded["band_images"], weights).astype(np.float32)


def try_load_combined_cell_from_band_cells(
    data_root: str | Path, projection: str, skycell: str, fp: str
) -> dict | None:
    """Drop-in for ``combined_store.try_load_combined_cell`` that reads band cells:
    ``{combined_image, combined_mask, headers_data, removed_stars}`` (``band_cache`` shape)."""
    loaded = try_load_band_cells(data_root, projection, skycell, fp)
    if loaded is None:
        return None
    return {
        "combined_image": combined_from_band_cells(loaded),
        "combined_mask": loaded["combined_mask"],
        "headers_data": loaded["headers_data"],
        "removed_stars": loaded["removed_stars"],
    }


def build_band_cells(
    bands: Mapping[str, np.ndarray],
    combined_before: np.ndarray,
    combined_after: np.ndarray,
    dtype: str = DEFAULT_BAND_DTYPE,
) -> dict[str, np.ndarray]:
    """``Z * F_b`` (UNWEIGHTED) with the combined NaN pattern in every band.

    ``bands`` are the flux-converted bands (``perband.weighted_band_images`` with
    ``perband.UNIT_WEIGHTS``), ``combined_before`` the (removal-weighted) combined image fed to
    ``remove_background`` and ``combined_after`` its result. ``Z`` is recovered with
    ``perband.zeroed_mask``. Raises ``ValueError`` if ``remove_background`` changed any pixel other
    than by zeroing it (then the cells could not reproduce the combined cell). Bands absent from
    ``bands`` come back as zero planes (see below); the caller records them in ``missing_bands``.
    """
    zeroed = perband.zeroed_mask(combined_before, combined_after)
    nanmask = np.isnan(combined_after)
    expected = np.where(zeroed, np.float32(0), combined_before)
    same = (expected == combined_after) | (np.isnan(expected) & nanmask)
    if not bool(same.all()):
        raise ValueError(
            f"remove_background changed {int((~same).sum())} pixels other than by zeroing; "
            "band cells would not reproduce the combined cell"
        )
    dt = np.dtype(dtype)
    out: dict[str, np.ndarray] = {}
    for b in BANDS:
        if b not in bands:
            # The PS1 cell has no such band: the combiner skips it (contribution 0), so the cell
            # stores an all-zero plane (with the combined NaN pattern) and records it in
            # ``missing_bands`` (see publish_band_cells).
            img = np.zeros(np.shape(combined_after), dtype=dt)
        else:
            img = np.where(zeroed, np.float64(0) if dt == np.float64 else np.float32(0), bands[b])
            img = np.asarray(img, dtype=dt)
        img[nanmask] = np.nan
        out[b] = img
    return out
