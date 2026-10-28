"""Shared, sky-keyed store of per-band star-removed PS1 skycells.

Companion of ``combined_store`` (design contract:
``dev_runs/s26_ownwcs_20261027/band_cells_design.md``). The combined cell is

    C = Z * sum_b w_b F_b,

with ``Z`` the 0/1 keep mask of ``band_utils.remove_background``. This store keeps the
four weighted band cells ``C_b = Z * w_b F_b`` (with the combined cell's NaN pattern added to
every band) under

    {data_root}/ps1_skycells_zarr/ps1_band_cells.zarr/{projection}/{skycell}/{fp}/

where ``fp`` is **the fingerprint of the combined cell of the same recipe**
(``combined_store.expected_combined_fingerprint``), so every lookup resolves through
``combined_store.resolve_combined_fingerprint_for_recipe`` unchanged. Layout, atomic
publish (``common.provenance.publish.try_publish_dir``), idempotence and the provenance sidecar
mirror ``combined_store.publish_combined_cell`` line by line. Only the sidecar ``kind``
differs (``KIND = "ps1_band_cells"``); the recipe and the fingerprint are the combined ones.

Payload: ``arrays.npz {band_r, band_i, band_z, band_y, combined_mask}``, ``headers.json``,
``removed_stars.json``; ``band_weights`` and ``band_dtype`` are recorded in the sidecar ``meta``.

Bitwise sum identity
--------------------
The combiner accumulates ``combined += float64_term`` into a float32 array, i.e. each add is done
in float64 and rounded once. ``perband.sum_bands`` reproduces that exactly when given the float64
weighted bands (``band_dtype="float64"``: ``combined_from_band_cells`` is then BITWISE the combined
image). With float32 band cells (the default, half the bytes) every term is rounded to float32 first
and the float32 sum differs from the combined image by at most 4 float32 ulp per pixel on roughly 30% of pixels (1 ulp typical); see ``tests/test_band_cell_store.py``.
"""
from __future__ import annotations

import json
import logging
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


def band_cells_complete(data_root: str | Path, projection: str, skycell: str, fp: str) -> bool:
    """True when the band-cell dir of ``fp`` has every required member."""
    return _payload_complete(band_cell_dir(data_root, projection, skycell, fp))


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
) -> dict | None:
    """Atomically publish the band cells of one skycell under the combined cell's fingerprint.

    ``band_images`` maps band name -> ``C_b = Z * w_b F_b`` (all four bands expected; the
    combined NaN pattern must already be in every band). ``recipe`` / ``input_fingerprints`` /
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

        final_dir = band_cell_dir(data_root, projection, skycell, fp)
        dest_root = final_dir.parent

        if _payload_complete(final_dir):
            return {"fingerprint": fp, "recipe_id": rid, "dir": final_dir, "already_published": True}

        weights = perband.band_weights_from_recipe(recipe) if recipe.get("band_weights") else None

        def _write_payload(tmp_dir: Path) -> None:
            arrays = {
                f"band_{b}": np.asarray(band_images[b], dtype=np.dtype(band_dtype)) for b in BANDS
            }
            np.savez_compressed(
                tmp_dir / _ARRAYS_FILENAME,
                combined_mask=np.asarray(combined_mask, dtype=np.uint16),
                **arrays,
            )
            with open(tmp_dir / _HEADERS_FILENAME, "w") as fh:
                json.dump(dict(headers_data), fh, default=combined_store._json_default)
            with open(tmp_dir / _REMOVED_STARS_FILENAME, "w") as fh:
                json.dump(list(removed_stars), fh, default=combined_store._json_default)

        meta = {"band_weights": weights, "band_dtype": band_dtype}
        if combined_store._HAVE_PROVENANCE_PUBLISH:
            published_dir = combined_store._prov_try_publish_dir(  # type: ignore[attr-defined]
                dest_root,
                fp,
                KIND,
                spatial_key,
                rid,
                code_version,
                input_fps,
                _write_payload,
                recipe_params=dict(recipe),
                spool_dir=combined_store._spool_dir(data_root),
                produced_by=producer,
                meta=meta,
            )
            if published_dir is None:
                return None
        else:  # pragma: no cover - provenance package always present in this tree
            raise RuntimeError("common.provenance.publish is required for band cells")

        return {"fingerprint": fp, "recipe_id": rid, "dir": published_dir, "already_published": False}
    except Exception:
        logger.warning(
            "publish_band_cells failed for projection=%s skycell=%s (best-effort, no raise)",
            projection,
            skycell,
            exc_info=True,
        )
        return None


def _read_band_weights(cell_dir: Path) -> Optional[dict]:
    try:
        with open(cell_dir / _PROVENANCE_SIDECAR_FILENAME) as fh:
            rec = json.load(fh)
        w = (rec.get("meta") or {}).get("band_weights")
        if w is None:
            w = (rec.get("recipe_params") or {}).get("band_weights")
        return None if w is None else {b: float(w[b]) for b in BANDS}
    except Exception:
        return None


def try_load_band_cells(
    data_root: str | Path, projection: str, skycell: str, fp: str
) -> dict | None:
    """Load a published band-cell dir, or ``None`` if absent/incomplete/corrupt.

    Returns ``{band_images: {b: C_b}, combined_mask, headers_data, removed_stars, band_weights}``
    (``band_weights`` is ``None`` if the sidecar records none). The band arrays keep the stored
    dtype.
    """
    cell_dir = band_cell_dir(data_root, projection, skycell, fp)
    try:
        if not _payload_complete(cell_dir):
            return None
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
            "band_weights": _read_band_weights(cell_dir),
        }
    except Exception:
        logger.warning("try_load_band_cells failed for %s (best-effort)", cell_dir, exc_info=True)
        return None


def combined_from_band_cells(loaded: Mapping[str, Any]) -> np.ndarray:
    """The combined image from loaded band cells: ``perband.sum_bands`` (float32, r,i,z,y order).

    Bitwise the stored combined image for float64 band cells; within 4 float32 ulp per pixel for float32
    band cells (module docstring).
    """
    return perband.sum_bands(loaded["band_images"])


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
    weighted_bands: Mapping[str, np.ndarray],
    combined_before: np.ndarray,
    combined_after: np.ndarray,
    dtype: str = DEFAULT_BAND_DTYPE,
) -> dict[str, np.ndarray]:
    """``C_b = Z * w_b F_b`` with the combined NaN pattern in every band.

    ``weighted_bands`` are ``perband.weighted_band_images`` outputs, ``combined_before`` the
    combined image fed to ``remove_background`` and ``combined_after`` its result. Raises
    ``ValueError`` if ``remove_background`` changed any pixel other than by zeroing it (then
    ``sum_b C_b`` could not equal the combined cell).
    """
    zeroed = perband.zeroed_mask(combined_before, combined_after)
    nanmask = np.isnan(combined_after)
    expected = np.where(zeroed, np.float32(0), combined_before)
    same = (expected == combined_after) | (np.isnan(expected) & nanmask)
    if not bool(same.all()):
        raise ValueError(
            f"remove_background changed {int((~same).sum())} pixels other than by zeroing; "
            "band cells would not sum to the combined cell"
        )
    dt = np.dtype(dtype)
    out: dict[str, np.ndarray] = {}
    for b in BANDS:
        if b not in weighted_bands:
            continue
        img = np.where(zeroed, np.float64(0) if dt == np.float64 else np.float32(0), weighted_bands[b])
        img = np.asarray(img, dtype=dt)
        img[nanmask] = np.nan
        out[b] = img
    return out
