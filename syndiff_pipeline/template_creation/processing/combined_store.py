"""Shared, sky-keyed store for post-star-removal combined PS1 skycells.

Pure data-layer groundwork for **PR4 — shared combined store**
(``doc/template_bookkeeping_plan.md`` §12, decision #14, artifact-kind
registry §6 ``combined_skycell``).

**Validated finding carried over (plan §12):** star removal runs *inside*
``process_single_cell`` downstream of band combine, and the uncertainty array
does not survive it. The shareable artifact is exactly what
``ps1_process.band_cache`` already holds for padding-source cells::

    {
        "combined_image": np.ndarray[float32],   # post band-combine, post star-removal
        "combined_mask": np.ndarray[uint16],     # PS1 bit-mask, bitwise-OR'd across bands
        "headers_data": dict[str, str],          # band -> FITS header string (e.g. "r"/"i"/"z"/"y")
        "removed_stars": list[dict],             # one record per removed segment/star
    }

This module persists exactly that shape cross-run under
``{data_root}/ps1_skycells_zarr/ps1_combined.zarr/{projection}/{skycell}/{fp}/``
so overlapping TESS sectors can skip the raw read + band combine + star
removal for a skycell they've already processed.

Scope note: this module does **not** import ``ps1_process`` and does **not**
wire into it. Live wiring (seeding/publishing from ``process_coordinator``)
is a later, separately-gated step per the plan.

Provenance note: ``common/provenance/`` is being authored concurrently on
this branch. This module imports the frozen contract
(``fingerprint``/``recipe_id``/``publish_record``/``ProvenanceStore``, plan
§9-10) lazily and falls back to a local, spec-shaped implementation when the
package isn't importable yet, so this module is usable standalone today and
upgrades automatically once the real package lands (see
``_HAVE_PROVENANCE_FINGERPRINT`` / ``_HAVE_PROVENANCE_PUBLISH`` below).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact-kind identity (plan §6 registry row: combined_skycell)
# ---------------------------------------------------------------------------

KIND = "combined_skycell"

# Hand-bumped per decision #5 ("code_version"): bump whenever this producer's
# algorithm (payload shape, recipe params) changes in a way that should mint
# new fingerprints for otherwise-identical inputs.
COMBINED_RECIPE_SCHEMA_VERSION = 1

# Same "hand-bumped code_version" contract (decision #5), but for the
# raw_skycell input-fingerprint helper below (bump if the version-token
# shape in ``_raw_skycell_version_token`` ever changes).
RAW_SKYCELL_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Store location (decision #14): all three PS1 stores live under the
# existing ps1_skycells_zarr/ folder. scc_paths.py is owned by a concurrent
# workstream on this branch and is being actively edited there, so the root
# is constructed inline here rather than imported (avoids a load-time
# dependency on a file this module doesn't own). As of this writing that
# workstream has already landed ``scc_paths.ps1_combined_zarr_path`` /
# ``ps1_convolved_zarr_path`` with exactly this basename
# (``ps1_skycells_zarr/ps1_combined.zarr``) -- TODO(PR4 integration): switch
# ``_ps1_combined_zarr_root`` below to call that helper directly once this
# module is wired into the concurrent branch's final state (plan §7, §20).
# ---------------------------------------------------------------------------

PS1_SKYCELLS_ZARR_DIRNAME = "ps1_skycells_zarr"
COMBINED_ZARR_BASENAME = "ps1_combined.zarr"

# Raw grizy PS1 skycell store (plan §7: "ps1_skycells.zarr # raw grizy
# (exists, unchanged)"), read-only from this module's point of view -- used
# only to derive a cheap raw-input version token, never to load pixel data.
RAW_ZARR_BASENAME = "ps1_skycells.zarr"

_ARRAYS_FILENAME = "arrays.npz"
_HEADERS_FILENAME = "headers.json"
_REMOVED_STARS_FILENAME = "removed_stars.json"
_PROVENANCE_SIDECAR_FILENAME = "_provenance.json"
_REQUIRED_MEMBERS = (_ARRAYS_FILENAME, _HEADERS_FILENAME, _REMOVED_STARS_FILENAME)

# Band-combine constants (band_utils.combine_rizy_bands default weights).
DEFAULT_BAND_WEIGHTS: dict[str, float] = {"r": 0.238, "i": 0.344, "z": 0.283, "y": 0.135}


# ---------------------------------------------------------------------------
# Provenance contract (plan §9-10) — lazy, best-effort import.
# ---------------------------------------------------------------------------

try:
    from syndiff_pipeline.common.provenance.fingerprint import (  # type: ignore
        fingerprint as _prov_fingerprint,
        recipe_id as _prov_recipe_id,
    )

    _HAVE_PROVENANCE_FINGERPRINT = True
except Exception:  # pragma: no cover - package mid-authoring on this branch
    _HAVE_PROVENANCE_FINGERPRINT = False

try:
    # publish_dir/try_publish_dir (plan §10) is the directory-shaped publish
    # primitive -- exactly this module's payload shape (arrays.npz +
    # headers.json + removed_stars.json + _provenance.json under one
    # fingerprinted directory). try_publish_dir already never raises and
    # already implements the tmp-dir + os.replace + idempotent-concurrent-
    # publish + spool-append protocol, so when it's importable this module
    # delegates to it entirely instead of duplicating that logic.
    from syndiff_pipeline.common.provenance.publish import (  # type: ignore
        try_publish_dir as _prov_try_publish_dir,
    )

    _HAVE_PROVENANCE_PUBLISH = True
except Exception:  # pragma: no cover - package mid-authoring on this branch
    _HAVE_PROVENANCE_PUBLISH = False

try:
    from syndiff_pipeline.common.provenance.model import SkycellKey  # type: ignore

    _HAVE_PROVENANCE_MODEL = True
except Exception:  # pragma: no cover - package mid-authoring on this branch
    _HAVE_PROVENANCE_MODEL = False


def _canonical_fallback(obj: Any) -> bytes:
    """Local stand-in for ``common.provenance.fingerprint.canonical``.

    Mirrors the frozen spec (plan §9): sorted keys, floats rounded to 1e-9
    with -0.0 normalized, tuples -> lists, NaN/inf rejected. Not guaranteed
    byte-identical to the eventual real implementation (that's fine —
    content addressing means a definition change simply mints new
    fingerprints, per the "config change mid-campaign" invariant, §17).
    """

    def _clean(o: Any) -> Any:
        if isinstance(o, bool):
            return o
        if isinstance(o, (np.floating,)):
            return _clean(float(o))
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, float):
            if o != o or o in (float("inf"), float("-inf")):
                raise ValueError("NaN/inf are not allowed in canonical recipe params")
            o = round(o, 9)
            return 0.0 if o == 0.0 else o
        if isinstance(o, Mapping):
            return {str(k): _clean(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        return o

    cleaned = _clean(obj)
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _recipe_id_fallback(kind: str, params: Mapping, code_version: int) -> str:
    payload = {"kind": kind, "params": params, "code_version": code_version}
    return hashlib.sha256(_canonical_fallback(payload)).hexdigest()[:16]


def _fingerprint_fallback(
    kind: str, spatial_key: Mapping, recipe_id_value: str, input_fps: Iterable[str]
) -> str:
    payload = {
        "kind": kind,
        "spatial_key": spatial_key,
        "recipe_id": recipe_id_value,
        "input_fingerprints": sorted(input_fps),
    }
    return hashlib.sha256(_canonical_fallback(payload)).hexdigest()[:24]


def _compute_recipe_id(kind: str, params: Mapping, code_version: int) -> str:
    if _HAVE_PROVENANCE_FINGERPRINT:
        try:
            return _prov_recipe_id(kind, params, code_version)
        except Exception:
            logger.debug("provenance.recipe_id failed; using local fallback", exc_info=True)
    return _recipe_id_fallback(kind, params, code_version)


def _compute_fingerprint(
    kind: str, spatial_key: Mapping, recipe_id_value: str, input_fps: Iterable[str]
) -> str:
    input_fps = list(input_fps)
    if _HAVE_PROVENANCE_FINGERPRINT:
        try:
            return _prov_fingerprint(kind, spatial_key, recipe_id_value, input_fps)
        except Exception:
            logger.debug("provenance.fingerprint failed; using local fallback", exc_info=True)
    return _fingerprint_fallback(kind, spatial_key, recipe_id_value, input_fps)


def _spatial_key(projection: str, skycell: str) -> dict:
    if _HAVE_PROVENANCE_MODEL:
        try:
            return SkycellKey(str(projection), str(skycell)).to_dict()  # type: ignore[name-defined]
        except Exception:
            logger.debug("provenance.model.SkycellKey failed; using local dict", exc_info=True)
    return {"projection": str(projection), "skycell": str(skycell)}


def _spool_dir(data_root: str | Path) -> Path:
    # Mirrors scc_paths.provenance_spool_dir (already landed on the
    # concurrent branch as {data_root}/bookkeeping/spool) without importing
    # it, for the same reason the zarr root is constructed inline above.
    return Path(data_root).expanduser() / "bookkeeping" / "spool"


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


def _param(source: Any, name: str, default: Any) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def combined_recipe(resolved_or_params: Any = None, **overrides: Any) -> dict:
    """Build the ``combined_skycell`` recipe dict (plan §6 registry row).

    Accepts either a resolved-config-like object (attribute access), a plain
    mapping, or ``None`` (all defaults). Extracts:
      - saturation/star-removal params: ``enable_saturation_correction``,
        ``remove_saturated_stars``, ``bright_star_mag_threshold``
      - band-combine constants: ``band_weights``, ``apply_flux_conv``
      - ``gaia_version``

    Keyword ``overrides`` win over both the source and the defaults.
    """
    recipe = {
        "enable_saturation_correction": bool(
            _param(resolved_or_params, "enable_saturation_correction", False)
        ),
        "remove_saturated_stars": bool(_param(resolved_or_params, "remove_saturated_stars", True)),
        "bright_star_mag_threshold": float(
            _param(resolved_or_params, "bright_star_mag_threshold", 13.0)
        ),
        "band_weights": dict(_param(resolved_or_params, "band_weights", DEFAULT_BAND_WEIGHTS)),
        "apply_flux_conv": bool(_param(resolved_or_params, "apply_flux_conv", True)),
        "gaia_version": _param(resolved_or_params, "gaia_version", None),
    }
    recipe.update(overrides)
    return recipe


def combined_recipe_id(recipe: Mapping, code_version: int = COMBINED_RECIPE_SCHEMA_VERSION) -> str:
    return _compute_recipe_id(KIND, recipe, code_version)


def combined_fingerprint(
    projection: str,
    skycell: str,
    recipe_id_value: str,
    input_fingerprints: Iterable[str] = (),
) -> str:
    spatial_key = {"projection": str(projection), "skycell": str(skycell)}
    return _compute_fingerprint(KIND, spatial_key, recipe_id_value, input_fingerprints)


# ---------------------------------------------------------------------------
# Store layout
# ---------------------------------------------------------------------------


def _ps1_combined_zarr_root(data_root: str | Path) -> Path:
    return Path(data_root).expanduser() / PS1_SKYCELLS_ZARR_DIRNAME / COMBINED_ZARR_BASENAME


def combined_cell_dir(data_root: str | Path, projection: str, skycell: str, fp: str) -> Path:
    """``{data_root}/ps1_skycells_zarr/ps1_combined.zarr/{projection}/{skycell}/{fp}/``."""
    return _ps1_combined_zarr_root(data_root) / str(projection) / str(skycell) / str(fp)


def _payload_complete(cell_dir: Path) -> bool:
    if not cell_dir.is_dir():
        return False
    return all((cell_dir / name).is_file() for name in _REQUIRED_MEMBERS)


# ---------------------------------------------------------------------------
# Publish / load
# ---------------------------------------------------------------------------


def _local_publish_dir_fallback(
    dest_root: Path,
    fp: str,
    spatial_key: Mapping,
    rid: str,
    code_version: int,
    input_fps: list,
    recipe: Mapping,
    producer: str,
    write_payload,
) -> Path:
    """Hand-rolled directory publish, used only when
    ``common.provenance.publish`` isn't importable (package mid-authoring).

    Same protocol as the real ``publish_dir``: write under
    ``_tmp_{fp}_{pid}``, then a single atomic ``os.replace`` onto the
    fingerprinted key. Deliberately does **not** catch/clean up on failure
    here (the caller's outer ``try/except`` does that at the
    ``publish_combined_cell`` level) -- an exception raised by
    ``write_payload`` leaves the ``_tmp_*`` directory behind as an orphan,
    matching the plan §17 failure-matrix invariant ("worker crash mid-write
    -> only ``_tmp_*`` orphan; no sidecar; index unaffected").
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    final_dir = dest_root / fp
    tmp_dir = dest_root / f"_tmp_{fp}_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    write_payload(tmp_dir)

    sidecar = {
        "fingerprint": fp,
        "kind": KIND,
        "spatial_key": dict(spatial_key),
        "recipe_id": rid,
        "recipe": dict(recipe),
        "code_version": code_version,
        "input_fingerprints": input_fps,
        "producer": producer,
        "created_at": time.time(),
    }
    with open(tmp_dir / _PROVENANCE_SIDECAR_FILENAME, "w") as fh:
        json.dump(sidecar, fh, default=_json_default)

    os.replace(tmp_dir, final_dir)
    return final_dir


def publish_combined_cell(
    data_root: str | Path,
    projection: str,
    skycell: str,
    *,
    combined_image: np.ndarray,
    combined_mask: np.ndarray,
    headers_data: Mapping[str, str],
    removed_stars: list,
    recipe: Mapping,
    input_fingerprints: Iterable[str] = (),
    code_version: int = COMBINED_RECIPE_SCHEMA_VERSION,
    producer: str = "template_creation.processing.combined_store.publish_combined_cell",
) -> dict | None:
    """Atomically publish a combined-skycell cell. Never raises (best-effort).

    Payload: ``{arrays.npz [combined_image, combined_mask], headers.json,
    removed_stars.json}`` plus a self-describing ``_provenance.json``
    sidecar (plan §10). Delegates to
    ``common.provenance.publish.try_publish_dir`` (tmp-dir + ``os.replace``
    + idempotent-concurrent-publish + spool-append, plan §10) when that
    package is importable; falls back to a local equivalent otherwise so
    this module keeps working standalone while ``common/provenance/`` is
    authored concurrently on this branch.

    Returns a small info dict on success (or when an identical fingerprint
    is already published), or ``None`` on any failure.
    """
    try:
        input_fps = sorted(set(input_fingerprints))
        rid = combined_recipe_id(recipe, code_version)
        fp = combined_fingerprint(projection, skycell, rid, input_fps)
        spatial_key = _spatial_key(projection, skycell)

        final_dir = combined_cell_dir(data_root, projection, skycell, fp)
        dest_root = final_dir.parent

        if _payload_complete(final_dir):
            # Idempotent: identical content already published under this fp
            # (plan §17: "two workers build same fp -> idempotent").
            return {"fingerprint": fp, "recipe_id": rid, "dir": final_dir, "already_published": True}

        def _write_payload(tmp_dir: Path) -> None:
            np.savez_compressed(
                tmp_dir / _ARRAYS_FILENAME,
                combined_image=np.asarray(combined_image, dtype=np.float32),
                combined_mask=np.asarray(combined_mask, dtype=np.uint16),
            )
            with open(tmp_dir / _HEADERS_FILENAME, "w") as fh:
                json.dump(dict(headers_data), fh, default=_json_default)
            with open(tmp_dir / _REMOVED_STARS_FILENAME, "w") as fh:
                json.dump(list(removed_stars), fh, default=_json_default)

        if _HAVE_PROVENANCE_PUBLISH:
            published_dir = _prov_try_publish_dir(  # type: ignore[name-defined]
                dest_root,
                fp,
                KIND,
                spatial_key,
                rid,
                code_version,
                input_fps,
                _write_payload,
                recipe_params=dict(recipe),
                spool_dir=_spool_dir(data_root),
                produced_by=producer,
            )
            if published_dir is None:
                return None
        else:
            published_dir = _local_publish_dir_fallback(
                dest_root, fp, spatial_key, rid, code_version, input_fps, recipe, producer, _write_payload
            )

        return {"fingerprint": fp, "recipe_id": rid, "dir": published_dir, "already_published": False}
    except Exception:
        logger.warning(
            "publish_combined_cell failed for projection=%s skycell=%s (best-effort, no raise)",
            projection,
            skycell,
            exc_info=True,
        )
        return None


def try_load_combined_cell(
    data_root: str | Path, projection: str, skycell: str, fp: str
) -> dict | None:
    """Load a published combined cell, or ``None`` if absent/incomplete/corrupt.

    Returns exactly the ``band_cache`` shape:
    ``{combined_image, combined_mask, headers_data, removed_stars}``.
    """
    cell_dir = combined_cell_dir(data_root, projection, skycell, fp)
    try:
        if not _payload_complete(cell_dir):
            return None
        with np.load(cell_dir / _ARRAYS_FILENAME, allow_pickle=False) as z:
            combined_image = np.array(z["combined_image"])
            combined_mask = np.array(z["combined_mask"])
        with open(cell_dir / _HEADERS_FILENAME) as fh:
            headers_data = json.load(fh)
        with open(cell_dir / _REMOVED_STARS_FILENAME) as fh:
            removed_stars = json.load(fh)
        return {
            "combined_image": combined_image,
            "combined_mask": combined_mask,
            "headers_data": headers_data,
            "removed_stars": removed_stars,
        }
    except Exception:
        logger.warning("try_load_combined_cell failed for %s (best-effort)", cell_dir, exc_info=True)
        return None


def resolve_current_combined_ref(
    data_root: str | Path, projection: str, skycell: str
):
    """Return the validated selected immutable combined artifact, if any."""
    from syndiff_pipeline.template_creation.processing.artifact_pointer import read_current_pointer

    return read_current_pointer(
        _ps1_combined_zarr_root(data_root) / str(projection) / str(skycell),
        expected_kind=KIND,
        projection=str(projection),
        skycell=str(skycell),
        required_members=(*_REQUIRED_MEMBERS, _PROVENANCE_SIDECAR_FILENAME),
    )


def update_current_pointer(
    data_root: str | Path, projection: str, skycell: str, fp: str
) -> bool:
    """Select an already-published combined artifact. Never selects by mtime."""
    from syndiff_pipeline.template_creation.processing.artifact_pointer import (
        validate_artifact,
        write_current_pointer,
    )

    root = _ps1_combined_zarr_root(data_root) / str(projection) / str(skycell)
    target = root / str(fp)
    try:
        # Validate the requested target using the same strict contract as a
        # reader, without first changing the current pointer.
        ref = validate_artifact(
            root, kind=KIND, projection=str(projection), skycell=str(skycell),
            fingerprint=str(fp), recipe_id=None,
            required_members=(*_REQUIRED_MEMBERS, _PROVENANCE_SIDECAR_FILENAME),
        )
        write_current_pointer(root, ref)
        return True
    except Exception:
        logger.warning("failed to update combined current pointer for %s/%s", projection, skycell, exc_info=True)
        return False


def gaia_version_stamp(catalog_path: str | None) -> str:
    """
    Gaia catalog identity for the shared-store recipe fingerprint.

    Any successfully-resolved catalog file stamps as one canonical
    ``"loaded"`` value -- not embedded path/size/mtime -- so cells built
    from different SCCs' own per-sector catalog files (e.g. adjacent CVZ
    sectors imaging much of the same sky, each with its own
    ``gaia_catalog_s{sector}_{camera}_{ccd}.csv``) can still share the
    shared PS1 store: Gaia is a static astrometric catalog, so its content
    does not meaningfully differ sector-to-sector for the same sky
    position, and per-file identity was defeating cross-sector cache
    sharing for no real correctness benefit. Only "no usable catalog"
    (unset or unreadable path) gets a different stamp -- paired with
    ps1_process's fail-loud catalog-load-failure path, an SCC only ever
    reaches this function with `catalog_path` pointing at a real, already
    successfully-loaded file, never a silently-failed one.
    """
    if not catalog_path:
        return "none"
    try:
        os.stat(catalog_path)
        return "loaded"
    except OSError:
        return "none"


def production_combined_recipe(
    ps1_process_config: Any,
    *,
    data_root: str | Path | None = None,
    sector: int | None = None,
    camera: int | None = None,
    ccd: int | None = None,
) -> dict:
    """Build the ``combined_skycell`` recipe from a resolved
    ``stages.ps1_process`` config (attribute-style object, mapping, or
    ``None``), exactly the way ``ps1_process.run_modern_sliding_window_pipeline``
    builds it for ``seed_band_cache_from_combined_store`` /
    ``publish_combined_cell``.

    This is deliberately the single, shared place that maps a ps1_process
    config to a recipe. Before this helper existed, the producer
    (``ps1_process.py``) and any downstream reader that needed to recompute
    "what recipe would this config produce" (e.g. ``downsample`` resolving
    its own cross-projection padding correction) each built the recipe
    dict/``gaia_version`` stamp independently -- exactly the kind of drift
    that let a reader silently pick a wrong-recipe cell (see
    ``field_downsample``/``padding_correction`` fingerprint discovery: they
    used to fall back to "newest mtime" with no recipe check at all).
    Producer and reader must always call this same function so they can
    never disagree.

    ``gaia_version`` is only stamped (rather than fixed at ``"none"``) when
    saturation handling is actually enabled, mirroring ps1_process's own
    logic: a catalog-independent config (both flags off) should not mint a
    new fingerprint just because some unrelated catalog file's mtime moved.

    ``catalog_path`` is almost never set explicitly in production configs --
    every real caller relies on ``ps1_process.load_gaia_catalog``'s implicit
    per-SCC default path. Stamping ``gaia_version="none"`` whenever
    ``catalog_path`` is merely unset (the historical behavior) made every
    such build indistinguishable from a genuine catalog-load failure, so a
    single run whose catalog load silently failed could permanently poison
    the shared, cross-sector/cross-run store for every other SCC requesting
    the same nominal recipe. When ``data_root``/``sector``/``camera``/``ccd``
    are supplied (every current caller has them available), an unset
    ``catalog_path`` resolves to that same implicit default via
    ``scc_paths.default_gaia_catalog_path`` before stamping, so the
    fingerprint reflects the catalog that will actually be loaded.
    """
    enable_saturation_correction = bool(
        _param(ps1_process_config, "enable_saturation_correction", False)
    )
    remove_saturated_stars = bool(_param(ps1_process_config, "remove_saturated_stars", True))
    bright_star_mag_threshold = float(_param(ps1_process_config, "bright_star_mag_threshold", 13.0))
    catalog_path = _param(ps1_process_config, "catalog_path", None)
    if catalog_path is None and None not in (data_root, sector, camera, ccd):
        from syndiff_pipeline.common.scc_paths import default_gaia_catalog_path

        catalog_path = str(default_gaia_catalog_path(data_root, sector, camera, ccd))
    gaia_version = (
        gaia_version_stamp(catalog_path)
        if (enable_saturation_correction or remove_saturated_stars)
        else "none"
    )
    return combined_recipe(
        enable_saturation_correction=enable_saturation_correction,
        remove_saturated_stars=remove_saturated_stars,
        bright_star_mag_threshold=bright_star_mag_threshold,
        gaia_version=gaia_version,
    )


# ---------------------------------------------------------------------------
# raw_skycell input fingerprint (bug fix: combined_skycell's Merkle
# fingerprint must incorporate its upstream raw_skycell identity -- plan §6
# registry row "combined_skycell inputs: raw_skycell, source_catalog";
# decision #6 "raw skycells version on (size, mtime, download_batch_id)").
#
# gaia_version (-> source_catalog) is deliberately NOT re-folded in here: it
# is already threaded into ``combined_recipe``/``combined_recipe_id`` (see
# ``gaia_version_stamp`` above and its call site in
# ``run_modern_sliding_window_pipeline``), so it already changes
# ``combined_fingerprint`` via ``recipe_id_value`` -- adding it again as an
# input_fingerprint would be redundant, not a second signal.
# ---------------------------------------------------------------------------


def _ps1_raw_zarr_root(data_root: str | Path) -> Path:
    return Path(data_root).expanduser() / PS1_SKYCELLS_ZARR_DIRNAME / RAW_ZARR_BASENAME


def _raw_skycell_group_dir(data_root: str | Path, projection: str, skycell: str) -> Path:
    """Directory of one raw skycell's zarr group.

    Mirrors ``zarr_utils.load_skycell_bands_masks_and_headers``'s key
    resolution: ``skycell`` is already the full ``skycell.PROJ.CELL`` id
    (starts with ``"skycell."``) when passed straight from a caller that
    hasn't split it, or -- as this module normally has it, via
    ``_projection_and_cell`` -- just the trailing cell number, in which case
    it's rebuilt as ``{projection}.{cell}``.
    """
    skycell_str = str(skycell)
    skycell_key = skycell_str if skycell_str.startswith("skycell.") else f"{projection}.{skycell_str}"
    return _ps1_raw_zarr_root(data_root) / str(projection) / skycell_key


def _raw_skycell_version_token(data_root: str | Path, projection: str, skycell: str) -> dict:
    """Cheap, non-recursive on-disk version token for one raw PS1 skycell.

    Decision #5 ("per-cell checksums are too costly on the hot path") rules
    out opening/checksumming the raw FITS pixel data here. Decision #6's
    ideal identity is ``(size, mtime, download_batch_id)``, but no
    ``download_batch_id`` is stamped anywhere in this codebase yet (grepped
    ``ps1_download.py``: nothing records one). So this uses the cheapest
    available proxy instead: a single ``os.scandir`` of the raw zarr store's
    per-skycell group directory (``ps1_skycells_zarr/ps1_skycells.zarr/
    {projection}/{skycell_key}/``) -- one directory listing plus one
    ``stat()`` per immediate entry (band arrays/subgroups + metadata files),
    never walking into chunk data. ``ps1_download.store_skycell_batch``
    replaces this skycell's array subdirectories wholesale on a re-download,
    so at least one immediate entry's size/mtime changes even though no
    chunk is ever opened here -- the same "stat, don't read" shape as
    ``gaia_version_stamp`` above.

    Returns a JSON-safe dict (fed through ``recipe_id``/``fingerprint``,
    never persisted directly). ``{"status": "missing"}`` when the raw store
    or this skycell's group isn't present on disk -- deliberately a stable,
    distinct value (not silently omitted) so a skycell that doesn't exist
    yet and one that later appears always mint different fingerprints.
    """
    group_dir = _raw_skycell_group_dir(data_root, projection, skycell)
    try:
        entries = list(os.scandir(group_dir))
    except OSError:
        return {"status": "missing"}

    total_size = 0
    newest_mtime_ns = 0
    names: list[str] = []
    for entry in entries:
        try:
            st = entry.stat()
        except OSError:
            continue
        total_size += st.st_size
        newest_mtime_ns = max(newest_mtime_ns, st.st_mtime_ns)
        names.append(entry.name)

    return {
        "status": "present",
        "n_entries": len(names),
        "total_size": total_size,
        "newest_mtime_ns": newest_mtime_ns,
    }


def resolve_combined_fingerprint_for_recipe(
    data_root: str | Path,
    projection: str,
    skycell: str,
    recipe: Mapping,
    *,
    raw_fp: str | None = None,
) -> str | None:
    """Deterministically resolve the ``combined_skycell`` fingerprint that
    matches ``recipe`` for ``(projection, skycell)``, or ``None`` if a
    payload for that exact recipe has not been published.

    This recomputes the fingerprint exactly the way ``publish_combined_cell``
    derives it for a given recipe -- it never inspects directory mtimes or
    "whichever fingerprint dir happens to exist". This matters because the
    store is shared cross-sector/cross-run: another sector or an unrelated
    experiment may have published a *different* recipe (e.g.
    ``remove_saturated_stars=False``) for the exact same sky cell, and that
    publish's mtime says nothing about whether it matches the caller's own
    config. Callers that know their own recipe (i.e. every production
    caller) should always prefer this over mtime-based discovery; mtime
    fallback should only ever be used, with a loud warning, when no recipe
    context is available at all (see ``padding_correction`` /
    ``field_downsample`` discovery helpers).

    ``raw_fp`` lets a caller that already computed
    ``raw_skycell_input_fingerprint`` reuse it (e.g. a hot loop over many
    recipes for the same skycell); otherwise it is recomputed here.
    """
    try:
        rid = combined_recipe_id(recipe)
        resolved_raw_fp = (
            raw_fp if raw_fp is not None else raw_skycell_input_fingerprint(data_root, projection, skycell)
        )
        fp = combined_fingerprint(projection, skycell, rid, [resolved_raw_fp])
    except Exception:
        logger.warning(
            "resolve_combined_fingerprint_for_recipe failed for %s/%s (best-effort)",
            projection,
            skycell,
            exc_info=True,
        )
        return None
    cell_dir = combined_cell_dir(data_root, projection, skycell, fp)
    if not _payload_complete(cell_dir):
        return None
    return fp


def raw_skycell_input_fingerprint(data_root: str | Path, projection: str, skycell: str) -> str:
    """``raw_skycell``-kind Merkle fingerprint for one PS1 skycell, suitable
    as a ``combined_fingerprint(..., input_fingerprints=[...])`` entry.

    Computed the same way for every caller (seed lookup and publish both
    call this function with the same ``(data_root, projection, skycell)``),
    which is what keeps lookup and publish symmetric: a cache built under
    fingerprint A is only ever matched by a lookup that also derives
    fingerprint A. See ``_raw_skycell_version_token`` for what the token
    covers and why.
    """
    token = _raw_skycell_version_token(data_root, projection, skycell)
    rid = _compute_recipe_id("raw_skycell", token, RAW_SKYCELL_SCHEMA_VERSION)
    spatial_key = {"projection": str(projection), "skycell": str(skycell)}
    return _compute_fingerprint("raw_skycell", spatial_key, rid, ())


def _projection_and_cell(skycell_id: str) -> tuple[str, str] | None:
    """Split ``skycell.PROJ.CELL`` into (``skycell.PROJ``, ``CELL``)."""
    parts = str(skycell_id).split(".")
    if len(parts) < 3:
        return None
    return ".".join(parts[:2]), parts[2]


def seed_band_cache_from_combined_store(
    data_root: str | Path,
    skycell_names: Iterable[str],
    recipe: Mapping,
) -> dict[str, dict]:
    """Load combined-store hits into a ``band_cache``-shaped dict. Never raises.

    Uses ``resolve_combined_fingerprint_for_recipe`` (recipe-deterministic,
    never mtime-based) so a seed hit is guaranteed to match ``recipe`` even
    when other recipes have also been published for the same skycell.
    """
    hits: dict[str, dict] = {}
    for name in skycell_names:
        parsed = _projection_and_cell(name)
        if parsed is None:
            continue
        projection, cell = parsed
        try:
            fp = resolve_combined_fingerprint_for_recipe(data_root, projection, cell, recipe)
            loaded = try_load_combined_cell(data_root, projection, cell, fp) if fp is not None else None
        except Exception:
            logger.warning(
                "Combined-store seed lookup failed for %s (skipping)", name, exc_info=True
            )
            continue
        if loaded is not None:
            hits[name] = loaded
    return hits
