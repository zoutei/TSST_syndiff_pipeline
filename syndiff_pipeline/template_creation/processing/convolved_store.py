"""Shared, sky-keyed store for canonical (same-projection-only) convolved
PS1 skycells.

Pure data-layer groundwork for **PR5 — shared convolved store**
(``doc/template_bookkeeping_plan.md`` §13, decision #14/#15, artifact-kind
registry §6 ``convolved_skycell``). Structural mirror of
``combined_store.py``: same tmp-dir + ``os.replace`` atomic publish, same
payload shape (arrays.npz + headers.json + removed_stars.json +
``_provenance.json`` sidecar), same lazy/best-effort provenance import.

**Canonical cell definition (plan §13):** convolve on a master array padded
by **same-projection** neighbors only (sector-independent). Cross-projection
seam correction is applied later at ``scc_assembly`` — *not* here — via the
validated linearity finding: ``convolve(canonical cell with gap zeroed) +
convolve(reprojected neighbor patch alone at its true position)``, added.
See ``tests/test_seam_correction_linearity.py`` for the re-landed
linearity/bias-guard tests that document why that correction is mandatory.

**Sharing key:** the canonical convolved cell carries no sector/camera/ccd
(same as its ``combined_skycell`` input), so it's cross-sector shareable.
Its fingerprint Merkle-inputs the upstream ``combined_skycell`` fingerprint,
so any change to star-removal/band-combine params (which re-fingerprints
``combined_skycell``) automatically re-fingerprints every downstream
``convolved_skycell`` that consumed it.

**Hard cut on writes (decision #15):** once PR5 actually lands, per-SCC
``convolved.zarr`` is never written again — that is a live-wiring change to
``ps1_process.py`` and is explicitly out of scope here.

Scope note: this module does **not** import ``ps1_process`` or
``cross_projection_padding.py`` and does **not** wire into either. Live
wiring is a later, separately-gated step per the plan.
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
# Artifact-kind identity (plan §6 registry row: convolved_skycell)
# ---------------------------------------------------------------------------

KIND = "convolved_skycell"

# Hand-bumped per decision #5 ("code_version"): bump whenever this producer's
# algorithm (payload shape, recipe params) changes in a way that should mint
# new fingerprints for otherwise-identical inputs.
CONVOLVED_RECIPE_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Store location (decision #14): all three PS1 stores live under the
# existing ps1_skycells_zarr/ folder. scc_paths.py is owned by a concurrent
# workstream on this branch and is being actively edited there, so the root
# is constructed inline here rather than imported (avoids a load-time
# dependency on a file this module doesn't own). As of this writing that
# workstream has already landed ``scc_paths.ps1_convolved_zarr_path`` with
# exactly this basename (``ps1_skycells_zarr/ps1_convolved.zarr``) --
# TODO(PR5 integration): switch ``_ps1_convolved_zarr_root`` below to call
# that helper directly once this module is wired into the concurrent
# branch's final state (plan §7, §20).
# ---------------------------------------------------------------------------

PS1_SKYCELLS_ZARR_DIRNAME = "ps1_skycells_zarr"
CONVOLVED_ZARR_BASENAME = "ps1_convolved.zarr"

_ARRAYS_FILENAME = "arrays.npz"
_HEADERS_FILENAME = "headers.json"
_REMOVED_STARS_FILENAME = "removed_stars.json"
_PROVENANCE_SIDECAR_FILENAME = "_provenance.json"
_REQUIRED_MEMBERS = (_ARRAYS_FILENAME, _HEADERS_FILENAME, _REMOVED_STARS_FILENAME)

# convolution_utils.apply_gaussian_convolution production defaults.
DEFAULT_PSF_SIGMA = 40.0
DEFAULT_RADIUS = 470
DEFAULT_MODE = "constant"

# Not a tunable — part of the recipe identity (plan §13 decision #4): the
# canonical cell is padded by same-projection neighbors only. Recorded in
# the recipe so a future padding-strategy change re-fingerprints downstream.
PADDING_MODE = "same_projection_only"


# ---------------------------------------------------------------------------
# Provenance contract (plan §9-10) — lazy, best-effort import. Mirrors
# combined_store.py exactly; kept duplicated (not shared) so each store
# module stays independently importable regardless of the other's state.
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
    # primitive -- exactly this module's payload shape. See
    # combined_store.py's identical import for the full rationale.
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

    See ``combined_store._canonical_fallback`` for the same rationale: this
    mirrors the frozen spec's shape (sorted keys, rounded floats, no -0.0,
    tuples->lists, NaN/inf rejected) without depending on the concurrently
    authored package.
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


def convolved_recipe(resolved_or_params: Any = None, **overrides: Any) -> dict:
    """Build the ``convolved_skycell`` recipe dict (plan §6 registry row).

    Extracts ``psf_sigma``, ``radius``, ``mode`` (the convolution boundary
    mode, see ``convolution_utils.apply_gaussian_convolution``); ``padding``
    is always ``"same_projection_only"`` for the canonical cell (not
    overridable via ``resolved_or_params`` — cross-projection padding is a
    ``scc_assembly``-time correction, not part of this artifact's identity
    knobs — but callers may still override it explicitly via ``**overrides``
    for testing param-sensitivity).
    """
    recipe = {
        "psf_sigma": float(_param(resolved_or_params, "psf_sigma", DEFAULT_PSF_SIGMA)),
        "radius": int(_param(resolved_or_params, "radius", DEFAULT_RADIUS)),
        "mode": str(_param(resolved_or_params, "mode", DEFAULT_MODE)),
        "padding": PADDING_MODE,
    }
    recipe.update(overrides)
    return recipe


def convolved_recipe_id(recipe: Mapping, code_version: int = CONVOLVED_RECIPE_SCHEMA_VERSION) -> str:
    return _compute_recipe_id(KIND, recipe, code_version)


def convolved_fingerprint(
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


def _ps1_convolved_zarr_root(data_root: str | Path) -> Path:
    return Path(data_root).expanduser() / PS1_SKYCELLS_ZARR_DIRNAME / CONVOLVED_ZARR_BASENAME


def convolved_cell_dir(data_root: str | Path, projection: str, skycell: str, fp: str) -> Path:
    """``{data_root}/ps1_skycells_zarr/ps1_convolved.zarr/{projection}/{skycell}/{fp}/``."""
    return _ps1_convolved_zarr_root(data_root) / str(projection) / str(skycell) / str(fp)


def _projection_and_cell(skycell_id: str) -> tuple[str, str] | None:
    """Split a canonical ``skycell.<projection>.<cell>`` identity."""
    parts = str(skycell_id).split(".")
    if len(parts) < 3 or parts[0] != "skycell":
        return None
    return ".".join(parts[:2]), parts[2]


def _payload_complete(cell_dir: Path) -> bool:
    if not cell_dir.is_dir():
        return False
    return all((cell_dir / name).is_file() for name in _REQUIRED_MEMBERS)


def resolve_convolved_fingerprint_for_recipe(
    data_root: str | Path,
    projection: str,
    skycell: str,
    recipe: Mapping,
    combined_fp: str,
    *,
    extra_input_fingerprints: Iterable[str] = (),
    code_version: int = CONVOLVED_RECIPE_SCHEMA_VERSION,
) -> str | None:
    """Deterministically resolve the ``convolved_skycell`` fingerprint that
    matches ``recipe`` (built on top of this exact upstream ``combined_fp``)
    for ``(projection, skycell)``, or ``None`` if that exact recipe has not
    been published.

    Mirrors exactly how ``publish_convolved_cell`` derives its fingerprint.
    Never inspects directory mtimes or "whichever fingerprint dir happens to
    exist" -- the shared store is cross-sector/cross-run, so a different
    recipe published later for the same cell (e.g. a different ``psf_sigma``
    or an upstream ``combined_skycell`` built with a different star-removal
    config) must never be picked over the one matching the caller's own
    config. Mtime-based fallback should only ever be used, with a loud
    warning, by callers that genuinely have no recipe context (see
    ``field_downsample`` / ``padding_correction`` discovery helpers).
    """
    try:
        input_fps = sorted({str(combined_fp), *extra_input_fingerprints})
        rid = convolved_recipe_id(recipe, code_version)
        fp = convolved_fingerprint(projection, skycell, rid, input_fps)
    except Exception:
        logger.warning(
            "resolve_convolved_fingerprint_for_recipe failed for %s/%s (best-effort)",
            projection,
            skycell,
            exc_info=True,
        )
        return None
    cell_dir = convolved_cell_dir(data_root, projection, skycell, fp)
    if not _payload_complete(cell_dir):
        return None
    return fp


def republish_convolved_cell_for_recipe(
    data_root: str | Path,
    projection: str,
    skycell: str,
    target_recipe: Mapping,
    target_combined_fp: str,
    *,
    code_version: int = CONVOLVED_RECIPE_SCHEMA_VERSION,
) -> dict | None:
    """Republish an already-published convolved cell under a new
    ``combined_fingerprint`` chain, when its upstream content is provably
    identical -- not a reprocess.

    A ``convolved_skycell`` fingerprint is chained on top of its upstream
    ``combined_skycell`` fingerprint (see :func:`resolve_convolved_fingerprint_for_recipe`).
    Any event that re-mints the *combined* fingerprint for unchanged
    content -- e.g. the ``gaia_version_stamp`` scheme migration handled by
    :func:`combined_store.republish_combined_cell_for_recipe` -- therefore
    also orphans every convolved cell chained on the old combined
    fingerprint, even though the convolution recipe (``psf_sigma``,
    ``radius``, ``mode``) never changed and the pixels are identical.

    Unlike the combined-store rekey, this does not need to reason about
    individual recipe fields: it only requires an exact
    ``convolved_recipe_id`` match (the convolution recipe has no
    ``gaia_version``-like field of its own), plus a **direct pixel
    equality check** between the old cell's recorded upstream combined
    image and ``target_combined_fp``'s combined image -- stronger than
    trusting fingerprint equality alone, and cheap since combined images
    are only a few MB. Only single-input cells are handled (production
    ``publish_convolved_cell`` calls never pass
    ``extra_input_fingerprints`` today); a multi-input cell is skipped,
    not guessed at.

    Returns ``None`` if no eligible candidate exists or republish fails;
    otherwise the same info dict :func:`publish_convolved_cell` returns.
    """
    cell_root = convolved_cell_dir(data_root, projection, skycell, "x").parent
    if not cell_root.is_dir():
        return None
    target_rid = convolved_recipe_id(target_recipe, code_version)
    try:
        candidates = sorted(p for p in cell_root.iterdir() if p.is_dir())
    except OSError:
        return None
    new_combined_payload = None
    for fp_dir in candidates:
        prov_path = fp_dir / _PROVENANCE_SIDECAR_FILENAME
        if not prov_path.is_file():
            continue
        try:
            prov = json.loads(prov_path.read_text())
        except Exception:
            continue
        rp = dict(prov.get("recipe_params", {}))
        try:
            if convolved_recipe_id(rp, code_version) != target_rid:
                continue
        except Exception:
            continue
        inputs = prov.get("inputs", [])
        if len(inputs) != 1:
            continue
        old_combined_fp = str(inputs[0])
        if old_combined_fp == str(target_combined_fp):
            continue
        if not _payload_complete(fp_dir):
            continue
        from syndiff_pipeline.template_creation.processing.combined_store import (
            try_load_combined_cell,
        )

        old_combined = try_load_combined_cell(data_root, projection, skycell, old_combined_fp)
        if old_combined is None:
            continue
        if new_combined_payload is None:
            new_combined_payload = try_load_combined_cell(data_root, projection, skycell, target_combined_fp)
            if new_combined_payload is None:
                return None
        if not np.array_equal(old_combined["combined_image"], new_combined_payload["combined_image"]):
            continue
        payload = try_load_convolved_cell(data_root, projection, skycell, fp_dir.name)
        if payload is None:
            continue
        return publish_convolved_cell(
            data_root,
            projection,
            skycell,
            convolved_image=payload["convolved_image"],
            convolved_mask=payload["convolved_mask"],
            headers_data=payload["headers_data"],
            removed_stars=payload["removed_stars"],
            recipe=target_recipe,
            combined_fingerprint=target_combined_fp,
            code_version=code_version,
            producer="template_creation.processing.convolved_store.republish_convolved_cell_for_recipe",
        )
    return None


def skycell_already_canonical(
    data_root: str | Path,
    projection: str,
    skycell: str,
    combined_recipe: Mapping,
    convolved_recipe: Mapping,
    *,
    raw_fp: str | None = None,
) -> bool:
    """``True`` iff a *complete* ``convolved_skycell`` payload for
    ``(projection, skycell)`` already exists under the exact recipe chain
    ``combined_recipe -> convolved_recipe`` that a caller's own config would
    produce -- i.e. this cell can be skipped entirely (no re-fetch, no
    re-combine, no re-convolve) rather than reprocessed.

    This is a per-cell, read-only, best-effort check (never raises; a
    resolution failure conservatively returns ``False``, forcing normal
    reprocessing) intended to run *before* any expensive PS1 fetch/combine/
    convolution work is dispatched for a cell, not just before the store
    write.

    Correctness is anchored entirely in the fingerprint chain, not in any
    parameter this function inspects directly: ``combined_recipe`` must be
    built the same way ``combined_store.production_combined_recipe`` builds
    it (baking in ``band_weights``, ``apply_flux_conv``,
    ``remove_saturated_stars``, ``enable_saturation_correction``,
    ``bright_star_mag_threshold``, ``gaia_version``), and
    ``convolved_recipe`` the same way ``convolved_store.convolved_recipe``
    builds it (``psf_sigma``, ``radius``, ``mode``). Two runs can only
    resolve to the same convolved fingerprint here if *every one* of those
    upstream combined-recipe parameters matches in addition to the
    convolution parameters -- there is no way to get a false "already
    canonical" positive on convolution params alone, because
    ``convolved_fingerprint`` always Merkles in the resolved
    ``combined_fingerprint`` as one of its inputs (mirrors exactly what
    ``ps1_process._publish_canonical_convolved_snapshot`` already does per
    cell, just checked *before* convolving instead of after).

    ``raw_fp`` lets a caller that already computed
    ``raw_skycell_input_fingerprint`` for this cell reuse it (e.g. a hot
    loop resolving many cells per row); otherwise it is recomputed here.
    """
    from syndiff_pipeline.template_creation.processing.combined_store import (
        raw_skycell_input_fingerprint,
        resolve_combined_fingerprint_for_recipe,
    )

    try:
        combined_fp = resolve_combined_fingerprint_for_recipe(
            data_root, projection, skycell, combined_recipe, raw_fp=raw_fp,
        )
        if combined_fp is None:
            # No published combined_skycell record under this exact recipe
            # yet -- the convolved cell (keyed on this combined_fp) cannot
            # exist either. Never fabricate an edge to a missing upstream.
            return False
        convolved_fp = resolve_convolved_fingerprint_for_recipe(
            data_root, projection, skycell, convolved_recipe, combined_fp,
        )
    except Exception:
        logger.warning(
            "skycell_already_canonical failed for %s/%s (best-effort, treating "
            "as not-canonical)",
            projection,
            skycell,
            exc_info=True,
        )
        return False
    return convolved_fp is not None


def classify_projection_missing_cells(
    data_root: str | Path,
    projection: str,
    cell_names: Iterable[str],
    combined_recipe: Mapping,
    convolved_recipe: Mapping,
) -> set[str]:
    """Cheap, upfront, per-projection classification for the tiered ingest
    architecture (see ``doc/ps1_process_tiered_ingest_architecture_plan.md``,
    Part B, step 1).

    Returns the subset of ``cell_names`` (each a full ``skycell.<projection>.<cell>``
    identity) that is NOT already canonical under the exact recipe chain
    ``combined_recipe -> convolved_recipe`` -- i.e. the cells that actually
    need (re)convolving. Built entirely from ``skycell_already_canonical``
    (fingerprint/path checks only, no pixel array loads), so this is cheap
    enough to run for every cell in a projection before any task dispatch.
    """
    missing: set[str] = set()
    for cell_name in cell_names:
        parsed = _projection_and_cell(cell_name)
        if parsed is None:
            missing.add(cell_name)
            continue
        cell_projection, cell = parsed
        try:
            already_canonical = skycell_already_canonical(
                data_root, cell_projection, cell, combined_recipe, convolved_recipe,
            )
        except Exception:
            logger.warning(
                "classify_projection_missing_cells: canonical check failed for "
                "%s (treating as missing/needs-conv)",
                cell_name,
                exc_info=True,
            )
            already_canonical = False
        if not already_canonical:
            missing.add(cell_name)
    return missing


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
    See ``combined_store._local_publish_dir_fallback`` for the identical
    rationale (deliberately no cleanup on failure -- that's what leaves the
    ``_tmp_*`` orphan the plan's failure matrix documents).
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


def publish_convolved_cell(
    data_root: str | Path,
    projection: str,
    skycell: str,
    *,
    convolved_image: np.ndarray,
    convolved_mask: np.ndarray,
    headers_data: Mapping[str, str],
    removed_stars: list,
    recipe: Mapping,
    combined_fingerprint: str,
    extra_input_fingerprints: Iterable[str] = (),
    code_version: int = CONVOLVED_RECIPE_SCHEMA_VERSION,
    producer: str = "template_creation.processing.convolved_store.publish_convolved_cell",
) -> dict | None:
    """Atomically publish a canonical convolved-skycell cell. Never raises.

    ``combined_fingerprint`` is the upstream ``combined_skycell`` artifact's
    fingerprint and is required — it is the Merkle input that ties this
    canonical cell to the exact star-removed image it was convolved from
    (plan §13). ``extra_input_fingerprints`` is reserved for a future
    same-projection-neighbor input set; unused today.

    Payload: ``{arrays.npz [convolved_image, convolved_mask], headers.json,
    removed_stars.json}`` (headers/removed_stars pass through from the
    upstream combined cell for lineage) plus a self-describing
    ``_provenance.json`` sidecar. Delegates to
    ``common.provenance.publish.try_publish_dir`` when importable, with the
    same local fallback strategy as ``combined_store.publish_combined_cell``.
    """
    try:
        input_fps = sorted({combined_fingerprint, *extra_input_fingerprints})
        rid = convolved_recipe_id(recipe, code_version)
        fp = convolved_fingerprint(projection, skycell, rid, input_fps)
        spatial_key = _spatial_key(projection, skycell)

        final_dir = convolved_cell_dir(data_root, projection, skycell, fp)
        dest_root = final_dir.parent

        if _payload_complete(final_dir):
            return {"fingerprint": fp, "recipe_id": rid, "dir": final_dir, "already_published": True}

        def _write_payload(tmp_dir: Path) -> None:
            np.savez_compressed(
                tmp_dir / _ARRAYS_FILENAME,
                convolved_image=np.asarray(convolved_image, dtype=np.float32),
                convolved_mask=np.asarray(convolved_mask, dtype=np.uint16),
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
            "publish_convolved_cell failed for projection=%s skycell=%s (best-effort, no raise)",
            projection,
            skycell,
            exc_info=True,
        )
        return None


def try_load_convolved_cell(
    data_root: str | Path, projection: str, skycell: str, fp: str
) -> dict | None:
    """Load a published convolved cell, or ``None`` if absent/incomplete/corrupt."""
    cell_dir = convolved_cell_dir(data_root, projection, skycell, fp)
    try:
        if not _payload_complete(cell_dir):
            return None
        with np.load(cell_dir / _ARRAYS_FILENAME, allow_pickle=False) as z:
            convolved_image = np.array(z["convolved_image"])
            convolved_mask = np.array(z["convolved_mask"])
        with open(cell_dir / _HEADERS_FILENAME) as fh:
            headers_data = json.load(fh)
        with open(cell_dir / _REMOVED_STARS_FILENAME) as fh:
            removed_stars = json.load(fh)
        return {
            "convolved_image": convolved_image,
            "convolved_mask": convolved_mask,
            "headers_data": headers_data,
            "removed_stars": removed_stars,
        }
    except Exception:
        logger.warning("try_load_convolved_cell failed for %s (best-effort)", cell_dir, exc_info=True)
        return None


def resolve_current_convolved_ref(
    data_root: str | Path, projection: str, skycell: str
):
    """Return the validated selected immutable canonical artifact, if any."""
    from syndiff_pipeline.template_creation.processing.artifact_pointer import read_current_pointer

    return read_current_pointer(
        _ps1_convolved_zarr_root(data_root) / str(projection) / str(skycell),
        expected_kind=KIND,
        projection=str(projection),
        skycell=str(skycell),
        required_members=(*_REQUIRED_MEMBERS, _PROVENANCE_SIDECAR_FILENAME),
    )


def convolved_store_inventory(
    data_root: str | Path,
    expected_skycells: Iterable[str],
    *,
    require_current_pointer: bool = False,
) -> dict[str, Any]:
    """Audit exact convolved-cell availability for a P2 master inventory.

    A directory containing ``arrays.npz`` is not sufficient: the immutable
    provenance sidecar and both payload members must be present, and the
    sidecar identity must match the requested projection/cell.  The returned
    partitions are intentionally JSON-serialisable so L5 can include them in
    a hard-failure diagnostic without opening pixel arrays.
    """
    root = _ps1_convolved_zarr_root(data_root)
    present: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    for full_name in sorted({str(v) for v in expected_skycells}):
        parsed = _projection_and_cell(full_name)
        if parsed is None:
            invalid.append(full_name)
            continue
        projection, cell = parsed
        cell_root = root / projection / cell
        candidates: list[Path] = []
        if require_current_pointer:
            ref = resolve_current_convolved_ref(data_root, projection, cell)
            if ref is not None:
                candidates.append(Path(ref.path))
        else:
            try:
                candidates = [p for p in cell_root.iterdir() if p.is_dir() and not p.name.startswith("_tmp_")]
            except OSError:
                candidates = []
        good = False
        for candidate in candidates:
            sidecar = candidate / _PROVENANCE_SIDECAR_FILENAME
            if not _payload_complete(candidate) or not sidecar.is_file():
                continue
            try:
                record = json.loads(sidecar.read_text())
                spatial = record.get("spatial_key", {})
                if (str(record.get("kind")) != KIND or
                        str(spatial.get("projection")) != projection or
                        str(spatial.get("skycell")) != cell):
                    continue
            except Exception:
                continue
            good = True
            break
        (present if good else missing).append(full_name)
    return {
        "expected": sorted({str(v) for v in expected_skycells}),
        "present": present,
        "missing": missing,
        "invalid_identity": invalid,
        "count_expected": len({str(v) for v in expected_skycells}),
        "count_present": len(present),
    }


def update_current_pointer(
    data_root: str | Path, projection: str, skycell: str, fp: str
) -> bool:
    """Select an already-published canonical artifact. Never selects by mtime."""
    from syndiff_pipeline.template_creation.processing.artifact_pointer import (
        validate_artifact,
        write_current_pointer,
    )

    root = _ps1_convolved_zarr_root(data_root) / str(projection) / str(skycell)
    try:
        ref = validate_artifact(
            root, kind=KIND, projection=str(projection), skycell=str(skycell),
            fingerprint=str(fp), recipe_id=None,
            required_members=(*_REQUIRED_MEMBERS, _PROVENANCE_SIDECAR_FILENAME),
        )
        write_current_pointer(root, ref)
        return True
    except Exception:
        logger.warning("failed to update convolved current pointer for %s/%s", projection, skycell, exc_info=True)
        return False
