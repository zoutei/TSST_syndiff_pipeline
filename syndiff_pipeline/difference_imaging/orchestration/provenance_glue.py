"""
provenance_glue.py
===================
Diff-side glue for the content-addressed provenance graph — PR-D1, Phase D1
"track in place" (``doc/template_bookkeeping_plan.md`` §14.1, kind registry
§6, publish/query protocol §9-10).

**Scope discipline (decision #16):** this module only *tracks* already-written
per-FFI diff artifacts (background, diff image, ePSF, shared mask). It never
changes what gets written or where — every product keeps landing at the exact
event-workspace path it writes today. ``emit_diff_artifact`` records a
best-effort sidecar record for a file that some other code already wrote to
disk: it appends one JSON line to this process's provenance spool file
(``common.provenance.publish.append_spool_record``); it never moves bytes,
never touches ``provenance.db`` directly (ingest is the supervisor's job —
out of this module's territory), and never raises into the pipeline.

**Provenance package status.** ``syndiff_pipeline.common.provenance`` (PR1:
fingerprint/model/publish/store) landed concurrently on this branch while
this module was being written. Every symbol this module needs is imported
from the *real* package (not a guessed shape); each import is still wrapped
in ``try/except`` so this module keeps working — degrading to a no-op for
anything that touches provenance — if a future refactor removes/renames a
symbol, or in a partial checkout where the package is absent.

Two adaptations from the literal spatial-key shapes named in this task's
brief, made to match the *actual* landed ``common.provenance.model`` kind
registry (the authoritative contract once the package exists — reusing it is
what makes fingerprints computed here agree with fingerprints computed by any
other producer of the same kind):

- Per-FFI diff kinds (``diff_background``, ``diff_image``, ``epsf``) use
  ``model.SccFfiKey(s, c, k, product_id, label=...)`` per §6 so distinct
  workspace labels over the same FFI do not collide in ``provenance.db``.
  Bare ``ffi`` input nodes omit ``label``. ``shared_mask`` uses
  ``model.SccKey(s, c, k)`` (SCC-scoped, not per-FFI).
- Diff-side ``recipe_params`` come from ``model.diff_image_recipe_params`` /
  ``diff_background_recipe_params`` / ``epsf_recipe_params`` /
  ``shared_mask_recipe_params`` (already landed, already doing exactly the
  ``dataclasses.asdict`` + classname-namespacing this task asked for) rather
  than a second, locally-reinvented implementation that could silently drift
  from the canonical one.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from syndiff_pipeline.common.fits_variants import fits_logical_path
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    tess_product_id_from_ffi_path,
)

log = logging.getLogger(__name__)

# ── Guarded imports against the real, landed provenance package ────────────
try:
    from syndiff_pipeline.common.provenance.fingerprint import (
        RECIPE_SCHEMA_VERSION as _RECIPE_SCHEMA_VERSION,
        canonical as _canonical,
        fingerprint as _fingerprint_fn,
        recipe_id as _recipe_id_fn,
    )
except Exception:  # pragma: no cover - exercised via PROVENANCE_AVAILABLE=False tests
    _RECIPE_SCHEMA_VERSION = 1
    _canonical = None
    _fingerprint_fn = None
    _recipe_id_fn = None

try:
    from syndiff_pipeline.common.provenance.model import (
        SccFfiKey as _SccFfiKey,
        SccKey as _SccKey,
        diff_background_recipe_params as _diff_background_recipe_params,
        diff_image_recipe_params as _diff_image_recipe_params,
        epsf_recipe_params as _epsf_recipe_params,
        shared_mask_recipe_params as _shared_mask_recipe_params,
    )
except Exception:  # pragma: no cover
    _SccFfiKey = None
    _SccKey = None
    _diff_background_recipe_params = None
    _diff_image_recipe_params = None
    _epsf_recipe_params = None
    _shared_mask_recipe_params = None

try:
    from syndiff_pipeline.common.provenance.publish import (
        append_spool_record as _append_spool_record,
        build_record as _build_record,
    )
except Exception:  # pragma: no cover
    _append_spool_record = None
    _build_record = None

try:
    from syndiff_pipeline.common.provenance.store import ProvenanceStore as _ProvenanceStore
except Exception:  # pragma: no cover
    _ProvenanceStore = None

try:
    # Read-only use of scc_paths' bookkeeping-location helpers (this module
    # never creates/edits scc_paths.py itself, per this PR's file ownership).
    from syndiff_pipeline.common.scc_paths import (
        provenance_db_path as _provenance_db_path,
        provenance_spool_dir as _provenance_spool_dir,
    )
except Exception:  # pragma: no cover
    _provenance_db_path = None
    _provenance_spool_dir = None

#: True once fingerprinting is importable (the minimum needed to compute a
#: content-addressed identity). Recipe/spatial-key/required-set helpers work
#: regardless of this flag; anything that needs a real fingerprint checks it.
PROVENANCE_AVAILABLE = _fingerprint_fn is not None and _recipe_id_fn is not None

#: True once the sidecar-spool write path is importable.
_SPOOL_AVAILABLE = (
    _append_spool_record is not None
    and _build_record is not None
    and _provenance_spool_dir is not None
)

#: True once the read-only completeness-query path is importable.
_STORE_AVAILABLE = _ProvenanceStore is not None and _provenance_db_path is not None

#: Fallback recipe-schema code_version used only when the real
#: ``fingerprint.RECIPE_SCHEMA_VERSION`` could not be imported (keeps this
#: module deterministic even in a partial checkout).
DIFF_RECIPE_SCHEMA_VERSION = _RECIPE_SCHEMA_VERSION

FFI_SCOPED_KINDS = frozenset({"diff_background", "diff_image", "epsf"})
SHARED_MASK_KIND = "shared_mask"


# ── Recipe builders (§6 kind registry, diff-side rows) ──────────────────────

_MODEL_RECIPE_BUILDERS = {
    "diff_background": lambda parts: _diff_background_recipe_params(parts[0]),
    "diff_image": lambda parts: _diff_image_recipe_params(*parts),
    "epsf": lambda parts: _epsf_recipe_params(parts[0]),
    "shared_mask": lambda parts: _shared_mask_recipe_params(parts[0]),
}


def _asdict_any(obj: Any) -> dict:
    import dataclasses
    from typing import Mapping

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(
        f"diff_recipe: params must be a dataclass instance or mapping, got {type(obj)!r}"
    )


def diff_recipe(kind: str, params_dataclass: Any) -> dict:
    """
    Materialize one or more strict param dataclasses into an allow-listed recipe dict.

    ``params_dataclass`` is normally a single dataclass instance
    (``HotpantsParams``, ``EpsfParams``, ``BackgroundParams`` — nested step
    params flatten automatically, ``SharedMaskParams``). For kinds whose
    recipe spans two stage configs (``KernelFitParams`` +
    ``KernelSubtractParams`` for kernel-matched ``diff_image``), pass a
    list/tuple of dataclass instances.

    Delegates to ``common.provenance.model``'s landed ``*_recipe_params``
    builders when available (the canonical implementation every other
    producer of these kinds also uses); falls back to a local
    ``dataclasses.asdict`` + classname-namespacing reimplementation of the
    exact same rule when the package is unavailable, so recipe dicts stay
    identical either way.

    Returns ``{"kind": kind, "params": <dict>, "code_version": <int>}``.
    """
    parts = params_dataclass if isinstance(params_dataclass, (list, tuple)) else [params_dataclass]
    builder = _MODEL_RECIPE_BUILDERS.get(kind)
    if builder is not None:
        try:
            params = builder(list(parts))
            return {"kind": str(kind), "params": params, "code_version": DIFF_RECIPE_SCHEMA_VERSION}
        except Exception:
            log.debug("diff_recipe(%s): model builder failed, using local fallback", kind, exc_info=True)
    if len(parts) == 1:
        params = _asdict_any(parts[0])
    else:
        params = {type(p).__name__: _asdict_any(p) for p in parts}
    return {"kind": str(kind), "params": params, "code_version": DIFF_RECIPE_SCHEMA_VERSION}


# ── Spatial keys ─────────────────────────────────────────────────────────────


def ffi_spatial_key(
    sector: int, camera: int, ccd: int, product_id: str, label: str = ""
) -> dict:
    """
    Spatial key for per-FFI kinds (``diff_background``, ``diff_image``, ``epsf``).

    Delegates to ``model.SccFfiKey(s, c, k, product_id, label=label)``.
    """
    if _SccFfiKey is not None:
        try:
            return _SccFfiKey(
                int(sector), int(camera), int(ccd), str(product_id), label=str(label or "")
            ).to_dict()
        except Exception:
            log.debug("ffi_spatial_key: model.SccFfiKey failed, using local fallback", exc_info=True)
    d = {"s": int(sector), "c": int(camera), "k": int(ccd), "product_id": str(product_id)}
    if label:
        d["label"] = str(label)
    return d


def shared_mask_spatial_key(sector: int, camera: int, ccd: int, *, label: str = "") -> dict:
    """Spatial key for ``shared_mask`` — ``model.SccKey(s, c, k)``, no ``label`` field."""
    if _SccKey is not None:
        try:
            return _SccKey(int(sector), int(camera), int(ccd)).to_dict()
        except Exception:
            log.debug("shared_mask_spatial_key: model.SccKey failed, using local fallback", exc_info=True)
    return {"s": int(sector), "c": int(camera), "k": int(ccd)}


def product_id_for_ffi(path_or_basename: str) -> Optional[str]:
    """``tess<digits>`` product id for an FFI path/basename (thin re-export)."""
    return tess_product_id_from_ffi_path(path_or_basename)


# ── Input-edge helpers ───────────────────────────────────────────────────────


def ffi_input_version(
    ffi_path: str, ffi_list_row: Optional["pd.Series"] = None
) -> dict:
    """
    Version token for an FFI input node (decision #6: logical basename, size, mtime).

    Prefers a direct ``stat`` of *ffi_path* (matches the spec exactly). Falls
    back to whatever the ``ffi_list`` row carries (``wcs_ok``, ``date_obs``)
    when the file itself is not reachable but a cached row is available
    (e.g. offline reindex / tests).
    """
    try:
        logical_basename = Path(fits_logical_path(ffi_path)).name
    except Exception:
        logical_basename = Path(str(ffi_path)).name
    token: dict = {"logical_basename": logical_basename}
    try:
        st = os.stat(ffi_path)
        token["size"] = int(st.st_size)
        token["mtime"] = float(st.st_mtime)
        return token
    except OSError:
        pass
    if ffi_list_row is not None:
        wcs_ok = ffi_list_row.get("wcs_ok") if hasattr(ffi_list_row, "get") else None
        token["wcs_ok"] = bool(wcs_ok) if wcs_ok is not None else None
        date_obs = ffi_list_row.get("date_obs") if hasattr(ffi_list_row, "get") else None
        if date_obs is not None and not (isinstance(date_obs, float) and pd.isna(date_obs)):
            token["date_obs"] = str(date_obs)
    return token


def ffi_input_fingerprint(
    sector: int,
    camera: int,
    ccd: int,
    ffi_path: str,
    ffi_list_row: Optional["pd.Series"] = None,
) -> Optional[str]:
    """
    Content-addressed fingerprint for one ``ffi`` input node, or ``None`` if unavailable.

    ``ffi``'s spatial key is ``scc_ffi`` too (registry: ``KIND_REGISTRY["ffi"]``)
    — it needs the consuming stage's ``(sector, camera, ccd)``, not just the
    FFI's own product id, hence the three leading positional args.
    """
    if not PROVENANCE_AVAILABLE:
        return None
    product_id = tess_product_id_from_ffi_path(ffi_path)
    if not product_id:
        return None
    spatial_key = ffi_spatial_key(sector, camera, ccd, product_id)
    version = ffi_input_version(ffi_path, ffi_list_row)
    try:
        rid = _recipe_id_fn("ffi", version, DIFF_RECIPE_SCHEMA_VERSION)
        return _fingerprint_fn("ffi", spatial_key, rid, [])
    except Exception:
        log.debug("ffi_input_fingerprint failed for %s", ffi_path, exc_info=True)
        return None


def _stable_location_token(location: str) -> str:
    """Deterministic stand-in fingerprint keyed on a logical file location.

    Used for edges into nodes this module does not own the recipe for
    (template ``downsample`` nodes) or cross-stage/cross-process upstream
    diff artifacts reachable here only as an on-disk file path. A live
    ``store`` lookup by ``(kind, spatial_key)`` is deliberately *not* used
    for these edges: the upstream artifact's sidecar record is very likely
    still sitting in a spool file (not yet drained by the supervisor) when a
    downstream stage of the *same* pipeline run reads it moments later, so a
    DB-backed lookup would almost always miss. A location hash is at least
    deterministic and available immediately. PENDING: once there is a
    reliable way to resolve the *real* upstream fingerprint at emit time
    (e.g. threading it forward in-process instead of round-tripping through
    the store), swap this for the real value — every diff-side recipe that
    consumes one of these edges already depends on the returned string, so
    the swap re-fingerprints the diff cone automatically (Merkle
    invalidation, plan §3.3).
    """
    return "loc:" + hashlib.sha256(str(location).encode("utf-8")).hexdigest()[:24]


def location_edge(kind: str, path: str, *, is_fits: bool = True) -> dict:
    """Best-effort edge to a node this module doesn't own the recipe for (see :func:`_stable_location_token`)."""
    location = fits_logical_path(path) if is_fits else str(path)
    return {"kind": str(kind), "location": location, "fingerprint": _stable_location_token(location)}


def template_input_edge(template_path: str) -> dict:
    """Edge to the ``downsample`` node that produced *template_path* (best-effort, see :func:`location_edge`)."""
    return location_edge("downsample", template_path)


def upstream_label_edge(kind: str, path: str) -> dict:
    """Edge to an upstream diff-side artifact referenced by its written file (best-effort, see :func:`location_edge`)."""
    return location_edge(kind, path)


def gaia_catalog_edge(path: str) -> dict:
    """Edge to the Gaia catalog CSV consumed by ``epsf`` (best-effort, see :func:`location_edge`)."""
    return location_edge("source_catalog", path, is_fits=False)


# ── Fingerprint computation ──────────────────────────────────────────────────


def diff_kind_fingerprint(
    kind: str,
    *,
    sector: int,
    camera: int,
    ccd: int,
    product_id: str,
    label: str = "",
    params: Any,
    input_fingerprints: Sequence[Optional[str]] = (),
) -> Optional[str]:
    """
    Content-addressed fingerprint for one per-FFI diff artifact.

    Returns ``None`` when the provenance package is unavailable, or when any
    entry of *input_fingerprints* is ``None`` (an edge that failed to
    resolve) — fail open rather than mint a fingerprint that silently omits
    a dependency it should have recorded.
    """
    if not PROVENANCE_AVAILABLE:
        return None
    if any(fp is None for fp in input_fingerprints):
        return None
    spatial_key = ffi_spatial_key(sector, camera, ccd, product_id, label)
    recipe = diff_recipe(kind, params)
    try:
        rid = _recipe_id_fn(kind, recipe["params"], recipe["code_version"])
        return _fingerprint_fn(
            kind, spatial_key, rid, sorted(fp for fp in input_fingerprints if fp)
        )
    except Exception:
        log.debug("diff_kind_fingerprint(%s) failed", kind, exc_info=True)
        return None


def diff_kind_fingerprint_shared_mask(
    sector: int,
    camera: int,
    ccd: int,
    params: Any,
    *,
    label: str = "shared_mask",
    input_fingerprints: Sequence[Optional[str]] = (),
) -> Optional[str]:
    """Content-addressed fingerprint for the ``shared_mask`` node."""
    if not PROVENANCE_AVAILABLE:
        return None
    if any(fp is None for fp in input_fingerprints):
        return None
    spatial_key = shared_mask_spatial_key(sector, camera, ccd, label=label)
    recipe = diff_recipe(SHARED_MASK_KIND, params)
    try:
        rid = _recipe_id_fn(SHARED_MASK_KIND, recipe["params"], recipe["code_version"])
        return _fingerprint_fn(
            SHARED_MASK_KIND, spatial_key, rid, sorted(fp for fp in input_fingerprints if fp)
        )
    except Exception:
        log.debug("diff_kind_fingerprint(shared_mask) failed", exc_info=True)
        return None


# ── Required-set derivation (frame manifest) ─────────────────────────────────


def required_product_ids(
    frames_df: "pd.DataFrame",
    *,
    require_wcs_ok: bool = True,
    require_group_assigned: bool = True,
) -> list[str]:
    """
    Expected per-FFI product ids for a diff stage, derived from ``frames.csv``.

    Respects the exclusion signal already recorded on the manifest by the
    rest of the diff pipeline: rows with ``wcs_ok`` false (when the column
    exists) and rows with no assigned ``group_id`` (< 0, i.e. ungrouped /
    excluded frames) are not required. Falls back to every row with a
    resolvable ``tess<digits>`` product id when neither column is present.
    """
    if frames_df is None or len(frames_df) == 0:
        return []
    df = frames_df
    mask = pd.Series(True, index=df.index)
    if require_wcs_ok and "wcs_ok" in df.columns:
        mask &= df["wcs_ok"].fillna(False).astype(bool)
    if require_group_assigned and "group_id" in df.columns:
        gid = pd.to_numeric(df["group_id"], errors="coerce")
        mask &= gid.notna() & (gid >= 0)

    from syndiff_pipeline.difference_imaging.support.manifest import (
        row_ffi_product_id_series,
    )

    pids = row_ffi_product_id_series(df)
    out = sorted({str(p) for p, keep in zip(pids, mask) if bool(keep) and p})
    return out


# ── Publish (best-effort sidecar record) ─────────────────────────────────────


def _emit_record(
    *,
    fp: str,
    kind: str,
    spatial_key: dict,
    recipe: dict,
    input_fingerprints: Sequence[Optional[str]],
    location: str,
    data_root: str,
    meta: Optional[dict],
) -> bool:
    """Build one sidecar record and append it to this process's spool file."""
    if not _SPOOL_AVAILABLE:
        return False
    rid = _recipe_id_fn(kind, recipe["params"], recipe["code_version"])
    record = _build_record(
        fp,
        kind,
        spatial_key,
        rid,
        recipe["code_version"],
        [f for f in input_fingerprints if f],
        location,
        recipe_params=recipe["params"],
        meta=meta,
    )
    spool_dir = _provenance_spool_dir(data_root)
    _append_spool_record(spool_dir, record)
    return True


def emit_diff_artifact(
    *,
    kind: str,
    sector: int,
    camera: int,
    ccd: int,
    product_id: str,
    label: str,
    params: Any,
    location: str,
    input_fingerprints: Sequence[Optional[str]] = (),
    data_root: Optional[str] = None,
    meta: Optional[dict] = None,
    is_fits: bool = True,
    workspace_root: Optional[str] = None,
    scc_primary: bool = False,
    output_store_name: Optional[str] = None,
) -> Optional[str]:
    """
    Best-effort sidecar record for an already-written per-FFI file.

    Never raises into the pipeline: every failure mode (missing provenance
    package, unwritable spool dir, bad params, ``data_root`` unset) is caught
    and logged at DEBUG; on any failure this returns ``None`` and the caller
    proceeds exactly as it would have without provenance tracking.

    On success, returns the computed fingerprint so the caller may pass it
    forward as an ``input_fingerprints`` entry for a downstream kind within
    the same process (e.g. ``diff_image``'s fingerprint feeding ``epsf``).
    """
    try:
        if not PROVENANCE_AVAILABLE or not _SPOOL_AVAILABLE or not data_root:
            return None
        fp = diff_kind_fingerprint(
            kind,
            sector=sector,
            camera=camera,
            ccd=ccd,
            product_id=product_id,
            label=label,
            params=params,
            input_fingerprints=input_fingerprints,
        )
        if fp is None:
            return None
        recipe = diff_recipe(kind, params)
        spatial_key = ffi_spatial_key(sector, camera, ccd, product_id, label)
        full_meta = dict(meta or {})
        full_meta.setdefault("label", label)
        if output_store_name is not None:
            full_meta.setdefault("output_store_name", output_store_name)
        location_str = fits_logical_path(location) if is_fits else str(location)
        if _emit_record(
            fp=fp,
            kind=kind,
            spatial_key=spatial_key,
            recipe=recipe,
            input_fingerprints=input_fingerprints,
            location=location_str,
            data_root=str(data_root),
            meta=full_meta,
        ):
            if scc_primary and workspace_root:
                try:
                    from syndiff_pipeline.difference_imaging.orchestration.diff_store import (
                        record_scc_artifact_pointer,
                    )

                    rid = _recipe_id_fn(kind, recipe["params"], recipe["code_version"])
                    record_scc_artifact_pointer(
                        workspace_root=workspace_root,
                        product_id=product_id,
                        label=label,
                        scc_path=location_str,
                        kind=kind,
                        fingerprint=fp,
                        stage_label=label,
                        recipe_fp=rid,
                    )
                except Exception:
                    log.debug("SCC diff-store pointer record failed", exc_info=True)
            return fp
        return None
    except Exception:
        log.debug(
            "emit_diff_artifact(kind=%s, product_id=%s) failed; continuing without provenance",
            kind,
            product_id,
            exc_info=True,
        )
        return None


def emit_shared_mask_artifact(
    *,
    sector: int,
    camera: int,
    ccd: int,
    params: Any,
    location: str,
    data_root: Optional[str] = None,
    label: str = "shared_mask",
    meta: Optional[dict] = None,
) -> Optional[str]:
    """``emit_diff_artifact`` counterpart for the ``shared_mask`` kind (different spatial key shape)."""
    try:
        if not PROVENANCE_AVAILABLE or not _SPOOL_AVAILABLE or not data_root:
            return None
        fp = diff_kind_fingerprint_shared_mask(sector, camera, ccd, params, label=label)
        if fp is None:
            return None
        recipe = diff_recipe(SHARED_MASK_KIND, params)
        spatial_key = shared_mask_spatial_key(sector, camera, ccd, label=label)
        full_meta = dict(meta or {})
        full_meta.setdefault("label", label)
        if _emit_record(
            fp=fp,
            kind=SHARED_MASK_KIND,
            spatial_key=spatial_key,
            recipe=recipe,
            input_fingerprints=(),
            location=fits_logical_path(location),
            data_root=str(data_root),
            meta=full_meta,
        ):
            return fp
        return None
    except Exception:
        log.debug("emit_shared_mask_artifact failed; continuing without provenance", exc_info=True)
        return None


# ── Query (used by diff_verify) ──────────────────────────────────────────────


def open_store(data_root: str) -> Optional[Any]:
    """Best-effort read-only ``ProvenanceStore`` handle, or ``None`` if unavailable/unconstructible."""
    if not _STORE_AVAILABLE or not data_root:
        return None
    try:
        db_path = _provenance_db_path(data_root)
        return _ProvenanceStore(db_path, read_only=True)
    except Exception:
        log.debug("open_store(%s) failed", data_root, exc_info=True)
        return None


def close_store(store: Any) -> None:
    """Best-effort close for a handle returned by :func:`open_store` (no-op: ProvenanceStore has no persistent handle)."""
    return None
