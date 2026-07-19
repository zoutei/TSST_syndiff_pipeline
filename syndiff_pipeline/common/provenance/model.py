"""
model.py
========
Artifact-kind registry and spatial-key types for the provenance graph.

See ``doc/template_bookkeeping_plan.md`` §5-6. Fifteen kinds total: ten
template-side, five diff-side (§6 table). Each kind has:

- a spatial-key *shape* (one of ``skycell``, ``scc``, ``scc_ffi``, ``event``),
  validated by the dataclasses below;
- a ``recipe_params(...)`` builder that returns the exact allow-listed params
  that affect the artifact's bytes.

Template-side ``recipe_params`` builders mirror the enumerations in
``template_creation/orchestration/verify.py:117-172`` (``config_fingerprint``)
field-for-field -- that module is read-only here, never imported for its
hashing (this package computes its own fingerprints via
``provenance.fingerprint``), only used as the source of truth for *which
fields matter per stage*. Diff-side builders wrap the existing strict
dataclasses in ``difference_imaging/orchestration/stage_params.py`` via
``dataclasses.asdict`` -- those are already the allow-listed recipe, so they
are drift-proof by construction (any new key added there flows through
automatically, and ``validate_stage_keys`` there rejects unknown YAML keys).

This module performs no I/O and has no effect on any compute path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "SPATIAL_KEY_KINDS",
    "SkycellKey",
    "SccKey",
    "SccFfiKey",
    "EventKey",
    "TEMPLATE_KINDS",
    "DIFF_KINDS",
    "ALL_KINDS",
    "KindSpec",
    "KIND_REGISTRY",
    "LEGACY_UNVERIFIED_SUFFIX",
    "legacy_unverified_kind",
    "mapping_recipe_params",
    "remap_store_recipe_params",
    "downsample_recipe_params",
    "ps1_download_recipe_params",
    "ps1_process_recipe_params",
    "ffi_set_recipe_params",
    "raw_skycell_recipe_params",
    "source_catalog_recipe_params",
    "combined_skycell_recipe_params",
    "convolved_skycell_recipe_params",
    "scc_assembly_recipe_params",
    "shared_mask_recipe_params",
    "diff_background_recipe_params",
    "diff_image_recipe_params",
    "epsf_recipe_params",
    "photometry_recipe_params",
]


# ---------------------------------------------------------------------------
# Spatial keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkycellKey:
    """Sky-addressed key: PS1 projection/skycell. No sector/camera/ccd."""

    projection: str
    skycell: str

    def __post_init__(self) -> None:
        if not str(self.projection).strip():
            raise ValueError("SkycellKey.projection must be non-empty")
        if not str(self.skycell).strip():
            raise ValueError("SkycellKey.skycell must be non-empty")

    def to_dict(self) -> dict:
        return {"projection": str(self.projection), "skycell": str(self.skycell)}


@dataclass(frozen=True)
class SccKey:
    """SCC-addressed key, optionally at one oversampling factor."""

    s: int
    c: int
    k: int
    os: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("s", "c", "k"):
            val = getattr(self, name)
            if int(val) != val or int(val) < 0:
                raise ValueError(f"SccKey.{name} must be a non-negative int, got {val!r}")
        if self.os is not None and (int(self.os) != self.os or int(self.os) < 1):
            raise ValueError(f"SccKey.os must be a positive int or None, got {self.os!r}")

    def to_dict(self) -> dict:
        d = {"s": int(self.s), "c": int(self.c), "k": int(self.k)}
        if self.os is not None:
            d["os"] = int(self.os)
        return d


@dataclass(frozen=True)
class SccFfiKey:
    """Per-FFI key within one SCC: ``(s, c, k, product_id[, label])``.

    ``label`` is the diff workspace stage label (e.g. ``hp_d``). Required for
    diff-side kinds per §6; omit or leave empty for bare ``ffi`` input nodes.
    """

    s: int
    c: int
    k: int
    product_id: str
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("s", "c", "k"):
            val = getattr(self, name)
            if int(val) != val or int(val) < 0:
                raise ValueError(f"SccFfiKey.{name} must be a non-negative int, got {val!r}")
        if not str(self.product_id).strip():
            raise ValueError("SccFfiKey.product_id must be non-empty")
        object.__setattr__(self, "label", str(self.label))

    def to_dict(self) -> dict:
        d = {
            "s": int(self.s),
            "c": int(self.c),
            "k": int(self.k),
            "product_id": str(self.product_id),
        }
        if self.label:
            d["label"] = str(self.label)
        return d


@dataclass(frozen=True)
class EventKey:
    """Event-scoped key: ``(event, s, c, k)`` -- photometry only."""

    event: str
    s: int
    c: int
    k: int

    def __post_init__(self) -> None:
        if not str(self.event).strip():
            raise ValueError("EventKey.event must be non-empty")
        for name in ("s", "c", "k"):
            val = getattr(self, name)
            if int(val) != val or int(val) < 0:
                raise ValueError(f"EventKey.{name} must be a non-negative int, got {val!r}")

    def to_dict(self) -> dict:
        return {
            "event": str(self.event),
            "s": int(self.s),
            "c": int(self.c),
            "k": int(self.k),
        }


SPATIAL_KEY_KINDS = ("skycell", "scc", "scc_ffi", "event")


# ---------------------------------------------------------------------------
# Kind registry
# ---------------------------------------------------------------------------

TEMPLATE_KINDS: tuple[str, ...] = (
    "ffi",
    "ffi_set",
    "raw_skycell",
    "source_catalog",
    "mapping",
    "remap_store",
    "combined_skycell",
    "convolved_skycell",
    "scc_assembly",
    "downsample",
)

DIFF_KINDS: tuple[str, ...] = (
    "shared_mask",
    "diff_background",
    "diff_image",
    "epsf",
    "photometry",
)

ALL_KINDS: tuple[str, ...] = TEMPLATE_KINDS + DIFF_KINDS

LEGACY_UNVERIFIED_SUFFIX = "_legacy_unverified"


def legacy_unverified_kind(kind: str) -> str:
    """Kind label for a reindexed product whose recipe could not be verified.

    Decision #8: legacy products discovered at reindex (no matching
    ``_provenance.json`` or a recipe that doesn't match any known builder)
    are recorded under ``{kind}_legacy_unverified`` rather than dropped, so
    they are still visible in queries but never satisfy a
    ``scc_stage_complete`` check against a freshly-computed fingerprint.
    """
    if kind.endswith(LEGACY_UNVERIFIED_SUFFIX):
        return kind
    return f"{kind}{LEGACY_UNVERIFIED_SUFFIX}"


@dataclass(frozen=True)
class KindSpec:
    """Static metadata for one artifact kind (§6 table row)."""

    kind: str
    spatial_key_kind: str  # one of SPATIAL_KEY_KINDS
    inputs: tuple[str, ...]  # input kinds this artifact consumes (documentation only)
    description: str = ""


KIND_REGISTRY: dict[str, KindSpec] = {
    "ffi": KindSpec("ffi", "scc_ffi", (), "TESS FFI input node; version = ffi_list row"),
    "ffi_set": KindSpec("ffi_set", "scc", ("ffi",), "N x ffi for one SCC"),
    "raw_skycell": KindSpec("raw_skycell", "skycell", (), "raw grizy PS1 skycell input"),
    "source_catalog": KindSpec("source_catalog", "skycell", (), "Gaia footprint catalog"),
    "mapping": KindSpec("mapping", "scc", ("ffi_set",), "pixel<->skycell mapping"),
    "remap_store": KindSpec("remap_store", "scc", ("ffi_set", "mapping"), "field L2-L4 remap store"),
    "combined_skycell": KindSpec(
        "combined_skycell", "skycell", ("raw_skycell", "source_catalog"), "band-combined + star-removed"
    ),
    "convolved_skycell": KindSpec(
        "convolved_skycell", "skycell", ("combined_skycell",), "PSF-convolved skycell"
    ),
    "scc_assembly": KindSpec(
        "scc_assembly", "scc", ("mapping", "convolved_skycell"), "per-SCC convolved.zarr checkpoint"
    ),
    "downsample": KindSpec(
        "downsample", "scc", ("scc_assembly", "remap_store"), "final template FITS products"
    ),
    "shared_mask": KindSpec("shared_mask", "scc", ("ffi_set",), "SCC-scoped shared diff mask"),
    "diff_background": KindSpec("diff_background", "scc_ffi", ("ffi",), "per-FFI background image"),
    "diff_image": KindSpec(
        "diff_image",
        "scc_ffi",
        ("ffi", "downsample", "shared_mask", "diff_background"),
        "per-FFI difference image",
    ),
    "epsf": KindSpec("epsf", "scc_ffi", ("diff_image",), "per-FFI gridded ePSF"),
    "photometry": KindSpec(
        "photometry", "event", ("diff_image", "epsf"), "per-event forced photometry"
    ),
}

assert set(KIND_REGISTRY) == set(ALL_KINDS)


def _asdict(obj: Any) -> dict:
    """``dataclasses.asdict`` for a stage-params dataclass, or pass through a dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"expected a dataclass instance or Mapping, got {type(obj)!r}")


# ---------------------------------------------------------------------------
# Template-side recipe_params builders.
#
# Field lists copied verbatim from
# template_creation/orchestration/verify.py:117-172 (`config_fingerprint`);
# that function is not imported or modified here.
# ---------------------------------------------------------------------------


def mapping_recipe_params(resolved) -> dict:
    """``mapping`` recipe params (verify.py:132)."""
    mp = resolved.stages.mapping
    return {
        "oversampling_factor": mp.oversampling_factor,
        "pad_distance": mp.pad_distance,
        "overwrite": mp.overwrite,
    }


def ps1_process_recipe_params(resolved) -> dict:
    """``ps1_process`` recipe params (verify.py's ``ps1_process`` branch).

    Used directly for ``combined_skycell``/``convolved_skycell``/
    ``scc_assembly`` in PR1: Phase 1/2 (§12/§13) have not landed on this
    branch, so those kinds do not yet have their own finer-grained param
    breakdown (band-combine constants, psf radius/mode, gaia_version). Until
    then this single builder stands in for all three per §11's checkpoint
    recipe (``psf_sigma``/saturation/removal params + ``projections_limit``).
    """
    pp = resolved.stages.ps1_process
    return {
        "projections_limit": pp.projections_limit,
        "psf_sigma": pp.psf_sigma,
        "enable_saturation_correction": pp.enable_saturation_correction,
        "remove_saturated_stars": pp.remove_saturated_stars,
        "bright_star_mag_threshold": pp.bright_star_mag_threshold,
    }


def remap_store_recipe_params(resolved) -> dict:
    """``remap_store`` recipe params (verify.py:146)."""
    rm = resolved.stages.remap
    mp = resolved.stages.mapping
    return {
        "oversampling_factor": mp.oversampling_factor,
        "cache_quantum_ps1_px": rm.cache_quantum_ps1_px,
        "keying": rm.keying,
        "apply_hybrid_exact": rm.apply_hybrid_exact,
        "hybrid_R": rm.hybrid_R,
        "include_abutting_border_exact": rm.include_abutting_border_exact,
    }


def downsample_recipe_params(resolved) -> dict:
    """``downsample`` recipe params (verify.py:159)."""
    ds = resolved.stages.downsample
    return {
        "oversampling_factor": ds.oversampling_factor,
        "single_offset": ds.single_offset,
        "ignore_mask_bits": list(ds.ignore_mask_bits),
        "output_base": ds.output_base or resolved.template_output_base,
    }


def ps1_download_recipe_params(resolved) -> dict:
    """``ps1_download`` recipe params (verify.py's ``ps1_download`` branch)."""
    pd = resolved.stages.ps1_download
    return {
        "overwrite": pd.overwrite,
        "use_local_files": pd.use_local_files,
    }


def ffi_set_recipe_params() -> dict:
    """``ffi_set`` recipe params.

    ``verify.config_fingerprint`` has no ``tess_ffi_download`` branch (that
    stage precedes the fingerprinted stage tuple, see §2.1). FFI byte content
    is defined entirely by the archive (MAST) and product id, not by download
    mechanics -- so this recipe is intentionally empty; the ``ffi_set``
    artifact's identity comes from its N x ``ffi`` input fingerprints.
    """
    return {}


def raw_skycell_recipe_params() -> dict:
    """``raw_skycell`` recipe params: none. Identity is the version token
    (size, mtime, download_batch_id) recorded in ``input_files``, per decision #6.
    """
    return {}


# Placeholder Gaia catalog version until Phase 1 (§12) plumbs a real
# per-footprint value through ps1_process's star-removal path. Scalar for
# now per §21 open question ("start scalar").
_DEFAULT_GAIA_VERSION = "dr3"


def source_catalog_recipe_params(
    *,
    gaia_version: str = _DEFAULT_GAIA_VERSION,
    mag_threshold: Optional[float] = None,
) -> dict:
    """``source_catalog`` recipe params: Gaia query params + version."""
    d: dict = {"gaia_version": str(gaia_version)}
    if mag_threshold is not None:
        d["mag_threshold"] = float(mag_threshold)
    return d


def combined_skycell_recipe_params(resolved, *, gaia_version: str = _DEFAULT_GAIA_VERSION) -> dict:
    """``combined_skycell`` recipe params.

    Phase 1 (§12) stub: saturation/star-removal params come from the current
    ``ps1_process`` stage config (there is no separate combine-stage config on
    this branch yet); ``gaia_version`` is threaded through explicitly.
    """
    params = ps1_process_recipe_params(resolved)
    params["gaia_version"] = str(gaia_version)
    return params


def convolved_skycell_recipe_params(resolved) -> dict:
    """``convolved_skycell`` recipe params.

    Phase 2 (§13) stub: ``radius``/``mode``/``padding`` are not yet
    independent config knobs (convolution today is one call inside
    ``ps1_process`` keyed only by ``psf_sigma``); recorded as fixed literals
    until the padding decouple lands.
    """
    pp = resolved.stages.ps1_process
    return {
        "psf_sigma": pp.psf_sigma,
        "radius": None,
        "mode": "same",
        "padding": "same_projection_only",
    }


def scc_assembly_recipe_params(resolved) -> dict:
    """``scc_assembly`` checkpoint recipe params (§11: same fields as
    ``ps1_process_recipe_params`` -- "psf_sigma/saturation/removal params +
    projections_limit").
    """
    return ps1_process_recipe_params(resolved)


# ---------------------------------------------------------------------------
# Diff-side recipe_params builders -- dataclasses.asdict of the strict
# stage_params dataclasses (already the exact allow-list).
# ---------------------------------------------------------------------------


def shared_mask_recipe_params(params: Any) -> dict:
    """``shared_mask`` recipe params from ``SharedMaskParams``."""
    return _asdict(params)


def diff_background_recipe_params(params: Any) -> dict:
    """``diff_background`` recipe params from ``BackgroundParams`` (+ steps)
    or the hotpants-internal background params dataclass, whichever the
    caller's stage used.
    """
    return _asdict(params)


def diff_image_recipe_params(*params: Any) -> dict:
    """``diff_image`` recipe params.

    Accepts one dataclass (``HotpantsParams``) or two (``KernelFitParams``,
    ``KernelSubtractParams``) and merges them under their dataclass type name
    so the two producer families never collide on a shared key.
    """
    if not params:
        raise ValueError("diff_image_recipe_params() requires at least one stage-params object")
    merged: dict = {}
    for p in params:
        merged[type(p).__name__] = _asdict(p)
    return merged


def epsf_recipe_params(params: Any) -> dict:
    """``epsf`` recipe params from ``EpsfParams``."""
    return _asdict(params)


def photometry_recipe_params(method_params: Any) -> dict:
    """``photometry`` recipe params from a photometry method params dataclass
    (``PsfPhotometryMethodParams`` | ``AperturePhotometryMethodParams``).
    """
    return _asdict(method_params)
