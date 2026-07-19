"""Provenance graph model: Recipe / Artifact dataclasses and per-kind builders.

This module is the *knowledge layer* of the content-addressed provenance graph
described in ``doc/template_bookkeeping_plan.md`` (§5 model, §6 kind registry,
§9 fingerprint). It migrates the per-stage "which params matter" enumeration that
used to live in :func:`verify.config_fingerprint` into one ``recipe_params``
builder per artifact kind.

Design constraints:

- Stdlib-light. We import the fingerprint contract lazily inside methods to
  avoid any import cycle, and we never import ``zarr``/``astropy`` here so the
  scheduler/daemon can use this model cheaply.
- Spatial keys and recipe params are plain JSON-serializable dicts with stable
  field names (see §6). ``fingerprint.canonical`` handles sorting/normalization,
  so key *order* here is irrelevant to identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Kind registry (§6). Kept as plain string constants + a frozenset so callers
# can validate without importing an enum machinery.
# ---------------------------------------------------------------------------
FFI_SET = "ffi_set"
RAW_SKYCELL = "raw_skycell"
SOURCE_CATALOG = "source_catalog"
MAPPING = "mapping"
WCS_GROUP = "wcs_group"
COMBINED_SKYCELL = "combined_skycell"
CONVOLVED_SKYCELL = "convolved_skycell"
SCC_ASSEMBLY = "scc_assembly"
TEMPLATE = "template"

KINDS: frozenset = frozenset(
    {
        FFI_SET,
        RAW_SKYCELL,
        SOURCE_CATALOG,
        MAPPING,
        WCS_GROUP,
        COMBINED_SKYCELL,
        CONVOLVED_SKYCELL,
        SCC_ASSEMBLY,
        TEMPLATE,
    }
)

# Convolution boundary constants. The producer
# (``convolution_utils.apply_gaussian_convolution``) is called with only
# ``sigma=psf_sigma``; ``radius`` and ``mode`` use the function defaults, so they
# are recorded here as constants (overridable via ``recipe_params`` extras) to
# keep the convolved recipe honest about what actually shaped the bytes.
CONVOLVE_RADIUS_DEFAULT: int = 470
CONVOLVE_MODE_DEFAULT: str = "constant"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Recipe:
    """A fully-materialized recipe: the params that produced an artifact kind.

    ``recipe_id`` is a content address ``H(kind, params, code_version)``.
    """

    kind: str
    params: Dict[str, Any]
    code_version: str
    git_sha: Optional[str] = None

    def recipe_id(self) -> str:
        """Content-address this recipe via the shared fingerprint contract."""
        # NB: import the names, not the module — the ``provenance`` package
        # ``__init__`` rebinds the ``fingerprint`` attribute to the *function*,
        # so ``import ...provenance.fingerprint as fp`` would yield that function.
        from syndiff_pipeline.common.provenance.fingerprint import recipe_id

        return recipe_id(self.kind, self.params, self.code_version)


@dataclass
class Artifact:
    """A node in the provenance DAG (§5).

    Identity is the Merkle fingerprint
    ``H(kind, spatial_key, recipe_id, sorted(input_fingerprints))``. Everything
    else (location, state, bytes, ...) is metadata about a materialization of
    that identity.
    """

    kind: str
    spatial_key: Dict[str, Any]
    recipe: Recipe
    inputs: List[str] = field(default_factory=list)
    location: Optional[str] = None
    state: str = "building"
    bytes: Optional[int] = None
    wall_time_s: Optional[float] = None
    produced_by: Optional[str] = None

    def fingerprint(self) -> str:
        """Merkle fingerprint of this artifact node."""
        from syndiff_pipeline.common.provenance.fingerprint import (
            fingerprint as merkle_fingerprint,
        )

        return merkle_fingerprint(
            self.kind,
            self.spatial_key,
            self.recipe.recipe_id(),
            list(self.inputs),
        )

    def to_record(self) -> Dict[str, Any]:
        """Serialize to the store ingest contract (§8 / §10 sidecar shape)."""
        return {
            "fingerprint": self.fingerprint(),
            "kind": self.kind,
            "spatial_key": self.spatial_key,
            "recipe_id": self.recipe.recipe_id(),
            "recipe": {
                "kind": self.recipe.kind,
                "params": self.recipe.params,
                "code_version": self.recipe.code_version,
                "git_sha": self.recipe.git_sha,
            },
            "inputs": list(self.inputs),
            "location": self.location,
            "state": self.state,
            "bytes": self.bytes,
            "wall_time_s": self.wall_time_s,
            "produced_by": self.produced_by,
        }


# ---------------------------------------------------------------------------
# Spatial-key helpers (§6). Plain dicts, stable field names.
# ---------------------------------------------------------------------------
def skycell_spatial_key(projection: Any, skycell: Any) -> Dict[str, Any]:
    """Spatial key for skycell-scoped kinds (raw/combined/convolved/source)."""
    return {"projection": projection, "skycell": skycell}


def scc_spatial_key(sector: int, camera: int, ccd: int, oversampling: int) -> Dict[str, Any]:
    """Spatial key for oversampling-scoped SCC kinds (mapping/scc_assembly/template)."""
    return {
        "sector": int(sector),
        "camera": int(camera),
        "ccd": int(ccd),
        "oversampling": int(oversampling),
    }


def scc_no_os_spatial_key(sector: int, camera: int, ccd: int) -> Dict[str, Any]:
    """Spatial key for SCC kinds with no oversampling axis (ffi_set/wcs_group)."""
    return {"sector": int(sector), "camera": int(camera), "ccd": int(ccd)}


# ---------------------------------------------------------------------------
# recipe_params builders — one per kind (§6). Each takes a ``resolved``
# (ResolvedTargetConfig) and returns a JSON-serializable param dict. Builders
# read only attributes confirmed to exist in
# ``template_creation/orchestration/stage_params.py`` / ``runner_config.py``.
# ---------------------------------------------------------------------------
def _mapping_stage(resolved):
    return resolved.stages.mapping


def _ps1_process_stage(resolved):
    return resolved.stages.ps1_process


def _templates_stage(resolved):
    # ``.templates`` is a property alias for the ``downsample`` stage params.
    return resolved.stages.templates


def _wcs_grouping_stage(resolved):
    return resolved.stages.wcs_grouping


# --- Fully derivable builders ---------------------------------------------
def mapping_params(resolved, **extra) -> Dict[str, Any]:
    """mapping recipe params (§6): oversampling_factor, pad_distance, tess_buffer."""
    mp = _mapping_stage(resolved)
    return {
        "oversampling_factor": mp.oversampling_factor,
        "pad_distance": mp.pad_distance,
        "tess_buffer": mp.tess_buffer,
    }


def wcs_group_params(resolved, **extra) -> Dict[str, Any]:
    """wcs_group recipe params (§6): offset_threshold, savgol window/order, crop_mode."""
    wg = _wcs_grouping_stage(resolved)
    return {
        "offset_threshold": wg.offset_threshold,
        "wcs_drift_savgol_window": wg.wcs_drift_savgol_window,
        "wcs_drift_savgol_polyorder": wg.wcs_drift_savgol_polyorder,
        "crop_mode": wg.crop_mode,
    }


def combined_skycell_params(resolved, *, gaia_version: Any = None, **extra) -> Dict[str, Any]:
    """combined_skycell recipe params (§6).

    Star-removal is folded into the combined recipe (locked decision #10): the
    Gaia catalog version + mag threshold + saturation flags fully determine the
    combined bytes for a footprint. ``gaia_version`` is a passthrough because it
    is not a resolved stage attribute today (see catalog source, TODO).

    No band-combine constants are configurable in the current stage params, so
    none are emitted.
    """
    pp = _ps1_process_stage(resolved)
    return {
        "enable_saturation_correction": pp.enable_saturation_correction,
        "remove_saturated_stars": pp.remove_saturated_stars,
        "bright_star_mag_threshold": pp.bright_star_mag_threshold,
        "gaia_version": gaia_version,
    }


def convolved_skycell_params(
    resolved,
    *,
    radius: int = CONVOLVE_RADIUS_DEFAULT,
    mode: str = CONVOLVE_MODE_DEFAULT,
    **extra,
) -> Dict[str, Any]:
    """convolved_skycell recipe params (§6): psf_sigma, radius, mode.

    ``psf_sigma`` is the only resolved knob; ``radius``/``mode`` are the
    convolution-utility defaults (constant, mode="constant") the producer uses,
    exposed as overridable extras so identity stays honest if they ever change.
    """
    pp = _ps1_process_stage(resolved)
    return {
        "psf_sigma": pp.psf_sigma,
        "radius": radius,
        "mode": mode,
    }


def template_params(resolved, **extra) -> Dict[str, Any]:
    """template recipe params (§6): oversampling_factor, single_offset,
    ignore_mask_bits, geometry_mode."""
    ds = _templates_stage(resolved)
    return {
        "oversampling_factor": ds.oversampling_factor,
        "single_offset": ds.single_offset,
        "ignore_mask_bits": list(ds.ignore_mask_bits),
        "geometry_mode": ds.geometry_mode,
    }


# --- Stub builders (params not cleanly derivable from ``resolved`` yet) -----
def ffi_set_params(resolved, **extra) -> Dict[str, Any]:
    """ffi_set recipe params (§6): download source/params.

    TODO: FFI download provenance (source archive, cadence, calibration level)
    is not represented in ResolvedTargetConfig today. Minimal placeholder keyed
    only on the SCC target so identity is stable; extend when the download stage
    records its source params.
    """
    t = resolved.target
    return {"sector": int(t.sector), "camera": int(t.camera), "ccd": int(t.ccd)}


def raw_skycell_params(resolved, *, version_token: Any = None, **extra) -> Dict[str, Any]:
    """raw_skycell recipe params (§6): ``{}`` + version_token (size,mtime,batch).

    Raw cells have no producer recipe of our own; identity is the downloaded
    bytes' version token (passthrough). TODO: populate version_token from the
    input_files table once the raw ingest records it.
    """
    return {"version_token": version_token}


def source_catalog_params(resolved, *, gaia_version: Any = None, **extra) -> Dict[str, Any]:
    """source_catalog recipe params (§6): gaia query params, gaia_version.

    TODO: Gaia query params (radius, mag limits, epoch) are not in
    ResolvedTargetConfig yet. Passthrough gaia_version so the footprint catalog
    re-fingerprints when the catalog release changes.
    """
    return {"gaia_version": gaia_version}


def scc_assembly_params(resolved, **extra) -> Dict[str, Any]:
    """scc_assembly recipe params (§6): seam-pad params (PAD_SIZE, edge exclusion).

    The assembly stitches N convolved cells onto the SCC grid; its identity is
    the seam handling. ``edge_exclusion`` is a confirmed mapping-stage attribute;
    PAD_SIZE is a producer constant not yet surfaced in stage params.
    TODO: surface PAD_SIZE / edge-buffer knobs when the assembly stage exposes
    them; today only edge_exclusion is derivable.
    """
    mp = _mapping_stage(resolved)
    return {"edge_exclusion": mp.edge_exclusion}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_PARAM_BUILDERS = {
    FFI_SET: ffi_set_params,
    RAW_SKYCELL: raw_skycell_params,
    SOURCE_CATALOG: source_catalog_params,
    MAPPING: mapping_params,
    WCS_GROUP: wcs_group_params,
    COMBINED_SKYCELL: combined_skycell_params,
    CONVOLVED_SKYCELL: convolved_skycell_params,
    SCC_ASSEMBLY: scc_assembly_params,
    TEMPLATE: template_params,
}


def recipe_params(kind: str, resolved, **extra) -> Dict[str, Any]:
    """Return the recipe param dict for ``kind`` built from ``resolved``."""
    if kind not in _PARAM_BUILDERS:
        raise ValueError(f"unknown artifact kind: {kind!r}")
    return _PARAM_BUILDERS[kind](resolved, **extra)


def build_recipe(kind: str, resolved, **extra) -> Recipe:
    """Build a :class:`Recipe` for ``kind`` with current code/git version."""
    from syndiff_pipeline.common.provenance.fingerprint import code_version, git_sha

    params = recipe_params(kind, resolved, **extra)
    return Recipe(
        kind=kind,
        params=params,
        code_version=code_version(),
        git_sha=git_sha(),
    )
