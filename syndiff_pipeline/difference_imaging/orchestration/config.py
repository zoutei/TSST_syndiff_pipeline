"""
config.py
=========
SynDiff pipeline configuration dataclass with YAML I/O and CLI argument parsing.

Usage:
    from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig, load_config, save_config

    cfg = load_config("config.yaml")
    save_config(cfg, "config_out.yaml")
"""

import argparse
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)


def discover_template_paths(template_dir: str) -> dict:
    """
    Build ``{group_id: abs_path}`` for PS1 template FITS under *template_dir*.

    Expects subdirectories named ``group_<id>`` each containing
    ``ps1_template.fits`` (or ``template.fits``).
    """
    root = Path(template_dir)
    if not root.is_dir():
        return {}
    out: dict[int, str] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if not name.startswith("group_"):
            continue
        try:
            gid = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        for fname in ("ps1_template.fits", "template.fits"):
            p = sub / fname
            if p.is_file():
                out[gid] = str(p.resolve())
                break
    if out:
        log.info(
            "Discovered %d template FITS under %s (group ids: %s)",
            len(out),
            root,
            sorted(out.keys()),
        )
    return out


@dataclass
class SynDiffConfig:
    """All parameters that drive the SynDiff pipeline."""

    # ── Required paths ────────────────────────────────────────────────────────
    ffi_dir: str = ""
    """Root ``tess_ffi`` directory. FFIs are read from
    ``{ffi_dir}/s{{sector:04d}}/cam{{camera}}_ccd{{ccd}}/`` (see ``download.nested_ffi_dir``)."""

    output_dir: str = ""
    """Root directory for all pipeline outputs."""

    manifest: str = ""
    """Optional absolute path to the per-FFI manifest CSV. If empty, uses
    ``syndiff_ffi_frames.csv`` under ``output_dir`` (see ``paths.DEFAULT_MANIFEST_BASENAME``)."""

    pipeline: list = field(default_factory=list)
    """Ordered list of stage dicts (``kind`` + fields). Required; :func:`run_pipeline`
    executes these stages in order."""

    gaia_catalog: str = ""
    """Path to the crop-local Gaia CSV produced with the PS1 template job
    (e.g. ``unique_gaia_stars_for_cropped_template.csv`` / ``ps1_sat_stars_gaia_catalog.csv``):
    columns include ``x``, ``y`` and photometry. Required for ``shared_mask`` and ePSF stages."""

    removed_stars_csv: str = ""
    """Optional PS1 ``removed_stars`` CSV. Used only when building sat templates
    if the loaded Gaia catalog lacks crop-local ``x``/``y`` (normally satisfied
    by ``gaia_catalog``)."""

    median_mask_path: str = ""
    """Path to TGLC ``median_mask.fits`` (bad-pixel / background mask)."""

    straps_csv: str = ""
    """CSV listing detector strap columns (TESS camera/CCD layout)."""

    # ── Template paths: filled by user after wcs_grouping ────────────────────
    template_paths: dict = field(default_factory=dict)
    """Mapping group_id → absolute path of the PS1 template FITS for that group.
    Example: {0: '/path/to/ps1_template_group0.fits', 1: '/path/...'}
    Leave empty until ``cluster_template_job.json`` from ``wcs_grouping`` exists
    and PS1 templates are available (or set ``template_dir`` for discovery)."""

    template_dir: str = ""
    """If set and ``template_paths`` is empty, fill ``template_paths`` by scanning
    subfolders ``group_<id>/ps1_template.fits`` (or ``template.fits``) under this directory."""

    # ── Optional reference FFI ────────────────────────────────────────────────
    ref_ffi_path: Optional[str] = None
    """Absolute path of the FFI to use as WCS reference for pixel-projecting
    Gaia stars.  If null (default), ``wcs_grouping`` chooses a frame using
    TESSVectors Earth/Moon angle cuts (see ``ref_ffi_min_*``), raw–smooth drift
    agreement when Savitzky–Golay smoothing ran, and proximity to the median
    smoothed drift; the path is recorded in ``output_dir/cluster_template_job.json``.
    Uses optional ``bkg_vector_path`` on the ``wcs_grouping`` pipeline stage for local
    TESSVectors CSV when set (else HEASARC)."""

    ref_ffi_min_earth_deg: float = 45.0
    """Minimum Earth–camera angle (degrees) for automatic reference FFI selection."""

    ref_ffi_min_moon_deg: float = 25.0
    """Minimum Moon–camera angle (degrees) for automatic reference FFI selection."""

    ref_ffi_max_smoothed_residual: float = 0.05
    """When ``delta_x_raw``/``delta_y_raw`` exist, automatic reference selection
    prefers frames with hypot(raw−smooth) at most this many pixels."""

    # ── Target ────────────────────────────────────────────────────────────────
    target_ra: Optional[float] = None
    """RA (deg, J2000) of the science target for light-curve extraction."""

    target_dec: Optional[float] = None
    """Dec (deg, J2000) of the science target."""

    additional_forced_targets: list = field(default_factory=list)
    """Extra entries ``{ra, dec, name}`` (degrees, J2000; ``name`` is a short label).
    Each produces ``ws/<forced_photometry output>/lightcurve_<name>.csv`` (primary
    stays ``lightcurve.csv``). WCS grouping and templates use ``target_ra`` /
    ``target_dec`` only."""

    # ── Instrument ────────────────────────────────────────────────────────────
    sector: int = 20
    camera: int = 3
    ccd: int = 3

    # ── Crop region ──────────────────────────────────────────────────────────
    x_min: Optional[int] = None
    """Left column of the crop (inclusive). If **any** of ``x_min``/``x_max``/``y_min``/``y_max`` is non-null, **explicit** crop: unset edges are filled from usable bounds (``x_left_dead``, ``x_right_dead``, ``y_edge_strip``); values are clamped to the FFI."""

    x_max: Optional[int] = None
    """Right column of the crop, exclusive. Same explicit-mode rules as ``x_min``."""

    y_min: Optional[int] = None
    """Bottom row of the crop (inclusive). Same explicit-mode rules as ``x_min``."""

    y_max: Optional[int] = None
    """Top row of the crop, exclusive. Same explicit-mode rules as ``x_min``."""

    crop_quadrant: str = "full"
    """When **none** of ``x_min``/``x_max``/``y_min``/``y_max`` are set: ``'tl'`` | ``'tr'`` | ``'bl'`` | ``'br'`` subdivide the usable area (dead strips removed) using chip midlines ``nx//2``, ``ny//2``; ``'full'`` uses the entire FFI array including dead columns/rows; ``'full_science'`` uses the usable rectangle only."""

    x_left_dead: int = 44
    """Dead columns on the left edge of the FFI (usable x starts here)."""

    x_right_dead: int = 44
    """Dead columns on the right edge (usable x ends before ``nx - x_right_dead``)."""

    y_edge_strip: int = 30
    """Dead rows along the **top** of the FFI only; usable y is ``[0, ny - y_edge_strip)``."""

    # ── Diagnostics & workspace ───────────────────────────────────────────────
    pipeline_plots: bool = False
    """If True, write diagnostic figures: after ``wcs_grouping``,
    ``wcs_drift_template_debug.png`` and ``lightcurve_<stage>.png`` under
    ``{output_dir}/{pipeline_plots_dir}/`` by default; adaptive-background GIF
    (when hooked up) in the same folder. Forced-photometry light-curve PNG titles
    include the stage ``output`` workspace label. CSVs live under ``ws/<output>/``
    (``lightcurve.csv`` for the primary, ``lightcurve_<name>.csv`` for each
    ``additional_forced_targets`` entry)."""

    pipeline_plot_dpi: int = 150
    """Resolution for PNGs written when ``pipeline_plots`` is True."""

    pipeline_plots_dir: str = "debug_plots"
    """Subdirectory of ``output_dir`` for WCS and light-curve diagnostic PNGs when
    ``pipeline_plots`` is True. If empty, diagnostics are written directly under
    ``output_dir`` (not recommended when using workspace subdirs for data)."""

    pipeline_external_workspace_labels: Optional[list[Any]] = None
    """Hotpants / workspace labels already populated under ``output_dir/ws/<label>/``
    from a previous run. Added to the dependency graph during validation so you can
    omit slow stages (e.g. re-run ``forced_photometry`` only). When ``wcs_grouping``
    is absent from ``pipeline:``, ``run_config_pipeline`` reloads the frame manifest
    and ``cluster_template_job.json`` from ``output_dir``."""

    master_fits_mirror: bool = True
    """If True (default), every per-FFI FITS under ``ws/<label>/`` is also exposed via a
    relative symlink directly under ``master/`` (flat layout), refreshed after each
    pipeline stage. Disable for read-only filesystems where symlink creation is unsupported."""

    # ── Parallelism ───────────────────────────────────────────────────────────
    n_jobs: int = 8
    """Default worker count (joblib **loky**) for stages that read this global value:
    forced photometry, background stacking / adaptive temporal smoothing, etc.
    Per-stage overrides (e.g. ``hotpants_n_jobs`` on ``kind: hotpants``) win."""

    max_ffis: Optional[int] = None
    """If set (positive int), use at most this many FFIs after **time sort**, skipping any
    file whose WCS cannot place ``target_ra``/``target_dec`` until enough valid frames are
    found. Useful for smoke tests without moving files. ``None`` means use every FFI on disk
    (invalid WCS rows remain in the table with ``wcs_ok=False``)."""


# ── YAML I/O ─────────────────────────────────────────────────────────────────


def _sanitize_forced_lightcurve_name(name: str) -> str:
    """Map user ``name`` to a safe fragment for ``lightcurve_<name>.csv`` / PNG."""
    s = str(name).strip()
    if not s:
        raise ValueError("additional_forced_targets entry 'name' must be non-empty")
    if os.sep in s or (os.altsep and os.altsep in s) or "/" in s or "\\" in s:
        raise ValueError("additional_forced_targets 'name' must not contain path separators")
    out = re.sub(r"[^0-9A-Za-z._+-]+", "_", s)
    out = re.sub(r"_+", "_", out).strip("_")
    if not out or out in (".", ".."):
        raise ValueError(f"invalid light curve name after sanitization: {name!r}")
    return out


def normalize_additional_forced_targets(raw: Any) -> List[Dict[str, Any]]:
    """
    Parse ``additional_forced_targets`` from YAML into a list of
    ``{"ra": float, "dec": float, "name": str}`` dicts.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            "additional_forced_targets must be a list of mappings with keys "
            "'ra', 'dec', and 'name'"
        )
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"additional_forced_targets[{i}] must be a mapping, got {type(item).__name__}"
            )
        if "ra" not in item or "dec" not in item:
            raise ValueError(
                f"additional_forced_targets[{i}] must include 'ra' and 'dec' (degrees)"
            )
        if "name" not in item or item["name"] is None or not str(item["name"]).strip():
            raise ValueError(
                f"additional_forced_targets[{i}]: non-empty 'name' is required "
                "(used for lightcurve_<name>.csv and debug PNG)"
            )
        try:
            ra = float(item["ra"])
            dec = float(item["dec"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"additional_forced_targets[{i}]: ra/dec must be numeric"
            ) from e
        sname = _sanitize_forced_lightcurve_name(str(item["name"]))
        if sname in seen:
            raise ValueError(
                f"duplicate additional_forced_targets name {sname!r}; names must be unique"
            )
        seen.add(sname)
        out.append({"ra": ra, "dec": dec, "name": sname})
    return out


def _cfg_to_dict(cfg: SynDiffConfig) -> dict:
    d = asdict(cfg)
    # Convert None to null-friendly representation (yaml.dump handles None as null)
    return d


def _resolve_config_path(value: Optional[str], base: Path) -> Optional[str]:
    """Resolve a single path: absolute paths unchanged; relative paths → base / path."""
    if value is None or value == "":
        return value
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((base / p).resolve())


def load_config(yaml_path: str) -> SynDiffConfig:
    """
    Load a SynDiffConfig from a YAML file.

    Unknown keys are ignored (forward-compatibility). Missing keys use the
    dataclass defaults.

    All string paths (``ffi_dir``, ``output_dir``, catalog paths, templates, etc.)
    that are relative in YAML are resolved against the **directory containing
    the config file**, not the process working directory.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.

    Returns
    -------
    SynDiffConfig
    """
    yaml_path = str(Path(yaml_path).expanduser())
    base = Path(yaml_path).resolve().parent

    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    # Filter to only known fields
    known = {f.name for f in SynDiffConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known}

    if not filtered.get("gaia_catalog") and raw.get("unique_gaia_catalog"):
        filtered["gaia_catalog"] = raw["unique_gaia_catalog"]
        log.warning(
            "Config key 'unique_gaia_catalog' is deprecated; use 'gaia_catalog'."
        )

    unknown = set(raw) - known
    if unknown:
        log.warning(f"Ignoring unknown config keys: {sorted(unknown)}")

    if "pipeline" in raw and raw["pipeline"] is not None:
        if not isinstance(raw["pipeline"], list):
            raise ValueError("pipeline must be a list of stage dicts")
        filtered["pipeline"] = raw["pipeline"]

    # YAML null for path strings → omit so dataclass defaults (e.g. "") apply.
    for key in (
        "ffi_dir",
        "output_dir",
        "gaia_catalog",
        "removed_stars_csv",
        "median_mask_path",
        "straps_csv",
        "template_dir",
        "manifest",
    ):
        if filtered.get(key) is None:
            filtered.pop(key, None)

    for key in (
        "ffi_dir",
        "output_dir",
        "gaia_catalog",
        "removed_stars_csv",
        "median_mask_path",
        "straps_csv",
        "ref_ffi_path",
        "template_dir",
        "manifest",
    ):
        if key in filtered and filtered[key] is not None:
            filtered[key] = _resolve_config_path(str(filtered[key]), base)

    # template_paths keys may be loaded as strings if they came from YAML
    if "template_paths" in filtered and isinstance(filtered["template_paths"], dict):
        filtered["template_paths"] = {
            int(k): _resolve_config_path(str(v), base)
            for k, v in filtered["template_paths"].items()
            if v is not None
        }

    cfg = SynDiffConfig(**filtered)
    if cfg.template_dir and not cfg.template_paths:
        cfg.template_paths = discover_template_paths(cfg.template_dir)
    cfg.additional_forced_targets = normalize_additional_forced_targets(
        cfg.additional_forced_targets
    )
    # Resolve relative bkg_vector_path on pipeline stages (same base as the YAML file).
    for st in cfg.pipeline or []:
        if isinstance(st, dict) and st.get("bkg_vector_path"):
            st["bkg_vector_path"] = _resolve_config_path(
                str(st["bkg_vector_path"]), base
            )
    return cfg


def save_config(cfg: SynDiffConfig, yaml_path: str) -> None:
    """
    Write a SynDiffConfig to a YAML file.

    Parameters
    ----------
    cfg : SynDiffConfig
    yaml_path : str
    """
    os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(_cfg_to_dict(cfg), fh, default_flow_style=False, sort_keys=False)
    log.info(f"Config saved to {yaml_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --config and per-field overrides to an argparse parser."""
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file."
    )
    # Allow a subset of high-frequency overrides on the CLI
    parser.add_argument("--sector",     type=int,   default=None)
    parser.add_argument("--camera",     type=int,   default=None)
    parser.add_argument("--ccd",        type=int,   default=None)
    parser.add_argument("--output-dir", type=str,   default=None, dest="output_dir")
    parser.add_argument("--ffi-dir",    type=str,   default=None, dest="ffi_dir")
    parser.add_argument("--n-jobs",     type=int,   default=None, dest="n_jobs")
    parser.add_argument(
        "--max-ffis", type=int, default=None, dest="max_ffis",
        help="Cap number of FFIs (after glob sort); for quick tests.",
    )
    parser.add_argument(
        "--pipeline-plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="pipeline_plots",
        help="Write pipeline diagnostic PNGs (overrides YAML).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SynDiffConfig:
    """
    Build a SynDiffConfig from parsed CLI args.
    YAML file is the base; explicit CLI overrides win.
    """
    cfg = load_config(args.config)
    for attr in (
        "sector", "camera", "ccd", "output_dir", "ffi_dir", "n_jobs", "max_ffis",
        "pipeline_plots",
    ):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)
    return cfg


if __name__ == "__main__":
    # Quick self-test: print default config as YAML
    import sys
    cfg = SynDiffConfig()
    yaml.dump(_cfg_to_dict(cfg), sys.stdout, default_flow_style=False, sort_keys=False)
