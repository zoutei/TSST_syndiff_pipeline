"""
config.py
=========
SynDiff pipeline configuration dataclass with YAML I/O and CLI argument parsing.

Usage:
    from syndiff_pipeline.config import SynDiffConfig, load_config, save_config

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
    Gaia stars.  If null (default), ``wcs_grouping`` picks the first valid-WCS
    frame and records it in ``output_dir/cluster_template_job.json`` (or legacy
    ``ref_ffi_path.txt``)."""

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

    crop_quadrant: str = "tr"
    """When **none** of ``x_min``/``x_max``/``y_min``/``y_max`` are set: ``'tl'`` | ``'tr'`` | ``'bl'`` | ``'br'`` subdivide the usable area (dead strips removed) using chip midlines ``nx//2``, ``ny//2``; ``'full'`` uses the whole usable rectangle."""

    x_left_dead: int = 44
    """Dead columns on the left edge of the FFI (usable x starts here)."""

    x_right_dead: int = 44
    """Dead columns on the right edge (usable x ends before ``nx - x_right_dead``)."""

    y_edge_strip: int = 30
    """Dead rows along the **top** of the FFI only; usable y is ``[0, ny - y_edge_strip)``."""

    # ── Template grouping ─────────────────────────────────────────────────────
    offset_threshold: float = 0.01
    """Maximum pixel offset (TESS pixels) before a new template group is needed."""

    wcs_drift_savgol_window: Optional[int] = 11
    """If not ``None`` and ≥ 3: apply a Savitzky–Golay filter along time-ordered
    valid frames to ``delta_x``/``delta_y`` before template grouping (must be odd;
    even values are bumped up). Set to ``None`` to disable."""

    wcs_drift_savgol_polyorder: int = 2
    """Polynomial order for :func:`wcs_grouping.smooth_wcs_drift_savgol`
    (must be ``< wcs_drift_savgol_window``)."""

    # ── Hotpants ──────────────────────────────────────────────────────────────
    sci_fwhm: float = 1.0
    """Science image FWHM in native TESS pixels. Drives kernel/substamp widths."""

    hp_ko: int = 2
    hp_bgo: int = 3
    hp_nstampx: int = 10
    hp_nstampy: int = 10
    hp_nss: int = 100
    hp_ngauss: int = 3
    hp_deg_fixe: list = field(default_factory=lambda: [6, 4, 2])
    hp_fitthresh: float = 5.0
    hp_stat_sig: float = 3.0
    hp_kf_spread_mask1: float = 0.0
    hp_ks: float = 3.0
    hp_kfm: float = 0.75
    hp_force_convolve: str = "t"
    hp_normalize: str = "i"

    # ── Masking ───────────────────────────────────────────────────────────────
    gaia_mag_bright: float = 13.0
    """Mask all Gaia stars brighter than this magnitude (TESSreduce Cat_mask)."""

    ref_mag_min: float = 13.5
    """Minimum tess_mag for hotpants reference stars."""

    ref_mag_max: float = 14.5
    """Maximum tess_mag for hotpants reference stars."""

    ref_isolation_mag: float = 13.5
    """Reject a reference-star candidate if any star brighter than this falls
    within ref_isolation_px of it."""

    ref_isolation_px: int = 8
    """Isolation radius in pixels (see ref_isolation_mag)."""

    ref_separation_px: int = 10
    """Minimum pixel separation between any two selected reference stars."""

    strapsize: int = 6
    """Width (pixels) of the strap mask kernel (Strap_mask size parameter)."""

    # ── TGLC ePSF ─────────────────────────────────────────────────────────────
    tile_nx: int = 4
    """Number of tiles along the x axis for TGLC ePSF fitting."""

    tile_ny: int = 4
    """Number of tiles along the y axis."""

    epsf_oversample: int = 2
    """ePSF oversampling factor.  over_size = 2 * psf_size + 1 (e.g. 23 for psf_size=11)."""

    psf_size: int = 11
    """Half-size of the ePSF stamp in native pixels (before oversampling).
    The saturated-star template is also built at this oversampling."""

    # ── Saturated-star template (high-res canvas) ─────────────────────────────
    high_res_os: int = 9
    """Oversampling for the high-resolution saturated-star template canvas in
    ``sat_template`` (block-sum downsampled to native pixels for storage)."""

    # ── Background temporal (AdaptiveBackground) & spatial (TESSreduce-like) ───
    bkg_vector_path: Optional[str] = None
    """Directory containing TESSVectors CSV (``TessVectors_SXXX_CY_FFI.csv``).
    If unset, files are downloaded from HEASARC when background smoothing runs."""

    bkg_adaptive_method: str = "savgol"
    """Temporal smooth on the rough background cube: ``\"savgol\"`` (Savitzky–Golay
    along time; default, matches upstream TESSreduce) or ``\"adaptive\"``
    (adaptive temporal median / ``adaptive_medfilt_3d``)."""

    bkg_adaptive_savgol_window: Optional[int] = None
    """Savitzky–Golay window length (odd frames) when ``bkg_adaptive_method`` is ``\"savgol\"``.
    ``None`` lets vendored :class:`adaptive_background.AdaptiveBackground` choose a
    cadence-based window (~6 h), matching upstream TESSreduce defaults."""

    bkg_adaptive_savgol_polyorder: int = 2
    """Savitzky–Golay polynomial order when ``bkg_adaptive_method`` is ``\"savgol\"``."""

    bkg_adaptive_w_min: int = 3
    """Minimum odd temporal window (frames) for adaptive background median filter."""

    bkg_adaptive_w_max: int = 51
    """Maximum odd temporal window (frames) for adaptive background median filter."""

    bkg_adaptive_block_size: int = 5
    """Spatial block size inside the adaptive background smoother (TESSreduce default)."""

    bkg_r1_recombine_hotpants: bool = False
    """If True, round-1 rough background uses ``Smooth_bkg(diff + hotpants_bkg)``
    before adding ``hotpants_bkg``; if False, uses ``Smooth_bkg(diff)`` (Hotpants
    background already removed in the diff FITS). Set True when the diff is in a
    domain where re-adding the polynomial bkg before ``Smooth_bkg`` matches your
    Hotpants convention."""

    bkg_tessreduce_spatial_pipeline: bool = True
    """If True, :func:`background.background_loop` stacks Hotpants frames and runs the
    spatial portion of ``TESSreduce.tessreduce.background()`` (``Smooth_bkg`` passes,
    optional strap QE, ``fix_background_anomalies``). The rough workspace then holds
    that cube; ``background_adaptive`` / ``background_estimate`` apply only
    :class:`adaptive_background.AdaptiveBackground` temporal Savitzky–Golay (no second
    Hotpants polynomial add). Incompatible with ``stream_load_rough``."""

    bkg_gauss_smooth: float = 2.0
    """Gaussian sigma for the refined ``Smooth_bkg`` pass when ``rerun_negative`` is True
    (TESSreduce ``bkg_gauss_sigma`` default)."""

    bkg_calc_qe: bool = True
    """Strap QE multiplication on the background cube (TESSreduce ``calc_qe``)."""

    bkg_strap_iso: bool = True
    """If True, valid sky pixels require ``mask == 0`` (TESSreduce ``strap_iso``)."""

    bkg_source_hunt: bool = False
    """PSF-based source hunt per frame; requires PRF kernel + residual cube wiring (not
    yet exposed from YAML — keep False unless extended)."""

    bkg_interpolate: bool = True
    """Passed to vendored ``Smooth_bkg`` as ``interpolate`` (griddata vs inpaint)."""

    bkg_rerun_negative: bool = False
    """TESSreduce ``rerun_negative`` oversubtraction mask refinement."""

    bkg_rerun_diff: bool = False
    """TESSreduce ``rerun_diff`` residual mask refinement."""

    bkg_use_error_image: bool = False
    """Use per-pixel flux errors for ``rerun_negative`` threshold (needs ``eflux`` cube)."""

    # ── Photometry ────────────────────────────────────────────────────────────
    psf_type: str = "epsf"
    """'epsf' — use the fitted empirical ePSF (EpsfLocator).
    'prf'  — use the official TESS PRF (TESS_PRF from the PRF package)."""

    phot_cutout_size: int = 15
    """Side length (native pixels) of the photometry cutout stamp."""

    phot_bkg_poly_order: int = 3
    """Polynomial order for the local background surface fit in create_psf.psf_flux."""

    phot_snap: str = "brightest"
    """Position-fit strategy: 'brightest' | 'ref' | 'fixed' (TESSreduce-compatible; see photometry)."""

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
    """Number of parallel workers (joblib **loky**) for :func:`photometry.run_forced_photometry`
    (cutout load + per-epoch ``psf_flux`` when ``n_jobs`` > 1),
    :func:`background.background_loop` (per-frame rough ``Smooth_bkg`` when ``n_jobs`` > 1),
    and sub-tasks inside :func:`adaptive_background.adaptive_medfilt_3d` when
    ``bkg_adaptive_method`` is ``\"adaptive\"`` (via ``cfg.n_jobs`` passed from
    :func:`background.adaptive_smooth_background`). For Hotpants only, see
    ``hotpants_n_jobs`` (defaults to this value when unset). ePSF fitting remains serial over frames."""

    hotpants_n_jobs: Optional[int] = None
    """If set, overrides ``n_jobs`` for :func:`hotpants_runner.hotpants_loop` only (per-frame
    differencing). Use ``1`` when peak RAM is tight: each worker holds a full ``HotpantsState``
    for the crop. When unset, ``n_jobs`` is used."""

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
        "bkg_vector_path",
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
        "bkg_vector_path",
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
        "--hotpants-n-jobs",
        type=int,
        default=None,
        dest="hotpants_n_jobs",
        help="Override n_jobs for the hotpants stage only (default: same as n_jobs).",
    )
    parser.add_argument(
        "--max-ffis", type=int, default=None, dest="max_ffis",
        help="Cap number of FFIs (after glob sort); for quick tests.",
    )
    parser.add_argument("--psf-type",   type=str,   default=None, dest="psf_type",
                        choices=["epsf", "prf"])
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
        "sector", "camera", "ccd", "output_dir", "ffi_dir", "n_jobs", "hotpants_n_jobs", "max_ffis",
        "psf_type", "pipeline_plots",
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
