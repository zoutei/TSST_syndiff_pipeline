"""
epsf_fitting.py
===============
Fit empirical PSFs (ePSF) on difference images with photutils, tiling each
frame into tile_nx × tile_ny sub-regions and building a per-frame
:class:`~photutils.psf.GriddedPSFModel`.

Persists per-frame ``*_gridded_epsf.npz`` archives, legacy smooth stacks for
downstream stages, and per-template-group median ePSFs.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from syndiff_pipeline.difference_imaging.support.ffi_naming import parse_workspace_frame_stem, strip_fits_suffix, tess_product_id_from_ffi_path

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ── ePSF stack bundle (stack + ffi_stem per axis-0 row) ───────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def epsf_stack_bundle_base(output_dir: str, round_id: int) -> str:
    """Base path without extension for ``epsf_stack_r{round_id}.npz``."""
    return os.path.join(output_dir, f"epsf_stack_r{round_id}")


def save_epsf_stack_bundle(
    stack: np.ndarray,
    ffi_stems: list,
    output_dir: str,
    round_id: int,
) -> str:
    """Save epsf stack bundle.
    
    Parameters
    ----------
    stack : np.ndarray
    ffi_stems : list
    output_dir : str
    round_id : int
    
    Returns
    -------
    str"""
    os.makedirs(output_dir, exist_ok=True)
    path = epsf_stack_bundle_base(output_dir, round_id) + ".npz"
    np.savez_compressed(
        path,
        stack=np.asarray(stack),
        ffi_stem=np.asarray(ffi_stems, dtype=object),
    )
    log.info("  ePSF stack saved to %s  shape=%s", path, stack.shape)
    return path


def load_epsf_stack_bundle(output_dir: str, round_id: int) -> tuple:
    """
    Load round ``round_id`` ePSF stack from ``epsf_stack_r{round_id}.npz``.

    Returns
    -------
    stack : ndarray, shape (n_frames, n_tiles, n_pix)
    ffi_stem : list of str
    """
    base = epsf_stack_bundle_base(output_dir, round_id)
    npz_p = base + ".npz"
    if not os.path.isfile(npz_p):
        raise FileNotFoundError(f"No ePSF stack at {npz_p}")
    z = np.load(npz_p, allow_pickle=True)
    try:
        stack = np.asarray(z["stack"])
        if "ffi_stem" not in z.files:
            raise ValueError(f"{npz_p!r} missing required array 'ffi_stem'")
        raw = z["ffi_stem"]
        ffi_stem = [str(x) for x in raw.tolist()]
    finally:
        z.close()
    return stack, ffi_stem


# ═══════════════════════════════════════════════════════════════════════════════
# ── Smoothed / repaired ePSF stack (saved alongside fitting workspace) ────────
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_epsf_stack(epsf_stack: np.ndarray) -> np.ndarray:
    """
    Per-frame fitted ePSFs with all-NaN tiles repaired (global median ePSF per pixel).

    No temporal filtering — aligns with using raw fits before downstream grouping.

    Parameters
    ----------
    epsf_stack : ndarray, shape (n_frames, n_tiles, over_size²)
    """
    n_frames, n_tiles, n_pix = epsf_stack.shape
    out = np.array(epsf_stack, copy=True)

    global_med = np.nanmedian(out.reshape(-1, n_pix), axis=0)
    n_nan_tiles = 0
    for t in range(n_tiles):
        if np.isnan(out[:, t, :]).all():
            out[:, t, :] = global_med
            n_nan_tiles += 1

    if n_nan_tiles:
        log.warning(
            "  prepare_epsf_stack: %d tiles were all-NaN; filled with global median ePSF.",
            n_nan_tiles,
        )

    return out


def compute_group_epsf(
    epsf_smooth: np.ndarray,
    group_ids: np.ndarray,
    output_dir: str | None = None,
    group_subdir: str = "group_epsf",
) -> dict:
    """
    One median ePSF per template group (median across frames in that group).

    Parameters
    ----------
    epsf_smooth : ndarray (n_frames, n_tiles, over_size²)
    group_ids : ndarray (n_frames,)
    output_dir : str, optional — saves ``group_epsf_{gid}.npy`` under ``group_subdir``
    """
    group_ids = np.asarray(group_ids)
    if group_ids.shape[0] != epsf_smooth.shape[0]:
        raise ValueError(
            f"group_ids length {group_ids.shape[0]} != n_frames "
            f"{epsf_smooth.shape[0]}"
        )
    unique_groups = [g for g in sorted(set(group_ids.tolist())) if g >= 0]

    group_epsf: dict[int, np.ndarray] = {}
    for gid in unique_groups:
        frame_mask = group_ids == gid
        if frame_mask.sum() == 0:
            continue
        group_stack = epsf_smooth[frame_mask]
        group_epsf[gid] = np.nanmedian(group_stack, axis=0)
        log.info(
            "  Group %s: %d frames → ePSF shape %s",
            gid,
            int(frame_mask.sum()),
            group_epsf[gid].shape,
        )

    if output_dir:
        out_subdir = os.path.join(output_dir, group_subdir)
        os.makedirs(out_subdir, exist_ok=True)
        for gid, epsf in group_epsf.items():
            np.save(os.path.join(out_subdir, f"group_epsf_{gid}.npy"), epsf)
        log.info("  Group ePSFs saved to %s/", out_subdir)

    return group_epsf


def save_epsf_smooth(
    epsf_smooth: np.ndarray,
    output_dir: str,
    round_id: int,
    ffi_stem: np.ndarray | list,
) -> str:
    """Save stack to ``epsf_rN_smooth.npz`` with ``ffi_stem`` (one per axis-0 row)."""
    if ffi_stem is None:
        raise TypeError("ffi_stem is required for save_epsf_smooth")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"epsf_r{round_id}_smooth.npz")
    np.savez_compressed(
        path,
        stack=np.asarray(epsf_smooth),
        ffi_stem=np.asarray(ffi_stem, dtype=object),
    )
    log.info("Smoothed ePSF stack saved to %s", path)
    return path


def load_epsf_smooth_stems_only(output_dir: str, round_id: int) -> list | None:
    """Load only ``ffi_stem`` from ``epsf_r{round_id}_smooth.npz``, if present."""
    npz_p = os.path.join(output_dir, f"epsf_r{round_id}_smooth.npz")
    if not os.path.isfile(npz_p):
        return None
    z = np.load(npz_p, allow_pickle=True)
    try:
        if "ffi_stem" in z.files:
            return [str(x) for x in z["ffi_stem"].tolist()]
    finally:
        z.close()
    return None


def load_epsf_smooth(output_dir: str, round_id: int) -> tuple:
    """Load ``epsf_r{round_id}_smooth.npz``. Returns ``stack``, ``ffi_stem`` list."""
    npz_p = os.path.join(output_dir, f"epsf_r{round_id}_smooth.npz")
    if not os.path.isfile(npz_p):
        raise FileNotFoundError(f"No smoothed ePSF at {npz_p}")
    z = np.load(npz_p, allow_pickle=True)
    try:
        stack = np.asarray(z["stack"])
        if "ffi_stem" not in z.files:
            raise ValueError(f"{npz_p!r} missing required array 'ffi_stem'")
        ffi_stem = [str(x) for x in z["ffi_stem"].tolist()]
    finally:
        z.close()
    return stack, ffi_stem


# ═══════════════════════════════════════════════════════════════════════════════
# ── Per-frame gridded ePSF fitting (photutils) ────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Static-mask bits ignored for ePSF star rejection:
# BRIGHT_CAT | SAT_CROSS | FAINT_CAT (1|2|32) — all catalog star stamps incl. crosses.
# Any other set bit rejects (straps, edges, PS1, TNS, asteroids).
from syndiff_pipeline.difference_imaging.masking.bits import EPSF_IGNORE_BITS, epsf_reject_mask

EPSF_SHARED_MASK_BITS = EPSF_IGNORE_BITS  # legacy name; means "ignored" bits
EPSF_STATIC_MASK_BITS = EPSF_IGNORE_BITS


def _load_static_mask_2d(
    static_mask_path: str | None, shape: tuple[int, int]
) -> np.ndarray | None:
    """
    Load boolean reject mask for ePSF from a static mask FITS (fallback path).

    Prefer :func:`epsf_reject_mask_at` with a ``MaskCatalog`` so asteroid bit
    128 is included per FFI. This path only sees the on-disk static layer.
    """
    if not static_mask_path or not os.path.isfile(static_mask_path):
        return None
    try:
        data = fits.getdata(static_mask_path)
        if data is None:
            return None
        mask = np.asarray(data)
        if mask.shape != shape:
            return None
        return epsf_reject_mask(mask)
    except Exception as exc:
        log.debug("static mask for ePSF not loaded: %s", exc)
        return None


# Backward-compatible alias
_load_shared_mask_2d = _load_static_mask_2d


def epsf_reject_mask_at(mask_catalog, time=None) -> np.ndarray:
    """Boolean ePSF reject mask for one epoch (in-memory ``mask_at``, no FITS I/O)."""
    return epsf_reject_mask(mask_catalog.mask_at(time, which="full"))


def btjd_by_stem_from_manifest(wcs_table) -> dict:
    """Map TESS product id / stem → BTJD from the frame manifest."""
    import pandas as pd

    from syndiff_pipeline.difference_imaging.support.ffi_naming import (
        tess_product_id_from_ffi_path,
    )

    out: dict = {}
    if wcs_table is None or len(wcs_table) == 0:
        return out
    btjd_col = None
    for c in ("btjd", "BTJD", "tjd", "TJD", "jd", "JD"):
        if c in wcs_table.columns:
            btjd_col = c
            break
    if btjd_col is None:
        return out

    if "product_id" in wcs_table.columns:
        stems = wcs_table["product_id"].astype(str)
    elif "filename" in wcs_table.columns:
        stems = wcs_table["filename"].map(
            lambda x: tess_product_id_from_ffi_path(str(x)) or ""
        )
    elif "path" in wcs_table.columns:
        stems = wcs_table["path"].map(
            lambda x: tess_product_id_from_ffi_path(str(x)) or ""
        )
    else:
        return out

    btjd = pd.to_numeric(wcs_table[btjd_col], errors="coerce")
    for stem, t in zip(stems, btjd):
        if stem and np.isfinite(t):
            out[str(stem)] = float(t)
    return out


def fit_epsf_all_frames(diff_paths: list,
                         gaia_df: pd.DataFrame,
                         cfg,
                         epsf,
                         output_dir: str = None,
                         round_id: int = 1,
                         *,
                         shared_mask_path: str | None = None,
                         static_mask_path: str | None = None,
                         mask_catalog=None,
                         btjd_by_stem: dict | None = None,
                         wcs_table: pd.DataFrame | None = None,
                         diff_log_path: str | None = None,
                         epsf_label: str | None = None,
                         diffs_input: str | None = None,
                         force_rerun: bool = False) -> tuple:
    """
    Fit gridded ePSF on every difference image in diff_paths.

    Parameters
    ----------
    diff_paths  : list of str (FITS files from hotpants)
    gaia_df     : pd.DataFrame (crop-local Gaia catalog)
    cfg         : SynDiffConfig
    epsf        : EpsfParams
    output_dir  : str, optional
    round_id    : int
    shared_mask_path : deprecated alias of ``static_mask_path``
    static_mask_path : optional on-disk static mask FITS (fallback if no catalog)
    mask_catalog : optional ``MaskCatalog`` — preferred; per-FFI ``mask_at`` (no FITS I/O)
    btjd_by_stem : optional stem→BTJD map (built from ``wcs_table`` when omitted)
    wcs_table : optional frame manifest for BTJD lookup
    force_rerun : when True, ignore already-computed per-frame ePSF models and
        recompute every frame (mirrors the hotpants stage's ``force_rerun``)
    """
    from syndiff_pipeline.difference_imaging.stages import gridded_epsf

    mask_path = static_mask_path or shared_mask_path
    if btjd_by_stem is None and wcs_table is not None:
        btjd_by_stem = btjd_by_stem_from_manifest(wcs_table)

    mask_2d = None
    first_path = next((p for p in diff_paths if p and os.path.exists(p)), None)
    if mask_catalog is None and mask_path and first_path:
        shape = fits.getdata(first_path).shape
        mask_2d = _load_static_mask_2d(mask_path, shape)

    if output_dir is None:
        raise ValueError("fit_epsf_all_frames requires output_dir for gridded ePSF")

    result = gridded_epsf.fit_gridded_epsf_all_frames(
        diff_paths,
        gaia_df,
        cfg,
        epsf,
        output_dir,
        mask_2d=mask_2d,
        mask_catalog=mask_catalog,
        btjd_by_stem=btjd_by_stem,
        round_id=round_id,
        diff_log_path=diff_log_path,
        epsf_label=epsf_label,
        diffs_input=diffs_input,
        skip_existing=not force_rerun,
    )
    epsf_stack, tile_centers, ffi_stems, epsf_ok = result
    save_epsf_stack_bundle(epsf_stack, ffi_stems, output_dir, round_id)
    return epsf_stack, tile_centers, ffi_stems, epsf_ok


def fit_epsf_tiled(diff_image: np.ndarray,
                   gaia_df: pd.DataFrame,
                   cfg,
                   epsf,
                   frame_label: str = "",
                   *,
                   mask_2d: np.ndarray | None = None) -> tuple:
    """Fit gridded ePSF on a single difference image (test / debug helper)."""
    from syndiff_pipeline.difference_imaging.stages import gridded_epsf

    _model, tile_centers, stack = gridded_epsf.build_gridded_psf_for_frame(
        diff_image,
        gaia_df,
        epsf,
        mask_2d=mask_2d,
        frame_label=frame_label,
    )
    if stack is None:
        n_tiles = epsf.tile_ny * epsf.tile_nx
        over_size = 2 * epsf.psf_size + 1
        return (
            np.full((n_tiles, over_size ** 2), np.nan),
            tile_centers,
        )
    flat = gridded_epsf.stack_from_gridded_cube(stack)
    return flat, tile_centers


def tess_mag_from_gaia_phot(
    g: np.ndarray, bp: np.ndarray, rp: np.ndarray
) -> np.ndarray:
    """
    TESS magnitude from Gaia ``phot_g_mean_mag`` / BP / RP (TGLC polynomial).

    Where G, BP, and RP are finite: color polynomial (same as TGLC ``ffi.py``).
    Otherwise (missing color): ``tess_mag = G - 0.430`` when G is finite.
    Non-finite polynomial values for valid color use ``G - 0.430``.

    For a catalog DataFrame including ``tess_flux`` / ``tess_flux_ratio``, use
    :func:`add_tess_flux_ratio` instead.
    """
    g = np.asarray(g, dtype=float)
    bp = np.asarray(bp, dtype=float)
    rp = np.asarray(rp, dtype=float)
    tess = np.full_like(g, np.nan, dtype=float)
    color_ok = np.isfinite(g) & np.isfinite(bp) & np.isfinite(rp)
    dif = np.where(color_ok, bp - rp, np.nan)
    tess_poly = (
        g
        - 0.00522555 * dif ** 3
        + 0.0891337 * dif ** 2
        - 0.633923 * dif
        + 0.0324473
    )
    tess[color_ok] = tess_poly[color_ok]
    bad_poly = color_ok & ~np.isfinite(tess_poly)
    tess[bad_poly] = g[bad_poly] - 0.430
    g_only = np.isfinite(g) & ~color_ok
    tess[g_only] = g[g_only] - 0.430
    return tess


def add_tess_flux_ratio(gaia_df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy of ``gaia_df`` with ``tess_mag`` (via :func:`tess_mag_from_gaia_phot`),
    ``tess_flux``, and ``tess_flux_ratio``.

    Merges any pre-existing ``tess_mag`` with photometry: NaN rows are filled
    from G/BP/RP; see :func:`tess_mag_from_gaia_phot` for the conversion.

    Parameters
    ----------
    gaia_df : pd.DataFrame
        Must have ``phot_g_mean_mag`` when ``tess_mag`` is absent or all NaN.
        ``phot_bp_mean_mag`` and ``phot_rp_mean_mag`` are optional (per column or per row).
        Optionally ``tess_mag`` (pre-computed; NaNs may be filled).

    Returns
    -------
    pd.DataFrame with columns ``tess_mag``, ``tess_flux``, ``tess_flux_ratio``.
    """
    df = gaia_df.copy()
    n = len(df)

    if "phot_g_mean_mag" not in df.columns:
        if "tess_mag" in df.columns and df["tess_mag"].notna().any():
            g = np.full(n, np.nan, dtype=float)
            bp = np.full(n, np.nan, dtype=float)
            rp = np.full(n, np.nan, dtype=float)
        else:
            raise ValueError(
                "add_tess_flux_ratio requires phot_g_mean_mag when tess_mag is "
                "absent or all NaN"
            )
    else:
        g = df["phot_g_mean_mag"].values.astype(float)

    if "phot_bp_mean_mag" in df.columns:
        bp = df["phot_bp_mean_mag"].values.astype(float)
    else:
        bp = np.full(n, np.nan, dtype=float)

    if "phot_rp_mean_mag" in df.columns:
        rp = df["phot_rp_mean_mag"].values.astype(float)
    else:
        rp = np.full(n, np.nan, dtype=float)

    synthesized = tess_mag_from_gaia_phot(g, bp, rp)
    if "tess_mag" not in df.columns or df["tess_mag"].isna().all():
        df["tess_mag"] = synthesized
    else:
        tm = df["tess_mag"].values.astype(float)
        fill = ~np.isfinite(tm)
        if fill.any():
            tm[fill] = synthesized[fill]
        df["tess_mag"] = tm

    df["tess_flux"] = 10.0 ** (-df["tess_mag"].values / 2.5)
    max_flux = np.nanmax(df["tess_flux"].values)
    df["tess_flux_ratio"] = df["tess_flux"] / max_flux if max_flux > 0 else df["tess_flux"]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ── Tile machinery ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _make_tile_grid(ny: int, nx: int, tile_ny: int, tile_nx: int) -> list:
    """
    Return list of (r0, c0, tile_size) tuples for a square tile grid.
    """
    tile_h = ny // tile_ny
    tile_w = nx // tile_nx
    tile_size = min(tile_h, tile_w)
    tiles = []
    for i in range(tile_ny):
        for j in range(tile_nx):
            r0 = i * tile_size
            c0 = j * tile_size
            tiles.append((r0, c0, tile_size))
    return tiles

