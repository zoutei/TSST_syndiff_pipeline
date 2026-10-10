"""``make_init --route photutils``: initial ePSF + WCS from the target frame's own difference image.

Steps (all on the TARGET frame's bootstrap hp_d; fold fits use TRAINING-fold stars only):

1. ePSF: photutils ``EPSFBuilder`` pooled per CCD cell (the 6 x 6 tile grid and every filter of
   ``init_study.build_epsf_prior`` -- its ``_build_pooled_from_prepared`` / ``_fit_pooled_tile`` are called
   unchanged) on the isolated T 8-11 scene contributors (role 0).
2. Convert each tile stamp to the forward model's node grid (``init_study.photutils_to_forward_epsf``), restoring
   unit flux first (photutils ImagePSF/GriddedPSFModel do not normalise their input; 07-31 lesson), then RECENTRE
   every node so its Gaussian-core centroid is at (0, 0): ``epsf_model.core_centroid_xy`` is the gauge
   ``decode_epsf_base`` enforces at every step of scene_fit (``recentre_core``: the same centroid / bilinear-shift
   primitives, iterated to convergence with a single interpolation). The applied offsets are recorded.
3. Free-x,y PSF photometry (photutils ``PSFPhotometry`` on a ``GriddedPSFModel`` built from the DECODED start ePSF,
   i.e. exactly what scene_fit renders at step 0) of the scene stars on the same hp_d.
4. ``cheb_wcs.fit_frame_cheb_warmstart`` (the routine that built the source bundle's ``params0``) on the
   centroids vs the epoch-propagated Gaia positions of the source bundle -> ``wcs_coeff`` (same degree / layout).

jax is imported first (pyarrow-before-jax segfaults XLA).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax  # noqa: F401  (import order is load-bearing: before pandas/pyarrow)
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .. import cheb_wcs as CW
from .. import epsf_model as EM
from . import folds as FO

FOLD_SEED = 20260929          # build_folds default
FOLD_TILE = 128
N_FOLDS = 5


def _log(msg: str) -> None:
    print(f"[photutils_init {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- folds / stars
def fold_assignment(x, y, *, n_folds: int = N_FOLDS, seed: int = FOLD_SEED, tile: int = FOLD_TILE,
                    pattern: str = "diagonal") -> np.ndarray:
    """Fold of science-local positions: ``build_folds`` layout (K=5: ``tile_map_diagonal``)."""
    tmap = (FO.tile_map_diagonal(seed, tile, n_folds) if pattern == "diagonal"
            else FO.tile_map(seed, tile, 2, n_folds))
    return FO.fold_of(tmap, tile, x, y)


def select_stars(z: dict, *, fold: int | None, n_folds: int = N_FOLDS, seed: int = FOLD_SEED, tile: int = FOLD_TILE,
                 pattern: str = "diagonal", min_sep_px: float = 6.0, contrib_mag=(8.0, 11.0),
                 neighbour_mag_max: float = 13.0, folds_csv=None) -> pd.DataFrame:
    """Per scene star: position, role, fold and the three use flags.

    ``train``    role != 2 (not a nuisance: scene.py demotes every excluded / masked-core star to role 2) and not in
                 the held-out fold -- the stars whose centroids may enter the WCS fit.
    ``pool``     train & role 0 (the T 8-11 ePSF contributors) & isolated: no other scene star brighter than
                 ``neighbour_mag_max`` within ``min_sep_px`` (``irregular_stamps.select_isolated_primaries``, the
                 rule the source bundle's isolated primaries used; 0 disables it).
    """
    from ..irregular_stamps import select_isolated_primaries

    x = np.asarray(z["x0"], float)
    y = np.asarray(z["y0"], float)
    T = np.asarray(z["tess_mag"], float)
    role = np.asarray(z["role"])
    fd = fold_assignment(x, y, n_folds=n_folds, seed=seed, tile=tile, pattern=pattern)
    if folds_csv is not None:       # the harness's own folds.csv (source_id, fold) is authoritative where it has the star
        ff = pd.read_csv(folds_csv, usecols=["source_id", "fold"]).drop_duplicates("source_id").set_index("source_id")["fold"]
        got = pd.Series(np.asarray(z["source_id"])).map(ff)
        fd = np.where(got.notna().to_numpy(), got.fillna(-1).to_numpy(), fd).astype(np.int8)
    held = np.zeros(len(x), bool) if fold is None else (fd == int(fold))
    train = (role != 2) & ~held
    iso = np.ones(len(x), bool)
    if min_sep_px > 0:
        keep = select_isolated_primaries(x, y, T, mag_lo=-np.inf, mag_hi=np.inf, bright_mag_max=neighbour_mag_max,
                                         min_sep_px=min_sep_px)
        iso[:] = False
        iso[keep] = True
    pool = train & (role == 0) & iso & (T >= contrib_mag[0]) & (T < contrib_mag[1])
    return pd.DataFrame(dict(idx=np.arange(len(x)), source_id=np.asarray(z["source_id"]), x=x, y=y, tess_mag=T,
                             tess_flux=np.asarray(z["tess_flux"], float), role=role, fold=fd, held=held,
                             train=train, isolated=iso, pool=pool))


# ---------------------------------------------------------------------------------------------- masks
def reject_mask(meta: dict, hpd_mask: np.ndarray, source: str = "scene") -> tuple[np.ndarray, str]:
    """True-where-rejected pixel mask for pooling and photometry.

    ``scene`` (default): the SCC ``shared_mask.fits.fz`` found through ``scene_meta['workspace']`` -- the very mask
    ``scene_export`` / ``chain.scene`` built the scene's ``valid`` from -- keeping the scene's masked bits except
    bit 1 (very bright star squares: stars sit inside these; ``masking.bits.epsf_reject_mask`` ignores 1|2|32 too).
    ``hpd``: the hp_d mask plane (HDU3) through ``epsf_reject_mask``.
    """
    from astropy.io import fits

    from syndiff_pipeline.difference_imaging.masking.bits import epsf_reject_mask

    if source == "hpd":
        return epsf_reject_mask(hpd_mask), "hp_d HDU3 via epsf_reject_mask"
    with fits.open(Path(meta["workspace"]) / "shared_mask.fits.fz") as h:
        m = [e for e in h if e.data is not None][0].data
    m = (m[0] if m.ndim == 3 else m).astype(np.int64)
    bits = set(int(b) for b in meta["masked_bits"]) - {1}
    if not meta.get("straps_masked", False):
        bits -= {4}
    sel = sum(bits)
    return (m & sel) != 0, f"shared_mask.fits.fz bits {sorted(bits)} (scene_meta masked_bits minus bit 1)"


# ---------------------------------------------------------------------------------------------- ePSF
def pool_epsf(image: np.ndarray, reject: np.ndarray, pool_xy: pd.DataFrame, *, tile_ny: int, tile_nx: int,
              n_jobs: int = 1, min_stars_per_tile: int = 5, ckpt_dir: Path | None = None):
    """Photutils ePSF per CCD cell: ``build_epsf_prior._build_pooled_from_prepared`` on one prepared frame.

    ``image`` is the hp_d difference image (science crop), ``reject`` a boolean True-where-rejected mask
    (``masking.bits.epsf_reject_mask`` of the hp_d mask plane), ``pool_xy`` has science-local ``x``, ``y``.
    Returns ``(stack (tile_ny*tile_nx, n, n) photutils density, grid_xypos, stats)``.
    """
    from ..init_study.build_epsf_prior import _build_pooled_from_prepared
    from .._vendor.ref_epsf_photometry.pool_epsf import EpsfBuildParams, _PreparedFrame

    img = np.array(image, dtype=np.float64)
    bad = ~np.isfinite(img)
    img[bad] = 0.0
    mask = np.asarray(reject, bool) | bad
    gaia_frame = pd.DataFrame({"x": np.asarray(pool_xy["x"], float), "y": np.asarray(pool_xy["y"], float),
                               "tess_mag": np.asarray(pool_xy["tess_mag"], float)})
    frame = _PreparedFrame(stem="target", diff_img=img, gaia_frame=gaia_frame, full_mask=mask)
    params = EpsfBuildParams(tile_nx=int(tile_nx), tile_ny=int(tile_ny), min_stars_per_tile=int(min_stars_per_tile),
                             n_jobs=int(n_jobs), progress_bar=False)
    return _build_pooled_from_prepared([frame], params, tile_ckpt_dir=ckpt_dir)


def recentre_core(node: np.ndarray, *, tol: float = 2e-6, max_iter: int = 60, clip_nonneg: bool = True):
    """Shift ``node`` (G, G) so its Gaussian-core centroid is at (0, 0) in the ``epsf_model.core_centroid_xy`` gauge.

    ``epsf_model.recenter_grid_core`` does two shift+renormalise passes. The windowed centroid of a symmetric
    profile offset by d is only ~d * w / (w + s) (w, s = window and PSF variances), so two passes leave ~25-40 % of d
    for a photutils stamp that starts a few hundredths of a pixel off, and every extra pass is another bilinear
    interpolation (a blur of ~f(1-f)/16 px^2). Here the total shift is found by a fixed-point iteration
    s <- s - centroid(shift(node, s)) always applied to the ORIGINAL grid, so the result carries exactly one bilinear
    interpolation and a centroid |c| < ``tol``. Returns ``(grid float32 sum 1, (sx, sy) applied shift in px,
    centroid before, centroid after)``.
    """
    g = jnp.asarray(node, jnp.float32)
    c0 = tuple(float(v) for v in EM.core_centroid_xy(g))
    sx = sy = 0.0
    out = g
    for _ in range(max_iter):
        out = EM.bilinear_shift_physical(g, jnp.float32(sx), jnp.float32(sy))
        if clip_nonneg:
            out = jnp.clip(out, 0.0)
        out = out / (jnp.sum(out) + 1e-12)
        cx, cy = (float(v) for v in EM.core_centroid_xy(out))
        if max(abs(cx), abs(cy)) < tol:
            break
        sx, sy = sx - cx, sy - cy
    c1 = tuple(float(v) for v in EM.core_centroid_xy(out))
    return np.asarray(out, np.float32), (sx, sy), c0, c1


def stack_to_node_base(stack: np.ndarray, *, tile_ny: int, tile_nx: int, oversample: int = 4):
    """Photutils tile stack -> forward node grids, recentred; returns ``(base, table)``.

    Per tile: density -> flux fraction (unit sum restored; the post-crop renormalisation lesson of 07-31), area-overlap
    resample onto the node grid, then ``recentre_core``. ``table`` (one row per node) holds the core centroid BEFORE
    recentring, the applied shift (px, = minus the before-centroid at first order), the centroid after, and the
    sums at each step.
    """
    from ..init_study.photutils_to_forward_epsf import (photutils_density_to_flux_fraction,
                                                        resample_oversampled_stamp_to_node)

    stack = np.asarray(stack, np.float64)
    if stack.shape[0] != tile_ny * tile_nx:
        raise ValueError(f"stack has {stack.shape[0]} tiles, expected {tile_ny * tile_nx}")
    G = EM.NODE_GRID_SIZE
    base = np.zeros((tile_ny, tile_nx, G, G), np.float32)
    rows = []
    for idx in range(tile_ny * tile_nx):
        i, j = divmod(idx, tile_nx)
        raw_sum = float(stack[idx].sum())
        frac = photutils_density_to_flux_fraction(stack[idx], oversample=oversample)
        node = resample_oversampled_stamp_to_node(frac, oversample_src=oversample)
        out, (sx, sy), c0, c1 = recentre_core(node)
        base[i, j] = out
        rows.append(dict(row=i, col=j, core_cx_before=c0[0], core_cy_before=c0[1], shift_x=sx, shift_y=sy,
                         core_cx_after=c1[0], core_cy_after=c1[1], photutils_stamp_sum=raw_sum,
                         node_sum_before=float(node.sum(dtype=np.float64)), node_sum_after=float(out.sum(dtype=np.float64))))
    return base, pd.DataFrame(rows)


def start_epsf(base_node: np.ndarray, stamp: int):
    """Node grids -> ``(epsf_base_raw, decoded)`` on the scene's ``stamp`` geometry.

    The wing extension is scene_export's (zero pad + r^-3 power law fitted to the 4-5 px annulus, so the outer
    softplus leaves are alive); ``decoded`` = ``decode_epsf_base(raw)``, i.e. exactly the ePSF scene_fit renders at
    step 0 (flux-rule + core recentre applied).
    """
    from ..scene_export import extend_epsf_seed

    ext = extend_epsf_seed(np.asarray(base_node, np.float64), int(stamp))
    raw = np.asarray(EM.encode_epsf_base(jnp.asarray(ext)), np.float32)
    dec = np.asarray(EM.decode_epsf_base(jnp.asarray(raw)), np.float32)
    return raw, dec


def gridded_model(decoded: np.ndarray, node_x: np.ndarray, node_y: np.ndarray, oversample: int = 4):
    """photutils ``GriddedPSFModel`` from forward node grids ``(ny, nx, G, G)`` (flux fraction, sum 1).

    Stamps are stored in photutils' density convention (sum = oversample^2); the grid is placed at the forward
    model's node positions (science-local px) in row-major tile order.
    """
    from astropy.nddata import NDData
    from photutils.psf import GriddedPSFModel

    ny, nx = decoded.shape[:2]
    data = (np.asarray(decoded, np.float64) * float(oversample) ** 2).reshape(ny * nx, *decoded.shape[2:])
    xy = np.array([(float(node_x[j]), float(node_y[i])) for i in range(ny) for j in range(nx)])
    return GriddedPSFModel(NDData(data=data, meta={"grid_xypos": xy, "oversampling": int(oversample)}))


# ---------------------------------------------------------------------------------------------- photometry
def header_xy(ffi, ra, dec) -> tuple[np.ndarray, np.ndarray]:
    """Science-local pixel positions of (ra, dec) from the FFI header WCS (the WCS the F=4 mapping stage used).

    Same projection as ``build_folds.project_catalogue`` but WITHOUT its median shift onto the scene's fitted
    positions (so no information from the earlier fit enters): x = column - 44, y = row, 0-based.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(ffi) as hd:
        h = next(x.header for x in hd if x.header.get("NAXIS") == 2 and "CTYPE1" in x.header)
    col, row = WCS(h).all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
    return col - 44.0, row


def find_ffi(meta: dict) -> Path:
    """The target FFI beside the scene's lane: ``<workspace>/../ffi/<tess product id>*``."""
    stem = str(meta["frame_stem"]).split("-s")[0]
    hits = sorted((Path(meta["workspace"]).parent / "ffi").glob(f"{stem}*"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one FFI {stem}* under {Path(meta['workspace']).parent / 'ffi'}, found {hits}")
    return hits[0]


def _phot_block(image, noise, mask, model, stars: pd.DataFrame, cfg_kwargs: dict, xy_bounds):
    from ..init_study.stamp_qa import StampQaConfig, run_gridded_photometry

    df, _ = run_gridded_photometry(image, model, stars, cfg=StampQaConfig(**cfg_kwargs), fix_xy=False,
                                   compute_stamp_chi2=False, mask=mask, error=noise, xy_bounds=xy_bounds)
    return df


def run_photometry(image, noise, reject, model, stars: pd.DataFrame, *, block: int = 512, margin: float = 16.0,
                   n_jobs: int = 1, cfg_kwargs: dict | None = None, xy_bounds: float | None = None) -> pd.DataFrame:
    """Free-x,y PSF photometry of ``stars`` (``x_init``, ``y_init``, ``tess_flux``, ``source_id``...).

    The image is cut into ``block``-px squares; each block fits the stars inside it plus a ``margin``-px ring of
    neighbours (so simultaneous groups are not truncated), keeps only the block's own stars, and blocks run in parallel
    (joblib/loky). Inputs are exactly those of ``init_study.stamp_qa.run_gridded_photometry`` (fit_shape 11, aperture
    4, grouper 7 px) plus the hp_d noise as ``error`` and the reject mask.
    """
    from joblib import Parallel, delayed

    cfg_kwargs = cfg_kwargs or {}
    H, W = image.shape
    x, y = stars["x_init"].to_numpy(float), stars["y_init"].to_numpy(float)
    jobs, owners = [], []
    for by in range(0, H, block):
        for bx in range(0, W, block):
            own = (x >= bx) & (x < bx + block) & (y >= by) & (y < by + block)
            if not own.any():
                continue
            near = (x >= bx - margin) & (x < bx + block + margin) & (y >= by - margin) & (y < by + block + margin)
            jobs.append(near)
            owners.append(own)
    _log(f"photometry: {len(stars)} stars in {len(jobs)} blocks (block {block}, margin {margin}, n_jobs {n_jobs})")
    img = np.asarray(image, np.float64)
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_phot_block)(img, noise, reject, model, stars.loc[near].reset_index(drop=True), cfg_kwargs, xy_bounds)
        for near in jobs)
    out = []
    for near, own, df in zip(jobs, owners, res):
        sid = stars.loc[near, "source_id"].to_numpy()
        keep = own[near]
        d = df.loc[keep].copy()
        d["scene_idx"] = stars.loc[near, "idx"].to_numpy()[keep]
        out.append(d)
        assert (sid[keep] == d["source_id"].to_numpy()).all()
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------- WCS
def fit_wcs(x_lin, y_lin, x_obs, y_obs, static: CW.ChebWcsStatic):
    """``cheb_wcs.fit_frame_cheb_warmstart`` -> ``(wcs_coeff (2*n_terms, 1) float32, kept mask)``."""
    cx, cy, kept = CW.fit_frame_cheb_warmstart(np.asarray(x_lin, float), np.asarray(y_lin, float),
                                               np.asarray(x_obs, float), np.asarray(y_obs, float), static)
    return np.concatenate([cx, cy])[:, None].astype(np.float32), kept


def predict_xy(x_lin, y_lin, static: CW.ChebWcsStatic, coeff) -> tuple[np.ndarray, np.ndarray]:
    """Static-frame pixel positions from the linear prediction and ``wcs_coeff`` (single column, no temporal basis)."""
    xh = (np.asarray(x_lin, float) - static.center[0]) / static.half_extents[0]
    yh = (np.asarray(y_lin, float) - static.center[1]) / static.half_extents[1]
    basis = np.asarray(CW.cheb_star_basis(jnp.asarray(xh), jnp.asarray(yh), static.poly_degree, static.exponents))
    c = np.asarray(coeff, float).reshape(-1)
    n = static.n_terms
    return np.asarray(x_lin, float) + basis @ c[:n], np.asarray(y_lin, float) + basis @ c[n:]


def cell_medians(x, y, dx, dy, cell: int = 128, npix: int = 2048):
    """Per-``cell`` median of dx, dy and count: three (n, n) arrays."""
    n = npix // cell
    ix = np.clip((np.asarray(x) // cell).astype(int), 0, n - 1)
    iy = np.clip((np.asarray(y) // cell).astype(int), 0, n - 1)
    mx, my, cnt = np.full((n, n), np.nan), np.full((n, n), np.nan), np.zeros((n, n), int)
    for a in range(n):
        for b in range(n):
            s = (iy == a) & (ix == b)
            cnt[a, b] = s.sum()
            if s.any():
                mx[a, b], my[a, b] = np.median(np.asarray(dx)[s]), np.median(np.asarray(dy)[s])
    return mx, my, cnt


# ---------------------------------------------------------------------------------------------- driver
def build_photutils_init(*, scene_dir, hp_d, out, fold: int | None = None, n_folds: int = N_FOLDS,
                         seed: int = FOLD_SEED, tile: int = FOLD_TILE, pattern: str = "diagonal",
                         min_sep_px: float = 6.0, n_jobs: int = 6, phot_init: str = "header", ffi=None, xy_bounds: float = 2.0,
                         wcs_tmag_max: float = 13.0, block: int = 512, donor_meta: dict | None = None,
                         max_shift_px: float = 1.9, mask_source: str = "scene", folds_csv=None) -> dict:
    """Run steps 1-4 and write ``params_init.npz`` + ``fit_meta.json`` (+ QA tables) into ``out``."""
    from astropy.io import fits

    from .. import fit_bundle as FB
    from ..scene_fit import G8_DEFAULT_EXTRAS, G8_DEFAULT_GAUGE
    scene_dir, out = Path(scene_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(scene_dir / "scene_bundle.npz"))
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    stamp = int(z["stamp"])
    n_rows, n_cols = z["epsf_seed"].shape[:2]
    b = FB.load_fit_bundle(Path(meta["source_bundle"]))
    sbi = z["star_bundle_index"]

    with fits.open(hp_d) as h:
        img = np.asarray(h[1].data, np.float64)
        err = np.asarray(h[2].data, np.float64)
        msk = np.asarray(h[3].data, np.int64)
    reject, mask_desc = reject_mask(meta, msk, mask_source)
    reject = reject | ~np.isfinite(img) | ~np.isfinite(err) | (err <= 0)
    err = np.where(np.isfinite(err) & (err > 0), err, 1.0)
    if img.shape != (2048, 2048):
        raise ValueError(f"hp_d shape {img.shape} != (2048, 2048)")

    st = select_stars(z, fold=fold, n_folds=n_folds, seed=seed, tile=tile, pattern=pattern, min_sep_px=min_sep_px,
                     folds_csv=folds_csv)
    st["x_lin"] = np.asarray(b.x_lin, float)[sbi]
    st["y_lin"] = np.asarray(b.y_lin, float)[sbi]
    counts = dict(n_scene=int(len(st)), n_role0=int((st.role == 0).sum()), n_role1=int((st.role == 1).sum()),
                  n_role2=int((st.role == 2).sum()), n_held_out=int(st.held.sum()), n_train=int(st.train.sum()),
                  n_pool=int(st.pool.sum()), n_role0_train=int(((st.role == 0) & st.train).sum()),
                  n_role0_train_not_isolated=int(((st.role == 0) & st.train & ~st.isolated).sum()))
    _log(f"fold {fold}: {counts}")
    if fold is not None and (st.pool & st.held).any():
        raise AssertionError("held-out star in the ePSF pool")

    # 1. ePSF
    t0 = time.time()
    stack, grid_xypos, pstats = pool_epsf(img, reject, st[st.pool], tile_ny=n_rows, tile_nx=n_cols, n_jobs=n_jobs,
                                          ckpt_dir=None)
    _log(f"pooled ePSF: tiles ok {pstats['n_tiles_ok']}/{pstats['n_tiles_total']} ({time.time() - t0:.0f}s)")
    # 2. node grids + recentre
    base_node, offs = stack_to_node_base(stack, tile_ny=n_rows, tile_nx=n_cols)
    ok = np.zeros(n_rows * n_cols, bool)
    for k, v in pstats["tile_cutout_counts"].items():
        i, j = (int(t) for t in k.split("_"))
        ok[i * n_cols + j] = v >= 5
    offs["n_cutouts"] = [pstats["tile_cutout_counts"][f"{r}_{c}"] for r, c in zip(offs.row, offs.col)]
    offs["fallback_mean_of_ok_tiles"] = ~ok
    raw, dec = start_epsf(base_node, stamp)
    cc = np.array([[float(v) for v in EM.core_centroid_xy(jnp.asarray(dec[i, j]))] for i in range(n_rows)
                   for j in range(n_cols)])
    offs["core_cx_start"], offs["core_cy_start"] = cc[:, 0], cc[:, 1]
    offs["start_sum"] = dec.reshape(n_rows * n_cols, -1).sum(1).astype(float)
    offs.to_csv(out / "recentring_offsets.csv", index=False)
    np.savez_compressed(out / "epsf_stages.npz", photutils_stack=stack.astype(np.float32), node_recentred=base_node,
                        start_decoded=dec)
    if not np.allclose(offs.start_sum, 1.0, atol=1e-4):
        raise AssertionError(f"start ePSF not unit flux: {offs.start_sum.min()}..{offs.start_sum.max()}")

    # 3. photometry
    model = gridded_model(dec, np.asarray(b.epsf_grid.node_x), np.asarray(b.epsf_grid.node_y))
    cand = st[(st.tess_mag <= 13.0) & np.isfinite(st.x) & (st.x > -5) & (st.x < 2053) & (st.y > -5) & (st.y < 2053)]
    if phot_init == "header":
        ffi = Path(ffi) if ffi else find_ffi(meta)
        xi, yi = header_xy(ffi, np.asarray(b.ra, float)[sbi][cand.idx.to_numpy()], np.asarray(b.dec, float)[sbi][cand.idx.to_numpy()])
        xi, yi = pd.Series(xi, index=cand.index), pd.Series(yi, index=cand.index)
    elif phot_init == "scene":
        xi, yi = cand.x, cand.y
    else:
        xi, yi = cand.x_lin, cand.y_lin
    stars = pd.DataFrame(dict(idx=cand.idx.to_numpy(), source_id=cand.source_id.to_numpy(), x_init=xi.to_numpy(),
                              y_init=yi.to_numpy(), tess_flux=cand.tess_flux.to_numpy(),
                              tess_mag=cand.tess_mag.to_numpy()))
    ph = run_photometry(img, err, reject, model, stars, block=block, n_jobs=n_jobs, xy_bounds=xy_bounds)
    ph = ph.merge(st[["idx", "role", "fold", "held", "train", "pool", "x_lin", "y_lin", "x", "y"]].rename(
        columns={"idx": "scene_idx", "x": "x_scene", "y": "y_scene"}), on="scene_idx")
    ph.to_parquet(out / "photometry.parquet", index=False)

    # 4. WCS from the TRAINING stars only
    fl = ph["flags"].to_numpy() if "flags" in ph else np.zeros(len(ph), int)
    good = (ph.train.to_numpy() & ((ph.role == 0) | (ph.role == 1)).to_numpy() & (ph.tess_mag <= wcs_tmag_max).to_numpy()
            & np.isfinite(ph.x_fit) & np.isfinite(ph.y_fit) & ((fl & (2 | 4 | 8 | 32)) == 0)
            & (np.hypot(ph.x_fit - ph.x_init, ph.y_fit - ph.y_init) < max_shift_px))
    g = ph[good]
    wcs_coeff, kept = fit_wcs(g.x_lin, g.y_lin, g.x_fit, g.y_fit, b.cheb_static)
    xp, yp = predict_xy(ph.x_lin, ph.y_lin, b.cheb_static, wcs_coeff)
    ph["x_wcs"], ph["y_wcs"] = xp, yp
    ph["dx"], ph["dy"] = ph.x_fit - xp, ph.y_fit - yp
    ph["wcs_fit_used"] = False
    ph.loc[g.index[kept], "wcs_fit_used"] = True
    ph.to_parquet(out / "photometry.parquet", index=False)

    def _rms(s):
        return dict(n=int(len(s)), dx_med=float(s.dx.median()), dy_med=float(s.dy.median()),
                    dx_mad_sigma=float(1.4826 * np.median(np.abs(s.dx - s.dx.median()))),
                    dy_mad_sigma=float(1.4826 * np.median(np.abs(s.dy - s.dy.median()))))
    used = ph[ph.wcs_fit_used]
    res = dict(n_phot=int(len(ph)), n_candidates=int(good.sum()), n_used=int(kept.sum()), fit_stars=_rms(used))
    ho = ph[ph.held & ((ph.role == 0) | (ph.role == 1)) & np.isfinite(ph.dx) & (np.hypot(ph.x_fit - ph.x_init,
            ph.y_fit - ph.y_init) < max_shift_px)] if fold is not None else ph.iloc[0:0]
    if len(ho):
        res["held_out_stars"] = _rms(ho)
    mx, my, cnt = cell_medians(used.x_fit, used.y_fit, used.dx, used.dy)
    res["cell128_median_abs_max_px"] = float(np.nanmax(np.hypot(mx, my)))
    res["cell128_median_abs_median_px"] = float(np.nanmedian(np.hypot(mx, my)))
    np.savez(out / "wcs_cell_residuals.npz", dx=mx, dy=my, n=cnt)

    # 5. outputs
    n8 = 8 + len([s for s in (donor_meta or {}).get("chroma_g8_extras", G8_DEFAULT_EXTRAS).split(",") if s.strip()])
    prm = {"wcs_coeff": wcs_coeff, "epsf_modes": np.zeros((0, n_rows, n_cols) + raw.shape[-2:], np.float32),
           "w_coeff": np.zeros((0, 1), np.float32), "epsf_base_raw": raw, "chroma_g8": np.zeros(n8, np.float32),
           "epsf_repr": np.asarray(EM.EPSF_REPR)}
    np.savez(out / "params_init.npz", **prm)
    dm = donor_meta or {}
    fm = {"chroma_g8_gauge": dm.get("chroma_g8_gauge", G8_DEFAULT_GAUGE),
          "chroma_g8_extras": dm.get("chroma_g8_extras", G8_DEFAULT_EXTRAS),
          "chroma_g8_blur_order": dm.get("chroma_g8_blur_order", 0), "chroma_g8_no_dil": dm.get("chroma_g8_no_dil", False),
          "chroma_g8_drop": dm.get("chroma_g8_drop", ""), "chroma_radial_knots": dm.get("chroma_radial_knots"),
          "crossfit_init": dict(
              route="photutils", target_scene=str(scene_dir), target_frame=meta.get("frame_stem"),
              hp_d=str(Path(hp_d).resolve()), mask=mask_desc, frac_rejected=float(reject.mean()),
              source_bundle=meta["source_bundle"], fold=fold,
              fold_spec=dict(n_folds=n_folds, seed=seed, tile=tile, pattern=pattern, folds_csv=str(folds_csv) if folds_csv else None),
              star_counts=counts, min_sep_px=min_sep_px, tile_grid=[n_rows, n_cols],
              pool_stats={k: v for k, v in pstats.items() if k != "tile_cutout_counts"},
              recentring_offsets_px=[{k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v))
                                      for k, v in r.items()} for r in offs.to_dict("records")],
              photometry=dict(init=phot_init, ffi=str(ffi) if phot_init == "header" else None, xy_bounds=xy_bounds, fit_shape=11, aperture_radius=4.0, grouper_min_separation=7.0,
                              model_positions="forward-model node positions (edge placement)",
                              tmag_max_wcs=wcs_tmag_max, max_shift_px=max_shift_px),
              wcs=dict(res, source="cheb_wcs.fit_frame_cheb_warmstart on photometry centroids vs bundle x_lin/y_lin "
                                   "(Gaia epoch-propagated, bundle meta gaia_pm_propagation)",
                       degree=int(b.cheb_static.poly_degree), shape=list(wcs_coeff.shape)),
              wing_extension="scene_export.extend_epsf_seed (r^-3 beyond 5 px)",
              chroma_g8="zeros; A3 extras zero")}
    (out / "fit_meta.json").write_text(json.dumps(fm, indent=1))
    _log(f"wrote {out / 'params_init.npz'}; WCS residual (fit stars) {res['fit_stars']}")
    return fm
