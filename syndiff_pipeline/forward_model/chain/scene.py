"""Scene construction: swap pixels into a source scene, demote excluded stars, optionally mask strap columns.

Ported from the e2e scripts ``s3/make_swap_scene.py`` + ``s3/demote_stars.py`` and
``localbg_20260930/mask_straps_scene.py`` (numerics unchanged).

A *scene* is ``scene_bundle.npz`` + ``scene_meta.json``. ``build_scene`` applies, in order:

1. ``swap_data``   data/noise replaced by the values of another full-CCD ``hp_d`` at the same stamp pixels
                   (geometry, roles, valid mask, pairs unchanged);
2. ``demote``      stars (Gaia source_id) listed in ``exclusion_csv`` -> role 2 (nuisance: still modelled as neighbours,
                   never trained on). The list may be much larger than the scene; unmatched ids are counted, not an
                   error (``inputs.exclusion_strict: true`` restores the e2e assert);
3. ``mask_straps`` (``inputs.strap_mask``) valid &= ~strap (bit 4 of the SCC ``shared_mask.fits.fz``, the same mask the
                   scene was exported with, found through ``scene_meta['workspace']``), then scene_export's demotion
                   rule (core pixels < ``min_core_valid`` or centre invalid -> role 2). If the source scene already has
                   ``straps_masked: true`` the step is a no-op (recorded in the meta).

The package resource ``tess_straps.csv`` is NOT used: the reference implementation takes the strap flag from the
shared mask, so the straps masked here are exactly the ones the diff image was built with.

Stage directories: ``scene_boot/`` (hp_d from the bootstrap) and ``scene_final/`` (hp_d of the final image); the scene
files live directly inside the stage dir.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np

from .config import ChainConfig, ConfigError, is_done, mark_done, write_provenance

STRAP_BIT = 4


# ---------------------------------------------------------------------- IO
def _load(scene_dir: Path) -> tuple[dict, dict]:
    z = dict(np.load(Path(scene_dir) / "scene_bundle.npz"))
    meta = json.loads((Path(scene_dir) / "scene_meta.json").read_text())
    return z, meta


def _save(out_dir: Path, z: dict, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scene_bundle.npz", **z)
    (out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------------- steps (in-memory)
def swap_data(z: dict, meta: dict, src_dir: Path, hp_d: str | Path) -> int:
    """Replace ``data``/``noise`` by ``hp_d`` extensions 1/2 at the scene's stamp pixels. Returns n changed pixels."""
    from astropy.io import fits

    s = int(z["stamp"])
    r = np.arange(s) - s // 2
    n = len(z["cx"])
    ys = np.broadcast_to(z["cy"][:, None, None] + r[None, :, None], (n, s, s)).reshape(n, -1)
    xs = np.broadcast_to(z["cx"][:, None, None] + r[None, None, :], (n, s, s)).reshape(n, -1)
    with fits.open(hp_d) as h:
        H, W = h[1].data.shape
        inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        ysc, xsc = np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)
        D = h[1].data[ysc, xsc].astype(np.float32)
        N = h[2].data[ysc, xsc].astype(np.float32)
    keep = z["finite"] & inb
    bad = keep & ~(np.isfinite(D) & np.isfinite(N) & (N > 0))
    if bad.sum() != 0:
        raise ValueError(f"{int(bad.sum())} scene pixels non-finite in {hp_d}")
    old = z["data"]
    z["data"] = np.where(keep, D, z["data"]).astype(np.float32)
    z["noise"] = np.where(keep, N, z["noise"]).astype(np.float32)
    meta["d14_swap"] = dict(source_scene=os.path.abspath(src_dir), hp_d=os.path.abspath(hp_d),
                            note="data/noise replaced at identical pixels; geometry, roles, valid mask, pairs unchanged")
    return int((keep & (z["data"] != old)).sum())


def demote(z: dict, meta: dict, csv: str | Path, strict: bool = False) -> int:
    """Stars whose Gaia ``source_id`` is in the csv -> role 2 (nuisance). Returns the number matched in the scene.

    The list may cover many more stars than the scene (e.g. the whole catalogue T8-13), so by default ids absent from
    the scene are only counted. ``strict=True`` reproduces the e2e ``demote_stars.py`` assert (every listed id must be
    in the scene). The meta records the list path + sha256, ``n_listed`` (unique ids), ``n_matched_in_scene``,
    ``n_listed_not_in_scene`` and the roles before/after."""
    import hashlib

    import pandas as pd

    ids = pd.read_csv(csv)["source_id"].to_numpy()
    sel = np.isin(z["source_id"], ids)
    n_listed = int(len(np.unique(ids)))
    if strict and sel.sum() != len(ids):
        raise ValueError(f"{int(sel.sum())} of {len(ids)} drop stars found in scene")
    before = np.bincount(z["role"], minlength=3)
    z["role"] = np.where(sel, 2, z["role"]).astype(z["role"].dtype)
    after = np.bincount(z["role"], minlength=3)
    meta["n_roles"] = {"contrib": int(after[0]), "anchor": int(after[1]), "nuisance": int(after[2])}
    meta["demoted"] = dict(csv=os.path.abspath(csv), n=int(sel.sum()),
                           roles_before=[int(v) for v in before], roles_after=[int(v) for v in after],
                           csv_sha256=hashlib.sha256(Path(csv).read_bytes()).hexdigest(), n_listed=n_listed,
                           n_matched_in_scene=int(sel.sum()), n_listed_not_in_scene=n_listed - int(sel.sum()),
                           strict=bool(strict))
    return int(sel.sum())


def strap_flags(meta: dict) -> np.ndarray:
    """Boolean full-CCD strap map (bit 4 of ``<workspace>/shared_mask.fits.fz``)."""
    from astropy.io import fits

    if not str(meta.get("mask_source", "")).startswith("shared_mask"):
        raise ValueError(f"strap masking needs a scene exported with the shared mask; mask_source={meta.get('mask_source')!r}")
    with fits.open(Path(meta["workspace"]) / "shared_mask.fits.fz") as h:
        m = [e for e in h if e.data is not None][0].data
    m = (m[0] if m.ndim == 3 else m).astype(np.int64)
    return (m & STRAP_BIT) != 0


def mask_straps(z: dict, meta: dict, src_dir: Path | str = "", strap: np.ndarray | None = None,
                reason: str = "strap columns biased in the difference image") -> dict:
    """valid &= ~strap at each stamp pixel, then scene_export's demotion rule. No-op if already masked."""
    if meta.get("straps_masked"):
        return {"skipped": "scene already has straps_masked=true"}
    if strap is None:
        strap = strap_flags(meta)
    S = int(z["stamp"])
    h = S // 2
    k = np.arange(S * S)
    px = z["cx"][:, None] + (k % S - h)[None]
    py = z["cy"][:, None] + (k // S - h)[None]
    inarr = (px >= 0) & (px < strap.shape[1]) & (py >= 0) & (py < strap.shape[0])
    st = np.zeros_like(inarr)
    st[inarr] = strap[py[inarr], px[inarr]]
    valid0 = np.asarray(z["valid"], bool)
    valid = valid0 & ~st
    off = np.arange(S) - h
    lr = np.hypot(*np.meshgrid(off, off)).reshape(-1)
    core = lr <= float(meta["core_radius"])
    n_core_valid = (valid & core[None]).sum(1)
    centre_ok = valid[:, (S * S) // 2]
    role0 = z["role"].copy()
    dem = (role0 != 2) & ((n_core_valid < int(meta["min_core_valid"])) | ~centre_ok)
    role = role0.copy()
    role[dem] = 2
    z["valid"] = valid
    z["role"] = role
    meta["straps_masked"] = True
    meta["masked_bits"] = sorted(set(meta["masked_bits"]) | {STRAP_BIT})
    meta["n_roles"] = {"contrib": int((role == 0).sum()), "anchor": int((role == 1).sum()),
                       "nuisance": int((role == 2).sum())}
    meta["strap_mask"] = dict(source_scene=str(src_dir), date=_dt.date.today().isoformat(), reason=reason,
                              n_stamp_px_masked=int((valid0 & st).sum()),
                              frac_valid_removed=float((valid0 & st).sum() / valid0.sum()),
                              n_demoted=int(dem.sum()), demoted_source_ids=z["source_id"][dem].tolist())
    return {k_: v for k_, v in meta["strap_mask"].items() if k_ != "demoted_source_ids"}


# hp_d pixels more negative than this many sigma are not a star (stars are positive in hp_d: T<13 are removed from the
# template) but over-subtraction artefacts, e.g. Hotpants around a saturated star's bleed (F2 2026-10-01: ~10 stamps at
# -1e4..-3e4 e-/s carried ~all of the loss and made scene_fit diverge). Data-driven gap: C4's most negative stamp pixel
# is -231 sigma (-119 e-/s), F2's corrupt stamps reach -43638 sigma. Identical for all fields.
NEG_SIGMA_MASK = 300.0
GUARD_ABS_MAX = 1.0e5   # e-/s; any valid stamp pixel beyond this (or non-finite) means a broken input image


def guard_swapped_data(z: dict, hp_d) -> None:
    """Refuse a scene whose swapped-in valid pixels are non-finite or |value| > ``GUARD_ABS_MAX`` (e.g. a mis-decoded
    FITS: the 2026-10-01 DS9 ~1e10 episode)."""
    v = np.asarray(z["valid"], bool)
    d, n = np.asarray(z["data"]), np.asarray(z["noise"])
    bad = v & (~np.isfinite(d) | ~np.isfinite(n) | (np.abs(d) > GUARD_ABS_MAX))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} valid scene pixels non-finite or |data| > {GUARD_ABS_MAX:g} e-/s "
                         f"after swapping in {hp_d}; refusing (broken input image?)")


def mask_negative_outliers(z: dict, meta: dict, nsigma: float = NEG_SIGMA_MASK) -> dict:
    """valid &= ~(data/noise < -nsigma), applied to every stamp sharing the pixel (union ``uid``), then scene_export's
    demotion rule (core pixels < ``min_core_valid`` or centre invalid -> role 2). Recorded in ``meta['neg_outlier_mask']``."""
    S = int(z["stamp"])
    valid0 = np.asarray(z["valid"], bool)
    d, n = np.asarray(z["data"], float), np.asarray(z["noise"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit = valid0 & (n > 0) & (d / np.where(n > 0, n, 1.0) < -float(nsigma))
    if not hit.any():
        meta["neg_outlier_mask"] = dict(rule=f"mask union pixels with data/noise < -{nsigma:g}; then core/centre demotion",
                                        nsigma=float(nsigma), n_union_px_masked=0, n_stamp_px_masked=0,
                                        n_stamps_touched=0, n_demoted=0, demoted_source_ids=[])
        return {k: v for k, v in meta["neg_outlier_mask"].items() if k != "demoted_source_ids"}
    if "uid" in z:
        uid = np.asarray(z["uid"])
        bad_uid = np.unique(uid[hit & (uid >= 0)])
        st = (np.isin(uid, bad_uid) & (uid >= 0)) | hit   # every stamp sharing a flagged union pixel
    else:
        bad_uid, st = np.flatnonzero(hit), hit
    valid = valid0 & ~st
    off = np.arange(S) - S // 2
    core = (np.hypot(*np.meshgrid(off, off)).reshape(-1) <= float(meta["core_radius"]))
    n_core_valid = (valid & core[None]).sum(1)
    centre_ok = valid[:, (S * S) // 2]
    role0 = np.asarray(z["role"]).copy()
    dem = (role0 != 2) & ((n_core_valid < int(meta["min_core_valid"])) | ~centre_ok)
    role = role0.copy()
    role[dem] = 2
    z["valid"], z["role"] = valid, role.astype(np.asarray(z["role"]).dtype)
    meta["n_roles"] = {"contrib": int((role == 0).sum()), "anchor": int((role == 1).sum()), "nuisance": int((role == 2).sum())}
    meta["neg_outlier_mask"] = dict(rule=f"mask union pixels with data/noise < -{nsigma:g}; then core/centre demotion",
                                    nsigma=float(nsigma), n_union_px_masked=int(len(bad_uid)),
                                    n_stamp_px_masked=int((valid0 & st).sum()),
                                    n_stamps_touched=int((valid0 & st).any(1).sum()), n_demoted=int(dem.sum()),
                                    demoted_source_ids=np.asarray(z["source_id"])[dem].tolist(),
                                    min_data_over_sigma=float(np.nanmin(np.where(valid0 & (n > 0), d / np.where(n > 0, n, 1.0), np.nan))))
    return {k: v for k, v in meta["neg_outlier_mask"].items() if k != "demoted_source_ids"}


# ---------------------------------------------------------------------- composed
def build_scene(src_dir: str | Path, out_dir: str | Path, hp_d: str | Path | None = None,
                exclusion_csv: str | Path | None = None, strap_mask: bool = False, exclusion_strict: bool = False) -> dict:
    """swap (if ``hp_d``; then guard + negative-outlier mask) -> demote (if csv) -> strap mask (if flag); writes
    ``out_dir``. Returns a summary dict."""
    z, meta = _load(Path(src_dir))
    summ: dict = {}
    if hp_d is not None:
        summ["n_changed_px"] = swap_data(z, meta, Path(src_dir), hp_d)
        guard_swapped_data(z, hp_d)
        summ["neg_outlier_mask"] = mask_negative_outliers(z, meta)
    if exclusion_csv is not None:
        summ["n_demoted"] = demote(z, meta, exclusion_csv, strict=exclusion_strict)
        summ["demoted"] = {k: meta["demoted"][k] for k in ("n_listed", "n_matched_in_scene", "n_listed_not_in_scene")}
    if strap_mask:
        summ["strap_mask"] = mask_straps(z, meta, src_dir=Path(src_dir))
    _save(Path(out_dir), z, meta)
    summ["n_roles"] = meta["n_roles"]
    return summ


def find_hp_d(root: Path, stem: str) -> Path:
    """The unique ``<stem>*hp_d*.fits[.fz]`` under ``root`` (searched recursively)."""
    hits = sorted({p for pat in (f"{stem}*hp_d*.fits.fz", f"{stem}*hp_d*.fits") for p in Path(root).rglob(pat)})
    if len(hits) != 1:
        raise ConfigError(f"expected exactly one {stem}*hp_d* image under {root}, found {len(hits)}"
                          f"{': ' + ', '.join(map(str, hits)) if hits else ''}; pass --hp-d explicitly")
    return hits[0]


def run_scene(cfg: ChainConfig, which: str, hp_d: str | Path | None = None, force: bool = False) -> Path:
    """Stage ``scene_boot`` (``which='boot'``) or ``scene_final`` (``'final'``). Returns the stage dir."""
    if which not in ("boot", "final"):
        raise ValueError(which)
    stage = cfg.stage_dir(f"scene_{which}")
    if is_done(stage) and not force:
        print(f"[scene_{which}] already done: {stage}")
        return stage
    src = cfg.need("inputs.source_scene")
    if hp_d is None:
        if which == "boot":
            hp_d = cfg.inputs.bootstrap_hp_d or find_hp_d(cfg.stage_dir("bootstrap"), cfg.stem)
        else:
            hp_d = find_hp_d(cfg.stage_dir("final"), cfg.stem)
    (stage / "DONE").unlink(missing_ok=True)
    summ = build_scene(src, stage, hp_d=hp_d, exclusion_csv=cfg.inputs.exclusion_csv, strap_mask=cfg.inputs.strap_mask,
                       exclusion_strict=cfg.inputs.exclusion_strict)
    print(f"[scene_{which}] {summ}")
    ins = {"source_scene_bundle": Path(src) / "scene_bundle.npz", "source_scene_meta": Path(src) / "scene_meta.json",
           "hp_d": hp_d}
    if cfg.inputs.exclusion_csv:
        ins["exclusion_csv"] = cfg.inputs.exclusion_csv     # path + sha256 (small file)
        ins["exclusion_counts"] = {"strict": cfg.inputs.exclusion_strict, **summ["demoted"]}
    write_provenance(stage, cfg, ins)
    mark_done(stage)
    return stage
