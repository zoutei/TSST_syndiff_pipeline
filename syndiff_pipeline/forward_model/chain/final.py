"""Stage ``final``: per-band / achromatic template-kernel difference images (e2e ``f05_dryrun.py``).

Variants (``VARIANTS``):
  band       T_b convolved with the band kernels K_b = K0 + delta_b K1, in the moment form
             C = K0 (*) sum_b T_b + K1 (*) sum_b delta_b T_b
  achrom     T_sum convolved with the achromatic kernel K_achrom (the model PSF at c_ref)
  band_w, achrom_w   the same with the ADOPTED band weights (D13, ``inputs.adopted_weights``):
             T'_b = s_b T_b with s_b = w'_b / w_b, and per band delta'_b = (alpha + beta lambda_b - u_ref)/(du/dc) so the
             A3 colour label is unchanged (D18); K0, K1 unchanged.

ADOPTED-WEIGHT RESCALE.  ``s_b`` is computed against the weights the combined store actually carried when the band
templates were built (``perband/band_templates/store_weights.json``, written by f04 from the cells' recorded recipe):
``s_b = weight_rescale(store_weights, adopted weights_rizy)``.  For a production-weight store this is
0.254/0.238, ... = ADOPTED_WEIGHTS.json ``scale_vs_production`` (bit-identical to the e2e constants); for a store already
built with the adopted D13 weights it is exactly 1 in every band (no second rescale), while delta' is recalibrated
either way.  The unweighted variants use the templates as built, so with an adopted-weight store ``band`` equals ``band_w``
up to the delta recalibration -- use the ``_w`` variants for the adopted pipeline.

Difference: target = FFI science region - ks_b background; a(x,y) Chebyshev order 2 + b const, weighted LSQ on the
reference-good pixels outside the T<13 footprints (``_tk.run_match``); D = target - (a C + b).  ext2/ext3 = noise and mask of
the Hotpants baseline of the same template and frame (``hotpants/<stem>/hp_d``).  The output is an hp_d-format FITS
``final/<stem>/<variant>/hp_d/<stem>_hp_d.fits.fz`` (diff, noise, mask), lossless GZIP_1.

Port of e2e ``f05_dryrun.py`` with the weight-aware rescale; numerics unchanged.  Variant names drop the historic
``_a3`` label (e2e ``band_a3w`` -> ``band_w``).  Outputs per frame also: ``match/match.json`` (fit stats).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from . import _tk as TK
from .config import is_done, mark_done, write_provenance
from .perband.paths import BANDS, LAM, chain_band_weights, chain_paths

log = logging.getLogger(__name__)

# variant -> (kernel file in kernels/, adopted weights?)
VARIANTS: dict[str, tuple[str, bool]] = {
    "band": ("band_epsf.npz", False),
    "achrom": ("K_achrom.npz", False),
    "band_w": ("band_epsf.npz", True),
    "achrom_w": ("K_achrom.npz", True),
}


def adopted_weights(cfg) -> dict[str, float]:
    w = json.loads(Path(cfg.inputs.adopted_weights).read_text())["weights_rizy"]
    return {b: float(v) for b, v in zip(BANDS, w)}


def store_weights(cfg) -> dict[str, float]:
    """The band weights the band templates were built with: f04's ``store_weights.json`` if present, else the chain's
    assumption (``inputs.combined_store_weights``)."""
    P = chain_paths(cfg)
    if P.store_weights_json.is_file():
        d = json.loads(P.store_weights_json.read_text())["store_band_weights"]
        return {b: float(d[b]) for b in BANDS}
    return chain_band_weights(cfg)


def band_scales(cfg, weighted: bool, store_w: Optional[Mapping[str, float]] = None) -> np.ndarray:
    """s_b = w'_b / w_b for the ``_w`` variants (adopted over store weights); ones for the unweighted variants."""
    from syndiff_pipeline.template_creation.processing import perband as PB
    if not weighted:
        return np.ones(4)
    sw = dict(store_w) if store_w is not None else store_weights(cfg)
    s = PB.weight_rescale(sw, adopted_weights(cfg))
    return np.array([s[b] for b in BANDS])


def delta_new(cfg, kz) -> np.ndarray:
    """delta'_b = (alpha + beta lambda_b - u_ref)/(du/dc): adopted-weight band labels on the unchanged A3 colour scale."""
    ad = json.loads(Path(cfg.inputs.adopted_weights).read_text())
    m = json.loads(str(kz["meta"]))
    al, be = ad["colour_map_label_from_new"]
    return (al + be * np.asarray(LAM) - m["u_ref_nm"]) / m["du_dc_nm_per_mag"]


def model_image(cfg, variant: str, *, workers: int = 8, store_w: Optional[Mapping[str, float]] = None,
                use_cache: bool = True) -> np.ndarray:
    """Native-grid (science-area) convolved template C of ``variant`` (cached as ``final/cache/C_<variant>.npy``)."""
    kfile, weighted = VARIANTS[variant]
    P = chain_paths(cfg)
    cache = P.final / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"C_{variant}.npy"
    if use_cache and p.exists():
        return np.load(p)
    grid = TK.load_grid(P.mapping_dir)
    sw = dict(store_w) if store_w is not None else store_weights(cfg)
    if not weighted:
        from .perband.paths import combined_store_weights_mode
        if combined_store_weights_mode(cfg) != "production":
            log.warning("variant %s uses the templates as built (store weights %s, not production); "
                        "use the _w variants for the adopted-weight pipeline", variant, sw)
    sc = band_scales(cfg, weighted, sw)
    T = {b: np.load(P.band_templates / f"T_{b}.npy") * float(k) for b, k in zip(BANDS, sc)}
    M0 = sum(T.values())
    kz = np.load(P.kernels / kfile, allow_pickle=False)
    geo = np.load(P.kernels / "band_epsf.npz", allow_pickle=False)
    nx, ny = geo["node_x"], geo["node_y"]
    t0 = time.time()
    if variant.startswith("band"):
        dl = delta_new(cfg, kz) if weighted else kz["delta_b"]
        print(variant, "delta_b", np.round(dl, 4), "scale_b", np.round(sc, 6), flush=True)
        M1 = sum(float(d) * T[b] for d, b in zip(dl, BANDS))
        conv = TK.convolve_blended(M0, kz["K0"], grid, nx, ny, workers=workers)
        conv += TK.convolve_blended(M1, kz["K1"], grid, nx, ny, workers=workers)
    else:
        conv = TK.convolve_blended(M0, kz["K"], grid, nx, ny, workers=workers)
    F = int(grid.oversampling)
    nat = TK.trim(TK.block_sum(conv, F), grid)
    Tn = TK.trim(TK.block_sum(M0, F), grid)
    np.save(p, nat)
    (cache / f"C_{variant}.json").write_text(json.dumps(dict(
        seconds=time.time() - t0, flux_ratio=float(nat.sum() / Tn.sum()),
        store_band_weights=sw, scale_b=sc.tolist(), weighted=weighted)))
    print(variant, "convolved", time.time() - t0, float(nat.sum() / Tn.sum()), flush=True)
    return nat


def scene_stars(cfg):
    """The fit scene's stars as the Tmag<13 exclusion table (science-local ``cx, cy``). Added Gaia neighbours
    (``is_added``, chain/neighbours.py) are not part of it, so the final image is the same with or without them."""
    import pandas as pd
    z = np.load(chain_paths(cfg).scene_dir / "scene_bundle.npz", allow_pickle=True)
    keep = ~z["is_added"] if "is_added" in z.files else np.ones(len(z["source_id"]), bool)
    return pd.DataFrame(dict(source_id=z["source_id"][keep], tmag=z["tess_mag"][keep], x=z["cx"][keep], y=z["cy"][keep]))


def final_image(target, C, noise, mask, stars, ko: int = 2, out_dir: Optional[Path] = None):
    """a, b match + difference: ``D = target - (a C + b)``.  Returns (D, fit result)."""
    res, a, _keep = TK.run_match(target, C, noise, mask, ko, stars, out_dir=out_dir)
    return target - (a * C + res["b"]), res


def run(cfg, variants: Optional[Sequence[str]] = None, workers: int = 8, force: bool = False) -> Path:
    """Stage ``final``: needs ``perband`` (band templates), ``kernels`` and ``hotpants`` done."""
    from astropy.io import fits
    P = chain_paths(cfg)
    stage = cfg.stage_dir("final")
    variants = list(variants) if variants else list(VARIANTS)
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant {v!r}; known: {list(VARIANTS)}")
    for dep in ("kernels", "hotpants"):
        if not is_done(cfg.stage_dir(dep)):
            raise FileNotFoundError(f"stage {dep} not done: {cfg.stage_dir(dep)}")
    stars = scene_stars(cfg)
    summary = {}
    for stem in P.frames:
        target = TK.load_science(P.ffi(stem), P.background(stem))
        ref = P.hotpants_ref(stem)
        _, noise, mask = TK.load_hp_planes(ref)
        for v in variants:
            Cn = model_image(cfg, v, workers=workers, use_cache=not force)
            od = P.final / stem / v
            D, res = final_image(target, Cn, noise, mask, stars, 2, out_dir=od / "match")
            hdr = fits.Header()
            hdr["COMMENT"] = "forward_model.chain final: template-kernel difference image"
            hdr["VARIANT"] = v
            hdr["FRAME"] = stem
            hdr["FIELD"] = P.field
            hdr["KO"] = 2
            TK.write_fits(od / "hp_d" / f"{stem}_hp_d.fits.fz", hdr,
                          [(D.astype(np.float32), fits.Header()), (noise.astype(np.float32), fits.Header()),
                           (mask.astype(np.int32), fits.Header())])
            summary[f"{stem}/{v}"] = dict(chi2_red_keep=res["chi2_red_keep"], b=res["b"], a=res["a_stats"],
                                          hp_d=str(od / "hp_d" / f"{stem}_hp_d.fits.fz"), ref=str(ref))
            print(stem, v, json.dumps(summary[f"{stem}/{v}"]), flush=True)
    P.final.mkdir(parents=True, exist_ok=True)
    sj = P.final / "summary.json"
    old = json.loads(sj.read_text()) if sj.exists() else {}
    sj.write_text(json.dumps({**old, **summary}, indent=1))
    if all((P.final / s / v / "hp_d" / f"{s}_hp_d.fits.fz").is_file() for s in P.frames for v in VARIANTS):
        write_provenance(stage, cfg, {"band_templates": P.band_templates / "sum_check.json",
                                      "kernels": P.kernels / "band_epsf.npz", "adopted_weights": cfg.inputs.adopted_weights,
                                      "store_band_weights": store_weights(cfg)})
        mark_done(stage)
    return stage


def main(argv: Optional[list] = None) -> int:
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("variants", nargs="*", help=f"default: all of {list(VARIANTS)}")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="recompute the cached convolved templates")
    a = ap.parse_args(argv)
    run(load_config(a.config), a.variants or None, a.workers, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
