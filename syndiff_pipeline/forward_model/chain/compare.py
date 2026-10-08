"""D14-style comparison of fits (e2e ``s8/compare_fits.py``, generalised from hard-coded FITS/PAIRS to arguments).

For each pair ``(test, ref, label)`` of scene_fit output dirs it reports, at the scene's stars:
WCS difference (Chebyshev coefficients -> per-star dx, dy), ePSF core-centroid difference, the effective position
(WCS + ePSF centroid at the star's node), second moments / size / ellipticity differences of ``epsf_base`` (Gaussian
window sigma 1.5 px), enclosed-flux difference, chroma-term differences, loss difference, and (if both fits carry them)
the brightness-width coefficient ``b`` (``bright_width`` leaf) and the constant background (``bg_coef``), plus the
solved-flux ratio by Tmag bin. Output: ``compare_fits.json`` / ``.md`` (the D14 table), ``fig_compare_fits.png``,
``fig_compare_epsf.png``.

Yardsticks (rehearsal, split-half) are just more pairs: pass any ``fits`` dict and ``pairs`` list. Missing fit dirs are
skipped. The fitter is deterministic on CPU, so a same-input rerun floor is exactly 0.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from .config import SCIENCE_ORIGIN_FFI, ChainConfig, is_done, mark_done, write_provenance

OS = 4
BINS = [(7, 9), (9, 10), (10, 11), (11, 12), (12, 13)]


def default_fits(cfg: ChainConfig) -> tuple[dict, list]:
    """boot = ``fit/``, refit = ``refit/`` (+ ``refit/warm`` if present)."""
    fits = {"boot": cfg.stage_dir("fit"), "refit": cfg.stage_dir("refit")}
    pairs = [("refit", "boot", "refit on FINAL image, same init + recipe (the D14 test)")]
    warm = cfg.stage_dir("refit") / "warm"
    if (warm / "params.npz").exists():
        fits["refit_warm"] = warm
        pairs.append(("refit_warm", "boot", "warm continuation on FINAL image from the calibration"))
    return fits, pairs


def _load(d):
    import numpy as np
    d = str(d)
    if not os.path.exists(f"{d}/params.npz"):
        return None
    p = dict(np.load(f"{d}/params.npz"))
    f = dict(np.load(f"{d}/flux_solved.npz")) if os.path.exists(f"{d}/flux_solved.npz") else None
    h = [json.loads(l) for l in open(f"{d}/history.jsonl")] if os.path.exists(f"{d}/history.jsonl") else []
    m = json.load(open(f"{d}/fit_meta.json")) if os.path.exists(f"{d}/fit_meta.json") else {}
    bwm = m.get("bright_width_model") or {}
    b = float(np.ravel(p["bright_width"])[0]) * float(bwm.get("leaf_unit", 1e-3)) if "bright_width" in p else 0.0
    # the smoothed stop rule appends event rows (stage_end, lr cuts) without a loss: use the last row that has one
    hl = [r for r in h if "loss" in r]
    return dict(p=p, f=f, loss=hl[-1]["loss"] if hl else np.nan, data_term=hl[-1].get("data_term", np.nan) if hl else np.nan,
                b=b, qref=float(bwm.get("q_ref", 0.0)))


def _moments(E, sig=1.5):
    import numpy as np
    G = E.shape[-1]
    c = (G - 1) / 2
    u = (np.arange(G) - c) / OS
    X, Y = np.meshgrid(u, u)
    W = np.exp(-(X ** 2 + Y ** 2) / (2 * sig ** 2))
    w = E * W
    s = w.sum((-1, -2))
    mx = (w * X).sum((-1, -2)) / s
    my = (w * Y).sum((-1, -2)) / s
    ixx = (w * (X - mx[..., None, None]) ** 2).sum((-1, -2)) / s
    iyy = (w * (Y - my[..., None, None]) ** 2).sum((-1, -2)) / s
    ixy = (w * (X - mx[..., None, None]) * (Y - my[..., None, None])).sum((-1, -2)) / s
    r = np.hypot(X, Y)
    ee3 = (E * (r <= 3)).sum((-1, -2)) / E.sum((-1, -2))
    return dict(mx=mx, my=my, T=ixx + iyy, e1=ixx - iyy, e2=2 * ixy, ee3=ee3)


def compare_fits(fits: Mapping[str, str | Path], pairs: Sequence[tuple[str, str, str]], scene_dir: str | Path,
                 out_dir: str | Path, source_bundle: str | Path | None = None) -> dict:
    """Run the comparison; returns the result dict (also written as ``compare_fits.json``)."""
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator as RGI

    scene_dir, out_dir = Path(scene_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if source_bundle is None:
        source_bundle = json.loads((scene_dir / "scene_meta.json").read_text())["source_bundle"]
    b = np.load(source_bundle)
    sc = np.load(scene_dir / "scene_bundle.npz")
    B = b["cheb_basis"][sc["star_bundle_index"]]
    NX = b["epsf_node_col_ccd"]
    NY = b["epsf_node_row_ccd"]
    sx = sc["cx"] + float(SCIENCE_ORIGIN_FFI[0])
    sy = sc["cy"].astype(float)  # node cols are CCD columns (science x + 44)

    def at_star(g):
        return RGI((NY, NX), g, bounds_error=False, fill_value=None)(np.c_[sy, sx])

    nt = B.shape[1]
    tm = sc["tess_mag"]
    role = sc["role"]
    F = {k: _load(v) for k, v in fits.items()}
    print({k: (v is not None) for k, v in F.items()})
    res = {}
    arrays = {}
    for t, r_, lab in pairs:
        a, c = F.get(t), F.get(r_)
        if a is None or c is None:
            continue
        dW = a["p"]["wcs_coeff"][:, 0] - c["p"]["wcs_coeff"][:, 0]
        dx = B @ dW[:nt]
        dy = B @ dW[nt:]
        ma, mc = _moments(a["p"]["epsf_base"]), _moments(c["p"]["epsf_base"])
        # effective image position = WCS position + ePSF core centroid at the star's node (bilinear skipped: node-mean)
        dmx = (ma["mx"] - mc["mx"])
        dmy = (ma["my"] - mc["my"])
        out = dict(
            label=lab,
            wcs_rms_mpx=1e3 * float(np.sqrt(np.mean(dx ** 2 + dy ** 2))), wcs_max_mpx=1e3 * float(np.max(np.hypot(dx, dy))),
            wcs_mean_dx_mpx=1e3 * float(dx.mean()), wcs_mean_dy_mpx=1e3 * float(dy.mean()),
            epsf_centroid_rms_mpx=1e3 * float(np.sqrt(np.mean(dmx ** 2 + dmy ** 2))),
            eff_pos_rms_mpx=1e3 * float(np.sqrt(np.mean((dx + at_star(dmx)) ** 2 + (dy + at_star(dmy)) ** 2))),
            eff_pos_max_mpx=1e3 * float(np.max(np.hypot(dx + at_star(dmx), dy + at_star(dmy)))),
            dT_1e3px2_median=1e3 * float(np.median(ma["T"] - mc["T"])),
            # effective ePSF size at fixed star brightness q (e- per 2-s read): base + isotropic b(q-q_ref)/1e4 per axis
            dTeff_q0_1e3px2=1e3 * float(np.median(ma["T"] - mc["T"]) + 2 * (a["b"] * (0 - a["qref"]) - c["b"] * (0 - c["qref"])) / 1e4),
            dTeff_q1e5_1e3px2=1e3 * float(np.median(ma["T"] - mc["T"]) + 2 * (a["b"] * (1e5 - a["qref"]) - c["b"] * (1e5 - c["qref"])) / 1e4),
            dT_1e3px2_rms=1e3 * float(np.sqrt(np.mean((ma["T"] - mc["T"]) ** 2))),
            de1_1e3_rms=1e3 * float(np.sqrt(np.mean((ma["e1"] - mc["e1"]) ** 2))),
            de2_1e3_rms=1e3 * float(np.sqrt(np.mean((ma["e2"] - mc["e2"]) ** 2))),
            dee3_ppt_median=1e3 * float(np.median(ma["ee3"] - mc["ee3"])),
            dchroma=(a["p"]["chroma_g8"] - c["p"]["chroma_g8"]).round(5).tolist(),
            dloss=float(a["loss"] - c["loss"]), ddata_term=float(a["data_term"] - c["data_term"]),
            db_bright=(float(np.ravel(a["p"]["bright_width"])[0] - np.ravel(c["p"]["bright_width"])[0]) * 1e-3
                       if "bright_width" in a["p"] and "bright_width" in c["p"] else None),
            b_bright=[float(np.ravel(x["p"]["bright_width"])[0]) * 1e-3 if "bright_width" in x["p"] else None for x in (a, c)],
            bg=[(float(np.ravel(x["f"]["bg_coef"])[0]) if x["f"] is not None and "bg_coef" in x["f"] else None) for x in (a, c)])
        rr = None
        if a["f"] is not None and c["f"] is not None:
            # fits trained on neighbour scenes (chain/neighbours.py) carry the added Gaia neighbours AFTER the original
            # scene stars (original rows unchanged), and boot/refit may add different numbers: compare the prefix
            n0 = len(role)
            for x in (a, c):
                if "source_id" in x["f"] and not np.array_equal(np.asarray(x["f"]["source_id"])[:n0], sc["source_id"]):
                    raise ValueError("flux_solved source_id prefix does not match the comparison scene")
            fa, fc = a["f"]["flux"][:n0], c["f"]["flux"][:n0]
            ok = (fc > 0) & (fa > 0) & np.isfinite(fa) & np.isfinite(fc) & (role != 2)
            rr = fa / fc - 1
            fb = {}
            for lo, hi in BINS:
                m = ok & (tm >= lo) & (tm < hi)
                x = rr[m]
                fb[f"{lo}-{hi}"] = dict(n=int(m.sum()), median_ppt=1e3 * float(np.median(x)),
                                        err_ppt=1e3 * float(1.2533 * 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(m.sum())),
                                        scatter_ppt=1e3 * float(1.4826 * np.median(np.abs(x - np.median(x)))))
            out["flux_ratio_by_T"] = fb
        res[f"{t} - {r_}"] = out
        arrays[f"{t} - {r_}"] = dict(dx=dx, dy=dy, dT=ma["T"] - mc["T"], de1=ma["e1"] - mc["e1"], rr=rr)
    (out_dir / "compare_fits.json").write_text(json.dumps(res, indent=1))
    # ---- table
    rows = []
    for k, v in res.items():
        fb = v.get("flux_ratio_by_T", {})
        rows.append(f"| {v['label']} (`{k}`) | {v['wcs_rms_mpx']:.2f} / {v['wcs_max_mpx']:.1f} | {v['eff_pos_rms_mpx']:.2f} / {v['eff_pos_max_mpx']:.1f} | {v['epsf_centroid_rms_mpx']:.2f} | {v['dT_1e3px2_median']:+.3f} ({v['dT_1e3px2_rms']:.3f}); eff q=0 {v['dTeff_q0_1e3px2']:+.3f}, q=1e5 {v['dTeff_q1e5_1e3px2']:+.3f}; b {v['b_bright'][0]}/{v['b_bright'][1]} | {v['de1_1e3_rms']:.3f} / {v['de2_1e3_rms']:.3f} | {v['dee3_ppt_median']:+.3f} | "
                    + " ".join(f"{fb[b_]['median_ppt']:+.2f}±{fb[b_]['err_ppt']:.2f}" for b_ in fb) + f" | {v['dloss']:+.4f} |")
    hdr = "| pair | WCS Δpos rms / max [mpx] | effective Δpos (WCS + ePSF centroid) rms / max [mpx] | ePSF centroid Δ rms [mpx] | ΔT base median (rms); effective at q=0 / 1e5 e-; b test/ref [1e-3 px²] | Δe1 / Δe2 rms [1e-3] | ΔEE(r≤3) [ppt] | flux ratio −1 by T 7-9/9-10/10-11/11-12/12-13 [ppt] | Δloss |\n|---|---|---|---|---|---|---|---|---|"
    (out_dir / "compare_fits.md").write_text(hdr + "\n" + "\n".join(rows) + "\n")
    print(hdr)
    print("\n".join(rows))
    if res:
        _figures(res, arrays, F, sc, tm, out_dir)
    return res


def _figures(res, arrays, F, sc, tm, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    ks = list(res)
    n = len(ks)
    fig, ax = plt.subplots(4, n, figsize=(3.3 * n, 13), squeeze=False)
    for j, k in enumerate(ks):
        A = arrays[k]
        ax[0, j].set_title(res[k]["label"], fontsize=8)
        ax[0, j].quiver(sc["cx"][::7], sc["cy"][::7], A["dx"][::7], A["dy"][::7], np.hypot(A["dx"], A["dy"])[::7] * 1e3, scale=0.05, cmap="viridis")
        ax[0, j].set_aspect("equal")
        ax[0, j].set_xticks([])
        ax[0, j].set_yticks([])
        ax[0, j].text(0.02, 0.02, f"rms {res[k]['wcs_rms_mpx']:.2f} mpx", transform=ax[0, j].transAxes, fontsize=8, color="k", bbox=dict(fc="w"))
        im = ax[1, j].imshow(1e3 * A["dT"], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax[1, j].set_title("ΔT per node [1e-3 px²]", fontsize=8)
        plt.colorbar(im, ax=ax[1, j], fraction=.046)
        im = ax[2, j].imshow(1e3 * A["de1"], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax[2, j].set_title("Δe1 per node [1e-3 px²]", fontsize=8)
        plt.colorbar(im, ax=ax[2, j], fraction=.046)
        if A["rr"] is not None:
            ax[3, j].scatter(tm, 1e3 * A["rr"], s=1, alpha=.2)
            ax[3, j].set_ylim(-10, 10)
            ax[3, j].axhline(0, c="k", lw=.5)
            fb = res[k]["flux_ratio_by_T"]
            ax[3, j].errorbar([np.mean([float(s) for s in b_.split("-")]) for b_ in fb], [fb[b_]["median_ppt"] for b_ in fb],
                              [fb[b_]["err_ppt"] for b_ in fb], fmt="ro-", ms=3)
            for y in (-1, 1):
                ax[3, j].axhline(y, c="r", ls=":", lw=.8)
            ax[3, j].set_xlabel("Tmag")
            ax[3, j].set_ylabel("flux ratio −1 [ppt]", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_compare_fits.png", dpi=90)
    plt.close(fig)
    # ePSF difference stacks (node-mean) for the pairs
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    for j, k in enumerate(ks):
        t, r_ = k.split(" - ")
        d = (F[t]["p"]["epsf_base"] - F[r_]["p"]["epsf_base"]).mean((0, 1))
        pk = F[r_]["p"]["epsf_base"].mean((0, 1)).max()
        im = ax[0, j].imshow(1e3 * d / pk, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, extent=[-7.9, 7.9, -7.9, 7.9])
        ax[0, j].set_title(res[k]["label"], fontsize=7)
        plt.colorbar(im, ax=ax[0, j], fraction=.046, label="ΔePSF / peak [ppt]")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_compare_epsf.png", dpi=90)
    plt.close(fig)


def run_compare(cfg: ChainConfig, extra_fits: Mapping[str, str] | None = None,
                extra_pairs: Sequence[tuple[str, str, str]] = (), force: bool = False) -> Path:
    """Stage ``compare``: ``refit`` vs ``fit`` (+ any extra fits/pairs, e.g. yardsticks)."""
    out = cfg.stage_dir("compare")
    if is_done(out) and not force:
        print(f"[compare] already done: {out}")
        return out
    fits, pairs = default_fits(cfg)
    fits.update(extra_fits or {})
    pairs = list(pairs) + list(extra_pairs)
    for dep in ("fit", "refit"):
        if not is_done(cfg.stage_dir(dep)):
            raise FileNotFoundError(f"stage {dep} not done: {cfg.stage_dir(dep)}")
    (out / "DONE").unlink(missing_ok=True)
    compare_fits(fits, pairs, cfg.stage_dir("scene_boot"), out)
    write_provenance(out, cfg, {**{f"fit:{k}": Path(v) / "params.npz" for k, v in fits.items() if (Path(v) / "params.npz").exists()},
                                "pairs": {"value": [list(p) for p in pairs]}})
    mark_done(out)
    return out
