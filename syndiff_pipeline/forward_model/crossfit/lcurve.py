# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Out-of-fold learning curve: score every saved checkpoint of one fit (model compiled once).

    python -m syndiff_pipeline.forward_model.crossfit.lcurve --fit-out <fit dir> --base-scene <scene> --folds <folds.npz> \
        --held-fold k --out lcurve.json [--every 1]

Checkpoints: <fit>/params_stage1.npz (= stage-2 step 0) and <fit>/checkpoints/params_s{2,3}_step*.npz, then
params_stage2/3.npz. For each: median core chi^2 of out-of-fold trainees (fold == k) and in-fold trainees per Tmag bin
(plain medians, fresh state on the unmodified scene; see score_oof). For an all-stars fit pass --held-fold k anyway:
"oof" is then just the fold-k stars (in-sample), a matched comparison for the fold fits.
"""
from __future__ import annotations

import jax  # noqa: F401
import jax.numpy as jnp

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .. import fit as FIT
from .. import scene_fit as SF
from .score_oof import build, core_chi2

TB = ((8, 9), (9, 10), (10, 11), (11, 12), (12, 13))


def checkpoints(fit_out: Path):
    out = []
    if (fit_out / "params_stage1.npz").exists():
        out.append((2, 0, fit_out / "params_stage1.npz"))
    for p in sorted((fit_out / "checkpoints").glob("params_s*_step*.npz")):
        m = re.match(r"params_s(\d)_step(\d+)\.npz", p.name)
        if m and int(m.group(1)) >= 2:
            out.append((int(m.group(1)), int(m.group(2)), p))
    for s in (2, 3):
        if (fit_out / f"params_stage{s}.npz").exists():
            out.append((s, 10 ** 6, fit_out / f"params_stage{s}.npz"))   # stage end (step unknown here)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-out", required=True)
    ap.add_argument("--base-scene", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--held-fold", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1, help="use every n-th checkpoint")
    a = ap.parse_args(argv)
    fo = Path(a.fit_out)
    meta, scene, diag, _, _ = build(fo, Path(a.base_scene))
    N = scene.N
    fz = np.load(a.folds)
    assert (fz["source_id"] == scene.z["source_id"]).all()
    trainee = scene.role != SF.ROLE_NUISANCE
    oof = trainee & (fz["fold"] == a.held_fold)
    inf = trainee & (fz["fold"] != a.held_fold)
    T = scene.z["tess_mag"].astype(float)
    pix = np.ones(scene.U + 1, np.float32)
    pix[scene.U] = 0.0
    st0 = {"pix_active_u": jnp.asarray(pix), "star_free": jnp.ones((N,), jnp.float32),
           "f_fixed": jnp.zeros((N,), jnp.float32), "f_prior": jnp.zeros((N,), jnp.float32)}
    hist = [json.loads(l) for l in open(fo / "history.jsonl") if '"loss"' in l]
    stage_len = {s: max([h["step"] for h in hist if h["stage"] == s], default=-1) + 1 for s in (1, 2, 3)}
    cps = checkpoints(fo)[:: a.every]
    prev = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else []
    done = {(r["stage"], r["step"]) for r in prev}
    rows = list(prev)
    for stage, step, p in cps:
        if step == 10 ** 6:
            step = stage_len.get(stage, 0)
        if (stage, step) in done:
            continue
        params = FIT.load_params_npz(p)
        extras = SF.g8_extras_tuple(meta.get("chroma_g8_extras", "") or "")
        if "chroma_g8" in params:
            params = SF.set_chroma_model(params, "global8", halo=False, g8_init=None, g8_extras=extras,
                                         g8_source_extras=extras)
        f0, _, _ = diag(params, st0)
        st, _ = SF.update_flux_prior(f0, st0, scene)
        f, _, chi = diag(params, st)
        c2, n = core_chi2(scene, np.asarray(chi, np.float64))
        ok = (np.asarray(f) > 0) & (n >= 15)
        r = dict(stage=stage, step=step, epsf_step=(0 if stage == 2 else stage_len.get(2, 0)) + step,
                 oof={f"{lo}-{hi}": float(np.median(c2[ok & oof & (T >= lo) & (T < hi)])) for lo, hi in TB},
                 inf={f"{lo}-{hi}": float(np.median(c2[ok & inf & (T >= lo) & (T < hi)])) for lo, hi in TB},
                 n_oof=int((ok & oof).sum()), n_inf=int((ok & inf).sum()), params=str(p))
        rows.append(r)
        rows.sort(key=lambda r: (r["stage"], r["step"]))
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(stage, step, r["oof"], flush=True)


if __name__ == "__main__":
    main()
