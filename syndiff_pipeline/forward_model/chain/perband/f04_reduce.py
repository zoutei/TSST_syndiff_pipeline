"""f04: sum the sparse per-cell contributions into full OS4 band templates and check sum_b T_b == production.

Outputs (``perband/band_templates/``): ``T_{r,i,z,y}.npy``, ``T_prodpath.npy``, ``count.npy``, ``T_sum.npy`` (float64,
full OS4 grid, same orientation as the production template FLUX_SUM), ``sum_check.json``, ``store_weights.json`` and
``perband/figs/sum_check.png``.  T_b are the four band shares of the template: sum_b T_b should equal the production-path
sum up to float rounding, edge cells whose canonical blur used neighbours outside this SCC's list, and any seam-correction
mismatch.  There is no production FLUX_SUM for the new mapping, so the check is against the production-path sum
(``T_prodpath``, the binned production convolved cells).

``store_weights.json`` records the band weights the combined store carried for ALL band cells (read from each band
cell's ``check.store_band_weights``; cells written before they were recorded count as production).  It asserts that every
cell has the same set -- a mixed store is an error -- and that it equals the chain's assumption; ``final`` reads this file
to compute the adopted-weight rescale ``s_b = w'_b / w_b``.

Port of e2e ``f04_reduce.py`` (single mapping, label ``a3`` in the contrib keys kept).
"""
from __future__ import annotations

import json

import numpy as np

from .paths import BANDS, MODEL, chain_band_weights, chain_paths

KEYS = ("prod",) + BANDS


def collect_store_weights(P, cfg=None) -> dict:
    """Band weights of the combined store as recorded by f02 in every band cell of ``cells.json``."""
    from syndiff_pipeline.template_creation.processing import perband as PB
    from syndiff_pipeline.template_creation.processing.combined_store import DEFAULT_BAND_WEIGHTS
    from .paths import combined_store_weights_mode

    names = json.loads(P.cells_json.read_text())["all_band_cells"]
    legacy = {b: float(DEFAULT_BAND_WEIGHTS[b]) for b in BANDS}
    found, missing = None, []
    for n in names:
        p = P.band_cells / f"{n}.npz"
        if not p.exists():
            missing.append(n)
            continue
        chk = json.loads(str(np.load(p)["check"]))
        w = chk.get("store_band_weights") or legacy      # pre-recording cells = production store
        w = {b: float(w[b]) for b in BANDS}
        if found is None:
            found = w
        elif not PB.same_band_weights(found, w):
            raise ValueError(f"band cells carry different store weights: {n}: {w} vs {found}")
    if found is None:
        raise FileNotFoundError(f"no band cells under {P.band_cells}")
    out = dict(store_band_weights=found, n_cells_checked=len(names) - len(missing), missing=missing)
    if cfg is not None:
        expected = chain_band_weights(cfg)
        if not PB.same_band_weights(found, expected):
            raise ValueError(f"store weights {found} != chain-assumed {expected} "
                             f"(inputs.combined_store_weights={combined_store_weights_mode(cfg)})")
        out["chain_assumed"] = expected
    return out


def run(cfg, make_figure: bool = True) -> dict:
    from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master

    P = chain_paths(cfg)
    bt = P.band_templates
    bt.mkdir(parents=True, exist_ok=True)
    sw = collect_store_weights(P, cfg)
    P.store_weights_json.write_text(json.dumps(sw, indent=1))
    shp = tuple(load_mapping_grid_from_master(P.mapping_dir / P.master_name).array_shape_os())
    n = shp[0] * shp[1]
    acc = {k: np.zeros(n) for k in KEYS}
    cnt = np.zeros(n)
    cells = json.loads(P.cells_json.read_text())["cells"]
    checks, missing = {}, []
    for name in cells:
        p = P.contrib / f"{name}.npz"
        if not p.exists():
            missing.append(name)
            continue
        z = np.load(p)
        checks[name] = json.loads(str(z["check"]))
        for k in KEYS:
            if f"{MODEL}_pix_{k}" in z.files:
                np.add.at(acc[k], z[f"{MODEL}_pix_{k}"], z[f"{MODEL}_sum_{k}"])
        if f"{MODEL}_count" in z.files:
            np.add.at(cnt, z[f"{MODEL}_pix_prod"], z[f"{MODEL}_count"])
    acc = {k: v.reshape(shp) for k, v in acc.items()}
    cnt = cnt.reshape(shp)
    for b in BANDS:
        np.save(bt / f"T_{b}.npy", acc[b])
    np.save(bt / "T_prodpath.npy", acc["prod"])
    np.save(bt / "count.npy", cnt)
    np.save(bt / "T_sum.npy", sum(acc[b] for b in BANDS))   # achromatic template = sum of band templates (Hotpants baseline input)
    T = acc["prod"]      # no production FLUX_SUM on this mapping: check against the production-path sum
    S = sum(acc[b] for b in BANDS)
    pk = float(np.abs(T).max())
    dS, dP = np.abs(S - T), np.abs(acc["prod"] - T)
    cc = [v for v in checks.values() if "error" not in v]
    summ = dict(n_cells=len(cells), n_contrib=len(checks), missing=missing,
                errors={k: v["error"] for k, v in checks.items() if "error" in v},
                store_band_weights=sw["store_band_weights"],
                prodpath_vs_template_max_rel=float(dP.max() / pk),
                bandsum_vs_template_max_rel=float(dS.max() / pk),
                bandsum_vs_template_p999_rel=float(np.quantile(dS, 0.999) / pk),
                bandsum_vs_template_frac_gt1e5=float(np.mean(dS > 1e-5 * pk)),
                flux=dict(template=float(T.sum()), prodpath=float(acc["prod"].sum()), bandsum=float(S.sum()),
                          **{b: float(acc[b].sum()) for b in BANDS}),
                band_share={b: float(acc[b].sum() / S.sum()) for b in BANDS},
                cell_sum_check=dict(max_rel=max(v["max_rel_to_peak"] for v in cc) if cc else None,
                                    n_gt_1e5=sum(v["max_rel_to_peak"] > 1e-5 for v in cc),
                                    worst=sorted(((v["max_rel_to_peak"], k) for k, v in checks.items() if "error" not in v))[-15:]),
                n_xproj=sum(v.get("xproj", False) for v in cc))
    (bt / "sum_check.json").write_text(json.dumps(summ, indent=1))
    if make_figure:
        _figure(P, acc, dS, S, pk, cc, summ)
    print(json.dumps({k: v for k, v in summ.items() if k != "cell_sum_check"}, indent=1))
    print(json.dumps(summ["cell_sum_check"], indent=1))
    return summ


def _figure(P, acc, dS, S, pk, cc, summ) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = 16

    def blk(a):
        h, w = (a.shape[0] // F) * F, (a.shape[1] // F) * F
        return a[:h, :w].reshape(h // F, F, w // F, F).max(axis=(1, 3))
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    im = ax[0].imshow(np.log10(blk(dS) / pk + 1e-12), origin="lower", vmin=-9, vmax=-2, cmap="magma")
    plt.colorbar(im, ax=ax[0], label="log10 max|Σ_b T_b − T| / peak (16-subcell blocks)")
    ax[0].set_title("band sum vs production-path template")
    u = sum(acc[b] * lam for b, lam in zip(BANDS, (617, 752, 866, 962))) / np.where(S > 0, S, np.nan)
    im = ax[1].imshow(blk(np.nan_to_num(u, nan=0)), origin="lower", vmin=760, vmax=880, cmap="RdYlBu_r")
    plt.colorbar(im, ax=ax[1], label="template-weighted band wavelength [nm] (max in block)")
    ax[1].set_title("per-pixel band-mix u")
    vals = sorted(v["max_rel_to_peak"] for v in cc)
    ax[2].semilogy(vals, ".-")
    ax[2].set_xlabel("cell rank")
    ax[2].set_ylabel("per-cell max|Σ_b blurred_b − prod cell|/peak")
    ax[2].set_title(f"per-cell blur check ({len(cc)} cells, {summ['n_xproj']} seam-corrected)")
    fig.tight_layout()
    P.figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(P.figs / f"sum_check_{P.field}.png", dpi=110)
    plt.close(fig)
