"""Verify the in-tree ``template_calibration: zpcell`` reduce against run4 and against the out-of-tree reference.

For each field F: (a) reduce run4's contribs with an all-ones factor file and require T_{r,i,z,y}, T_sum, T_prodpath and
count to be bitwise equal to run4's band_templates; (b) reduce with the real ``factors_{F}_hyb.json`` and compare the
float32-cast T_b with ``Tprime/{F}/zpcell_hyb/T_{b}.npy`` (bandw_kernel_shift_20261009).  Writes a JSON report.

Usage: python tools/template_zp/verify_port.py [--fields F1 F2 ...] [--out DIR] [--report JSON]
Scratch outputs (big) go under --out, never /home.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

RUN4 = Path("/astro/armin/koji/syndiff/dev_runs/e2e_final_recipe_20261007/run4")
FACT = Path("/astro/armin/koji/syndiff/dev_runs/template_zp_factors_v1")
REF = Path("/astro/armin/koji/syndiff/dev_runs/bandw_kernel_shift_20261009/template_zp/Tprime")
OUT = Path("/astro/armin/koji/syndiff/dev_runs/template_zp_port_20261013")
FIELDS = ("F1", "F2", "S22", "C1", "C4")
BANDS = ("r", "i", "z", "y")


def run4_cfg(field: str, out_root: Path, cal: dict | None):
    """ChainConfig of run4's perband stage (from its provenance) re-rooted at ``out_root``, optional calibration section."""
    from syndiff_pipeline.forward_model.chain import config as C
    raw = json.loads((RUN4 / field / "perband/provenance.json").read_text())["config"]
    raw = dict(raw, out_root=str(out_root))
    raw.pop("template_calibration", None)
    if cal:
        raw["template_calibration"] = cal
    return C.config_from_dict(raw, config_path=out_root / "cfg.yaml")


def stage_inputs(field: str, out_root: Path) -> None:
    """Symlink run4's cells.json, contrib and mapping into ``out_root``; band_templates is written fresh."""
    (out_root / "perband").mkdir(parents=True, exist_ok=True)
    for dst, src in ((out_root / "perband/cells.json", RUN4 / field / "perband/cells.json"),
                     (out_root / "perband/contrib", RUN4 / field / "perband/contrib"),
                     (out_root / "mapping", RUN4 / field / "mapping")):
        if not dst.is_symlink() and not dst.exists():
            dst.symlink_to(src)


def ones_factors(field: str, dst: Path) -> Path:
    doc = json.loads((FACT / f"factors_{field}_hyb.json").read_text())
    doc["zpcell"] = {c: {b: 1.0 for b in BANDS} for c in doc["zpcell"]}
    dst.write_text(json.dumps(doc))
    return dst


def reduce_field(field: str, out_root: Path, factors: Path) -> dict:
    from syndiff_pipeline.forward_model.chain.perband import f04_reduce
    stage_inputs(field, out_root)
    cfg = run4_cfg(field, out_root, dict(mode="zpcell", factors=str(factors)))
    t0 = time.time()
    f04_reduce.run(cfg, make_figure=False)
    return dict(seconds=round(time.time() - t0, 1), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b)
    m = np.abs(b) > 0
    return dict(bitwise=bool(np.array_equal(a, b, equal_nan=True)), max_abs=float(np.nanmax(d)),
                max_rel=float(np.nanmax(d[m] / np.abs(b[m]))) if m.any() else 0.0)


def verify(field: str, out: Path, keep: bool = False) -> dict:
    res = {}
    d = out / field
    d.mkdir(parents=True, exist_ok=True)
    # (a) all-ones factors vs run4 band_templates, bitwise
    o1 = d / "ones"
    res["ones_run"] = reduce_field(field, o1, ones_factors(field, d / "factors_ones.json"))
    bt = o1 / "perband/band_templates"
    res["ones_vs_run4"] = {k: _diff(np.load(bt / f"{k}.npy"), np.load(RUN4 / field / f"perband/band_templates/{k}.npy"))
                           for k in ("T_r", "T_i", "T_z", "T_y", "T_sum", "T_prodpath", "count")}
    # (b) real factors vs the out-of-tree reference (float32)
    o2 = d / "real"
    res["real_run"] = reduce_field(field, o2, FACT / f"factors_{field}_hyb.json")
    bt = o2 / "perband/band_templates"
    res["real_vs_reference_f32"] = {b: _diff(np.load(bt / f"T_{b}.npy").astype(np.float32), np.load(REF / field / f"zpcell_hyb/T_{b}.npy"))
                                    for b in BANDS}
    res["sum_check_template_calibration"] = json.loads((bt / "sum_check.json").read_text())["template_calibration"]
    if not keep:
        for o in (o1, o2):
            for f in (o / "perband/band_templates").glob("*.npy"):
                f.unlink()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs="*", default=list(FIELDS))
    ap.add_argument("--out", type=Path, default=OUT / "scratch")
    ap.add_argument("--report", type=Path, default=OUT / "verify_port.json")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    rep = {}
    if a.report.exists():
        rep = json.loads(a.report.read_text())
    for f in a.fields:
        rep[f] = verify(f, a.out, keep=a.keep)
        a.report.write_text(json.dumps(rep, indent=1))
        print(f, json.dumps({k: v for k, v in rep[f].items() if k != "sum_check_template_calibration"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
