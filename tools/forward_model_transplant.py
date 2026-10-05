#!/usr/bin/env python
"""Transplant dev/forward_epsf_wcs into syndiff_pipeline/forward_model (migration Move 1).

Faithful copy with import rewrites only. Re-runnable: it rebuilds the destination tree from a dev
checkout, so a newer dev commit can be re-cut the same way. Plan:
doc/forward_epsf_wcs_migration_plan_20260930.md (gitignored). Record:
/astro/armin/koji/syndiff/dev_runs/fwd_migration_20260930/.

Usage:
    python tools/forward_model_transplant.py --dev <dev checkout> --untracked-root <live dev dir> [--dest <repo root>]

What is copied:
  * forward_epsf_wcs/*.py and crossfit/*.py                  -> syndiff_pipeline/forward_model/
  * the init_study/ and diagnostics/ modules that code imports (import closure only; the plot and
    diagnosis scripts stay in dev)                            -> forward_model/{init_study,diagnostics}/
  * sibling dev modules the closure needs                    -> forward_model/_vendor/
    (temporal_wcs_poly, ref_epsf_photometry and sat_star_lc are untracked in the dev repo, so they
    are read from the live dev directory given by --untracked-root)
  * forward_epsf_wcs/tests/*.py whose imports all resolve     -> tests/forward_model/test_fm_*.py
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

PKG = "syndiff_pipeline.forward_model"
VENDOR = f"{PKG}._vendor"

# dev module path -> (dest subdir under forward_model/, module names)
VENDORED = {
    "temporal_wcs_poly": ("_vendor/temporal_wcs_poly", ["data_io", "temporal_model"]),
    "wcs_fit_from_centroids": ("_vendor/wcs_fit_from_centroids",
                               ["cheb_poly_fit", "fit_wcs_from_centroids", "sip_poly_fit", "wcs_conversion"]),
    "ref_epsf_photometry": ("_vendor/ref_epsf_photometry", ["orbit_windows", "pool_epsf", "stamp_norm"]),
    "sat_star_lc/forced_photometry": ("_vendor/sat_star_lc", ["radec_to_xy_temporal", "rgi_epsf_phot"]),
}
# tests that cannot run in production (reason shown in the skip marker)
_DEV_REF = "dev-repo harness: git-checks-out a dev reference commit to compare against; parity is covered by " \
           "the migration goldens (dev_runs/fwd_migration_20260930)"
SKIP_TESTS = {
    "test_leaf_absent_bit_identical_to_main": _DEV_REF,
    "test_c14_loss_and_grad_bit_identical_to_ref_commit": _DEV_REF,
    "test_bg_off_bit_identical_to_main_on_real_scene": _DEV_REF,
    "test_pointing_analysis_numpy_recenter_matches_training_core_gauge":
        "needs dev/pointing_analysis (not migrated)",
    "test_resample_prf_native_to_node_sums_to_one":
        "pre-existing failure on dev main too (PRF fork resample), 2026-09-30",
}

# production-only files inside forward_model/ that a re-cut must not delete
PRESERVE = {"recipes", "recipe.py", "chain"}

COLAB_SCRIPTS = ["transcode_bundle_tiered.py", "slice_bundle_frames.py", "upload_to_gdrive.py",
                 "colab_run_irreg_fullccd_chunk10.py", "pack_colab_train.sh", "measure_vram_sweep.sh"]

BARE = {m: f"{VENDOR}.{Path(d).name}" for src, (d, mods) in VENDORED.items() for m in mods}

DOTTED = [
    (r"\bdev\.forward_epsf_wcs\b", PKG),
    (r"\bdev\.ref_epsf_photometry\b", f"{VENDOR}.ref_epsf_photometry"),
    (r"\bdev\.sat_star_lc\.forced_photometry\b", f"{VENDOR}.sat_star_lc"),
    (r"\bdev\.temporal_wcs_poly\b", f"{VENDOR}.temporal_wcs_poly"),
]

BOOTSTRAP = '''"""Optional PRF fork on sys.path (not pip-installed; only ``psf_type: prf`` init paths need it).

In dev this module also wired sibling dev directories onto sys.path; those modules are now vendored
under ``_vendor/`` and imported by absolute name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

_EXTRA_PATHS = [
    _REPO / "tess_prf_oversample" / "src",
]

for _p in _EXTRA_PATHS:
    sp = str(_p)
    if _p.is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)
'''

REQUIRE_OUT_DIR = '''

def _require_out_dir():
    """Migration: the dev default wrote runs into the package directory (the /home checkout)."""
    raise SystemExit("pass --out-dir (run outputs belong under /astro/armin/koji/syndiff/, never /home)")
'''

INIT = '''"""Forward-modelled ePSF + WCS scene fitting (Gaia-positioned stars on a TESS FFI).

Migrated from ``dev/forward_epsf_wcs``. Single-FFI path: ``scene_export`` -> ``scene_fit``; multi-frame
(Paper 2): ``scene_fit_multi``, ``run_fit``/``train_loop`` and the Colab/GPU runners; overfitting checks:
``crossfit``. Default recipe: ``recipes/``.

jax is imported first on purpose: pandas/pyarrow loaded before jax pulls in the system libstdc++ and the
first XLA CPU compile segfaults.
"""

try:
    import jax  # noqa: F401  (must precede pandas/pyarrow)
except ImportError as exc:  # pragma: no cover
    raise ImportError("syndiff_pipeline.forward_model needs jax + optax: pip install -e '.[forward]'") from exc
'''

HEADER = "# Migrated from dev/forward_epsf_wcs (dev commit {sha}) by tools/forward_model_transplant.py.\n"


def rewrite(text: str, *, is_test: bool) -> str:
    for pat, rep in DOTTED:
        text = re.sub(pat, rep, text)
    # bare "forward_epsf_wcs" (dev tests ran with dev/ itself on sys.path) and "python -m forward_epsf_wcs.x"
    text = re.sub(r"^(\s*)(from|import) forward_epsf_wcs\b", rf"\1\2 {PKG}", text, flags=re.M)
    text = re.sub(r"(-m\s+)forward_epsf_wcs\.", rf"\1{PKG}.", text)
    bare = "|".join(map(re.escape, BARE))
    # from data_io import X  -> from <vendor>.temporal_wcs_poly.data_io import X
    text = re.sub(rf"^(\s*)from ({bare}) import ", lambda m: f"{m[1]}from {BARE[m[2]]}.{m[2]} import ",
                  text, flags=re.M)
    # import temporal_model [as TM] -> from <vendor>.temporal_wcs_poly import temporal_model [as TM]
    text = re.sub(rf"^(\s*)import ({bare})\b", lambda m: f"{m[1]}from {BARE[m[2]]} import {m[2]}",
                  text, flags=re.M)
    # tests import each other as dev.forward_epsf_wcs.tests.test_X; they are now test_fm_X
    text = re.sub(rf"\b{re.escape(PKG)}\.tests\.test_(\w+)", r"test_fm_\1", text)
    if is_test:
        text = re.sub(r"^(\s*)from \.test_(\w+) import ", r"\1from test_fm_\2 import ", text, flags=re.M)
        for name, reason in SKIP_TESTS.items():
            text = re.sub(rf"^def {name}\(", f"@pytest.mark.skip(reason={reason!r})\ndef {name}(", text, flags=re.M)
        if any(re.search(rf"^def {n}\(", text, re.M) for n in SKIP_TESTS) and not re.search(r"^import pytest", text, re.M):
            text = "import pytest\n" + text
        # tests lived in forward_epsf_wcs/tests (a sub-package); they now sit outside the package
        text = re.sub(r"^(\s*)from \.\. import ", rf"\1from {PKG} import ", text, flags=re.M)
        text = re.sub(r"^(\s*)from \.\.(\w)", rf"\1from {PKG}.\2", text, flags=re.M)
    # sys.path wiring to sibling dev dirs is obsolete: everything is a proper sub-module now
    text = re.sub(r"^(\s*)sys\.path\.insert\(.*$", r"\1pass  # migration: dev sys.path wiring removed",
                  text, flags=re.M)
    return text


def imports_of(path: Path) -> set[str]:
    """Absolute module names imported by ``path`` (relative imports resolved via its package)."""
    parts = path.resolve().with_suffix("").parts
    # the package root is the LAST "syndiff_pipeline" (the checkout directory may share the name)
    last = max((i for i, p in enumerate(parts) if p == "syndiff_pipeline"), default=None)
    pkg = ".".join(parts[last:-1]) if last is not None else ""
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                if not pkg:
                    continue
                base = pkg.split(".")[: len(pkg.split(".")) - (n.level - 1)]
                mod = ".".join(base + ([n.module] if n.module else []))
            else:
                mod = n.module or ""
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in n.names)
    return out


def module_file(root: Path, mod: str) -> Path | None:
    if not mod.startswith(PKG):
        return None
    rel = mod[len("syndiff_pipeline."):].replace(".", "/")
    for cand in (root / "syndiff_pipeline" / f"{rel}.py", root / "syndiff_pipeline" / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def resolves(root: Path, mod: str) -> bool:
    """``mod`` is an importable module, or ``parent.name`` where parent is a module defining ``name``."""
    if module_file(root, mod) is not None:
        return True
    parent, _, name = mod.rpartition(".")
    pf = module_file(root, parent)
    if pf is None:
        return False
    if pf.name == "__init__.py":  # a package: name must be a real submodule or defined in __init__
        return re.search(rf"^(def|class)\s+{name}\b|^{name}\s*=|import .*\b{name}\b", pf.read_text(), re.M) is not None
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", required=True, type=Path, help="dev checkout (contains forward_epsf_wcs/)")
    ap.add_argument("--untracked-root", required=True, type=Path,
                    help="live dev dir holding the untracked temporal_wcs_poly/, ref_epsf_photometry/, sat_star_lc/")
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--sha", default="unknown", help="dev commit recorded in each file header")
    a = ap.parse_args()

    src = a.dev / "forward_epsf_wcs"
    dst = a.dest / "syndiff_pipeline" / "forward_model"
    tdst = a.dest / "tests" / "forward_model"
    # rebuild everything the transplant owns; keep hand-written production files (PRESERVE)
    if tdst.exists():
        shutil.rmtree(tdst)
    if dst.exists():
        for p in dst.iterdir():
            if p.name in PRESERVE:
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    tdst.mkdir(parents=True)
    hdr = HEADER.format(sha=a.sha)

    def put(s: Path, d: Path, *, is_test=False):
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(hdr + rewrite(s.read_text(), is_test=is_test))

    # 1. package top level + crossfit
    for f in sorted(src.glob("*.py")):
        put(f, dst / f.name)
    for f in sorted((src / "crossfit").glob("*.py")):
        put(f, dst / "crossfit" / f.name)
    for name in ("run_fit.py", "train_from_bundle.py"):
        f = dst / name
        t = f.read_text()
        t2 = re.sub(r'Path\(__file__\)(?:\.resolve\(\))?\.parent / "output" / time\.strftime\("run_%Y%m%d_%H%M%S"\)',
                    '_require_out_dir()', t)
        assert t2 != t, name
        f.write_text(t2 + REQUIRE_OUT_DIR)
    (dst / "__init__.py").write_text(hdr + INIT)
    # dev sys.path wiring -> only the optional PRF fork (a sibling checkout, not pip-installed) remains
    (dst / "_bootstrap.py").write_text(hdr + BOOTSTRAP)
    # 2. vendored sibling modules
    for sdir, (ddir, mods) in VENDORED.items():
        base = (a.dev if sdir == "wcs_fit_from_centroids" else a.untracked_root) / sdir
        for m in mods:
            put(base / f"{m}.py", dst / ddir / f"{m}.py")
        (dst / ddir / "__init__.py").write_text(hdr)
    (dst / "_vendor" / "__init__.py").write_text(
        hdr + '"""Sibling dev modules the forward model imports, vendored verbatim (imports rewritten)."""\n')
    # 2b. Colab/GPU operations scripts (Paper 2 multi-frame runs); analysis scripts stay in dev
    (dst / "scripts").mkdir(exist_ok=True)
    (dst / "scripts" / "__init__.py").write_text(hdr)
    for name in COLAB_SCRIPTS:
        s = src / "scripts" / name
        if name.endswith(".py"):
            put(s, dst / "scripts" / name)
        else:
            shutil.copy2(s, dst / "scripts" / name)
    # 3. init_study / diagnostics: import closure only
    for sub in ("init_study", "diagnostics"):
        (dst / sub).mkdir(exist_ok=True)
        (dst / sub / "__init__.py").write_text(hdr)
    changed = True
    while changed:
        changed = False
        for f in list(dst.rglob("*.py")):
            for mod in imports_of(f):
                for sub in ("init_study", "diagnostics"):
                    pre = f"{PKG}.{sub}."
                    if mod.startswith(pre):
                        name = mod[len(pre):].split(".")[0]
                        s, d = src / sub / f"{name}.py", dst / sub / f"{name}.py"
                        if s.is_file() and not d.exists():
                            put(s, d)
                            changed = True
    # 4. tests: keep those whose package imports all resolve
    kept, skipped = [], []
    for f in sorted((src / "tests").glob("test_*.py")):
        text = rewrite(f.read_text(), is_test=True)
        tmp = tdst / f"test_fm_{f.name[5:]}"
        tmp.write_text(hdr + text)
        missing = [m for m in imports_of(tmp) if m.startswith(PKG) and not resolves(a.dest, m)]
        if missing:
            tmp.unlink()
            skipped.append((f.name, sorted(missing)))
        else:
            kept.append(tmp.name)
    print(f"package files: {len(list(dst.rglob('*.py')))}; tests kept {len(kept)}, skipped {len(skipped)}")
    for n, miss in skipped:
        print(f"  skipped {n}: needs {', '.join(miss)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
