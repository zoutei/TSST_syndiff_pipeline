"""Helpers for the Agent-B tests (perband / kernels / hotpants_ref / final / score): a config proxy that carries the
extra ``inputs.*`` keys these stages read (colour_map, xp_synth, scorer_dir, pass2_hp, combined_store_weights, skylist)
on top of a real ``ChainConfig`` loaded from a dict, plus the e2e F1 product locations for the slow parity tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.forward_model.chain import config as C

E2E = Path("/astro/armin/koji/syndiff/dev_runs/e2e_f1_20260930")
E2E_RUN = E2E / "run"
STEM = "tess2020120182919-s0024-2-2"
UNSEEN = "tess2020121015919-s0024-2-2"
COLOUR_FILE = "/astro/armin/koji/syndiff/dev_runs/chroma_v2_20260928/colour/F1_s24c2k2_u.csv"
ADOPTED = "/astro/armin/koji/syndiff/dev_runs/bandpass_rizy_20260929/weights_oof/ADOPTED_WEIGHTS.json"
F1_EXTRA = dict(
    colour_map=dict(summary_json="/astro/armin/koji/syndiff/dev_runs/chroma_v2_20260928/colour/scripts/f1f2_summary.json",
                    key="F1_s24c2k2"),
    xp_synth="/astro/armin/koji/syndiff/dev_runs/xp_dimension_20260925/xp_synth.csv",
    scorer_dir="/astro/armin/koji/syndiff/dev_runs/perband_3field_20260929/scorer",
    pass2_hp={STEM: str(E2E / "s12_bootstrap/diff_tvwcs/hp_d" / f"{STEM}_hp_d.fits.fz")},
)


class _View:
    """Attribute view of ``base`` with extra attributes (only for names the base object lacks)."""

    def __init__(self, base, **extra):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_extra", extra)

    def __getattr__(self, name):
        extra = object.__getattribute__(self, "_extra")
        if name in extra:
            return extra[name]
        return getattr(object.__getattribute__(self, "_base"), name)


def make_cfg(tmp_path: Path, extra_inputs: dict | None = None, **over):
    """A real ChainConfig (field T1, SCC 24/2/2) in ``tmp_path`` + extra ``inputs`` attributes."""
    raw = {
        "field": "T1", "scc": {"sector": 24, "camera": 2, "ccd": 2}, "stem": STEM, "unseen_stem": UNSEEN,
        "data_root": str(tmp_path / "data"), "out_root": str(tmp_path / "out"),
        "inputs": {"colour_file": str(tmp_path / "colour.csv"), "adopted_weights": str(tmp_path / "adopted.json")},
        "code": {"forward_model_root": str(Path(C.__file__).resolve().parents[3])},
    }
    raw.update(over)
    cfg = C.config_from_dict(raw, config_path=tmp_path / "cfg.yaml")
    if extra_inputs:
        return _View(cfg, inputs=_View(cfg.inputs, **extra_inputs))
    return cfg


def f1_cfg(tmp_path: Path, **over_inputs):
    """F1 config with out_root = tmp_path/F1 and the e2e extras; stage dirs are filled by the caller (symlinks)."""
    raw = {
        "field": "F1", "scc": {"sector": 24, "camera": 2, "ccd": 2}, "stem": STEM, "unseen_stem": UNSEEN,
        "data_root": "/astro/armin/koji/syndiff/data", "out_root": str(tmp_path / "F1"),
        "inputs": {"colour_file": COLOUR_FILE, "adopted_weights": ADOPTED},
        "code": {"forward_model_root": str(Path(C.__file__).resolve().parents[3])},
    }
    cfg = C.config_from_dict(raw, config_path=tmp_path / "F1.yaml")
    return _View(cfg, inputs=_View(cfg.inputs, **{**F1_EXTRA, **over_inputs}))


def need(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"{p} not available")


def link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(src)


def arrays_equal(a, b, keys=None, skip=()):
    """Assert every array key of two npz files is bitwise equal; returns the keys compared."""
    a, b = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    keys = keys or [k for k in a.files if k not in skip]
    for k in keys:
        x, y = a[k], b[k]
        if x.dtype.kind in "OUS":
            continue
        assert x.shape == y.shape and np.array_equal(x, y, equal_nan=True), k
    return keys
