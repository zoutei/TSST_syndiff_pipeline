"""Synthetic inputs shared by the chain tests (no /astro access)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

CHAIN_DIR = Path(__file__).resolve().parents[2] / "syndiff_pipeline/forward_model/chain"


def raw_config(tmp_path: Path, **over) -> dict:
    raw = {
        "field": "T1",
        "scc": {"sector": 24, "camera": 2, "ccd": 2},
        "stem": "tess2020120182919-s0024-2-2",
        "data_root": str(tmp_path / "data"),
        "out_root": str(tmp_path / "out"),
        "inputs": {"colour_file": str(tmp_path / "colour.csv")},
    }
    raw.update(over)
    return raw


def write_config(tmp_path: Path, **over) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw_config(tmp_path, **over)))
    return p


def make_scene(d: Path, n=6, stamp=5, hp_shape=(40, 40), seed=0, workspace: Path | None = None, mask_source="shared_mask.fits.fz (static only)"):
    """Tiny scene: n stars on a grid of centres inside a ``hp_shape`` CCD."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    cx = np.array([8, 14, 20, 26, 8, 14][:n])
    cy = np.array([8, 8, 8, 8, 20, 20][:n])
    npx = stamp * stamp
    z = dict(
        stamp=np.int64(stamp), cx=cx, cy=cy,
        finite=np.ones((n, npx), bool), valid=np.ones((n, npx), bool),
        data=rng.normal(size=(n, npx)).astype(np.float32), noise=np.ones((n, npx), np.float32),
        role=np.array([0, 1, 1, 1, 1, 2][:n], np.int8),
        source_id=np.arange(1000, 1000 + n, dtype=np.int64),
        tess_mag=np.linspace(8, 12.5, n), star_bundle_index=np.arange(n),
    )
    np.savez(d / "scene_bundle.npz", **z)
    meta = dict(scene_version=1, source_bundle=str(d / "bundle.npz"), workspace=str(workspace) if workspace else "",
                frame_btjd=1969.0, stamp=stamp, core_radius=1.0, min_core_valid=3, masked_bits=[1, 8],
                straps_masked=False, mask_source=mask_source,
                n_roles={"contrib": 1, "anchor": 4, "nuisance": 1})
    (d / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    return z


def make_hp_d(path: Path, shape=(40, 40), value=3.0, noise=2.0):
    from astropy.io import fits
    h = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.full(shape, value, np.float32)),
                      fits.ImageHDU(np.full(shape, noise, np.float32))])
    path.parent.mkdir(parents=True, exist_ok=True)
    h.writeto(path, overwrite=True)
    return path
