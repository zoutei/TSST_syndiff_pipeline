# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Load init-study WCS coefficients for export_fit_bundle (skip centroids warmstart)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FIT_STARS_SNAPSHOT = "_fit_stars_export.parquet"
FIT_STARS_META = "_fit_stars_export_meta.json"


def load_init_wcs_coeff(
    path: Path,
    frames: list,
    wcs_tb,
    *,
    cheb_degree: int,
    btjd_atol: float = 1e-6,
    basis_rtol: float = 1e-5,
    basis_atol: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate ``wcs_coeff.npz`` from an init-study warmstart bundle."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"init WCS coeff file not found: {path}")

    z = np.load(path, allow_pickle=True)
    if "wcs_coeff" not in z.files:
        raise ValueError(f"{path} missing 'wcs_coeff' array")

    wcs_coeff = np.asarray(z["wcs_coeff"], dtype=np.float64)
    saved_basis = np.asarray(z["wcs_frame_basis"], dtype=np.float64)
    saved_btjd = np.asarray(z["btjd"], dtype=np.float64)
    saved_stems = [str(s) for s in np.asarray(z["stems"], dtype=object)]
    n_terms = int(np.asarray(z["n_terms"]).item() if np.ndim(z["n_terms"]) else z["n_terms"])
    saved_cheb = int(np.asarray(z["cheb_poly_degree"]).item()) if "cheb_poly_degree" in z.files else None

    exp_stems = [f.stem for f in frames]
    exp_btjd = np.asarray([f.btjd for f in frames], dtype=np.float64)
    exp_basis = np.asarray(wcs_tb.frame_basis, dtype=np.float64)

    if saved_stems != exp_stems:
        raise ValueError(
            f"{path} frame stems mismatch export window "
            f"(saved n={len(saved_stems)}, export n={len(exp_stems)}; "
            f"first saved={saved_stems[0]!r} export={exp_stems[0]!r})"
        )
    if saved_btjd.shape != exp_btjd.shape or not np.allclose(saved_btjd, exp_btjd, atol=btjd_atol):
        raise ValueError(f"{path} btjd vector does not match selected export frames")
    if saved_basis.shape != exp_basis.shape or not np.allclose(
        saved_basis, exp_basis, rtol=basis_rtol, atol=basis_atol,
    ):
        raise ValueError(
            f"{path} temporal frame_basis does not match export knot settings "
            f"(saved {saved_basis.shape}, export {exp_basis.shape})"
        )
    if saved_cheb is not None and saved_cheb != int(cheb_degree):
        raise ValueError(f"{path} cheb_poly_degree={saved_cheb} != export --cheb-degree={cheb_degree}")

    expected_rows = 2 * n_terms
    if wcs_coeff.shape != (expected_rows, wcs_tb.n_basis):
        raise ValueError(
            f"{path} wcs_coeff shape {wcs_coeff.shape} != "
            f"expected ({expected_rows}, {wcs_tb.n_basis})"
        )

    meta = {
        "path": str(path),
        "n_terms": n_terms,
        "n_basis": int(wcs_tb.n_basis),
        "n_frames": len(frames),
        "cheb_degree": int(cheb_degree),
    }
    return wcs_coeff.astype(np.float32), meta


def _snapshot_paths(study_dir: Path) -> tuple[Path, Path]:
    d = study_dir / "stamp_qa"
    d.mkdir(parents=True, exist_ok=True)
    return d / FIT_STARS_SNAPSHOT, d / FIT_STARS_META


def _snapshot_key(
    *,
    frame_stems: list[str],
    region,
    tess_mag: tuple[float, float],
    qc_min_frac: float,
    no_prefilter_qc: bool,
) -> dict[str, Any]:
    return {
        "frame_stems": list(frame_stems),
        "region": [int(region.x_min), int(region.y_min), int(region.x_max), int(region.y_max)],
        "tess_mag": [float(tess_mag[0]), float(tess_mag[1])],
        "qc_min_frac": float(qc_min_frac),
        "no_prefilter_qc": bool(no_prefilter_qc),
    }


def try_load_fit_stars_snapshot(
    study_dir: Path,
    *,
    frame_stems: list[str],
    region,
    tess_mag: tuple[float, float],
    qc_min_frac: float,
    no_prefilter_qc: bool,
) -> pd.DataFrame | None:
    """Return cached primary table when export settings match the saved snapshot."""
    snap_path, meta_path = _snapshot_paths(study_dir)
    if not snap_path.is_file() or not meta_path.is_file():
        return None
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    key = _snapshot_key(
        frame_stems=frame_stems,
        region=region,
        tess_mag=tess_mag,
        qc_min_frac=qc_min_frac,
        no_prefilter_qc=no_prefilter_qc,
    )
    if saved != key:
        return None
    df = pd.read_parquet(snap_path)
    if "source_id" not in df.columns:
        raise ValueError(f"{snap_path} missing source_id column")
    return df.reset_index(drop=True)


def save_fit_stars_snapshot(
    study_dir: Path,
    fit_stars: pd.DataFrame,
    *,
    frame_stems: list[str],
    region,
    tess_mag: tuple[float, float],
    qc_min_frac: float,
    no_prefilter_qc: bool,
) -> Path:
    snap_path, meta_path = _snapshot_paths(study_dir)
    fit_stars.to_parquet(snap_path, index=False)
    meta_path.write_text(
        json.dumps(
            _snapshot_key(
                frame_stems=frame_stems,
                region=region,
                tess_mag=tess_mag,
                qc_min_frac=qc_min_frac,
                no_prefilter_qc=no_prefilter_qc,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return snap_path
