"""
Per-FFI frame manifest (WCS drift, template groups, pipeline step outcomes).

The default on-disk name is ``syndiff_ffi_frames.csv`` under ``output_dir``;
an alternate path may be set via config ``manifest``.

Step-status columns are added when stages run (hotpants / ePSF / labeled workspaces).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DEFAULT_MANIFEST_BASENAME, workspace_dir


def manifest_path_from_output_dir(output_dir: str, manifest_abspath: str | None = None) -> str:
    """Path to CSV; if *manifest_abspath* is None, use default under *output_dir*."""
    if manifest_abspath:
        return manifest_abspath
    return os.path.join(os.path.abspath(output_dir), DEFAULT_MANIFEST_BASENAME)


def load_frame_manifest(output_dir: str, manifest_path: str | None = None) -> pd.DataFrame:
    path = manifest_path_from_output_dir(output_dir, manifest_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing frame manifest {path!r} (expected {DEFAULT_MANIFEST_BASENAME!r} "
            f"under output_dir when manifest is unset)."
        )
    return pd.read_csv(path)


def save_frame_manifest(
    df: pd.DataFrame, output_dir: str, manifest_path: str | None = None
) -> str:
    path = manifest_path_from_output_dir(output_dir, manifest_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def manifest_csv_exists(output_dir: str, manifest_path: str | None = None) -> bool:
    return os.path.isfile(manifest_path_from_output_dir(output_dir, manifest_path))


def _ensure_hotpants_columns(df: pd.DataFrame, round_id: int) -> pd.DataFrame:
    df = df.copy()
    cols = (
        (f"hotpants_r{round_id}_ok", pd.NA),
        (f"hotpants_r{round_id}_error", ""),
        (f"diff_r{round_id}_path", ""),
    )
    for c, init in cols:
        if c not in df.columns:
            df[c] = init
    return df


def _ensure_hotpants_label_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    safe = str(label).replace(" ", "_")
    for c, init in (
        (f"hotpants_{safe}_ok", pd.NA),
        (f"hotpants_{safe}_error", ""),
        (f"diff_{safe}_path", ""),
    ):
        if c not in df.columns:
            df[c] = init
    return df


def _ensure_epsf_columns(df: pd.DataFrame, round_id: int) -> pd.DataFrame:
    df = df.copy()
    for c, init in (
        (f"epsf_r{round_id}_ok", pd.NA),
        (f"epsf_r{round_id}_error", ""),
    ):
        if c not in df.columns:
            df[c] = init
    return df


def manifest_has_hotpants_status(
    output_dir: str, round_id: int, manifest_path: str | None = None
) -> bool:
    path = manifest_path_from_output_dir(output_dir, manifest_path)
    if not os.path.isfile(path):
        return False
    df = pd.read_csv(path, nrows=0)
    return f"hotpants_r{round_id}_ok" in df.columns


def manifest_has_hotpants_label(
    output_dir: str, label: str, manifest_path: str | None = None
) -> bool:
    path = manifest_path_from_output_dir(output_dir, manifest_path)
    if not os.path.isfile(path):
        return False
    safe = str(label).replace(" ", "_")
    df = pd.read_csv(path, nrows=0)
    return f"hotpants_{safe}_ok" in df.columns


def row_stem_series(df: pd.DataFrame) -> pd.Series:
    if "filename" in df.columns:
        return df["filename"].map(lambda f: Path(str(f)).stem)
    return df["path"].map(lambda p: Path(str(p)).stem)


def apply_hotpants_results(
    df: pd.DataFrame,
    ffi_paths: list,
    results: list,
    round_id: int,
) -> pd.DataFrame:
    df = _ensure_hotpants_columns(df, round_id)
    ok_col = f"hotpants_r{round_id}_ok"
    err_col = f"hotpants_r{round_id}_error"
    path_col = f"diff_r{round_id}_path"
    stems_index = row_stem_series(df)

    for ffi_path, res in zip(ffi_paths, results):
        stem = Path(ffi_path).stem
        m = stems_index == stem
        if not m.any():
            continue
        success = bool(res.get("success", False))
        msg = res.get("error_msg", "") or ""
        df.loc[m, ok_col] = success
        df.loc[m, err_col] = str(msg) if msg else ""
        if success and res.get("path"):
            df.loc[m, path_col] = str(res["path"])
        else:
            df.loc[m, path_col] = ""
    return df


def apply_hotpants_workspace_results(
    df: pd.DataFrame,
    ffi_paths: list,
    results: list,
    label: str,
) -> pd.DataFrame:
    """Update manifest columns for a labeled Hotpants workspace (``diff_<label>_path``)."""
    df = _ensure_hotpants_label_columns(df, label)
    safe = str(label).replace(" ", "_")
    ok_col = f"hotpants_{safe}_ok"
    err_col = f"hotpants_{safe}_error"
    path_col = f"diff_{safe}_path"
    stems_index = row_stem_series(df)

    for ffi_path, res in zip(ffi_paths, results):
        stem = Path(ffi_path).stem
        m = stems_index == stem
        if not m.any():
            continue
        success = bool(res.get("success", False))
        msg = res.get("error_msg", "") or ""
        df.loc[m, ok_col] = success
        df.loc[m, err_col] = str(msg) if msg else ""
        if success and res.get("path"):
            df.loc[m, path_col] = str(res["path"])
        else:
            df.loc[m, path_col] = ""
    return df


def apply_epsf_status(
    df: pd.DataFrame,
    ffi_stems: list,
    epsf_ok: list,
    round_id: int,
) -> pd.DataFrame:
    df = _ensure_epsf_columns(df, round_id)
    ok_col = f"epsf_r{round_id}_ok"
    err_col = f"epsf_r{round_id}_error"
    stems_index = row_stem_series(df)

    for stem, ok in zip(ffi_stems, epsf_ok):
        m = stems_index == str(stem)
        if not m.any():
            continue
        ok_b = bool(ok)
        df.loc[m, ok_col] = ok_b
        df.loc[m, err_col] = "" if ok_b else "epsf_fit_failed_or_missing_diff"
    return df


def epsf_row_indices_for_group(
    df: pd.DataFrame,
    ffi_stems: np.ndarray | list,
    group_id: int,
) -> np.ndarray:
    stems_index = row_stem_series(df)
    stem_to_epsf = {str(s): j for j, s in enumerate(ffi_stems)}
    rows = []
    for stem, gid in zip(stems_index, df["group_id"]):
        try:
            gid_i = int(gid) if pd.notna(gid) else -999
        except (TypeError, ValueError):
            gid_i = -999
        if gid_i != group_id:
            continue
        k = str(stem)
        if k in stem_to_epsf:
            rows.append(stem_to_epsf[k])
    return np.asarray(rows, dtype=np.int64)


def group_ids_from_ffi_stems(df: pd.DataFrame, ffi_stems: np.ndarray | list) -> np.ndarray:
    if "group_id" not in df.columns:
        raise ValueError("frame manifest table missing group_id")
    stems_index = row_stem_series(df)
    stem_to_gid = {}
    for i, s in enumerate(stems_index):
        sk = str(s)
        if sk in stem_to_gid:
            continue
        gid = df.iloc[i]["group_id"]
        try:
            gid_i = int(gid) if pd.notna(gid) else -1
        except (TypeError, ValueError):
            gid_i = -1
        stem_to_gid[sk] = gid_i

    out = []
    for s in ffi_stems:
        out.append(stem_to_gid.get(str(s), -1))
    return np.asarray(out, dtype=np.int64)


def ordered_photometry_diff_paths(
    df: pd.DataFrame,
    output_dir: str,
    source: str,
) -> list:
    src = source.lower().strip()
    if src not in ("final", "r1", "r2"):
        raise ValueError(
            f"source must be 'final', 'r1', or 'r2', got {source!r}"
        )
    subdir = {"final": "diff_final", "r1": "diff_r1", "r2": "diff_r2"}[src]
    path_col = f"diff_{src}_path"
    stems = row_stem_series(df).reset_index(drop=True)
    df_reset = df.reset_index(drop=True)
    use_col = path_col if path_col in df_reset.columns else None
    out: list = []
    for i in range(len(df_reset)):
        p = None
        if use_col is not None:
            cell = df_reset.iloc[i][path_col]
            if pd.notna(cell) and str(cell).strip():
                cand = str(cell).strip()
                if os.path.isfile(cand):
                    p = str(Path(cand).resolve())
        if p is None:
            stem = str(stems.iloc[i])
            cand = os.path.join(output_dir, subdir, f"{stem}.fits")
            if os.path.isfile(cand):
                p = str(Path(cand).resolve())
        out.append(p)
    return out


def ordered_diff_paths_for_workspace(
    df: pd.DataFrame,
    output_dir: str,
    label: str,
    manifest_path: str | None = None,
) -> list:
    """
    One FITS path per manifest row for a workspace label (``ws/<label>/``).

    Uses column ``diff_<label>_path`` when present and valid; otherwise
    ``{output_dir}/ws/{label}/{stem}.fits``.
    """
    safe = str(label).replace(" ", "_")
    path_col = f"diff_{safe}_path"
    stems = row_stem_series(df).reset_index(drop=True)
    df_reset = df.reset_index(drop=True)
    ws = workspace_dir(output_dir, label)
    use_col = path_col if path_col in df_reset.columns else None
    out: list = []
    for i in range(len(df_reset)):
        p = None
        if use_col is not None:
            cell = df_reset.iloc[i][path_col]
            if pd.notna(cell) and str(cell).strip():
                cand = str(cell).strip()
                if os.path.isfile(cand):
                    p = str(Path(cand).resolve())
        if p is None:
            stem = str(stems.iloc[i])
            cand = os.path.join(ws, f"{stem}.fits")
            if os.path.isfile(cand):
                p = str(Path(cand).resolve())
        out.append(p)
    return out


