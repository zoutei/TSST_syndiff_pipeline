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

from .ffi_naming import (
    parse_workspace_frame_stem,
    sanitize_workspace_label,
    tess_product_id_from_ffi_path,
    workspace_frame_stem,
)
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


def row_ffi_product_id_series(df: pd.DataFrame) -> pd.Series:
    """Series of ``tess<digits>`` product ids parsed from manifest ``filename``/``path``."""
    if "filename" in df.columns:
        return df["filename"].map(lambda f: tess_product_id_from_ffi_path(str(f)) or "")
    return df["path"].map(lambda p: tess_product_id_from_ffi_path(str(p)) or "")


def _result_product_id(res: dict) -> str:
    """Pull product id from a hotpants-shaped result dict (explicit field or stem)."""
    pid = res.get("ffi_product_id")
    if pid:
        return str(pid)
    stem = res.get("stem")
    if stem:
        parsed = parse_workspace_frame_stem(str(stem))
        if parsed is not None:
            return parsed[0]
        guess = tess_product_id_from_ffi_path(str(stem))
        if guess:
            return guess
    return ""


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
    pids_index = row_ffi_product_id_series(df)

    for ffi_path, res in zip(ffi_paths, results):
        pid = tess_product_id_from_ffi_path(str(ffi_path))
        if not pid:
            continue
        m = pids_index == pid
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
    safe = sanitize_workspace_label(label)
    ok_col = f"hotpants_{safe}_ok"
    err_col = f"hotpants_{safe}_error"
    path_col = f"diff_{safe}_path"
    pids_index = row_ffi_product_id_series(df)

    for ffi_path, res in zip(ffi_paths, results):
        pid = tess_product_id_from_ffi_path(str(ffi_path))
        if not pid:
            continue
        m = pids_index == pid
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


def _coerce_to_product_id(value: object) -> str:
    """Extract ``tess<digits>`` from a product id, workspace stem, path, or basename."""
    if value is None:
        return ""
    s = str(value)
    if not s:
        return ""
    parsed = parse_workspace_frame_stem(s)
    if parsed is not None:
        return parsed[0]
    pid = tess_product_id_from_ffi_path(s)
    return pid or ""


def apply_epsf_status(
    df: pd.DataFrame,
    ffi_product_ids: list,
    epsf_ok: list,
    round_id: int,
) -> pd.DataFrame:
    """Update epsf status columns by joining on ``tess<digits>`` product ids."""
    df = _ensure_epsf_columns(df, round_id)
    ok_col = f"epsf_r{round_id}_ok"
    err_col = f"epsf_r{round_id}_error"
    pids_index = row_ffi_product_id_series(df)

    for raw, ok in zip(ffi_product_ids, epsf_ok):
        pid = _coerce_to_product_id(raw)
        if not pid:
            continue
        m = pids_index == pid
        if not m.any():
            continue
        ok_b = bool(ok)
        df.loc[m, ok_col] = ok_b
        df.loc[m, err_col] = "" if ok_b else "epsf_fit_failed_or_missing_diff"
    return df


def epsf_row_indices_for_group(
    df: pd.DataFrame,
    ffi_product_ids: np.ndarray | list,
    group_id: int,
) -> np.ndarray:
    """Indices into *ffi_product_ids* for rows of *df* in *group_id* (matched by product id)."""
    pids_index = row_ffi_product_id_series(df)
    pid_to_epsf = {
        _coerce_to_product_id(s): j for j, s in enumerate(ffi_product_ids)
    }
    pid_to_epsf.pop("", None)
    rows = []
    for pid, gid in zip(pids_index, df["group_id"]):
        try:
            gid_i = int(gid) if pd.notna(gid) else -999
        except (TypeError, ValueError):
            gid_i = -999
        if gid_i != group_id:
            continue
        k = str(pid)
        if k in pid_to_epsf:
            rows.append(pid_to_epsf[k])
    return np.asarray(rows, dtype=np.int64)


def group_ids_from_ffi_stems(
    df: pd.DataFrame, ffi_product_ids: np.ndarray | list
) -> np.ndarray:
    """Group id per *ffi_product_ids* entry (looked up via the manifest product id)."""
    if "group_id" not in df.columns:
        raise ValueError("frame manifest table missing group_id")
    pids_index = row_ffi_product_id_series(df)
    pid_to_gid: dict[str, int] = {}
    for i, s in enumerate(pids_index):
        sk = str(s)
        if sk in pid_to_gid:
            continue
        gid = df.iloc[i]["group_id"]
        try:
            gid_i = int(gid) if pd.notna(gid) else -1
        except (TypeError, ValueError):
            gid_i = -1
        pid_to_gid[sk] = gid_i

    out = []
    for s in ffi_product_ids:
        pid = _coerce_to_product_id(s)
        out.append(pid_to_gid.get(pid, -1))
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
    pids = row_ffi_product_id_series(df).reset_index(drop=True)
    df_reset = df.reset_index(drop=True)
    use_col = path_col if path_col in df_reset.columns else None
    label = sanitize_workspace_label(subdir)
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
            pid = str(pids.iloc[i])
            if pid:
                cand = os.path.join(
                    output_dir, subdir, f"{workspace_frame_stem(pid, label)}.fits"
                )
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
    ``{output_dir}/ws/{label}/{tess<digits>}_{label}.fits``.
    """
    safe = sanitize_workspace_label(label)
    path_col = f"diff_{safe}_path"
    pids = row_ffi_product_id_series(df).reset_index(drop=True)
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
            pid = str(pids.iloc[i])
            if pid:
                cand = os.path.join(ws, f"{workspace_frame_stem(pid, safe)}.fits")
                if os.path.isfile(cand):
                    p = str(Path(cand).resolve())
        out.append(p)
    return out


