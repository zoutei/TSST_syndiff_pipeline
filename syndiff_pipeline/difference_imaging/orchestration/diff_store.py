"""SCC-scoped diff product store (lane-primary layout).

Per-FFI artifacts are written under
``{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/{workspace_label}/{recipe_fp}/``.
Event workspaces may record pointers in ``scc_diff_index.json``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

from syndiff_pipeline.common.scc_paths import (
    normalize_store_name,
    scc_diff_workspace_dir,
    scc_diff_workspace_index_path,
)

log = logging.getLogger(__name__)

_INDEX_VERSION = 1


def diff_artifact_basename(product_id: str, label: str, *, suffix: str = ".fits.fz") -> str:
    stem = f"{product_id}_{label}" if label else product_id
    return f"{stem}{suffix}"


def scc_diff_artifact_path(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    stage_label: str,
    recipe_fp: str,
    product_id: str,
    label: str,
    *,
    output_store_name: str | None = None,
    suffix: str = ".fits.fz",
) -> Path:
    return scc_diff_workspace_dir(
        data_root,
        sector,
        camera,
        ccd,
        store_name=normalize_store_name(output_store_name),
        workspace_label=stage_label,
        recipe_fp=recipe_fp,
    ) / diff_artifact_basename(product_id, label, suffix=suffix)


def resolve_diff_write_path(
    *,
    data_root: str | None,
    sck: tuple[int, int, int] | None,
    kind: str,
    stage_label: str,
    product_id: str,
    label: str,
    params: Any,
    workspace_path: str | Path,
    output_store_name: str | None = None,
    suffix: str = ".fits.fz",
) -> tuple[Path, bool]:
    """
    Choose the on-disk write target for a per-FFI diff artifact.

    When ``data_root`` and SCC context are available and a recipe fingerprint
    resolves, returns the lane path under ``diff_{store}/``; otherwise returns
    the event workspace path.
    """
    ws = Path(workspace_path)
    if not data_root or sck is None:
        return ws, False
    recipe_fp = recipe_fp_for_artifact(kind, params)
    if not recipe_fp:
        return ws, False
    sector, camera, ccd = sck
    return (
        scc_diff_artifact_path(
            data_root,
            sector,
            camera,
            ccd,
            stage_label,
            recipe_fp,
            product_id,
            label,
            output_store_name=output_store_name,
            suffix=suffix,
        ),
        True,
    )


def record_scc_artifact_pointer(
    *,
    workspace_root: str | Path,
    product_id: str,
    label: str,
    scc_path: str | Path,
    kind: str,
    fingerprint: Optional[str],
    stage_label: str,
    recipe_fp: str,
) -> None:
    """Record a workspace index entry for an SCC-primary artifact."""
    record_workspace_pointer(
        workspace_root,
        f"{product_id}:{label}",
        {
            "kind": kind,
            "fingerprint": fingerprint,
            "scc_path": str(scc_path),
            "stage_label": stage_label,
            "recipe_fp": recipe_fp,
        },
    )


def mirror_to_scc_store(source: str | Path, dest: Path) -> bool:
    """Atomically copy *source* to *dest*. Never raises."""
    try:
        src = Path(source)
        if not src.is_file():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file():
            return True
        tmp = dest.parent / f"_tmp_{dest.name}_{os.getpid()}"
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
        return True
    except Exception:
        log.debug("mirror_to_scc_store failed %s -> %s", source, dest, exc_info=True)
        if "tmp" in locals() and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return False


def _load_index(path: Path) -> dict:
    if not path.is_file():
        return {"version": _INDEX_VERSION, "artifacts": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "artifacts" in data:
            return data
    except Exception:
        pass
    return {"version": _INDEX_VERSION, "artifacts": {}}


def record_workspace_pointer(
    workspace_root: str | Path,
    key: str,
    entry: Mapping[str, Any],
) -> None:
    """Append/update one artifact pointer in the workspace index. Never raises."""
    try:
        path = scc_diff_workspace_index_path(workspace_root)
        data = _load_index(path)
        artifacts = data.setdefault("artifacts", {})
        artifacts[str(key)] = dict(entry)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        log.debug("record_workspace_pointer failed for %s", key, exc_info=True)


def recipe_fp_for_artifact(kind: str, params: Any) -> Optional[str]:
    """Compute the recipe fingerprint for a diff artifact kind."""
    try:
        from syndiff_pipeline.difference_imaging.orchestration import provenance_glue
        from syndiff_pipeline.common.provenance.fingerprint import recipe_id

        recipe = provenance_glue.diff_recipe(kind, params)
        return recipe_id(kind, recipe["params"], recipe["code_version"])
    except Exception:
        log.debug("recipe_fp_for_artifact failed for kind=%s", kind, exc_info=True)
        return None


def try_materialize_workspace_artifact(
    *,
    data_root: str | None,
    sck: tuple[int, int, int] | None,
    kind: str,
    stage_label: str,
    product_id: str,
    label: str,
    params: Any,
    workspace_dest: str | Path,
    workspace_root: Optional[str] = None,
    output_store_name: str | None = None,
    suffix: str = ".fits.fz",
    fingerprint: Optional[str] = None,
) -> bool:
    """Copy a finalized artifact from the SCC diff lane into an event workspace.

    Returns True when *workspace_dest* exists after the call (already present or
    freshly materialized). Never raises.
    """
    if not data_root or sck is None:
        return False
    dest = Path(workspace_dest)
    if dest.is_file():
        return True
    recipe_fp = recipe_fp_for_artifact(kind, params)
    if not recipe_fp:
        return False
    sector, camera, ccd = sck
    src = scc_diff_artifact_path(
        data_root,
        sector,
        camera,
        ccd,
        stage_label,
        recipe_fp,
        product_id,
        label,
        output_store_name=output_store_name,
        suffix=suffix,
    )
    if not src.is_file():
        return False
    if not mirror_to_scc_store(src, dest):
        return False
    if workspace_root:
        record_workspace_pointer(
            workspace_root,
            f"{product_id}:{label}",
            {
                "kind": kind,
                "fingerprint": fingerprint,
                "scc_path": str(src),
                "stage_label": stage_label,
                "recipe_fp": recipe_fp,
                "materialized": str(dest),
            },
        )
    return True

