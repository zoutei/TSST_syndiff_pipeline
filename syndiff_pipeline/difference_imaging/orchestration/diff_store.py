"""SCC-scoped diff product store (flat lane layout).

Per-FFI artifacts live under
``{data_root}/s{SSSS}/c{C}/k{K}/diff_{lane}/{label}/{ffi_stem}_{label}.fits.fz``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from syndiff_pipeline.common.scc_paths import (
    normalize_store_name,
    scc_diff_label_dir,
)

log = logging.getLogger(__name__)


def diff_artifact_basename(ffi_stem: str, label: str, *, suffix: str = ".fits.fz") -> str:
    stem = f"{ffi_stem}_{label}" if label else ffi_stem
    return f"{stem}{suffix}"


def scc_diff_artifact_path(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    label: str,
    ffi_stem: str,
    *,
    output_store_name: str | None = None,
    suffix: str = ".fits.fz",
) -> Path:
    return scc_diff_label_dir(
        data_root,
        sector,
        camera,
        ccd,
        store_name=normalize_store_name(output_store_name),
        label=label,
    ) / diff_artifact_basename(ffi_stem, label, suffix=suffix)


def resolve_diff_write_path(
    *,
    data_root: str | Path,
    sck: tuple[int, int, int],
    kind: str,
    stage_label: str,
    ffi_stem: str,
    label: str,
    params: Any,
    output_store_name: str | None = None,
    suffix: str = ".fits.fz",
) -> Path:
    """Return the SCC diff lane write path for one per-FFI artifact."""
    if not data_root or sck is None:
        raise ValueError("resolve_diff_write_path requires data_root and sck")
    sector, camera, ccd = sck
    return scc_diff_artifact_path(
        data_root,
        sector,
        camera,
        ccd,
        stage_label,
        ffi_stem,
        output_store_name=output_store_name,
        suffix=suffix,
    )


def recipe_fp_for_artifact(kind: str, params: Any) -> Optional[str]:
    """Compute the recipe fingerprint for a diff artifact kind (provenance only)."""
    try:
        from syndiff_pipeline.difference_imaging.orchestration import provenance_glue
        from syndiff_pipeline.common.provenance.fingerprint import recipe_id

        recipe = provenance_glue.diff_recipe(kind, params)
        return recipe_id(kind, recipe["params"], recipe["code_version"])
    except Exception:
        log.debug("recipe_fp_for_artifact failed for kind=%s", kind, exc_info=True)
        return None
