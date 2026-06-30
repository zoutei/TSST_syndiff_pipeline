"""Workspace-aware diff completion checks for orchestrator verification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.pipeline_entries import split_pipeline
from syndiff_pipeline.difference_imaging.orchestration.site_config import freeze_target_diff_config
from syndiff_pipeline.difference_imaging.orchestration.validate import _outputs_for_stage
from syndiff_pipeline.difference_imaging.stages.photometry import lightcurve_csv_basename
from syndiff_pipeline.difference_imaging.stages.kernel import (
    KERNEL_FIT_META_BASENAME,
    KERNEL_R2_NPZ_BASENAME,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    is_pipeline_fits_filename,
    resolve_pipeline_artifact_path,
)
from syndiff_pipeline.difference_imaging.support.manifest import manifest_path_from_output_dir
from syndiff_pipeline.difference_imaging.support.paths import (
    SHARED_MASK_FITS_BASENAME,
    workspace_root,
)

if TYPE_CHECKING:
    from syndiff_pipeline.common.orchestration.spec import StageRunContext
    from syndiff_pipeline.template_creation.orchestration.runner_config import RunnerConfig


def resolve_diff_site_config_path(
    *,
    meta: dict | None = None,
    runner_cfg: RunnerConfig | None = None,
) -> Path:
    """Match execute-time config resolution: run_meta overrides runner default."""
    for key in ("source_diff_config_path", "diff_config_path"):
        raw = (meta or {}).get(key) or getattr(runner_cfg, "diff_config_path", "")
        if raw:
            return Path(str(raw)).expanduser().resolve()
    raise ValueError(
        "Diff verification requires source_diff_config_path in run_meta or "
        "diff_config_path on RunnerConfig"
    )


def apply_workspace_run_id_override(
    cfg: SynDiffConfig,
    meta: dict | None,
) -> SynDiffConfig:
    """Apply orchestrator run meta override on top of frozen site defaults."""
    if not meta:
        return cfg
    override = meta.get("workspace_run_id")
    if override is not None and str(override).strip():
        cfg.workspace_run_id = str(override).strip()
    return cfg


def frozen_diff_config_for_verify(
    site_config_path: str | Path,
    target: Target,
    *,
    meta: dict | None = None,
) -> SynDiffConfig:
    """Load site policy + deployment and apply optional run-level workspace override."""
    cfg = freeze_target_diff_config(site_config_path, target)
    return apply_workspace_run_id_override(cfg, meta)


def frozen_diff_config_for_context(ctx: StageRunContext) -> SynDiffConfig:
    return frozen_diff_config_for_verify(
        resolve_diff_site_config_path(meta=ctx.meta, runner_cfg=ctx.runner_cfg),
        ctx.target,
        meta=ctx.meta,
    )


def diff_workspace_root(cfg: SynDiffConfig, event_dir: str | Path) -> Path:
    return Path(
        workspace_root(str(event_dir), run_id=getattr(cfg, "workspace_run_id", None))
    )


def _last_executable_stage(cfg: SynDiffConfig) -> dict | None:
    _, _, stages = split_pipeline(cfg.pipeline)
    if not stages:
        return None
    return stages[-1][1]


def _label_dir_has_files(ws_dir: Path, label: str) -> bool:
    d = ws_dir / label
    if not d.is_dir():
        return False
    return any(p.is_file() for p in d.rglob("*"))


def _label_dir_has_fits(ws_dir: Path, label: str) -> bool:
    d = ws_dir / label
    if not d.is_dir():
        return False
    return any(
        is_pipeline_fits_filename(p.name) for p in d.rglob("*") if p.is_file()
    )


def _final_stage_complete(cfg: SynDiffConfig, ws_dir: Path) -> bool:
    stage = _last_executable_stage(cfg)
    if stage is None:
        return False

    kind = stage.get("kind")

    if kind == "shared_mask":
        return (
            resolve_pipeline_artifact_path(str(ws_dir), SHARED_MASK_FITS_BASENAME)
            is not None
        )

    if kind == "forced_photometry":
        label = str(stage["output"]).strip()
        phot_dir = ws_dir / label
        if not phot_dir.is_dir():
            return False
        methods = stage.get("methods") or []
        if not methods:
            return False
        extras = getattr(cfg, "additional_forced_targets", None) or []
        for entry in methods:
            if not isinstance(entry, dict):
                return False
            name = str(entry.get("name", "")).strip()
            if not name:
                return False
            primary_csv = lightcurve_csv_basename(name)
            if not (phot_dir / primary_csv).is_file():
                return False
            for pt in extras:
                if not isinstance(pt, dict):
                    continue
                extra_name = str(pt.get("name", "")).strip()
                if not extra_name:
                    continue
                extra_csv = lightcurve_csv_basename(name, extra_name)
                if not (phot_dir / extra_csv).is_file():
                    return False
        return True

    if kind == "kernel_subtract":
        o = stage.get("output") or {}
        diffs = str(o.get("diffs", "")).strip()
        return bool(diffs) and _label_dir_has_fits(ws_dir, diffs)

    if kind == "background":
        label = str(stage.get("output", "")).strip()
        if not label:
            return False
        out_dir = ws_dir / label
        if not out_dir.is_dir():
            return False
        stack_npz = out_dir / "stack.npz"
        stack_npy = out_dir / "stack.npy"
        if stack_npz.is_file() or stack_npy.is_file():
            return True
        return _label_dir_has_fits(ws_dir, label)

    if kind in (
        "subtract",
        "sat_template",
    ):
        outputs = _outputs_for_stage(stage)
        return bool(outputs) and all(_label_dir_has_files(ws_dir, lab) for lab in outputs)

    if kind == "epsf":
        label = str(stage["output"]).strip()
        epsf_dir = ws_dir / label
        if not epsf_dir.is_dir():
            return False
        return any(epsf_dir.rglob("group_epsf_*.npy"))

    if kind == "kernel_fit":
        label = str(stage.get("output", "")).strip()
        if not label:
            return False
        d = ws_dir / label
        return (d / KERNEL_FIT_META_BASENAME).is_file() and (d / KERNEL_R2_NPZ_BASENAME).is_file()

    if kind == "convolved_templates":
        label = str(stage.get("output", "")).strip()
        return bool(label) and (ws_dir / label / "convolved_templates.csv").is_file()

    if kind == "hotpants":
        o = stage.get("output") or {}
        diffs = str(o.get("diffs", "")).strip()
        return bool(diffs) and _label_dir_has_fits(ws_dir, diffs)

    return False


def diff_workspace_complete(cfg: SynDiffConfig, event_dir: str | Path) -> bool:
    """True when handoff manifest and final pipeline outputs exist in the active workspace tree."""
    event_dir = Path(event_dir)
    manifest_csv = Path(manifest_path_from_output_dir(str(event_dir), None))
    if not manifest_csv.is_file():
        return False
    ws_dir = diff_workspace_root(cfg, event_dir)
    if not ws_dir.is_dir():
        return False
    return _final_stage_complete(cfg, ws_dir)


def collect_diff_workspace_artifacts(cfg: SynDiffConfig, event_dir: str | Path) -> list[str]:
    """List artifact paths under the active workspace tree for diff manifest collection."""
    from syndiff_pipeline.difference_imaging.support.paths import DEFAULT_MANIFEST_BASENAME

    event_dir = Path(event_dir)
    artifacts: list[str] = []
    manifest_csv = manifest_path_from_output_dir(str(event_dir), None)
    if Path(manifest_csv).is_file():
        artifacts.append(manifest_csv)

    ws_dir = diff_workspace_root(cfg, event_dir)
    if not ws_dir.is_dir():
        return artifacts

    for child in sorted(ws_dir.iterdir()):
        if not child.is_dir():
            if child.is_file():
                artifacts.append(str(child.resolve()))
            continue
        if child.name == "master":
            master_manifest = child / DEFAULT_MANIFEST_BASENAME
            if master_manifest.is_file():
                artifacts.append(str(master_manifest.resolve()))
            continue
        for path in sorted(child.rglob("*")):
            if path.is_file():
                artifacts.append(str(path.resolve()))
    return artifacts
