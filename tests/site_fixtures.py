"""Test helpers for template site config + deployment (not diff site_config loader)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_site_deployment(
    config_dir: Path,
    *,
    workspace_root: str,
    data_root: str,
    deployment_file: str = "deployment.yaml",
) -> None:
    path = config_dir / deployment_file
    path.write_text(
        "\n".join(
            [
                f"workspace_root: {workspace_root}",
                f"data_root: {data_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_site_config(
    path: Path,
    *,
    workspace_root: str,
    data_root: str,
    notifications_enabled: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "deployment_file: deployment.yaml",
                "stages:",
                "  mapping: {}",
                "notifications:",
                f"  enabled: {'true' if notifications_enabled else 'false'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_site_deployment(path.parent, workspace_root=workspace_root, data_root=data_root)


def write_unified_site_config(
    path: Path,
    *,
    workspace_root: str,
    data_root: str,
    diff: dict[str, Any] | None = None,
    stages: dict[str, Any] | None = None,
    deployment_file: str = "deployment.yaml",
    config_schema_version: int = 2,
    notifications_enabled: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a unified (schema v2) site ``pipeline.yaml`` + its ``deployment.yaml``.

    Parameters
    ----------
    path : Path
        Where to write the ``pipeline.yaml``. Its parent directory also gets
        a sibling ``deployment.yaml`` (via :func:`write_site_deployment`).
    workspace_root, data_root : str
        Values written into the sibling ``deployment.yaml``.
    diff : dict[str, Any] | None
        Raw mapping embedded verbatim under the top-level ``diff:`` key,
        e.g. ``{"pipeline": [{"kind": "shared_mask"}], "condor": {...}}``.
        Omit (``None``, the default) to write a config with no diff policy
        at all -- a legitimate template-only/photometry-only site.
    stages : dict[str, Any] | None
        Raw mapping for the top-level ``stages:`` key. Defaults to a minimal
        ``{"mapping": {}}`` when omitted.
    deployment_file : str
        Written as the top-level ``deployment_file:`` pointer, and also used
        as the sibling deployment file's name.
    config_schema_version : int
        Written as the top-level ``config_schema_version:`` marker.
    notifications_enabled : bool
        Value for ``notifications.enabled``.
    extra : dict[str, Any] | None
        Additional top-level keys merged in verbatim after the ones above --
        e.g. a legacy ``diff_config:`` pointer, to build a
        both-forms-rejected fixture alongside *diff*.

    Notes
    -----
    Serializes via ``yaml.safe_dump`` so a caller can assert deep-equality
    against *diff* after a round trip through the loader -- content survives
    verbatim even though YAML dump/load is not byte-identical to the input
    dict's repr.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {
        "config_schema_version": config_schema_version,
        "deployment_file": deployment_file,
        "stages": stages if stages is not None else {"mapping": {}},
        "notifications": {"enabled": notifications_enabled},
    }
    if diff is not None:
        doc["diff"] = diff
    if extra:
        doc.update(extra)
    path.write_text(
        yaml.safe_dump(doc, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    write_site_deployment(
        path.parent,
        workspace_root=workspace_root,
        data_root=data_root,
        deployment_file=deployment_file,
    )


def resolve_target_diff_config(site_dir: Path, target, *, config_filename: str = "pipeline.yaml"):
    """Load a unified site config's embedded ``diff:`` policy, resolved for one target.

    The v2 replacement for the removed ``freeze_target_diff_config(diff_config.yaml,
    target)``: instead of pointing at a standalone ``diff_config.yaml``, this
    loads ``{site_dir}/{config_filename}`` (written by :func:`write_unified_site_config`,
    with a ``diff=`` block) via :func:`load_runner_config` and resolves its
    embedded ``diff:`` policy against *target* + the sibling ``deployment.yaml``.
    """
    from syndiff_pipeline.common.orchestration.deployment import (
        deployment_path_for_config,
        load_deployment,
    )
    from syndiff_pipeline.difference_imaging.orchestration.site_config import (
        resolve_diff_config,
    )
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        load_runner_config,
    )

    site_dir = Path(site_dir)
    config_path = site_dir / config_filename
    runner_cfg = load_runner_config(config_path)
    if runner_cfg.diff is None:
        raise ValueError(f"{config_path} has no embedded 'diff:' policy")
    deploy_path = deployment_path_for_config(config_path, runner_cfg.deployment_file)
    deployment = load_deployment(config_path, runner_cfg.deployment_file)
    return resolve_diff_config(
        target,
        runner_cfg.diff,
        deployment,
        deployment_path=deploy_path,
        site_config_dir=site_dir,
    )


def write_materialized_config(
    path: Path,
    *,
    workspace_root: str,
    data_root: str,
    runs_root: str,
    state_db_path: str,
) -> None:
    """Frozen run config with embedded paths (no deployment file required)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"data_root: {data_root}",
                f"workspace_root: {workspace_root}",
                f"runs_root: {runs_root}",
                f"state_db_path: {state_db_path}",
                "stages:",
                "  mapping: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
