"""Helpers for site config + deployment in tests."""

from __future__ import annotations

from pathlib import Path


def write_site_deployment(
    config_dir: Path,
    *,
    handoff_root: str,
    data_root: str,
    deployment_file: str = "deployment.yaml",
) -> None:
    path = config_dir / deployment_file
    path.write_text(
        "\n".join(
            [
                f"handoff_root: {handoff_root}",
                f"data_root: {data_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_site_config(
    path: Path,
    *,
    handoff_root: str,
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
    write_site_deployment(path.parent, handoff_root=handoff_root, data_root=data_root)


def write_materialized_config(
    path: Path,
    *,
    handoff_root: str,
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
                f"handoff_root: {handoff_root}",
                f"runs_root: {runs_root}",
                f"state_db_path: {state_db_path}",
                "stages:",
                "  mapping: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
