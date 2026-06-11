"""Site deployment.yaml loading (paths + credentials beside config)."""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml

log = logging.getLogger(__name__)

_LEGACY_PATH_KEYS = frozenset(
    {
        "data_root",
        "workspace_root",
        "runs_root",
        "state_db_path",
        "ffi_dir",
        "skycell_wcs_csv",
        "gaia_credentials",
    }
)


def deployment_path_for_config(
    config_path: str | Path, deployment_file: str = "deployment.yaml"
) -> Path:
    return Path(config_path).expanduser().resolve().parent / deployment_file


def load_deployment_file(deployment_path: str | Path) -> dict:
    """Load a deployment YAML file from an explicit path."""
    path = Path(deployment_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Deployment file not found: {path}. "
            "Copy deployment.yaml.example to deployment.yaml beside your site config."
        )
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise FileNotFoundError(f"Failed to read deployment file {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Deployment file must be a YAML mapping: {path}")
    return data


def load_workspace_root_from_deployment(deployment_path: str | Path) -> Path:
    path = Path(deployment_path).expanduser().resolve()
    deployment = load_deployment_file(path)
    return Path(require_deployment_path(deployment, "workspace_root", deployment_path=path))


def load_deployment(config_path: str | Path, deployment_file: str = "deployment.yaml") -> dict:
    path = deployment_path_for_config(config_path, deployment_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"Deployment file not found: {path}. "
            "Copy deployment.yaml.example to deployment.yaml beside your site config."
        )
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise FileNotFoundError(f"Failed to read deployment file {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Deployment file must be a YAML mapping: {path}")
    return data


def warn_legacy_config_paths(raw: dict, *, config_path: Path) -> None:
    for key in _LEGACY_PATH_KEYS:
        if raw.get(key):
            log.warning(
                "Ignoring legacy config path %s in %s (use deployment.yaml)",
                key,
                config_path,
            )


def require_deployment_path(deployment: dict, key: str, *, deployment_path: Path) -> str:
    value = str(deployment.get(key, "")).strip()
    if not value:
        raise ValueError(f"deployment.yaml requires {key} ({deployment_path})")
    return str(Path(value).expanduser().resolve())


@contextmanager
def gaia_credentials_file(
    deployment: dict,
    *,
    deployment_path: Path,
) -> Iterator[str | None]:
    """Yield a two-line Gaia credentials file path, or None for anonymous TAP."""
    username = str(deployment.get("gaia_username", "")).strip()
    password = str(deployment.get("gaia_password", "")).strip()
    if not username and not password:
        yield None
        return
    if not username or not password:
        raise ValueError(
            f"deployment.yaml must set both gaia_username and gaia_password ({deployment_path})"
        )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="syndiff_gaia_",
        suffix=".credentials",
        delete=True,
    ) as fh:
        fh.write(f"{username}\n{password}\n")
        fh.flush()
        yield fh.name
