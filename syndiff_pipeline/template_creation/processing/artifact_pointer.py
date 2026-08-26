"""Validated, atomic ``current.json`` pointers for immutable PS1 artifacts.

An artifact directory is content-addressed and immutable.  A pointer is the
only permitted way to choose one of several complete versions for a skycell;
directory mtime is not provenance and must never participate in selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable


CURRENT_FILENAME = "current.json"
PROVENANCE_FILENAME = "_provenance.json"
POINTER_SCHEMA_VERSION = 1


class ArtifactPointerError(RuntimeError):
    """A current pointer is malformed, dangling, or inconsistent with its artifact."""


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    projection: str
    skycell: str
    fingerprint: str
    recipe_id: str
    path: Path


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactPointerError(f"current pointer has invalid {name}")
    return value


def read_current_pointer(
    cell_root: str | Path,
    *,
    expected_kind: str,
    projection: str,
    skycell: str,
    required_members: Iterable[str],
) -> ArtifactRef | None:
    """Resolve and validate ``cell_root/current.json``.

    ``None`` means no pointer was ever selected.  A present but invalid pointer
    raises: silently selecting a newer directory would make the correction's
    inputs nondeterministic and scientifically unauditable.
    """
    root = Path(cell_root)
    pointer_path = root / CURRENT_FILENAME
    if not pointer_path.is_file():
        return None
    try:
        payload = json.loads(pointer_path.read_text())
    except Exception as exc:
        raise ArtifactPointerError(f"cannot read current pointer {pointer_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise ArtifactPointerError(f"current pointer has unsupported schema: {pointer_path}")

    kind = _require_string(payload.get("kind"), "kind")
    fp = _require_string(payload.get("fingerprint"), "fingerprint")
    recipe_id = _require_string(payload.get("recipe_id"), "recipe_id")
    spatial = payload.get("spatial_key")
    if (
        kind != expected_kind
        or not isinstance(spatial, dict)
        or spatial.get("projection") != str(projection)
        or spatial.get("skycell") != str(skycell)
    ):
        raise ArtifactPointerError(f"current pointer identity mismatch: {pointer_path}")

    return validate_artifact(
        root,
        kind=expected_kind,
        projection=str(projection),
        skycell=str(skycell),
        fingerprint=fp,
        recipe_id=recipe_id,
        required_members=required_members,
    )


def validate_artifact(
    cell_root: str | Path,
    *,
    kind: str,
    projection: str,
    skycell: str,
    fingerprint: str,
    recipe_id: str | None,
    required_members: Iterable[str],
) -> ArtifactRef:
    """Validate one immutable artifact directory without changing a pointer."""
    root = Path(cell_root)
    artifact_dir = root / fingerprint
    if not artifact_dir.is_dir() or any(not (artifact_dir / name).is_file() for name in required_members):
        raise ArtifactPointerError(f"current pointer target is incomplete: {artifact_dir}")
    try:
        sidecar = json.loads((artifact_dir / PROVENANCE_FILENAME).read_text())
    except Exception as exc:
        raise ArtifactPointerError(f"current pointer target lacks valid provenance: {artifact_dir}") from exc
    if (
        sidecar.get("fingerprint") != fingerprint
        or sidecar.get("kind") != kind
        or (recipe_id is not None and sidecar.get("recipe_id") != recipe_id)
        or sidecar.get("spatial_key") != {"projection": str(projection), "skycell": str(skycell)}
    ):
        raise ArtifactPointerError(f"current pointer target provenance mismatch: {artifact_dir}")
    sidecar_recipe_id = _require_string(sidecar.get("recipe_id"), "target recipe_id")
    return ArtifactRef(kind, str(projection), str(skycell), fingerprint, sidecar_recipe_id, artifact_dir)


def write_current_pointer(
    cell_root: str | Path,
    artifact: ArtifactRef,
) -> None:
    """Atomically point one skycell at a previously validated immutable artifact."""
    root = Path(cell_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "kind": artifact.kind,
        "spatial_key": {"projection": artifact.projection, "skycell": artifact.skycell},
        "fingerprint": artifact.fingerprint,
        "recipe_id": artifact.recipe_id,
    }
    temporary = root / f"_tmp_current_{os.getpid()}.json"
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        os.replace(temporary, root / CURRENT_FILENAME)
    finally:
        # An interrupted replacement may leave the temporary file, which is
        # harmless and deliberately never treated as a pointer.
        if temporary.exists():
            temporary.unlink(missing_ok=True)
