"""SCC-scoped field template store: sparse contribs + assemble by group_id.

Layout (canonical)::

    {data_root}/field_templates/sector_{S}_camera_{C}_ccd_{K}/[oversampling_{N}/]
      template_manifest.json
      shift_schedule.npz
      template_group_shifts.parquet
      contribs/skycell.{name}_sx{±N}_sy{±N}.npz
      .lock
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from filelock import FileLock

SCHEMA_VERSION = 1
MANIFEST_NAME = "template_manifest.json"
CONTRIBS_DIRNAME = "contribs"
LOCK_NAME = ".lock"

_CONTRIB_RE = re.compile(
    r"^(?P<skycell>skycell\.\d+\.\d+)_sx(?P<sx>[+-]?\d+)_sy(?P<sy>[+-]?\d+)\.npz$",
    re.IGNORECASE,
)


def field_templates_root(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
) -> Path:
    """Return the SCC field-templates directory (does not create it)."""
    root = (
        Path(data_root).expanduser().resolve()
        / "field_templates"
        / f"sector_{int(sector):04d}_camera_{int(camera)}_ccd_{int(ccd)}"
    )
    if int(oversampling_factor) > 1:
        root = root / f"oversampling_{int(oversampling_factor)}"
    return root


def field_store_lock(store_root: str | Path) -> FileLock:
    """Process-wide lock for writers of one SCC field store."""
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(str(root / LOCK_NAME), timeout=-1)


def contrib_basename(skycell: str, sx_int: int, sy_int: int) -> str:
    """Filename for one sparse contribution key."""
    name = str(skycell).strip()
    if not name.startswith("skycell."):
        name = f"skycell.{name}" if not name.startswith("skycell") else name
    return f"{name}_sx{int(sx_int):+d}_sy{int(sy_int):+d}.npz"


def contrib_path(store_root: str | Path, skycell: str, sx_int: int, sy_int: int) -> Path:
    return Path(store_root) / CONTRIBS_DIRNAME / contrib_basename(skycell, sx_int, sy_int)


def parse_contrib_basename(name: str) -> Optional[tuple[str, int, int]]:
    m = _CONTRIB_RE.match(Path(name).name)
    if not m:
        return None
    return m.group("skycell"), int(m.group("sx")), int(m.group("sy"))


def write_contrib(
    store_root: str | Path,
    skycell: str,
    sx_int: int,
    sy_int: int,
    *,
    indices: np.ndarray,
    flux_sum: np.ndarray,
    count: np.ndarray,
    mask_count: np.ndarray | None = None,
) -> Path:
    """Write one sparse contrib NPZ under lock. ``indices`` are flat TESS pixel ids."""
    root = Path(store_root)
    out = contrib_path(root, skycell, sx_int, sy_int)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indices": np.asarray(indices, dtype=np.int64),
        "flux_sum": np.asarray(flux_sum, dtype=np.float64),
        "count": np.asarray(count, dtype=np.float64),
        "skycell": np.asarray(str(skycell)),
        "sx_int": np.asarray(int(sx_int), dtype=np.int16),
        "sy_int": np.asarray(int(sy_int), dtype=np.int16),
    }
    if mask_count is not None:
        payload["mask_count"] = np.asarray(mask_count, dtype=np.float64)
    with field_store_lock(root):
        np.savez_compressed(out, **payload)
    return out


def load_contrib(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@dataclass(frozen=True)
class FieldManifest:
    geometry_mode: str
    scope: str
    assembly: str
    materialize_fits: bool
    sector: int
    camera: int
    ccd: int
    contribs_dir: str
    groups: list[dict[str, Any]]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "geometry_mode": self.geometry_mode,
            "scope": self.scope,
            "assembly": self.assembly,
            "materialize_fits": self.materialize_fits,
            "sector": int(self.sector),
            "camera": int(self.camera),
            "ccd": int(self.ccd),
            "contribs_dir": self.contribs_dir,
            "groups": list(self.groups),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


def write_template_manifest(store_root: str | Path, manifest: FieldManifest | Mapping[str, Any]) -> Path:
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_NAME
    payload = manifest.to_dict() if isinstance(manifest, FieldManifest) else dict(manifest)
    with field_store_lock(root):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_template_manifest(store_root: str | Path) -> dict[str, Any]:
    path = Path(store_root) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"field template manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_group_from_contribs(
    store_root: str | Path,
    shifts: Sequence[tuple[str, int, int]],
    *,
    shape: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Sum sparse contribs for one signature group.

    Parameters
    ----------
    shifts
        Iterable of ``(skycell, sx_int, sy_int)`` for this ``group_id``.
    shape
        Full-chip ``(ny, nx)`` TESS shape.
    crop
        Optional ``(x_min, x_max, y_min, y_max)`` half-open crop in full-FFI pixels.
    """
    ny, nx = int(shape[0]), int(shape[1])
    flux = np.zeros(ny * nx, dtype=np.float64)
    count = np.zeros(ny * nx, dtype=np.float64)
    mask_count = np.zeros(ny * nx, dtype=np.float64)
    root = Path(store_root)
    n_loaded = 0
    for skycell, sx_i, sy_i in shifts:
        path = contrib_path(root, skycell, sx_i, sy_i)
        if not path.is_file():
            raise FileNotFoundError(f"missing field contrib: {path}")
        data = load_contrib(path)
        idx = np.asarray(data["indices"], dtype=np.int64)
        flux[idx] += np.asarray(data["flux_sum"], dtype=np.float64)
        count[idx] += np.asarray(data["count"], dtype=np.float64)
        if "mask_count" in data:
            mask_count[idx] += np.asarray(data["mask_count"], dtype=np.float64)
        n_loaded += 1
    flux_2d = flux.reshape(ny, nx)
    count_2d = count.reshape(ny, nx)
    mask_2d = mask_count.reshape(ny, nx)
    if crop is not None:
        x0, x1, y0, y1 = (int(v) for v in crop)
        flux_2d = flux_2d[y0:y1, x0:x1]
        count_2d = count_2d[y0:y1, x0:x1]
        mask_2d = mask_2d[y0:y1, x0:x1]
    return {
        "flux_sum": flux_2d,
        "count": count_2d,
        "mask_count": mask_2d,
        "n_contribs": np.asarray(n_loaded, dtype=np.int32),
    }


def verify_field_store(
    store_root: str | Path,
    *,
    required_keys: Iterable[tuple[str, int, int]] | None = None,
    require_nonempty: bool = False,
) -> dict[str, Any]:
    """Thin completeness check for SCC field store reuse."""
    root = Path(store_root)
    reasons: list[str] = []
    if not root.is_dir():
        return {"ok": False, "reasons": [f"missing store root {root}"]}
    man = root / MANIFEST_NAME
    if not man.is_file():
        reasons.append(f"missing {MANIFEST_NAME}")
    contrib_dir = root / CONTRIBS_DIRNAME
    if not contrib_dir.is_dir():
        reasons.append(f"missing {CONTRIBS_DIRNAME}/")
    missing = []
    empty = []
    if required_keys is not None and contrib_dir.is_dir():
        for skycell, sx_i, sy_i in required_keys:
            p = contrib_path(root, skycell, sx_i, sy_i)
            if not p.is_file():
                missing.append(p.name)
                continue
            if require_nonempty:
                data = load_contrib(p)
                if len(np.asarray(data["indices"])) == 0:
                    empty.append(p.name)
        if missing:
            reasons.append(f"missing {len(missing)} contrib keys (e.g. {missing[:3]})")
        if empty:
            reasons.append(f"{len(empty)} empty contrib keys (e.g. {empty[:3]})")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "missing_contribs": missing,
        "empty_contribs": empty,
    }
