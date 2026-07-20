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
FITS_DIRNAME = "fits"
MATERIALIZED_FITS_SIDECAR = "materialized_fits.json"
LOCK_NAME = ".lock"

_CONTRIB_RE = re.compile(
    r"^(?P<skycell>skycell\.\d+\.\d+)_sx(?P<sx>[+-]?\d+)_sy(?P<sy>[+-]?\d+)"
    r"(?:_gid(?P<gid>\d+))?\.npz$",
    re.IGNORECASE,
)


from syndiff_pipeline.common.scc_paths import scc_templates_dir


def templates_root(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
    store_name: str | None = None,
) -> Path:
    """Return the SCC templates store directory (does not create it)."""
    return scc_templates_dir(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=oversampling_factor,
        store_name=store_name,
    )


def field_templates_root(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
    store_name: str | None = None,
) -> Path:
    """Legacy alias for :func:`templates_root`."""
    return templates_root(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=oversampling_factor,
        store_name=store_name,
    )


def field_store_lock(store_root: str | Path) -> FileLock:
    """Process-wide lock for writers of one SCC field store."""
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(str(root / LOCK_NAME), timeout=-1)


def contrib_basename(
    skycell: str,
    sx_int: int,
    sy_int: int,
    *,
    group_id: int | None = None,
) -> str:
    """Filename for one sparse contribution key.

    When ``group_id`` is set, the basename is group-qualified:
    ``{skycell}_sx…_sy…_gid{N}.npz``. Field mode always uses ``_gid``.
    """
    name = str(skycell).strip()
    if not name.startswith("skycell."):
        name = f"skycell.{name}" if not name.startswith("skycell") else name
    stem = f"{name}_sx{int(sx_int):+d}_sy{int(sy_int):+d}"
    if group_id is not None:
        stem += f"_gid{int(group_id)}"
    return f"{stem}.npz"


def contrib_path(
    store_root: str | Path,
    skycell: str,
    sx_int: int,
    sy_int: int,
    *,
    group_id: int | None = None,
) -> Path:
    return Path(store_root) / CONTRIBS_DIRNAME / contrib_basename(
        skycell, sx_int, sy_int, group_id=group_id
    )


def field_fits_basename(
    sector: int,
    camera: int,
    ccd: int,
    group_id: int,
    *,
    oversampling_factor: int = 1,
) -> str:
    """Basename for one materialized field template FITS (logical ``.fits``)."""
    os_part = f"_os{int(oversampling_factor)}" if int(oversampling_factor) > 1 else ""
    return (
        f"syndiff_field_s{int(sector):04d}_{int(camera)}_{int(ccd)}"
        f"{os_part}_gid{int(group_id)}.fits"
    )


def field_fits_path(
    store_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    group_id: int,
    *,
    oversampling_factor: int = 1,
) -> Path:
    """Path under ``fits/`` for one group's materialized template."""
    return (
        Path(store_root)
        / FITS_DIRNAME
        / field_fits_basename(
            sector, camera, ccd, group_id, oversampling_factor=oversampling_factor
        )
    )


def _roi_bounds_to_assemble_crop(
    roi_bounds: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if roi_bounds is None:
        return None
    x_min, y_min, x_max, y_max = (int(v) for v in roi_bounds)
    return (x_min, x_max, y_min, y_max)


def build_field_fits_header(
    *,
    sector: int,
    camera: int,
    ccd: int,
    group_id: int,
    oversampling_factor: int = 1,
    roi_bounds: tuple[int, int, int, int] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Any:
    """Minimal FITS header for a materialized field template."""
    from astropy.io import fits

    hdr = fits.Header()
    hdr["SYNDIFF"] = (True, "SynDiff template")
    hdr["SYNDMODE"] = ("field", "SynDiff geometry mode")
    hdr["SECTOR"] = (int(sector), "TESS sector")
    hdr["CAMERA"] = (int(camera), "TESS camera")
    hdr["CCD"] = (int(ccd), "TESS CCD")
    hdr["GROUP_ID"] = (int(group_id), "WCS signature group id")
    os_factor = max(1, int(oversampling_factor))
    if os_factor > 1:
        hdr["OVERSAMP"] = (os_factor, "Oversampling factor")
    if roi_bounds is not None:
        x_min, y_min, x_max, y_max = (int(v) for v in roi_bounds)
        hdr["XMIN"] = (x_min, "ROI xmin in base TESS pixels")
        hdr["XMAX"] = (x_max, "ROI xmax (exclusive) in base TESS pixels")
        hdr["YMIN"] = (y_min, "ROI ymin in base TESS pixels")
        hdr["YMAX"] = (y_max, "ROI ymax (exclusive) in base TESS pixels")
        hdr["ROIW"] = (x_max - x_min, "ROI width in base TESS pixels")
        hdr["ROIH"] = (y_max - y_min, "ROI height in base TESS pixels")
    if provenance:
        if "intra_skycell_R" in provenance:
            hdr["INTRA_R"] = (int(provenance["intra_skycell_R"]), "Intra-skycell dilation R")
        elif "hybrid_R" in provenance:
            hdr["INTRA_R"] = (int(provenance["hybrid_R"]), "Intra-skycell dilation R")
        if "n_intra_skycell_keys" in provenance and provenance["n_intra_skycell_keys"] is not None:
            hdr["NINTRKEY"] = (
                int(provenance["n_intra_skycell_keys"]),
                "Intra-skycell exact cache keys",
            )
        elif "n_exact_keys" in provenance and provenance["n_exact_keys"] is not None:
            hdr["NINTRKEY"] = (int(provenance["n_exact_keys"]), "Intra-skycell exact cache keys")
        if (
            "n_inter_skycell_pair_states" in provenance
            and provenance["n_inter_skycell_pair_states"] is not None
        ):
            hdr["NINTERPR"] = (
                int(provenance["n_inter_skycell_pair_states"]),
                "Inter-skycell pair-state cache keys",
            )
        elif "n_l4b_pair_states" in provenance and provenance["n_l4b_pair_states"] is not None:
            hdr["NINTERPR"] = (
                int(provenance["n_l4b_pair_states"]),
                "Inter-skycell pair-state cache keys",
            )
    return hdr


def write_field_group_fits(
    out_path: str | Path,
    flux: np.ndarray,
    count: np.ndarray,
    *,
    header: Any | None = None,
) -> str:
    """Write one group's mean-flux template FITS (+ COUNT extension) as ``.fits.fz``."""
    from astropy.io import fits

    from syndiff_pipeline.common.fits_io import write_hdul_fits

    hdr = fits.Header(header) if header is not None else fits.Header()
    flux_arr = np.asarray(flux, dtype=np.float32)
    count_arr = np.asarray(count, dtype=np.float32)
    count_hdr = hdr.copy()
    count_hdr["EXTNAME"] = "COUNT"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(flux_arr, header=hdr),
            fits.ImageHDU(count_arr, header=count_hdr, name="COUNT"),
        ]
    )
    return write_hdul_fits(out_path, hdul)


def parse_contrib_basename(
    name: str,
) -> Optional[tuple[str, int, int] | tuple[str, int, int, int]]:
    m = _CONTRIB_RE.match(Path(name).name)
    if not m:
        return None
    base = (m.group("skycell"), int(m.group("sx")), int(m.group("sy")))
    gid = m.group("gid")
    if gid is None:
        return base
    return base[0], base[1], base[2], int(gid)


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
    group_id: int | None = None,
) -> Path:
    """Write one sparse contrib NPZ via temp file + atomic replace.

    No store-wide lock: concurrent writers use distinct final paths
    (``…_gid{N}.npz``) and NFS-safe ``Path.replace``.
    """
    import os
    import tempfile

    root = Path(store_root)
    out = contrib_path(root, skycell, sx_int, sy_int, group_id=group_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indices": np.asarray(indices, dtype=np.int64),
        "flux_sum": np.asarray(flux_sum, dtype=np.float64),
        "count": np.asarray(count, dtype=np.float64),
        "skycell": np.asarray(str(skycell)),
        "sx_int": np.asarray(int(sx_int), dtype=np.int32),
        "sy_int": np.asarray(int(sy_int), dtype=np.int32),
    }
    if mask_count is not None:
        payload["mask_count"] = np.asarray(mask_count, dtype=np.float64)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out.name}.",
        suffix=".tmp.npz",
        dir=str(out.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(out)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
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
    group_id: int | None = None,
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
        path = contrib_path(root, skycell, sx_i, sy_i, group_id=group_id)
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
    required_keys: Iterable[tuple] | None = None,
    require_nonempty: bool = False,
    group_id: int | None = None,
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
        for key in required_keys:
            if len(key) == 4:
                gid_i, skycell, sx_i, sy_i = key
                p = contrib_path(root, skycell, sx_i, sy_i, group_id=int(gid_i))
            else:
                skycell, sx_i, sy_i = key
                p = contrib_path(
                    root, skycell, sx_i, sy_i, group_id=group_id
                )
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
