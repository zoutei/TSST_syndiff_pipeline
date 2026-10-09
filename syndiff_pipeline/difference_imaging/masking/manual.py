"""Manual extra-mask file (``manual_masks.yaml``): hand-listed bad columns / rectangles / circles per SCC.

Coordinates are FULL-FFI 0-based science-array pixels (the frame of ``crop_bounds``); ranges are ``[min, max)``.
The crop-local position is ``x - crop_bounds['x_min']``, ``y - crop_bounds['y_min']``. Entries for other SCCs are
ignored. Every region is OR-ed into the static shared mask (bit ``edge`` = 8 by default; ``sat_cross`` = 2 on request).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from syndiff_pipeline.difference_imaging.masking import bits

log = logging.getLogger(__name__)

MANUAL_MASKS_BASENAME = "manual_masks.yaml"
_BIT_NAMES = {"edge": bits.EDGE, "sat_cross": bits.SAT_CROSS}
_KINDS = {
    "column": {"kind", "x", "y", "bit", "reason"},
    "rect": {"kind", "x", "y", "bit", "reason"},
    "circle": {"kind", "x", "y", "r", "bit", "reason"},
}
_ENTRY_KEYS = {"sector", "camera", "ccd", "regions"}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(float(v))


def _range(v, name: str, where: str, *, lo_limit: Optional[float] = None) -> tuple[float, float]:
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_num(a) for a in v)):
        raise ValueError(f"{where}: {name} must be a [min, max) pair of numbers, got {v!r}")
    a, b = float(v[0]), float(v[1])
    if not b > a:
        raise ValueError(f"{where}: {name} range must have max > min, got {list(v)!r}")
    return a, b


def parse_region(region: Any, where: str) -> dict:
    """Validate one region mapping and return it normalised (``bit`` resolved to an int in ``bit_value``)."""
    if not isinstance(region, dict):
        raise ValueError(f"{where}: region must be a mapping, got {type(region).__name__}")
    kind = region.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"{where}: unknown kind {kind!r}; expected one of {sorted(_KINDS)}")
    unknown = set(region) - _KINDS[kind]
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} for kind {kind!r}")
    reason = region.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{where}: missing 'reason'")
    bit = region.get("bit", "edge")
    if bit not in _BIT_NAMES:
        raise ValueError(f"{where}: bit must be one of {sorted(_BIT_NAMES)}, got {bit!r}")
    out: dict = {"kind": kind, "bit": bit, "bit_value": _BIT_NAMES[bit], "reason": reason}
    if "x" not in region:
        raise ValueError(f"{where}: missing 'x'")
    if kind == "column":
        x = region["x"]
        if _is_num(x):
            if float(x) != int(x) or int(x) < 0:
                raise ValueError(f"{where}: column x must be a non-negative integer or [min, max), got {x!r}")
            out["x"] = (int(x), int(x) + 1)
        else:
            a, b = _range(x, "x", where)
            out["x"] = (int(a), int(b))
        out["y"] = _range(region["y"], "y", where) if region.get("y") is not None else None
    elif kind == "rect":
        if "y" not in region:
            raise ValueError(f"{where}: missing 'y'")
        out["x"] = _range(region["x"], "x", where)
        out["y"] = _range(region["y"], "y", where)
    else:
        if "y" not in region or "r" not in region:
            raise ValueError(f"{where}: circle needs 'x', 'y' and 'r'")
        if not (_is_num(region["x"]) and _is_num(region["y"])):
            raise ValueError(f"{where}: circle x, y must be numbers")
        if not (_is_num(region["r"]) and float(region["r"]) > 0):
            raise ValueError(f"{where}: circle r must be a positive number, got {region['r']!r}")
        out["x"], out["y"], out["r"] = float(region["x"]), float(region["y"]), float(region["r"])
    return out


def load_manual_regions(path: str | Path, sector: int, camera: int, ccd: int) -> list[dict]:
    """Parse ``path`` and return the validated regions of this SCC, in file order (``[]`` when none)."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict) or set(raw) - {"version", "masks"}:
        raise ValueError(f"{p}: top level must be a mapping with only 'version' and 'masks'")
    if raw.get("version") != 1:
        raise ValueError(f"{p}: unsupported version {raw.get('version')!r} (expected 1)")
    masks = raw.get("masks")
    if masks is None:
        masks = []
    if not isinstance(masks, list):
        raise ValueError(f"{p}: 'masks' must be a list")
    out: list[dict] = []
    for i, entry in enumerate(masks):
        where = f"{p}: masks[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: must be a mapping")
        unknown = set(entry) - _ENTRY_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        for k in ("sector", "camera", "ccd"):
            if not isinstance(entry.get(k), int) or isinstance(entry.get(k), bool):
                raise ValueError(f"{where}: '{k}' must be an integer, got {entry.get(k)!r}")
        regions = entry.get("regions")
        if not isinstance(regions, list):
            raise ValueError(f"{where}: 'regions' must be a list")
        if (entry["sector"], entry["camera"], entry["ccd"]) != (int(sector), int(camera), int(ccd)):
            continue
        for j, region in enumerate(regions):
            out.append(parse_region(region, f"{where}.regions[{j}]"))
    return out


def regions_for_recipe(regions: list[dict]) -> list[dict]:
    """JSON/YAML-friendly canonical form (lists, no derived keys) for fingerprints and the frozen copy."""
    out = []
    for r in regions:
        d = {"kind": r["kind"], "bit": r["bit"], "reason": r["reason"]}
        for k in ("x", "y"):
            if r.get(k) is not None:
                d[k] = list(r[k]) if isinstance(r[k], tuple) else r[k]
        if "r" in r:
            d["r"] = r["r"]
        out.append(d)
    return out


def resolve_manual_mask_path(manual_mask_file: Optional[str], site_dir: str | Path | None) -> Optional[Path]:
    """``manual_mask_file`` (relative → under ``site_dir``) or ``{site_dir}/manual_masks.yaml``; None if absent."""
    if manual_mask_file:
        p = Path(manual_mask_file).expanduser()
        if not p.is_absolute() and site_dir is not None:
            p = Path(site_dir) / p
        if not p.is_file():
            raise FileNotFoundError(f"manual_mask_file {str(p)!r} not found")
        return p
    if site_dir is not None:
        p = Path(site_dir) / MANUAL_MASKS_BASENAME
        if p.is_file():
            return p
    return None


def rasterize_region(shape: tuple[int, int], crop_bounds: dict, region: dict) -> np.ndarray:
    """Boolean crop-local raster of one region (full-FFI coords, clipped to the crop)."""
    ny, nx = int(shape[0]), int(shape[1])
    x0, y0 = int(crop_bounds.get("x_min", 0)), int(crop_bounds.get("y_min", 0))
    out = np.zeros((ny, nx), dtype=bool)
    kind = region["kind"]
    if kind in ("column", "rect"):
        xa, xb = region["x"]
        ya, yb = region["y"] if region.get("y") is not None else (y0, y0 + ny)
        # pixel p is in [min, max) when min <= p < max (integer pixel indices)
        j0, j1 = int(np.ceil(xa)) - x0, int(np.ceil(xb)) - x0
        i0, i1 = int(np.ceil(ya)) - y0, int(np.ceil(yb)) - y0
        j0, j1, i0, i1 = max(j0, 0), min(j1, nx), max(i0, 0), min(i1, ny)
        if j1 > j0 and i1 > i0:
            out[i0:i1, j0:j1] = True
    else:
        cx, cy, r = region["x"] - x0, region["y"] - y0, region["r"]
        i0, i1 = max(0, int(np.floor(cy - r))), min(ny, int(np.ceil(cy + r)) + 1)
        j0, j1 = max(0, int(np.floor(cx - r))), min(nx, int(np.ceil(cx + r)) + 1)
        if i1 > i0 and j1 > j0:
            yy, xx = np.ogrid[i0:i1, j0:j1]
            out[i0:i1, j0:j1] = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return out


def apply_manual_masks(mask: np.ndarray, crop_bounds: dict, regions: Optional[list[dict]]) -> np.ndarray:
    """OR the regions into ``mask`` (returned unchanged, same object, when ``regions`` is empty/None)."""
    if not regions:
        return mask
    dtype = mask.dtype
    out = np.array(mask, copy=True)
    for k, region in enumerate(regions):
        hit = rasterize_region(out.shape, crop_bounds, region)
        newly = hit & ((out & region["bit_value"]) == 0)
        out[hit] |= np.asarray(region["bit_value"], dtype=dtype)
        log.info(
            "  manual mask [%d] %s x=%s y=%s bit=%s: %d px in crop (%d newly set) - %s",
            k, region["kind"], region.get("x"), region.get("y"), region["bit"],
            int(hit.sum()), int(newly.sum()), region["reason"],
        )
    return out


def freeze_manual_masks(regions: list[dict], lane_root: str | Path, sector: int, camera: int, ccd: int) -> Path:
    """Write the SCC-filtered entries as an immutable ``{lane}/manual_masks.yaml`` (chmod 444)."""
    path = Path(lane_root) / MANUAL_MASKS_BASENAME
    doc = {
        "version": 1,
        "masks": [{"sector": int(sector), "camera": int(camera), "ccd": int(ccd), "regions": [
            {k: v for k, v in r.items() if k != "bit"} | ({"bit": r["bit"]} if r["bit"] != "edge" else {})
            for r in regions_for_recipe(regions)]}],
    }
    if path.exists():
        os.chmod(path, 0o644)
        path.unlink()
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=None)
    os.chmod(path, 0o444)
    return path
