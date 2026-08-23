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
INTERIOR_CONTRIBS_DIRNAME = "interior_contribs"
SEAM_DELTA_CONTRIBS_DIRNAME = "seam_delta_contribs"
FITS_DIRNAME = "fits"
MATERIALIZED_FITS_SIDECAR = "materialized_fits.json"
LOCK_NAME = ".lock"

_CONTRIB_RE = re.compile(
    r"^(?P<skycell>skycell\.\d+\.\d+)_sx(?P<sx>[+-]?\d+)_sy(?P<sy>[+-]?\d+)"
    r"(?:_gid(?P<gid>\d+))?\.npz$",
    re.IGNORECASE,
)


from syndiff_pipeline.common.scc_paths import scc_templates_dir


def validate_frozen_field_geometry(
    store_root: str | Path,
    mapping_grid: Any,
    *,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the immutable geometry contract before field assembly.

    The field sidecar is the handoff between L5 and diff/bootstrap.  Once a
    ``MappingGrid`` is supplied, a missing or contradictory sidecar is a hard
    error; callers must rebuild the store instead of silently assembling with
    a different crop.  Temporal provenance is checked when requested by the
    caller (typically against the remap manifest).
    """
    if mapping_grid is None:
        raise ValueError("mapping_grid is required for frozen field geometry validation")
    sidecar_path = Path(store_root) / "field_mode_assembly.json"
    if not sidecar_path.is_file():
        raise ValueError(f"missing frozen field geometry sidecar: {sidecar_path}")
    side = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if int(side.get("schema_version", 0)) < 3:
        raise ValueError("field geometry sidecar must use schema_version >= 3")
    saved = side.get("mapping_grid")
    current = mapping_grid.to_mapping_dict()
    if not isinstance(saved, Mapping):
        raise ValueError("field geometry sidecar missing mapping_grid")
    for key in ("geometry_fingerprint", "oversampling_factor", "conv_pad_native"):
        if str(saved.get(key)) != str(current.get(key)):
            raise ValueError(
                f"field geometry mismatch for {key}: saved={saved.get(key)!r}, "
                f"requested={current.get(key)!r}"
            )
    # MAPGRID=3 is the paired-padding contract.  A sidecar that merely
    # contains a v3 MappingGrid is insufficient: the science pad is
    # deliberately neutral/invalid and T must be explicit at the handoff.
    if int(getattr(mapping_grid, "mapgrid_version", 0)) != 3:
        raise ValueError("field template assembly requires MAPGRID=3 geometry")
    if int(getattr(mapping_grid, "mapgrid_version", 0)) == 3:
        if str(side.get("science_pad_policy", "")) != "neutral_invalid":
            raise ValueError(
                "MAPGRID=3 field sidecar must declare science_pad_policy=neutral_invalid"
            )
        support = side.get("template_support_bounds_ffi")
        expected_support = {
            "x_min": int(mapping_grid.template_xmin),
            "x_max": int(mapping_grid.template_xmax),
            "y_min": int(mapping_grid.template_ymin),
            "y_max": int(mapping_grid.template_ymax),
        }
        if support != expected_support:
            raise ValueError(
                "MAPGRID=3 field sidecar template_support_bounds_ffi does not match T"
            )
        pads = side.get("pad_native")
        expected_pads = {
            "left": int(mapping_grid.pad_left),
            "right": int(mapping_grid.pad_right),
            "bottom": int(mapping_grid.pad_bottom),
            "top": int(mapping_grid.pad_top),
        }
        if pads != expected_pads:
            raise ValueError("MAPGRID=3 field sidecar pad_native does not match MappingGrid")
    shape = tuple(int(v) for v in side.get("base_tess_shape", ()))
    expected_shape = tuple(int(v) for v in mapping_grid.array_shape_os())
    if shape and shape != expected_shape:
        raise ValueError(
            f"field geometry base_tess_shape {shape} != MappingGrid OS shape {expected_shape}"
        )
    saved_prov = dict(side.get("geometry_provenance") or {})
    for key, value in (expected_provenance or {}).items():
        if value is None:
            continue
        if str(saved_prov.get(key)) != str(value):
            raise ValueError(
                f"field temporal provenance mismatch for {key}: "
                f"saved={saved_prov.get(key)!r}, requested={value!r}"
            )
    return side


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
    *,
    oversampling_factor: int = 1,
    mapping_grid: Any | None = None,
) -> tuple[int, int, int, int] | None:
    if roi_bounds is None:
        return None
    oversampling_factor = int(oversampling_factor)
    if oversampling_factor < 1:
        raise ValueError(
            f"oversampling_factor must be >= 1, got {oversampling_factor}"
        )
    x_min, y_min, x_max, y_max = (int(v) for v in roi_bounds)
    if mapping_grid is not None:
        # ``roi_bounds`` are native FFI coordinates, while assembled arrays
        # are indexed in the local, oversampled MappingGrid coordinate frame.
        # In particular, ffi_ymin may be negative because the grid includes
        # convolution-pad rows; passing it directly to a NumPy slice turns it
        # into a from-the-end index and can produce an empty image.
        x_min -= int(mapping_grid.ffi_xmin)
        x_max -= int(mapping_grid.ffi_xmin)
        y_min -= int(mapping_grid.ffi_ymin)
        y_max -= int(mapping_grid.ffi_ymin)
    return (
        x_min * oversampling_factor,
        x_max * oversampling_factor,
        y_min * oversampling_factor,
        y_max * oversampling_factor,
    )


def build_field_fits_header(
    *,
    sector: int,
    camera: int,
    ccd: int,
    group_id: int,
    oversampling_factor: int = 1,
    roi_bounds: tuple[int, int, int, int] | None = None,
    provenance: Mapping[str, Any] | None = None,
    mapping_grid: Any | None = None,
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
    if mapping_grid is not None:
        for key, value in mapping_grid.to_fits_header_updates().items():
            # MappingGrid metadata is mostly integral geometry, but the
            # coordinate-frame declaration is intentionally textual.
            # Preserve FITS-compatible strings instead of coercing every
            # value to int.
            serialized = value if isinstance(value, (str, bool)) else int(value)
            hdr[key] = (serialized, f"MappingGrid {key}")
    elif roi_bounds is not None:
        x_min, y_min, x_max, y_max = (int(v) for v in roi_bounds)
        hdr["XMIN"] = (x_min, "ROI xmin in base TESS pixels")
        hdr["XMAX"] = (x_max, "ROI xmax (exclusive) in base TESS pixels")
        hdr["YMIN"] = (y_min, "ROI ymin in base TESS pixels")
        hdr["YMAX"] = (y_max, "ROI ymax (exclusive) in base TESS pixels")
        hdr["ROIW"] = (x_max - x_min, "ROI width in base TESS pixels")
        hdr["ROIH"] = (y_max - y_min, "ROI height in base TESS pixels")
    if provenance:
        # Keep compact geometry provenance in every materialized FITS.  The
        # complete values are retained in the JSON manifest/sidecar.
        for key, prov_key, comment in (
            ("TVWCSVER", "temporal_wcs_version", "Temporal WCS model version"),
            ("TVWCSFP", "temporal_wcs_fingerprint", "Temporal WCS fingerprint"),
            ("MAPFP", "mapping_fingerprint", "Mapping fingerprint"),
            ("REMAPFP", "remap_fingerprint", "Remap fingerprint"),
        ):
            value = provenance.get(prov_key)
            if value not in (None, ""):
                hdr[key] = (str(value), comment)
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
    """Write one group's flux-sum template FITS (+ COUNT extension) as ``.fits.fz``."""
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


def native_debug_fits_basename(source_name: str, *, oversampling_factor: int) -> str:
    """Name a native debug FITS derived from an oversampled FITS."""
    factor = int(oversampling_factor)
    if factor < 2:
        raise ValueError(f"native debug conversion requires oversampling >= 2, got {factor}")
    source = Path(source_name).name
    suffix = ".fits.fz" if source.endswith(".fits.fz") else ".fits"
    stem = source[: -len(suffix)] if source.endswith(suffix) else Path(source).stem
    return f"{stem}_native{suffix}"


def _sum_native_blocks(array: np.ndarray, *, oversampling_factor: int) -> np.ndarray:
    factor = int(oversampling_factor)
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError(f"native debug conversion requires 2-D arrays, got shape {values.shape}")
    height, width = values.shape
    if height % factor or width % factor:
        raise ValueError(
            f"array shape {values.shape} is not divisible by oversampling factor {factor}"
        )
    return values.reshape(height // factor, factor, width // factor, factor).sum(axis=(1, 3))


def write_native_debug_fits(
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
    oversampling_factor: int,
) -> str:
    """Write a native-scale debug FITS by summing each oversampled block."""
    from astropy.io import fits

    from syndiff_pipeline.common.fits_io import write_hdul_fits

    source = Path(source_path)
    factor = int(oversampling_factor)
    if factor < 2:
        raise ValueError(f"native debug conversion requires oversampling >= 2, got {factor}")
    with fits.open(source, memmap=False) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            raise ValueError(f"source FITS has no flux image extension: {source}")
        flux = _sum_native_blocks(hdul[1].data, oversampling_factor=factor)
        if len(hdul) < 3 or hdul[2].data is None:
            raise ValueError(f"source FITS has no COUNT image extension: {source}")
        count = _sum_native_blocks(hdul[2].data, oversampling_factor=factor)
        header = hdul[1].header.copy()

    header["OVERSAMP"] = (1, "Native pixel scale")
    header["SRCOS"] = (factor, "Source oversampling factor")
    header["BLKSUM"] = (factor, f"{factor}x{factor} block sum")
    count_header = header.copy()
    count_header["EXTNAME"] = "COUNT"
    destination = Path(output_path) if output_path is not None else source.with_name(
        native_debug_fits_basename(source.name, oversampling_factor=factor)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    native_hdul = fits.HDUList(
        [
            fits.PrimaryHDU(flux.astype(np.float32), header=header),
            fits.ImageHDU(count.astype(np.float32), header=count_header, name="COUNT"),
        ]
    )
    return write_hdul_fits(destination, native_hdul)


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


def load_contrib(path: str | Path, keys: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Load a contrib NPZ.

    ``keys`` restricts which members are decompressed (NpzFile access is
    lazy per-member); omit to load everything, as before.
    """
    with np.load(path, allow_pickle=False) as z:
        wanted = z.files if keys is None else [k for k in keys if k in z.files]
        return {k: z[k] for k in wanted}


# ---------------------------------------------------------------------------
# Interior/seam-delta contrib split (H.1, 2026-08-23).
#
# The plain ``contribs/`` files above are group-qualified because a thin
# neighbor-shift-dependent border correction is baked into every write --
# see ``field_hybrid_exact.py::compose_group_hybrid_assignment``'s
# ``apply_inter_skycell`` patch. Everywhere except that thin border, a
# skycell's contrib at a given ``(sx_int, sy_int)`` is byte-identical no
# matter which ``group_id`` it's written under (many groups share it). The
# functions below split that fully-corrected write into two pieces so the
# group-independent majority can be written/read once and reused, instead
# of duplicated under every group that shares it:
#
# - "interior": the intra-skycell-only result (``apply_inter_skycell=False``
#   at compose time), keyed by ``(skycell, sx_int, sy_int)`` -- no group_id.
# - "seam delta": the sparse difference the inter-skycell rim patch made,
#   keyed by ``(skycell, sx_int, sy_int, group_id)`` -- small, group-specific.
#
# ``interior + seam_delta`` reconstructs the plain ``contribs/`` value
# exactly, by construction (it *is* that difference) -- see
# ``assemble_group_from_split_contribs`` below and its regression test
# against ``assemble_group_from_contribs``.


def interior_contrib_basename(skycell: str, sx_int: int, sy_int: int) -> str:
    """Filename for one group-independent interior contrib (no ``_gid``)."""
    return contrib_basename(skycell, sx_int, sy_int, group_id=None)


def interior_contrib_path(store_root: str | Path, skycell: str, sx_int: int, sy_int: int) -> Path:
    return Path(store_root) / INTERIOR_CONTRIBS_DIRNAME / interior_contrib_basename(
        skycell, sx_int, sy_int
    )


def seam_delta_contrib_path(
    store_root: str | Path, skycell: str, sx_int: int, sy_int: int, group_id: int
) -> Path:
    return Path(store_root) / SEAM_DELTA_CONTRIBS_DIRNAME / contrib_basename(
        skycell, sx_int, sy_int, group_id=group_id
    )


def write_interior_contrib(
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
    """Write the group-independent interior contrib (atomic temp+replace)."""
    import os
    import tempfile

    root = Path(store_root)
    out = interior_contrib_path(root, skycell, sx_int, sy_int)
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
        prefix=f".{out.name}.", suffix=".tmp.npz", dir=str(out.parent)
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


def write_seam_delta_contrib(
    store_root: str | Path,
    skycell: str,
    sx_int: int,
    sy_int: int,
    *,
    indices: np.ndarray,
    flux_sum: np.ndarray,
    count: np.ndarray,
    mask_count: np.ndarray | None,
    group_id: int,
) -> Path:
    """Write the small, group-specific seam-delta contrib (atomic temp+replace).

    ``indices``/``flux_sum``/``count``/``mask_count`` should already be the
    sparse *difference* (full - interior), e.g. from
    :func:`compute_seam_delta`; an empty (zero-length) delta is still
    written (as an empty array), so its presence on disk means "this
    skycell/shift/group combination was checked and found to need no
    correction" rather than "not yet computed".
    """
    import os
    import tempfile

    root = Path(store_root)
    out = seam_delta_contrib_path(root, skycell, sx_int, sy_int, group_id)
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
        prefix=f".{out.name}.", suffix=".tmp.npz", dir=str(out.parent)
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


def compute_seam_delta(
    *,
    idx_full: np.ndarray,
    flux_full: np.ndarray,
    count_full: np.ndarray,
    mcount_full: np.ndarray | None,
    idx_interior: np.ndarray,
    flux_interior: np.ndarray,
    count_interior: np.ndarray,
    mcount_interior: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Sparse ``full - interior`` difference, restricted to nonzero entries.

    Both inputs are one skycell's own (indices, flux_sum, count, mask_count)
    tuples -- ``full`` from binning the fully-corrected (intra+inter)
    assignment, ``interior`` from binning the intra-only assignment. Uses
    each side's OWN index set (not the whole SCC grid), so cost scales with
    this one skycell's footprint, not the full array. Exact by construction:
    ``interior + seam_delta == full`` for every touched pixel, since
    ``seam_delta`` *is* that arithmetic difference.
    """
    idx_full = np.asarray(idx_full, dtype=np.int64)
    idx_interior = np.asarray(idx_interior, dtype=np.int64)
    union_idx = np.union1d(idx_full, idx_interior)

    def _scatter(idx: np.ndarray, val: np.ndarray) -> np.ndarray:
        out = np.zeros(union_idx.shape, dtype=np.float64)
        if idx.size:
            pos = np.searchsorted(union_idx, idx)
            np.add.at(out, pos, np.asarray(val, dtype=np.float64))
        return out

    flux_delta = _scatter(idx_full, flux_full) - _scatter(idx_interior, flux_interior)
    count_delta = _scatter(idx_full, count_full) - _scatter(idx_interior, count_interior)
    if mcount_full is not None and mcount_interior is not None:
        mcount_delta = _scatter(idx_full, mcount_full) - _scatter(idx_interior, mcount_interior)
    else:
        mcount_delta = None

    keep = flux_delta != 0.0
    if mcount_delta is not None:
        keep = keep | (count_delta != 0.0) | (mcount_delta != 0.0)
    else:
        keep = keep | (count_delta != 0.0)

    out_idx = union_idx[keep]
    out_flux = flux_delta[keep]
    out_count = count_delta[keep]
    out_mcount = mcount_delta[keep] if mcount_delta is not None else None
    return out_idx, out_flux, out_count, out_mcount


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
    provenance: dict[str, Any] | None = None

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
            "provenance": dict(self.provenance or {}),
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
    need_count: bool = True,
    need_mask: bool = True,
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
    need_count, need_mask
        When ``False``, skip allocating/accumulating that plane entirely
        (returned as a zero-size placeholder) and skip decompressing its
        NPZ member. Callers that only want ``flux_sum`` (e.g. the hotpants
        field-mode template loader, which is on a per-frame hot path) can
        cut out roughly two-thirds of this function's array work by passing
        both as ``False``.
    """
    ny, nx = int(shape[0]), int(shape[1])
    flux = np.zeros(ny * nx, dtype=np.float64)
    count = np.zeros(ny * nx, dtype=np.float64) if need_count else None
    mask_count = np.zeros(ny * nx, dtype=np.float64) if need_mask else None
    load_keys = ["indices", "flux_sum"]
    if need_count:
        load_keys.append("count")
    if need_mask:
        load_keys.append("mask_count")
    root = Path(store_root)
    n_loaded = 0
    for skycell, sx_i, sy_i in shifts:
        path = contrib_path(root, skycell, sx_i, sy_i, group_id=group_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing field contrib: {path}")
        data = load_contrib(path, keys=load_keys)
        idx = np.asarray(data["indices"], dtype=np.int64)
        flux[idx] += np.asarray(data["flux_sum"], dtype=np.float64)
        if count is not None:
            count[idx] += np.asarray(data["count"], dtype=np.float64)
        if mask_count is not None and "mask_count" in data:
            mask_count[idx] += np.asarray(data["mask_count"], dtype=np.float64)
        n_loaded += 1
    flux_2d = flux.reshape(ny, nx)
    count_2d = count.reshape(ny, nx) if count is not None else np.zeros((0, 0), dtype=np.float64)
    mask_2d = mask_count.reshape(ny, nx) if mask_count is not None else np.zeros((0, 0), dtype=np.float64)
    if crop is not None:
        x0, x1, y0, y1 = (int(v) for v in crop)
        flux_2d = flux_2d[y0:y1, x0:x1]
        if count is not None:
            count_2d = count_2d[y0:y1, x0:x1]
        if mask_count is not None:
            mask_2d = mask_2d[y0:y1, x0:x1]
    return {
        "flux_sum": flux_2d,
        "count": count_2d,
        "mask_count": mask_2d,
        "n_contribs": np.asarray(n_loaded, dtype=np.int32),
    }


def assemble_group_from_split_contribs(
    store_root: str | Path,
    shifts: Sequence[tuple[str, int, int]],
    *,
    shape: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
    group_id: int | None = None,
    need_count: bool = True,
    need_mask: bool = True,
) -> dict[str, np.ndarray]:
    """Sum interior + seam-delta contribs for one group -- must reproduce
    :func:`assemble_group_from_contribs`'s output exactly.

    Same signature/return shape as :func:`assemble_group_from_contribs`;
    for each ``(skycell, sx_int, sy_int)`` in *shifts*, sums the
    group-independent interior contrib plus (if present) this group's
    sparse seam-delta contrib. A missing seam-delta file means "no
    correction was needed for this skycell in this group" (see
    :func:`write_seam_delta_contrib`), not an error.
    """
    ny, nx = int(shape[0]), int(shape[1])
    flux = np.zeros(ny * nx, dtype=np.float64)
    count = np.zeros(ny * nx, dtype=np.float64) if need_count else None
    mask_count = np.zeros(ny * nx, dtype=np.float64) if need_mask else None
    load_keys = ["indices", "flux_sum"]
    if need_count:
        load_keys.append("count")
    if need_mask:
        load_keys.append("mask_count")
    root = Path(store_root)
    n_loaded = 0
    for skycell, sx_i, sy_i in shifts:
        interior_path = interior_contrib_path(root, skycell, sx_i, sy_i)
        if not interior_path.is_file():
            raise FileNotFoundError(f"missing interior contrib: {interior_path}")
        data = load_contrib(interior_path, keys=load_keys)
        idx = np.asarray(data["indices"], dtype=np.int64)
        flux[idx] += np.asarray(data["flux_sum"], dtype=np.float64)
        if count is not None:
            count[idx] += np.asarray(data["count"], dtype=np.float64)
        if mask_count is not None and "mask_count" in data:
            mask_count[idx] += np.asarray(data["mask_count"], dtype=np.float64)

        if group_id is not None:
            delta_path = seam_delta_contrib_path(root, skycell, sx_i, sy_i, group_id)
            if delta_path.is_file():
                ddata = load_contrib(delta_path, keys=load_keys)
                didx = np.asarray(ddata["indices"], dtype=np.int64)
                if didx.size:
                    flux[didx] += np.asarray(ddata["flux_sum"], dtype=np.float64)
                    if count is not None and "count" in ddata:
                        count[didx] += np.asarray(ddata["count"], dtype=np.float64)
                    if mask_count is not None and "mask_count" in ddata:
                        mask_count[didx] += np.asarray(ddata["mask_count"], dtype=np.float64)
        n_loaded += 1
    flux_2d = flux.reshape(ny, nx)
    count_2d = count.reshape(ny, nx) if count is not None else np.zeros((0, 0), dtype=np.float64)
    mask_2d = mask_count.reshape(ny, nx) if mask_count is not None else np.zeros((0, 0), dtype=np.float64)
    if crop is not None:
        x0, x1, y0, y1 = (int(v) for v in crop)
        flux_2d = flux_2d[y0:y1, x0:x1]
        if count is not None:
            count_2d = count_2d[y0:y1, x0:x1]
        if mask_count is not None:
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
