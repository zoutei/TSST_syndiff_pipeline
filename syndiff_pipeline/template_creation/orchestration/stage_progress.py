"""Log-derived progress for running pipeline stages.

Parses the tail of per-target stage logs so ``syndiff progress`` can
show fractional progress without importing verify/template modules or scanning
NFS artifact trees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class StageProgress:
    """StageProgress."""
    text: str
    kind: str  # "fraction" | "phase" | "elapsed"


_RE_PS1_DL_FINISHED = re.compile(r"Finished skycell .+ \((\d+)/(\d+)\)")
_RE_PS1_DL_DASK = re.compile(r"Dask progress: (\d+)/(\d+) skycells finished")
_RE_PS1_DL_TOTAL = re.compile(r"Found (\d+) total skycells to process")

_RE_PS1_PR_PROJ_ROW_PROGRESS = re.compile(
    r"\[Pipeline\] Progress: projection (\d+)/(\d+) row (\d+)/(\d+)"
)
_RE_PS1_PR_PROCESSING_PROJECTIONS = re.compile(r"\[Pipeline\] Processing (\d+) projections")
_RE_PS1_PR_ROW_STEP = re.compile(
    r"\[SequentialProcessor\] --- Processing step for row (\d+)/(\d+):"
)
_RE_PS1_PR_PROJ_FINISHED = re.compile(
    r"\[SequentialProcessor\] --- Finished sequential processing for projection:"
)

_RE_DOWN_SKYCELLS = re.compile(r"Processing (\d+) skycells in (\d+) batches")
_RE_DOWN_BATCHES = re.compile(r"Processing \d+ skycells in (\d+) batches")
_RE_DOWN_COMPLETED = re.compile(r"Completed batch (\d+)")
_RE_REMAP_EXACT_DONE = re.compile(r"Exact cache: (\d+) keys, (\d+) written")

_RE_TESS_TOTAL = re.compile(r"Downloading (\d+) FITS(?: file\(s\)| files)")
_RE_TESS_PROGRESS = re.compile(r"FFI download progress: (\d+)/(\d+)")
_RE_TESS_TQDM_FRAC = re.compile(r"(\d+)/(\d+)\s*\[")
_RE_TESS_TQDM_PCT = re.compile(r"(\d+)%\|")

_RE_MAP_SKYCELLS = re.compile(r"Processing skycells:.*?(\d+)/(\d+)")
_MAP_COMPUTE_MARKERS = (
    "Starting optimized TESS image processing",
    "Creating optimized TESS-to-skycell mapping",
    "Converting ",
    "Processing individual skycell mappings",
)

_RE_HOTPANTS_FRAMES = re.compile(
    r"hotpants \[(\w+)\] round \d+: (\d+)/(\d+) frames succeeded"
)
_RE_EPSF_FRAMES = re.compile(
    r"ePSF \[(\w+)\] round \d+: (\d+)/(\d+) frames succeeded"
)
_RE_CENTROIDS_FRAMES = re.compile(
    r"centroids \[(\w+)\]: (\d+)/(\d+) frames wrote"
)

_PHASE_LINES = (
    ("Combining results", "combining"),
    ("Saving outputs", "saving"),
    ("Loading Zarr metadata", "loading_zarr"),
    ("Precomputing shifts", "precomputing_shifts"),
    ("Getting registration files", "registration_files"),
    ("Loading skycell info", "loading_skycells"),
    ("Loading TESS data and WCS", "loading_tess"),
    ("Fetching tesscurl manifest", "fetching_manifest"),
    ("MOC filtering complete", "moc_filter"),
    ("Master skycell CSV saved", "mapping_done"),
    ("Gaia catalog saved", "gaia_done"),
    ("Starting Gaia catalog download", "gaia_download"),
)


_TAIL_CHUNK_BYTES = 65536
_MAX_TAIL_SCAN_BYTES = 4 * 1024 * 1024
_PROGRESS_CLI_MAX_TAIL_SCAN_BYTES = 512 * 1024
PROGRESS_CLI_MAX_TAIL_SCAN_BYTES = _PROGRESS_CLI_MAX_TAIL_SCAN_BYTES


def _read_tail_bytes(log_path: Path, nbytes: int) -> str:
    """Read tail bytes.
    
    Parameters
    ----------
    log_path : Path
    nbytes : int
    
    Returns
    -------
    str"""
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as fh:
            fh.seek(max(0, size - nbytes))
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _tail_text(log_path: Path, *, tail_bytes: int = _TAIL_CHUNK_BYTES) -> str:
    """Tail text.
    
    Parameters
    ----------
    log_path : Path
    tail_bytes : int, optional, default ``_TAIL_CHUNK_BYTES``
    
    Returns
    -------
    str"""
    return _read_tail_bytes(log_path, tail_bytes)


def _scan_tail_text(
    log_path: Path,
    parser,
    *,
    chunk_bytes: int = _TAIL_CHUNK_BYTES,
    max_scan_bytes: int = _MAX_TAIL_SCAN_BYTES,
) -> StageProgress | None:
    """Expand the read window backward from EOF until *parser* finds progress."""
    try:
        size = log_path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None

    scan = min(chunk_bytes, size)
    limit = min(max_scan_bytes, size)
    while True:
        result = parser(_read_tail_bytes(log_path, scan))
        if result is not None:
            return result
        if scan >= limit:
            break
        scan = min(scan + chunk_bytes, limit)
    return None


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """Last match.
    
    Parameters
    ----------
    pattern : re.Pattern[str]
    text : str
    
    Returns
    -------
    re.Match[str] | None"""
    last: re.Match[str] | None = None
    for match in pattern.finditer(text):
        last = match
    return last


def _phase_from_text(text: str) -> StageProgress | None:
    """Phase from text.
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    StageProgress | None"""
    last_label: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for needle, label in _PHASE_LINES:
            if needle in stripped:
                last_label = label
    if last_label is None:
        return None
    return StageProgress(last_label, "phase")


def _parse_ps1_download(text: str) -> StageProgress | None:
    """Parse ps1 download.
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    StageProgress | None"""
    match = _last_match(_RE_PS1_DL_FINISHED, text)
    if match:
        return StageProgress(f"{match.group(1)}/{match.group(2)}", "fraction")
    match = _last_match(_RE_PS1_DL_DASK, text)
    if match:
        return StageProgress(f"{match.group(1)}/{match.group(2)}", "fraction")
    total_match = _last_match(_RE_PS1_DL_TOTAL, text)
    if total_match and not _RE_PS1_DL_FINISHED.search(text) and not _RE_PS1_DL_DASK.search(text):
        return StageProgress(f"0/{total_match.group(1)}", "fraction")
    return _phase_from_text(text)


def _parse_ps1_process(text: str) -> StageProgress | None:
    """Parse ps1 process.
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    StageProgress | None"""
    match = _last_match(_RE_PS1_PR_PROJ_ROW_PROGRESS, text)
    if match:
        p_done, p_total, r_done, r_total = match.groups()
        return StageProgress(
            f"{p_done}/{p_total} projections {r_done}/{r_total} rows",
            "fraction",
        )

    total_proj_match = _last_match(_RE_PS1_PR_PROCESSING_PROJECTIONS, text)
    row_match = _last_match(_RE_PS1_PR_ROW_STEP, text)
    if total_proj_match and row_match:
        p_total = int(total_proj_match.group(1))
        p_done = len(_RE_PS1_PR_PROJ_FINISHED.findall(text))
        p_done = min(p_done, p_total)
        return StageProgress(
            f"{p_done}/{p_total} projections {row_match.group(1)}/{row_match.group(2)} rows",
            "fraction",
        )

    if total_proj_match:
        p_total = int(total_proj_match.group(1))
        p_done = len(_RE_PS1_PR_PROJ_FINISHED.findall(text))
        if p_done:
            return StageProgress(f"{p_done}/{p_total} projections", "fraction")
        return StageProgress(f"0/{p_total} projections", "fraction")

    return _phase_from_text(text)


def _downsample_progress_label(
    done: int, total: int, data: dict, *, prefix: str | None = None
) -> str:
    """Format skycell/composite-key fraction, tagging oversampling when os > 1."""
    os_factor = int(data.get("oversampling_factor") or 1)
    base = f"{prefix} {done}/{total}" if prefix else f"{done}/{total}"
    if os_factor > 1:
        return f"{base} os{os_factor}"
    return base


def _parse_downsample_sidecar(log_path: Path) -> StageProgress | None:
    """Parse downsample sidecar.
    
    Parameters
    ----------
    log_path : Path
    
    Returns
    -------
    StageProgress | None"""
    from syndiff_pipeline.template_creation.processing.downsample_progress import progress_path_for_log, read_progress

    data = read_progress(progress_path_for_log(log_path))
    if not data:
        return None

    phase = data.get("phase")
    if phase == "setup":
        return StageProgress("setup", "phase")
    if phase in ("combining", "saving"):
        return StageProgress(str(phase), "phase")
    if phase == "precomputing_shifts":
        offsets_done = data.get("offsets_done")
        offsets_total = data.get("offsets_total")
        if offsets_done is not None and offsets_total is not None:
            return StageProgress(f"shifts {offsets_done}/{offsets_total}", "phase")
        return StageProgress("precomputing_shifts", "phase")

    # Field L5: composite-key / skycell-batch progress
    if data.get("geometry_mode") == "field" or phase == "field_composite_keys":
        ck_total = int(data.get("composite_keys_total", 0) or 0)
        ck_done = int(data.get("composite_keys_done", 0) or 0)
        if phase == "complete" and ck_total > 0:
            return StageProgress(
                _downsample_progress_label(ck_total, ck_total, data, prefix="ckeys"),
                "fraction",
            )
        if ck_total > 0:
            return StageProgress(
                _downsample_progress_label(ck_done, ck_total, data, prefix="ckeys"),
                "fraction",
            )
        sk_total = int(data.get("total_skycells", 0) or 0)
        sk_done = int(data.get("skycells_done", 0) or 0)
        if sk_total > 0:
            return StageProgress(
                _downsample_progress_label(sk_done, sk_total, data),
                "fraction",
            )

    if phase == "complete":
        total = int(data.get("total_skycells", 0))
        if total > 0:
            return StageProgress(_downsample_progress_label(total, total, data), "fraction")

    total = int(data.get("total_skycells", 0))
    done = int(data.get("skycells_done", 0))
    if total > 0:
        return StageProgress(_downsample_progress_label(done, total, data), "fraction")
    return None


def _remap_progress_label(done: int, total: int, data: dict, *, prefix: str = "exact") -> str:
    """Format exact-cache fraction, tagging oversampling when os > 1."""
    os_factor = int(data.get("oversampling_factor") or 1)
    base = f"{prefix} {done}/{total}"
    if os_factor > 1:
        return f"{base} os{os_factor}"
    return base


def _parse_remap_sidecar(log_path: Path) -> StageProgress | None:
    """Parse remap sidecar."""
    from syndiff_pipeline.template_creation.processing.remap_progress import (
        progress_path_for_log,
        read_progress,
    )

    data = read_progress(progress_path_for_log(log_path))
    if not data:
        return None

    phase = data.get("phase")
    if phase in ("shift_schedule", "grouping"):
        return StageProgress(str(phase), "phase")

    l4a_total = int(data.get("exact_l4a_total", data.get("exact_total", 0)) or 0)
    l4a_done = int(data.get("exact_l4a_done", data.get("exact_done", 0)) or 0)
    l4b_total = int(data.get("exact_l4b_total", 0) or 0)
    l4b_done = int(data.get("exact_l4b_done", 0) or 0)

    if phase == "exact_l4a" and l4a_total > 0:
        return StageProgress(_remap_progress_label(l4a_done, l4a_total, data, prefix="l4a"), "fraction")
    if phase == "exact_l4b" and l4b_total > 0:
        return StageProgress(_remap_progress_label(l4b_done, l4b_total, data, prefix="l4b"), "fraction")
    if phase == "exact_cache" and l4a_total > 0:
        return StageProgress(_remap_progress_label(l4a_done, l4a_total, data), "fraction")

    if phase == "complete":
        if l4b_total > 0:
            return StageProgress(
                _remap_progress_label(l4b_total, l4b_total, data, prefix="l4b"),
                "fraction",
            )
        if l4a_total > 0:
            return StageProgress(_remap_progress_label(l4a_total, l4a_total, data, prefix="l4a"), "fraction")
        return StageProgress("complete", "phase")

    if l4b_total > 0:
        return StageProgress(_remap_progress_label(l4b_done, l4b_total, data, prefix="l4b"), "fraction")
    if l4a_total > 0:
        return StageProgress(_remap_progress_label(l4a_done, l4a_total, data, prefix="l4a"), "fraction")
    return None


def _parse_remap(text: str) -> StageProgress | None:
    """Parse remap log tail (fallback when sidecar is unavailable)."""
    match = _last_match(_RE_REMAP_EXACT_DONE, text)
    if match:
        total = int(match.group(1))
        return StageProgress(f"exact {total}/{total}", "fraction")
    return _phase_from_text(text)


def _parse_downsample(text: str) -> StageProgress | None:
    """Parse downsample.
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    StageProgress | None"""
    skycells_match = _last_match(_RE_DOWN_SKYCELLS, text)
    batches_match = _last_match(_RE_DOWN_BATCHES, text)
    completed_count = len(_RE_DOWN_COMPLETED.findall(text))
    if skycells_match and completed_count and batches_match:
        total_skycells = int(skycells_match.group(1))
        total_batches = int(batches_match.group(1))
        if total_batches > 0 and total_skycells > 0:
            est_done = min(
                total_skycells,
                int(round(completed_count * total_skycells / total_batches)),
            )
            return StageProgress(f"~{est_done}/{total_skycells}", "fraction")
        return StageProgress(f"batch {completed_count}/{total_batches}", "fraction")
    if batches_match and completed_count:
        total_batches = int(batches_match.group(1))
        return StageProgress(f"batch {completed_count}/{total_batches}", "fraction")
    if skycells_match:
        total_skycells = int(skycells_match.group(1))
        return StageProgress(f"0/{total_skycells}", "fraction")
    if batches_match:
        total_batches = int(batches_match.group(1))
        return StageProgress(f"batch 0/{total_batches}", "fraction")
    return _phase_from_text(text)


def _parse_tess_ffi_download(text: str) -> StageProgress | None:
    """Parse tess ffi download.
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    StageProgress | None"""
    match = _last_match(_RE_TESS_PROGRESS, text)
    if match:
        return StageProgress(f"{match.group(1)}/{match.group(2)}", "fraction")
    match = _last_match(_RE_TESS_TQDM_FRAC, text)
    if match:
        return StageProgress(f"{match.group(1)}/{match.group(2)}", "fraction")
    total_match = _last_match(_RE_TESS_TOTAL, text)
    pct_match = _last_match(_RE_TESS_TQDM_PCT, text)
    if total_match and pct_match:
        total = int(total_match.group(1))
        done = max(0, round(int(pct_match.group(1)) * total / 100))
        return StageProgress(f"{done}/{total}", "fraction")
    if total_match:
        return StageProgress(f"0/{total_match.group(1)}", "fraction")
    return _phase_from_text(text)


def _parse_mapping(text: str) -> StageProgress | None:
    """Parse mapping.
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    StageProgress | None"""
    match = _last_match(_RE_MAP_SKYCELLS, text)
    if match:
        return StageProgress(f"{match.group(1)}/{match.group(2)}", "fraction")
    # Mapping starts a legacy-named Gaia helper concurrently.  Once the main
    # coordinate/mapping work is visible, report the actual mapping phase
    # rather than the stale ``gaia_download`` marker from that helper.
    if any(marker in text for marker in _MAP_COMPUTE_MARKERS):
        return StageProgress("mapping", "phase")
    return _phase_from_text(text)


def _parse_wcs_grouping(text: str) -> StageProgress | None:
    """Parse wcs grouping.
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    StageProgress | None"""
    if text.strip():
        return StageProgress("running", "phase")
    return None


def _parse_diff_sidecar(log_path: Path) -> StageProgress | None:
    """Parse diff sidecar.
    
    Parameters
    ----------
    log_path : Path
    
    Returns
    -------
    StageProgress | None"""
    from syndiff_pipeline.difference_imaging.stages.centroids_progress import (
        format_progress_text as format_centroids_progress,
        progress_path_for_diff_log as centroids_sidecar_path,
        read_progress_merged as read_centroids_progress_merged,
    )
    from syndiff_pipeline.difference_imaging.stages.convolved_templates_progress import (
        format_progress_text as format_convolved_templates_progress,
        progress_path_for_diff_log as convolved_templates_sidecar_path,
        read_progress as read_convolved_templates_progress,
    )
    from syndiff_pipeline.difference_imaging.stages.kernel_subtract_progress import (
        format_progress_text as format_kernel_subtract_progress,
        progress_path_for_diff_log as kernel_subtract_sidecar_path,
        read_progress as read_kernel_subtract_progress,
    )
    from syndiff_pipeline.difference_imaging.stages.epsf_progress import (
        format_progress_text as format_epsf_progress,
        progress_path_for_diff_log as epsf_sidecar_path,
        read_progress_merged,
    )
    from syndiff_pipeline.difference_imaging.stages.hotpants_progress import (
        format_progress_text as format_hotpants_progress,
        progress_path_for_diff_log as hotpants_sidecar_path,
        read_progress,
    )
    from syndiff_pipeline.difference_imaging.stages.photometry_progress import (
        format_progress_text as format_photometry_progress,
        progress_path_for_diff_log as photometry_sidecar_path,
    )

    best: tuple[str, str, str] | None = None
    for sidecar_path, format_fn, read_fn in (
        (
            convolved_templates_sidecar_path(log_path),
            format_convolved_templates_progress,
            read_convolved_templates_progress,
        ),
        (
            kernel_subtract_sidecar_path(log_path),
            format_kernel_subtract_progress,
            read_kernel_subtract_progress,
        ),
        (hotpants_sidecar_path(log_path), format_hotpants_progress, read_progress),
        (epsf_sidecar_path(log_path), format_epsf_progress, read_progress_merged),
        (centroids_sidecar_path(log_path), format_centroids_progress, read_centroids_progress_merged),
        (photometry_sidecar_path(log_path), format_photometry_progress, read_progress),
    ):
        data = read_fn(sidecar_path)
        if not data:
            continue
        text = format_fn(data)
        if text is None:
            continue
        updated = str(data.get("updated_at") or "")
        if best is None or updated >= best[0]:
            phase = str(data.get("phase", ""))
            kind = "fraction" if phase != "complete" else "phase"
            best = (updated, text, kind)
    if best is None:
        return None
    return StageProgress(best[1], best[2])


def _parse_diff(text: str) -> StageProgress | None:
    """Parse diff.
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    StageProgress | None"""
    match = _last_match(_RE_HOTPANTS_FRAMES, text)
    if match:
        label, done, total = match.groups()
        return StageProgress(f"hotpants {label} complete {done}/{total}", "phase")
    match = _last_match(_RE_EPSF_FRAMES, text)
    if match:
        label, done, total = match.groups()
        return StageProgress(f"epsf {label} complete {done}/{total}", "phase")
    match = _last_match(_RE_CENTROIDS_FRAMES, text)
    if match:
        label, done, total = match.groups()
        return StageProgress(f"centroids {label} complete {done}/{total}", "phase")
    return _phase_from_text(text)


def _elapsed_progress(started_at: str | None) -> StageProgress | None:
    """Elapsed progress.
    
    Parameters
    ----------
    started_at : str | None
    
    Returns
    -------
    StageProgress | None"""
    if not started_at:
        return None
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed_s = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
    if elapsed_s < 60:
        return StageProgress(f"{elapsed_s}s", "elapsed")
    return StageProgress(f"{elapsed_s // 60}m", "elapsed")


_PARSERS = {
    "ps1_download": _parse_ps1_download,
    "ps1_process": _parse_ps1_process,
    "remap": _parse_remap,
    "downsample": _parse_downsample,
    "tess_ffi_download": _parse_tess_ffi_download,
    "mapping": _parse_mapping,
    "wcs_grouping": _parse_wcs_grouping,
    "diff": _parse_diff,
}


def read_log_progress(
    log_path: Path | str,
    stage: str,
    *,
    tail_bytes: int = _TAIL_CHUNK_BYTES,
    max_scan_bytes: int = _MAX_TAIL_SCAN_BYTES,
    started_at: str | None = None,
) -> StageProgress | None:
    """Return log-derived progress for *stage*, or None if unavailable."""
    path = Path(log_path)
    if stage == "downsample":
        sidecar_prog = _parse_downsample_sidecar(path)
        if sidecar_prog is not None:
            return sidecar_prog
    if stage == "remap":
        sidecar_prog = _parse_remap_sidecar(path)
        if sidecar_prog is not None:
            return sidecar_prog
    if stage == "diff":
        sidecar_prog = _parse_diff_sidecar(path)
        if sidecar_prog is not None:
            return sidecar_prog

    if not path.is_file():
        if stage == "wcs_grouping":
            return _elapsed_progress(started_at)
        return None

    parser = _PARSERS.get(stage)
    if parser is None:
        text = _tail_text(path, tail_bytes=tail_bytes)
        if not text.strip():
            return None
        return _elapsed_progress(started_at)

    result = _scan_tail_text(
        path,
        parser,
        chunk_bytes=tail_bytes,
        max_scan_bytes=max_scan_bytes,
    )
    if result is not None:
        return result
    if stage == "wcs_grouping":
        return _elapsed_progress(started_at)
    return None
