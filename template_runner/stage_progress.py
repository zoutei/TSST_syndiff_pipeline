"""Log-derived progress for running pipeline stages.

Parses the tail of per-target stage logs so ``syndiff-template progress`` can
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

_RE_TESS_TOTAL = re.compile(r"Downloading (\d+) FITS(?: file\(s\)| files)")
_RE_TESS_PROGRESS = re.compile(r"FFI download progress: (\d+)/(\d+)")
_RE_TESS_TQDM_FRAC = re.compile(r"(\d+)/(\d+)\s*\[")
_RE_TESS_TQDM_PCT = re.compile(r"(\d+)%\|")

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


def _tail_text(log_path: Path, *, tail_bytes: int = 65536) -> str:
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
            data = fh.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    last: re.Match[str] | None = None
    for match in pattern.finditer(text):
        last = match
    return last


def _phase_from_text(text: str) -> StageProgress | None:
    last_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            last_line = stripped
    for needle, label in _PHASE_LINES:
        if needle in last_line or needle in text:
            return StageProgress(label, "phase")
    return None


def _parse_ps1_download(text: str) -> StageProgress | None:
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


def _parse_downsample(text: str) -> StageProgress | None:
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
    return _phase_from_text(text)


def _parse_wcs_grouping(text: str) -> StageProgress | None:
    if text.strip():
        return StageProgress("running", "phase")
    return None


def _elapsed_progress(started_at: str | None) -> StageProgress | None:
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
    "downsample": _parse_downsample,
    "tess_ffi_download": _parse_tess_ffi_download,
    "mapping": _parse_mapping,
    "wcs_grouping": _parse_wcs_grouping,
}


def read_log_progress(
    log_path: Path | str,
    stage: str,
    *,
    tail_bytes: int = 65536,
    started_at: str | None = None,
) -> StageProgress | None:
    """Return log-derived progress for *stage*, or None if unavailable."""
    path = Path(log_path)
    if not path.is_file():
        if stage == "wcs_grouping":
            return _elapsed_progress(started_at)
        return None

    text = _tail_text(path, tail_bytes=tail_bytes)
    if not text.strip():
        if stage == "wcs_grouping":
            return _elapsed_progress(started_at)
        return None

    parser = _PARSERS.get(stage)
    if parser is None:
        return _elapsed_progress(started_at)

    result = parser(text)
    if result is not None:
        return result
    if stage == "wcs_grouping":
        return _elapsed_progress(started_at)
    return None
