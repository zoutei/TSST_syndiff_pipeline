"""
download.py
===========
Download calibrated TESS Full Frame Images (FFICs) for a given sector/camera/CCD.

By default this uses the STScI tesscurl sector script only as a **URL manifest**:
``https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_<N>_ffic.sh``
lines are parsed for MAST download URLs; FITS are fetched with :mod:`urllib` (no subprocess curl).

Default local layout is nested: ``data/tess_ffi/s{sector:04d}/cam{camera}_ccd{ccd}/``.

Usage (CLI):
    python -m syndiff_pipeline.common.download --sector 20 --camera 3 --ccd 3
    # or:
    python -m syndiff_pipeline.common.download --sector 20 --camera 3 --ccd 3 \
        --output-dir data/tess_ffi/s0020/cam3_ccd3/

Usage (Python):
    from syndiff_pipeline.common.download import download_ffis, nested_ffi_dir
    paths = download_ffis(
        sector=20, camera=3, ccd=3, output_dir=nested_ffi_dir(20, 3, 3),
    )
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

log = logging.getLogger(__name__)

TESSCURL_SCRIPT_URL = (
    "https://archive.stsci.edu/missions/tess/download_scripts/sector/"
    "tesscurl_sector_{sector}_ffic.sh"
)

# tesscurl lines: curl ... -o <file> <url>
_CURL_LINE_RE = re.compile(
    r"^\s*curl\b.*\s-o\s+(?P<out>\"[^\"]+\"|'[^']+'|\S+)\s+(?P<url>\S+)",
    re.IGNORECASE,
)

_DOWNLOAD_TIMEOUT_SCRIPT_S = 120.0
_DOWNLOAD_TIMEOUT_FITS_S = 600.0
_CHUNK_BYTES = 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b"
_GZIP_COMPRESSLEVEL = 1
_DEFAULT_MAX_WORKERS = 8
_USER_AGENT = "syndiff_pipeline/TESS-FFI"

_HTTP_ERROR_HELP = (
    " If tesscurl is missing or the archive is down, retry later."
)


def _progress_iterate(length: int, desc: str):
    """Yield ``range(length)`` wrapped in ``tqdm`` when available."""
    r = range(length)
    try:
        from tqdm import tqdm

        return tqdm(r, desc=desc, unit="file")
    except ImportError:
        log.info("Install tqdm to show a download progress bar.")
        return r


def nested_ffi_dir(sector: int, camera: int, ccd: int, root: str = "data/tess_ffi") -> str:
    """
    Conventional nested directory for one sector/camera/CCD under ``root``.

    Example: ``data/tess_ffi/s0020/cam3_ccd3``
    """
    return str(Path(root) / f"s{sector:04d}" / f"cam{camera}_ccd{ccd}")


FFIC_FPACK_SUFFIX = ".fits.fz"
FFIC_GZIP_SUFFIX = ".fits.gz"
FFIC_PLAIN_SUFFIX = ".fits"


def spoc_ffi_fpack_basename(spoc_basename: str) -> str:
    """``tess..._ffic.fits`` → ``tess..._ffic.fits.fz``."""
    base = os.path.basename(str(spoc_basename))
    lower = base.lower()
    if lower.endswith(FFIC_FPACK_SUFFIX):
        return base
    if lower.endswith(FFIC_GZIP_SUFFIX):
        return base[: -len(FFIC_GZIP_SUFFIX)] + FFIC_FPACK_SUFFIX
    if lower.endswith(FFIC_PLAIN_SUFFIX):
        return base + ".fz"
    return base + FFIC_FPACK_SUFFIX


def spoc_ffi_gzip_basename(spoc_basename: str) -> str:
    """``tess..._ffic.fits`` → ``tess..._ffic.fits.gz`` (legacy)."""
    base = os.path.basename(str(spoc_basename))
    if base.lower().endswith(FFIC_GZIP_SUFFIX):
        return base
    if base.lower().endswith(FFIC_FPACK_SUFFIX):
        return base[: -len(FFIC_FPACK_SUFFIX)] + FFIC_GZIP_SUFFIX
    if base.lower().endswith(FFIC_PLAIN_SUFFIX):
        return base + ".gz"
    return base + FFIC_GZIP_SUFFIX


def spoc_ffi_basename_from_local(path_or_name: str) -> str:
    """Map a local FFI path to the tesscurl manifest basename (``.fits``)."""
    return manifest_basename_from_local(path_or_name)


def is_spoc_ffi_filename(name: str) -> bool:
    """True for SPOC calibrated FFI basenames (``.fits.fz`` / ``.gz`` / ``.fits``)."""
    lower = os.path.basename(str(name)).lower()
    return (
        lower.endswith("_ffic.fits.fz")
        or lower.endswith("_ffic.fits.gz")
        or lower.endswith("_ffic.fits")
    )


def resolve_local_ffi_path(directory: str, spoc_basename: str) -> str | None:
    """Return on-disk FFI path, preferring ``.fits.fz`` over ``.gz`` over plain."""
    from syndiff_pipeline.common.fits_variants import strip_fits_storage_suffix

    stem = strip_fits_storage_suffix(os.path.basename(str(spoc_basename)))
    # Manifest basenames are ``…_ffic.fits``; stem is ``…_ffic``.
    from syndiff_pipeline.common.fits_variants import resolve_stem_in_directory

    return resolve_stem_in_directory(directory, stem)


def ffi_glob_patterns(sector: int, camera: int, ccd: int) -> list[str]:
    """Glob patterns for SPOC FFIs (fpack first, then gzip, then plain)."""
    return [
        _ffi_filename_pattern(sector, camera, ccd, kind="fz"),
        _ffi_filename_pattern(sector, camera, ccd, kind="gz"),
        _ffi_filename_pattern(sector, camera, ccd, kind="plain"),
    ]


def _ffi_filename_pattern(
    sector: int, camera: int, ccd: int, *, kind: str = "fz"
) -> str:
    """Return glob pattern for TESS FFI calibrated files."""
    if kind == "fz":
        suffix = "_ffic.fits.fz"
    elif kind == "gz":
        suffix = "_ffic.fits.gz"
    else:
        suffix = "_ffic.fits"
    return f"tess*-s{sector:04d}-{camera}-{ccd}-*{suffix}"


def _ffic_product_basename_matches(
    product_filename: str, sector: int, camera: int, ccd: int
) -> bool:
    """
    True if ``productFilename`` is a calibrated FFI for exactly this sector/camera/CCD.

    SPOC names look like ``tess2020019142923-s0020-3-3-0165-s_ffic.fits`` (manifest)
    or ``..._ffic.fits.fz`` / ``.fits.gz`` on disk after compression migration.
    """
    base = os.path.basename(str(product_filename))
    pat = re.compile(
        rf"^tess[0-9]+-s{sector:04d}-{camera}-{ccd}-.+_ffic\.fits(?:\.fz|\.gz)?$",
        re.IGNORECASE,
    )
    return pat.match(base) is not None


def manifest_basename_from_local(path_or_name: str) -> str:
    """Map a local FFI path to the tesscurl manifest basename (``.fits``)."""
    from syndiff_pipeline.common.fits_variants import fits_logical_path

    return os.path.basename(fits_logical_path(path_or_name))


def local_ffi_manifest_basenames(paths: list[str]) -> set[str]:
    """Manifest basenames for local FFI paths (``.fits`` / ``.gz`` / ``.fz``)."""
    return {manifest_basename_from_local(p) for p in paths}


def compress_spoc_ffi_to_fpack(plain_path: str) -> str:
    """Compress ``path.fits`` to ``path.fits.fz`` and remove the uncompressed file."""
    from syndiff_pipeline.common.fits_io import fpack_plain_fits

    return str(fpack_plain_fits(plain_path, delete_plain=True))


def compress_spoc_ffi_to_gzip(plain_path: str) -> str:
    """Legacy: compress ``path.fits`` to ``path.fits.gz`` (prefer :func:`compress_spoc_ffi_to_fpack`)."""
    return _gzip_fits_file(plain_path)


def _gzip_fits_file(fits_path: str) -> str:
    """Compress ``path.fits`` to ``path.fits.gz`` and remove the uncompressed file."""
    import gzip
    import shutil

    gz_path = fits_path + ".gz"
    part = gz_path + ".part"
    try:
        with open(fits_path, "rb") as f_in, gzip.open(part, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.replace(part, gz_path)
        os.remove(fits_path)
    except BaseException:
        if os.path.isfile(part):
            try:
                os.remove(part)
            except OSError:
                pass
        raise
    return gz_path


def list_local_ffis(ffi_dir: str, sector: int, camera: int, ccd: int) -> list:
    """
    Glob for already-downloaded FFI files matching sector/camera/CCD.

    Prefers ``.fits.fz`` over ``.fits.gz`` over legacy ``.fits`` when both exist.

    Parameters
    ----------
    ffi_dir : str
        Directory to search.
    sector, camera, ccd : int
        TESS sector, camera, and CCD numbers.

    Returns
    -------
    list of str
        Sorted list of absolute file paths.
    """
    from syndiff_pipeline.common.fits_variants import storage_suffix_rank

    by_manifest: dict[str, str] = {}
    for kind in ("fz", "gz", "plain"):
        pattern = os.path.join(
            ffi_dir, _ffi_filename_pattern(sector, camera, ccd, kind=kind)
        )
        for path in sorted(glob.glob(pattern)):
            key = manifest_basename_from_local(path)
            existing = by_manifest.get(key)
            if existing is None or storage_suffix_rank(path) < storage_suffix_rank(
                existing
            ):
                by_manifest[key] = path
    return [by_manifest[k] for k in sorted(by_manifest)]


def _fetch_bytes(url: str, timeout: float) -> bytes:
    """Fetch bytes.
    
    Parameters
    ----------
    url : str
    timeout : float
    
    Returns
    -------
    bytes"""
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _strip_shell_quote(s: str) -> str:
    """Strip shell quote.
    
    Parameters
    ----------
    s : str
    
    Returns
    -------
    str"""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def parse_tesscurl_script(text: str) -> List[Tuple[str, str]]:
    """
    Parse a tesscurl ``.sh`` body into ``(fits_basename, download_url)`` pairs.

    Each relevant line contains ``curl ... -o <file> <url>``.
    """
    pairs: List[Tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _CURL_LINE_RE.match(line)
        if not m:
            continue
        out = _strip_shell_quote(m.group("out"))
        url = m.group("url").strip().rstrip("'\"")
        if out and url:
            pairs.append((os.path.basename(out), url))
    return pairs


def tesscurl_script_path(output_dir: str, sector: int) -> str:
    """Path where a tesscurl sector manifest is cached under ``output_dir``."""
    return os.path.join(output_dir, f"tesscurl_sector_{sector}_ffic.sh")


def load_tesscurl_script_text(
    sector: int,
    output_dir: str | None = None,
    *,
    local_only: bool = False,
) -> str | None:
    """Load tesscurl manifest text from a cached script or MAST.

    When ``local_only`` is True (artifact verify), only the on-disk cache is read;
  never contact MAST.
    """
    if output_dir:
        cached = tesscurl_script_path(output_dir, sector)
        if os.path.isfile(cached):
            try:
                return Path(cached).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log.warning("Could not read cached tesscurl script %s: %s", cached, exc)

    if local_only:
        return None

    script_url = TESSCURL_SCRIPT_URL.format(sector=sector)
    try:
        script_bytes = _fetch_bytes(script_url, _DOWNLOAD_TIMEOUT_SCRIPT_S)
    except (HTTPError, URLError) as exc:
        log.debug("Could not fetch tesscurl script for sector %s: %s", sector, exc)
        return None
    return script_bytes.decode("utf-8", errors="replace")


def expected_ffi_basenames(
    sector: int,
    camera: int,
    ccd: int,
    output_dir: str | None = None,
    *,
    local_only: bool = False,
) -> list[str] | None:
    """Return sorted expected FFI basenames from the tesscurl manifest.

    Returns ``None`` when the manifest cannot be loaded.
    """
    script_text = load_tesscurl_script_text(
        sector, output_dir, local_only=local_only
    )
    if script_text is None:
        return None
    pairs = parse_tesscurl_script(script_text)
    return sorted(
        bn
        for bn, _ in pairs
        if _ffic_product_basename_matches(bn, sector, camera, ccd)
    )


def _stream_url_to_file(url: str, dest_path: str, timeout: float) -> None:
    """Stream ``url`` to ``dest_path`` (atomic replace on success)."""
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    part = dest_path + ".part"
    try:
        with urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
            while True:
                chunk = resp.read(_CHUNK_BYTES)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(part, dest_path)
    except BaseException:
        if os.path.isfile(part):
            try:
                os.remove(part)
            except OSError:
                pass
        raise


_FFI_LIST_FLUSH_BATCH = 20


class _FfiListIngestBuffer:
    """Thread-safe buffer for batched ``ffi_list`` upserts during parallel download."""

    def __init__(self, ffi_list_path: str | Path) -> None:
        self._path = Path(ffi_list_path)
        self._rows: list[dict] = []
        self._lock = threading.Lock()

    def add(self, row: dict) -> None:
        with self._lock:
            self._rows.append(row)
            if len(self._rows) >= _FFI_LIST_FLUSH_BATCH:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._rows:
            return
        from syndiff_pipeline.common.wcs_header_cache import upsert_ffi_list_rows

        upsert_ffi_list_rows(self._path, self._rows)
        self._rows = []


def _stream_url_to_fpack_fits(
    url: str,
    fz_dest_path: str,
    timeout: float,
    *,
    on_plain_fits: Callable[[str], None] | None = None,
) -> None:
    """
    Download ``url`` to a plain FITS temp, then ``fpack -F`` to ``fz_dest_path``.

    If the HTTP payload is already gzip-compressed (MAST FITS+GZIP), bytes are
    gunzipped to plain FITS first. Plain FITS payloads are written as-is before
    fpack. Never leaves a plain ``.fits`` beside the final ``.fits.fz``.
    """
    import gzip
    import io

    from syndiff_pipeline.common.fits_io import fpack_plain_fits
    from syndiff_pipeline.common.fits_variants import fits_logical_path

    fz_dest = Path(fz_dest_path)
    plain_path = Path(fits_logical_path(fz_dest))
    part = str(plain_path) + ".part"
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(req, timeout=timeout) as resp:
            first_chunk = resp.read(_CHUNK_BYTES)
            if not first_chunk:
                raise OSError(f"Empty response from {url}")

            already_gzip = first_chunk[:2] == _GZIP_MAGIC
            if already_gzip:
                # Accumulate gzip payload then decompress to plain FITS.
                buf = io.BytesIO()
                buf.write(first_chunk)
                while True:
                    chunk = resp.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    buf.write(chunk)
                buf.seek(0)
                with gzip.GzipFile(fileobj=buf) as gz_in, open(part, "wb") as out:
                    while True:
                        chunk = gz_in.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        out.write(chunk)
            else:
                with open(part, "wb") as fh:
                    fh.write(first_chunk)
                    while True:
                        chunk = resp.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        fh.write(chunk)
        os.replace(part, plain_path)
        if on_plain_fits is not None:
            on_plain_fits(str(plain_path))
        fpack_plain_fits(plain_path, delete_plain=True)
        # fpack writes sibling .fits.fz next to plain; ensure final name matches dest.
        produced = Path(str(plain_path) + ".fz")
        if produced.resolve() != fz_dest.resolve():
            if fz_dest.is_file():
                fz_dest.unlink()
            os.replace(produced, fz_dest)
    except BaseException:
        if os.path.isfile(part):
            try:
                os.remove(part)
            except OSError:
                pass
        if plain_path.is_file():
            try:
                plain_path.unlink()
            except OSError:
                pass
        raise


# Back-compat alias used by older tests/callers.
def _stream_url_to_gzip_fits(url: str, gz_dest_path: str, timeout: float) -> None:
    """Deprecated: download and fpack to ``.fits.fz`` (gz path rewritten)."""
    fz_dest = spoc_ffi_fpack_basename(gz_dest_path)
    if not gz_dest_path.endswith(FFIC_FPACK_SUFFIX):
        parent = os.path.dirname(gz_dest_path) or "."
        fz_dest = os.path.join(parent, os.path.basename(fz_dest))
    else:
        fz_dest = gz_dest_path
    _stream_url_to_fpack_fits(url, fz_dest, timeout)


def _run_one_ffi_download(basename: str, download_fn: Callable[[], None]) -> tuple[str, str | None]:
    """Run a single FFI download; return ``(basename, error_message)``."""
    try:
        download_fn()
        return basename, None
    except (HTTPError, URLError, OSError) as exc:
        return basename, str(exc)


def _execute_ffi_downloads(
    tasks: list[tuple[str, Callable[[], None]]],
    max_workers: int,
) -> tuple[int, int]:
    """Run FFI download tasks sequentially or in parallel. Returns ``(n_ok, n_err)``."""
    if not tasks:
        return 0, 0

    n_ok, n_err = 0, 0
    lock = threading.Lock()
    last_progress_log = time.monotonic()
    completed = 0

    def _record_result(basename: str, error: str | None) -> None:
        """Record result.
        
        Parameters
        ----------
        basename : str
        error : str | None"""
        nonlocal n_ok, n_err, completed, last_progress_log
        with lock:
            completed += 1
            if error:
                n_err += 1
                log.warning("File %s: %s", basename, error)
            else:
                n_ok += 1
            now = time.monotonic()
            if completed % 10 == 0 or now - last_progress_log >= 30.0:
                log.info("FFI download progress: %d/%d", completed, len(tasks))
                last_progress_log = now

    if max_workers <= 1:
        for i in _progress_iterate(len(tasks), desc="FFI download"):
            basename, download_fn = tasks[i]
            _record_result(*_run_one_ffi_download(basename, download_fn))
        return n_ok, n_err

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_run_one_ffi_download, basename, download_fn)
            for basename, download_fn in tasks
        ]
        for fut in as_completed(futures):
            _record_result(*fut.result())
    return n_ok, n_err


def _download_ffis_via_tesscurl(
    sector: int,
    camera: int,
    ccd: int,
    output_dir: str,
    overwrite: bool,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    *,
    ffi_list_path: str | Path | None = None,
) -> list:
    """Download ffis via tesscurl.
    
    Parameters
    ----------
    sector : int
    camera : int
    ccd : int
    output_dir : str
    overwrite : bool
    max_workers : int, optional, default ``_DEFAULT_MAX_WORKERS``
    
    Returns
    -------
    list"""
    script_url = TESSCURL_SCRIPT_URL.format(sector=sector)
    log.info("Fetching tesscurl manifest %s ...", script_url)
    try:
        script_bytes = _fetch_bytes(script_url, _DOWNLOAD_TIMEOUT_SCRIPT_S)
    except HTTPError as e:
        log.error(
            "Could not download tesscurl script (%s): %s.%s",
            script_url,
            e,
            _HTTP_ERROR_HELP,
        )
        return []
    except URLError as e:
        log.error("Network error fetching tesscurl script: %s.%s", e, _HTTP_ERROR_HELP)
        return []

    script_text = script_bytes.decode("utf-8", errors="replace")
    script_path = tesscurl_script_path(output_dir, sector)
    try:
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(script_text)
    except OSError as e:
        log.warning("Could not save tesscurl script to %s: %s", script_path, e)

    pairs = parse_tesscurl_script(script_text)
    url_by_basename = {bn: url for bn, url in pairs}
    expected_basenames = sorted(
        bn
        for bn in url_by_basename
        if _ffic_product_basename_matches(bn, sector, camera, ccd)
    )
    filtered = [(bn, url_by_basename[bn]) for bn in expected_basenames]
    n_drop = len(pairs) - len(filtered)
    if n_drop and pairs:
        log.debug(
            "Filtered tesscurl manifest to camera=%s ccd=%s (%s of %s lines).",
            camera,
            ccd,
            len(filtered),
            len(pairs),
        )

    if not filtered:
        log.warning(
            "No FFIC URLs for sector=%s camera=%s ccd=%s in tesscurl manifest "
            "(%s curl lines parsed).%s",
            sector,
            camera,
            ccd,
            len(pairs),
            _HTTP_ERROR_HELP,
        )
        return list_local_ffis(output_dir, sector, camera, ccd)

    log.info("Found %s FFIC file(s) for this camera/CCD in tesscurl manifest.", len(filtered))

    if not overwrite:
        existing = local_ffi_manifest_basenames(
            list_local_ffis(output_dir, sector, camera, ccd)
        )
        before = len(filtered)
        filtered = [(bn, url) for bn, url in filtered if bn not in existing]
        n_skip = before - len(filtered)
        if n_skip > 0:
            log.info("Skipping %s already-downloaded file(s).", n_skip)

    if filtered:
        log.info(
            "Downloading %s FITS file(s) to %s (workers=%s) ...",
            len(filtered),
            output_dir,
            max_workers,
        )
        ingest_buffer = (
            _FfiListIngestBuffer(ffi_list_path) if ffi_list_path is not None else None
        )
        tasks: list[tuple[str, Callable[[], None]]] = []
        for bn, url in filtered:
            fz_path = os.path.join(output_dir, spoc_ffi_fpack_basename(bn))

            def _download(
                url: str = url,
                fz_path: str = fz_path,
                buffer: _FfiListIngestBuffer | None = ingest_buffer,
            ) -> None:
                on_plain = None
                if buffer is not None:
                    from syndiff_pipeline.common.wcs_grouping import open_fits_memmap
                    from syndiff_pipeline.common.wcs_header_cache import extract_ffi_header_record

                    def on_plain(plain_path: str, buffer: _FfiListIngestBuffer = buffer) -> None:
                        row = extract_ffi_header_record(
                            plain_path, open_fits=open_fits_memmap
                        )
                        buffer.add(row)

                _stream_url_to_fpack_fits(
                    url,
                    fz_path,
                    _DOWNLOAD_TIMEOUT_FITS_S,
                    on_plain_fits=on_plain,
                )

            tasks.append((bn, _download))
        n_ok, n_err = _execute_ffi_downloads(tasks, max_workers)
        if ingest_buffer is not None:
            ingest_buffer.flush()
        log.info("Download finished (%s ok, %s errors).", n_ok, n_err)
        if n_err:
            log.warning("Some downloads failed; re-run with --overwrite or check network.")
    else:
        log.info("Nothing new to download.")

    return list_local_ffis(output_dir, sector, camera, ccd)


def download_ffis(
    sector: int,
    camera: int,
    ccd: int,
    output_dir: str,
    overwrite: bool = False,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    *,
    data_root: str | None = None,
    update_ffi_list: bool = True,
) -> list:
    """
    Download all calibrated TESS FFIs for a given sector/camera/CCD from MAST.

    Parameters
    ----------
    sector, camera, ccd : int
        TESS sector, camera, and CCD numbers.
    output_dir : str
        Destination directory. Created if it does not exist.
    overwrite : bool
        If True, re-download files that already exist locally.
    max_workers : int
        Concurrent download workers (default 8). Use 1 for strictly sequential
        downloads.
    data_root : str or None
        When set with ``update_ffi_list=True``, populate ``ffi_list.parquet`` under
        the SCC root and run end-of-batch ensure for skipped locals.
    update_ffi_list : bool
        When ``data_root`` is set, extract HDU1 headers during download.

    Returns
    -------
    list of str
        Sorted list of local FITS file paths (downloaded + pre-existing).
    """
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)
    max_workers = max(1, int(max_workers))

    ffi_list_path = None
    if data_root and update_ffi_list:
        from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet

        ffi_list_path = scc_ffi_list_parquet(data_root, sector, camera, ccd)
    elif update_ffi_list and not data_root:
        log.debug("ffi_list ingest skipped: data_root not provided")

    paths = _download_ffis_via_tesscurl(
        sector,
        camera,
        ccd,
        output_dir,
        overwrite,
        max_workers,
        ffi_list_path=ffi_list_path,
    )

    if data_root and update_ffi_list:
        from syndiff_pipeline.common.wcs_grouping import open_fits_memmap
        from syndiff_pipeline.common.wcs_header_cache import ensure_scc_ffi_list

        ensure_scc_ffi_list(
            data_root,
            sector,
            camera,
            ccd,
            paths,
            open_fits=open_fits_memmap,
        )

    return paths


def rebuild_ffi_list_for_scc(
    data_root: str,
    sector: int,
    camera: int,
    ccd: int,
    ffi_dir: str,
) -> int:
    """Cold-rebuild ``ffi_list`` for one SCC. Returns row count."""
    from syndiff_pipeline.common.wcs_grouping import open_fits_memmap
    from syndiff_pipeline.common.wcs_header_cache import rebuild_scc_ffi_list

    paths = list_local_ffis(ffi_dir, sector, camera, ccd)
    df = rebuild_scc_ffi_list(
        data_root,
        sector,
        camera,
        ccd,
        paths,
        open_fits=open_fits_memmap,
    )
    return len(df)


def main():
    """CLI for FFI download and optional ``ffi_list`` rebuild."""
    parser = argparse.ArgumentParser(
        description="Download TESS FFI calibrated images (tesscurl manifest + urllib).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sector", type=int, required=True, help="TESS sector number")
    parser.add_argument("--camera", type=int, required=True, help="Camera (1-4)")
    parser.add_argument("--ccd", type=int, required=True, help="CCD (1-4)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory (default: data/tess_ffi/sNNNN/camM_ccdK under cwd)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="SCC data root for ffi_list ingest (e.g. pipeline data_root)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files")
    parser.add_argument(
        "--no-update-ffi-list",
        action="store_true",
        help="Skip ffi_list header extraction during download",
    )
    parser.add_argument(
        "--rebuild-ffi-list",
        action="store_true",
        help="Cold-rebuild ffi_list.parquet from local FFIs (no network download)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_MAX_WORKERS,
        help="Concurrent FFI download workers",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = args.output_dir or nested_ffi_dir(args.sector, args.camera, args.ccd)
    if args.rebuild_ffi_list:
        if not args.data_root:
            parser.error("--rebuild-ffi-list requires --data-root")
        n_rows = rebuild_ffi_list_for_scc(
            args.data_root,
            args.sector,
            args.camera,
            args.ccd,
            output_dir,
        )
        print(f"\nRebuilt ffi_list with {n_rows} row(s)")
        return

    paths = download_ffis(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        output_dir=output_dir,
        overwrite=args.overwrite,
        max_workers=args.workers,
        data_root=args.data_root,
        update_ffi_list=not args.no_update_ffi_list,
    )
    print(f"\nTotal local FFI files: {len(paths)}")
    for p in paths[:5]:
        print(f"  {p}")
    if len(paths) > 5:
        print(f"  ... and {len(paths) - 5} more")


if __name__ == "__main__":
    main()
