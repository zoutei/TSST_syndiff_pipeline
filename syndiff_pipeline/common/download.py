"""
download.py
===========
Download calibrated TESS Full Frame Images (FFICs) for a given sector/camera/CCD.

By default this uses the STScI tesscurl sector script only as a **URL manifest**:
``https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_<N>_ffic.sh``
lines are parsed for MAST download URLs; FITS are fetched with :mod:`urllib` (no subprocess curl).

Use ``use_mast_astroquery=True`` (or CLI ``--via-mast``) for the legacy astroquery catalog +
MAST download path if the tesscurl script is unavailable.

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
import time
from pathlib import Path
from typing import List, Tuple
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
_USER_AGENT = "syndiff_pipeline/TESS-FFI"

_HTTP_ERROR_HELP = (
    " If tesscurl is missing or the archive is down, retry later or use --via-mast."
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


def _ffi_filename_pattern(sector: int, camera: int, ccd: int) -> str:
    """Return glob pattern for TESS FFI calibrated files."""
    return f"tess*-s{sector:04d}-{camera}-{ccd}-*_ffic.fits"


def _ffic_product_basename_matches(
    product_filename: str, sector: int, camera: int, ccd: int
) -> bool:
    """
    True if ``productFilename`` is a calibrated FFI for exactly this sector/camera/CCD.

    SPOC names look like ``tess2020019142923-s0020-3-3-0165-s_ffic.fits``.
    """
    base = os.path.basename(str(product_filename))
    pat = re.compile(
        rf"^tess[0-9]+-s{sector:04d}-{camera}-{ccd}-.+_ffic\.fits$",
        re.IGNORECASE,
    )
    return pat.match(base) is not None


def list_local_ffis(ffi_dir: str, sector: int, camera: int, ccd: int) -> list:
    """
    Glob for already-downloaded FFI files matching sector/camera/CCD.

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
    pattern = os.path.join(ffi_dir, _ffi_filename_pattern(sector, camera, ccd))
    return sorted(glob.glob(pattern))


def _fetch_bytes(url: str, timeout: float) -> bytes:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _strip_shell_quote(s: str) -> str:
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


def _download_ffis_via_tesscurl(
    sector: int,
    camera: int,
    ccd: int,
    output_dir: str,
    overwrite: bool,
) -> list:
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
        existing = set(
            os.path.basename(p) for p in list_local_ffis(output_dir, sector, camera, ccd)
        )
        before = len(filtered)
        filtered = [(bn, url) for bn, url in filtered if bn not in existing]
        n_skip = before - len(filtered)
        if n_skip > 0:
            log.info("Skipping %s already-downloaded file(s).", n_skip)

    if filtered:
        log.info("Downloading %s FITS file(s) to %s ...", len(filtered), output_dir)
        n_ok, n_err = 0, 0
        last_progress_log = time.monotonic()
        for i in _progress_iterate(len(filtered), desc="FFI download"):
            bn, url = filtered[i]
            local_path = os.path.join(output_dir, bn)
            try:
                _stream_url_to_file(url, local_path, _DOWNLOAD_TIMEOUT_FITS_S)
                n_ok += 1
            except (HTTPError, URLError, OSError) as e:
                n_err += 1
                log.warning("File %s: %s", bn, e)
            now = time.monotonic()
            if (i + 1) % 10 == 0 or now - last_progress_log >= 30.0:
                log.info("FFI download progress: %d/%d", i + 1, len(filtered))
                last_progress_log = now
        log.info("Download finished (%s ok, %s errors).", n_ok, n_err)
        if n_err:
            log.warning("Some downloads failed; re-run with --overwrite or check network.")
    else:
        log.info("Nothing new to download.")

    return list_local_ffis(output_dir, sector, camera, ccd)


def _download_ffis_via_astroquery(
    sector: int,
    camera: int,
    ccd: int,
    output_dir: str,
    overwrite: bool,
) -> list:
    try:
        from astroquery.mast import Observations
    except ImportError as e:
        raise ImportError(
            "astroquery is required for --via-mast downloads. "
            "Install with: conda install -c conda-forge astroquery"
        ) from e

    expected_obs_id = f"tess-s{sector:04d}-{camera}-{ccd}"
    log.info(
        "Querying MAST for this camera/CCD only (obs_id=%s, sector=%s camera=%s ccd=%s) ...",
        expected_obs_id,
        sector,
        camera,
        ccd,
    )

    common = dict(
        obs_collection="TESS",
        dataproduct_type="image",
        sequence_number=sector,
        provenance_name="SPOC",
    )
    obs_table = Observations.query_criteria(obs_id=expected_obs_id, **common)

    if len(obs_table) == 0:
        log.info(
            "Narrow query returned no rows; trying full sector list and selecting %s ...",
            expected_obs_id,
        )
        obs_table = Observations.query_criteria(**common)
        if len(obs_table) == 0:
            log.warning("No TESS observations found for sector %s.", sector)
            return []
        oid_col = np.asarray(obs_table["obs_id"], dtype=str)
        sel = oid_col == expected_obs_id
        if not sel.any():
            needle = f"s{sector:04d}-{camera}-{ccd}"
            sel = np.array([needle in s for s in oid_col], dtype=bool)
        obs_table = obs_table[sel]
        if len(obs_table) == 0:
            log.warning(
                "No MAST observation with obs_id matching %s (sector %s camera %s ccd %s).",
                expected_obs_id,
                sector,
                camera,
                ccd,
            )
            return []

    obsids = np.unique(np.asarray(obs_table["obsid"], dtype=str))
    obsids = obsids[obsids != ""]
    if obsids.size == 0:
        log.warning("No valid obsid in MAST query results.")
        return []
    if obsids.size != 1:
        log.error(
            "Expected a single MAST obsid for %s; got %s. Refusing to download.",
            expected_obs_id,
            obsids.tolist(),
        )
        return []

    log.info(
        "Fetching product list for %s only (metadata, not FITS yet) ...",
        expected_obs_id,
    )
    products = Observations.get_product_list(obsids)

    ffic_mask = products["productSubGroupDescription"] == "FFIC"
    ffic_products = products[ffic_mask]

    n_ffic = len(ffic_products)
    cam_mask = [
        _ffic_product_basename_matches(fn, sector, camera, ccd)
        for fn in ffic_products["productFilename"]
    ]
    ffic_products = ffic_products[cam_mask]
    n_drop = n_ffic - len(ffic_products)
    if n_drop:
        log.warning(
            "Dropped %s FFIC product row(s) whose filenames do not match camera=%s ccd=%s.",
            n_drop,
            camera,
            ccd,
        )

    if len(ffic_products) == 0:
        log.warning(
            "No FFIC products found for sector=%s, camera=%s, ccd=%s.",
            sector,
            camera,
            ccd,
        )
        return []

    log.info("Found %s FFIC files to download.", len(ffic_products))

    if not overwrite:
        existing = set(
            os.path.basename(p) for p in list_local_ffis(output_dir, sector, camera, ccd)
        )
        to_download_mask = [
            str(fn) not in existing for fn in ffic_products["productFilename"]
        ]
        n_skip = sum(1 for x in to_download_mask if not x)
        if n_skip > 0:
            log.info("Skipping %s already-downloaded files.", n_skip)
        ffic_products = ffic_products[to_download_mask]

    if len(ffic_products) > 0:
        log.info("Downloading %s FITS files to %s ...", len(ffic_products), output_dir)
        n_ok, n_err = 0, 0
        last_progress_log = time.monotonic()
        for i in _progress_iterate(len(ffic_products), desc="FFI download"):
            row = ffic_products[i]
            local_path = os.path.join(
                output_dir, os.path.basename(row["productFilename"])
            )
            status, msg, _url = Observations.download_file(
                row["dataURI"],
                local_path=local_path,
                cache=not overwrite,
                verbose=False,
            )
            if status == "COMPLETE":
                n_ok += 1
            else:
                n_err += 1
                log.warning("File %s: %s %s", row["productFilename"], status, msg or "")
            now = time.monotonic()
            if (i + 1) % 10 == 0 or now - last_progress_log >= 30.0:
                log.info("FFI download progress: %d/%d", i + 1, len(ffic_products))
                last_progress_log = now
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
    use_mast_astroquery: bool = False,
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
    use_mast_astroquery : bool
        If True, use astroquery CAOM queries and ``Observations.download_file``
        instead of the default tesscurl manifest + ``urllib`` downloads.

    Returns
    -------
    list of str
        Sorted list of local FITS file paths (downloaded + pre-existing).
    """
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    if use_mast_astroquery:
        return _download_ffis_via_astroquery(
            sector, camera, ccd, output_dir, overwrite
        )
    return _download_ffis_via_tesscurl(sector, camera, ccd, output_dir, overwrite)


def main():
    parser = argparse.ArgumentParser(
        description="Download TESS FFI calibrated images (tesscurl manifest + urllib by default).",
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
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing files")
    parser.add_argument(
        "--via-mast",
        action="store_true",
        help="Use astroquery MAST queries + Observations.download_file instead of tesscurl.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = args.output_dir or nested_ffi_dir(args.sector, args.camera, args.ccd)
    paths = download_ffis(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        output_dir=output_dir,
        overwrite=args.overwrite,
        use_mast_astroquery=args.via_mast,
    )
    print(f"\nTotal local FFI files: {len(paths)}")
    for p in paths[:5]:
        print(f"  {p}")
    if len(paths) > 5:
        print(f"  ... and {len(paths) - 5} more")


if __name__ == "__main__":
    main()
