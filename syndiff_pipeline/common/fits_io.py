"""
fits_io.py
==========
Write SynDiff FITS products as CFITSIO fpack (``.fits.fz``).

Flow: normalize target → atomic plain ``.fits`` write → ``fpack -g -q 0 -Y -D`` →
``.fits.fz`` (GZIP tile compression; lossless for floats).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits

from syndiff_pipeline.common.fits_variants import (
    FITS_FPACK_EXT,
    FITS_PLAIN_EXT,
    fits_fpack_path,
    fits_logical_path,
    is_compressed_fits_path,
)

log = logging.getLogger(__name__)


def _require_fpack() -> str:
    path = shutil.which("fpack")
    if not path:
        raise RuntimeError(
            "fpack not found on PATH. Activate the syndiff conda env "
            "(CFITSIO) before writing pipeline FITS."
        )
    return path


def fpack_plain_fits(plain_path: str | Path, *, delete_plain: bool = True) -> Path:
    """
    Run ``fpack -g -q 0 -Y`` on a plain ``.fits`` file; return the ``.fits.fz`` path.

    Uses GZIP tile compression with ``-q 0`` (lossless for floats). ``-Y``
    suppresses the interactive lossy-prompt; an existing sibling ``.fits.fz``
    is removed first.
    """
    plain = Path(plain_path)
    if not plain.is_file():
        raise FileNotFoundError(f"Plain FITS not found for fpack: {plain}")
    if plain.name.lower().endswith(FITS_FPACK_EXT) or plain.name.lower().endswith(
        ".fits.gz"
    ):
        raise ValueError(f"Expected plain .fits for fpack, got {plain.name}")

    fpack_bin = _require_fpack()
    fz_path = Path(fits_fpack_path(plain))
    if fz_path.is_file():
        try:
            fz_path.unlink()
        except OSError as exc:
            raise RuntimeError(f"Could not remove existing {fz_path}: {exc}") from exc

    cmd = [fpack_bin, "-g", "-q", "0", "-Y"]
    if delete_plain:
        cmd.append("-D")
    cmd.append(str(plain))
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not fz_path.is_file():
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"fpack failed for {plain} (rc={proc.returncode}): {err or 'no output'}"
        )
    if delete_plain and plain.is_file():
        try:
            plain.unlink()
        except OSError as exc:
            log.warning("Could not remove plain FITS after fpack %s: %s", plain, exc)
    return fz_path


def _atomic_writeto_plain(plain_path: Path, write_fn) -> None:
    """Call *write_fn(tmp_path)* then replace into *plain_path*."""
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=plain_path.stem + ".",
        suffix=".fits.part",
        dir=str(plain_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, plain_path)
    except BaseException:
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def write_hdul_fits(out_path: str | Path, hdul: fits.HDUList) -> str:
    """
    Write an HDUList to ``.fits.fz`` via plain temp + fpack.

    *out_path* may be ``.fits``, ``.fits.gz``, or ``.fits.fz``; the on-disk
    product is always ``.fits.fz``.
    """
    target = Path(fits_fpack_path(out_path))
    plain = Path(fits_logical_path(target))

    def _write(tmp: Path) -> None:
        # Copy HDUs so we do not mutate the caller's list via close.
        hdul.writeto(tmp, overwrite=True)

    _atomic_writeto_plain(plain, _write)
    fz = fpack_plain_fits(plain, delete_plain=True)
    return str(fz)


def write_image_fits(
    out_path: str | Path,
    data: np.ndarray,
    *,
    header: Optional[fits.Header] = None,
) -> str:
    """Write a single 2D image FITS (float32) as ``.fits.fz``."""
    target = Path(fits_fpack_path(out_path))
    plain = Path(fits_logical_path(target))
    hdr = fits.Header(header) if header is not None else None
    arr = np.asarray(data, dtype=np.float32)

    def _write(tmp: Path) -> None:
        fits.writeto(tmp, arr, header=hdr, overwrite=True)

    _atomic_writeto_plain(plain, _write)
    fz = fpack_plain_fits(plain, delete_plain=True)
    return str(fz)


def write_primary_hdu_fits(
    out_path: str | Path,
    hdu: fits.PrimaryHDU | fits.ImageHDU,
) -> str:
    """Write a single Primary/Image HDU as ``.fits.fz``."""
    return write_hdul_fits(out_path, fits.HDUList([hdu]))


def _promote_fpack_primary_image(hdul: fits.HDUList) -> None:
    """
    Restore primary-image access after fpack.

    ``fpack`` moves a data-bearing PRIMARY into a ``COMPRESSED_IMAGE`` extension
    and leaves PRIMARY empty. Many SynDiff readers use ``hdul[0].data``; promote
    the decompressed array back onto PRIMARY and drop the redundant extension.
    Custom keywords that fpack relocated onto ``COMPRESSED_IMAGE`` are copied
    back onto PRIMARY. Named extensions (NOISE, MASK, SCI, …) are left in place.
    """
    if len(hdul) < 2:
        return
    if hdul[0].data is not None:
        return
    h1 = hdul[1]
    if getattr(h1, "name", None) != "COMPRESSED_IMAGE":
        return
    if h1.data is None:
        return
    hdul[0].data = np.asarray(h1.data)
    # fpack relocates non-structural primary keywords onto COMPRESSED_IMAGE.
    _STRUCTURAL = {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "NAXIS3",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTNAME",
        "INHERIT",
    }
    for card in h1.header.cards:
        key = card.keyword
        if not key or key in _STRUCTURAL:
            continue
        if key.startswith(("Z", "TFORM", "TTYPE", "TUNIT", "TDIM")):
            continue
        if key in hdul[0].header:
            continue
        hdul[0].header.append(card)
    del hdul[1]


def open_fits(path: str | Path, **kwargs):
    """
    Open a FITS file after resolving storage variants.

    Compressed variants (``.fits.fz``, ``.fits.gz``) default to ``memmap=False``.
    Plain ``.fits`` defaults to ``memmap=True`` unless overridden.

    For ``.fits.fz`` files produced by fpack from a primary image, the empty
    PRIMARY + ``COMPRESSED_IMAGE`` layout is normalized so ``hdul[0].data`` works.
    """
    from syndiff_pipeline.common.fits_variants import resolve_fits_variant

    resolved = resolve_fits_variant(path)
    if "memmap" not in kwargs:
        kwargs["memmap"] = not is_compressed_fits_path(resolved)
    hdul = fits.open(resolved, **kwargs)
    try:
        _promote_fpack_primary_image(hdul)
    except Exception as exc:
        log.debug("fpack primary promote skipped for %s: %s", resolved, exc)
    return hdul
