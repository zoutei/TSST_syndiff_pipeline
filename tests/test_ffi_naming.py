"""Tests for pipeline FITS naming helpers."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    iter_pipeline_fits_paths,
    parse_workspace_frame_stem,
    resolve_pipeline_fits_path,
    strip_fits_suffix,
    workspace_frame_fits_basename,
    workspace_frame_fits_path,
    workspace_frame_stem,
)


def test_strip_fits_suffix():
    assert strip_fits_suffix("tess123_hp_d.fits.gz") == "tess123_hp_d"
    assert strip_fits_suffix("tess123_hp_d.fits") == "tess123_hp_d"
    assert strip_fits_suffix("/ws/ks_d/tess123_hp_d.fits.gz") == "tess123_hp_d"


def test_parse_workspace_frame_stem_with_gzip():
    parsed = parse_workspace_frame_stem("tess2020057105921_ks_d.fits.gz")
    assert parsed == ("tess2020057105921", "ks_d")


def test_workspace_frame_fits_basename():
    stem = workspace_frame_stem("tess123", "hp_d")
    assert workspace_frame_fits_basename(stem) == f"{stem}.fits.gz"


def test_resolve_pipeline_fits_path_prefers_gzip():
    stem = workspace_frame_stem("tess999", "hp_d")
    with tempfile.TemporaryDirectory() as td:
        legacy = workspace_frame_fits_path(td, stem).replace(".fits.gz", ".fits")
        gzip = workspace_frame_fits_path(td, stem)
        with open(legacy, "wb") as fh:
            fh.write(b"SIMPLE  = T")
        with open(gzip, "wb") as fh:
            fh.write(b"SIMPLE  = T")
        assert resolve_pipeline_fits_path(td, stem) == gzip


def test_resolve_pipeline_fits_path_falls_back_to_legacy():
    stem = workspace_frame_stem("tess999", "hp_b")
    with tempfile.TemporaryDirectory() as td:
        legacy = os.path.join(td, f"{stem}.fits")
        fits.writeto(legacy, np.array([[1.0]], dtype=np.float32), overwrite=True)
        assert resolve_pipeline_fits_path(td, stem) == legacy


def test_iter_pipeline_fits_paths_dedupes_by_stem():
    stem = workspace_frame_stem("tess999", "ks_d")
    with tempfile.TemporaryDirectory() as td:
        legacy = os.path.join(td, f"{stem}.fits")
        gzip = workspace_frame_fits_path(td, stem)
        fits.writeto(legacy, np.array([[1.0]], dtype=np.float32), overwrite=True)
        fits.writeto(gzip, np.array([[2.0]], dtype=np.float32), overwrite=True)
        paths = iter_pipeline_fits_paths(td)
        assert len(paths) == 1
        assert paths[0] == gzip
