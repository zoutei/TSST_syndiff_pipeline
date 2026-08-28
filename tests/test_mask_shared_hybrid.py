"""Hybrid empirical shared mask stamps."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.masking import bits
from syndiff_pipeline.difference_imaging.masking.settings import MaskSettings, SharedMaskSettings
from syndiff_pipeline.difference_imaging.masking.shared import build_static_mask, make_shared_mask


def _settings(**kwargs):
    shared = SharedMaskSettings(
        style="empirical",
        include_straps=False,
        include_edges=False,
        ps1_min_hit_count=0,
        **kwargs,
    )
    return MaskSettings(shared=shared)


def test_hybrid_stamp_table():
    image = np.zeros((80, 80), dtype=np.float64)
    crop = {"x_min": 100, "x_max": 180, "y_min": 100, "y_max": 180, "shape": (80, 80)}
    gaia = pd.DataFrame(
        {
            "x": [20.0, 40.0, 60.0, 70.0],
            "y": [20.0, 40.0, 60.0, 70.0],
            "mag": [11.0, 6.0, 15.0, 19.0],
        }
    )
    mask = build_static_mask(
        image,
        gaia,
        crop,
        settings=_settings(),
        straps_csv="/nonexistent/straps.csv",
    )
    # mag 11 → bit 2 (mid bright)
    assert mask[20, 20] & bits.SAT_CROSS
    assert not (mask[20, 20] & bits.BRIGHT_CAT)
    # mag 6 → bit 1 only (very bright)
    assert mask[40, 40] & bits.BRIGHT_CAT
    assert not (mask[40, 40] & bits.SAT_CROSS)
    # mag 15 → bit 32
    assert mask[60, 60] & bits.FAINT_CAT
    assert not (mask[60, 60] & bits.BRIGHT_CAT)
    # mag 19 → none
    assert mask[70, 70] == 0


def test_bsc_cross_only():
    image = np.zeros((80, 80), dtype=np.float64)
    crop = {"x_min": 0, "x_max": 80, "y_min": 0, "y_max": 80, "shape": (80, 80)}
    gaia = pd.DataFrame({"x": [10.0], "y": [10.0], "mag": [12.0]})
    # Without real FFI, inject BSC via monkeypatch of _project_bsc
    import syndiff_pipeline.difference_imaging.masking.shared as shared_mod

    bsc = pd.DataFrame({"x": [50.0], "y": [50.0], "vmag": [5.0]})
    orig = shared_mod._project_bsc
    shared_mod._project_bsc = lambda *a, **k: bsc
    try:
        mask = build_static_mask(
            image,
            gaia,
            crop,
            settings=_settings(),
            straps_csv="/nonexistent/straps.csv",
            ref_ffi_path="/tmp/fake.fits",
        )
    finally:
        shared_mod._project_bsc = orig
    assert mask[50, 50] & bits.BRIGHT_CAT
    assert not (mask[50, 50] & bits.SAT_CROSS)


def test_tessreduce_no_bit32():
    image = np.zeros((60, 60), dtype=np.float64)
    crop = {"x_min": 0, "x_max": 60, "y_min": 0, "y_max": 60, "shape": (60, 60)}
    gaia = pd.DataFrame({"x": [30.0], "y": [30.0], "mag": [15.0]})
    settings = MaskSettings(
        shared=SharedMaskSettings(
            style="tessreduce",
            include_straps=False,
            include_edges=False,
            ps1_min_hit_count=0,
            bright_maglim=18.0,
        )
    )
    mask = build_static_mask(
        image,
        gaia,
        crop,
        settings=settings,
        straps_csv="/nonexistent/straps.csv",
    )
    assert not (mask & bits.FAINT_CAT).any()
    # tessreduce mid-bright (mag 15) → bit 2 squares
    assert (mask & bits.SAT_CROSS).any()


def test_make_shared_mask_writes_canonical_basename(tmp_path):
    image = np.zeros((40, 40), dtype=np.float64)
    crop = {"x_min": 0, "x_max": 40, "y_min": 0, "y_max": 40, "shape": (40, 40)}
    gaia = pd.DataFrame({"x": [20.0], "y": [20.0], "mag": [12.0]})
    mask = make_shared_mask(
        image,
        gaia,
        crop,
        straps_csv="/nonexistent/straps.csv",
        strapsize=0,
        output_dir=str(tmp_path),
        ps1_min_hit_count=0,
    )
    from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME

    assert (tmp_path / SHARED_MASK_FITS_BASENAME).is_file()
    assert SHARED_MASK_FITS_BASENAME.endswith(".fits.fz")
    assert mask.dtype == np.int16


def test_build_static_mask_forwards_resolved_mask_settings_to_emit(tmp_path):
    """Contract A2 (emitters side): build_static_mask's *resolved* ``settings``
    (not just ``mask_params``, the path-bearing stage params) must reach
    ``provenance_glue.emit_shared_mask_artifact`` as ``mask_settings=``, so the
    recorded recipe hashes the policy that actually painted the mask.
    """
    image = np.zeros((40, 40), dtype=np.float64)
    crop = {"x_min": 0, "x_max": 40, "y_min": 0, "y_max": 40, "shape": (40, 40)}
    gaia = pd.DataFrame({"x": [20.0], "y": [20.0], "mag": [12.0]})
    settings = _settings(bright_maglim=14.0)

    with patch(
        "syndiff_pipeline.difference_imaging.orchestration.provenance_glue"
        ".emit_shared_mask_artifact"
    ) as mock_emit:
        build_static_mask(
            image,
            gaia,
            crop,
            settings=settings,
            straps_csv="/nonexistent/straps.csv",
            output_dir=str(tmp_path),
            sck=(20, 3, 3),
            data_root=str(tmp_path),
            mask_params=object(),
        )

    mock_emit.assert_called_once()
    assert mock_emit.call_args.kwargs["mask_settings"] is settings
