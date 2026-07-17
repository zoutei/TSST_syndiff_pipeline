"""Tests for shift_schedule.py (synthetic WCS fixtures only, no /astro data)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
    build_ps1_wcs,
)
from syndiff_pipeline.template_creation.processing.shift_schedule import (
    ShiftSchedule,
    _sg_smooth_series,
    _split_orbit_segments,
    assign_groups_from_schedule,
    build_rle_dataframe,
    build_skycell_shift_schedule,
    cache_key_for,
    encode_quantized_shift,
    hysteresis_round_series,
    quantize_shift,
    write_group_artifacts,
)

PS1_SCALE_DEG = 0.25 / 3600.0


def _make_wcs(
    crval=(150.0, 20.0),
    crpix=(1024.5, 1024.5),
    scale_deg: float = 20.25 / 3600.0,
) -> WCS:
    """A simple linear (no-SIP) TAN WCS at ~TESS pixel scale (mirrors test_wcs_drift_field.py)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cd = np.array([[-scale_deg, 0.0], [0.0, scale_deg]])
    w.wcs.set()
    return w


def _shifted_wcs(ref_wcs: WCS, shift_xy: tuple[float, float]) -> WCS:
    """A WCS identical to ``ref_wcs`` except CRPIX translated by ``shift_xy`` (uniform drift)."""
    w = WCS(naxis=2)
    w.wcs.ctype = list(ref_wcs.wcs.ctype)
    w.wcs.crval = list(ref_wcs.wcs.crval)
    w.wcs.crpix = [ref_wcs.wcs.crpix[0] + shift_xy[0], ref_wcs.wcs.crpix[1] + shift_xy[1]]
    w.wcs.cd = np.array(ref_wcs.wcs.cd)
    w.wcs.set()
    return w


def _make_ps1_row(name: str, ra: float, dec: float, naxis: int = 2000) -> pd.Series:
    """A synthetic PS1 skycell row with the WCS header columns build_ps1_wcs reads."""
    return pd.Series(
        {
            "NAME": name,
            "RA": ra,
            "DEC": dec,
            "NAXIS1": naxis,
            "NAXIS2": naxis,
            "CRVAL1": ra,
            "CRVAL2": dec,
            "CRPIX1": naxis / 2.0,
            "CRPIX2": naxis / 2.0,
            "PC1_1": 1.0,
            "PC1_2": 0.0,
            "PC2_1": 0.0,
            "PC2_2": 1.0,
            "CDELT1": -PS1_SCALE_DEG,
            "CDELT2": PS1_SCALE_DEG,
            "RADESYS": "ICRS",
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
        }
    )


def _make_skycell_df() -> pd.DataFrame:
    centers = [
        ("skycell.0001.001", 149.99, 19.99),
        ("skycell.0001.002", 150.01, 19.99),
        ("skycell.0001.003", 149.99, 20.01),
        ("skycell.0001.004", 150.01, 20.01),
    ]
    return pd.DataFrame([_make_ps1_row(*c) for c in centers])


# ── (a) shift schedule values vs. independent WCS round-trip ───────────────

def test_shift_schedule_matches_independent_wcs_roundtrip():
    """
    build_skycell_shift_schedule now measures real per-frame WCS drift
    directly at each skycell center (no anchor-grid interpolation) -- so the
    "independent" check here is simply doing that same WCS round-trip by
    hand and comparing. The injected shifts are exactly linear in time, so
    SG-smoothing (polyorder=1) reproduces them with no residual.
    """
    ref_wcs = _make_wcs()
    shifts = [(0.05 * t, -0.03 * t) for t in range(4)]
    frames = [(f"f{t}.fits", _shifted_wcs(ref_wcs, s)) for t, s in enumerate(shifts)]
    btjd = np.arange(4, dtype=float)

    skycell_df = _make_skycell_df()
    schedule = build_skycell_shift_schedule(
        frames, skycell_df, ref_wcs, btjd=btjd, savgol_window=3, savgol_polyorder=1,
    )

    ra = skycell_df["RA"].to_numpy(dtype=np.float64)
    dec = skycell_df["DEC"].to_numpy(dtype=np.float64)
    x_ref, y_ref = ref_wcs.world_to_pixel_values(ra, dec)

    for f in (0, 2):
        for c in (0, 1):
            wcs_f = frames[f][1]
            fx, fy = wcs_f.world_to_pixel_values(ra[c], dec[c])
            dx, dy = float(fx) - float(x_ref[c]), float(fy) - float(y_ref[c])
            ps1_wcs, _ = build_ps1_wcs(skycell_df.iloc[c])

            x_tess, y_tess = ref_wcs.world_to_pixel_values(ra[c], dec[c])
            ra1, dec1 = ref_wcs.pixel_to_world_values(x_tess, y_tess)
            ra2, dec2 = ref_wcs.pixel_to_world_values(x_tess + dx, y_tess + dy)
            u1, v1 = ps1_wcs.world_to_pixel_values(ra1, dec1)
            u2, v2 = ps1_wcs.world_to_pixel_values(ra2, dec2)
            expected_sx, expected_sy = float(u2 - u1), float(v2 - v1)

            assert schedule.sx_float[f, c] == pytest.approx(expected_sx, abs=1e-3)
            assert schedule.sy_float[f, c] == pytest.approx(expected_sy, abs=1e-3)

    assert schedule.frame_valid.all()
    assert list(schedule.skycell_names) == list(skycell_df["NAME"])


def test_shift_schedule_invalid_frame_stays_nan_in_float_and_holds_in_int():
    ref_wcs = _make_wcs()
    shifts = [(0.1 * t, 0.0) for t in range(5)]
    wcses: list[WCS | None] = [_shifted_wcs(ref_wcs, s) for s in shifts]
    wcses[2] = None
    frames = [(f"f{t}.fits", w) for t, w in enumerate(wcses)]

    skycell_df = _make_skycell_df()
    schedule = build_skycell_shift_schedule(
        frames, skycell_df, ref_wcs, btjd=np.arange(5, dtype=float), savgol_window=3,
    )

    assert not schedule.frame_valid[2]
    assert np.all(np.isnan(schedule.sx_float[2]))
    assert np.all(np.isnan(schedule.sy_float[2]))
    # int schedule still carries a sensible (interpolated) value through the gap
    assert np.all(np.isfinite(schedule.sx_int[2]))


def test_shift_schedule_btjd_length_mismatch_raises():
    ref_wcs = _make_wcs()
    frames = [("f0.fits", ref_wcs), ("f1.fits", ref_wcs)]
    skycell_df = _make_skycell_df()
    with pytest.raises(ValueError):
        build_skycell_shift_schedule(frames, skycell_df, ref_wcs, btjd=np.array([0.0]))


# ── orbit-segment splitting + SG smoothing (relocated from the deleted
#    common/wcs_drift_field.py "Layer 1"; see module docstring) ───────────

def test_split_orbit_segments_splits_on_gap():
    n1, n2 = 6, 6
    btjd = np.concatenate([np.linspace(0.0, 1.0, n1), np.linspace(0.0, 1.0, n2) + 3.0])
    bounds = _split_orbit_segments(btjd, n1 + n2, 0.5)
    assert bounds.tolist() == [[0, n1], [n1, n1 + n2]]


def test_split_orbit_segments_no_btjd_gives_single_segment():
    bounds = _split_orbit_segments(None, 5, 0.5)
    assert bounds.tolist() == [[0, 5]]


def test_sg_smooth_series_orbit_segmentation_prevents_smearing_across_gap():
    n1, n2 = 6, 6
    raw = np.concatenate([0.01 * np.arange(n1), 5.0 + 0.01 * np.arange(n2)])
    valid = np.ones(n1 + n2, dtype=bool)
    btjd = np.concatenate([np.linspace(0.0, 1.0, n1), np.linspace(0.0, 1.0, n2) + 3.0])
    segment_bounds = _split_orbit_segments(btjd, n1 + n2, 0.5)
    assert segment_bounds.tolist() == [[0, n1], [n1, n1 + n2]]

    smoothed = raw.copy()
    for seg_start, seg_end in segment_bounds:
        seg = slice(int(seg_start), int(seg_end))
        smoothed[seg] = _sg_smooth_series(raw[seg], valid[seg], 5, 2)

    # first frame of segment 2 stays close to its own raw value (~5.0), not
    # smeared toward segment 1's near-zero values by a single SG window.
    assert abs(smoothed[n1] - 5.0) < 0.5
    assert abs(smoothed[n1 - 1] - raw[n1 - 1]) < 0.5


def test_sg_smooth_series_skips_invalid_and_leaves_nan():
    raw = np.array([0.0, 0.1, np.nan, 0.3, 0.4])
    valid = np.array([True, True, False, True, True])
    out = _sg_smooth_series(raw, valid, 5, 2)
    assert np.isnan(out[2])
    assert np.all(np.isfinite(out[[0, 1, 3, 4]]))


def test_sg_smooth_series_too_few_valid_returns_unchanged():
    raw = np.array([1.0, 2.0])
    valid = np.array([True, True])
    out = _sg_smooth_series(raw, valid, 5, 2)
    np.testing.assert_array_equal(out, raw)


# ── (b) hysteresis: oscillation within margin does not flap ────────────────

def test_hysteresis_round_series_does_not_flap_within_margin():
    # Oscillates +-0.55 around the integer 2; margin=0.1 -> switch threshold 0.6.
    frac = np.array([2.0, 2.55, 1.45, 2.55, 1.45, 2.55, 1.45], dtype=np.float64)
    out = hysteresis_round_series(frac, margin=0.1)
    assert np.all(out == 2)


def test_hysteresis_round_series_switches_beyond_margin():
    # A jump that exceeds the 0.6 threshold must switch bins.
    frac = np.array([2.0, 2.0, 3.7, 3.7, 3.7], dtype=np.float64)
    out = hysteresis_round_series(frac, margin=0.1)
    assert out.tolist() == [2, 2, 4, 4, 4]


# ── (c) RLE round-trip ──────────────────────────────────────────────────────

def test_rle_round_trip_reconstructs_int_arrays():
    skycell_names = np.array(["cellA", "cellB"])
    sx_int = np.array([[0, -1], [0, -1], [0, -1], [1, -1], [1, -1], [2, -1]], dtype=np.int16)
    sy_int = np.array([[0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 4]], dtype=np.int16)

    schedule = ShiftSchedule(
        skycell_names=skycell_names,
        sx_float=sx_int.astype(np.float32),
        sy_float=sy_int.astype(np.float32),
        sx_int=sx_int,
        sy_int=sy_int,
        frame_valid=np.ones(6, dtype=bool),
        meta={"schema_version": 1},
    )
    rle_df = schedule.to_rle_dataframe()

    n_frames = sx_int.shape[0]
    recon_sx = np.zeros_like(sx_int)
    recon_sy = np.zeros_like(sy_int)
    for c, name in enumerate(skycell_names):
        for _, row in rle_df[rle_df["skycell"] == name].iterrows():
            s, e = int(row["seg_start_frame_idx"]), int(row["seg_end_frame_idx"])
            recon_sx[s : e + 1, c] = row["sx_int"]
            recon_sy[s : e + 1, c] = row["sy_int"]

    np.testing.assert_array_equal(recon_sx, sx_int)
    np.testing.assert_array_equal(recon_sy, sy_int)
    # sanity: RLE is strictly more compact than the dense representation here
    assert len(rle_df) < n_frames * len(skycell_names)
    assert set(rle_df.columns) == {
        "skycell", "seg_start_frame_idx", "seg_end_frame_idx", "sx_int", "sy_int", "n_frames",
    }


def _make_grouping_schedule() -> ShiftSchedule:
    """
    6 frames x 3 skycells. Frame signatures: 0 and 3 share signature A,
    1 and 4 share signature B, frame 2 is invalid, frame 5 is a new
    signature C.
    """
    skycell_names = np.array(["c0", "c1", "c2"])
    sig_a = np.array([0, 1, 2], dtype=np.int16)
    sig_b = np.array([1, 1, 2], dtype=np.int16)
    sig_c = np.array([5, -3, 0], dtype=np.int16)
    sy_a = np.array([0, 1, 2], dtype=np.int16)
    sy_b = np.array([0, 2, 2], dtype=np.int16)
    sy_c = np.array([-1, 4, 4], dtype=np.int16)

    sx_int = np.stack([sig_a, sig_b, sig_a, sig_a, sig_b, sig_c])
    sy_int = np.stack([sy_a, sy_b, sy_a, sy_a, sy_b, sy_c])
    # float = int + a small fixed phase, so qx/qy math is well-defined everywhere
    sx_float = (sx_int + 0.2).astype(np.float32)
    sy_float = (sy_int - 0.1).astype(np.float32)

    frame_valid = np.array([True, True, False, True, True, True])

    return ShiftSchedule(
        skycell_names=skycell_names,
        sx_float=sx_float,
        sy_float=sy_float,
        sx_int=sx_int,
        sy_int=sy_int,
        frame_valid=frame_valid,
        meta={"schema_version": 1, "n_frames": 6, "n_skycells": 3},
    )


# ── (d) grouping ─────────────────────────────────────────────────────────

def test_assign_groups_dense_first_appearance_and_invalid_frames():
    schedule = _make_grouping_schedule()
    assignment = assign_groups_from_schedule(
        schedule, grouping_quantum_ps1_px=1.0, cache_quantum_ps1_px=0.25, keying="phase",
    )

    assert assignment.group_id_per_frame.tolist() == [0, 1, -1, 0, 1, 2]
    assert assignment.group_id_per_frame.dtype == np.int32
    group_ids = [g["group_id"] for g in assignment.groups]
    assert group_ids == [0, 1, 2]
    n_frames_by_group = {g["group_id"]: g["n_frames"] for g in assignment.groups}
    assert n_frames_by_group == {0: 2, 1: 2, 2: 1}
    # every group has a distinct signature hash
    assert len({g["signature_hash"] for g in assignment.groups}) == 3
    # shifts_df has one row per (group, skycell)
    assert len(assignment.shifts_df) == 3 * 3
    assert set(assignment.shifts_df["group_id"].unique()) == {0, 1, 2}


def test_assign_groups_coarser_quantum_can_merge_groups():
    schedule = _make_grouping_schedule()
    # At grouping_quantum=1.0 frames 0/3 (sig A) and 1/4 (sig B) are distinct
    # groups because c0 differs (0 vs 1). A very coarse grouping quantum
    # collapses that 1-unit difference, merging A and B into one group.
    assignment = assign_groups_from_schedule(
        schedule, grouping_quantum_ps1_px=10.0, cache_quantum_ps1_px=0.25, keying="absolute",
    )
    valid_ids = assignment.group_id_per_frame[assignment.group_id_per_frame >= 0]
    assert len(set(valid_ids.tolist())) < 3


# ── (e) phase vs absolute keying relationship ───────────────────────────────

def test_phase_vs_absolute_keying_relationship():
    schedule = _make_grouping_schedule()
    cache_quantum = 0.1

    abs_assign = assign_groups_from_schedule(
        schedule, grouping_quantum_ps1_px=1.0, cache_quantum_ps1_px=cache_quantum, keying="absolute",
    )
    phase_assign = assign_groups_from_schedule(
        schedule, grouping_quantum_ps1_px=1.0, cache_quantum_ps1_px=cache_quantum, keying="phase",
    )

    abs_row = abs_assign.shifts_df[
        (abs_assign.shifts_df["group_id"] == 0) & (abs_assign.shifts_df["skycell"] == "c0")
    ].iloc[0]
    phase_row = phase_assign.shifts_df[
        (phase_assign.shifts_df["group_id"] == 0) & (phase_assign.shifts_df["skycell"] == "c0")
    ].iloc[0]

    # absolute qx quantizes the full float value; phase qx quantizes only the
    # fractional part, so int + phase-qx must reconstruct absolute-qx to
    # within one cache quantum.
    reconstructed = phase_row["sx_int"] + phase_row["qx"]
    assert abs(float(abs_row["qx"]) - float(reconstructed)) <= cache_quantum + 1e-6

    reconstructed_y = phase_row["sy_int"] + phase_row["qy"]
    assert abs(float(abs_row["qy"]) - float(reconstructed_y)) <= cache_quantum + 1e-6

    assert abs_row["cache_key"] != phase_row["cache_key"] or abs_row["sx_int"] == 0


# ── quantize_shift / encode_quantized_shift / cache_key_for ────────────────

def test_quantize_shift_scalar_and_array():
    assert quantize_shift(1.24, 0.25) == pytest.approx(1.25)
    assert quantize_shift(-0.51, 0.25) == pytest.approx(-0.5)
    out = quantize_shift(np.array([1.24, -0.51]), 0.25)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.25, -0.5])


def test_encode_quantized_shift_examples():
    assert encode_quantized_shift(1.25) == "p1p25"
    assert encode_quantized_shift(-0.5) == "n0p50"
    assert encode_quantized_shift(0.0) == "p0p00"


def test_cache_key_for():
    assert cache_key_for(1.25, -0.5) == "qxp1p25_qyn0p50"


# ── (f) parquet + JSON artifact round trip ──────────────────────────────────

def test_write_group_artifacts_schema_round_trip(tmp_path):
    schedule = _make_grouping_schedule()
    assignment = assign_groups_from_schedule(
        schedule, grouping_quantum_ps1_px=1.0, cache_quantum_ps1_px=0.25, keying="phase",
    )
    parquet_path, json_path = write_group_artifacts(
        assignment,
        tmp_path,
        geometry_mode="field",
        grouping_quantum_ps1_px=1.0,
        cache_quantum_ps1_px=0.25,
    )
    assert parquet_path.name == "template_group_shifts.parquet"
    assert json_path.name == "template_groups.json"
    assert parquet_path.is_file()
    assert json_path.is_file()

    df = pd.read_parquet(parquet_path)
    assert list(df.columns) == ["group_id", "skycell", "sx_int", "sy_int", "qx", "qy", "cache_key"]
    assert df["group_id"].dtype == np.int32
    assert df["sx_int"].dtype == np.int16
    assert df["sy_int"].dtype == np.int16
    assert df["qx"].dtype == np.float32
    assert df["qy"].dtype == np.float32
    assert len(df) == len(assignment.shifts_df)

    with open(json_path) as fh:
        payload = json.load(fh)
    assert payload["schema_version"] == 1
    assert payload["geometry_mode"] == "field"
    assert payload["grouping_quantum_ps1_px"] == 1.0
    assert payload["cache_quantum_ps1_px"] == 0.25
    assert payload["n_groups"] == len(assignment.groups)
    assert len(payload["groups"]) == len(assignment.groups)
    for g in payload["groups"]:
        assert set(g.keys()) == {"group_id", "n_frames", "signature_hash"}


# ── (g) ShiftSchedule save/load round trip ──────────────────────────────────

def test_shift_schedule_save_load_round_trip(tmp_path):
    ref_wcs = _make_wcs()
    frames = [
        ("frame0.fits", ref_wcs),
        ("frame1.fits", _shifted_wcs(ref_wcs, (0.2, -0.1))),
        ("frame2.fits", _shifted_wcs(ref_wcs, (0.4, -0.2))),
    ]
    skycell_df = _make_skycell_df()
    schedule = build_skycell_shift_schedule(
        frames, skycell_df, ref_wcs, btjd=np.array([0.0, 0.1, 0.2]),
    )

    npz_path = tmp_path / "shift_schedule.npz"
    schedule.save(npz_path)
    assert npz_path.is_file()
    assert (tmp_path / "shift_schedule.json").is_file()

    loaded = ShiftSchedule.load(npz_path)
    np.testing.assert_array_equal(loaded.skycell_names, schedule.skycell_names)
    np.testing.assert_array_equal(loaded.sx_float, schedule.sx_float)
    np.testing.assert_array_equal(loaded.sy_float, schedule.sy_float)
    np.testing.assert_array_equal(loaded.sx_int, schedule.sx_int)
    np.testing.assert_array_equal(loaded.sy_int, schedule.sy_int)
    np.testing.assert_array_equal(loaded.frame_valid, schedule.frame_valid)
    assert loaded.meta == schedule.meta
    assert loaded.meta["schema_version"] == 1
