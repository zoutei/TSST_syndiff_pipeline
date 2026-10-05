# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for irregular stamp segmentation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.forward_model import irregular_stamps as IS  # noqa: E402


def test_detect_merged_segments_pads_and_joins_neighbors():
    """Two islands 1 px apart merge after pad_px=1 dilation."""
    cal = np.zeros((20, 20), dtype=float)
    noise = np.ones((20, 20), dtype=float)
    # Two 2x2 islands separated by a 1-pixel gap (columns 8 and 10).
    cal[8:10, 6:8] = 10.0
    cal[8:10, 10:12] = 10.0
    # Without pad: two separate detections.
    raw = IS.detect_merged_segments(cal, noise, n_sigma=3.0, npixels=3, pad_px=0)
    assert raw.n_labels == 2
    # With pad=1: islands grow into the gap and fuse.
    merged = IS.detect_merged_segments(cal, noise, n_sigma=3.0, npixels=3, pad_px=1)
    assert merged.n_labels == 1
    assert int(merged.label_map[8, 7]) > 0
    assert int(merged.label_map[8, 10]) == int(merged.label_map[8, 7])
    # Gap pixel itself should be filled by dilation.
    assert int(merged.label_map[8, 9]) == int(merged.label_map[8, 7])


def test_select_isolated_primaries_min_sep_from_bright_pool():
    # Positions: primary candidates at mag 8; bright neighbors at mag 12.
    x = np.array([10.0, 20.0, 50.0, 12.0, 80.0], dtype=float)
    y = np.array([10.0, 20.0, 50.0, 10.5, 80.0], dtype=float)
    mag = np.array([8.0, 8.5, 9.0, 12.0, 12.5], dtype=float)
    # idx 0 is 2 px from idx 3 (bright) → not isolated at min_sep=6
    # idx 1 is far from all bright → isolated
    # idx 2 is far → isolated
    keep = IS.select_isolated_primaries(
        x, y, mag, mag_lo=7.0, mag_hi=10.0, bright_mag_max=13.0, min_sep_px=6.0,
    )
    assert set(keep.tolist()) == {1, 2}


def test_assign_segment_members_collects_gaia_in_mask():
    label_map = np.zeros((30, 30), dtype=np.int32)
    label_map[10:16, 10:18] = 1
    label_map[20:24, 20:24] = 2
    # Crop-local xy = pixel centers roughly; region origin (0,0).
    x = np.array([12.2, 15.1, 21.0, 5.0], dtype=float)
    y = np.array([12.2, 12.5, 21.0, 5.0], dtype=float)
    mag = np.array([8.0, 11.5, 9.0, 8.5], dtype=float)
    primaries = np.array([0], dtype=int)
    assignments = IS.assign_segment_members(
        label_map, primaries, x, y, mag, bright_mag_max=13.0,
        region_x_min=0.0, region_y_min=0.0,
    )
    assert len(assignments) == 1
    a = assignments[0]
    assert a.segment_label == 1
    assert set(a.member_indices.tolist()) == {0, 1}  # primary + companion in seg 1
    assert a.n_pixels == int((label_map == 1).sum())


def test_assign_skips_primary_on_background():
    label_map = np.zeros((10, 10), dtype=np.int32)
    label_map[2:5, 2:5] = 1
    x = np.array([8.0], dtype=float)
    y = np.array([8.0], dtype=float)
    mag = np.array([8.0], dtype=float)
    assignments = IS.assign_segment_members(
        label_map, np.array([0]), x, y, mag, bright_mag_max=13.0,
    )
    assert assignments == []


def test_mask_to_ds9_polygons_and_reg_roundtrip(tmp_path):
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:12, 5:12] = True
    polys = IS.mask_to_ds9_polygons(mask)
    assert len(polys) >= 1
    assert polys[0].startswith("polygon(")

    a = IS.SegmentAssignment(
        primary_index=0,
        segment_label=1,
        mask=mask,
        member_indices=np.array([0, 1], dtype=int),
        n_pixels=int(mask.sum()),
    )
    x = np.array([8.0, 10.0])
    y = np.array([8.0, 10.0])
    mag = np.array([8.1, 11.2])
    path = IS.write_fit_regions_reg(
        tmp_path / "fit_regions.reg",
        [a],
        x=x, y=y, mag=mag,
        region_x_min=0.0, region_y_min=0.0,
    )
    text = path.read_text()
    assert "polygon(" in text
    assert "circle(" in text
    assert "8.10" in text
    assert "11.20" in text


def test_contaminates_primary_stamp_chebyshev():
    # S=13 → contaminates if L∞ ≤ 12
    assert IS.contaminates_primary_stamp(100.0, 100.0, 106.0, 100.0, stamp_physical=13)
    assert IS.contaminates_primary_stamp(100.0, 100.0, 112.0, 100.0, stamp_physical=13)
    assert not IS.contaminates_primary_stamp(100.0, 100.0, 113.0, 100.0, stamp_physical=13)


def test_detect_erode_breaks_thin_bridge():
    """1-px bridge merges without erode; erode_px=1 splits into two labels."""
    cal = np.zeros((24, 30), dtype=float)
    noise = np.ones_like(cal)
    # Two 5×5 blobs linked by a 1-px-wide horizontal bridge.
    cal[8:13, 5:10] = 20.0
    cal[8:13, 16:21] = 20.0
    cal[10, 10:16] = 20.0  # bridge row
    linked = IS.detect_merged_segments(cal, noise, n_sigma=3.0, npixels=3, pad_px=0, erode_px=0)
    assert linked.n_labels == 1
    split = IS.detect_merged_segments(cal, noise, n_sigma=3.0, npixels=3, pad_px=0, erode_px=1)
    assert split.n_labels == 2
    assert int(split.label_map[10, 7]) > 0
    assert int(split.label_map[10, 18]) > 0
    assert int(split.label_map[10, 7]) != int(split.label_map[10, 18])
    # Bridge pixels should be gone after erosion.
    assert int(split.label_map[10, 12]) == 0


def test_grow_enclosing_compact_segment():
    # 5×5 block → d_max=2 from center → S=5
    ys, xs = np.mgrid[18:23, 18:23]
    s = IS.grow_enclosing_square_size(
        ys.ravel(), xs.ravel(), 20, 20, s_max=13, s_min=5,
    )
    assert s == 5


def test_grow_enclosing_with_pad():
    ys, xs = np.mgrid[18:23, 18:23]
    s = IS.grow_enclosing_square_size(
        ys.ravel(), xs.ravel(), 20, 20, s_max=13, s_min=5, enclose_pad_px=1,
    )
    assert s == 7  # 5 + 2


def test_grow_enclosing_pad_caps_at_smax():
    ys, xs = np.mgrid[14:27, 14:27]  # 13×13 → S=13, pad would be 15 → clamp 13
    s = IS.grow_enclosing_square_size(
        ys.ravel(), xs.ravel(), 20, 20, s_max=13, s_min=5, enclose_pad_px=1,
    )
    assert s == 13


def test_grow_enclosing_larger_segment():
    # 9×9 block → d_max=4 → S=9
    ys, xs = np.mgrid[16:25, 16:25]
    s = IS.grow_enclosing_square_size(
        ys.ravel(), xs.ravel(), 20, 20, s_max=13, s_min=5,
    )
    assert s == 9


def test_grow_enclosing_empty_is_smin():
    s = IS.grow_enclosing_square_size(
        np.array([], dtype=int), np.array([], dtype=int), 20, 20, s_max=13, s_min=5,
    )
    assert s == 5


def test_grow_enclosing_caps_at_smax():
    ys, xs = np.mgrid[10:31, 10:31]  # 21×21 → would need S=21
    s = IS.grow_enclosing_square_size(
        ys.ravel(), xs.ravel(), 20, 20, s_max=13, s_min=5,
    )
    assert s == 13


def test_shrink_when_outer_ring_empty():
    label_map = np.zeros((40, 40), dtype=np.int32)
    label_map[18:23, 18:23] = 1
    s = IS.shrink_square_size(label_map, 20, 20, star_label=1, s_max=13, s_min=5)
    assert s == 5


def test_shrink_stops_when_outer_ring_hits_segment():
    label_map = np.zeros((40, 40), dtype=np.int32)
    label_map[16:25, 16:25] = 1
    s = IS.shrink_square_size(label_map, 20, 20, star_label=1, s_max=13, s_min=5)
    assert s == 9


def test_shrink_no_segment_falls_back_to_smax():
    label_map = np.zeros((40, 40), dtype=np.int32)
    s = IS.shrink_square_size(label_map, 20, 20, star_label=0, s_max=13, s_min=5)
    assert s == 13


def test_build_epsf_support_isolated_is_s_by_s():
    ny = nx = 64
    cal = np.zeros((ny, nx), dtype=float)
    noise = np.ones((ny, nx), dtype=float)
    cal[29:36, 29:36] = 20.0  # 7×7 blob around (32,32)
    x = np.array([32.2, 55.0], dtype=float)
    y = np.array([32.1, 55.0], dtype=float)
    mag = np.array([8.0, 11.0], dtype=float)
    asn = IS.build_epsf_support_stamps(
        np.array([0]), x, y, mag,
        cal=cal, noise=noise, stamp_physical=13, stamp_min=5,
        region_x_min=0, region_y_min=0,
        prune_snr=False, n_sigma=5.0, npixels=3,
        erode_px=1, enclose_pad_px=0,
    )
    assert len(asn) == 1
    a = asn[0]
    assert list(a.member_indices) == [0]
    assert a.member_stamp_sizes is not None
    # Erode 7×7 → ~5×5 enclose S=5 (no square pad)
    assert int(a.member_stamp_sizes[0]) == 5
    assert a.n_pixels == 5 * 5
    assert a.stamp_center_x == 32 and a.stamp_center_y == 32
    assert a.segment_label > 0


def test_build_epsf_support_same_segment_companions_enclose():
    """Companions attach only if they share the eroded segment; sizes enclose it."""
    ny = nx = 80
    cal = np.zeros((ny, nx), dtype=float)
    noise = np.ones((ny, nx), dtype=float)
    # Thick connected bar (survives 1-px erode) covering primary and companion.
    cal[37:44, 37:52] = 20.0
    cal[68:73, 68:73] = 20.0  # separate blob
    x = np.array([40.0, 48.0, 70.0], dtype=float)
    y = np.array([40.0, 40.0, 70.0], dtype=float)
    mag = np.array([8.0, 11.5, 12.0], dtype=float)
    asn = IS.build_epsf_support_stamps(
        np.array([0]), x, y, mag,
        cal=cal, noise=noise, stamp_physical=13, stamp_min=5,
        region_x_min=0, region_y_min=0,
        prune_snr=False, n_sigma=5.0, npixels=3,
        erode_px=1, enclose_pad_px=0,
    )
    a = asn[0]
    assert set(a.member_indices.tolist()) == {0, 1}
    assert 2 not in set(a.member_indices.tolist())
    lab = int(a.segment_label)
    assert lab > 0
    seg = IS.detect_merged_segments(
        cal, noise, n_sigma=5.0, npixels=3, pad_px=0, erode_px=1,
    )
    ys, xs = np.where(seg.label_map == lab)
    half = 13 // 2
    for iy, ix in zip(ys, xs):
        d0 = max(abs(ix - 40), abs(iy - 40))
        d1 = max(abs(ix - 48), abs(iy - 40))
        if min(d0, d1) > half:
            continue
        assert a.mask[iy, ix], f"segment pixel ({iy},{ix}) not covered"


def test_build_epsf_support_bridge_does_not_attach():
    """Thin bridge between islands is broken by erode → no false companion."""
    cal = np.zeros((40, 50), dtype=float)
    noise = np.ones_like(cal)
    cal[15:22, 10:17] = 20.0
    cal[15:22, 25:32] = 20.0
    cal[18, 17:25] = 20.0  # 1-px bridge
    x = np.array([13.0, 28.0], dtype=float)
    y = np.array([18.0, 18.0], dtype=float)
    mag = np.array([8.0, 11.0], dtype=float)
    asn = IS.build_epsf_support_stamps(
        np.array([0]), x, y, mag,
        cal=cal, noise=noise, stamp_physical=13, stamp_min=5,
        prune_snr=False, n_sigma=5.0, npixels=3,
        erode_px=1, enclose_pad_px=0,
    )
    assert list(asn[0].member_indices) == [0]


def test_build_epsf_support_separate_segments_no_chebyshev_attach():
    """Nearby but separate islands must not become companions (no pad)."""
    ny = nx = 80
    cal = np.zeros((ny, nx), dtype=float)
    noise = np.ones((ny, nx), dtype=float)
    cal[38:43, 38:43] = 20.0
    cal[38:43, 46:51] = 20.0
    x = np.array([40.0, 48.0], dtype=float)
    y = np.array([40.0, 40.0], dtype=float)
    mag = np.array([8.0, 11.5], dtype=float)
    asn = IS.build_epsf_support_stamps(
        np.array([0]), x, y, mag,
        cal=cal, noise=noise, stamp_physical=13, stamp_min=5,
        prune_snr=False, n_sigma=5.0, npixels=3,
        erode_px=1, enclose_pad_px=0,
    )
    assert list(asn[0].member_indices) == [0]


def test_build_epsf_support_merges_co_segment_primaries():
    """Two isolated primaries on one eroded island → one shared stamp."""
    cal = np.zeros((80, 80), dtype=float)
    noise = np.ones((80, 80), dtype=float)
    cal[37:44, 37:52] = 20.0  # thick bar covering both
    x = np.array([40.0, 48.0], dtype=float)
    y = np.array([40.0, 40.0], dtype=float)
    mag = np.array([8.0, 8.5], dtype=float)
    asn = IS.build_epsf_support_stamps(
        np.array([0, 1]), x, y, mag,
        cal=cal, noise=noise, stamp_physical=13,
        prune_snr=False, erode_px=1, enclose_pad_px=0, npixels=3,
    )
    assert len(asn) == 1
    assert asn[0].primary_index == 0  # brighter primary
    assert set(asn[0].member_indices.tolist()) == {0, 1}


def test_build_epsf_support_rejects_when_k_gt_max():
    """More than max_group_size bright stars on the segment → no stamp."""
    cal = np.zeros((80, 80), dtype=float)
    noise = np.ones((80, 80), dtype=float)
    cal[30:55, 30:55] = 20.0
    # 1 primary + 4 companions = K=5 > 4
    x = np.array([40.0, 42.0, 44.0, 46.0, 48.0], dtype=float)
    y = np.array([40.0, 42.0, 44.0, 46.0, 48.0], dtype=float)
    mag = np.array([8.0, 11.0, 11.2, 11.4, 11.6], dtype=float)
    asn = IS.build_epsf_support_stamps(
        np.array([0]), x, y, mag,
        cal=cal, noise=noise, stamp_physical=13, max_group_size=4,
        prune_snr=False, erode_px=1, enclose_pad_px=0, npixels=3,
    )
    assert asn == []


def test_find_overlapping_stamp_pairs():
    m0 = np.zeros((20, 20), dtype=bool)
    m1 = np.zeros((20, 20), dtype=bool)
    m2 = np.zeros((20, 20), dtype=bool)
    m0[5:10, 5:10] = True
    m1[8:14, 8:14] = True  # overlaps m0
    m2[15:18, 15:18] = True  # isolated
    a0 = IS.SegmentAssignment(0, 1, m0, np.array([0]), int(m0.sum()), bbox=(5, 10, 5, 10))
    a1 = IS.SegmentAssignment(1, 2, m1, np.array([1]), int(m1.sum()), bbox=(8, 14, 8, 14))
    a2 = IS.SegmentAssignment(2, 3, m2, np.array([2]), int(m2.sum()), bbox=(15, 18, 15, 18))
    pairs = IS.find_overlapping_stamp_pairs([a0, a1, a2])
    assert len(pairs) == 1
    assert pairs[0][0] == 0 and pairs[0][1] == 1
    assert pairs[0][2] == int(np.logical_and(m0, m1).sum())


def test_resolve_overlapping_pixels_nearest_brightest():
    """Contested pixels stay with the stamp whose brightest star is closer."""
    m0 = np.zeros((30, 30), dtype=bool)
    m1 = np.zeros((30, 30), dtype=bool)
    m0[10:18, 10:18] = True
    m1[14:22, 14:22] = True  # overlap block [14:18, 14:18]
    # Star 0 at (12,12) brighter; star 1 at (20,20) fainter.
    x = np.array([12.0, 20.0], dtype=float)
    y = np.array([12.0, 20.0], dtype=float)
    mag = np.array([8.0, 10.0], dtype=float)
    a0 = IS.SegmentAssignment(
        0, 1, m0.copy(), np.array([0]), int(m0.sum()),
        stamp_center_x=12, stamp_center_y=12, bbox=(10, 18, 10, 18),
    )
    a1 = IS.SegmentAssignment(
        1, 2, m1.copy(), np.array([1]), int(m1.sum()),
        stamp_center_x=20, stamp_center_y=20, bbox=(14, 22, 14, 22),
    )
    out, stats = IS.resolve_overlapping_stamp_pixels([a0, a1], x, y, mag)
    assert stats["n_contested_px"] > 0
    assert stats["n_pixels_removed"] == stats["n_contested_px"]
    assert IS.find_overlapping_stamp_pairs(out) == []
    # Corner of overlap nearer star0 should remain on stamp0 only.
    assert out[0].mask[14, 14] and not out[1].mask[14, 14]
    # Corner nearer star1 should remain on stamp1 only.
    assert out[1].mask[17, 17] and not out[0].mask[17, 17]


def test_build_epsf_support_resolves_overlaps():
    """build_epsf_support_stamps returns exclusive supports (no mask overlaps)."""
    cal = np.zeros((60, 60), dtype=float)
    noise = np.ones_like(cal)
    # Two bright islands close enough that 13×13 squares can touch.
    cal[20:29, 20:29] = 20.0
    cal[20:29, 28:37] = 20.0  # 1 px gap before erode; after erode may still be close
    x = np.array([24.0, 32.0], dtype=float)
    y = np.array([24.0, 24.0], dtype=float)
    mag = np.array([8.0, 8.5], dtype=float)
    asn = IS.build_epsf_support_stamps(
        np.array([0, 1]), x, y, mag,
        cal=cal, noise=noise, stamp_physical=13, stamp_min=5,
        prune_snr=False, erode_px=1, enclose_pad_px=0, npixels=3,
        max_group_size=4,
    )
    assert len(asn) >= 1
    assert IS.find_overlapping_stamp_pairs(asn) == []


def test_prune_snr_removes_low_snr_pixels():
    cal = np.zeros((40, 40), dtype=float)
    noise = np.ones((40, 40), dtype=float)
    cal[18:23, 18:23] = 10.0  # survive erode_px=1
    x = np.array([20.0], dtype=float)
    y = np.array([20.0], dtype=float)
    mag = np.array([8.0], dtype=float)
    asn = IS.build_epsf_support_stamps(
        np.array([0]), x, y, mag,
        cal=cal, noise=noise, stamp_physical=13,
        prune_snr=True, n_sigma=5.0, npixels=1,
        erode_px=1, enclose_pad_px=0,
    )
    assert asn[0].n_pixels >= 1
    assert asn[0].pix_x is not None


def test_build_epsf_support_foreign_touch_merges():
    """If the sized square covers a foreign Gaia-bearing label, merge islands."""
    # Adjacent labels sharing a column in the primary's enclosing square.
    label_map = np.zeros((40, 40), dtype=np.int32)
    label_map[10:21, 10:20] = 1  # primary island
    label_map[10:21, 20:30] = 2  # foreign island (col 20 is in S=11 around 15)
    cal = (label_map > 0).astype(float) * 20.0
    noise = np.ones_like(cal)
    x = np.array([15.0, 24.0], dtype=float)
    y = np.array([15.0, 15.0], dtype=float)
    mag = np.array([8.0, 11.0], dtype=float)
    # Separate enough that Chebyshev attach is irrelevant; merge via square touch.
    asn = IS.build_epsf_support_stamps(
        np.array([0]), x, y, mag,
        cal=cal, noise=noise, label_map=label_map, stamp_physical=13,
        prune_snr=False, erode_px=0, enclose_pad_px=0, npixels=3,
    )
    assert len(asn) == 1
    assert set(asn[0].member_indices.tolist()) == {0, 1}


def test_build_epsf_support_close_islands_no_pad_stay_separate():
    """With enclose_pad_px=0, close-but-separate islands stay separate."""
    cal = np.zeros((60, 60), dtype=float)
    noise = np.ones_like(cal)
    cal[20:29, 20:29] = 20.0
    cal[20:29, 31:40] = 20.0  # 2 px gap
    x = np.array([24.0, 35.0], dtype=float)
    y = np.array([24.0, 24.0], dtype=float)
    mag = np.array([8.0, 11.0], dtype=float)
    asn = IS.build_epsf_support_stamps(
        np.array([0]), x, y, mag,
        cal=cal, noise=noise, stamp_physical=13,
        prune_snr=False, erode_px=1, enclose_pad_px=0, npixels=3,
    )
    assert len(asn) == 1
    assert list(asn[0].member_indices) == [0]

