# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Direct v2 creation and tier-native transcode regression tests."""

import numpy as np
import pytest

from syndiff_pipeline.forward_model import fit_bundle as FB
from syndiff_pipeline.forward_model import packed_support as PS
from syndiff_pipeline.forward_model.scripts.transcode_bundle_tiered import transcode
from test_fm_fit_bundle_ragged import (
    _packed_dense_bundle_and_tiers,
)


def test_dense_to_tiered_roundtrip(tmp_path):
    dense, _expected, _ = _packed_dense_bundle_and_tiers()
    tiered = FB.tiered_packed_bundle(dense)
    assert tiered.packed_tiers is not None
    path = FB.save_fit_bundle(tmp_path, tiered)
    loaded = FB.load_fit_bundle(path)
    np.testing.assert_array_equal(loaded.data, dense.data)
    np.testing.assert_array_equal(loaded.noise, dense.noise)
    np.testing.assert_array_equal(loaded.weight, dense.weight)
    np.testing.assert_array_equal(loaded.pix_valid, dense.pix_valid)


def test_direct_tiers_from_batches_match_dense():
    dense, expected, _ = _packed_dense_bundle_and_tiers()
    batches = []
    for tier in expected:
        gi = tier.group_idx
        batches.append(PS.PackedStampBatch(
            data=tier.data, noise=tier.noise, weight=tier.weight_f32,
            pix_x=tier.pix_x, pix_y=tier.pix_y, pix_valid=tier.pix_valid,
            members=dense.members[gi, :tier.k_tier],
            valid=dense.valid[gi, :tier.k_tier],
            k_tier=tier.k_tier, p_tier=tier.p_tier,
            orig_stamp_idx=gi,
        ))
    direct = FB.bundle_from_tiers(dense, PS.packed_tiers_from_batches(batches))
    np.testing.assert_array_equal(direct.data, dense.data)
    np.testing.assert_array_equal(direct.noise, dense.noise)
    np.testing.assert_array_equal(direct.weight, dense.weight)


def test_already_tiered_transcode_never_materializes_dense_views():
    dense, expected, _ = _packed_dense_bundle_and_tiers()
    tiered = FB.bundle_from_tiers(dense, expected)
    assert not any(k in tiered.__dict__ for k in ("data", "noise", "weight", "pix_valid"))
    out = transcode(tiered, k_tiers=(1, 2), p_tiers=(4, 8), log_fn=lambda *_: None)
    assert not any(k in tiered.__dict__ for k in ("data", "noise", "weight", "pix_valid"))
    assert out.packed_tiers is not None
    for got, want in zip(out.packed_tiers, expected):
        np.testing.assert_array_equal(got.data, want.data)
        np.testing.assert_array_equal(got.group_idx, want.group_idx)


def test_tier_validation_rejects_duplicate_group():
    dense, tiers, _ = _packed_dense_bundle_and_tiers()
    tiers[1].group_idx[0] = 0
    with pytest.raises(ValueError, match="overlaps"):
        FB.bundle_from_tiers(dense, tiers)


def test_direct_batches_reject_nonbinary_weight():
    dense, tiers, _ = _packed_dense_bundle_and_tiers()
    tier = tiers[0]
    batch = PS.PackedStampBatch(
        data=tier.data, noise=tier.noise, weight=tier.weight_f32,
        pix_x=tier.pix_x, pix_y=tier.pix_y, pix_valid=tier.pix_valid,
        members=dense.members[[0], :1], valid=dense.valid[[0], :1],
        k_tier=1, p_tier=4, orig_stamp_idx=np.array([0]),
    )
    batch.weight[0, 0, 0] = .5
    with pytest.raises(ValueError, match="binary"):
        PS.packed_tiers_from_batches([batch])
