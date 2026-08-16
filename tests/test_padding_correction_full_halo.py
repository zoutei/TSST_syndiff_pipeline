"""Unit guardrails for the standalone cross-projection seam correction.

Covers ``convolve_local_padding_delta`` (the geometry-free local-domain
convolution primitive) and ``load_padding_aware_convolved_cell`` (the public
entry point implementing
``doc/shared_convolved_cross_projection_simple_fix_plan.md``), reusing the
same primitive for both edges and corners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.template_creation.processing.convolution_utils import (
    apply_gaussian_convolution,
)
from syndiff_pipeline.template_creation.processing.padding_correction import (
    PaddingCorrectionError,
    convolve_local_padding_delta,
    cross_projection_padding_spec,
    load_padding_aware_convolved_cell,
    padding_spec_fingerprint,
)


def test_external_flux_is_convolved_before_recipient_crop():
    """Flux outside the recipient image must contribute across its top edge."""
    # Native y=[7, 24) surrounds a 20x16 recipient with a 3px top halo.
    # The source pixel at y=21 is outside the recipient but within one
    # truncated-kernel radius of its top edge.
    a = np.zeros((17, 16), dtype=np.float64)
    f = a.copy()
    f[14, 8] = 1.0  # native (x=8, y=21)

    got = convolve_local_padding_delta(
        a, f, local_origin_xy=(0, 7), canonical_shape=(20, 16),
        psf_sigma=1.0, kernel_radius=3,
    )

    assert got[19, 8] > 0.0
    assert np.allclose(got[:16], 0.0, atol=1e-15)


def test_delta_crop_matches_full_domain_convolution_with_overlap_replacement():
    """An overlap replacement contributes F-A, not an independently added F."""
    a = np.zeros((17, 16), dtype=np.float64)
    a[12, 8] = 5.0  # native y=19, inside the recipient overlap strip
    f = a.copy()
    f[12, 8] = 7.0  # producer-style replacement makes D exactly +2 here
    f[14, 8] = 11.0  # and adds genuinely exterior source flux

    got = convolve_local_padding_delta(
        a, f, local_origin_xy=(0, 7), canonical_shape=(20, 16),
        psf_sigma=1.0, kernel_radius=3,
    )

    expected_local = apply_gaussian_convolution(
        f - a, sigma=1.0, radius=3, cval=0.0
    )
    expected = np.zeros((20, 16), dtype=np.float64)
    expected[7:20] = expected_local[:13]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    # If F had been convolved and added directly, the old A=5 overlap flux
    # would have been double-counted.  The exact correction is only +2.
    assert got[19, 8] < apply_gaussian_convolution(
        f, sigma=1.0, radius=3, cval=0.0
    )[12, 8]


def test_invalid_local_delta_inputs_fail_explicitly():
    with pytest.raises(PaddingCorrectionError, match="same-shaped 2-D"):
        convolve_local_padding_delta(
            np.zeros((2, 2)), np.zeros((3, 2)), local_origin_xy=(0, 0),
            canonical_shape=(2, 2), psf_sigma=1.0,
        )


def test_padding_spec_preserves_column_and_slash_source_priority():
    row = pd.Series({
        "projection": "skycell.1234",
        "pad_skycell_top": "skycell.9999.001/skycell.8888.002",
        "pad_skycell_right": "skycell.7777.003",
        # Must be recognized as same-projection despite different spelling.
        "pad_skycell_bottom": "skycell.1234.004",
    })
    spec = cross_projection_padding_spec(row)
    assert spec == [
        {"neighbor": "skycell.9999.001", "location": "top"},
        {"neighbor": "skycell.8888.002", "location": "top"},
        {"neighbor": "skycell.7777.003", "location": "right"},
    ]
    assert padding_spec_fingerprint(spec) != padding_spec_fingerprint(list(reversed(spec)))


def _flat_cell_row(*, projection: str, crval2: float, crpix: float = 1.0) -> pd.Series:
    return pd.Series({
        "projection": projection,
        "CRVAL1": 10.0,
        "CRVAL2": crval2,
        "CRPIX1": crpix,
        "CRPIX2": crpix,
        "CD1_1": -1.0 / 3600, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1.0 / 3600,
    })


@pytest.fixture
def cross_projection_fixture(monkeypatch):
    """A recipient cell with one same-projection "top" neighbor requiring
    cross-projection padding, plus fakes for the shared canonical/combined
    stores so the correction can run without any real published artifacts."""
    size = 700  # >> PAD_SIZE(480)+EDGE_EXCLUSION(10) so bounds stay well-posed
    recipient = "skycell.1111.050"
    neighbor = "skycell.2222.099"

    recipient_row = _flat_cell_row(projection="1111", crval2=20.0)
    # Placed directly above the recipient (world offset ~= cell height in deg).
    neighbor_row = _flat_cell_row(projection="2222", crval2=20.0 + size / 3600.0)
    recipient_row["pad_skycell_top"] = neighbor
    skycell_df = pd.DataFrame([recipient_row, neighbor_row], index=[recipient, neighbor])

    own_combined = np.zeros((size, size), dtype=np.float64)
    own_combined[-5, size // 2] = 3.0  # a value inside the native overlap strip

    neighbor_combined = np.zeros((size, size), dtype=np.float64)
    # A bright source close to the neighbor's edge facing the recipient (its
    # own low-row/low-Dec edge, since the neighbor sits north of the
    # recipient) -- just outside the neighbor's own 10px edge exclusion.
    neighbor_combined[15, size // 2] = 300.0

    canonical_image = np.zeros((size, size), dtype=np.float32)
    canonical_mask = np.zeros((size, size), dtype=np.int32)

    def fake_shared(data_root, skycell):
        assert skycell == recipient
        return canonical_image.copy(), canonical_mask.copy()

    def fake_combined(data_root, projection, cell):
        if projection == "skycell.1111":
            return own_combined.copy()
        if projection == "skycell.2222":
            return neighbor_combined.copy()
        return None

    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_downsample."
        "_try_load_shared_convolved_arrays",
        fake_shared,
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.padding_correction._load_combined_image",
        fake_combined,
    )
    return {
        "skycell_df": skycell_df, "recipient": recipient, "size": size,
        "own_combined": own_combined, "neighbor_combined": neighbor_combined,
    }


def test_required_cross_projection_cell_is_corrected_not_raised(
    tmp_path, cross_projection_fixture
):
    """Doc test 1 (edge source): a required cross-projection cell now gets an
    additive correction instead of the old hard ``PaddingCorrectionError``."""
    f = cross_projection_fixture
    corrected_image, corrected_mask = load_padding_aware_convolved_cell(
        tmp_path, f["recipient"], skycell_df=f["skycell_df"], psf_sigma=5.0, kernel_radius=20,
    )
    size = f["size"]
    assert corrected_image.shape == (size, size)
    # Flux blurred in from the neighbor's bright pixel must reach the top rows.
    assert corrected_image[-1, size // 2] > 0.0
    # Far from the seam, the correction must vanish (no changes elsewhere).
    assert np.allclose(corrected_image[: size - 60], 0.0, atol=1e-8)
    np.testing.assert_array_equal(corrected_mask, np.zeros((size, size), dtype=np.int32))


def test_unaffected_cell_is_returned_unchanged(tmp_path, cross_projection_fixture):
    f = cross_projection_fixture
    other_row = f["skycell_df"].loc[f["recipient"]].copy()
    other_row["pad_skycell_top"] = ""  # no cross-projection requirement
    other_df = f["skycell_df"].copy()
    other_df.loc[f["recipient"]] = other_row

    image, mask = load_padding_aware_convolved_cell(
        tmp_path, f["recipient"], skycell_df=other_df, psf_sigma=5.0, kernel_radius=20,
    )
    np.testing.assert_array_equal(image, np.zeros((f["size"], f["size"]), dtype=np.float32))


def test_overlap_strip_replacement_does_not_double_count(tmp_path, monkeypatch):
    """Doc test 3: when the reprojected source exactly matches the recipient's
    own pre-existing value in the native 10px overlap strip, the correction
    there must be ~zero (the strip is a *replacement*, not an addition)."""
    size = 700
    recipient, neighbor = "skycell.1111.050", "skycell.2222.099"
    recipient_row = _flat_cell_row(projection="1111", crval2=20.0)
    neighbor_row = _flat_cell_row(projection="2222", crval2=20.0 + size / 3600.0)
    recipient_row["pad_skycell_top"] = neighbor
    skycell_df = pd.DataFrame([recipient_row, neighbor_row], index=[recipient, neighbor])

    own_combined = np.zeros((size, size), dtype=np.float64)
    own_combined[-5:, :] = 9.0  # native overlap strip pre-existing value

    # Neighbor reprojects (approximately) to the same flat value everywhere,
    # so the exact replacement contributes zero net change in that strip.
    neighbor_combined = np.full((size, size), 9.0, dtype=np.float64)

    canonical_image = np.zeros((size, size), dtype=np.float32)
    canonical_mask = np.zeros((size, size), dtype=np.int32)

    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_downsample."
        "_try_load_shared_convolved_arrays",
        lambda data_root, skycell: (canonical_image.copy(), canonical_mask.copy()),
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.padding_correction._load_combined_image",
        lambda data_root, projection, cell: (
            own_combined.copy() if projection == "skycell.1111" else neighbor_combined.copy()
        ),
    )

    corrected_image, _ = load_padding_aware_convolved_cell(
        tmp_path, recipient, skycell_df=skycell_df, psf_sigma=5.0, kernel_radius=20,
    )
    # Same value in and out -> the delta (F - A) is ~0 near the seam, so the
    # correction must not add flux there. A narrow band right at the fixed
    # 10px edge-exclusion boundary can show small bilinear blending against
    # the excluded (NaN) neighbor rows; that is unrelated to double-counting
    # (which would show up as ~9.0, not a small fraction of it).
    assert np.nanmax(np.abs(corrected_image)) < 1.0


def test_missing_neighbor_combined_cell_raises(tmp_path, monkeypatch):
    recipient, neighbor = "skycell.1111.050", "skycell.2222.099"
    recipient_row = _flat_cell_row(projection="1111", crval2=20.0)
    neighbor_row = _flat_cell_row(projection="2222", crval2=20.0 + 700 / 3600.0)
    recipient_row["pad_skycell_top"] = neighbor
    skycell_df = pd.DataFrame([recipient_row, neighbor_row], index=[recipient, neighbor])

    canonical_image = np.zeros((700, 700), dtype=np.float32)
    canonical_mask = np.zeros((700, 700), dtype=np.int32)
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_downsample."
        "_try_load_shared_convolved_arrays",
        lambda data_root, skycell: (canonical_image.copy(), canonical_mask.copy()),
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.padding_correction._load_combined_image",
        lambda data_root, projection, cell: None,
    )

    with pytest.raises(PaddingCorrectionError, match="is unavailable"):
        load_padding_aware_convolved_cell(
            tmp_path, recipient, skycell_df=skycell_df, psf_sigma=5.0, kernel_radius=20,
        )


def test_multi_source_replacement_follows_mapping_table_slash_order(tmp_path, monkeypatch):
    """Doc test 4: two slash-separated sources at one location must be
    composed by replacement in mapping-table/slash order -- the later source
    wins wherever its footprint overlaps the earlier one, matching
    ``process_ps1``'s own ordered-replacement composition."""
    size = 700
    recipient = "skycell.1111.050"
    first, second = "skycell.2222.099", "skycell.3333.077"
    recipient_row = _flat_cell_row(projection="1111", crval2=20.0)
    neighbor_row_a = _flat_cell_row(projection="2222", crval2=20.0 + size / 3600.0)
    neighbor_row_b = _flat_cell_row(projection="3333", crval2=20.0 + size / 3600.0)
    recipient_row["pad_skycell_top"] = f"{first}/{second}"
    skycell_df = pd.DataFrame(
        [recipient_row, neighbor_row_a, neighbor_row_b],
        index=[recipient, first, second],
    )

    own_combined = np.zeros((size, size), dtype=np.float64)
    first_combined = np.full((size, size), 100.0, dtype=np.float64)
    second_combined = np.full((size, size), 5.0, dtype=np.float64)

    canonical_image = np.zeros((size, size), dtype=np.float32)
    canonical_mask = np.zeros((size, size), dtype=np.int32)

    def fake_combined(data_root, projection, cell):
        if projection == "skycell.1111":
            return own_combined.copy()
        if projection == "skycell.2222":
            return first_combined.copy()
        if projection == "skycell.3333":
            return second_combined.copy()
        return None

    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_downsample."
        "_try_load_shared_convolved_arrays",
        lambda data_root, skycell: (canonical_image.copy(), canonical_mask.copy()),
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.padding_correction._load_combined_image",
        fake_combined,
    )

    corrected_second_last, _ = load_padding_aware_convolved_cell(
        tmp_path, recipient, skycell_df=skycell_df, psf_sigma=5.0, kernel_radius=20,
    )

    # Swap the slash order: the second source now comes first.
    skycell_df_reversed = skycell_df.copy()
    skycell_df_reversed.loc[recipient, "pad_skycell_top"] = f"{second}/{first}"
    corrected_first_last, _ = load_padding_aware_convolved_cell(
        tmp_path, recipient, skycell_df=skycell_df_reversed, psf_sigma=5.0, kernel_radius=20,
    )

    # Both full-flat neighbors cover the whole padding box, so whichever
    # source is *last* in slash order should dominate the final replacement
    # -- the two orderings must therefore disagree, and the magnitude of the
    # correction should track the *last* source's (larger) delta from the
    # recipient's own value, not the first/discarded one.
    assert not np.allclose(corrected_second_last, corrected_first_last)
    center = size // 2
    assert abs(corrected_second_last[-1, center]) < abs(corrected_first_last[-1, center])


def test_linear_and_field_receive_identical_corrected_array(tmp_path, monkeypatch):
    """Doc test 6: linear and field downsample must call the same standalone
    correction and see an identical corrected shared array for the same
    skycell. Patched at the lowest level (fingerprint discovery + raw array
    load + neighbor combined images) so both modules' *real*, unpatched
    dispatch functions run end-to-end through ``load_padding_aware_convolved_cell``."""
    from syndiff_pipeline.template_creation.processing import field_downsample, linear_downsample

    size = 700
    recipient, neighbor = "skycell.1111.050", "skycell.2222.099"
    recipient_row = _flat_cell_row(projection="1111", crval2=20.0)
    neighbor_row = _flat_cell_row(projection="2222", crval2=20.0 + size / 3600.0)
    recipient_row["pad_skycell_top"] = neighbor
    skycell_df = pd.DataFrame([recipient_row, neighbor_row], index=[recipient, neighbor])

    own_combined = np.zeros((size, size), dtype=np.float64)
    own_combined[-5, size // 2] = 3.0
    neighbor_combined = np.zeros((size, size), dtype=np.float64)
    neighbor_combined[15, size // 2] = 300.0

    canonical_image = np.zeros((size, size), dtype=np.float32)
    canonical_mask = np.zeros((size, size), dtype=np.int32)

    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.field_downsample."
        "_discover_shared_convolved_fp",
        lambda data_root, projection, cell: "fake-fp",
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.convolved_store."
        "try_load_convolved_cell",
        lambda data_root, projection, cell, fp: {
            "convolved_image": canonical_image.copy(),
            "convolved_mask": canonical_mask.copy(),
        },
    )
    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.padding_correction._load_combined_image",
        lambda data_root, projection, cell: (
            own_combined.copy() if projection == "skycell.1111" else neighbor_combined.copy()
        ),
    )

    field_image, field_mask = field_downsample._try_load_shared_convolved_arrays(
        tmp_path, recipient, skycell_df=skycell_df, psf_sigma=5.0,
    )
    linear_image, linear_mask = linear_downsample._load_ps1_skycell(
        recipient,
        data_root=tmp_path,
        shared_convolved_store=True,
        legacy_zarr_path=None,
        zstore_cache={},
        skycell_df=skycell_df,
        psf_sigma=5.0,
    )

    assert field_image is not None and np.any(field_image != 0.0)
    np.testing.assert_array_equal(field_image, linear_image)
    np.testing.assert_array_equal(field_mask, linear_mask)
