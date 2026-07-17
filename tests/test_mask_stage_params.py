"""shared_mask stage params: mask_settings + legacy overrides."""

from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    SHARED_MASK_ALLOWED,
    legacy_mask_stage_overrides,
    parse_shared_mask,
)


def test_mask_settings_allowlisted():
    assert "mask_settings" in SHARED_MASK_ALLOWED
    p = parse_shared_mask(
        {"kind": "shared_mask", "mask_settings": "/tmp/m.yaml", "gaia_mag_bright": 12},
        0,
    )
    assert p.mask_settings == "/tmp/m.yaml"
    assert not hasattr(p, "gaia_mag_bright")


def test_legacy_overrides_only_when_explicit():
    assert legacy_mask_stage_overrides({"kind": "shared_mask"}) == {}
    assert legacy_mask_stage_overrides(
        {"kind": "shared_mask", "gaia_mag_bright": 12.0, "strapsize": 8}
    ) == {"gaia_mag_bright": 12.0, "strapsize": 8}
