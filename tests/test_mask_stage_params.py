"""shared_mask stage param allowlist includes mask_settings."""

from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    SHARED_MASK_ALLOWED,
    parse_shared_mask,
)


def test_mask_settings_allowlisted():
    assert "mask_settings" in SHARED_MASK_ALLOWED
    p = parse_shared_mask(
        {"kind": "shared_mask", "mask_settings": "/tmp/m.yaml", "gaia_mag_bright": 12},
        0,
    )
    assert p.mask_settings == "/tmp/m.yaml"
    assert p.gaia_mag_bright == 12.0
