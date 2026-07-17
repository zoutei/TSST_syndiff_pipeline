"""Mask settings defaults and resolve order."""

from pathlib import Path

import yaml

from syndiff_pipeline.masking.settings import (
    DEFAULT_TNS_PUBLIC_ZIP_URL,
    DEFAULT_TESS_ORBIT_TIMES_URL,
    MaskSettings,
    apply_stage_overrides,
    load_mask_settings,
    mask_settings_from_dict,
    resolve_mask_settings,
    write_mask_settings,
)


def test_defaults_tns_asteroids_enabled():
    s = MaskSettings()
    assert s.shared.style == "empirical"
    assert s.tns.enabled is True
    assert s.asteroids.enabled is True
    assert s.tns.download_url == DEFAULT_TNS_PUBLIC_ZIP_URL
    assert s.asteroids.orbit_times_url == DEFAULT_TESS_ORBIT_TIMES_URL
    assert s.asteroids.run_discover is True


def test_omitted_download_url_uses_code_default(tmp_path):
    p = tmp_path / "mask_settings.yaml"
    p.write_text(
        "tns:\n  enabled: true\n  public_csv: null\nasteroids:\n  enabled: false\n"
    )
    s = load_mask_settings(p)
    assert s.tns.download_url == DEFAULT_TNS_PUBLIC_ZIP_URL
    assert s.asteroids.enabled is False
    assert s.asteroids.orbit_times_url == DEFAULT_TESS_ORBIT_TIMES_URL


def test_resolve_order_stage_wins(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "mask_settings.yaml").write_text("shared:\n  style: tessreduce\n")
    stage = tmp_path / "stage.yaml"
    stage.write_text("shared:\n  style: empirical\n  bright_maglim: 12.0\n")
    s, used = resolve_mask_settings(
        stage_mask_settings=str(stage),
        site_dir=site,
        ws_root=tmp_path / "ws",
    )
    assert used == stage.resolve()
    assert s.shared.style == "empirical"
    assert s.shared.bright_maglim == 12.0


def test_stage_overrides_bc():
    s = MaskSettings()
    s2 = apply_stage_overrides(s, gaia_mag_bright=12.5, strapsize=8, ps1_min_hit_count=100)
    assert s2.shared.bright_maglim == 12.5
    assert s2.shared.strapsize == 8
    assert s2.shared.ps1_min_hit_count == 100


def test_write_omits_default_download_url(tmp_path):
    path = write_mask_settings(MaskSettings(), tmp_path / "out.yaml")
    raw = yaml.safe_load(path.read_text())
    assert "download_url" not in (raw.get("tns") or {})
    assert "orbit_times_url" not in (raw.get("asteroids") or {})


def test_invalid_style_raises():
    try:
        mask_settings_from_dict({"shared": {"style": "nope"}})
        assert False, "expected ValueError"
    except ValueError:
        pass
