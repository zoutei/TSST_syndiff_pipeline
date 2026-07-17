"""Slim frozen / saved SynDiffConfig YAML snapshots."""

from pathlib import Path

from syndiff_pipeline.difference_imaging.orchestration.config import (
    SynDiffConfig,
    cfg_to_snapshot_dict,
    load_config,
    save_config,
)
from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
    tess_straps_csv,
)


def test_snapshot_omits_empties_and_defaults():
    cfg = SynDiffConfig(
        ffi_dir="/data/tess_ffi",
        output_dir="/ws/events/t",
        data_root="/data",
        sector=20,
        camera=3,
        ccd=1,
        target_ra=1.0,
        target_dec=2.0,
        target_name="t",
        max_ffis=25,
        pipeline=[{"kind": "shared_mask"}],
    )
    d = cfg_to_snapshot_dict(cfg)
    assert d["max_ffis"] == 25
    assert "manifest" not in d
    assert "removed_stars_csv" not in d
    assert "ref_ffi_min_earth_deg" not in d
    assert "template_paths" not in d
    assert d["pipeline"] == [{"kind": "shared_mask"}]


def test_snapshot_omits_bundled_straps_path():
    cfg = SynDiffConfig(
        ffi_dir="/data/tess_ffi",
        output_dir="/ws/events/t",
        data_root="/data",
        straps_csv=str(tess_straps_csv()),
        pipeline=[{"kind": "shared_mask"}],
    )
    d = cfg_to_snapshot_dict(cfg)
    assert "straps_csv" not in d


def test_snapshot_slims_hotpants_defaults():
    cfg = SynDiffConfig(
        ffi_dir="/data/tess_ffi",
        output_dir="/ws/events/t",
        data_root="/data",
        pipeline=[
            {
                "kind": "hotpants",
                "hp_ko": 2,  # default
                "write_convolved": False,
                "output": {"diffs": "hp_d"},
            }
        ],
    )
    stage = cfg_to_snapshot_dict(cfg)["pipeline"][0]
    assert stage["kind"] == "hotpants"
    assert "hp_ko" not in stage
    assert stage["write_convolved"] is False
    assert stage["output"] == {"diffs": "hp_d"}


def test_load_config_accepts_legacy_full_dump(tmp_path: Path):
    legacy = tmp_path / "full.yaml"
    legacy.write_text(
        "\n".join(
            [
                "ffi_dir: /data/tess_ffi",
                "output_dir: /ws/events/t",
                "manifest: ''",
                "pipeline:",
                "  - kind: shared_mask",
                "gaia_catalog: ''",
                "removed_stars_csv: ''",
                "straps_csv: ''",
                "bsc_catalog: ''",
                "data_root: /data",
                "sector: 20",
                "camera: 3",
                "ccd: 1",
                "ref_ffi_min_earth_deg: 45.0",
                "n_jobs: 8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(str(legacy))
    assert cfg.sector == 20
    assert cfg.n_jobs == 8
    assert cfg.pipeline == [{"kind": "shared_mask"}]


def test_save_config_writes_slim(tmp_path: Path):
    cfg = SynDiffConfig(
        ffi_dir="/data/tess_ffi",
        output_dir="/ws/events/t",
        data_root="/data",
        max_ffis=10,
        pipeline=[{"kind": "shared_mask"}],
    )
    path = tmp_path / "out.yaml"
    save_config(cfg, str(path))
    text = path.read_text(encoding="utf-8")
    assert "max_ffis: 10" in text
    assert "manifest:" not in text
