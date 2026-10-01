import json

import pytest
import yaml

from syndiff_pipeline.forward_model.chain import config as C

from chain_fixtures import CHAIN_DIR, raw_config, write_config

FIELDS = ["F1", "F2", "S22", "C1", "C4"]


@pytest.mark.parametrize("f", FIELDS)
def test_example_configs_load(f):
    cfg = C.load_config(CHAIN_DIR / "configs" / f"{f}.yaml")
    assert cfg.field == f
    assert str(cfg.out_root) == f"/astro/armin/koji/syndiff/dev_runs/paper_dataset_20261001/{f}"
    assert cfg.stem.endswith(f"-s{cfg.scc.sector:04d}-{cfg.scc.camera}-{cfg.scc.ccd}")
    assert cfg.inputs.strap_mask is (f == "F2")
    assert cfg.fit.recipe == "paper1_dataset"
    assert cfg.stage_dir("fit") == cfg.out_root / "fit"


def test_f1_known_inputs():
    cfg = C.load_config(CHAIN_DIR / "configs/F1.yaml")
    assert cfg.inputs.source_scene.name == "F1_s24c2k2"
    assert cfg.inputs.init_params.name == "params_init.npz"
    assert cfg.unseen_stem == "tess2020121015919-s0024-2-2"
    assert str(cfg.reference.old_store).startswith(str(cfg.scc_root))     # {scc} expanded


def test_defaults_and_derived(tmp_path):
    cfg = C.config_from_dict(raw_config(tmp_path))
    assert cfg.kernels == C.KernelsCfg("k_sigma", 0.0)
    assert cfg.condor.request_cpus == C.DEFAULT_CPUS
    assert cfg.wcs_version == "T1_v1"
    assert cfg.scc_root == tmp_path / "data/s0024/c2/k2"
    assert cfg.code.forward_model_root.name  # default: the checkout this module lives in
    with pytest.raises(Exception):
        cfg.field = "x"            # frozen


def test_condor_override_merges(tmp_path):
    cfg = C.config_from_dict(raw_config(tmp_path, condor={"request_cpus": {"fit": 4}}))
    assert cfg.condor.request_cpus["fit"] == 4 and cfg.condor.request_cpus["mapping"] == 48


@pytest.mark.parametrize("mut,msg", [
    (lambda r: r.update(bogus=1), "unknown top-level"),
    (lambda r: r["inputs"].update(colour="x"), "unknown key"),
    (lambda r: r.update(stem="tess2020120182919-s0024-1-2"), "stem must look like"),
    (lambda r: r["scc"].update(camera=5), "scc.camera"),
    (lambda r: r.update(data_root="relative/path"), "absolute"),
    (lambda r: r.update(kernels={"source": "nope"}), "kernels.source"),
    (lambda r: r.update(kernels={"kernel_bright_q": -1}), "kernel_bright_q"),
    (lambda r: r["inputs"].update(strap_mask="yes"), "strap_mask"),
    (lambda r: r.update(mask={"bit2_radius": 0}), "bit2_radius"),
    (lambda r: r.update(condor={"request_cpus": {"fit": 0}}), "positive"),
    (lambda r: r.update(field="a/b"), "field"),
    (lambda r: r["inputs"].pop("colour_file"), "colour_file"),
    (lambda r: r.pop("scc"), "scc"),
    (lambda r: r.update(fit={"extra_flags": "--x"}), "extra_flags"),
])
def test_validation_errors(tmp_path, mut, msg):
    raw = raw_config(tmp_path)
    mut(raw)
    with pytest.raises(C.ConfigError, match=msg):
        C.config_from_dict(raw)


def test_hash_ignores_formatting_and_tracks_content(tmp_path):
    p = write_config(tmp_path)
    h1 = C.load_config(p).config_hash()
    # re-dump with reversed key order + a comment: same hash
    raw = yaml.safe_load(p.read_text())
    p.write_text("# comment\n" + yaml.safe_dump(dict(reversed(list(raw.items())))))
    assert C.load_config(p).config_hash() == h1
    raw["kernels"] = {"kernel_bright_q": 1.0}
    p.write_text(yaml.safe_dump(raw))
    assert C.load_config(p).config_hash() != h1


def test_need_and_stage_dir(tmp_path):
    cfg = C.config_from_dict(raw_config(tmp_path))
    with pytest.raises(C.ConfigError, match="inputs.source_scene"):
        cfg.need("inputs.source_scene")
    assert cfg.need("inputs.colour_file") == tmp_path / "colour.csv"
    with pytest.raises(C.ConfigError):
        cfg.stage_dir("nonsense")
    assert [cfg.stage_dir(s).name for s in C.STAGES] == list(C.STAGES)


def test_ffi_path_resolution(tmp_path):
    cfg = C.config_from_dict(raw_config(tmp_path))
    d = cfg.scc_root / "ffi"
    d.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        cfg.ffi_path()
    (d / f"{cfg.stem}-0180-s_ffic.fits.gz").touch()
    assert cfg.ffi_path().name.endswith(".fits.gz")
    (d / f"{cfg.stem}-0180-s_ffic.fits.fz").touch()      # .fz preferred
    assert cfg.ffi_path().name.endswith(".fits.fz")


def test_provenance_done(tmp_path):
    cfg = C.config_from_dict(raw_config(tmp_path, code={"sha": "abc123"}))
    small = tmp_path / "small.txt"
    small.write_text("hello")
    big = tmp_path / "big.bin"
    big.write_bytes(b"0" * 10)
    d = cfg.stage_dir("fit")
    assert not C.is_done(d)
    p = C.write_provenance(d, cfg, {"small": small, "dir": tmp_path, "missing": tmp_path / "nope", "n": {"value": 3}})
    prov = json.loads(p.read_text())
    assert prov["config_hash"] == cfg.config_hash() and prov["code"]["sha"] == "abc123" and prov["code"]["pinned"]
    assert prov["inputs"]["small"]["sha256"].startswith("2cf24dba")          # sha256("hello")
    assert prov["inputs"]["dir"]["kind"] == "dir" and prov["inputs"]["missing"]["exists"] is False
    assert prov["inputs"]["n"] == {"value": 3}
    C.mark_done(d)
    assert C.is_done(d)


def test_large_file_gets_no_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SHA_MAX_BYTES", 4)
    f = tmp_path / "f.bin"
    f.write_bytes(b"0123456789")
    rec = C._describe(f)
    assert "sha256" not in rec and rec["size"] == 10 and "mtime" in rec


def test_check_code_sha(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    (repo / "syndiff_pipeline").mkdir(parents=True)
    (repo / "syndiff_pipeline" / "a.py").write_text("x = 1\n")
    g = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(g + ["add", "."], check=True)
    subprocess.run(g + ["commit", "-qm", "c"], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    mk = lambda sha: C.config_from_dict(raw_config(tmp_path, code={"sha": sha, "forward_model_root": str(repo)}))
    mk(None).check_code_sha()                       # unpinned: no check
    mk(head).check_code_sha()
    with pytest.raises(C.ConfigError, match="is at"):
        mk("0" * 40).check_code_sha()
    (repo / "syndiff_pipeline" / "a.py").write_text("x = 2\n")
    with pytest.raises(C.ConfigError, match="local changes"):
        mk(head).check_code_sha()
