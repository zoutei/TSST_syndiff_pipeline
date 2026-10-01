import pytest

from syndiff_pipeline.forward_model.chain import condor as D
from syndiff_pipeline.forward_model.chain import config as C
from syndiff_pipeline.forward_model.chain import fit as F

from chain_fixtures import raw_config, write_config


def _cfg(tmp_path, **over):
    raw = raw_config(tmp_path, **over)
    return C.config_from_dict(raw)


def test_condor_quote():
    assert D._condor_quote("a") == "a"
    assert D._condor_quote("a b") == "'a b'"
    assert D._condor_quote("it's") == "'it''s'"
    assert D._condor_quote('x"y') == 'x""y'


def test_submit_text_fields(tmp_path):
    cfg = _cfg(tmp_path, code={"forward_model_root": str(tmp_path / "wt")})
    txt = D.submit_text(cfg, "fit", ["python", "-m", "x", "--a", "b c"], tmp_path / "logs", tag="fit")
    assert f"executable = {tmp_path}/wt/{D.WRAPPER_REL}" in txt
    assert 'arguments = "python -m x --a \'b c\'"' in txt
    assert f"PYTHONPATH={tmp_path}/wt" in txt and "JAX_PLATFORMS=cpu" in txt and "OPENBLAS_NUM_THREADS=1" in txt
    assert "request_cpus = 16" in txt and "request_memory = 16000" in txt
    assert "getenv = false" in txt and txt.rstrip().endswith("queue 1")
    assert "batch_name = T1_fit" in txt
    m = D.submit_text(cfg, "mapping", ["python"], tmp_path, tag="m")
    assert "request_cpus = 48" in m and "request_memory = 64000" in m
    assert "request_cpus = 8" in D.submit_text(cfg, "contrib", ["python"], tmp_path)
    assert "request_cpus = 3" in D.submit_text(cfg, "fit", ["python"], tmp_path, request_cpus=3)


def test_write_submit_and_stage_argv(tmp_path):
    p = write_config(tmp_path)
    cfg = C.load_config(p)
    argv = D.stage_argv(cfg, "gates", ["--force"])
    assert argv == ["python", "-m", "syndiff_pipeline.forward_model.chain", "gates", "--config", str(p.resolve()), "--force"]
    sub = D.write_submit(cfg, "mapping", argv)
    assert sub == cfg.out_root / "condor/mapping.sub" and "chain gates" in sub.read_text()
    no_path = C.config_from_dict(raw_config(tmp_path))
    with pytest.raises(ValueError):
        D.stage_argv(no_path, "fit")


def test_fit_command_boot_and_warm(tmp_path):
    cfg = _cfg(tmp_path, inputs={"colour_file": str(tmp_path / "c.csv"), "init_params": str(tmp_path / "p.npz")},
               fit={"recipe": "paper1_dataset", "extra_flags": ["--steps-per-stage", "1,1,1"]})
    scene, out = F.fit_dirs(cfg, "boot")
    assert (scene, out) == (cfg.stage_dir("scene_boot"), cfg.stage_dir("fit"))
    cmd = F.fit_command(cfg, scene, out, resume=True)
    assert cmd[1:5] == ["-m", "syndiff_pipeline.forward_model.recipe", "paper1_dataset", "--"]
    assert "--resume" in cmd and cmd[cmd.index("--init-params-file") + 1] == str(tmp_path / "p.npz")
    assert cmd[cmd.index("--colour-file") + 1] == str(tmp_path / "c.csv")
    assert cmd[-2:] == ["--steps-per-stage", "1,1,1"]                        # extra flags last (win in argparse)
    s2, o2 = F.fit_dirs(cfg, "refit", warm=True)
    assert (s2, o2) == (cfg.stage_dir("scene_final"), cfg.stage_dir("refit") / "warm")
    w = F.fit_command(cfg, s2, o2, warm=True)
    assert w[w.index("--init-params-file") + 1] == str(cfg.stage_dir("fit") / "params.npz")
    assert w[w.index("--init-state-file") + 1].endswith("state_stage3.npz")
    with pytest.raises(ValueError):
        F.fit_dirs(cfg, "boot", warm=True)


def test_fit_needs_init_and_scene(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(C.ConfigError, match="init_params"):
        F.fit_command(cfg, tmp_path, tmp_path)
    with pytest.raises(FileNotFoundError, match="scene_boot"):
        F.run_fit(cfg, "boot")


def test_fit_done_is_skipped(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    C.mark_done(cfg.stage_dir("fit"))
    monkeypatch.setattr(F.subprocess, "run", lambda *a, **k: pytest.fail("should not run"))
    assert F.run_fit(cfg, "boot") == cfg.stage_dir("fit")


def test_fit_runs_subprocess_with_env(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, inputs={"colour_file": str(tmp_path / "c.csv"), "init_params": str(tmp_path / "p.npz")})
    (tmp_path / "p.npz").write_bytes(b"x")
    C.mark_done(cfg.stage_dir("scene_boot"))
    (cfg.stage_dir("scene_boot") / "scene_bundle.npz").write_bytes(b"x")
    seen = {}

    def fake_run(cmd, check, env, cwd):
        seen.update(cmd=cmd, env=env, cwd=cwd)
        (cfg.stage_dir("fit") / "DONE").write_text("x")

    monkeypatch.setattr(F.subprocess, "run", fake_run)
    F.run_fit(cfg, "boot")
    assert seen["env"]["PYTHONPATH"] == str(cfg.code.forward_model_root) and seen["env"]["JAX_PLATFORMS"] == "cpu"
    assert "--resume" not in seen["cmd"]
    assert (cfg.stage_dir("fit") / "provenance.json").exists()
    # resume when progress.json exists and DONE is forced away
    (cfg.stage_dir("fit") / "progress.json").write_text("{}")
    F.run_fit(cfg, "boot", force=True)
    assert "--resume" not in seen["cmd"]                                   # force = fresh
    (cfg.stage_dir("fit") / "DONE").unlink()
    F.run_fit(cfg, "boot")
    assert "--resume" in seen["cmd"]


def test_prior_guard_accepts_the_dataset_recipe(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.fit.recipe == "paper1_dataset"
    assert F.check_prior_flags(cfg) == {"lambda-fine-nbr": 0.0, "lambda-local-poly": 3.0e8, "local-poly-window": 7.0}


def test_prior_guard_refuses_extra_flag_override(tmp_path):
    cfg = _cfg(tmp_path, fit={"recipe": "paper1_dataset", "extra_flags": ["--lambda-fine-nbr", "1e8"]})
    with pytest.raises(ValueError, match="overrides"):
        F.check_prior_flags(cfg)


def test_prior_guard_refuses_missing_or_wrong_recipe_values(tmp_path, monkeypatch):
    import syndiff_pipeline.forward_model.recipe as RC
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(RC, "recipe_argv", lambda name: ["--lambda-lap", "0.01", "--lambda-local-poly", "3e8"])
    with pytest.raises(ValueError, match="explicitly"):
        F.check_prior_flags(cfg)
    monkeypatch.setattr(RC, "recipe_argv", lambda name: ["--lambda-fine-nbr", "1e8", "--lambda-local-poly", "3e8",
                                                         "--local-poly-window", "7"])
    with pytest.raises(ValueError, match="required"):
        F.check_prior_flags(cfg)


def test_run_fit_refuses_before_launch(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, fit={"recipe": "paper1_dataset", "extra_flags": ["--local-poly-window=5"]})
    monkeypatch.setattr(F, "submit", lambda sub: pytest.fail("must not submit"))
    with pytest.raises(ValueError, match="overrides"):
        F.run_fit(cfg, "boot", condor=True)
