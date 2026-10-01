import sys
import types

import pytest

from syndiff_pipeline.forward_model.chain import cli

from chain_fixtures import write_config


def test_parser_stage_choices():
    ap = cli.build_parser()
    assert ap.parse_args(["kernels", "--config", "x"]).stage == "kernels"
    assert set(cli.DISPATCH) <= set(cli.ALL_STAGES) and set(cli.OWN_STAGES) <= set(cli.ALL_STAGES)
    with pytest.raises(SystemExit):
        ap.parse_args(["nonsense", "--config", "x"])
    with pytest.raises(SystemExit):
        ap.parse_args(["fit"])                                            # --config required


def test_pair_parser():
    assert cli._parse_pair("a:b") == ("a", "b", "a vs b")
    assert cli._parse_pair("a:b:some: label") == ("a", "b", "some: label")
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_pair("a")


def test_bad_config_returns_2(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text("field: x\n")
    assert cli.main(["status", "--config", str(p)]) == 2
    assert "config error" in capsys.readouterr().err
    assert cli.main(["status", "--config", str(tmp_path / "missing.yaml")]) == 2


def test_status(tmp_path, capsys):
    p = write_config(tmp_path)
    assert cli.main(["status", "--config", str(p)]) == 0
    out = capsys.readouterr().out
    assert "T1" in out and "scene_boot" in out


def test_own_stage_rejects_extra_args(tmp_path):
    p = write_config(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["fit", "--config", str(p), "--bogus"])


def test_missing_external_module_is_clear(tmp_path, capsys, monkeypatch):
    p = write_config(tmp_path)
    monkeypatch.delitem(sys.modules, "syndiff_pipeline.forward_model.chain.hotpants_ref", raising=False)
    monkeypatch.setattr(cli.importlib, "import_module",
                        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)))
    assert cli.main(["hotpants", "--config", str(p)]) == 1
    assert "hotpants_ref" in capsys.readouterr().err


@pytest.mark.parametrize("stage,mod", [(s, m) for s, m in cli.DISPATCH.items()])
def test_dispatch_to_module_run(tmp_path, monkeypatch, stage, mod):
    p = write_config(tmp_path)
    calls = []
    fake = types.ModuleType("fake")
    fake.run = lambda cfg, force=False, condor=False, args=(): calls.append((cfg.field, force, condor, list(args)))
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake if name.endswith(mod) else pytest.fail(name))
    assert cli.main([stage, "--config", str(p), "--force", "3", "7"]) == 0
    assert calls == [("T1", True, False, ["3", "7"])]


def test_dispatch_only_passes_accepted_kwargs(tmp_path, monkeypatch):
    p = write_config(tmp_path)
    calls = []
    fake = types.ModuleType("fake")
    fake.run = lambda cfg: calls.append(cfg.field)
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake)
    assert cli.main(["score", "--config", str(p), "--force"]) == 0 and calls == ["T1"]


def test_dispatch_no_run_attr(tmp_path, monkeypatch, capsys):
    p = write_config(tmp_path)
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: types.ModuleType("m"))
    assert cli.main(["final", "--config", str(p)]) == 1
    assert "run(cfg" in capsys.readouterr().err


def test_dispatch_condor_fallback_writes_submit(tmp_path, monkeypatch):
    p = write_config(tmp_path)
    fake = types.ModuleType("fake")
    fake.run = lambda cfg: pytest.fail("must not run locally")
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: fake)
    submitted = []
    import syndiff_pipeline.forward_model.chain.condor as D
    monkeypatch.setattr(D, "submit", lambda sub: submitted.append(sub) or "1 job(s) submitted")
    assert cli.main(["contrib", "--config", str(p), "--condor", "--force", "2", "5"]) == 0
    sub = submitted[0]
    txt = sub.read_text()
    assert "chain contrib" in txt and "2 5" in txt and "--force" in txt and "request_cpus = 8" in txt


def test_own_stage_dispatch(tmp_path, monkeypatch):
    p = write_config(tmp_path)
    got = {}
    import syndiff_pipeline.forward_model.chain.scene as S
    import syndiff_pipeline.forward_model.chain.mapping as M
    import syndiff_pipeline.forward_model.chain.fit as F
    monkeypatch.setattr(S, "run_scene", lambda cfg, which, hp_d=None, force=False: got.update(scene=(which, hp_d, force)))
    monkeypatch.setattr(M, "run_mapping", lambda cfg, mode, force=False, condor=False: got.update(mapping=(mode, force, condor)))
    monkeypatch.setattr(F, "run_fit", lambda cfg, which, warm=False, force=False, condor=False: got.update(fit=(which, warm, force, condor)))
    assert cli.main(["scene_final", "--config", str(p), "--hp-d", "/x/y.fits", "--force"]) == 0
    assert cli.main(["mapping", "--config", str(p), "--header-wcs", "--condor"]) == 0
    assert cli.main(["refit", "--config", str(p), "--warm"]) == 0
    assert got == {"scene": ("final", "/x/y.fits", True), "mapping": ("header", False, True), "fit": ("refit", True, False, False)}
