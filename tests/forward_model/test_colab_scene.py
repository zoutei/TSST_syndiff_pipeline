"""Scene packaging must preserve scientific inputs and reject ambiguous DAGs."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from syndiff_pipeline.forward_model.colab_scene import (
    build_scene_archive, ordered_jobs, verify_plan, contract_spec,
)
from syndiff_pipeline.forward_model.colab_submit import build_provision_worker_source


@pytest.fixture
def campaign(tmp_path):
    source = tmp_path / "source"
    package = source / "syndiff_pipeline/forward_model"
    (package / "recipes").mkdir(parents=True)
    (package / "scene_fit.py").write_text("# exact experimental fitter\n")
    (package / "recipes/test.yaml").write_text("scene_fit:\n  freeze-wcs: true\n  reject-every: 0\n  steps-per-stage: 0,0,8000\n")
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "scene_bundle.npz").write_bytes(b"scene pixels")
    (scene / "fit_bundle.npz").write_bytes(b"source model bundle")
    (scene / "scene_meta.json").write_text(json.dumps({"source_bundle": str(scene / "fit_bundle.npz")}))
    init = tmp_path / "fit"
    init.mkdir()
    (init / "params.npz").write_bytes(b"warm params")
    (init / "fit_meta.json").write_text('{"chroma_g8_extras":"full-law"}')
    (init / "flux_solved.npz").write_bytes(b"flux state")
    spec = dict(source_root=str(source), max_seconds=600, jobs=[dict(name="frozen", scene_dir=str(scene),
               init_params=str(init / "params.npz"), recipe="test", argv=["--steps-per-stage", "0,0,160"]),
               dict(name="polish", scene_dir=str(scene), recipe="test", init_from_job="frozen", depends_on=["frozen"])])
    return spec


def test_archive_is_portable_and_carries_warm_start_metadata(campaign, tmp_path):
    archive = tmp_path / "campaign.zip"
    plan = build_scene_archive(campaign, archive)
    root = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
        names = z.namelist()
    assert verify_plan(root) == plan
    first = plan["jobs"][0]
    assert first["argv"][-2:] == ["--steps-per-stage", "0,0,160"]
    assert not Path(first["scene_dir"]).is_absolute()
    meta = json.loads((root / first["scene_dir"] / "scene_meta.json").read_text())
    assert (root / meta["source_bundle"]).read_bytes() == b"source model bundle"
    assert (root / first["init_params"]).with_name("fit_meta.json").is_file()
    assert (root / first["init_params"]).with_name("flux_solved.npz").is_file()
    assert sum(n.endswith("scene_bundle.npz") for n in names) == 1
    assert (root / "syndiff_pipeline/forward_model/recipes/test.yaml").is_file()
    assert (root / "syndiff_pipeline/forward_model/scene_fit.py").read_text() == "# exact experimental fitter\n"


def test_preflight_rejects_changed_pixels(campaign, tmp_path):
    archive = tmp_path / "campaign.zip"
    plan = build_scene_archive(campaign, archive)
    root = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    (root / plan["jobs"][0]["scene_dir"] / "scene_bundle.npz").write_bytes(b"wrong image")
    with pytest.raises(ValueError, match="verification failed"):
        verify_plan(root)


@pytest.mark.parametrize("jobs", [
    [{"name": "a"}, {"name": "a"}],
    [{"name": "../a"}],
    [{"name": "a", "depends_on": ["missing"]}],
    [{"name": "a", "depends_on": ["b"]}, {"name": "b", "depends_on": ["a"]}],
    [{"name": "a"}, {"name": "b", "init_from_job": "a"}],
])
def test_invalid_campaigns_fail_before_allocation(jobs):
    with pytest.raises(ValueError):
        ordered_jobs(jobs)


def test_dependency_order():
    jobs = [{"name": "child", "depends_on": ["parent"], "init_from_job": "parent"}, {"name": "parent"}]
    assert [j["name"] for j in ordered_jobs(jobs)] == ["parent", "child"]


def test_reserved_path_override_is_rejected(campaign, tmp_path):
    campaign["jobs"][0]["argv"] += ["--scene-dir=/other/image"]
    with pytest.raises(ValueError, match="job fields"):
        build_scene_archive(campaign, tmp_path / "bad.zip")


def test_scene_preflight_keeps_temporal_preflight_available():
    source = build_provision_worker_source(dict(run_id="test", job_dir="/content/jobs/test",
        bundle_sha256="abc", bundle_file_id="drive-id", scene_campaign=True))
    compile(source, "worker.py", "exec")
    assert 'SPEC.get("scene_campaign")' in source
    assert '"syndiff_pipeline.forward_model.colab_scene", "preflight"' in source
    assert '"syndiff_pipeline.forward_model.gpu_preflight"' in source


def load_controller():
    path = Path(__file__).parents[2] / "scripts/colab_job.py"
    spec = importlib.util.spec_from_file_location("scene_test_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_download_cannot_report_verified_final_sync(tmp_path, monkeypatch):
    c = load_controller()
    state = {"session": "owned", "kind": "scene"}
    monkeypatch.setattr(c, "job_state", lambda run: (tmp_path, state))
    monkeypatch.setattr(c, "_fetch_remote_manifest", lambda *a: {"frozen/params.npz": {"sha256": "x"}})
    monkeypatch.setattr(c, "_download_artifact", lambda *a, **kw: False)
    ok, changed = c.sync_artifacts_local("test", final=True)
    assert not ok and changed == []


def test_nested_scene_parameters_are_detected(tmp_path):
    c = load_controller()
    art = tmp_path / "artifacts"
    (art / "frozen").mkdir(parents=True)
    (art / "scene_campaign.json").write_text("{}")
    (art / "frozen/params.npz").write_bytes(b"model")
    assert c._has_training_outputs(tmp_path)
    assert c._newest_parameter_artifact(tmp_path) == "frozen/params.npz"


def test_contract_import_preserves_ablation_and_schedule(tmp_path):
    path = tmp_path / "launch_contract.json"
    contract = dict(argv=["--freeze-wcs", "--scene-dir", "/scene", "--out-dir", "/runs/F1/drop_width/fold0",
                         "--colour-file=/colour.csv", "--init-params-file", "/init/params.npz", "--steps-per-stage", "3000,20000,20000"],
                    dropped_slots=["blur", "eps", "dil_r"], hashes={"/adapter/ccd_colour_shift.py": "hash"})
    path.write_text(json.dumps(contract))
    job = contract_spec(tmp_path, [path], "0,0,160")["jobs"][0]
    assert job["name"] == "F1_drop_width_fold0"
    assert job["drop_slots"] == ["blur", "eps", "dil_r"]
    assert job["colour_file"] == "/colour.csv"
    assert job["argv"][-2:] == ["--steps-per-stage", "0,0,160"]
    assert "--scene-dir" not in job["argv"]


def test_cached_verified_file_is_not_a_sync_failure(tmp_path, monkeypatch):
    c = load_controller()
    (tmp_path / "artifacts/frozen").mkdir(parents=True)
    (tmp_path / "artifacts/frozen/params.npz").write_bytes(b"params")
    (tmp_path / "sync_manifest.json").write_text('{"frozen/params.npz":"same-hash"}')
    monkeypatch.setattr(c, "job_state", lambda run: (tmp_path, {"session": "owned", "kind": "scene"}))
    monkeypatch.setattr(c, "_fetch_remote_manifest", lambda *a: {"frozen/params.npz": {"sha256": "same-hash"}})
    monkeypatch.setattr(c, "_download_artifact", lambda *a, **kw: pytest.fail("cached file downloaded again"))
    assert c.sync_artifacts_local("test", final=True) == (True, [])


def test_gpu_worker_requests_cuda_not_rocm(campaign, tmp_path, monkeypatch):
    from syndiff_pipeline.forward_model import colab_scene as m
    archive, root, results = tmp_path / "campaign.zip", tmp_path / "package", tmp_path / "results"
    build_scene_archive(campaign, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    seen = []

    class Child:
        returncode = 0
        def __init__(self, cmd, **kw):
            seen.append(kw["env"])
            out = results / cmd[cmd.index("--name") + 1]
            out.mkdir(parents=True)
            (out / "DONE").write_text("complete")
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def poll(self): return 0

    monkeypatch.setattr(m.subprocess, "Popen", Child)
    m.run_campaign(root, results, "gpu")
    assert all(env["JAX_PLATFORMS"] == "cuda" for env in seen)
    assert all(env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" for env in seen)
    assert (results / "SCENE_DONE").is_file()
    with pytest.raises(ValueError, match="already exist"):
        m.run_campaign(root, results, "gpu")


def test_local_materialization_preserves_verified_bytes(tmp_path):
    from syndiff_pipeline.forward_model.colab_scene import materialize_results, digest
    directory = tmp_path / "run"
    artifacts = directory / "artifacts"
    fit = artifacts / "frozen"
    fit.mkdir(parents=True)
    (fit / "DONE").write_text("done")
    (fit / "params.npz").write_bytes(b"verified model")
    meta = {"scene_dir": "/content/syndiff/scene-runtime-xxx", "out_dir": "/content/jobs/run/frozen",
            "colour_file": "/content/syndiff/inputs/colour.csv"}
    (fit / "fit_meta.json").write_text(json.dumps(meta))
    (artifacts / "scene_campaign.json").write_text(json.dumps(dict(source_commit="pinned", jobs=[dict(name="frozen", scene_dir="scenes/hash")],
       origins={"scenes/hash/scene_bundle.npz": "/astro/original/scene/scene_bundle.npz", "inputs/colour.csv": "/astro/original/colour.csv"})))
    old_hash = digest(fit / "fit_meta.json")
    results = materialize_results(directory)
    restored = json.loads((results / "frozen/fit_meta.json").read_text())
    assert restored["scene_dir"] == "/astro/original/scene"
    assert restored["colour_file"] == "/astro/original/colour.csv"
    assert (results / "frozen/params.npz").is_symlink()
    assert digest(fit / "fit_meta.json") == old_hash


def test_direct_publication_needs_no_drive_credentials(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from syndiff_pipeline.forward_model import colab_scene as m
    c = load_controller()
    source = tmp_path / "spec.json"
    source.write_text("{}")
    archive = tmp_path / "scene.zip"
    archive.write_bytes(b"package")
    monkeypatch.setattr(m, "build_scene_archive", lambda *a: {"source_commit": "pin", "jobs": [{"name": "fit"}]})
    monkeypatch.setattr(m, "validate_archive", lambda *a: None)
    monkeypatch.setattr(c, "drive_auth_paths", lambda: pytest.fail("direct upload touched Drive auth"))
    monkeypatch.setattr(c, "MANIFEST", tmp_path / "manifest.json")
    assert c.publish_scene(SimpleNamespace(spec=source, archive=archive, transport="upload")) == 0
    entry = json.loads(c.MANIFEST.read_text())["bundles"]["scene.zip"]
    assert entry["transport"] == "upload" and entry["file_id"] is None


def test_submit_scene_defaults_to_t4_and_uses_packaged_science():
    c = load_controller()
    args = c.parser().parse_args(["submit-scene", "--archive", "/astro/test.zip"])
    assert args.gpu == "T4" and args.mode == "full" and args.steps_per_stage is None
    assert args.min_ram_gb == 10 and not args.no_supervisor
    with pytest.raises(SystemExit):
        c.parser().parse_args(["submit-scene", "--archive", "/astro/test.zip", "--steps-per-stage", "0,0,20"])
