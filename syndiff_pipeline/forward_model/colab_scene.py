"""Portable scene-fit campaigns for the existing Colab controller.

Packaging preserves the input files; each fit runs in a fresh JAX process.
The archive holds the actual producing fitter, resolved recipes, and hashed
inputs. No shared-storage path is used by the GPU worker.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile

SCHEMA = 1
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MODULE = "syndiff_pipeline.forward_model.colab_scene"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ordered_jobs(jobs):
    """Validate names and dependencies, then stable topological order."""
    names = [j["name"] for j in jobs]
    if not names or len(set(names)) != len(names) or any(not NAME.fullmatch(n) for n in names):
        raise ValueError("campaign needs unique, path-safe job names")
    remaining = list(jobs)
    result, done = [], set()
    while remaining:
        ready = next((j for j in remaining if set(j.get("depends_on", [])) <= done), None)
        if ready is None:
            raise ValueError("unknown dependency or cycle in scene campaign")
        parent = ready.get("init_from_job")
        if parent and parent not in ready.get("depends_on", []):
            raise ValueError("init_from_job must be listed in depends_on")
        result.append(ready)
        done.add(ready["name"])
        remaining.remove(ready)
    return result


def recipe_flags(source, name):
    import yaml
    if not NAME.fullmatch(name):
        raise ValueError("invalid recipe name")
    data = yaml.safe_load((source / "syndiff_pipeline/forward_model/recipes" / (name + ".yaml")).read_text())
    result = []
    for key, value in data["scene_fit"].items():
        if value is True:
            result.append("--" + key)
        elif value is not None and value is not False:
            result.extend(["--" + key, str(value)])
    return result


def contract_spec(source, contracts, steps=None):
    """Import native scene-fit launch contracts (including W3 drop-group runs)."""
    jobs = []
    for path in contracts:
        path = Path(path).resolve()
        contract = json.loads(path.read_text())
        flags, paths = [], {}
        argv, i = contract["argv"], 0
        managed = {"--scene-dir": "scene_dir", "--out-dir": "out_dir", "--colour-file": "colour_file",
                   "--init-params-file": "init_params", "--init-from": "init_from"}
        while i < len(argv):
            token = argv[i]
            key, sep, value = token.partition("=")
            if key in managed:
                if not sep:
                    i += 1
                    value = argv[i]
                paths[managed[key]] = value
            else:
                flags.append(token)
            i += 1
        if "init_from" in paths:
            paths["init_params"] = str(Path(paths.pop("init_from")) / "params.npz")
        name = "_".join(Path(paths.pop("out_dir")).parts[-3:])
        job = dict(name=name, argv=flags, **paths, provenance=dict(launch_contract=str(path), launch_contract_sha256=digest(path)))
        adapters = [p for p in contract.get("hashes", {}) if Path(p).name == "ccd_colour_shift.py"]
        if adapters:
            job["adapter"] = adapters[0]
        if contract.get("dropped_slots"):
            job["drop_slots"] = contract["dropped_slots"]
        if steps:
            job["argv"] += ["--steps-per-stage", steps]
        jobs.append(job)
    return dict(schema=SCHEMA, source_root=str(Path(source).resolve()), jobs=ordered_jobs(jobs))


def colour_spec(fits, source, arms, fields, folds, steps=None):
    """Import the current colour.dag arm table, including its checkpoint edges."""
    import yaml
    fits, source = Path(fits).resolve(), Path(source).resolve()
    table = json.loads((fits / "code/arms.json").read_text())
    base = fits.parents[1] / "paper1_final_fits_20261007"
    crossfit = fits.parents[1] / "epsf_model_closure_20261004/crossfit"
    selected = list(arms)
    for arm in selected:
        parent = table["arms"][arm].get("from")
        if parent and parent not in selected:
            raise ValueError(f"{arm} requires selecting {parent} too")
    jobs = []
    for arm in selected:
        cfg = table["arms"][arm]
        recipe = next((r for r in cfg["recipes"] if (source / "syndiff_pipeline/forward_model/recipes" / (r + ".yaml")).is_file()), None)
        if recipe is None:
            raise ValueError(f"no recipe for {arm} in {source}")
        for field in fields:
            colour = yaml.safe_load((crossfit / (field + "_full_generation/config.yaml")).read_text())["inputs"]["colour_file"]
            for fold in folds:
                name = f"{arm}_{field}_{fold}"
                job = dict(name=name, recipe=recipe, scene_dir=str(base / field / "nbr" / f"fold{fold}_T17"),
                           colour_file=colour, adapter=str(crossfit / "adapter/ccd_colour_shift.py"),
                           argv=list(cfg["extra"]), provenance=dict(arm=arm, field=field, fold=fold, arm_table=str(fits / "code/arms.json")))
                if cfg["mode"] == "from_arm":
                    job["init_from_job"] = f"{cfg['from']}_{field}_{fold}"
                    job["depends_on"] = [job["init_from_job"]]
                    schedule = table[cfg["steps_key"]]
                else:
                    warm = cfg["mode"] == "warm"
                    job["init_params"] = str(base / field / ("fits" if warm else "init") / f"fold{fold}" / ("params.npz" if warm else "params_init.npz"))
                    schedule = table["warm_steps"] if warm else None
                if steps or schedule:
                    job["argv"] += ["--steps-per-stage", steps or schedule]
                jobs.append(job)
    return dict(schema=SCHEMA, source_root=str(source), jobs=ordered_jobs(jobs))


def build_scene_archive(spec, output):
    if spec.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("unsupported scene spec schema")
    source = Path(spec["source_root"]).resolve()
    package = source / "syndiff_pipeline/forward_model"
    if not (package / "scene_fit.py").is_file():
        raise ValueError(f"no scene fitter in {source}")
    jobs = ordered_jobs(spec["jobs"])
    files, contents, hashes, origin = {}, {}, {}, {}

    def add(path, target=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        sha = digest(path)
        target = target or f"inputs/{sha[:20]}/{path.name}"
        if target in hashes and hashes[target] != sha:
            raise ValueError(f"archive member collision: {target}")
        files[target], hashes[target], origin[target] = path, sha, str(path)
        return target

    def add_json(value, target):
        raw = (json.dumps(value, indent=2) + "\n").encode()
        contents[target], hashes[target] = raw, hashlib.sha256(raw).hexdigest()
        return target

    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts:
            add(path, "syndiff_pipeline/forward_model/" + path.relative_to(package).as_posix())
    # Infrastructure from this controller; scientific model from source_root.
    here = Path(__file__).parent
    for name in ("colab_scene.py", "gpu_job_runner.py", "colab_emergency_upload.py"):
        target = "syndiff_pipeline/forward_model/" + name
        hashes.pop(target, None)
        add(here / name, target)
    contents["syndiff_pipeline/__init__.py"] = b""
    packed = []
    for job in jobs:
        item = {k: v for k, v in job.items() if k not in ("scene_dir", "init_params", "colour_file", "adapter", "argv")}
        scene = Path(job["scene_dir"]).resolve()
        sid = digest(scene / "scene_bundle.npz")[:20]
        prefix = "scenes/" + sid
        add(scene / "scene_bundle.npz", prefix + "/scene_bundle.npz")
        meta = json.loads((scene / "scene_meta.json").read_text())
        bundle = Path(meta["source_bundle"])
        if not bundle.is_absolute():
            bundle = scene / bundle
        btarget = add(bundle)
        # Rebased at execution, retaining original metadata in provenance.
        meta["source_bundle"] = btarget
        add_json(meta, prefix + "/scene_meta.json")
        item["scene_dir"] = prefix
        item["argv"] = recipe_flags(source, job["recipe"]) if job.get("recipe") else []
        extra = list(job.get("argv", []))
        reserved = {"--scene-dir", "--out-dir", "--init-params-file", "--init-from", "--colour-file", "--resume"}
        if any(x.split("=")[0] in reserved for x in extra):
            raise ValueError("set scene/checkpoint/colour in job fields, not argv")
        item["argv"] += extra
        if job.get("init_params"):
            init = Path(job["init_params"]).resolve()
            iprefix = "inits/" + hashlib.sha256(str(init.parent).encode()).hexdigest()[:20]
            item["init_params"] = add(init, iprefix + "/" + init.name)
            for name in ("fit_meta.json", "flux_solved.npz", "bright_width_q.npz"):
                if (init.parent / name).is_file():
                    add(init.parent / name, iprefix + "/" + name)
        for key in ("colour_file", "adapter"):
            if job.get(key):
                item[key] = add(job[key])
        packed.append(item)
    try:
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    except subprocess.CalledProcessError:
        revision = (source / "PIN_SHA").read_text().strip() if (source / "PIN_SHA").is_file() else None
        dirty = "non-Git source snapshot (content hashes recorded)"
    plan = dict(schema=SCHEMA, kind="scene", source_root=str(source), source_commit=revision,
                source_dirty=dirty, jobs=packed, hashes=hashes, origins=origin,
                max_seconds=int(spec.get("max_seconds", 7200)))
    if plan["max_seconds"] <= 0:
        raise ValueError("max_seconds must be positive")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f".tmp.{os.getpid()}")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for target, path in files.items():
                z.write(path, target)
            for target, raw in contents.items():
                z.writestr(target, raw)
            z.writestr("scene_campaign.json", json.dumps(plan, indent=2))
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return plan


def verify_plan(root):
    root = Path(root).resolve()
    plan = json.loads((root / "scene_campaign.json").read_text())
    if plan.get("schema") != SCHEMA or plan.get("kind") != "scene":
        raise ValueError("unsupported scene campaign")
    ordered_jobs(plan["jobs"])
    for name, expected in plan["hashes"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError(f"scene campaign input verification failed: {name}")
    return plan


def validate_archive(archive):
    """Check the package and actual producing parser before allocating a VM."""
    with tempfile.TemporaryDirectory(prefix="scene-validate-", dir=Path(archive).resolve().parent,
                                     ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise ValueError("unsafe scene archive member")
            z.extractall(root)
        verify_plan(root)
        script = '''import json
from pathlib import Path
from syndiff_pipeline.forward_model import scene_fit as SF
plan=json.loads(Path('scene_campaign.json').read_text())
for job in plan['jobs']:
    SF.build_parser().parse_args(job['argv']+['--scene-dir',job['scene_dir'],'--out-dir','/tmp/parse-only'])
print('scene parsers validated:',len(plan['jobs']))
'''
        subprocess.run([sys.executable, "-c", script], cwd=root,
                       env=dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS="cpu"),
                       timeout=120, check=True)


def materialize_results(directory):
    """Make locally usable scene fits without editing the verified artifact copies."""
    directory = Path(directory).resolve()
    artifacts = directory / "artifacts"
    plan = json.loads((artifacts / "scene_campaign.json").read_text())
    results = directory / "results"
    remote_jobs = "/content/jobs/" + directory.name

    def rebase(value):
        if isinstance(value, list):
            return [rebase(v) for v in value]
        if isinstance(value, dict):
            return {k: rebase(v) for k, v in value.items()}
        if isinstance(value, str):
            if value.startswith("/content/syndiff/"):
                return plan["origins"].get(value[len("/content/syndiff/"):], value)
            if value.startswith(remote_jobs + "/"):
                return str(results / value[len(remote_jobs) + 1:])
        return value

    for job in plan["jobs"]:
        src, dst = artifacts / job["name"], results / job["name"]
        if not (src / "DONE").is_file():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for path in src.iterdir():
            target = dst / path.name
            if path.name == "fit_meta.json":
                meta = rebase(json.loads(path.read_text()))
                meta["scene_dir"] = str(Path(plan["origins"][job["scene_dir"] + "/scene_bundle.npz"]).parent)
                meta["out_dir"] = str(dst)
                target.write_text(json.dumps(meta, indent=2))
            elif not target.exists():
                target.symlink_to(path)
        (dst / "local_provenance.json").write_text(json.dumps(dict(verified_artifacts=str(src), source_commit=plan["source_commit"], job=job,
                                                               note="fit_meta paths rebased; binary artifacts unchanged"), indent=2))
    return results


def run_one(root, job_dir, name, backend):
    # Import JAX before pandas/pyarrow in scene_fit.
    import jax
    import jax.numpy as jnp
    import numpy as np
    plan = verify_plan(root)
    job = next(j for j in plan["jobs"] if j["name"] == name)
    if jax.default_backend() != backend:
        raise RuntimeError(f"expected {backend}, got {jax.default_backend()}")
    print(f"jax backend: {jax.default_backend()} devices={jax.devices()}", flush=True)
    root, job_dir = Path(root).resolve(), Path(job_dir).resolve()
    original_scene = root / job["scene_dir"]
    runtime_scene = tempfile.TemporaryDirectory(prefix="scene-runtime-", dir=root)
    scene = Path(runtime_scene.name)
    (scene / "scene_bundle.npz").symlink_to(original_scene / "scene_bundle.npz")
    meta = json.loads((original_scene / "scene_meta.json").read_text())
    meta["source_bundle"] = str(root / meta["source_bundle"])
    (scene / "scene_meta.json").write_text(json.dumps(meta))
    try:
        adapter = None
        if job.get("adapter"):
            spec = importlib.util.spec_from_file_location("scene_campaign_adapter", root / job["adapter"])
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
            adapter.install(initialize=True, freeze=True)
        from . import scene_fit as SF, loss as L
        mask = None
        if job.get("drop_slots"):
            parser = SF.build_parser()
            probe = parser.parse_args(job["argv"] + ["--scene-dir", str(scene), "--out-dir", str(job_dir / name)])
            SF.resolve_g8_defaults(probe, job_dir / name)
            extras = SF.g8_extras_tuple(probe.chroma_g8_extras, probe.chroma_g8_blur_order, probe.chroma_g8_no_dil)
            names = ["s0", "s1", "s2", "k0", "k1", "blur", "eps", "trefoil"] + list(extras)
            mask = np.ones(len(names), np.float32)
            for slot in job["drop_slots"]:
                mask[names.index(slot)] = 0
            orig, jmask = L._chroma_g8_slot_terms, jnp.asarray(mask)
            L._chroma_g8_slot_terms = lambda p, c, s: orig(dict(p, chroma_g8=p["chroma_g8"] * jmask), c, s)
        out = job_dir / name
        out.mkdir(parents=True, exist_ok=True)
        argv = list(job["argv"]) + ["--scene-dir", str(scene), "--out-dir", str(out)]
        init = job.get("init_params")
        if job.get("init_from_job"):
            parent = job_dir / job["init_from_job"]
            if not (parent / "DONE").is_file():
                raise RuntimeError("dependency checkpoint is not complete")
            argv += ["--init-params-file", str(parent / "params.npz")]
        elif init:
            argv += ["--init-params-file", str(root / init)]
        if job.get("colour_file"):
            argv += ["--colour-file", str(root / job["colour_file"])]
        args = SF.build_parser().parse_args(argv)
        if adapter:
            args.chroma_ccd_shift_model = dict(adapter.provenance(), coefficients_frozen=True)
        (out / "launch_contract.json").write_text(json.dumps(dict(job=job, argv=argv, source_commit=plan["source_commit"], input_hashes=plan["hashes"], backend=backend), indent=2))
        started = time.monotonic()
        SF.run(args)
        if mask is not None:
            final = np.load(out / "params.npz")["chroma_g8"]
            if not np.all(final[mask == 0] == 0):
                raise RuntimeError("dropped coefficients moved; use a zero initialization")
        (out / "execution.json").write_text(json.dumps(dict(backend=jax.default_backend(), devices=[str(d) for d in jax.devices()], device_kind=jax.devices()[0].device_kind, elapsed_seconds=time.monotonic() - started, jax=jax.__version__, numpy=np.__version__), indent=2))
    finally:
        runtime_scene.cleanup()


def run_campaign(root, job_dir, backend):
    import signal
    root, job_dir = Path(root).resolve(), Path(job_dir).resolve()
    plan = verify_plan(root)
    job_dir.mkdir(parents=True, exist_ok=True)
    if any((job_dir / job["name"]).exists() for job in plan["jobs"]):
        raise ValueError("scene outputs already exist; use a new run directory")
    (job_dir / "scene_campaign.json").write_text(json.dumps(plan, indent=2))
    deadline = time.monotonic() + plan["max_seconds"]
    for job in ordered_jobs(plan["jobs"]):
        print(f"scene campaign job {job['name']} starting", flush=True)
        cmd = [sys.executable, "-m", MODULE, "one", "--root", str(root), "--job-dir", str(job_dir), "--name", job["name"], "--backend", backend]
        env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cuda" if backend == "gpu" else "cpu",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with subprocess.Popen(cmd, cwd=root, env=env) as proc:
            while proc.poll() is None:
                if (job_dir / "STOP").exists() or time.monotonic() >= deadline:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                    raise RuntimeError("scene campaign stopped or wall-time budget exhausted")
                time.sleep(1)
            if proc.returncode:
                raise RuntimeError(f"scene job {job['name']} failed rc={proc.returncode}")
        if not (job_dir / job["name"] / "DONE").is_file():
            raise RuntimeError(f"scene job {job['name']} missing completion marker")
    (job_dir / "SCENE_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack", help="build a scene archive without uploading")
    pack.add_argument("--spec", type=Path, required=True)
    pack.add_argument("--archive", type=Path, required=True)
    con = sub.add_parser("contract-spec", help="import existing scene-fit launch contracts, including W3 ablations")
    con.add_argument("--source-root", type=Path, required=True)
    con.add_argument("--contract", type=Path, action="append", required=True)
    con.add_argument("--steps-per-stage")
    con.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("colour-spec")
    c.add_argument("--fits", type=Path, required=True); c.add_argument("--source-root", type=Path, required=True)
    c.add_argument("--arms", required=True); c.add_argument("--fields", required=True); c.add_argument("--folds", required=True)
    c.add_argument("--steps-per-stage"); c.add_argument("--output", type=Path, required=True)
    for verb in ("run", "one", "preflight"):
        q = sub.add_parser(verb); q.add_argument("--root", type=Path, default=Path.cwd())
        if verb != "preflight":
            q.add_argument("--job-dir", type=Path, required=True); q.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
        if verb == "one":
            q.add_argument("--name", required=True)
    a = p.parse_args()
    if a.command == "pack":
        plan = build_scene_archive(json.loads(a.spec.read_text()), a.archive)
        validate_archive(a.archive)
        print(f"packaged {len(plan['jobs'])} jobs: {a.archive}")
    elif a.command == "contract-spec":
        data = contract_spec(a.source_root, a.contract, a.steps_per_stage)
        a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(data, indent=2))
    elif a.command == "colour-spec":
        data = colour_spec(a.fits, a.source_root, a.arms.split(","), a.fields.split(","), a.folds.split(","), a.steps_per_stage)
        a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(data, indent=2))
    elif a.command == "preflight":
        plan = verify_plan(a.root)
        print(f"scene preflight: {len(plan['jobs'])} jobs; all packaged inputs verified", flush=True)
    elif a.command == "one":
        run_one(a.root, a.job_dir, a.name, a.backend)
    else:
        run_campaign(a.root, a.job_dir, a.backend)


if __name__ == "__main__":
    main()
