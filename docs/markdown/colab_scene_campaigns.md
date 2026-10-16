# Scene-fit campaigns on Colab

The Colab controller supports the single-FFI `scene_fit` experiments used by
`colour.dag`, native fit recipes, and drop-group ablations such as `w3drop`.
It packages the producing worktree, scenes, FitBundles, colour catalogues,
warm-start parameters **and their adjacent metadata/flux state**, plus optional
process-local adapters. Each job runs in a fresh process; checkpoint dependencies
run in order on one GPU. This releases each fit's JAX allocations before the next
fit starts. It does not dispatch or replace existing Condor jobs.

Activate `syndiff` first. Keep archives and outputs under `/astro`, and use a
unique run ID. A T4 is supported; GPU memory and speed depend on the scene and
model. Short pilots establish compatibility and timing, not convergence.

## Import colour.dag arms

Generate a spec from the campaign's current `code/arms.json` and the actual
producing checkout. For example:

```bash
python -m syndiff_pipeline.forward_model.colab_scene colour-spec \
  --fits /astro/armin/koji/syndiff/dev_runs/colour_shoulder_20261008/fits \
  --source-root /home/kshukawa/syndiff_pipeline/.claude/worktrees/colour-radial \
  --arms continue,frozen,polish --fields F1 --folds 0 \
  --steps-per-stage 0,0,160 \
  --output /astro/armin/koji/syndiff/dev_runs/MY_RUN/pilot.json
```

Omit `--steps-per-stage` to retain each arm's actual schedule. Select a parent
arm together with its dependent child (for example `frozen,polish`). Selection
does not change the live DAG or its files. All jobs in one spec use one producing
checkout. Use separate packages for experiments with different scientific code.

## Import another recorded scene fit

Launch contracts containing native `scene_fit` argv can be imported, including
the `w3drop` contracts' `dropped_slots` renderer mask:

```bash
python -m syndiff_pipeline.forward_model.colab_scene contract-spec \
  --source-root /astro/armin/koji/syndiff/dev_runs/paper1_final_fits_20261007/pin \
  --contract /absolute/path/to/launch_contract.json \
  --steps-per-stage 0,0,160 \
  --output /astro/armin/koji/syndiff/dev_runs/MY_RUN/ablation.json
```

Repeat `--contract` for multiple jobs. Point `--source-root` to the implementation
that produced the recorded experiment. Imported drop-group tests require an
initial checkpoint whose dropped coefficients are zero, as in the original W3
launcher. Resuming a nonzero masked checkpoint is not equivalent.

## Package, submit, monitor, stop

```bash
python scripts/colab_job.py publish-scene \
  --spec /astro/armin/koji/syndiff/dev_runs/MY_RUN/pilot.json \
  --archive /astro/armin/koji/syndiff/dev_runs/MY_RUN/pilot.zip

python scripts/colab_job.py submit-scene --gpu T4 \
  --run-id MY_UNIQUE_RUN_ID \
  --archive /astro/armin/koji/syndiff/dev_runs/MY_RUN/pilot.zip \
  --no-discord

python scripts/colab_job.py status MY_UNIQUE_RUN_ID
python scripts/colab_job.py tail MY_UNIQUE_RUN_ID
python scripts/colab_job.py stop MY_UNIQUE_RUN_ID
colab sessions
colab usage
```

`publish-scene` validates the actual packaged parser and input hashes before
allocation. Its default delivery is direct `colab upload`; it registers the
archive locally and needs no Drive authentication. `--transport drive` uploads
to Drive using the existing credentials instead. Scene archives contain the
fitter package's Python modules **and YAML recipes**; hashes bind source and data.
Controller/runner infrastructure comes from the controller checkout, while the
scientific fitter comes from `source_root`. Both are recorded in the manifest.

Scene schedules and science options are frozen in the spec; temporal submit flags
such as `--steps-per-stage`, `--bundle`, `--bootstrap-init-params`, and
`--isolate-stages` are rejected for scene campaigns. `--mode probe` does not
override scene schedules: set the pilot's step count in its spec. `--no-postfit`
is automatic because temporal post-fit diagnostics do not describe a scene fit.
`submit-scene` exposes only scene-runtime options and defaults to T4. The existing
`submit` command also accepts registered scene archives for compatibility.

The controller verifies the requested GPU and JAX backend, monitors telemetry,
syncs hashed nested artifacts, and stops the owned VM after terminal sync.
Each job's `params.npz`, `DONE`, and `execution.json`, plus `SCENE_DONE`, are
required for a successful campaign. A failed scene run's logs are synced even
when no parameter checkpoint exists, so a trainer failure can still end billing.
Keep the controller worktree available until its supervisor has finished.

Results live under
`/astro/armin/koji/syndiff/output/colab/jobs/MY_UNIQUE_RUN_ID/artifacts/JOB_NAME/`.
They include the native fit outputs, launch contract, timing history, and runtime
versions. Evaluate GPU fits on the same held-out scene and support as their CPU
counterparts, then generate/publish the standard residual grids. Floating-point
CPU/GPU agreement is not guaranteed to be bitwise.

After final sync, `results/JOB_NAME/` beside `artifacts/` provides fits usable by
local evaluation: metadata paths point back to the original scenes/catalogues and
binary files link to the verified raw artifacts. The controller records this
directory as `scene_results`. It preserves the remote artifacts and their hashes.
Use the same producing scientific checkout for local scoring, especially for
experimental model terms absent from main.

## Generic spec and offline CPU validation

The input JSON format is:

```json
{
  "schema": 1,
  "source_root": "/absolute/producing/checkout",
  "max_seconds": 7200,
  "jobs": [
    {
      "name": "baseline",
      "scene_dir": "/absolute/scene",
      "init_params": "/absolute/fit/params.npz",
      "colour_file": "/absolute/colour.csv",
      "recipe": "paper1_final",
      "argv": ["--steps-per-stage", "0,0,160"]
    },
    {
      "name": "polish",
      "scene_dir": "/absolute/scene",
      "recipe": "paper1_final",
      "depends_on": ["baseline"],
      "init_from_job": "baseline",
      "argv": ["--steps-per-stage", "0,0,160"]
    }
  ]
}
```

Optional `adapter` points to a Python adapter implementing
`install(initialize=True, freeze=True)` and `provenance()`. Optional `drop_slots`
lists named global8 coefficients to mask inside the renderer. Paths are specified
in job fields; `argv` appends science flags after recipe defaults.

`max_seconds` limits total campaign training/compilation time, defaulting to two
hours. Increase it explicitly for long runs; it does not include provisioning or
sync time. A STOP request or budget expiry ends the running child. Existing output
directories are rejected to prevent mixing experiments. Use a fresh run directory
for reruns; this initial scene-campaign implementation does not resume partial
optimizers across a new session. Temporal Colab resume behavior is unchanged.

To build without registration/upload:

```bash
python -m syndiff_pipeline.forward_model.colab_scene pack \
  --spec /absolute/spec.json --archive /astro/absolute/campaign.zip
```

Extract into an owned `/astro` directory, set `PYTHONPATH` to that directory, and
run the same packaged campaign under a measured Condor CPU allocation:

```bash
python -m syndiff_pipeline.forward_model.colab_scene run \
  --root /astro/absolute/extracted_package \
  --job-dir /astro/absolute/cpu_results --backend cpu
```
