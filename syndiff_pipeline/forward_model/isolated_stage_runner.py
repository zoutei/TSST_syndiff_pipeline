# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Run training stages in fresh Python processes to release JAX GPU allocations.

The normal trainer intentionally keeps all stages in one process so that it can
share prepared bundle arrays.  On a 16 GiB T4, however, compiled executables
and the BFC allocator from stage 1 can coexist with stage-2 compilation.  This
small launcher trades that reuse for a clean CUDA/JAX process at each stage
boundary.  Parameters are handed off through the atomic stage checkpoints.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


N_STAGES = 4


def _stage_steps(value: str) -> list[int]:
    """3 or 4 non-negative step counts.

    3 entries is the historical form and still means "no stage 4" -- it is padded
    with a zero, so every pre-stage-4 command line behaves exactly as before.
    """
    values = [int(part) for part in value.split(",")]
    if len(values) not in (3, N_STAGES) or any(part < 0 for part in values):
        raise argparse.ArgumentTypeError(
            f"must be 3 or {N_STAGES} non-negative comma-separated integers"
        )
    return values + [0] * (N_STAGES - len(values))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps-per-stage", type=_stage_steps, required=True)
    parser.add_argument("--first-stage", type=int, choices=tuple(range(1, N_STAGES + 1)), default=1)
    parser.add_argument("--last-stage", type=int, choices=tuple(range(1, N_STAGES + 1)), default=3)
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="Cooperative stop sentinel shared with the VM runner")
    parser.add_argument(
        "--bootstrap-init-params", type=Path, default=None,
        help="--init-params for the first stage when --first-stage > 1",
    )
    parser.add_argument(
        "train_args", nargs=argparse.REMAINDER,
        help="arguments forwarded to train_from_bundle; put them after --",
    )
    args = parser.parse_args(argv)
    if args.first_stage > args.last_stage:
        parser.error("--first-stage must not exceed --last-stage")
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    forbidden = {
        "--from-bundle", "--out-dir", "--stage", "--start-stage",
        "--init-params", "--steps-per-stage", "--stop-file",
    }
    if any(item in forbidden for item in args.train_args):
        parser.error("stage/bundle/output arguments are managed by isolated_stage_runner")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stop_file = args.stop_file or (args.out_dir / "STOP")
    advance_file = args.out_dir / "ADVANCE_STAGE"
    previous: Path | None = None
    for stage in range(args.first_stage, args.last_stage + 1):
        if stop_file.is_file():
            print(f"=== isolated stage {stage}: stop requested; not launching ===", flush=True)
            return
        stage_steps = [0] * N_STAGES
        stage_steps[stage - 1] = args.steps_per_stage[stage - 1]
        command = [
            sys.executable, "-m", "syndiff_pipeline.forward_model.train_from_bundle",
            "--from-bundle", str(args.from_bundle),
            "--out-dir", str(args.out_dir),
            "--stop-file", str(stop_file),
            "--advance-file", str(advance_file),
            "--stage", str(stage),
            "--start-stage", str(stage),
            "--steps-per-stage", ",".join(map(str, stage_steps)),
        ]
        if stage == args.first_stage and args.bootstrap_init_params is not None:
            bootstrap = args.bootstrap_init_params
            if not bootstrap.is_file():
                raise RuntimeError(f"missing bootstrap init params: {bootstrap}")
            command.extend(("--init-params", str(bootstrap)))
        elif previous is not None:
            if not previous.is_file():
                raise RuntimeError(f"missing checkpoint from stage {stage - 1}: {previous}")
            command.extend(("--init-params", str(previous)))
        command.extend(args.train_args)
        print(f"=== isolated stage {stage}: launching fresh trainer process ===", flush=True)
        subprocess.run(command, check=True)
        previous = args.out_dir / f"params_stage{stage}.npz"
        if not previous.is_file():
            raise RuntimeError(f"stage {stage} exited without {previous.name}")
        print(f"=== isolated stage {stage}: checkpoint verified; releasing process ===", flush=True)
        if stop_file.is_file():
            print(f"=== isolated stage {stage}: stop requested; stopping stage chain ===", flush=True)
            return


if __name__ == "__main__":
    main()
