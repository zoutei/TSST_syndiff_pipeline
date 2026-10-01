# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
#!/usr/bin/env python3
"""Colab driver: unpack already-uploaded zip and run half-orbit irregular fit.

Expects /content/colab_fullccd_mag710_irreg_295.zip already uploaded.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ZIP = Path("/content/colab_fullccd_mag710_irreg_295.zip")
ROOT = Path("/content")
OUT = Path("/content/fullccd_mag710_irreg_295_chunk10")
STAMP_CHUNK = 10


def sh(cmd: list[str] | str, check: bool = True) -> None:
    print(f"+ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=check)


def main() -> None:
    t0 = time.time()
    print("jax/device probe…", flush=True)
    import jax
    print(f"jax={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}", flush=True)

    if not ZIP.is_file():
        raise SystemExit(f"missing upload: {ZIP}")

    sh(["unzip", "-qo", str(ZIP), "-d", str(ROOT)])
    os.environ["PYTHONPATH"] = f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"

    bundle = ROOT / "bundle" / "fit_bundle.npz"
    if not bundle.is_file():
        raise SystemExit(f"missing {bundle} after unzip")

    cmd = [
        sys.executable, "-m", "syndiff_pipeline.forward_model.train_from_bundle",
        "--from-bundle", str(bundle),
        "--stage", "3",
        "--start-stage", "1",
        "--steps-per-stage", "8,600,600",
        "--lr-per-stage", "1e-2,3e-4,1e-4",
        "--epsf-lr-scale", "1.0",
        "--stage2-freeze-wcs-steps", "20",
        "--stamp-chunk", str(STAMP_CHUNK),
        "--log-every", "20",
        "--reject-every", "20",
        "--checkpoint-every", "50",
        "--out-dir", str(OUT),
    ]
    print("starting fit:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    rc = subprocess.call(cmd, env=env)
    print(f"fit exit={rc} wall={time.time() - t0:.1f}s", flush=True)
    if rc != 0:
        raise SystemExit(rc)
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
