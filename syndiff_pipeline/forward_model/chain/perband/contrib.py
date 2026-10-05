"""CLI stage ``contrib`` (e2e f03): blurred, seam-corrected, binned band contributions.

``contrib --condor [--n-chunks 40] [--jobs 4]``  writes and submits the Condor array (one job per chunk of cells);
``contrib --chunk K N [--jobs J]``              runs chunk K of N in this process (what an array job executes);
``contrib``                                      runs every cell in this process (chunk 0 of 1)."""
from __future__ import annotations

import argparse
from typing import Sequence

from . import f03_band_contrib


def run(cfg, force: bool = False, condor: bool = False, args: Sequence[str] = ()):
    ap = argparse.ArgumentParser(prog="contrib")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n-chunks", type=int, default=40)
    ap.add_argument("--chunk", nargs=2, type=int, metavar=("K", "N"), default=[0, 1])
    a = ap.parse_args(list(args))
    if condor:
        return f03_band_contrib.submit(cfg, a.n_chunks, a.jobs, do_submit=True)
    return f03_band_contrib.run(cfg, a.chunk[0], a.chunk[1], a.jobs)
