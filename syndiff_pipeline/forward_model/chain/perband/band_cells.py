"""CLI stage ``band_cells`` (e2e f02): band cells ``C_b = Z w_b F_b`` from the combined store.

Extra tokens (after the stage name): ``--jobs N`` (default 12), ``--extra`` (also the publisher-list neighbours),
``--chunk K/N`` (this process takes cells[K::N])."""
from __future__ import annotations

import argparse
from typing import Sequence

from . import f02_band_cells


def run(cfg, force: bool = False, args: Sequence[str] = ()):
    ap = argparse.ArgumentParser(prog="band_cells")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--chunk", default=None)
    a = ap.parse_args(list(args))
    return f02_band_cells.run(cfg, n_jobs=a.jobs, extra=a.extra, chunk=a.chunk)
