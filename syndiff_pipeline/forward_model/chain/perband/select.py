"""CLI stage ``select`` (e2e f01): cells, same-projection neighbours and cross-projection sources -> ``perband/cells.json``."""
from __future__ import annotations

from . import f01_select
from .paths import chain_paths


def run(cfg, force: bool = False):
    P = chain_paths(cfg)
    if P.cells_json.is_file() and not force:
        return P.cells_json
    f01_select.run(cfg)
    return P.cells_json
