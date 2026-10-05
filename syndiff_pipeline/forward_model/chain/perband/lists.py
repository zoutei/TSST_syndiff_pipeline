"""CLI stage ``lists`` (e2e f01b): candidate neighbour sets from every skycell list on disk -> ``perband/publisher_lists.json``."""
from __future__ import annotations

from . import f01b_lists
from .paths import chain_paths


def run(cfg, force: bool = False):
    P = chain_paths(cfg)
    if P.publisher_lists.is_file() and not force:
        return P.publisher_lists
    f01b_lists.run(cfg)
    return P.publisher_lists
