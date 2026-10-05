"""CLI stage ``reduce`` (e2e f04): sum the contributions into the band templates; marks stage ``perband`` DONE.

The stage is only marked done when every selected cell contributed without error."""
from __future__ import annotations

from ..config import is_done, mark_done, write_provenance
from . import f04_reduce
from .paths import chain_paths


def run(cfg, force: bool = False, make_figure: bool = True):
    P = chain_paths(cfg)
    stage = cfg.stage_dir("perband")
    if is_done(stage) and not force:
        return stage
    summ = f04_reduce.run(cfg, make_figure=make_figure)
    if summ["missing"] or summ["errors"]:
        raise RuntimeError(f"perband incomplete: {len(summ['missing'])} cells missing, {len(summ['errors'])} with errors "
                           f"(see {P.band_templates / 'sum_check.json'}); not marking the stage done")
    write_provenance(stage, cfg, {"cells": P.cells_json, "publisher_lists": P.publisher_lists,
                                  "sum_check": P.band_templates / "sum_check.json",
                                  "store_weights": P.store_weights_json, "mapping": P.mapping_dir})
    mark_done(stage)
    return stage
