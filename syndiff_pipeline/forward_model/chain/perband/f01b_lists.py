"""f01b: candidate same-projection neighbour sets per cell, from EVERY skycell list on disk.

The stored canonical convolved cell was blurred by whichever run published it, with that run's skycell list.
Instead of mapping publishers by git SHA, collect every ``master_skycells_list*.csv`` under ``data/`` that contains
the cell, compute the neighbour set each gives (``ps1_process._find_projection_neighbors``) and deduplicate.
f03 tries the distinct candidates (own list first, then by how many lists give that candidate) and keeps the one that
reproduces the stored cell.  Output ``perband/publisher_lists.json``
({lists, cells: {name: {order, neighbours, n_lists}}, extra_band_cells}).

Grouping key (e2e fix, 2026-09-30): the neighbour NAMES **and** the row layout of rows R-1..R+1 of the list's
projection metadata. Lists with identical neighbour names can still differ in the cells/x-indices of the adjacent rows
the row-path blur assembles, and then give different blurs (skycell.2486.029: own list 0.81 vs 1e-7 relative error).
"""
from __future__ import annotations

import collections
import glob
import json

from .paths import chain_paths


def group_key(md: dict, cell: str, neighbours_of) -> tuple:
    """(neighbour-name set, row layout of rows R-1..R+1) of ``cell`` under projection metadata ``md``.

    ``neighbours_of(md, cell, R, X)`` yields ``(name, _, _)`` tuples (``_find_projection_neighbors``)."""
    row_of = {n: (r, x) for r, cs in md["rows"].items() for n, x in cs}
    R, X = row_of[cell]
    nb = frozenset(n for n, _, _ in neighbours_of(md, cell, R, X))
    sig = tuple((r, tuple(md["rows"].get(r, ()))) for r in (R - 1, R, R + 1))
    return nb, sig


def rank_candidates(sets: dict, own: str) -> list:
    """Candidate groups ordered own-list-first, then by descending number of lists giving the group."""
    return sorted(sets.items(), key=lambda kv: (own not in kv[1], -len(kv[1])))


def run(cfg) -> dict:
    import pandas as pd
    from syndiff_pipeline.template_creation.processing.ps1_process import (
        _find_projection_neighbors, extract_projection_metadata)

    P = chain_paths(cfg)
    paths = sorted(set(glob.glob(str(P.data / "s*/c*/k*/mapping*/oversampling_*/*skycells_list*.csv"))
                       + glob.glob(str(P.data / "s*/c*/k*/single_ffi_epsf_test/mapping/oversampling_*/*skycells_list*.csv"))))
    own = str(P.skylist)
    if own not in paths:
        paths.insert(0, own)
    sel = json.loads(P.cells_json.read_text())
    cells = sel["cells"]
    want = set(cells)
    dfs = {}
    for p in paths:
        try:
            d = pd.read_csv(p).set_index("NAME", drop=False)
        except Exception as e:
            print("skip", p, e)
            continue
        if want & set(d.index):
            dfs[p] = d
    print(len(dfs), "lists overlap this field")
    mdc, out, extra, have = {}, {}, set(), set(sel["all_band_cells"])
    for c in cells:
        sets = {}   # (frozenset(neighbours), row layout) -> [list paths]
        for p, d in dfs.items():
            if c not in d.index:
                continue
            proj = str(d.loc[c, "projection"])
            if (p, proj) not in mdc:
                mdc[(p, proj)] = extract_projection_metadata(d.reset_index(drop=True), proj)
            sets.setdefault(group_key(mdc[(p, proj)], c, _find_projection_neighbors), []).append(p)
        ranked = rank_candidates(sets, own)
        order, nbrs = [], {}
        for (nb, _sig), ps in ranked:
            key = own if own in ps else ps[0]
            order.append(key)
            nbrs[key] = sorted(nb)
            extra |= set(nb) - have
        out[c] = dict(order=order, neighbours=nbrs, n_lists={o: len(ps) for o, (_k, ps) in zip(order, ranked)})
    res = dict(lists={p: p for p in dfs}, cells=out, extra_band_cells=sorted(extra), n_extra=len(extra))
    P.publisher_lists.write_text(json.dumps(res, indent=1))
    print("extra band cells:", len(extra), "| distinct candidate groups per cell:",
          collections.Counter(len(v["order"]) for v in out.values()))
    return res
