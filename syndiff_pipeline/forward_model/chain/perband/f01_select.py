"""f01: all skycells feeding the whole CCD template, plus every cell their blur/seam correction reads.

cells      = skycells in the OS4 mapping's skycell list (each gets a band template contribution)
same_nbrs  = same-projection neighbours used by the whole-row-path blur (ps1_process._find_projection_neighbors)
xproj_src  = cross-projection padding sources (padding_correction.cross_projection_padding_spec)
band cells are needed for cells | same_nbrs | xproj_src. Output ``perband/cells.json``.

Port of e2e ``f01_select.py``; logic unchanged.
"""
from __future__ import annotations

import json

from .paths import chain_paths


def run(cfg) -> dict:
    import pandas as pd
    from syndiff_pipeline.template_creation.processing.padding_correction import cross_projection_padding_spec
    from syndiff_pipeline.template_creation.processing.ps1_process import (
        _find_projection_neighbors, extract_projection_metadata)

    P = chain_paths(cfg)
    P.perband.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(P.skylist).set_index("NAME", drop=False)
    cells = sorted(df.index)
    md_cache, same, xproj = {}, {}, {}
    for name in cells:
        proj = str(df.loc[name, "projection"])
        if proj not in md_cache:
            md_cache[proj] = extract_projection_metadata(df, proj)
        nb = _find_projection_neighbors(md_cache[proj], name, int(df.loc[name, "y"]), int(df.loc[name, "x"]))
        same[name] = sorted({n for n, _, _ in nb})
        spec = cross_projection_padding_spec(df.loc[name])
        if spec:
            xproj[name] = spec
    need = set(cells) | {n for v in same.values() for n in v} | {s["neighbor"] for v in xproj.values() for s in v}
    out = dict(cells=cells, same_neighbours=same, xproj=xproj, all_band_cells=sorted(need),
               n_cells=len(cells), n_band_cells=len(need), n_xproj_cells=len(xproj),
               projections=sorted({str(df.loc[c, "projection"]) for c in cells}),
               outside_list=sorted(need - set(cells)))
    P.cells_json.write_text(json.dumps(out, indent=1))
    return out
