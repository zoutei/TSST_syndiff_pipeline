"""f02: band cells ``C_b = Z * w_b F_b`` for every cell in ``cells.json`` (``all_band_cells``).

``Z`` is recovered from the STORED production combined cell (``perband.split_with_store_weights``); raw bands come
from the shared raw zarr, or are streamed from PS1 in memory when absent there (no zarr write).  ``w_b`` are the
band weights recorded in the combined store's recipe (``_provenance.json`` ``recipe_params.band_weights`` of the
resolved cell), not a hard-coded set: a production-weight store gives exactly the old arithmetic, an adopted-weight
(D13) store gives band cells that already carry the adopted weights.  Each cell's recorded weights must equal the
set the chain assumes (``paths.chain_band_weights``; config ``inputs.combined_store_weights``).

Per cell: ``band_cells/<name>.npz`` (float32 r,i,z,y + JSON ``check``, which records ``store_band_weights``).
Summary: ``band_cells/split_check*.json``.  Port of e2e ``f02_band_cells.py`` (validated bitwise on 57 cells).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .paths import BANDS, RECIPE_CFG, chain_band_weights, chain_paths, combined_store_weights_mode


def chain_band_weights_production() -> dict:
    from syndiff_pipeline.template_creation.processing.combined_store import DEFAULT_BAND_WEIGHTS
    return {b: float(DEFAULT_BAND_WEIGHTS[b]) for b in BANDS}


def store_recipe(cfg, band_weights: Optional[Mapping[str, float]] = None) -> dict:
    """The ``combined_skycell`` recipe of the field's store.  ``band_weights=None`` -> production defaults
    (exactly the recipe of the old store)."""
    from syndiff_pipeline.template_creation.processing.combined_store import production_combined_recipe
    P = chain_paths(cfg)
    recipe = production_combined_recipe(RECIPE_CFG, data_root=P.data, sector=P.sector, camera=P.camera, ccd=P.ccd)
    if band_weights is not None and combined_store_weights_mode(cfg) != "production":
        recipe = {**recipe, "band_weights": {b: float(band_weights[b]) for b in BANDS}}
    return recipe


def load_store_cell(data_root, name: str, recipe: Mapping):
    """(cell dict | None, band weights the store recorded for it | None).

    The recorded weights are read from the cell's ``_provenance.json`` (``recipe_params.band_weights``); a cell dir
    without the sidecar falls back to the weights of the recipe it was resolved with (the fingerprint includes them)."""
    from syndiff_pipeline.template_creation.processing import perband as PB
    from syndiff_pipeline.template_creation.processing.combined_store import (
        combined_cell_dir, resolve_combined_fingerprint_for_recipe, try_load_combined_cell)
    parts = str(name).split(".")
    if len(parts) < 3:
        return None, None
    projection, cell = ".".join(parts[:2]), parts[2]
    fp = resolve_combined_fingerprint_for_recipe(data_root, projection, cell, recipe)
    loaded = try_load_combined_cell(data_root, projection, cell, fp) if fp is not None else None
    if loaded is None:
        return None, None
    side = combined_cell_dir(data_root, projection, cell, fp) / "_provenance.json"
    rp = recipe
    if side.is_file():
        rp = json.loads(side.read_text()).get("recipe_params", recipe)
    return loaded, PB.band_weights_from_recipe(rp)


def one(cfg, name: str):
    import zarr
    from syndiff_pipeline.template_creation.processing import perband as PB
    from syndiff_pipeline.template_creation.processing.zarr_utils import load_skycell_bands_masks_and_headers

    P = chain_paths(cfg)
    out = P.band_cells / f"{name}.npz"
    expected = chain_band_weights(cfg)
    if out.exists():
        try:
            chk = json.loads(str(np.load(out)["check"]))
            # cells written before the weights were recorded are production-weight cells
            have = chk.get("store_band_weights") or (
                chain_band_weights_production() if combined_store_weights_mode(cfg) == "production" else None)
            if "error" not in chk and have is not None and PB.same_band_weights(have, expected):
                return name, chk
        except Exception:
            pass
    t0 = time.time()
    try:
        hit, recorded = load_store_cell(P.data, name, store_recipe(cfg, expected))
        if hit is None:
            return name, {"error": "no combined-store cell"}
        stored = np.asarray(hit["combined_image"])
        proj = name.split(".")[1]
        z = zarr.open(str(P.raw_zarr), mode="r")
        bands, masks, weights, headers, _hw = load_skycell_bands_masks_and_headers(z, proj, name)
        source = "zarr"
        if not bands:
            from syndiff_pipeline.template_creation.processing.ps1_download import fetch_skycell_bands_masks_and_headers
            bands, masks, weights, headers, _hw = fetch_skycell_bands_masks_and_headers(name)
            source = "stream"
        if not bands:
            return name, {"error": "no raw bands (zarr or stream)"}
        wb, split, before = PB.split_with_store_weights(bands, headers, stored, recorded, expected_band_weights=expected)
        chk = PB.split_residual(split, stored)
        chk.update(source=source, bands=sorted(split), shape=list(stored.shape),
                   store_band_weights=recorded,
                   zeroed_frac=float(PB.zeroed_mask(before, stored).mean()),
                   zero_after_frac=float((stored == 0).mean()),
                   before_eq_stored_on_kept=bool(np.array_equal(before[stored != 0], stored[stored != 0], equal_nan=True)),
                   band_flux={b: float(np.nansum(split[b], dtype=np.float64)) for b in split},
                   stored_flux=float(np.nansum(stored, dtype=np.float64)), seconds=time.time() - t0)
        P.band_cells.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{b: split[b] for b in split}, check=json.dumps(chk))
        tmp.rename(out)
        return name, chk
    except Exception as e:
        return name, {"error": f"{type(e).__name__}: {e}"}


def run(cfg, n_jobs: int = 12, extra: bool = False, chunk: Optional[str] = None) -> dict:
    """Build the band cells.  ``extra``: also the publisher-list neighbours of ``publisher_lists.json``;
    ``chunk`` ``"k/n"``: this process takes ``names[k::n]``."""
    from joblib import Parallel, delayed

    P = chain_paths(cfg)
    P.band_cells.mkdir(parents=True, exist_ok=True)
    names = json.loads(P.cells_json.read_text())["all_band_cells"]
    if extra:
        names = sorted(set(json.loads(P.publisher_lists.read_text())["extra_band_cells"]) | set(names))
    if chunk:
        k, n = map(int, chunk.split("/"))
        names = names[k::n]
    res = dict(Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(delayed(one)(cfg, n) for n in names))
    tag = f"_{chunk.replace('/', 'of')}" if chunk else ("_extra" if extra else "")
    (P.band_cells / f"split_check{tag}.json").write_text(json.dumps(res, indent=1))
    bad = {k: v for k, v in res.items() if "error" in v or v["max_rel_to_peak"] > 1e-6 or v["n_nan_mismatch"]}
    src = {}
    for v in res.values():
        src[v.get("source", "error")] = src.get(v.get("source", "error"), 0) + 1
    print(len(res), "cells;", len(bad), "flagged; sources", src)
    print(json.dumps(dict(list(bad.items())[:20]), indent=1)[:4000])
    ok = [v for v in res.values() if "error" not in v]
    if ok:
        print("max rel", max(v["max_rel_to_peak"] for v in ok),
              "kept-pixels bitwise:", sum(v["before_eq_stored_on_kept"] for v in ok), "/", len(ok))
    return res
