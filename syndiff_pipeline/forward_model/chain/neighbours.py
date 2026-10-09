"""Neighbour scenes: add recorded-but-omitted Gaia neighbours (``T <= tmax``) to a scene as nuisance sources.

Ported from ``dev_runs/neighbour_ps1_fixed_training_20261003/code/gaia_joint_scene.py`` (union recovery, topology
rebuild, source preparation, augmentation, scene writing) and ``dev_runs/paper1_final_fits_20261007/code/nbr.py``
(candidate selection, support filter, fold-table padding). Numerics unchanged; only the module loading (``modules(pin)``)
and the false-position control (unused by the chain) were dropped.

An added source is a role-2 nuisance: its template is stop-gradient, its flux is solved with the scene's weak catalogue
prior; the original pixel union, owners and source arrays are bitwise unchanged. Pairs/islands are rebuilt from actual
shared fitted pixels.

Removal-evidence gate: the builder refuses rows without ``full_positive_allowed``. ``neighbours.gate_override: true``
(user decision 2026-10-07, training_fixes option #5) marks every candidate allowed; a neighbour that was not actually
erased from the template should then fit to ~0 flux.

Config::

    neighbours:
      ledger: /astro/.../gaia_neighbour_joint_fit_20261002/ledger/<F>/candidates.csv
      tmax: 17
      gate_override: true

Stage ``nbr_boot`` / ``nbr_final``: ``scene_<which>`` + neighbours placed with the WCS of ``init_<which>`` (the
photutils init on that image; training-only for fold scenes, see ``crossfit``).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

STAR_KEYS = ('star_bundle_index', 'source_id', 'tess_mag', 'tess_flux', 'role',
             'cx', 'cy', 'x0', 'y0', 'data', 'noise', 'valid', 'finite', 'uid', 'owner')
GATE_OVERRIDE_NOTE = "user 2026-10-07: removal_status unverified, overridden for training_fixes option 5"


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def union_arrays(z):
    """Recover unique pixels from their immutable owner copies and check consistency.

    Returns U+1 arrays; slot U is the off-union sentinel.  Data and noise must agree bitwise across all existing
    copies, including invalid pixels.
    """
    U, S = int(z['n_union']), int(z['stamp'])
    uid = z['uid']
    owner = z['owner'].astype(bool)
    if np.any(owner & (uid >= U)):
        raise AssertionError('dummy pixel has an owner')
    count = np.bincount(uid[owner], minlength=U)
    if not np.all(count == 1):
        raise AssertionError('each real union pixel must have exactly one owner')
    k = np.arange(S * S)
    xx = z['cx'][:, None] + (k % S - S // 2)
    yy = z['cy'][:, None] + (k // S - S // 2)
    out = {}
    good = uid < U
    for key, arr in dict(x=xx, y=yy, data=z['data'], noise=z['noise'],
                         valid=z['valid'], finite=z['finite']).items():
        a = np.zeros(U + 1, dtype=arr.dtype)
        a[uid[owner]] = arr[owner]
        if key == 'noise':
            a[U] = 1
        if not np.array_equal(a[uid[good]], arr[good], equal_nan=True):
            raise AssertionError(f'inconsistent duplicate pixel {key}')
        out[key] = a
    if len(np.unique(np.c_[out['x'][:U], out['y'][:U]], axis=0)) != U:
        raise AssertionError('distinct UIDs refer to the same physical pixel')
    return out


def rebuild_topology(z, pixel_active_u=None):
    """Replace pair/island tables using actual shared fitted pixels (geometric pairs with no common active
    likelihood pixel are disconnected)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    z = {k: v for k, v in z.items() if not k.startswith('tier')}
    N, S, U = len(z['role']), int(z['stamp']), int(z['n_union'])
    active = np.ones(U + 1, bool) if pixel_active_u is None else np.asarray(pixel_active_u, bool).copy()
    if active.shape != (U + 1,):
        raise ValueError('pixel_active_u must have shape (U+1,)')
    active[U] = False
    centres = np.c_[z['cx'], z['cy']]
    pairs = cKDTree(centres).query_pairs(S - .5, p=np.inf, output_type='ndarray')
    pairs = np.asarray(pairs).reshape(-1, 2)
    # Chunking avoids a large intermediate (# geometric pairs x S x S).
    pi_all, pj_all, lj_all, pm_all = [], [], [], []
    ky, kx = np.divmod(np.arange(S * S), S)
    for start in range(0, len(pairs), 8192):
        pi, pj = pairs[start:start + 8192].T
        dx = z['cx'][pj] - z['cx'][pi]
        dy = z['cy'][pj] - z['cy'][pi]
        jx, jy = kx[None] - dx[:, None], ky[None] - dy[:, None]
        geom = (jx >= 0) & (jx < S) & (jy >= 0) & (jy < S)
        lj = (np.clip(jy, 0, S - 1) * S + np.clip(jx, 0, S - 1)).astype(np.int16)
        other_uid = np.take_along_axis(z['uid'][pj], lj, axis=1)
        common = geom & (z['uid'][pi] < U) & (z['uid'][pi] == other_uid)
        weighted = common & z['valid'][pi] & active[z['uid'][pi]]
        keep = weighted.any(axis=1)
        pi_all.append(pi[keep]); pj_all.append(pj[keep])
        lj_all.append(lj[keep]); pm_all.append(common[keep])
    pi = np.concatenate(pi_all).astype(np.int32) if pi_all else np.empty(0, np.int32)
    pj = np.concatenate(pj_all).astype(np.int32) if pj_all else np.empty(0, np.int32)
    lj = np.concatenate(lj_all) if lj_all else np.empty((0, S * S), np.int16)
    pm = np.concatenate(pm_all) if pm_all else np.empty((0, S * S), bool)
    n_island, island = connected_components(coo_matrix((np.ones(len(pi)), (pi, pj)), shape=(N, N)), directed=False)
    sizes = np.bincount(island, minlength=n_island)
    ceil2 = np.array([1 << (int(s) - 1).bit_length() for s in sizes])
    tiers = np.unique(ceil2)
    star_tier, star_row, star_slot = np.zeros(N, np.int16), np.zeros(N, np.int32), np.zeros(N, np.int32)
    for ti, K in enumerate(tiers):
        ids = np.flatnonzero(ceil2 == K)
        row_of = np.full(n_island, -1, np.int32); row_of[ids] = np.arange(len(ids))
        mat = np.full((len(ids), K), -1, np.int32)
        fill = np.zeros(len(ids), np.int32)
        for s in np.flatnonzero(ceil2[island] == K):
            r = row_of[island[s]]; slot = fill[r]
            mat[r, slot] = s
            star_tier[s], star_row[s], star_slot[s] = ti, r, slot
            fill[r] += 1
        z[f'tier{K}_star_idx'] = mat
    z.update(pair_i=pi, pair_j=pj, pair_lj=lj, pair_mask=pm,
             island=island.astype(np.int32), star_tier=star_tier, star_row=star_row,
             star_slot=star_slot, tiers=tiers.astype(np.int32))
    return z


def prepare_sources(scene, rows, *, coeff=None, colour_file=None):
    """Exact catalogue IDs + the bundle's PM/astrometry convention; no tmag cut."""
    import pandas as pd
    from syndiff_pipeline.forward_model import cheb_wcs as CW, gaia_pm as GP

    rows = rows.copy().reset_index(drop=True)
    if not pd.api.types.is_integer_dtype(rows['source_id'].dtype):
        raise ValueError('source_id must be exact integer, never a float64 catalogue ID')
    if rows['source_id'].duplicated().any():
        raise ValueError('ledger must deduplicate source_id before building scene')
    if np.isin(rows['source_id'], scene.z['source_id']).any():
        raise ValueError('added source already exists in frozen scene')
    if 'model_x' in rows or 'model_y' in rows:
        raise ValueError('detector-position overrides are not supported here')
    pm = scene.src.meta.get('gaia_pm_propagation', {})
    ra, dec = rows['ra'].to_numpy(float), rows['dec'].to_numpy(float)
    if pm.get('applied', False):
        if 'ref_epoch' in rows and not np.allclose(rows['ref_epoch'].fillna(2016), 2016):
            raise ValueError('baseline uses Gaia epoch2016; input ledger has another epoch')
        pma = rows['pmra'].to_numpy(float) if 'pmra' in rows else np.full(len(rows), np.nan)
        pmd = rows['pmdec'].to_numpy(float) if 'pmdec' in rows else np.full(len(rows), np.nan)
        ra, dec = GP.propagate_ra_dec(ra, dec, pma, pmd, GP.btjd_to_time(pm['target_btjd']))
    if not np.all(np.isfinite(ra) & np.isfinite(dec)):
        raise ValueError('nonfinite Gaia coordinate')
    # run_fit built the original basis from float32 sky coordinates: keep that convention for the added sources.
    xl, yl, basis = CW.star_basis(ra.astype(np.float32), dec.astype(np.float32), scene.src.cheb_static)
    xl, yl, basis = map(np.asarray, (xl, yl, basis))
    coeff = np.asarray(scene.src.params0['wcs_coeff'] if coeff is None else coeff)
    xx, yy = CW.eval_all_positions(xl, yl, basis, coeff, np.asarray(scene.src.wcs_frame_basis), scene.src.cheb_static.n_terms)
    xx, yy = np.asarray(xx)[:, 0], np.asarray(yy)[:, 0]
    fallback = rows['bp_rp'].to_numpy(float) if 'bp_rp' in rows else np.full(len(rows), np.nan)
    colour = fallback.copy()
    origin = np.where(np.isfinite(fallback), 'gaia_bp_rp', 'unknown').astype('<U24')
    if colour_file:
        cf = pd.read_csv(colour_file, dtype={'source_id': 'int64'})
        table = dict(zip(cf.source_id, cf.colour))
        got = np.array([table.get(int(s), np.nan) for s in rows.source_id], float)
        take = np.isfinite(got); colour[take] = got[take]; origin[take] = 'frozen_colour_file'
    src_arrays = dict(ra=ra, dec=dec, x_lin=xl, y_lin=yl, cheb_basis=basis, bp_rp=colour)
    rows['model_ra_epoch'] = ra; rows['model_dec_epoch'] = dec
    rows['model_x'] = xx; rows['model_y'] = yy; rows['colour'] = colour; rows['colour_origin'] = origin
    rows['colour_uncertainty_unresolved'] = origin != 'frozen_colour_file'
    return rows, src_arrays


def augment_arrays(z, rows, source_start):
    """Add source stamps mapped only onto the unchanged existing union."""
    U, S, N = int(z['n_union']), int(z['stamp']), len(z['role'])
    u = union_arrays(z)
    cx = np.round(rows.model_x).astype(np.int32).to_numpy()
    cy = np.round(rows.model_y).astype(np.int32).to_numpy()
    k = np.arange(S * S)
    xx = cx[:, None] + (k % S - S // 2); yy = cy[:, None] + (k // S - S // 2)
    # A collision-free integer encoding supports negative/off-array coordinates.
    lo = int(min(u['x'].min(), xx.min(initial=0))) - 1
    width = int(max(u['x'].max(), xx.max(initial=0)) - lo) + 2
    old = (u['y'][:U].astype(np.int64) * width + u['x'][:U] - lo)
    order = np.argsort(old); ordered = old[order]
    ask = yy.astype(np.int64) * width + xx - lo
    pos = np.searchsorted(ordered, ask); safe = np.clip(pos, 0, max(U - 1, 0))
    on = (pos < U) & (ordered[safe] == ask)
    uid = np.where(on, order[safe], U).astype(np.int32)
    support = (u['valid'][uid]).sum(axis=1)
    if np.any(support == 0):
        bad = rows.loc[support == 0, 'source_id'].tolist()
        raise ValueError(f'{len(bad)} added sources have no common valid pixel; remove in ledger selection: {bad[:5]}')
    flux = rows.tess_flux.to_numpy(float) if 'tess_flux' in rows else np.full(len(rows), np.nan)
    mag = rows.tess_mag.to_numpy(float) if 'tess_mag' in rows else np.full(len(rows), np.nan)
    # A missing flux is never a physical zero-flux prior: consumers must use prior_available.
    available = np.isfinite(flux) & (flux > 0)
    new = dict(star_bundle_index=np.arange(source_start, source_start + len(rows), dtype=np.int32),
               source_id=rows.source_id.to_numpy(np.int64), tess_mag=mag.astype(np.float32),
               tess_flux=np.where(available, flux, 0).astype(np.float32), role=np.full(len(rows), 2, np.int8),
               cx=cx, cy=cy, x0=rows.model_x.to_numpy(np.float32), y0=rows.model_y.to_numpy(np.float32),
               uid=uid, owner=np.zeros_like(uid, bool), data=u['data'][uid], noise=u['noise'][uid],
               valid=u['valid'][uid], finite=u['finite'][uid])
    out = {key: value.copy() for key, value in z.items()}
    for key in STAR_KEYS:
        out[key] = np.concatenate([z[key], new[key]], axis=0)
    out['is_added'] = np.r_[np.zeros(N, bool), np.ones(len(rows), bool)]
    out['prior_available'] = np.r_[np.isfinite(z['tess_flux']) & (z['tess_flux'] > 0), available]
    out['added_valid_support'] = np.r_[np.zeros(N, np.int32), support.astype(np.int32)]
    for key in STAR_KEYS:
        if not np.array_equal(out[key][:N], z[key], equal_nan=True):
            raise AssertionError(f'original source array changed: {key}')
    union_arrays(out)
    return rebuild_topology(out)


def write_scene(scene_dir, rows, out_dir, *, params_file=None, colour_file=None):
    """Write the augmented Scene (``scene_bundle.npz``, ``scene_meta.json``, ``fit_bundle.npz``, ``added_sources.csv``)
    into ``out_dir`` (must not already hold a scene)."""
    import pandas as pd
    from syndiff_pipeline.forward_model import scene_fit as SF

    scene_dir, out_dir = Path(scene_dir), Path(out_dir)
    scene = SF.Scene(scene_dir)
    if ('full_positive_allowed' not in rows or not pd.api.types.is_bool_dtype(rows.full_positive_allowed.dtype)
            or not rows.full_positive_allowed.fillna(False).all()):
        raise ValueError('Full-removal evidence gate failed; causal full-positive scene forbidden')
    if (out_dir / 'scene_bundle.npz').exists():
        raise FileExistsError(f'{out_dir} already holds a scene')
    coeff = None
    if params_file:
        with np.load(params_file) as p:
            coeff = p['wcs_coeff']
    rows, addition = prepare_sources(scene, rows, coeff=coeff, colour_file=colour_file)
    z = augment_arrays(scene.z, rows, len(scene.src.ra))
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(scene.meta['source_bundle'])
    with np.load(source, allow_pickle=False) as old:
        raw = {k: old[k] for k in old.files}
    for key, value in addition.items():
        old = raw.get(key, np.full(len(scene.src.ra), np.nan))
        raw[key] = np.concatenate([old, np.asarray(value, dtype=old.dtype)], axis=0)
    raw['kept_star_mask'] = np.r_[raw['kept_star_mask'], np.zeros(len(rows), bool)]
    np.savez_compressed(out_dir / 'fit_bundle.npz', **raw)
    srcmeta = copy.deepcopy(scene.src.meta)
    srcmeta['gaia_joint_augmentation'] = dict(original_source_bundle=str(source), n_added=len(rows),
        packed_groups_unchanged=True, usage='Scene renderer; added sources intentionally have no packed groups')
    (out_dir / 'fit_bundle_meta.json').write_text(json.dumps(srcmeta, indent=2))
    np.savez_compressed(out_dir / 'scene_bundle.npz', **z)
    rows.to_csv(out_dir / 'added_sources.csv', index=False)
    sizes = np.bincount(z['island'])
    meta = copy.deepcopy(scene.meta)
    meta.update(source_bundle=str((out_dir / 'fit_bundle.npz').resolve()), n_stars=len(z['role']),
                n_pairs=len(z['pair_i']), n_islands=len(sizes), max_island=int(sizes.max()),
                n_roles={k: int((z['role'] == r).sum()) for r, k in enumerate(('contrib', 'anchor', 'nuisance'))},
                tiers={str(K): len(z[f'tier{K}_star_idx']) for K in z['tiers']})
    meta['gaia_joint_augmentation'] = dict(original_scene=str(scene_dir.resolve()), n_original=scene.N,
        n_added=len(rows), geometry_only=False, false_control=False,
        original_scene_sha256=sha256(scene_dir / 'scene_bundle.npz'),
        source_bundle_sha256=sha256(source), old_pixel_union_unchanged=True,
        original_source_arrays_bitwise_unchanged=True, owner_policy='original owners unchanged; appended owners false',
        pair_policy='common active valid pixel overlap, no geometric-only coupling',
        role2='full template stop-gradient; flux solve remains differentiated',
        missing_flux_prior_count=int((~z['prior_available'][scene.N:]).sum()),
        requires_per_source_prior_adapter=bool((~z['prior_available'][scene.N:]).any()),
        colour_policy='same frozen colour file; Gaia BP-RP fallback; missing stays NaN with unresolved uncertainty flag',
        astrometry='baseline GP epoch propagation + float32 CW.star_basis convention',
        common_pixel_mask_required=False)
    (out_dir / 'scene_meta.json').write_text(json.dumps(meta, indent=2))
    # The standard loader must accept this representation.
    check = SF.Scene(out_dir)
    if check.N != scene.N + len(rows) or check.U != scene.U:
        raise AssertionError('Scene reload geometry mismatch')
    for key in STAR_KEYS:
        if not np.array_equal(check.z[key][:scene.N], scene.z[key], equal_nan=True):
            raise AssertionError(f'reload changed original {key}')
    return meta


# ---------------------------------------------------------------------- candidate selection
VERIFIED_NOTE = "verified removal ledger (assoc_r2, removal_ledger/revision.py); no gate override"


def assoc_r2_table(ledger, gaia_catalog) -> tuple["pd.DataFrame", dict]:
    """The verified removal-ledger neighbour table as builder rows (``source_id, ra, dec, pmra, pmdec, ref_epoch,
    tess_mag, bp_rp``). Gaia rows only (PS1-only rows have no Gaia T). Astrometry: the field Gaia catalogue by
    source_id (DR3 epoch 2016 + proper motion; the table's own ra/dec are not plain epoch-2016 Gaia positions, median
    0.04 arcsec off on F1); rows outside the catalogue keep the table ra/dec with no proper motion (counted).
    ``tess_mag`` and the Gaia photometry come from the table."""
    import pandas as pd

    t = pd.read_parquet(ledger)
    g = t[t["gaia_id"].notna()].copy()
    g["source_id"] = g["gaia_id"].astype(str).str.strip().astype("int64")   # exact: never via float64
    if g["source_id"].duplicated().any():
        raise ValueError(f"{ledger}: duplicate gaia_id rows")
    cols = set(pd.read_csv(gaia_catalog, nrows=0).columns)
    cat = pd.read_csv(gaia_catalog, usecols=["source_id", "ra", "dec"] + [c for c in ("pmra", "pmdec") if c in cols],
                      dtype={"source_id": "int64"})
    for c in ("pmra", "pmdec"):         # some field catalogues carry no proper motion (C1); their scenes do not propagate
        if c not in cat:
            cat[c] = np.nan
    m = g.drop(columns=["ra", "dec"]).merge(cat, on="source_id", how="left")
    fb = m["ra"].isna().to_numpy()
    m.loc[fb, "ra"] = g.set_index("source_id").loc[m.loc[fb, "source_id"], "ra"].to_numpy()
    m.loc[fb, "dec"] = g.set_index("source_id").loc[m.loc[fb, "source_id"], "dec"].to_numpy()
    m["ref_epoch"] = 2016.0
    m["bp_rp"] = m["phot_bp_mean_mag"] - m["phot_rp_mean_mag"]
    info = dict(n_rows=int(len(t)), n_gaia_rows=int(len(g)), n_without_gaia=int(len(t) - len(g)),
                n_not_in_gaia_catalogue=int(fb.sum()), gaia_catalog=str(gaia_catalog))
    keep = ["source_id", "ra", "dec", "pmra", "pmdec", "ref_epoch", "tess_mag", "bp_rp", "phot_g_mean_mag",
            "phot_bp_mean_mag", "phot_rp_mean_mag", "canonical_entity", "link_kind", "is_region_trigger", "in_trigger_core"]
    return m[keep], info


def candidate_rows(ledger, full_scene, tmax: float, gate_override: bool, source: str = "candidates", gaia_catalog=None):
    """Ledger rows with ``tess_mag <= tmax``, finite position, not already in ``full_scene``; ``tess_flux`` on the
    scene's own flux scale (median zero point of ``log10 f + 0.4 T`` over its stars)."""
    import pandas as pd
    from syndiff_pipeline.forward_model import scene_fit as SF

    sc0 = SF.Scene(full_scene)
    tf, tm = sc0.z["tess_flux"].astype(float), sc0.z["tess_mag"].astype(float)
    ok = np.isfinite(tf) & (tf > 0) & np.isfinite(tm)
    zp = float(np.median(np.log10(tf[ok]) + 0.4 * tm[ok]))
    if source == "assoc_r2":
        cand, _ = assoc_r2_table(ledger, gaia_catalog)
    elif source == "candidates":
        cand = pd.read_csv(ledger, dtype={"source_id": "int64"})
    else:
        raise ValueError(f"unknown neighbour source {source!r}")
    rows = cand[(cand.tess_mag <= tmax) & np.isfinite(cand.ra) & np.isfinite(cand.dec)].drop_duplicates("source_id").copy()
    rows = rows[~rows.source_id.isin(sc0.z["source_id"])].reset_index(drop=True)
    rows["tess_flux"] = 10 ** (zp - 0.4 * rows.tess_mag.to_numpy(float))
    if source == "assoc_r2":
        if gate_override:
            raise ValueError("gate_override is not used with the verified assoc_r2 ledger")
        rows["full_positive_allowed"] = True         # every row is a verified removal
        rows["removal_evidence"] = VERIFIED_NOTE
    elif gate_override:
        rows["full_positive_allowed"] = True
        rows["gate_override"] = GATE_OVERRIDE_NOTE
    return rows, zp


def support_filter(scene, rows, wcs_params, colour_file):
    """Keep the rows whose stamp (placed with ``wcs_params``) touches at least one valid pixel of ``scene``'s union."""
    prep, _ = prepare_sources(scene, rows, coeff=np.load(wcs_params)["wcs_coeff"], colour_file=colour_file)
    z = scene.z; U, S = int(z["n_union"]), int(z["stamp"]); u = union_arrays(z)
    cx = np.round(prep.model_x).astype(np.int64).to_numpy(); cy = np.round(prep.model_y).astype(np.int64).to_numpy()
    k = np.arange(S * S); xx = cx[:, None] + (k % S - S // 2); yy = cy[:, None] + (k // S - S // 2)
    lo = int(min(u["x"].min(), xx.min())) - 1; width = int(max(u["x"].max(), xx.max()) - lo) + 2
    old = u["y"][:U].astype(np.int64) * width + u["x"][:U] - lo; order = np.argsort(old); ordered = old[order]
    ask = yy * width + xx - lo; pos = np.searchsorted(ordered, ask); safe = np.clip(pos, 0, U - 1)
    on = (pos < U) & (ordered[safe] == ask); uid = np.where(on, order[safe], U)
    return rows[(u["valid"][uid].sum(1) > 0)].reset_index(drop=True)


def build(scene_dir, out_dir, *, full_scene, ledger, tmax, gate_override, wcs_params, colour_file,
          source: str = "candidates", gaia_catalog=None) -> dict:
    """Neighbour scene for ``scene_dir`` (the full scene or one fold scene of it); candidates are selected against
    ``full_scene`` so every fold sees the same candidate list."""
    from syndiff_pipeline.forward_model import scene_fit as SF

    rows, zp = candidate_rows(ledger, full_scene, tmax, gate_override, source, gaia_catalog)
    r = support_filter(SF.Scene(scene_dir), rows, wcs_params, colour_file)
    meta = write_scene(scene_dir, r, out_dir, params_file=str(wcs_params), colour_file=colour_file)
    if gate_override:
        meta["gate_override"] = GATE_OVERRIDE_NOTE
    meta["placement_wcs"] = str(wcs_params)
    meta["neighbours"] = dict(ledger=str(ledger), ledger_sha256=sha256(ledger), source=source, tmax=float(tmax),
                              gate_override=bool(gate_override),
                              gaia_catalog=None if gaia_catalog is None else str(gaia_catalog),
                              gaia_catalog_sha256=None if gaia_catalog is None else sha256(gaia_catalog))
    if source == "assoc_r2":
        meta["neighbours"]["removal_evidence"] = VERIFIED_NOTE
        meta["neighbours"]["n_added_in_trigger_core"] = int(r["in_trigger_core"].fillna(False).astype(bool).sum())
        meta["neighbours"]["n_added_triggers"] = int(r["is_region_trigger"].fillna(False).astype(bool).sum())
    (Path(out_dir) / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    return dict(scene=str(scene_dir), wcs=str(wcs_params), zp=zp, n_candidates=int(len(rows)), n_added=int(len(r)),
                n_total=meta["n_stars"], max_island=meta["max_island"], tiers=meta["tiers"])


def _pad(v, nadd):
    fill = -1 if v.dtype.kind in "iu" else (False if v.dtype == bool else np.nan)
    return np.concatenate([v, np.full((nadd,) + v.shape[1:], fill, dtype=v.dtype)])


def pad_table(table: dict, n0: int, nadd: int, source_id: np.ndarray) -> dict:
    """Pad every per-star array (leading dim ``n0``) of a folds/contract table to the augmented scene; added rows get
    -1 / False / NaN (never trainees, never held)."""
    out = {k: (_pad(v, nadd) if v.shape[:1] == (n0,) else v) for k, v in table.items()}
    out["source_id"] = source_id
    return out
