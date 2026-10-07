"""Derived revision layer over published (immutable) removal ledgers.

Reads a published cell ledger and writes a SEPARATE revision product; it never
modifies, republishes or re-derives the pixel accounting (no SEP, no removal).
Three fixes to source association:

1. Saturated-core trigger identity. A trigger star's centre often lies on a
   zero/NaN core that is not in the region's exact support (the core was zeroed
   by the earlier background stage, so the catalogue operation's support has a
   hole there). For IDENTITY ONLY (never flux or deletion support) we define
   ``identity_support = binary_fill_holes(support) | enclosed nonfinite core``:
   * hole pixels (enclosed by the support within its padded bbox) that are not
     retained finite non-zero image content, and
   * nonfinite pixels of the combined image (NaN survives only where no
     operation selected the pixel) in 8-connected components that touch the
     support/hole, are smaller than ``NAN_CAP`` and do not reach the window edge.
   A source whose calibrated centre rounds into this extension gets the new
   association status ``centre_in_enclosed_core`` (distinct from
   ``centre_membership``). ``trigger_declared`` (the id the removal code used)
   is kept apart from the geometric evidence ``trigger_centre_evidence``.
2. Missing referenced sources. Sources were trimmed to +-600 px, so accepted
   pairs may reference rows absent from sources.parquet. The revision table is
   original + every row referenced by any candidate/association/region trigger,
   re-derived from the catalogue cache with the same functions as pilot.py and
   the stored per-catalogue image calibration shift.
3. Removed-neighbour list: removed centres (centre_removed + enclosed trigger)
   excluding PS1-only identities (no Gaia candidate of any status); excluded rows
   are kept in a separate table with the reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

REVISION_SCHEMA = 1
MIN_CORE_PX = 30        # smallest connected core component under a centre accepted as enclosed (see README: speckle holes <=30 px vs trigger cores median 390 px)
NAN_CAP = 5000          # largest nonfinite component accepted as an enclosed core (pixels)
WINDOW_PAD = 48         # padding of the per-region window used for hole filling / NaN components
STATUS_ENCLOSED = 'centre_in_enclosed_core'
CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PAIR_RANK = {'accepted_unique_calibrated': 3, 'ambiguous': 2, 'unconfirmed_candidate': 1}
PAIR_NAME = {3: 'accepted', 2: 'ambiguous', 1: 'unconfirmed'}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_parquet(table, path):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    table.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------- fix 1: geometry

def identity_core(support, retained_finite, nonfinite, *, nan_cap=NAN_CAP):
    """Enclosed-core pixels of one region, window-local arrays of equal shape.

    support         bool, the region's exact deletion support (placed in the window)
    retained_finite bool, pixels whose finite non-zero value survives (never a core)
    nonfinite       bool, nonfinite pixels of the image
    Returns (hole_core, nan_core, identity_support); the first two exclude ``support``.
    """
    support = np.asarray(support, bool)
    filled = ndi.binary_fill_holes(support)
    hole = filled & ~support
    hole_core = hole & ~np.asarray(retained_finite, bool)
    nan_core = np.zeros_like(support)
    nf = np.asarray(nonfinite, bool) & ~support
    if nf.any():
        lab, n = ndi.label(nf, structure=np.ones((3, 3), int))
        touch_zone = ndi.binary_dilation(support | hole_core, structure=np.ones((3, 3), int))
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        edge = np.zeros(n + 1, bool)
        for sl in (lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]):
            edge[np.unique(sl)] = True
        touching = np.zeros(n + 1, bool)
        touching[np.unique(lab[touch_zone & nf])] = True
        ok = touching & ~edge & (sizes <= nan_cap)
        ok[0] = False
        nan_core = ok[lab]
    return hole_core, nan_core, support | hole_core | nan_core


def unpack_support(geom, row):
    return np.unpackbits(geom[f"support_{int(row['operation'])}"], count=int(row['height']) * int(row['width'])
                         ).reshape(int(row['height']), int(row['width'])).astype(bool)


def centre_pixels(src, shape):
    """Same rounding as CellLedger.associate_centres; -1 where outside/unlocalized."""
    x = src.pixel_x.to_numpy(float)
    y = src.pixel_y.to_numpy(float)
    fin = np.isfinite(x) & np.isfinite(y)
    inside = fin & (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    ix = np.full(len(src), -1, int)
    iy = np.full(len(src), -1, int)
    ix[inside] = np.minimum(np.rint(x[inside]).astype(int), shape[1] - 1)
    iy[inside] = np.minimum(np.rint(y[inside]).astype(int), shape[0] - 1)
    return ix, iy, inside, fin


def enclosed_links(regions, geom_get, combined, selected_stage, src, shape, min_core_px=None):
    """Per non-background region: identity-core pixels and the sources whose centre lies in them.

    geom_get(row) -> bool support. Returns (region_stats DataFrame, links DataFrame).
    ``links`` rows: source_index, region_id, core_kind in {'hole','nan'}.
    """
    min_core_px = MIN_CORE_PX if min_core_px is None else min_core_px
    ix, iy, inside, _ = centre_pixels(src, shape)
    H, W = shape
    stats, links = [], []
    for row in regions.to_dict('records'):
        if row['reason'] == 'background':
            continue
        sup = geom_get(row)
        y0, x0, h, w = int(row['y0']), int(row['x0']), int(row['height']), int(row['width'])
        wy0, wx0 = max(0, y0 - WINDOW_PAD), max(0, x0 - WINDOW_PAD)
        wy1, wx1 = min(H, y0 + h + WINDOW_PAD), min(W, x0 + w + WINDOW_PAD)
        win_sup = np.zeros((wy1 - wy0, wx1 - wx0), bool)
        win_sup[y0 - wy0:y0 - wy0 + h, x0 - wx0:x0 - wx0 + w] = sup
        comb = combined[wy0:wy1, wx0:wx1]
        sel = selected_stage[wy0:wy1, wx0:wx1]
        retained = (sel == 0) & np.isfinite(comb) & (comb != 0)
        hole_core, nan_core, ident = identity_core(win_sup, retained, ~np.isfinite(comb))
        stats.append(dict(region_id=row['region_id'], identity_hole_pixels=int(hole_core.sum()),
                          identity_nan_pixels=int(nan_core.sum()), support_pixels_exact=int(sup.sum())))
        if not (hole_core.any() or nan_core.any()):
            continue
        cand = np.nonzero(inside & (ix >= wx0) & (ix < wx1) & (iy >= wy0) & (iy < wy1))[0]
        if not len(cand):
            continue
        # size of the connected core component under each centre: a saturated core is one compact blob, whereas
        # a porous merged support has hundreds of 1-10 px holes that must not count as "enclosed"
        lab, _ = ndi.label(hole_core | nan_core)
        sizes = np.bincount(lab.ravel())
        for si in cand:
            yy, xx = iy[si] - wy0, ix[si] - wx0
            if win_sup[yy, xx]:
                continue
            k = lab[yy, xx]
            if k and sizes[k] >= min_core_px:
                links.append((int(si), row['region_id'], 'hole' if hole_core[yy, xx] else 'nan', int(sizes[k])))
    return (pd.DataFrame(stats, columns=['region_id', 'identity_hole_pixels', 'identity_nan_pixels', 'support_pixels_exact']),
            pd.DataFrame(links, columns=['source_index', 'region_id', 'core_kind', 'core_component_px']))


def apply_enclosed(sources, associations, links):
    """Return associations_r2 and per-source flags. Original rows are never dropped."""
    a = associations.copy()
    a['original_association_status'] = a['association_status']
    a['centre_in_identity_support'] = a['centre_in_support'].fillna(False).astype(bool)
    a['identity_core_kind'] = None
    a['core_component_px'] = np.nan
    a['revision'] = 'original'
    if len(links) == 0:
        return a
    ln = links.copy()
    ln['source_key'] = sources.source_key.to_numpy()[ln.source_index.to_numpy()]
    ln = ln.drop(columns='source_index')
    idx = pd.MultiIndex.from_frame(a[['source_key', 'region_id']])
    key = pd.MultiIndex.from_frame(ln[['source_key', 'region_id']])
    hit = idx.isin(key)
    kind = ln.set_index(['source_key', 'region_id']).core_kind
    csz = ln.set_index(['source_key', 'region_id']).core_component_px
    # a source can sit in the enclosed core while not being in the exact support (never the reverse)
    upgrade = hit & ~a['centre_in_support'].fillna(False).to_numpy(bool)
    a.loc[upgrade, 'association_status'] = STATUS_ENCLOSED
    a.loc[upgrade, 'centre_in_identity_support'] = True
    a.loc[upgrade, 'identity_core_kind'] = kind.reindex(idx[upgrade]).to_numpy()
    a.loc[upgrade, 'core_component_px'] = csz.reindex(idx[upgrade]).to_numpy()
    a.loc[upgrade, 'revision'] = 'upgraded_possible_partial_support'
    new = ln[~key.isin(idx)]
    if len(new):
        reason = a.drop_duplicates('region_id').set_index('region_id').reason
        rows = pd.DataFrame(dict(source_key=new.source_key.to_numpy(), region_id=new.region_id.to_numpy()))
        rows['reason'] = rows.region_id.map(reason).fillna('catalog')
        rows['centre_in_support'] = False
        rows['candidate_overlap_pixels'] = 0
        rows['support_radius_px'] = np.nan
        rows['support_in_region_bbox_pixels'] = None
        rows['association_status'] = STATUS_ENCLOSED
        rows['stellar_flux_change'] = None
        rows['original_association_status'] = None
        rows['centre_in_identity_support'] = True
        rows['identity_core_kind'] = new.core_kind.to_numpy()
        rows['core_component_px'] = new.core_component_px.to_numpy()
        rows['revision'] = 'new_enclosed_core_link'
        a = pd.concat([a, rows], ignore_index=True)
    return a


def trigger_status(regions, sources, associations_r2, shape):
    """trigger_declared stays the removal code's id; evidence is geometric and separate."""
    pos = sources.drop_duplicates('source_key').set_index('source_key')[['pixel_x', 'pixel_y']]
    link = associations_r2[associations_r2.association_status.isin(['centre_membership', STATUS_ENCLOSED])]
    kinds = {}
    for r in link.itertuples():
        k = 'support' if r.association_status == 'centre_membership' else 'enclosed_core'
        kinds.setdefault((r.source_key, r.region_id), set()).add(k)
    out = []
    H, W = shape
    for r in regions.to_dict('records'):
        tid = r.get('trigger_gaia_id')
        has = isinstance(tid, str) and tid not in ('', 'None')
        rec = dict(region_id=r['region_id'], trigger_declared=tid if has else None,
                   trigger_centre_evidence='none', trigger_in_cell=None)
        if not has:
            rec['trigger_identity_status'] = 'no_declared_trigger'
        else:
            key = 'gaia:' + tid
            ev = kinds.get((key, r['region_id']), set())
            known = key in pos.index
            x = pos.pixel_x.get(key, np.nan) if known else np.nan
            y = pos.pixel_y.get(key, np.nan) if known else np.nan
            in_cell = bool(known and np.isfinite(x) and np.isfinite(y) and 0 <= x < W and 0 <= y < H)
            rec['trigger_in_cell'] = in_cell if known else None
            if 'support' in ev:
                rec['trigger_centre_evidence'] = 'support'
            elif 'enclosed_core' in ev:
                rec['trigger_centre_evidence'] = 'enclosed_core'
            if ev:
                rec['trigger_identity_status'] = 'declared_and_centre_enclosed'
            elif known and not in_cell:
                rec['trigger_identity_status'] = 'off_cell'
            elif not known:
                rec['trigger_identity_status'] = 'declared_trigger_not_in_sources'
            else:
                rec['trigger_identity_status'] = 'declared_candidate_only'
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- fix 2: referenced rows

def referenced_keys(candidates, associations, regions):
    gaia = set(candidates.gaia_key) if len(candidates) else set()
    ps1e = set(candidates.ps1_entity_key) if len(candidates) else set()
    keys = set(associations.source_key) if len(associations) else set()
    trig = {'gaia:' + t for t in regions.trigger_gaia_id.dropna().astype(str) if t not in ('', 'None')}
    return gaia | trig, ps1e, keys


def integrity(sources, candidates, associations, regions):
    """Counts of referenced identities absent from ``sources`` (0 == referentially complete)."""
    gk, pe, ak = referenced_keys(candidates, associations, regions)
    have_s = set(sources.source_key)
    have_e = set(sources.entity_key[sources.catalogue == 'ps1_dr2_stack']) if len(sources) else set()
    pairs_total = len(candidates)
    miss_g = candidates.gaia_key.map(lambda k: k not in have_s) if pairs_total else pd.Series(dtype=bool)
    miss_p = candidates.ps1_entity_key.map(lambda k: k not in have_e) if pairs_total else pd.Series(dtype=bool)
    acc = candidates.status == 'accepted_unique_calibrated' if pairs_total else pd.Series(dtype=bool)
    return dict(pairs=int(pairs_total), pairs_missing_end=int((miss_g | miss_p).sum()) if pairs_total else 0,
                accepted_pairs=int(acc.sum()) if pairs_total else 0,
                accepted_missing_end=int(((miss_g | miss_p) & acc).sum()) if pairs_total else 0,
                missing_gaia_keys=len(gk - have_s), missing_ps1_entities=len(pe - have_e),
                missing_association_sources=len(ak - have_s))


def derive_catalogue_rows(manifest, directory, regions_unused=None):
    """Re-derive the FULL per-cell catalogue table exactly as pilot.py did (before margin trim)."""
    from astropy.io import fits
    from astropy.time import Time
    from astropy.wcs import WCS
    from .matching import gaia_sources, ps1_sources, match_gaia_ps1, attach_entities
    heads = json.loads((Path(directory) / 'headers.json').read_text())
    header = fits.Header.fromstring(next(iter(heads.values())))
    wcs = WCS(header)
    cats = {c['catalogue']: Path(c['path']) / 'catalogue.parquet' for c in manifest['catalogues']}
    g = pd.read_parquet(cats['gaia'])
    ps = pd.read_parquet(cats['ps1'])
    epoch = float(Time(header['MJD-OBS'], format='mjd').jyear) if 'MJD-OBS' in header else None
    gs = gaia_sources(g, wcs, epoch_year=epoch)
    pss = ps1_sources(ps, wcs)
    return gs, pss


def pull_missing(sources, candidates, associations, regions, manifest, directory, shape):
    """Rows referenced but absent from ``sources``, derived from the catalogue cache; plus a rederivation check."""
    from . import matching
    from .. import band_utils as bu
    gk, pe, ak = referenced_keys(candidates, associations, regions)
    have = set(sources.source_key)
    have_e = set(sources.entity_key[sources.catalogue == 'ps1_dr2_stack'])
    need_g = {k for k in (gk | {k for k in ak if k.startswith('gaia:')}) if k not in have}
    need_pe = {k for k in pe if k not in have_e}
    selfcheck = dict(rederived_rows=0, max_abs_dpix=None)
    if not need_g and not need_pe:
        return sources.iloc[0:0].copy(), selfcheck
    gs, pss = derive_catalogue_rows(manifest, directory)
    full = pd.concat([gs, pss], ignore_index=True)
    # same entity attachment as pilot.py (accepted aliases), using the stored candidate table
    full = matching.attach_entities(full, candidates)
    rep = manifest['metadata'].get('image_calibration') or {}
    full['catalogue_pixel_x'] = full.pixel_x
    full['catalogue_pixel_y'] = full.pixel_y
    full['image_calibration_status'] = 'insufficient_calibrators'
    full['image_position_scatter_px'] = np.nan
    for cat in ('gaia_dr3', 'ps1_dr2_stack'):
        r = rep.get(cat)
        if not r:
            continue
        sel = full.catalogue == cat
        full.loc[sel, 'image_calibration_status'] = r['status']
        if r['status'] == 'calibrated':
            full.loc[sel, 'pixel_x'] += r['dx_px']
            full.loc[sel, 'pixel_y'] += r['dy_px']
            full.loc[sel, 'image_position_scatter_px'] = float(np.hypot(r['sigma_x_px'], r['sigma_y_px']))
    full['source_type'] = np.where(full.catalogue == 'gaia_dr3', 'catalogue_source', 'stack_detection')
    full['accounting_radius_px'] = 5.
    full['support_definition'] = '5-pixel candidate aperture; unmeasured exterior PSF remains unknown'
    is_g = full.catalogue == 'gaia_dr3'
    t = bu.compute_tess_mag(full.loc[is_g, 'phot_g_mean_mag'].to_numpy(float),
                            full.loc[is_g, 'phot_bp_mean_mag'].to_numpy(float),
                            full.loc[is_g, 'phot_rp_mean_mag'].to_numpy(float))
    bright = full.index[is_g][t < 13]
    full.loc[bright, 'accounting_radius_px'] = bu.star_footprint_radius(t[t < 13])
    full.loc[bright, 'support_definition'] = 'historical bright-star halo search radius; association candidate only'
    # rederivation check against the stored rows (shows the column derivation is identical)
    m = sources[['source_key', 'pixel_x', 'pixel_y']].merge(full[['source_key', 'pixel_x', 'pixel_y']], on='source_key',
                                                          suffixes=('', '_r'))
    if len(m):
        d = np.nanmax(np.abs(np.r_[m.pixel_x - m.pixel_x_r, m.pixel_y - m.pixel_y_r]))
        selfcheck = dict(rederived_rows=int(len(m)), max_abs_dpix=float(d))
    pick = full[(~full.source_key.isin(have)) &
                (full.source_key.isin(need_g) |
                 ((full.catalogue == 'ps1_dr2_stack') & full.entity_key.isin(need_pe)))].copy()
    # Gaia rows: only identity-bearing ones are in need_g; entity of an accepted PS1 pair is its Gaia key
    pick['catalogue_coordinate_role'] = pick['coordinate_role']
    pick['coordinate_role'] = 'pair_reference_outside_margin'
    ix, iy, inside, fin = centre_pixels(pick, shape)
    pick['centre_status'] = np.where(inside, 'in_cell_not_classified', np.where(fin, 'outside', 'unlocalized'))
    pick['centre_changed_stage'] = np.uint8(0)
    pick['support_status'] = 'candidate_circle'
    cols = [c for c in sources.columns if c in pick.columns]
    pick = pick[cols + [c for c in ('catalogue_coordinate_role',) if c not in cols]]
    return pick.reset_index(drop=True), selfcheck


def classify_added_centres(added, selected_stage, changed_stage, shape):
    """Added rows that fall in the cell get the same centre_status rule as the original table."""
    ix, iy, inside, _ = centre_pixels(added, shape)
    if inside.any():
        st = selected_stage[iy[inside], ix[inside]]
        ch = changed_stage[iy[inside], ix[inside]]
        status = np.where(np.isin(ch, (2, 3)), 'centre_removed', np.where(ch == 1, 'background_zeroed',
                          np.where(st > 0, 'selected_no_finite_change', 'unaffected')))
        added.loc[inside, 'centre_status'] = status
        added.loc[inside, 'centre_changed_stage'] = ch.astype(np.uint8)
    return added


# ---------------------------------------------------------------- fix 3: neighbours

def pairing_tables(candidates):
    c = candidates.copy()
    c['rank'] = c.status.map(PAIR_RANK)
    gstate = c.groupby('gaia_key')['rank'].max().map(PAIR_NAME)
    pstate = c.groupby('ps1_entity_key')['rank'].max().map(PAIR_NAME)
    acc = c[c.status == 'accepted_unique_calibrated']
    ids = c.groupby('ps1_entity_key').gaia_key.agg(lambda s: ';'.join(sorted(set(k[5:] for k in s))))
    return gstate, pstate, acc.set_index('ps1_entity_key').gaia_key.to_dict(), ids


def build_neighbours(sources, associations_r2, regions_r2, candidates, changed_operation_at, cell, field):
    """Removed-neighbour list for one cell and the PS1-only exclusions (row level, deduplicated by entity)."""
    gstate, pstate, gaia_of_ps1, gaia_ids_of_ps1 = pairing_tables(candidates) if len(candidates) else (
        pd.Series(dtype=object), pd.Series(dtype=object), {}, pd.Series(dtype=object))
    s = sources[~sources.catalogue.isin(['image_component'])].copy().reset_index(drop=True)
    trig = regions_r2[regions_r2.trigger_declared.notna()]
    enclosed_trigger_keys = set()
    enc = associations_r2[associations_r2.association_status == STATUS_ENCLOSED]
    trig_pairs = {('gaia:' + str(r.trigger_declared), r.region_id) for r in trig.itertuples()
                  if r.trigger_centre_evidence == 'enclosed_core'}
    for r in enc.itertuples():
        if (r.source_key, r.region_id) in trig_pairs:
            enclosed_trigger_keys.add(r.source_key)
    # Non-trigger sources whose centre lies in a trigger's enclosed (zeroed) core: their light was removed with the
    # trigger. Kept as real stars (e.g. binaries / close companions), flagged in_trigger_core; PS1-only ones are still
    # excluded below by the PS1-only rule.
    enclosed_any_keys = set(enc.source_key)
    s['link_kind'] = np.where(s.centre_status == 'centre_removed', 'centre_removed',
                              np.where(s.source_key.isin(enclosed_trigger_keys), 'enclosed_core_trigger',
                                       np.where(s.source_key.isin(enclosed_any_keys), 'in_trigger_core', None)))
    s = s[s.link_kind.notna()].copy()
    s['in_trigger_core'] = s.source_key.isin(enclosed_any_keys) & ~s.source_key.isin(enclosed_trigger_keys)
    if not len(s):
        return s, s.copy()
    is_g = s.catalogue == 'gaia_dr3'
    s['gaia_id'] = np.where(is_g, s.source_key.str[5:], None)
    s['ps1_entity'] = np.where(~is_g, s.entity_key, None)
    s['pair_state'] = np.where(is_g, s.source_key.map(gstate), s.entity_key.map(pstate))
    s['pair_state'] = s.pair_state.where(pd.notna(s.pair_state), np.where(is_g, 'gaia_only', 'ps1_only'))
    acc_gaia = s.entity_key.map(gaia_of_ps1)
    s.loc[~is_g, "gaia_id"] = acc_gaia[~is_g].map(lambda k: k[5:] if isinstance(k, str) else None)
    s['gaia_candidate_ids'] = np.where(is_g, s.gaia_id, s.entity_key.map(gaia_ids_of_ps1))
    ps1_only = (~is_g) & (s.pair_state == 'ps1_only')
    # region of the removed pixel: owner of the centre pixel when known, else first centre/enclosed link
    owner = changed_operation_at(s)
    link = associations_r2[associations_r2.association_status.isin(['centre_membership', STATUS_ENCLOSED]) &
                           (associations_r2.reason != 'background')]
    first = link.drop_duplicates('source_key').set_index('source_key')
    prefix = regions_r2.region_id.iloc[0].rsplit(':', 1)[0] if len(regions_r2) else None
    s['region_id'] = [f'{prefix}:{int(o)}' if o and o > 0 else first.region_id.get(k) for k, o in zip(s.source_key, owner)]
    s['association_status'] = [first.association_status.get(k) if k in first.index else None for k in s.source_key]
    n_links = link.groupby('source_key').region_id.nunique()
    s['n_regions_linked'] = s.source_key.map(n_links).fillna(0).astype(int)
    rt = regions_r2.set_index('region_id')
    s['region_trigger_declared'] = s.region_id.map(rt.trigger_declared)
    s['is_region_trigger'] = (s.gaia_id.notna() & (s.gaia_id.astype(object) == s.region_trigger_declared))
    s['cell'] = cell
    s['field'] = field
    from .. import band_utils as bu
    t = np.full(len(s), np.nan)
    if is_g.any():
        t[is_g.to_numpy()] = bu.compute_tess_mag(pd.to_numeric(s.loc[is_g, 'phot_g_mean_mag']).to_numpy(float),
                                                 pd.to_numeric(s.loc[is_g, 'phot_bp_mean_mag']).to_numpy(float),
                                                 pd.to_numeric(s.loc[is_g, 'phot_rp_mean_mag']).to_numpy(float))
    s['tess_mag'] = t
    s['canonical_entity'] = np.where(s.gaia_id.notna(), 'gaia:' + s.gaia_id.astype(str), s.entity_key)
    # dedupe by entity within the cell: prefer the Gaia row, then a removed-centre row over a trigger-only row
    s['_pref'] = (~is_g).astype(int) * 2 + (s.link_kind != 'centre_removed').astype(int)
    s = s.sort_values(['canonical_entity', '_pref', 'source_key'])
    s['n_rows_merged'] = s.groupby('canonical_entity').source_key.transform('size')
    s = s.drop_duplicates('canonical_entity').drop(columns='_pref')
    s['_ps1only'] = s.gaia_id.isna() & (s.pair_state == 'ps1_only')
    keep_cols = ['field', 'cell', 'canonical_entity', 'gaia_id', 'gaia_candidate_ids', 'ps1_entity', 'source_key', 'catalogue',
                 'ra', 'dec', 'pixel_x', 'pixel_y', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'tess_mag',
                 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag', 'centre_status', 'link_kind', 'region_id', 'association_status',
                 'n_regions_linked', 'is_region_trigger', 'in_trigger_core', 'pair_state', 'identity_status', 'coordinate_role', 'n_rows_merged']
    keep_cols = [c for c in keep_cols if c in s.columns]
    out = s[~s._ps1only][keep_cols].reset_index(drop=True)
    ex = s[s._ps1only][keep_cols].copy().reset_index(drop=True)
    ex['exclusion_reason'] = 'ps1_only_no_gaia_candidate'
    return out, ex


# ---------------------------------------------------------------- cell driver

def ledger_path(root, cell, fingerprint):
    return json.loads((Path(root) / 'cell_versions' / cell / fingerprint / 'result.json').read_text())['ledger']


def process_cell(root, field, rec, out, force=False):
    start = time.monotonic()
    cell = rec['cell']
    L = Path(ledger_path(root, cell, rec['fingerprint']))
    dest = Path(out) / field / 'cells' / cell
    mpath = dest / 'manifest.json'
    parent_sha = sha256_file(L / 'manifest.json')
    if mpath.exists() and not force:
        m = json.loads(mpath.read_text())
        if m.get('parent_manifest_sha256') == parent_sha and m.get('code_sha256') == CODE_SHA:
            return m['summary']
    with np.load(Path(rec['directory']) / 'arrays.npz') as z:
        combined = z['combined_image']
    return revise_ledger(L, combined, dest, field=field, cell=cell, catalogue_directory=rec['directory'],
                         start=start)


def revise_ledger(L, combined, dest, *, field, cell, catalogue_directory=None, start=None):
    """Revise one published ledger ``L`` given its cell's stored (post-removal) ``combined`` image.

    Writes the revision tables and ``manifest.json`` into ``dest`` and returns the summary. Fix 2 (rows referenced
    but trimmed from ``sources``) needs the PS1-stack candidate table and the cell's catalogue cache
    (``catalogue_directory``); an inline ``ps1_process`` ledger is Gaia-only, so it has neither and fix 2 is skipped
    (its associations reference only rows already in ``sources``). Fixes 1 and 3 always run.
    """
    start = time.monotonic() if start is None else start
    L = Path(L)
    dest = Path(dest)
    parent_sha = sha256_file(L / 'manifest.json')
    dest.mkdir(parents=True, exist_ok=True)
    mpath = dest / 'manifest.json'
    pm = json.loads((L / 'manifest.json').read_text())
    shape = tuple(pm['shape'])
    sources = pd.read_parquet(L / 'sources.parquet')
    assoc = pd.read_parquet(L / 'associations.parquet')
    regions = pd.read_parquet(L / 'regions.parquet')
    cand_path = L / 'gaia_ps1_candidates.parquet'
    have_cand = cand_path.is_file()
    cand = pd.read_parquet(cand_path) if have_cand else pd.DataFrame(
        {'gaia_key': pd.Series(dtype=object), 'ps1_entity_key': pd.Series(dtype=object),
         'status': pd.Series(dtype=object)})
    before_integrity = integrity(sources, cand, assoc, regions)
    with np.load(L / 'geometry.npz') as geom:
        selected = geom['selected_stage']
        changed = geom['changed_stage']
        owner_map = geom['changed_operation']
        supports = {int(r.operation): unpack_support(geom, r._asdict()) for r in regions.itertuples()
                    if r.reason != 'background'}
    # fix 2
    if have_cand and catalogue_directory is not None:
        added, selfcheck = pull_missing(sources, cand, assoc, regions, pm, catalogue_directory, shape)
        added = classify_added_centres(added, selected, changed, shape)
    else:
        added, selfcheck = sources.iloc[0:0].copy(), None
    sources['row_origin'] = 'original'
    added['row_origin'] = 'pair_reference_outside_margin'
    sources_r2 = pd.concat([sources, added], ignore_index=True)
    after_integrity = integrity(sources_r2, cand, assoc, regions)
    # fix 1
    stats, links = enclosed_links(regions, lambda r: supports[int(r['operation'])], combined, selected, sources_r2, shape)
    assoc_r2 = apply_enclosed(sources_r2, assoc, links)
    # new links / status for added rows inside the cell are included above (sources_r2 passed)
    link_src = links.assign(source_key=sources_r2.source_key.to_numpy()[links.source_index.to_numpy()]) if len(links) else links
    regions_r2 = regions.merge(stats, on='region_id', how='left')
    trig = trigger_status(regions, sources_r2, assoc_r2, shape)
    regions_r2 = regions_r2.merge(trig, on='region_id', how='left')
    enclosed_keys = set(link_src.source_key) if len(links) else set()
    sources_r2['identity_core_link'] = sources_r2.source_key.isin(enclosed_keys)

    def owner_at(s):
        ix, iy, inside, _ = centre_pixels(s, shape)
        o = np.zeros(len(s), int)
        o[inside] = owner_map[iy[inside], ix[inside]]
        return o
    nb, ex = build_neighbours(sources_r2, assoc_r2, regions_r2, cand, owner_at, cell, field)
    # outputs
    files = {}
    for name, tab in (('regions_r2', regions_r2), ('associations_r2', assoc_r2), ('sources_r2', sources_r2),
                      ('neighbours', nb), ('excluded_ps1_only', ex)):
        atomic_parquet(tab, dest / f'{name}.parquet')
        files[f'{name}.parquet'] = sha256_file(dest / f'{name}.parquet')
    nz = regions[(regions.reason != 'background') & (regions.signed_flux_change != 0)]
    rr = regions_r2[regions_r2.region_id.isin(nz.region_id)]
    flux = rr.set_index('region_id').signed_flux_change
    summary = dict(field=field, cell=cell,
        n_regions_nonbg=int((regions.reason != 'background').sum()), n_regions_nonzero=int(len(nz)),
        flux_nonzero=float(nz.signed_flux_change.sum()),
        trigger_status_counts=trig.trigger_identity_status.value_counts().to_dict(),
        nonzero_trigger_status={k: dict(n=int(v), flux=float(flux[rr.set_index('region_id').trigger_identity_status == k].sum()))
                                for k, v in rr.trigger_identity_status.value_counts().items()},
        nonzero_evidence={k: dict(n=int(v), flux=float(flux[rr.set_index('region_id').trigger_centre_evidence == k].sum()))
                          for k, v in rr.trigger_centre_evidence.value_counts().items()},
        enclosed_links=int(len(links)),
        enclosed_links_trigger=int(((assoc_r2.association_status == STATUS_ENCLOSED) &
                                    assoc_r2.set_index(['source_key', 'region_id']).index.isin(
                                        [('gaia:' + str(r.trigger_declared), r.region_id) for r in regions_r2.itertuples()
                                         if isinstance(r.trigger_declared, str)])).sum()),
        integrity_before=before_integrity, integrity_after=after_integrity, added_rows=int(len(added)),
        added_rows_in_cell=int((added.centre_status.isin(['centre_removed', 'background_zeroed', 'unaffected',
                                                          'selected_no_finite_change'])).sum()),
        neighbours=int(len(nb)), excluded_ps1_only=int(len(ex)), rederivation=selfcheck,
        seconds=time.monotonic() - start)
    manifest = dict(schema=REVISION_SCHEMA, kind='ps1_removal_ledger_assoc_revision', field=field, cell=cell,
        parent_ledger=str(L), parent_manifest_sha256=parent_sha, parent_fingerprint=pm['fingerprint'],
        parent_source_digest=pm.get('sources_digest'), code_sha256=CODE_SHA, nan_cap=NAN_CAP, window_pad=WINDOW_PAD,
        identity_support_definition=('binary_fill_holes(support) minus retained finite non-zero pixels, plus nonfinite '
                                     'combined-image components touching the support/hole (<=nan_cap px, not reaching '
                                     'the window edge); identity only, never flux or deletion support'),
        files=files, summary=summary)
    atomic_json(manifest, mpath)
    return summary


def _job(t):
    try:
        return process_cell(*t)
    except Exception:
        return dict(cell=t[2]['cell'], error=traceback.format_exc()[-1500:])


# ---------------------------------------------------------------- field aggregation

def aggregate_field(out, field, cells):
    base = Path(out) / field
    nbs, exs, sums = [], [], []
    for c in cells:
        d = base / 'cells' / c
        if not (d / 'manifest.json').exists():
            continue
        nbs.append(pd.read_parquet(d / 'neighbours.parquet'))
        exs.append(pd.read_parquet(d / 'excluded_ps1_only.parquet'))
        sums.append(json.loads((d / 'manifest.json').read_text())['summary'])
    nb = pd.concat([n for n in nbs if len(n)], ignore_index=True) if any(len(n) for n in nbs) else pd.DataFrame()
    ex = pd.concat([e for e in exs if len(e)], ignore_index=True) if any(len(e) for e in exs) else pd.DataFrame()
    dims = {}
    for c in cells:
        mp = base / 'cells' / c / 'manifest.json'
        if mp.exists():
            dims[c] = json.loads(Path(json.loads(mp.read_text())['parent_ledger'] + '/manifest.json').read_text())['shape']

    def edge_dist(df):
        if not len(df):
            return df
        H = df.cell.map(lambda c: dims[c][0]).to_numpy()
        W = df.cell.map(lambda c: dims[c][1]).to_numpy()
        x, y = df.pixel_x.to_numpy(float), df.pixel_y.to_numpy(float)
        df = df.copy()
        df['edge_distance_px'] = np.minimum.reduce([x, W - 1 - x, y, H - 1 - y])
        return df
    nb_all = edge_dist(nb)
    ex_all = edge_dist(ex)
    n_rows_cell = len(nb_all)
    if len(nb_all):
        # a star seen from several cells (cells overlap) is listed once, in the cell where it is most interior
        nb_all['n_cells_listed'] = nb_all.groupby('canonical_entity').cell.transform('nunique')
        nb_all = nb_all.sort_values(['canonical_entity', 'edge_distance_px', 'cell'], ascending=[True, False, True])
        nb_all = nb_all.drop_duplicates('canonical_entity').reset_index(drop=True)
    if len(ex_all):
        ex_all['n_cells_listed'] = ex_all.groupby('canonical_entity').cell.transform('nunique')
        ex_all = ex_all.sort_values(['canonical_entity', 'edge_distance_px', 'cell'], ascending=[True, False, True])
        ex_all = ex_all.drop_duplicates('canonical_entity').reset_index(drop=True)
    atomic_parquet(nb_all, base / 'neighbours.parquet')
    atomic_parquet(ex_all, base / 'excluded_ps1_only.parquet')
    summary = dict(field=field, cells=len(sums), code_sha256=CODE_SHA,
                   neighbour_rows_per_cell=int(n_rows_cell), neighbours_unique=int(len(nb_all)),
                   excluded_ps1_only_unique=int(len(ex_all)),
                   neighbour_link_kind=nb_all.link_kind.value_counts().to_dict() if len(nb_all) else {},
                   neighbour_pair_state=nb_all.pair_state.value_counts().to_dict() if len(nb_all) else {},
                   neighbour_catalogue=nb_all.catalogue.value_counts().to_dict() if len(nb_all) else {},
                   errors=0)
    atomic_json(summary, base / 'summary.json')
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', required=True, type=Path, help='campaign root holding inventory/ and cell_versions/')
    ap.add_argument('--field', required=True)
    ap.add_argument('--cells', nargs='*')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--no-aggregate', action='store_true')
    a = ap.parse_args(argv)
    warnings.filterwarnings('ignore')
    inv = json.loads((a.root / 'inventory' / f'{a.field}.json').read_text())['cells']
    if a.cells:
        inv = [r for r in inv if r['cell'] in set(a.cells)]
    if a.limit:
        inv = inv[:a.limit]
    tasks = [(str(a.root), a.field, r, str(a.out), a.force) for r in inv]
    errors = []
    n = 0
    if a.workers <= 1:
        results = map(_job, tasks)
    else:
        pool = Pool(a.workers, maxtasksperchild=8)
        results = pool.imap_unordered(_job, tasks)
    for r in results:
        n += 1
        if 'error' in r:
            errors.append(r)
            print('FAIL', r['cell'], r['error'][-400:], flush=True)
        elif n % 25 == 0:
            print(n, len(tasks), r['cell'], f"{r['seconds']:.1f}s", flush=True)
    if not a.no_aggregate:
        s = aggregate_field(a.out, a.field, [r['cell'] for r in inv])
        s['errors'] = len(errors)
        atomic_json(s, a.out / a.field / 'summary.json')
        print(json.dumps(s, indent=1), flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
