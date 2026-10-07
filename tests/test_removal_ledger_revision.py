import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing.removal_ledger import revision as rv


def ring_support(n=41, r_out=12, r_in=5, c=20):
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - c, yy - c)
    return (rr <= r_out) & (rr > r_in)


def test_hole_fill_is_identity_only():
    sup = ring_support()
    retained = np.zeros(sup.shape, bool)
    hole, nan, ident = rv.identity_core(sup, retained, np.zeros(sup.shape, bool))
    assert hole[20, 20] and not sup[20, 20]
    assert not np.any(hole & sup) and not nan.any()
    assert np.array_equal(ident, sup | hole)
    # the exact support is untouched
    assert sup.sum() == ring_support().sum()


def test_retained_finite_pixels_inside_hole_are_not_core():
    sup = ring_support()
    retained = np.zeros(sup.shape, bool)
    retained[20, 20] = True
    hole, _, ident = rv.identity_core(sup, retained, np.zeros(sup.shape, bool))
    assert not hole[20, 20] and not ident[20, 20]
    assert hole[20, 21]


def test_open_notch_is_not_a_hole():
    sup = ring_support()
    sup[19:22, 7:16] = False          # cut the ring: core is no longer enclosed
    hole, _, _ = rv.identity_core(sup, np.zeros(sup.shape, bool), np.zeros(sup.shape, bool))
    assert not hole.any()


def test_nonfinite_core_touching_support_is_included_but_bounded():
    sup = np.zeros((41, 41), bool)
    sup[10:30, 10:30] = True
    sup[:, 30:] = False
    nonfinite = np.zeros_like(sup)
    nonfinite[18:22, 30:34] = True          # touches the support, small
    nonfinite[0, 0:3] = True                # far from support, window edge
    nonfinite[35:41, 35:41] = True          # far, touches window edge
    hole, nan, ident = rv.identity_core(sup, np.zeros_like(sup), nonfinite)
    assert nan[18:22, 30:34].all()
    assert not nan[0, 0:3].any() and not nan[35:41, 35:41].any()
    # a huge adjacent strip is rejected by the area cap
    big = np.zeros_like(sup)
    big[10:30, 30:33] = True
    _, nan2, _ = rv.identity_core(sup, np.zeros_like(sup), big, nan_cap=10)
    assert not nan2.any()


def _toy(trigger_xy, other_xy=(), shape=(60, 60)):
    sup = np.zeros(shape, bool)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    rr = np.hypot(xx - 30, yy - 30)
    sup = (rr <= 14) & (rr > 5)
    regions = pd.DataFrame([dict(region_id='p:1', operation=1, reason='saturation', y0=0, x0=0, height=shape[0],
                                 width=shape[1], trigger_gaia_id='111', signed_flux_change=1000.)])
    comb = np.where(sup, 0., 5.).astype(np.float32)   # finite image everywhere, background-level 5
    comb[rr <= 5] = 0.                                 # zeroed core
    sel = np.zeros(shape, np.uint8)
    sel[rr <= 14] = 2
    rows = [dict(source_key='gaia:111', entity_key='gaia:111', catalogue='gaia_dr3', pixel_x=trigger_xy[0], pixel_y=trigger_xy[1])]
    for i, (x, y) in enumerate(other_xy):
        rows.append(dict(source_key=f'gaia:{200 + i}', entity_key=f'gaia:{200 + i}', catalogue='gaia_dr3', pixel_x=x, pixel_y=y))
    return regions, sup, comb, sel, pd.DataFrame(rows)


def _run(regions, sup, comb, sel, src, shape):
    stats, links = rv.enclosed_links(regions, lambda r: sup, comb, sel, src, shape)
    assoc = pd.DataFrame(dict(source_key=['gaia:111'], region_id=['p:1'], reason=['saturation'], centre_in_support=[False],
                              candidate_overlap_pixels=[50], support_radius_px=[300.], support_in_region_bbox_pixels=[10],
                              association_status=['possible_partial_support'], stellar_flux_change=[None]))
    a2 = rv.apply_enclosed(src, assoc, links)
    trig = rv.trigger_status(regions, src, a2, shape)
    return stats, links, a2, trig


def test_enclosed_trigger_gets_distinct_status():
    regions, sup, comb, sel, src = _toy((30., 30.), other_xy=[(32., 29.), (3., 3.)])
    stats, links, a2, trig = _run(regions, sup, comb, sel, src, sup.shape)
    row = a2[(a2.source_key == 'gaia:111')].iloc[0]
    assert row.association_status == rv.STATUS_ENCLOSED and row.original_association_status == 'possible_partial_support'
    assert row.revision == 'upgraded_possible_partial_support'
    t = trig.iloc[0]
    assert t.trigger_declared == '111' and t.trigger_identity_status == 'declared_and_centre_enclosed'
    assert t.trigger_centre_evidence == 'enclosed_core'
    # a non-trigger source in the same core gains a NEW link (false-gain bookkeeping); a far one does not
    assert (a2.source_key == 'gaia:200').sum() == 1 and a2[a2.source_key == 'gaia:200'].revision.iloc[0] == 'new_enclosed_core_link'
    assert not (a2.source_key == 'gaia:201').any()
    assert stats.identity_hole_pixels.iloc[0] > 0


def test_candidate_only_and_off_cell_triggers():
    regions, sup, comb, sel, src = _toy((50., 50.))       # inside the cell, outside the core and support
    _, _, a2, trig = _run(regions, sup, comb, sel, src, sup.shape)
    assert trig.iloc[0].trigger_identity_status == 'declared_candidate_only'
    assert a2.association_status.tolist() == ['possible_partial_support']
    regions, sup, comb, sel, src = _toy((-40., 30.))      # outside the cell
    _, _, a2, trig = _run(regions, sup, comb, sel, src, sup.shape)
    assert trig.iloc[0].trigger_identity_status == 'off_cell'
    assert trig.iloc[0].trigger_centre_evidence == 'none'


def test_exact_membership_is_not_relabelled():
    regions, sup, comb, sel, src = _toy((30., 18.))        # on the ring support itself
    stats, links, _, _ = _run(regions, sup, comb, sel, src, sup.shape)
    assert len(links) == 0


def _src(rows):
    return pd.DataFrame(rows)


def test_referential_integrity_counts():
    cand = pd.DataFrame(dict(gaia_key=['gaia:1', 'gaia:2'], ps1_entity_key=['ps1:10', 'ps1:20'], ps1_obj_id=['10', '20'],
                             status=['accepted_unique_calibrated'] * 2))
    src = _src([dict(source_key='gaia:1', entity_key='gaia:1', catalogue='gaia_dr3'),
                dict(source_key='ps1det:9', entity_key='ps1:10', catalogue='ps1_dr2_stack')])
    assoc = pd.DataFrame(dict(source_key=['gaia:1']))
    regions = pd.DataFrame(dict(trigger_gaia_id=['3']))
    before = rv.integrity(src, cand, assoc, regions)
    assert before['pairs_missing_end'] == 1 and before['accepted_missing_end'] == 1
    assert before['missing_gaia_keys'] == 2 and before['missing_ps1_entities'] == 1   # gaia:2, trigger gaia:3 ; ps1:20
    full = pd.concat([src, _src([dict(source_key='gaia:2', entity_key='gaia:2', catalogue='gaia_dr3'),
                                 dict(source_key='gaia:3', entity_key='gaia:3', catalogue='gaia_dr3'),
                                 dict(source_key='ps1det:19', entity_key='ps1:20', catalogue='ps1_dr2_stack')])])
    after = rv.integrity(full, cand, assoc, regions)
    assert after['pairs_missing_end'] == 0 and after['missing_gaia_keys'] == 0 and after['missing_ps1_entities'] == 0


def test_ps1_only_removed_centres_are_excluded_not_dropped():
    cand = pd.DataFrame(dict(gaia_key=['gaia:1', 'gaia:2'], ps1_entity_key=['ps1:10', 'ps1:11'], ps1_obj_id=['10', '11'],
                             status=['accepted_unique_calibrated', 'ambiguous']))
    base = dict(centre_status='centre_removed', pixel_x=5., pixel_y=5., ra=1., dec=2.)
    src = _src([dict(source_key='gaia:1', entity_key='gaia:1', catalogue='gaia_dr3', phot_g_mean_mag=15., phot_bp_mean_mag=15.5,
                     phot_rp_mean_mag=14.5, **base),
                dict(source_key='ps1det:a', entity_key='ps1:10', catalogue='ps1_dr2_stack', **base),     # duplicate of gaia:1 entity
                dict(source_key='ps1det:b', entity_key='ps1:11', catalogue='ps1_dr2_stack', **base),     # ambiguous: kept
                dict(source_key='ps1det:c', entity_key='ps1:99', catalogue='ps1_dr2_stack', **base),     # PS1-only: excluded
                dict(source_key='gaia:5', entity_key='gaia:5', catalogue='gaia_dr3', phot_g_mean_mag=18., phot_bp_mean_mag=18.5,
                     phot_rp_mean_mag=17.5, **{**base, 'centre_status': 'unaffected'})])
    # make accepted pairing resolve ps1:10 -> gaia:1 via the candidates
    assoc = pd.DataFrame(dict(source_key=['gaia:1', 'ps1det:a', 'ps1det:b', 'ps1det:c'], region_id=['p:1'] * 4,
                              reason=['catalog'] * 4, association_status=['centre_membership'] * 4))
    regions = pd.DataFrame(dict(region_id=['p:1'], trigger_declared=[None], trigger_centre_evidence=['none']))
    nb, ex = rv.build_neighbours(src, assoc, regions, cand, lambda s: np.ones(len(s), int), 'cell', 'F')
    assert sorted(nb.canonical_entity) == ['gaia:1', 'ps1:11']
    assert nb[nb.canonical_entity == 'gaia:1'].source_key.iloc[0] == 'gaia:1'     # Gaia row preferred, entities merged
    assert nb[nb.canonical_entity == 'gaia:1'].n_rows_merged.iloc[0] == 2
    assert list(ex.canonical_entity) == ['ps1:99'] and ex.exclusion_reason.iloc[0] == 'ps1_only_no_gaia_candidate'
    assert 'gaia:5' not in set(nb.canonical_entity) | set(ex.canonical_entity)    # not removed, never listed


def test_speckle_holes_in_porous_support_are_not_enclosed_cores():
    shape = (80, 80)
    sup = np.ones(shape, bool)
    sup[0, :] = sup[-1, :] = sup[:, 0] = sup[:, -1] = True
    sup[10:13, 10:13] = False                 # 9 px speckle hole
    sup[40:50, 40:50] = False                 # 100 px compact core
    regions = pd.DataFrame([dict(region_id='p:1', operation=1, reason='catalog', y0=0, x0=0, height=80, width=80,
                                 trigger_gaia_id='111', signed_flux_change=1.)])
    comb = np.zeros(shape, np.float32)
    sel = np.full(shape, 2, np.uint8)
    src = pd.DataFrame([dict(source_key='gaia:111', entity_key='gaia:111', catalogue='gaia_dr3', pixel_x=45., pixel_y=45.),
                        dict(source_key='gaia:2', entity_key='gaia:2', catalogue='gaia_dr3', pixel_x=11., pixel_y=11.)])
    _, links = rv.enclosed_links(regions, lambda r: sup, comb, sel, src, shape)
    assert links.source_index.tolist() == [0]
    assert links.core_component_px.iloc[0] == 100


def test_non_trigger_gaia_in_trigger_core_is_kept_and_flagged():
    cand = pd.DataFrame(dict(gaia_key=['gaia:7', 'gaia:111'], ps1_entity_key=['ps1:70', 'ps1:71'], ps1_obj_id=['70', '71'],
                             status=['ambiguous', 'ambiguous']))
    core = dict(centre_status='selected_no_finite_change', pixel_x=5., pixel_y=5., ra=1., dec=2.)
    src = _src([dict(source_key='gaia:111', entity_key='gaia:111', catalogue='gaia_dr3', phot_g_mean_mag=8., phot_bp_mean_mag=8.5,
                     phot_rp_mean_mag=7.5, **core),                                                           # the trigger
                dict(source_key='gaia:7', entity_key='gaia:7', catalogue='gaia_dr3', phot_g_mean_mag=9., phot_bp_mean_mag=9.5,
                     phot_rp_mean_mag=8.5, **core),                                                           # companion in the core
                dict(source_key='ps1det:z', entity_key='ps1:99', catalogue='ps1_dr2_stack', **core),          # PS1-only in the core
                dict(source_key='ps1det:t', entity_key='ps1:71', catalogue='ps1_dr2_stack', **core)])         # split detection of the trigger
    st = rv.STATUS_ENCLOSED
    assoc = pd.DataFrame(dict(source_key=['gaia:111', 'gaia:7', 'ps1det:z', 'ps1det:t'], region_id=['p:1'] * 4,
                              reason=['catalog'] * 4, association_status=[st] * 4))
    regions = pd.DataFrame(dict(region_id=['p:1'], trigger_declared=['111'], trigger_centre_evidence=['enclosed_core']))
    nb, ex = rv.build_neighbours(src, assoc, regions, cand, lambda s: np.ones(len(s), int), 'cell', 'F')
    kinds = dict(zip(nb.canonical_entity, nb.link_kind))
    assert kinds == {'gaia:111': 'enclosed_core_trigger', 'gaia:7': 'in_trigger_core'}
    flags = dict(zip(nb.canonical_entity, nb.in_trigger_core))
    assert flags == {'gaia:111': False, 'gaia:7': True}
    assert list(ex.canonical_entity) == []                                        # PS1 rows in a core are not listed at all
    assert 'ps1:71' not in set(nb.canonical_entity)                               # trigger's split detection not a neighbour
