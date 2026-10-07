"""End-to-end: ``revision.revise_ledger`` on a Gaia-only (inline ``ps1_process``) ledger with no PS1 candidates."""
import json

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing.removal_ledger import revision
from syndiff_pipeline.template_creation.processing.removal_ledger.cell import CellLedger


def _gaia(rows):
    df = pd.DataFrame(rows)
    df['catalogue'] = 'gaia_dr3'
    df['entity_key'] = df.source_key
    df['identity_status'] = 'gaia'
    for b in ('phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'):
        df[b] = df.get(b, 15.0)
    return df


def test_revise_ledger_inline_gaia_only(tmp_path):
    shape = (80, 80)
    raw = np.full(shape, 1.0)
    yy, xx = np.mgrid[:80, :80]
    r = np.hypot(xx - 40, yy - 40)
    core = r <= 4                               # saturated NaN core (49 px >= MIN_CORE_PX) of the trigger
    raw[core] = np.nan
    ring = (r > 4) & (r <= 10)                  # trigger footprint removed around the core
    faint = (np.abs(xx - 65) <= 2) & (np.abs(yy - 15) <= 2)   # a removed faint star
    ledger = CellLedger(raw, dict(cell='skycell.0001.001', combined_fingerprint='fp'))
    ledger.observe(raw, ring, reason='catalog', component=1, trigger=dict(source_id=111, ra=0.0, dec=0.0))
    ledger.observe(raw, faint, reason='catalog', component=2)
    result = raw.copy()
    result[ring | faint] = 0.0
    ledger.finish(raw, result)
    src = _gaia([dict(source_key='gaia:111', pixel_x=40.0, pixel_y=40.0, phot_g_mean_mag=8.0,
                      phot_bp_mean_mag=8.3, phot_rp_mean_mag=7.6),
                 dict(source_key='gaia:222', pixel_x=65.0, pixel_y=15.0),
                 dict(source_key='gaia:333', pixel_x=20.0, pixel_y=70.0)])
    src, links = ledger.associate_centres(src, support_radius_px=5.0)
    L = ledger.publish(tmp_path / 'ledger', sources=src, associations=links)
    assert not (L / 'gaia_ps1_candidates.parquet').exists()

    dest = tmp_path / 'assoc_r2'
    summary = revision.revise_ledger(L, result, dest, field='P', cell='skycell.0001.001')

    nb = pd.read_parquet(dest / 'neighbours.parquet')
    kinds = dict(zip(nb.gaia_id.astype(str), nb.link_kind))
    assert kinds == {'111': 'enclosed_core_trigger', '222': 'centre_removed'}
    assert len(pd.read_parquet(dest / 'excluded_ps1_only.parquet')) == 0
    regions = pd.read_parquet(dest / 'regions_r2.parquet').set_index('operation')
    assert regions.loc[1, 'trigger_centre_evidence'] == 'enclosed_core'
    assert summary['rederivation'] is None and summary['added_rows'] == 0
    m = json.loads((dest / 'manifest.json').read_text())
    assert m['parent_ledger'] == str(L)
