"""Read-only historical backfill CLI; writes only the explicitly chosen output."""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

from .cell import replay_cell, file_digest
from .catalogues import cell_query_geometry, fetch_ps1_cone, fetch_gaia_box
from .matching import gaia_sources, ps1_sources, match_gaia_ps1, attach_entities
from .. import band_utils as bu

warnings.filterwarnings('ignore',category=FITSFixedWarning)
RUNS=Path('/astro/armin/koji/syndiff/dev_runs')
PAPER=RUNS/'paper_dataset_20261001'
TRACE=RUNS/'epsf_stage_trace_20261002/fields'
AUDIT=RUNS/'ps1_removal_audit_20261002'
ATLAS=RUNS/'epsf_atlas_20261001/dip'


def inventory(out):
    records=[]
    for field in ['C4','F2','F1','S22','C1']:
        ctx=json.loads((TRACE/field/'inputs.json').read_text())
        inv=json.loads((TRACE/field/'tables/cell_inventory.json').read_text())
        rows=[]
        for row in inv:
            p=Path(row['removed_file']).parent
            meta=json.loads((p/'_provenance.json').read_text())
            if meta['fingerprint']!=row['fingerprint']:raise ValueError('Inventory fingerprint mismatch')
            cache=ATLAS/'stagetrace/cells'/f"{row['cell']}.npz"
            representation=None
            if cache.exists():
                with np.load(cache) as z:
                    if {'I0','I1','M0','segb'}<=set(z.files):representation='cached_union'
            rows.append(dict(cell=row['cell'],fingerprint=row['fingerprint'],directory=str(p),
                recipe_id=meta['recipe_id'],recipe=meta['recipe_params'],producer_sha=meta.get('git_sha'),
                cache=str(cache) if representation else None,segmentation=representation,
                ledger_sha256=row['removed_sha256']))
        # Do not declare newer products complete from directory existence alone.
        newer=PAPER/'template_v3'/field/'bootstrap'
        newer_meta=newer/'downsample_result.json'
        record=dict(field=field,original_inputs=ctx,cells=rows,
            original_ownership='same_projection_only_v2',
            candidate_v3_root=str(newer),candidate_v3_result=json.loads(newer_meta.read_text()) if newer_meta.exists() else None,
            candidate_v3_status='requires_input_resolution' if newer_meta.exists() else 'not_published_at_this_location')
        (out/'inventory'/f'{field}.json').write_text(json.dumps(record,indent=2))
        records.append(dict(field=field,cells=len(rows),cached_union=sum(r['segmentation'] is not None for r in rows),
            candidate_v3_status=record['candidate_v3_status']))
    (out/'inventory/summary.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2),flush=True)


def run_cell(field,cell,out,*,catalogues=True,download_raw=False):
    start=time.monotonic()
    data=json.loads((out/'inventory'/f'{field}.json').read_text())
    record=next(r for r in data['cells'] if r['cell']==cell)
    directory=Path(record['directory']); heads=json.loads((directory/'headers.json').read_text())
    header=fits.Header.fromstring(next(iter(heads.values())));wcs=WCS(header)
    with np.load(directory/'arrays.npz') as z:expected=z['combined_image'];mask=z['combined_mask']
    raw=None;uncert=None;union=None;cache=None
    if record['cache']:
        cache=Path(record['cache'])
        with np.load(cache) as z:raw=z['I0'];union=z['segb'];mask=z['M0']
    elif (AUDIT/'tables'/f'{cell}_raw.npz').exists():
        cache=AUDIT/'tables'/f'{cell}_raw.npz'
        with np.load(cache) as z:raw=z['I0'];uncert=z['U0'];mask=z['M0']
    elif download_raw:
        from ..ps1_download import fetch_skycell_bands_masks_and_headers
        bands,masks,weights,headers,weight_headers=fetch_skycell_bands_masks_and_headers(cell,max_workers=4)
        if set(bands)!=set('rizy'):raise ValueError('Incomplete original PS1 download')
        raw,mask,uncert=bu.process_skycell_bands(bands,masks,weights,headers,weight_headers,band_weights=record['recipe']['band_weights'])
        del bands,masks,weights
    else:raise ValueError(f'No reusable pre-removal cache for {cell}; explicit --download-raw required')
    proj=cell.split('.')[1]
    trigger_path=PAPER/f'data_root/catalogs/gaia_projections/gaia_dr3_projection_rp18_v1/proj_{proj}.parquet'
    triggers=bu.select_catalog_for_cell(pd.read_parquet(trigger_path),wcs,raw.shape)
    identity=dict(cell=cell,combined_fingerprint=record['fingerprint'],recipe_id=record['recipe_id'],
        producer_sha=record['producer_sha'],trigger_sha256=file_digest(trigger_path),header_sha256=file_digest(directory/'headers.json'))
    print(cell,'replay', 'cached_union' if union is not None else 'SEP_required',flush=True)
    segmentation=None
    if union is None:
        segmentation=bu.build_sep_background_segmentation(raw,uncert,close_bright_mask=True)
        union=(segmentation.segmap>0)|segmentation.mask_bright_stars
    result,legacy,ledger=replay_cell(raw,None,mask,triggers,identity,expected=expected,cached_union=union)
    ledger.geometry['segmentation_union']=np.packbits(union.ravel())
    print(cell,'pixel replay exact; operations',len(ledger.regions),flush=True)
    sources=None;links=None;manifests=[];calibration=None;matches=None
    if catalogues:
        geometry=cell_query_geometry(wcs,raw.shape)
        print(cell,'catalogue query geometry',geometry,flush=True)
        g,gm,gp=fetch_gaia_box(out/'catalogs/gaia',*geometry['bbox'])
        ps,pm,pp=fetch_ps1_cone(out/'catalogs/ps1',geometry['ra'],geometry['dec'],geometry['radius_deg'])
        # Mean observation date is a reference for accounting only. Original
        # trigger coordinates and image pixels remain untouched.
        epoch=float(Time(header['MJD-OBS'],format='mjd').jyear) if 'MJD-OBS' in header else None
        gs=gaia_sources(g,wcs,epoch_year=epoch);pss=ps1_sources(ps,wcs)
        matches,calibration=match_gaia_ps1(gs,pss)
        sources=pd.concat([gs,pss],ignore_index=True)
        sources=attach_entities(sources,matches)
        # Keep full catalogue tables in the cache; cell associations need only
        # nearby positions. Outside centres remain outside, never edge-clipped.
        keep=(sources.pixel_x>=-600)&(sources.pixel_x<raw.shape[1]+600)&(sources.pixel_y>=-600)&(sources.pixel_y<raw.shape[0]+600)
        sources=sources[keep].reset_index(drop=True)
        print(cell,'associate',len(sources),'catalogue measurements',flush=True)
        sources,links=ledger.associate_centres(sources,support_radius_px=5.)
        manifests=[dict(catalogue='gaia',path=str(gp),manifest=gm),dict(catalogue='ps1',path=str(pp),manifest=pm)]
    dest=ledger.publish(out/'pilot'/cell,sources=sources,associations=links,catalogue_manifests=manifests,
        metadata=dict(segmentation_reuse='cached_union' if record['cache'] else 'computed_once',
            geometry_units='native PS1 pixels; source positions zero based',
            candidate_support_radius_px=5.,support_interpretation='candidate only; not total stellar PSF',
            matching=calibration))
    summary=dict(field=field,cell=cell,ledger=str(dest),replay_exact=True,operations=len(ledger.regions),
                 seconds=time.monotonic()-start,segmentation_reused=bool(record['cache']),
                 source_rows=0 if sources is None else len(sources),association_rows=0 if links is None else len(links),
                 source_status=None if sources is None else sources.groupby(['catalogue','centre_status']).size().to_string())
    parent=out/'pilot'/cell
    if matches is not None:matches.to_parquet(parent/'gaia_ps1_matches.parquet',index=False)
    (parent/'result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--inventory',action='store_true')
    p.add_argument('--field',default='C4');p.add_argument('--cell');p.add_argument('--no-catalogues',action='store_true');p.add_argument('--download-raw',action='store_true');a=p.parse_args()
    for sub in ['inventory','catalogs','pilot']:(a.out/sub).mkdir(parents=True,exist_ok=True)
    if a.inventory:inventory(a.out)
    if a.cell:run_cell(a.field,a.cell,a.out,catalogues=not a.no_catalogues,download_raw=a.download_raw)


if __name__=='__main__':main()
