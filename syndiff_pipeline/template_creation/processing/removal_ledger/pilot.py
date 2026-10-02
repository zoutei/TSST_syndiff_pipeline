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

from .cell import replay_cell, file_digest, recover_segmentation_union, validate_published, array_digest
from .catalogues import cell_query_geometry, fetch_ps1_cone, fetch_gaia_box
from .matching import gaia_sources, ps1_sources, match_gaia_ps1, attach_entities
from .astrometry import calibrate_image_positions
from .. import band_utils as bu

warnings.filterwarnings('ignore',category=FITSFixedWarning)
RUNS=Path('/astro/armin/koji/syndiff/dev_runs')
PAPER=RUNS/'paper_dataset_20261001'
TRACE=RUNS/'epsf_stage_trace_20261002/fields'
AUDIT=RUNS/'ps1_removal_audit_20261002'
ATLAS=RUNS/'epsf_atlas_20261001/dip'
CODE_FILES={p.name:file_digest(p) for p in Path(__file__).parent.glob('*.py')}


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
            if representation is None:
                candidate=TRACE/field/'cells'/f"{row['cell']}.npz"
                if candidate.exists():
                    with np.load(candidate) as z:
                        if {'I0','Ibk','I1','M0'}<=set(z.files):
                            cache=candidate;representation='derive_union_if_identifiable'
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
    raw=None;uncert=None;union=None;cache=None;reuse='computed_once'
    if record['cache']:
        cache=Path(record['cache'])
        with np.load(cache) as z:
            raw=z['I0'];mask=z['M0']
            union=z['segb'] if 'segb' in z.files else recover_segmentation_union(raw,z['Ibk'])
            reuse='saved_union' if 'segb' in z.files else 'identifiable_background_cache'
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
    previous=out/'pilot'/cell/'result.json'
    if union is None and previous.exists():
        old=Path(json.loads(previous.read_text())['ledger'])
        old_manifest=validate_published(old)
        if old_manifest['identity']['combined_fingerprint']!=record['fingerprint'] or old_manifest['input_digest']!=array_digest(raw):
            raise ValueError('Prior ledger cache does not match these raw/combined inputs')
        with np.load(old/'geometry.npz') as geom:
            if 'segmentation_union' in geom:
                union=np.unpackbits(geom['segmentation_union'],count=raw.size).reshape(raw.shape).astype(bool)
                reuse='verified_prior_ledger_union'
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
    sources=None;links=None;manifests=[];calibration=None;matches=None;image_calibration=None;calibrators=None
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
        sources,image_calibration,calibrators=calibrate_image_positions(raw,mask,sources)
        sources['source_type']=np.where(sources.catalogue=='gaia_dr3','catalogue_source','stack_detection')
        sources['accounting_radius_px']=5.
        sources['support_definition']='5-pixel candidate aperture; unmeasured exterior PSF remains unknown'
        is_gaia=sources.catalogue=='gaia_dr3'
        t=bu.compute_tess_mag(sources.loc[is_gaia,'phot_g_mean_mag'].to_numpy(float),
            sources.loc[is_gaia,'phot_bp_mean_mag'].to_numpy(float),sources.loc[is_gaia,'phot_rp_mean_mag'].to_numpy(float))
        bright_indices=sources.index[is_gaia][t<13]
        sources.loc[bright_indices,'accounting_radius_px']=bu.star_footprint_radius(t[t<13])
        sources.loc[bright_indices,'support_definition']='historical bright-star halo search radius; association candidate only'
        print(cell,'associate',len(sources),'catalogue measurements',flush=True)
        sources,links=ledger.associate_centres(sources,support_radius_column='accounting_radius_px')
        manifests=[dict(catalogue='gaia',path=str(gp),manifest=gm),dict(catalogue='ps1',path=str(pp),manifest=pm)]
    extra={'legacy_records':pd.DataFrame(legacy)}
    if matches is not None:extra['gaia_ps1_candidates']=matches
    if calibrators is not None:extra['image_calibrators']=calibrators
    dest=ledger.publish(out/'pilot'/cell,sources=sources,associations=links,catalogue_manifests=manifests,extra_tables=extra,
        metadata=dict(segmentation_reuse=reuse,
            code_file_sha256=CODE_FILES,
            geometry_units='native PS1 pixels; source positions zero based',
            candidate_support_radius_px=5.,support_interpretation='candidate only; not total stellar PSF',
            matching=calibration,image_calibration=image_calibration))
    summary=dict(field=field,cell=cell,ledger=str(dest),replay_exact=True,operations=len(ledger.regions),
                 seconds=time.monotonic()-start,segmentation_reused=reuse!='computed_once',segmentation_source=reuse,
                 source_rows=0 if sources is None else len(sources),association_rows=0 if links is None else len(links),
                 source_status=None if sources is None else sources.groupby(['catalogue','centre_status']).size().to_string())
    parent=out/'pilot'/cell
    if matches is not None:matches.to_parquet(parent/'gaia_ps1_matches.parquet',index=False)
    if calibrators is not None:
        calibrators.to_parquet(parent/'image_calibrators.parquet',index=False)
        (parent/'image_calibration.json').write_text(json.dumps(image_calibration,indent=2))
    (parent/'result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--inventory',action='store_true')
    p.add_argument('--field',default='C4');p.add_argument('--cell');p.add_argument('--no-catalogues',action='store_true');p.add_argument('--download-raw',action='store_true');a=p.parse_args()
    for sub in ['inventory','catalogs','pilot']:(a.out/sub).mkdir(parents=True,exist_ok=True)
    if a.inventory:inventory(a.out)
    if a.cell:run_cell(a.field,a.cell,a.out,catalogues=not a.no_catalogues,download_raw=a.download_raw)


if __name__=='__main__':main()
