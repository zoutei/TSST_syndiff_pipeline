"""Validate every recipient against frozen contributions, using captured signal."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import FITSFixedWarning

from .. import convolved_store as cv
from .cell import explicit_deleted_image,validate_published,file_digest
from .field_operator import FrozenFieldOperator
from .transport import transport_cell,required_inputs


def historical_canonical():
    # Byte-for-byte snapshot from paper-pin 648c0bc. Never read another live
    # worktree at execution time. Only rendering/geometry functions are used;
    # convolved-store schema resolution is explicit in this module.
    from . import historical_v2
    return historical_v2,Path(historical_v2.__file__)


def validate_recipient(out,field,cell):
    warnings.filterwarnings('ignore',category=FITSFixedWarning)
    import dask
    dask.config.set(num_workers=2)
    start=time.monotonic();out=Path(out)
    inventory=json.loads((out/'inventory'/f'{field}.json').read_text());ctx=inventory['original_inputs']
    records={r['cell']:r for r in inventory['cells']}
    fieldop=FrozenFieldOperator(ctx)
    table_path=next(Path(ctx['mapping']).parent.glob('*master_skycells_list_os4.csv'))
    mapping=pd.read_csv(table_path).set_index('NAME',drop=False)
    old,oldpath=historical_canonical();md=old.metadata_for_cell(mapping,cell)
    names=required_inputs(cell,md,mapping);images={};deltas={};ledgers={}
    for name in names:
        if name not in records:raise ValueError(f'Uninventoried donor {name}')
        record=records[name]
        result=json.loads((out/'cell_versions'/name/record['fingerprint']/'result.json').read_text());ledger=Path(result['ledger'])
        manifest=validate_published(ledger)
        if manifest['identity']['combined_fingerprint']!=record['fingerprint']:raise ValueError('Ledger generation mismatch')
        with np.load(Path(record['directory'])/'arrays.npz') as z:images[name]=z['combined_image']
        deltas[name]=explicit_deleted_image(ledger,images[name])
        ledgers[name]=dict(path=str(ledger),fingerprint=manifest['fingerprint'],combined_fingerprint=record['fingerprint'])
    def before(name):return images[name]+deltas[name] if name in images else None
    def after(name):return images.get(name)
    result=transport_cell(cell,md,mapping,before,after,sigma=40.,radius=470,canonical_renderer=old.canonical_cell_image)
    projection,sc=cell.rsplit('.',1)
    own=records[cell]['fingerprint']
    nbs=[f'nbr:{n}={records[n]["fingerprint"]}' for n in old.canonical_neighbour_names(md,cell)]
    recipe=cv.convolved_recipe(psf_sigma=40.,radius=470,padding='same_projection_only_v2')
    root=Path(records[cell]['directory']).parents[4]
    # directory is data_root/ps1_skycells_zarr/ps1_combined.zarr/proj/cell/fp.
    fp=cv.resolve_convolved_fingerprint_for_recipe(root,projection,sc,recipe,own,
        extra_input_fingerprints=nbs,code_version=2)
    if fp is None:raise ValueError('Exact v2 convolved input missing; no latest-pointer fallback')
    convolved=cv.convolved_cell_dir(root,projection,sc,fp)
    with np.load(convolved/'arrays.npz') as z:mask=z['convolved_mask']
    assignment,assignment_meta=fieldop.assignment(cell)
    bins=[fieldop.bin(assignment,x,mask) for x in [result['before'],result['after'],result['transported']]]
    pub=fieldop.verify_published_contribution(cell,bins[1])
    if not pub['exact']:raise ValueError(f'Published contribution differs: {pub}')
    dest=out/'field_validation'/field/cell;dest.mkdir(parents=True,exist_ok=True)
    if bins[0] is None:
        if any(x is not None for x in bins):raise ValueError('Empty/nonempty support mismatch')
        idx=np.zeros(0,np.int64);b=a=d=bound=np.zeros(0,float);l5_pass=True
    else:
        idx=bins[0][0]
        if not all(np.array_equal(idx,x[0]) for x in bins[1:]):raise ValueError('L5 assignment changed in paired render')
        b,a,d=[x[1] for x in bins];bound=8*np.finfo(np.float32).eps*(abs(b)+abs(a)+abs(d))+1e-8
        l5_pass=bool(np.all(abs(b-a-d)<=bound))
    report=dict(field=field,cell=cell,seconds=time.monotonic()-start,
        scope='all captured explicit removals in every contributing source cell',
        status='validated_recipient' if l5_pass and result['report']['closure_passed'] else 'failed',
        operator=result['report'],assignment=assignment_meta,field_provenance=fieldop.provenance,
        canonical_source_sha256=file_digest(oldpath),convolved_fingerprint=fp,
        convolved_provenance_sha256=file_digest(convolved/'_provenance.json'),
        input_ledgers=ledgers,published_contribution=pub,
        l5_closure_passed=l5_pass,l5_max_residual=float(np.max(abs(b-a-d))) if len(idx) else 0.,
        integrated_deleted_signal=float(d.sum()))
    tmp=dest/f'.contribution-{os.getpid()}.npz'
    np.savez_compressed(tmp,indices=idx,before=b,after=a,deleted=d,bound=bound)
    os.replace(tmp,dest/'contribution.npz')
    (dest/'report.json').write_text(json.dumps(report,indent=2))
    if report['status']!='validated_recipient':raise ValueError('Field transport arithmetic closure failed')
    return report


def reduce_field(out,field):
    out=Path(out);inventory=json.loads((out/'inventory'/f'{field}.json').read_text());ctx=inventory['original_inputs']
    target=fits.getdata(ctx['template'],1)
    shape=target.shape;n=target.size
    before=np.zeros(n,float);after=np.zeros(n,float);deleted=np.zeros(n,float);bound=np.zeros(n,float)
    reports=[]
    for record in inventory['cells']:
        path=out/'field_validation'/field/record['cell']
        report=json.loads((path/'report.json').read_text())
        if report['status']!='validated_recipient':raise ValueError(f'Unvalidated recipient {record["cell"]}')
        with np.load(path/'contribution.npz') as z:
            ix=z['indices'];before[ix]+=z['before'];after[ix]+=z['after'];deleted[ix]+=z['deleted'];bound[ix]+=z['bound']
        reports.append(dict(cell=record['cell'],convolved_fingerprint=report['convolved_fingerprint'],
            source_ledgers=json.dumps(report['input_ledgers'],sort_keys=True),
            signed_flux_change=report['integrated_deleted_signal'],status='validated_recipient'))
    reconstructed=after.reshape(shape).astype(np.float32)
    if not np.array_equal(reconstructed,target,equal_nan=True):raise ValueError('Assembled after image differs from selected frozen template')
    residual=before-after-deleted
    if not np.all(abs(residual)<=bound+1e-8):raise ValueError('Assembled deletion closure failed')
    dest=out/'field_validation'/field
    pd.DataFrame(reports).to_parquet(dest/'template_contributions.parquet',index=False)
    # Keep measured absolute and deletion images for subsequent visual review.
    np.savez_compressed(dest/'template_accounting.npz',before=before.reshape(shape).astype(np.float32),
        after=reconstructed,deleted=deleted.reshape(shape).astype(np.float32),residual=residual.reshape(shape).astype(np.float32))
    summary=dict(status='numerically_validated_pending_visual_review',field=field,cells=len(reports),
        published_template_sha256=file_digest(ctx['template']),published_template_exact=True,
        max_closure_residual=float(np.max(abs(residual))),integrated_deleted_signal=float(deleted.sum()),
        limitations='Catalogue completeness and individual blended stellar flux are not certified; existing image artifacts are unchanged.')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary
