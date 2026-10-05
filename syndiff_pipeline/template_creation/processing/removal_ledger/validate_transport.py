"""Real historical seam validation for a scoped one-cell removal intervention."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import FITSFixedWarning

from .. import canonical_cell, combined_store
from .transport import transport_cell,bin_regmap
from .cell import file_digest
from .field_operator import FrozenFieldOperator


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--cell',default='skycell.2528.005');a=p.parse_args()
    warnings.filterwarnings('ignore',category=FITSFixedWarning)
    runs=Path('/astro/armin/koji/syndiff/dev_runs');paper=runs/'paper_dataset_20261001';data_root=paper/'data_root'
    ctx=json.loads((runs/'epsf_stage_trace_20261002/fields/C4/inputs.json').read_text())
    table_path=next(Path(ctx['mapping']).parent.glob('*master_skycells_list_os4.csv'))
    mapping=pd.read_csv(table_path).set_index('NAME',drop=False)
    # The adapter loads the exact historical ownership functions, not v3 defaults.
    pinned=Path('/home/kshukawa/syndiff_pipeline/.claude/worktrees/paper-pin/syndiff_pipeline/template_creation/processing/canonical_cell.py')
    spec=importlib.util.spec_from_file_location('_ledger_historical_canonical_v2',pinned)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    metadata=old.metadata_for_cell(mapping,a.cell)
    recipe=json.loads((paper/'C4/bootstrap/step_template.json').read_text())['combined_recipe']
    loaded={};inputs={}
    def after(name):
        if name not in loaded:
            proj,sc=name.rsplit('.',1)
            fp=combined_store.resolve_combined_fingerprint_for_recipe(data_root,proj,sc,recipe)
            if fp is None:raise ValueError(f'No immutable combined input for {name}')
            directory=combined_store.combined_cell_dir(data_root,proj,sc,fp)
            with np.load(directory/'arrays.npz') as z:loaded[name]=z['combined_image']
            inputs[name]=dict(fingerprint=fp,provenance_sha256=file_digest(directory/'_provenance.json'))
        return loaded[name]
    source=runs/f'epsf_atlas_20261001/dip/stagetrace/cells/{a.cell}.npz'
    with np.load(source) as z:
        baseline=z['Ibk'];published_pad=z['Ipad'];mask=z['Mpad']
    def before(name):return baseline if name==a.cell else after(name)
    print('Render full-halo before/after/delta for',a.cell,flush=True)
    res=transport_cell(a.cell,metadata,mapping,before,after,sigma=40.,radius=470,canonical_renderer=old.canonical_cell_image)
    fin=np.isfinite(published_pad)&np.isfinite(res['after'])
    err=np.where(fin,res['after'].astype(float)-published_pad,0.)
    res['report'].update(scope='Restore explicit removals in one source cell only; all other combined images fixed',
        cell=a.cell,ownership='same_projection_only_v2',ownership_source=str(pinned),ownership_sha256=file_digest(pinned),
        mapping_sha256=file_digest(table_path),input_fingerprints=inputs,
        published_same_nan_pattern=bool(np.array_equal(np.isnan(published_pad),np.isnan(res['after']))),
        published_max_absolute_difference=float(abs(err).max()),published_exact=bool(np.array_equal(published_pad,res['after'],equal_nan=True)))
    reg_path=paper/f'C4/bootstrap/data_priv/s0023/c4/k2/mapping/oversampling_4/tess_s23_4_2_{a.cell}_os4.fits.fz'
    reg=fits.getdata(reg_path,1)
    th=fits.getheader(ctx['template'],1);shape=(int(th['NAXIS2']),int(th['NAXIS1']))
    # Bin only this cell's actual assignment. Other recipient cells that consume
    # this donor must also be rendered before claiming a whole-template sum.
    delta=bin_regmap(res['transported'],reg,shape,quality_mask=mask)
    paired=bin_regmap(res['before'],reg,shape,quality_mask=mask)-bin_regmap(res['after'],reg,shape,quality_mask=mask)
    res['report'].update(binned_max_residual=float(abs(paired-delta).max()),
                         binned_integrated_residual=float((paired-delta).sum()),
                         binned_deleted_flux=float(delta.sum()))
    # The simple regmap diagnostic above is not the final L5 operator: real
    # inter-skycell rim patches also apply at zero shift. Verify the real sparse
    # contribution, including those patches and historical float32 narrowing.
    field=FrozenFieldOperator(ctx)
    assignment,assignment_meta=field.assignment(a.cell)
    bins=[field.bin(assignment,plane,mask) for plane in [res['before'],res['after'],res['transported']]]
    baseline_check=field.verify_published_contribution(a.cell,bins[1])
    if not baseline_check['exact']:raise RuntimeError('Frozen L5 contribution mismatch')
    if any(b is None for b in bins):raise RuntimeError('Empty L5 test contribution')
    if not all(np.array_equal(bins[0][0],b[0]) for b in bins[1:]):raise RuntimeError('Counterfactual L5 support changed')
    bv,av,dv=[b[1] for b in bins]
    residual=bv-av-dv
    # Conservative convolution-plus-binning roundoff bound at final precision.
    bound=8*np.finfo(np.float32).eps*(abs(bv)+abs(av)+abs(dv))+1e-8
    l5_passed=bool(np.all(abs(residual)<=bound))
    res['report']['actual_l5']=dict(assignment=assignment_meta,provenance=field.provenance,
        published_contribution=baseline_check,closure_passed=l5_passed,
        max_absolute_residual=float(abs(residual).max()),integrated_residual=float(residual.sum()),
        integrated_deleted_signal=float(dv.sum()),
        allowance='8*float32_eps*(abs(before)+abs(after)+abs(delta))+1e-8')
    out=a.out/'transport'/a.cell;out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(res['report'],indent=2))
    np.savez_compressed(out/'actual_l5.npz',indices=bins[0][0],before=bv,after=av,transported=dv,residual=residual,bound=bound)
    pd.DataFrame([dict(source_cell=a.cell,recipient_cell=a.cell,group_id=0,
        operation_scope='all explicit removals in this source cell',
        status='scoped_contribution_validated' if l5_passed else 'failed',
        signed_flux_change=float(dv.sum()),source_fingerprint=inputs[a.cell]['fingerprint'],
        mapping_sha256=field.provenance['master_sha256'])]).to_parquet(out/'template_contributions.parquet',index=False)
    # Diagnostic previews preserve absolute images and the measured differences.
    def reduce(x,f=8):
        h,w=x.shape;h=h//f*f;w=w//f*f
        return np.nan_to_num(x[:h,:w]).reshape(h//f,f,w//f,f).mean((1,3))
    np.savez_compressed(out/'maps.npz',before=reduce(res['before']),after=reduce(res['after']),
        transported=reduce(res['transported']),residual=reduce(res['residual']),published_residual=reduce(err))
    print(json.dumps(res['report'],indent=2),flush=True)
    if not res['report']['closure_passed']:raise RuntimeError('Transport closure failed')
    if not res['report']['published_exact']:raise RuntimeError('Historical reconstruction differs from published pixels')
    if not l5_passed:raise RuntimeError('Actual L5 closure failed')


if __name__=='__main__':main()
