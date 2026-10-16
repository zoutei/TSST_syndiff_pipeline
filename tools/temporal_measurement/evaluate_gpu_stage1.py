"""Evaluate an already completed GPU WCS-only stage on the local native scene.
This is a separately labelled intermediate diagnostic; it never replaces the CPU campaign fit.
"""
import jax
import jax.numpy as jnp
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
from syndiff_pipeline.forward_model.crossfit.score_oof import build
from syndiff_pipeline.forward_model.scene_fit import save_state
from export import export
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
def main(frame,run):
 src=Path('/astro/armin/koji/syndiff/output/colab/jobs')/run/'artifacts'/f'full_f{frame:04d}'
 events=[json.loads(s) for s in (src/'history.jsonl').read_text().splitlines() if s.strip()]
 ends=[s for s in events if s.get('event')=='stage_end' and s.get('stage')==1]
 assert len(ends)==1 and ends[0].get('reason',ends[0].get('stop_reason'))=='plateau',ends
 out=R/'frames'/f'f{frame:04d}'/'gpu_wcs_fit';out.mkdir(exist_ok=True)
 shutil.copy2(src/'params_stage1.npz',out/'params.npz');shutil.copy2(src/'state_stage1.npz',out/'state_stage1.npz')
 meta=json.loads((src/'fit_meta.json').read_text());meta['scene_dir']=str(out.parent/'scene');meta['colour_file']='/astro/armin/koji/syndiff/dev_runs/chroma_v2_20260928/colour/F1_s24c2k2_u.csv';meta['stage_outcomes']=ends;meta['diagnostic_origin']=dict(run=run,checkpoint='params_stage1.npz',flux_evaluation='CPU at completed GPU WCS-stage checkpoint; fixed reference shape')
 (out/'fit_meta.json').write_text(json.dumps(meta,indent=1))
 _,scene,diag,params,_=build(out,out.parent/'scene',params_name='params.npz');sz=np.load(out/'state_stage1.npz');st={k:jnp.asarray(sz[k]) for k in sz.files}
 f,chi,_=diag(params,st);f=np.asarray(f);save_state(out/'flux_solved.npz',st,extra=dict(flux=f,chi2_core=np.asarray(chi),source_id=scene.z['source_id'],role=scene.role,tess_mag=scene.z['tess_mag']))
 (out/'provenance.json').write_text(json.dumps(dict(run=run,params_sha256=hashlib.sha256((out/'params.npz').read_bytes()).hexdigest(),stage=ends[0],scientific_scope='intermediate measurement, not full free-shape fit; CPU/GPU numerical comparison pending'),indent=2))
 export(frame,'gpu_wcs_fit','gpu_wcs_analysis')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('frame',type=int);a.add_argument('run');v=a.parse_args();main(v.frame,v.run)
