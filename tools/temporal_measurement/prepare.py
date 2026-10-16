"""Freeze timing-selected exposures and private modern-bootstrap inputs. No training."""
import json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
E=Path('/astro/armin/koji/syndiff/dev_runs/e2e_final_recipe_20261007')
REF='tess2020120182919-s0024-2-2'

def link(dst,src):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink(): assert dst.resolve()==src.resolve(); return
 assert not dst.exists(),dst
 dst.symlink_to(src)

def main():
 f=pd.read_csv('/astro/armin/koji/syndiff/dev_runs/temporal_raw_epsf_20261001/inputs/frames.csv')
 j=pd.read_csv('/astro/armin/koji/syndiff/dev_runs/temporal_epsf_20260929/eng/jitter_per_ffi_cube_s24c2k2.csv')
 assert len(f)==len(j)
 f['jitter_trace']=j.vxx+j.vyy
 f['stem']=f.file.map(lambda p:Path(p).name.split('-0180')[0])
 picks=[]
 for orbit,g in f[(f.dquality==0)&f.hpd_ok].groupby('orbit'):
  t0,t1=g.tmid.min(),g.tmid.max()
  targets=[t0,t0+1/48,t0+.25,t0+1,t0+2,t0+.4*(t1-t0),t0+.65*(t1-t0),t1-1,t1-1/48,t1]
  ids=[int((g.tmid-t).abs().idxmin()) for t in targets]
  peak=int(g.jitter_trace.idxmax()); ids += [peak,int((g.tmid-(f.loc[peak,'tmid']+1/48)).abs().idxmin())]
  if orbit==2:
   ref=int(g.index[g.stem==REF][0]); ids[5]=ref
  ids=list(dict.fromkeys(ids))
  for q in np.linspace(0,1,30):
   if len(ids)>=12:break
   ix=int((g.tmid-(t0+q*(t1-t0))).abs().idxmin())
   if ix not in ids:ids.append(ix)
  assert len(ids)==12
  picks+=ids
 chosen=f.loc[sorted(picks)].copy(); (R/'inputs').mkdir(exist_ok=True)
 chosen.to_csv(R/'inputs/frames.csv',index=False)
 (R/'inputs/selection.json').write_text(json.dumps(dict(rule='12 per orbit: timing endpoints/startup/interior, jitter maximum and adjacent control; reference replaces one interior orbit2 sample',reference_stem=REF,frame_indices=chosen.frame_index.tolist()),indent=2))
 raw=yaml.safe_load((E/'configs/run5/F1.yaml').read_text())
 raw['code']['sha']=None;raw['code']['forward_model_root']=str(R/'worktree');raw['out_root']=str(R/'chain');raw['data_root']=str(R/'data_root')
 raw['code']['pythonpath_extra']=[]
 scc=R/'data_root/s0024/c2/k2'; lane=scc/'diff_linear';lane.mkdir(parents=True,exist_ok=True)
 original=Path('/astro/armin/koji/syndiff/data/s0024/c2/k2')
 modern=E/'rf5_data/F1/lane'
 link(scc/'ffi',original/'ffi')
 for name in ['diff_config.yaml','shared_mask.fits.fz','gaia_catalog_pipeline.csv','hotpants_substamp_stars.csv','mask_settings.yaml']:
  link(lane/name,modern/name)
 link(lane/'hp_d',original/'diff_linear/hp_d')
 (lane/'ks_b').mkdir(exist_ok=True)
 for name in ['mapping','perband']:
  link(R/'chain'/name,E/'run5/F1'/name)
 raw['inputs']['lane_dir']=str(lane)
 (R/'inputs/chain.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
 for p in ['logs','frames','figures','condor']:(R/p).mkdir(exist_ok=True)
 print(chosen[['frame_index','orbit','tmid','jitter_trace','stem']].to_string(index=False))
if __name__=='__main__':main()
