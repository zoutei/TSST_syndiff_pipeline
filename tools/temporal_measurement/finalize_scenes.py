"""Freeze science support (not Hotpants kernel masks) and differential PM before cadence fits."""
import json
import jax  # before pandas/pyarrow: avoid the environment native-library import-order conflict
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import binary_dilation
from epoch_bundle import build
from support import science_support
from syndiff_pipeline.forward_model import scene_export as SE
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
REF=Path('/astro/armin/koji/syndiff/dev_runs/e2e_final_recipe_20261007/run5/F1/nbr_final')
OLD=Path('/astro/armin/koji/syndiff/data/s0024/c2/k2')
def main():
 frames=pd.read_csv(R/'inputs/frames.csv');dirs=[R/'frames'/f'f{i:04d}'/'scene' for i in frames.frame_index]
 reference=dict(np.load(REF/'scene_bundle.npz'));uid=reference['uid'];sid=reference['source_id'];valid=reference['valid'].copy();perframe=[]
 static_defect=fits.getdata('/astro/armin/koji/syndiff/dev_runs/ksb_stability_20261008/rollout/F1/defect_mask.fits.fz',1)!=0
 for (_,row),d in zip(frames.iterrows(),dirs):
  z=np.load(d/'scene_bundle.npz');assert np.array_equal(uid,z['uid']) and np.array_equal(sid,z['source_id'])
  hp=R/'chain/hotpants'/row.stem/'hp_d'/f'{row.stem}_hp_d.fits.fz'
  with fits.open(hp) as h:finite=np.isfinite(h[1].data)&np.isfinite(h[2].data)&(h[2].data>0)
  with fits.open(row.file) as h:raw=np.asarray(h[1].data[:2048,44:2092])
  defect=static_defect|binary_dilation(raw>8e4,structure=np.ones((25,7)))
  mask,ms=SE.load_frame_mask(OLD/'diff_linear',data_root=OLD.parents[2],sector=24,camera=2,ccd=2,btjd=row.tmid,shape=finite.shape)
  v=science_support(reference,mask,finite,defect);valid &=v
  perframe.append(dict(frame=int(row.frame_index),mask_source=ms,valid=int(v.sum())))
 core=reference['core'].astype(bool);train=reference['role']!=2
 retained=train&(valid[:,core].sum(1)>=15);baseline=train&(reference['valid'][:,core].sum(1)>=15)
 assert retained.sum()>.8*baseline.sum(),(retained.sum(),baseline.sum())
 report=dict(n_frames=len(dirs),sources=len(sid),stamp_valid=int(valid.sum()),unique_valid=int(len(np.unique(uid[valid]))),fully_lost_stars=int((valid.sum(1)==0).sum()),trainers_core15=int(retained.sum()),baseline_trainers_core15=int(baseline.sum()),perframe=perframe,mask_rule='reference scene support intersect current science mask excluding edge/PS1/TNS/asteroid, finite noise, current saturation and fixed defects; bright template-mask bit allowed; kernel-fitting flags not used')
 np.savez_compressed(R/'inputs/common_support.npz',valid=valid,source_id=sid)
 for (_,row),d in zip(frames.iterrows(),dirs):
  z=dict(np.load(d/'scene_bundle.npz'));z['valid']=valid.copy();meta=json.loads((d/'scene_meta.json').read_text())
  pm=build(REF/'fit_bundle.npz',z,d/'fit_bundle.npz',row.tmid)
  meta['source_bundle']=str(d/'fit_bundle.npz');meta['temporal_measurement']['proper_motion']=pm;meta['temporal_measurement']['common_support']=report
  meta['temporal_measurement']['valid']=report['mask_rule']
  np.savez(d/'scene_bundle.npz',**z);(d/'scene_meta.json').write_text(json.dumps(meta,indent=1))
 (R/'inputs/SCENES_FROZEN.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
