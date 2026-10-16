"""Freeze common measured support and differential PM before any cadence fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from epoch_bundle import build
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
def main():
 frames=pd.read_csv(R/'inputs/frames.csv');dirs=[R/'frames'/f'f{i:04d}'/'scene' for i in frames.frame_index]
 first=np.load(dirs[0]/'scene_bundle.npz');uid=first['uid'];sid=first['source_id'];valid=np.ones_like(first['valid'],bool)
 for d in dirs:
  z=np.load(d/'scene_bundle.npz');assert np.array_equal(uid,z['uid']) and np.array_equal(sid,z['source_id']);valid &=z['valid']
 report=dict(n_frames=len(dirs),sources=len(sid),stamp_valid=int(valid.sum()),unique_valid=int(len(np.unique(uid[valid]))),fully_lost_stars=int((valid.sum(1)==0).sum()))
 np.savez_compressed(R/'inputs/common_support.npz',valid=valid,source_id=sid)
 for (_,row),d in zip(frames.iterrows(),dirs):
  z=dict(np.load(d/'scene_bundle.npz'));z['valid']=valid.copy();meta=json.loads((d/'scene_meta.json').read_text());src=meta['source_bundle']
  pm=build(src,z,d/'fit_bundle.npz',row.tmid)
  meta['source_bundle']=str(d/'fit_bundle.npz');meta['temporal_measurement']['proper_motion']=pm;meta['temporal_measurement']['common_support']=report
  np.savez(d/'scene_bundle.npz',**z);(d/'scene_meta.json').write_text(json.dumps(meta,indent=1))
 (R/'inputs/SCENES_FROZEN.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
