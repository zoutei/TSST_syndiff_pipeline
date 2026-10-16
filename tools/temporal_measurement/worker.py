"""One isolated exposure: modern bootstrap, reference-profile WCS fit, residual export."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
import jax
import argparse,json,time,subprocess,sys,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates,binary_dilation
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
E=Path('/astro/armin/koji/syndiff/dev_runs/e2e_final_recipe_20261007')
REF=E/'run5/F1'; OLD=Path('/astro/armin/koji/syndiff/data/s0024/c2/k2')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def prepare(row,out):
 from syndiff_pipeline.difference_imaging.stages.background import tessreduce_residual as TR
 from syndiff_pipeline.forward_model.chain.config import load_config
 from syndiff_pipeline.forward_model.chain.hotpants_ref import build_frame
 from syndiff_pipeline.forward_model.chain._tk import write_fz
 from syndiff_pipeline.forward_model.chain.scene import swap_data
 stem=row.stem; dest=R/'data_root/s0024/c2/k2/diff_linear/ks_b'/f'{stem}_ks_b.fits.fz'
 if not dest.exists():
  with fits.open(row.file) as h: raw=np.asarray(h[1].data[:2048,44:2092],float);hdr=h[1].header.copy()
  rf=list((OLD/'ffi').glob('tess2020120182919-s0024-2-2-*-s_ffic.fits*'))[0]
  refw=WCS(fits.getheader(rf,1));fw=WCS(hdr)
  yy,xx=np.mgrid[:2048,:2048]
  ra,dec=fw.all_pix2world(xx+44,yy,0);rx,ry=refw.all_world2pix(ra,dec,0);rx-=44
  seed_path=E/'rf5_data/F1/lane/tmpl_conv/convolved_template_dx0.000_dy0.000.fits.fz'
  seed=np.asarray(fits.getdata(seed_path),float)
  isref=stem=='tess2020120182919-s0024-2-2'
  mapped=seed.copy() if isref else map_coordinates(seed,[ry,rx],order=3,mode='nearest')
  sm=np.asarray(fits.getdata(R/'data_root/s0024/c2/k2/diff_linear/shared_mask.fits.fz'),np.int64)
  if sm.ndim==3:sm=sm[0]
  masks=Path('/astro/armin/koji/syndiff/dev_runs/ksb_stability_20261008/rollout/F1/masks')
  stars=np.load(masks/'wing.npy')|np.load(masks/'rf5.npy')
  ex=stars.copy() if isref else map_coordinates(stars.astype(float),[ry,rx],order=0,mode='constant',cval=1)>0
  ex |= np.load(masks/'badcol.npy')
  ex |= binary_dilation(raw>8e4,structure=np.ones((25,7)))
  t=time.time();b,_,_=TR.estimate_tessreduce_residual_background(raw-mapped,sm,extra_exclude=ex,fill_method='harmonic',residual_exclude_percentile=50)
  write_fz(dest,fits.Header(),[(b,hdr)])
  np.savez_compressed(out/'background_check.npz',raw=raw[::8,::8],seed=mapped[::8,::8],background=b[::8,::8],excluded=ex[::8,::8])
  (out/'background.json').write_text(json.dumps(dict(seconds=time.time()-t,seed=str(seed_path),seed_sha256=sha(seed_path),warp='reference header WCS to current header WCS; cubic interpolation of background seed only; native observed pixels untouched',extra='stellar wing/rf5 masks move by header WCS; bad columns fixed; saturation from current FFI',fill='harmonic',residual_exclude_percentile=50,background_sha256=sha(dest)),indent=2))
  del raw,mapped,seed,sm,ex,b,rx,ry,xx,yy,ra,dec
  if isref:
   baseline=fits.getdata(E/'rf5_data/F1/lane/ks_b'/f'{stem}_ks_b.fits.fz',1)
   current=fits.getdata(dest,1); delta=current-baseline
   check=dict(max_abs=float(np.max(np.abs(delta))),rms=float(np.sqrt(np.mean(delta**2))),median=float(np.median(delta)))
   (out/'reference_background_replay.json').write_text(json.dumps(check,indent=2))
   if check['max_abs']>1e-4:raise RuntimeError(f'reference background replay failed: {check}')
 cfg=load_config(R/'inputs/chain.yaml')
 hp=R/'chain/hotpants'/stem/'hp_d'/f'{stem}_hp_d.fits.fz'
 if not hp.exists():build_frame(cfg,stem)
 scene=out/'scene';scene.mkdir(exist_ok=True)
 if not (scene/'scene_bundle.npz').exists():
  src=REF/'nbr_final';z=dict(np.load(src/'scene_bundle.npz'));meta=json.loads((src/'scene_meta.json').read_text())
  swap_data(z,meta,src,hp)
  S=int(z['stamp']);k=np.arange(S*S);x=z['cx'][:,None]+k%S-S//2;y=z['cy'][:,None]+k//S-S//2
  with fits.open(hp) as h:
   m=np.asarray(h[3].data); H,W=m.shape; inside=(x>=0)&(x<W)&(y>=0)&(y<H);bad=m[np.clip(y,0,H-1),np.clip(x,0,W-1)]
  # HP flags 0 / 0x40 accepted by the current final-image chain; preserve inherited scene exclusions.
  good=inside&((bad==0)|(bad==0x40));uid=z['uid'];kill=np.zeros(int(z['n_union'])+1,bool);kill[uid[~good]]=True;kill[-1]=True
  z['valid']=z['valid']&~kill[uid]
  meta.update(frame_stem=stem,frame_btjd=float(row.tmid),temporal_measurement=dict(reference_scene=str(src),image=str(hp),roles='fixed reference roles',valid='reference valid intersect current HP flags; unique-pixel propagation',shape='run5 reference profile until independent shape fit'))
  np.savez(scene/'scene_bundle.npz',**z);(scene/'scene_meta.json').write_text(json.dumps(meta,indent=1))
  (out/'scene_inventory.json').write_text(json.dumps(dict(N=len(z['role']),nuisance=int((z['role']==2).sum()),valid=int(z['valid'].sum()),source_bundle=meta['source_bundle'],source_ids_sha256=hashlib.sha256(z['source_id'].tobytes()).hexdigest(),image_sha256=sha(hp)),indent=2))
 return scene

def fit(scene,out,steps):
 fitout=out/'wcs_fit'
 if (fitout/'DONE').exists():return
 command=[sys.executable,'-m','syndiff_pipeline.forward_model.recipe','paper1_final_step1','--','--scene-dir',str(scene),'--out-dir',str(fitout),'--init-params-file',str(REF/'refit/params.npz'),'--colour-file','/astro/armin/koji/syndiff/dev_runs/chroma_v2_20260928/colour/F1_s24c2k2_u.csv','--steps-per-stage',f'{steps},0,0','--checkpoint-every','200','--log-every','50']
 (out/'fit_command.json').write_text(json.dumps(command,indent=1));subprocess.run(command,check=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('frame',type=int);p.add_argument('--prepare-only',action='store_true');p.add_argument('--steps',type=int,default=3000);a=p.parse_args()
 row=pd.read_csv(R/'inputs/frames.csv').set_index('frame_index').loc[a.frame];out=R/'frames'/f'f{a.frame:04d}';out.mkdir(exist_ok=True)
 scene=prepare(row,out)
 if not a.prepare_only:fit(scene,out,a.steps)
 (out/('PREPARED' if a.prepare_only else 'FITTED')).write_text(time.strftime('%Y-%m-%dT%H:%M:%S'))
if __name__=='__main__':main()
