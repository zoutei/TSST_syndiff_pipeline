"""Render native images and descriptive temporal fields; never selects a temporal model."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016');FIG=R/'figures'
def main():
 frames=pd.read_csv(R/'inputs/frames.csv');available=[int(i) for i in frames.frame_index if (R/'frames'/f'f{i:04d}'/'native.npz').exists()]
 if 602 not in available:raise RuntimeError('Reference fit not exported yet')
 ref=dict(np.load(R/'frames/f0602/native.npz'));sid=ref['source_id'];S=int(ref['stamp']);disp=np.load(R/'inputs/display_sources.npz')['index'];tm=ref['tess_mag'];role=ref['role'];valid=ref['valid'];flux0=ref['flux'];xx=ref['x'];yy=ref['y']
 k=np.arange(S*S);dx=ref['cx'][:,None]+k%S-S//2-xx[:,None];dy=ref['cy'][:,None]+k//S-S//2-yy[:,None]
 rr=np.hypot(dx,dy);core=(rr<=3)&valid
 # Fixed diagnostics by reference catalogue and native support. No temporal residual clipping.
 good=np.zeros(len(sid),bool);cohort=np.load(R/'frames/f0602/shape_response.npz');good[cohort['index']]=True;good &=flux0>0
 edges=np.linspace(0,2048,4);cell=np.clip(np.digitize(yy,edges)-1,0,2)*3+np.clip(np.digitize(xx,edges)-1,0,2)
 scene=np.load(R/'frames/f0602/scene/scene_bundle.npz');island=scene['island'];rng=np.random.default_rng(20261016);halves=rng.integers(0,2,size=int(island.max())+1)[island]
 assert all(np.sum(good&(halves==h))>100 for h in [0,1])
 np.savez(R/'inputs/diagnostic_halves.npz',source_id=sid,island=island,half=halves,seed=20261016)
 metrics=[];positions=[];stacks=[];stack_frames=[]
 rref=ref['residual']/np.where(flux0>0,flux0,np.nan)[:,None]
 for frame in available:
  row=frames[frames.frame_index==frame].iloc[0];p=R/'frames'/f'f{frame:04d}';z=dict(np.load(p/'native.npz'));assert np.array_equal(z['source_id'],sid);assert np.array_equal(z['valid'],valid)
  # Nine actual source stamps, every frame with the same reference flux scale per row.
  fig,ax=plt.subplots(3,9,figsize=(18,6.6))
  for j,i in enumerate(disp):
   denom=max(float(flux0[i]),1);scale=max(float(np.max(ref['own'][i])/denom),.01)
   for a,(key,title) in enumerate([('data','data'),('model','all-source model'),('residual','data − model')]):
    image=z[key][i].reshape(S,S)/denom;mask=z['valid'][i].reshape(S,S)
    im=ax[a,j].imshow(np.where(mask,image,np.nan),origin='lower',cmap='RdBu_r' if a==2 else 'gray',vmin=-.01 if a==2 else 0,vmax=.01 if a==2 else scale)
    ax[a,j].set_xticks([]);ax[a,j].set_yticks([])
    if a==0:ax[a,j].set_title(f'T={tm[i]:.2f}\n({ref["cx"][i]},{ref["cy"][i]})',fontsize=8)
    if j==0:ax[a,j].set_ylabel(title)
  fig.colorbar(im,ax=ax[2,:].tolist(),fraction=.015,pad=.02,label='Residual / reference fitted source flux')
  fig.suptitle(f'FFI {frame} · orbit {int(row.orbit)} · BTJD {row.tmid:.5f}\nNative prepared pixels after sky/template subtraction; current radial/coma PSF, simultaneous faint neighbours; shape held fixed, WCS/flux fitted',fontsize=11)
  fig.subplots_adjust(top=.86,bottom=.12,wspace=.08,hspace=.08);fig.savefig(FIG/f'native_f{frame:04d}.png',dpi=120);plt.close(fig)
  delta=np.c_[z['x']-xx,z['y']-yy];positions.append(delta)
  norm=z['residual']/np.where(z['flux']>0,z['flux'],np.nan)[:,None]
  # These are residual-shape diagnostics, not calibrated intrinsic PSF covariance.
  weighted=valid*np.exp(-.5*(rr/2.5)**2)
  radial=np.nansum((norm-rref)*weighted*(rr*rr-2),axis=1)
  rec=dict(frame=frame,orbit=int(row.orbit),tmid=float(row.tmid),jitter_trace=float(row.jitter_trace),n_stars=int(good.sum()),dx_median_mpx=float(1e3*np.median(delta[good,0])),dy_median_mpx=float(1e3*np.median(delta[good,1])),residual_radial_diagnostic=float(np.median(radial[good])))
  for h in [0,1]:
   # Whole blend-island split: no shared fitted pixels across halves. Common fitted WCS is still conditional.
   g=good&(halves==h);rec[f'radial_half{h}']=float(np.median(radial[g]))
  metrics.append(rec)
  r=np.load(p/'raster_res_model.npz');index={int(s):i for i,s in enumerate(r['source_id'])};sels=np.array([index[int(s)] for s in sid[good] if int(s) in index])
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);stack=np.nanmedian(r['res'][sels],axis=0)
  stacks.append(stack);stack_frames.append(frame)
 pd.DataFrame(metrics).to_csv(R/'temporal_metrics.csv',index=False)
 pos=np.asarray(positions);np.savez_compressed(R/'positions.npz',frame=available,source_id=sid,delta=pos,good=good,cell=cell)
 m=pd.DataFrame(metrics);fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True)
 ax[0].plot(m.tmid,m.dx_median_mpx,'o-',label='x');ax[0].plot(m.tmid,m.dy_median_mpx,'o-',label='y');ax[0].set_ylabel('Median WCS-origin change [mpx]');ax[0].legend()
 ax[1].plot(m.tmid,m.radial_half0,'o-',label='star half A');ax[1].plot(m.tmid,m.radial_half1,'o-',label='star half B');ax[1].set(xlabel='BTJD',ylabel='Weighted residual radial diagnostic\n[fractional flux × pixel²]');ax[1].legend()
 gap=(frames.loc[frames.orbit==1,'tstop'].max(),frames.loc[frames.orbit==2,'tstart'].min())
 for a in ax:a.axvspan(*gap,color='.9',zorder=-5)
 fig.suptitle('Descriptive measurements relative to FFI 602 — no temporal law fitted\nWCS conditional on reference PSF; grey = orbit gap; lines connect samples only',fontsize=11);fig.tight_layout();fig.savefig(FIG/'temporal_tracks.png',dpi=140);plt.close(fig)
 base=stacks[stack_frames.index(602)];nc=6;nr=int(np.ceil(len(stacks)/nc));fig,ax=plt.subplots(2*nr,nc,figsize=(15,5*nr),squeeze=False)
 for j,frame in enumerate(stack_frames):
  ir,ic=divmod(j,nc)
  for b,img in enumerate([stacks[j],stacks[j]-base]):
   im=ax[2*ir+b,ic].imshow(img,origin='lower',extent=(-4,4,-4,4),cmap='RdBu_r',vmin=-.003,vmax=.003);ax[2*ir+b,ic].set_title(f'{frame}: '+('residual stack' if b==0 else 'minus reference stack'),fontsize=9)
 for j in range(len(stacks),nr*nc):
  ir,ic=divmod(j,nc);ax[2*ir,ic].axis('off');ax[2*ir+1,ic].axis('off')
 fig.subplots_adjust(right=.9,hspace=.3);fig.colorbar(im,cax=fig.add_axes([.92,.2,.015,.6]),label='Residual / own fitted flux')
 fig.suptitle('Underlying residual stacks and temporal differences (same stars, shared scale)\nDisplay rasterization is not new subpixel information; residual changes are not yet uniquely attributed',fontsize=12);fig.savefig(FIG/'residual_stack_history.png',dpi=130);plt.close(fig)
 out=dict(available_frames=available,expected_frames=frames.frame_index.tolist(),n_diagnostic_stars=int(good.sum()),status='descriptive reference-profile measurement; model selection pending',caveats=['WCS and shape coupled','in-sample WCS; split-star diagnostic is not held-source WCS validation','bootstrap images, not completed temporal per-band final chain','raster sampling and static profile errors can create temporal residual structure','no light-curve flattening'])
 (R/'measurement_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
