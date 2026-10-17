"""First measured stress/reference pair, including population images and conditional uncertainty."""
import json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from syndiff_pipeline.forward_model.diagnostics.stacked_temporal_residuals import rasterize_packed_values
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
def main():
 refdir=R/'frames/f0602/gpu_wcs_analysis';highdir=R/'frames/f0469';a=np.load(highdir/'shape_response.npz');b=np.load(refdir/'shape_response.npz');assert np.array_equal(a['source_id'],b['source_id'])
 d=a['coeff'][0]-b['coeff'][0];trace=d[:,3]+d[:,4];z=np.load(R/'frames/f0602/scene/scene_bundle.npz');island=z['island'][a['index']];ok=np.isfinite(trace);groups=np.unique(island[ok]);members=[np.flatnonzero(ok&(island==g)) for g in groups];rng=np.random.default_rng(20261016)
 boot=[np.median(trace[np.concatenate([members[j] for j in rng.integers(0,len(members),len(members))])]) for _ in range(400)]
 frames=pd.read_csv(R/'inputs/frames.csv').set_index('frame_index');jitter=float(frames.loc[469,'jitter_trace']-frames.loc[602,'jitter_trace'])
 info=dict(reference='completed GPU WCS-only stage602, CPU flux evaluation',high='CPU WCS-only fit469',n_sources=int(ok.sum()),n_blend_islands=len(groups),trace_median_px2=float(np.median(trace[ok])),trace_cluster_bootstrap95_px2=np.percentile(boot,[2.5,97.5]).tolist(),telemetry_camera_trace_difference_px2=jitter,held_shape_over_shift_median_high=float(np.nanmedian(a['held_shape_over_shift_mse'])),held_shape_over_shift_median_ref=float(np.nanmedian(b['held_shape_over_shift_mse'])),scope='Equivalent local shape response; bootstrap conditional on shared fitted models/WCS/background; mixed-backend reference with matched CPU check pending. Telemetry camera-frame trace is not a full detector-projected covariance law.')
 (R/'pair_measurements.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2),flush=True)
 allmaps=[];allres=[];ns=[]
 for folder in [refdir,highdir]:
  q=dict(np.load(folder/'native.npz'));assert np.array_equal(q['source_id'][a['index']],a['source_id']);S=int(q['stamp']);k=np.arange(S*S);maps=[];resmaps=[]
  for i in a['index']:
   px=q['cx'][i]+k%S-S//2;py=q['cy'][i]+k//S-S//2;valid=q['valid'][i];flux=q['flux'][i];args=(np.array([q['x_effective'][i]]),np.array([q['y_effective'][i]]),px,py,valid[None].astype(float))
   maps.append(rasterize_packed_values(((q['own'][i]+q['residual'][i])/flux)[None],*args,oversample=4,half_width=4)[0]);resmaps.append(rasterize_packed_values((q['residual'][i]/flux)[None],*args,oversample=4,half_width=4)[0])
  allmaps.append(np.asarray(maps));allres.append(np.asarray(resmaps))
 edges=np.linspace(0,2048,4);cell=np.clip(np.digitize(b['y'],edges)-1,0,2)*3+np.clip(np.digitize(b['x'],edges)-1,0,2)
 fig,ax=plt.subplots(9,4,figsize=(12,22))
 for c in range(9):
  sel=ok&(cell==c)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);D0=np.nanmedian(allmaps[0][sel],0);D1=np.nanmedian(allmaps[1][sel],0);Q0=np.nanmedian(allres[0][sel],0);Q1=np.nanmedian(allres[1][sel],0)
  for j,(img,title) in enumerate([(D0,'Reference deblended data'),(D1,'High-jitter deblended data'),(D1-D0,'Data stack change'),(Q1-Q0,'Residual stack change')]):
   im=ax[c,j].imshow(img,origin='lower',extent=(-4,4,-4,4),cmap='RdBu_r' if j>=2 else 'gray',vmin=-.004 if j>=2 else 0,vmax=.004 if j>=2 else max(float(np.nanmax(D0)),.01));ax[c,j].tick_params(labelsize=6)
   if c==0:ax[c,j].set_title(title,fontsize=9)
  ax[c,0].set_ylabel(f'CCD cell ({c%3},{c//3}), N={sel.sum()}',fontsize=8)
 fig.subplots_adjust(top=.94,right=.88,hspace=.2,wspace=.2);fig.colorbar(im,cax=fig.add_axes([.91,.2,.012,.6]),label='Change / own fitted flux');fig.suptitle('Actual images behind the high-jitter/reference response measurement\nSame sources and support; other-source model subtracted; native pixels splatted only for display',fontsize=12);fig.savefig(R/'figures/pair_population_images.png',dpi=130,bbox_inches='tight');plt.close(fig)
 null=np.load(R/'static_profile_null.npz');sid=np.load(R/'inputs/display_sources.npz')['source_id'];lookup={int(s):i for i,s in enumerate(a['source_id'])};ii=np.array([lookup[int(s)] for s in sid]);dn=null['coeff'][2]-null['coeff'][0]
 fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].plot(np.arange(1,10),1000*trace[ii],'o-',label='measured image response');ax[0].plot(np.arange(1,10),1000*(dn[:,3]+dn[:,4]),'o-',label='constant-profile conditional null');ax[0].set(xlabel='Fixed location',ylabel='Trace change [10⁻³ px²]');ax[0].legend()
 bins=np.linspace(-1.5,1.5,50)
 for name,q in [('reference',b),('high jitter',a)]:
  v=q['held_shape_over_shift_mse'].ravel();v=v[np.isfinite(v)&(v>0)];ax[1].hist(np.log10(v),bins=bins,histtype='step',density=True,label=name)
 ax[1].axvline(0,color='.5',ls='--');ax[1].set(xlabel='log10 held-pixel MSE: shape+shift / shift-only',ylabel='Density');ax[1].legend();fig.suptitle('Local shape change and held-pixel prediction; no temporal law fitted\nThe nine-location null is conditional, not a bound on all instrumental errors',fontsize=11);fig.tight_layout();fig.savefig(R/'figures/pair_response_checks.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
