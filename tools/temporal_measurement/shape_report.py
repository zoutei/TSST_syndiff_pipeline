"""Spatial displacement and native images behind local shape-response measurements."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016');FIG=R/'figures'
def main():
 frames=pd.read_csv(R/'inputs/frames.csv');pos=np.load(R/'positions.npz');avail=pos['frame'];delta=pos['delta'];good=pos['good'];cell=pos['cell']
 med=np.array([[np.median(d[good&(cell==c)],axis=0) for c in range(9)] for d in delta]);amp=np.nanmax(np.linalg.norm(med,axis=2));amp=max(float(amp),.001)
 nc=6;nr=int(np.ceil(len(avail)/nc));fig,axes=plt.subplots(nr,nc,figsize=(16,3*nr),squeeze=False)
 xx,yy=np.meshgrid(np.linspace(2048/6,2048-2048/6,3),np.linspace(2048/6,2048-2048/6,3))
 for j,frame in enumerate(avail):
  ax=axes.flat[j];d=med[j];im=ax.quiver(xx.ravel(),yy.ravel(),d[:,0],d[:,1],1e3*np.linalg.norm(d,axis=1),angles='xy',scale_units='xy',scale=amp/400,cmap='viridis',clim=(0,1000*amp));ax.set(xlim=(0,2048),ylim=(0,2048),title=f'FFI {frame}',aspect='equal');ax.tick_params(labelsize=6)
 for ax in list(axes.flat)[len(avail):]:ax.axis('off')
 fig.subplots_adjust(top=.88,right=.9,hspace=.35);fig.colorbar(im,cax=fig.add_axes([.92,.2,.015,.6]),label='WCS-origin displacement [mpx]');fig.suptitle('Measured field changes relative to FFI 602\nSame model-origin convention; reference PSF fixed; arrow scale common to every exposure',fontsize=12);fig.savefig(FIG/'wcs_fields.png',dpi=130);plt.close(fig)
 refp=R/'frames/f0602/shape_response.npz'
 if not refp.exists():return
 ref=np.load(refp);lookup={int(s):i for i,s in enumerate(ref['source_id'])};table=[]
 for _,row in frames.iterrows():
  p=R/'frames'/f'f{int(row.frame_index):04d}';path=p/'shape_response.npz'
  if not path.exists():continue
  q=dict(np.load(path));pairs=[(i,lookup[int(s)]) for i,s in enumerate(q['source_id']) if int(s) in lookup];ix,jx=np.array(pairs).T
  c=q['coeff'][0,ix]-ref['coeff'][0,jx];ratio=q['held_shape_over_shift_mse'][:,ix]
  ok=np.isfinite(c).all(axis=1);rec=dict(frame=int(row.frame_index),tmid=float(row.tmid),orbit=int(row.orbit),n=int(ok.sum()),trace=float(np.nanmedian(c[ok,3]+c[ok,4])),shear=float(np.nanmedian(c[ok,3]-c[ok,4])),cross=float(np.nanmedian(c[ok,5])),local_dx=float(np.nanmedian(c[ok,1])),local_dy=float(np.nanmedian(c[ok,2])),held_mse_ratio=float(np.nanmedian(ratio)))
  table.append(rec)
  # Actual residual, locally predicted response, and remainder on the same native pixels.
  z=dict(np.load(p/'native.npz'));sel=np.load(R/'inputs/display_sources.npz')['source_id'];qmap={int(s):i for i,s in enumerate(q['source_id'])};zmap={int(s):i for i,s in enumerate(z['source_id'])};show=[int(s) for s in sel if int(s) in qmap]
  if show:
   fig,ax=plt.subplots(3,len(show),figsize=(2*len(show),6.3),squeeze=False)
   for j,s in enumerate(show):
    a,b=zmap[s],qmap[s];den=float(z['flux'][a]);valid=z['valid'][a].reshape(15,15);rr=z['residual'][a];pred=q['predicted_residual'][b]
    for k,(v,label) in enumerate([(rr,'observed residual'),(pred,'local response fit'),(rr-pred,'remaining residual')]):
     im=ax[k,j].imshow(np.where(valid,v.reshape(15,15)/den,np.nan),origin='lower',cmap='RdBu_r',vmin=-.01,vmax=.01);ax[k,j].set_xticks([]);ax[k,j].set_yticks([])
     if j==0:ax[k,j].set_ylabel(label)
     if k==0:ax[k,j].set_title(f'T={z["tess_mag"][a]:.2f}',fontsize=9)
   fig.subplots_adjust(top=.85,right=.9,hspace=.05,wspace=.05);fig.colorbar(im,cax=fig.add_axes([.92,.2,.015,.6]),label='Residual / fitted source flux');fig.suptitle(f'FFI {int(row.frame_index)}: native images behind local shift/shape diagnostics\nLocal response includes amplitude/background; in-sample illustration, held-pixel score reported separately',fontsize=11);fig.savefig(FIG/f'response_f{int(row.frame_index):04d}.png',dpi=120);plt.close(fig)
 t=pd.DataFrame(table);t.to_csv(R/'shape_response_metrics.csv',index=False)
 fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
 for col,lab in [('trace','trace'),('shear','xx − yy'),('cross','xy')]:ax[0].plot(t.tmid,1e3*t[col],'o-',label=lab)
 ax[0].set_ylabel('Linearized covariance change\n[10⁻³ pixel²]');ax[0].legend()
 ax[1].plot(t.tmid,1e3*t.local_dx,'o-',label='x');ax[1].plot(t.tmid,1e3*t.local_dy,'o-',label='y');ax[1].set_ylabel('Additional local displacement\n[mpx]');ax[1].legend()
 ax[2].plot(t.tmid,t.held_mse_ratio,'o-');ax[2].axhline(1,color='.5',ls='--');ax[2].set(xlabel='BTJD',ylabel='Held-pixel MSE ratio\nshape+shift / shift-only')
 fig.suptitle('Exposure-level local diagnostics; no temporal shape law fitted\nConditional on reference profile/neighbour solve; covariance is an equivalent response, not intrinsic optics',fontsize=12);fig.tight_layout();fig.savefig(FIG/'shape_response_tracks.png',dpi=140);plt.close(fig)
 print('shape reports',len(t),flush=True)
if __name__=='__main__':main()
