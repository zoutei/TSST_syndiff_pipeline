"""Timing selection and native observed images; display sky subtraction only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
f=pd.read_csv(R/'inputs/frames.csv')
a=pd.read_csv('/astro/armin/koji/syndiff/dev_runs/temporal_raw_epsf_20261001/inputs/frames.csv')
j=pd.read_csv('/astro/armin/koji/syndiff/dev_runs/temporal_epsf_20260929/eng/jitter_per_ffi_cube_s24c2k2.csv')
z=np.load('/astro/armin/koji/syndiff/dev_runs/e2e_final_recipe_20261007/run5/F1/nbr_final/scene_bundle.npz')
xy=np.c_[z['x0'],z['y0']];tm=z['tess_mag'];ids=z['source_id'];role=z['role']
bright=(tm<13)&(role!=2);tree=cKDTree(xy[bright]);dist,_=tree.query(xy,k=2)
sel=[]
for cy in [400,1024,1650]:
 for cx in [400,1024,1650]:
  good=(tm>9)&(tm<10.5)&(role!=2)&(dist[:,1]>5)
  score=np.sum((xy-[cx,cy])**2,axis=1);score[~good]=np.inf
  ii=int(np.argmin(score));assert np.isfinite(score[ii]);sel.append(ii)
np.savez(R/'inputs/display_sources.npz',index=sel,source_id=ids[sel],xy=xy[sel],tess_mag=tm[sel])
fig,ax=plt.subplots(figsize=(12,3.6));ax.plot(a.tmid,np.where(a.dquality==0,1e3*(j.vxx+j.vyy),np.nan),lw=.6,color='.55',label='quality-zero exposures: camera-frame jitter trace')
ax.scatter(f.tmid,1e3*f.jitter_trace,c=f.orbit,cmap='coolwarm',s=35,zorder=3,label='selected 24 exposures')
for _,r in f.iterrows():ax.annotate(str(r.frame_index),(r.tmid,r.jitter_trace*1e3),xytext=(0,5),textcoords='offset points',fontsize=6,rotation=60)
ax.set(xlabel='BTJD (FFI midpoint)',ylabel='Telemetry variance trace [10⁻³ pixel²]',title='S24 C2K2: exposure selection before fitting');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(R/'figures/exposure_selection.png',dpi=140);plt.close(fig)
frames=[1,469,602,1225];imgs=[]
for frame in frames:
 row=f[f.frame_index==frame].iloc[0]
 with fits.open(row.file) as h:d=np.asarray(h[1].data[:2048,44:2092],float)
 stamps=[]
 for i in sel:
  x,y=np.rint(xy[i]).astype(int);p=d[y-7:y+8,x-7:x+8].copy();yy,xx=np.mgrid[-7:8,-7:8];ann=np.hypot(xx,yy)>=6
  p-=np.median(p[ann]);stamps.append(p)
 imgs.append(stamps)
fig,ax=plt.subplots(9,4,figsize=(9,18))
for row,i in enumerate(sel):
 scale=max(np.max(imgs[2][row]),1)
 for col,frame in enumerate(frames):
  p=imgs[col][row]/scale;im=ax[row,col].imshow(p,origin='lower',cmap='gray',vmin=0,vmax=1);ax[row,col].set_xticks([]);ax[row,col].set_yticks([])
  if row==0:ax[row,col].set_title(f'FFI {frame}\norbit {1 if frame<602 else 2}')
  if col==0:ax[row,col].set_ylabel(f'T={tm[i]:.2f}\n({xy[i,0]:.0f},{xy[i,1]:.0f})',fontsize=8)
fig.suptitle('Actual native FFI stamps: local annulus removed for display only\nEach row shares the reference-frame peak scale; no resampling of observed pixels',fontsize=12,y=.995)
fig.tight_layout(rect=(0,0,1,.97));fig.savefig(R/'figures/native_input_stamps.png',dpi=130);plt.close(fig)
print('Saved native input atlas and timing selection; no fit or temporal inference.',flush=True)
