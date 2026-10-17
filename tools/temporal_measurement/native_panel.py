import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
p=argparse.ArgumentParser();p.add_argument('native');p.add_argument('output');p.add_argument('title');p.add_argument('--reference');a=p.parse_args();z=dict(np.load(a.native));ref=dict(np.load(a.reference)) if a.reference else z;assert np.array_equal(z['source_id'],ref['source_id']);rows=np.load(R/'inputs/display_sources.npz')['index'];S=int(z['stamp'])
fig,ax=plt.subplots(3,len(rows),figsize=(18,6.6))
for j,i in enumerate(rows):
 den=max(float(ref['flux'][i]),1);v=z['valid'][i].reshape(S,S);peak=max(float(np.max(ref['own'][i])/den),.01)
 for k,(key,label) in enumerate([('data','prepared image'),('model','all-source model'),('residual','data − model')]):
  im=ax[k,j].imshow(np.where(v,z[key][i].reshape(S,S)/den,np.nan),origin='lower',cmap='RdBu_r' if k==2 else 'gray',vmin=-.01 if k==2 else 0,vmax=.01 if k==2 else peak);ax[k,j].set_xticks([]);ax[k,j].set_yticks([])
  if k==0:ax[k,j].set_title(f'T={z["tess_mag"][i]:.2f}\n({z["cx"][i]},{z["cy"][i]})',fontsize=8)
  if j==0:ax[k,j].set_ylabel(label)
fig.subplots_adjust(top=.82,bottom=.1,right=.9,wspace=.06,hspace=.06);fig.colorbar(im,cax=fig.add_axes([.92,.13,.012,.2]),label='Residual / reference source flux')
fig.suptitle(a.title+'\nNative pixels after background/template subtraction; jointly fitted faint neighbours; blank pixels excluded',fontsize=12)
fig.savefig(a.output,dpi=130);plt.close(fig)
