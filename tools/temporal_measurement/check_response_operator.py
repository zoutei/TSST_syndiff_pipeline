"""Check the finite-difference response on the actual integrated-grid renderer."""
import jax
import jax.numpy as jnp
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from syndiff_pipeline.forward_model import scene_fit as SF,fit as FIT,loss as L,epsf_model as EM
from response import responses,OFF
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016');F=Path('/astro/armin/koji/syndiff/dev_runs/e2e_final_recipe_20261007/run5/F1/refit')
m=json.loads((F/'fit_meta.json').read_text());s=SF.Scene(Path(m['scene_dir']));s.use_colour_file(m['colour_file']);i=np.load(R/'inputs/display_sources.npz')['index'];s.role[:]=2;s.role[i]=0
EM.set_radial_knots(m['chroma_radial_knots']);EM.set_coma_knots(m['chroma_coma_knots']);EM.set_radial_mode(m['chroma_radial_mode']);p=FIT.load_params_npz(F/'params.npz')
ctxs=s.contexts(m['colour_ref'],chroma_axis=tuple(m['chroma_axis']),chroma_g8_gauge=m['chroma_g8_gauge'],chroma_g8_extras=SF.g8_extras_tuple(m['chroma_g8_extras']),delta2_mean=m['chroma_delta2_mean'],**SF.meta_colour_kwargs(m))
r,idx,c=next(x for x in ctxs if x[0]==0)
@jax.jit
def render(off):return L.forward_model(p,replace(c,stamp_center_x=c.stamp_center_x-off[0],stamp_center_y=c.stamp_center_y-off[1]))[0].reshape(len(idx),-1)
result={};samples={};bases={}
for h in [.03,.25,.5]:
 a=np.array([np.asarray(render(jnp.asarray(o*h,np.float32))).astype(float) for o in OFF]);b=responses(a,h);samples[h]=a;bases[h]=b;result[str(h)]=dict(second_response_l2=np.sqrt(np.sum(b[:,:,3:6]**2,axis=1)).tolist(),rank=[int(np.linalg.matrix_rank(x)) for x in b])
(R/'inputs/response_operator.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(3,4,figsize=(12,9))
for row,star in enumerate([0,3,7]):
 ax[row,0].imshow(samples[.25][0,star].reshape(15,15),origin='lower',cmap='gray');ax[row,0].set_title('Actual rendered profile')
 for col,h in enumerate([.03,.25,.5],1):
  im=ax[row,col].imshow(bases[h][star,:,3].reshape(15,15),origin='lower',cmap='RdBu_r',vmin=-.35,vmax=.35);ax[row,col].set_title(f'xx response, h={h} px')
 for a in ax[row]:a.set_xticks([]);a.set_yticks([])
fig.subplots_adjust(top=.88,right=.9);fig.colorbar(im,cax=fig.add_axes([.92,.2,.015,.6]),label='Finite-grid second response [fraction/pixel²]')
fig.suptitle('Renderer preflight: tiny displacement differences do not measure smooth curvature\nThe bilinear integrated-grid representation is piecewise linear; use finite-grid responses and retain this limit',fontsize=12)
fig.savefig(R/'figures/response_operator_check.png',dpi=130);plt.close(fig)
