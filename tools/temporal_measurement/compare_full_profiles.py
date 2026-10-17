"""Compare independently fitted PSFs at identical positions, colours and pixel phases.
Only model profiles are sampled on a finer grid; observed native pixels are never resampled here.
"""
import jax
import jax.numpy as jnp
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from syndiff_pipeline.forward_model import scene_fit as SF,fit as FIT,loss as L,epsf_model as EM
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
def main():
 frames=[i for i in [602,1,469,1225] if (R/'frames'/f'f{i:04d}'/'full_fit/params.npz').exists()]
 assert 602 in frames and len(frames)>=2,frames
 scene=SF.Scene(R/'frames/f0602/scene');idx=np.load(R/'inputs/display_sources.npz')['index'];scene.role[:]=2;scene.role[idx]=0
 base=FIT.load_params_npz(R/'frames/f0602/full_fit/params.npz');profiles=[];offset=(np.arange(4)-1.5)/4
 for frame in frames:
  fit=R/'frames'/f'f{frame:04d}'/'full_fit';m=json.loads((fit/'fit_meta.json').read_text());scene.use_colour_file(m['colour_file'])
  EM.set_radial_knots(m['chroma_radial_knots']);EM.set_coma_knots(m['chroma_coma_knots']);EM.set_radial_mode(m['chroma_radial_mode']);p=FIT.load_params_npz(fit/'params.npz');p['wcs_coeff']=base['wcs_coeff']
  contexts=scene.contexts(m['colour_ref'],chroma_axis=tuple(m['chroma_axis']),chroma_g8_gauge=m['chroma_g8_gauge'],chroma_g8_extras=SF.g8_extras_tuple(m['chroma_g8_extras']),delta2_mean=m['chroma_delta2_mean'],**SF.meta_colour_kwargs(m));_,ids,c=next(t for t in contexts if t[0]==0)
  mapping={int(s):j for j,s in enumerate(np.asarray(ids))};order=np.array([mapping[int(s)] for s in idx]);_,xx,yy,_=L.forward_model(p,c);star_occ=c.members.reshape(-1)[c.occ_flat_idx];ch=L._colour_slot_terms(p,c,star_occ)
  xe=xx.reshape(-1)+ch[1];ye=yy.reshape(-1)+ch[2]
  @jax.jit
  def render(off):return L.forward_model(p,replace(c,stamp_center_x=xe+off[0],stamp_center_y=ye+off[1]))[0].reshape(len(ids),15,15)
  P=np.zeros((len(idx),60,60),np.float32)
  for iy,dy in enumerate(offset):
   for ix,dx in enumerate(offset):P[:,iy::4,ix::4]=np.asarray(render(jnp.array([dx,dy],dtype=jnp.float32)))[order]
  profiles.append(P);print('profile rendered',frame,flush=True)
 profiles=np.asarray(profiles);np.savez_compressed(R/'full_profile_comparison.npz',frame=frames,source_id=scene.z['source_id'][idx],profile=profiles,step=.25,coordinate_min=-7.375,alignment='same reference WCS/spatial coordinates; explicit colour origin removed per model; same subpixel sampling')
 fig,ax=plt.subplots(len(idx),len(frames),figsize=(3*len(frames),2.5*len(idx)),squeeze=False)
 for i in range(len(idx)):
  ax[i,0].imshow(profiles[0,i],origin='lower',extent=(-7.5,7.5,-7.5,7.5),cmap='gray');ax[i,0].set_ylabel(f'({scene.z["cx"][idx[i]]},{scene.z["cy"][idx[i]]})')
  for j in range(1,len(frames)):
   im=ax[i,j].imshow(profiles[j,i]-profiles[0,i],origin='lower',extent=(-7.5,7.5,-7.5,7.5),cmap='RdBu_r',vmin=-.005,vmax=.005)
  for j,frame in enumerate(frames):
   ax[i,j].set_xlim(-4,4);ax[i,j].set_ylim(-4,4);ax[i,j].tick_params(labelsize=6)
   if i==0:ax[i,j].set_title('Reference FFI 602' if j==0 else f'FFI {frame} − 602')
 fig.subplots_adjust(top=.94,right=.9,hspace=.18,wspace=.18);fig.colorbar(im,cax=fig.add_axes([.92,.2,.015,.6]),label='Difference of unit-flux pixel-integrated profiles')
 fig.suptitle('Independent full single-FFI fits: PSF-shape changes at matched origins and sampling\nSame nine locations/colours; WCS translations and explicit colour-origin changes removed; in-sample fitted models',fontsize=12)
 fig.savefig(R/'figures/full_profile_differences.png',dpi=130);plt.close(fig)
 print('comparison complete',frames)
if __name__=='__main__':main()
