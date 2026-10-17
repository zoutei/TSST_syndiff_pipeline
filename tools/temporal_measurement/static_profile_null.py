"""Conditional sampling/weight counterfactual using a fixed independently refitted profile.
No neighbours, global WCS re-fit or full image-generation transfer is simulated.
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
from response import OFF,responses,solve
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016');BASE=Path('/astro/armin/koji/syndiff/dev_runs/e2e_final_recipe_20261007/run5/F1/refit')
def main():
 frames=[602,1,469,1225];idx=np.load(R/'inputs/display_sources.npz')['index'];m=json.loads((BASE/'fit_meta.json').read_text());b=FIT.load_params_npz(BASE/'params.npz');truth=FIT.load_params_npz(R/'frames/f0602/full_fit/params.npz')
 EM.set_radial_knots(m['chroma_radial_knots']);EM.set_coma_knots(m['chroma_coma_knots']);EM.set_radial_mode(m['chroma_radial_mode']);coeff=[];residual=[];truth_images=[]
 for frame in frames:
  root=R/'frames'/f'f{frame:04d}';scene=SF.Scene(root/'scene');scene.use_colour_file(m['colour_file']);scene.role[:]=2;scene.role[idx]=0
  ctxs=scene.contexts(m['colour_ref'],chroma_axis=tuple(m['chroma_axis']),chroma_g8_gauge=m['chroma_g8_gauge'],chroma_g8_extras=SF.g8_extras_tuple(m['chroma_g8_extras']),delta2_mean=m['chroma_delta2_mean'],**SF.meta_colour_kwargs(m));_,ids,c=next(t for t in ctxs if t[0]==0);order=np.array([list(np.asarray(ids)).index(i) for i in idx]);wcs=FIT.load_params_npz(root/'full_fit/params_stage1.npz')['wcs_coeff'];pb=dict(b,wcs_coeff=wcs);pt=dict(truth,wcs_coeff=wcs)
  @jax.jit
  def render(p,off):return L.forward_model(p,replace(c,stamp_center_x=c.stamp_center_x-off[0],stamp_center_y=c.stamp_center_y-off[1]))[0].reshape(len(ids),225)
  sample=lambda p,h,off=OFF:np.array([np.asarray(render(p,jnp.asarray(o*h,np.float32)))[order].astype(float) for o in off])
  a=sample(pb,.25);small=sample(pb,.03,OFF[:5]);wide=sample(pb,.5);basis=responses(a,.25,small,.03,wide);true=np.asarray(render(pt,jnp.zeros(2)))[order].astype(float)
  k=np.arange(225);dx=scene.z['cx'][idx,None]+k%15-7-scene.z['x0'][idx,None];dy=scene.z['cy'][idx,None]+k//15-7-scene.z['y0'][idx,None];valid=scene.z['valid'][idx]&(np.hypot(dx,dy)<=5);sigma=scene.z['noise'][idx].astype(float)
  out=[];rr=[]
  for j in range(len(idx)):
   w=valid[j]/sigma[j]**2;amp=np.sum(w*a[0,j]*true[j])/np.sum(w*a[0,j]**2)
   d=true[j]/amp-a[0,j];A=np.column_stack([basis[j],np.ones(225),dx[j],dy[j]]);co,_=solve(A,d,sigma[j],valid[j]);out.append(co);rr.append(d)
  coeff.append(out);residual.append(rr);truth_images.append(true);print('static-profile null',frame,flush=True)
 coeff=np.asarray(coeff);residual=np.asarray(residual);truth_images=np.asarray(truth_images);np.savez_compressed(R/'static_profile_null.npz',frame=frames,index=idx,coeff=coeff,residual=residual,truth=truth_images,scope='constant full-refit reference profile sampled at measured GPU-stage1 WCS/PM and per-frame weights, fitted with published reference profile; conditional isolated-source response only')
 fig,ax=plt.subplots(3,5,figsize=(14,8))
 for row,j in enumerate([0,4,6]):
  ax[row,0].imshow(truth_images[0,j].reshape(15,15),origin='lower',cmap='gray');ax[row,0].set_title('Fixed truth profile')
  for col,frame in enumerate(frames,1):
   im=ax[row,col].imshow(residual[col-1,j].reshape(15,15),origin='lower',cmap='RdBu_r',vmin=-.003,vmax=.003);ax[row,col].set_title(f'Null residual FFI {frame}')
  for a in ax[row]:a.set_xticks([]);a.set_yticks([])
 fig.subplots_adjust(top=.88,right=.9);fig.colorbar(im,cax=fig.add_axes([.92,.2,.012,.6]),label='Residual / fitted unit-source amplitude');fig.suptitle('Conditional constant-profile counterfactual: actual rendered input and native residuals\nTruth shape fixed; only measured pointing/PM and weights change. No full pipeline or neighbour retraining.',fontsize=11);fig.savefig(R/'figures/static_profile_null_images.png',dpi=130,bbox_inches='tight');plt.close(fig)
 delta=coeff-coeff[0:1];fig,ax=plt.subplots(1,2,figsize=(12,4))
 for i,f in enumerate(frames[1:],1):ax[0].plot(np.arange(1,10),1000*(delta[i,:,3]+delta[i,:,4]),'o-',label=f'FFI {f}');ax[1].plot(np.arange(1,10),1000*delta[i,:,2],'o-',label=f'FFI {f}')
 ax[0].set_ylabel('False covariance-trace change [10⁻³ px²]');ax[1].set_ylabel('False local y change [mpx]')
 for a in ax:a.axhline(0,color='.5',lw=.7);a.set_xlabel('Fixed location');a.legend()
 fig.suptitle('Apparent changes from static profile mismatch + sampling/weights alone\nConditional sensitivity test, not a bound on all real systematics',fontsize=11);fig.tight_layout();fig.savefig(R/'figures/static_profile_null_response.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()
