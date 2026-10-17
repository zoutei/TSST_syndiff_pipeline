"""Independent PSFs at common locations/colours, retaining both origin conventions.
Only models are sampled and recentered; native observed pixels and fitted predictions are unchanged.
"""
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # diagnostic model-coordinate arithmetic only
from dataclasses import replace
from pathlib import Path
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from syndiff_pipeline.forward_model import scene_fit as SF,fit as FIT,loss as L,epsf_model as EM
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
NS=int(os.environ.get('S24_DIAGNOSTIC_PROFILE_OS','4'));NG=15*NS
def plot(profiles,frames,scene,idx,name,title):
 fig,ax=plt.subplots(len(idx),len(frames),figsize=(max(12,3*len(frames)),2.3*len(idx)),squeeze=False)
 for i in range(len(idx)):
  ax[i,0].imshow(profiles[0,i],origin='lower',extent=(-7.5,7.5,-7.5,7.5),cmap='gray');ax[i,0].set_ylabel(f'({scene.z["cx"][idx[i]]},{scene.z["cy"][idx[i]]})')
  for j in range(1,len(frames)):im=ax[i,j].imshow(profiles[j,i]-profiles[0,i],origin='lower',extent=(-7.5,7.5,-7.5,7.5),cmap='RdBu_r',vmin=-.005,vmax=.005)
  for j,frame in enumerate(frames):
   ax[i,j].set_xlim(-4,4);ax[i,j].set_ylim(-4,4);ax[i,j].tick_params(labelsize=6)
   if i==0:ax[i,j].set_title('Reference FFI 602' if j==0 else f'FFI {frame} − 602')
 fig.subplots_adjust(top=.94,right=.88,hspace=.18,wspace=.18);fig.colorbar(im,cax=fig.add_axes([.91,.2,.012,.6]),label='Difference of unit-flux pixel-integrated profiles')
 fig.suptitle(title+'\nSame nine locations and colours; in-sample models, not independent optical truth',fontsize=11)
 fig.savefig(R/'figures'/name,dpi=130,bbox_inches='tight');plt.close(fig)
def main():
 frames=[i for i in [602,1,469,1225] if (R/'frames'/f'f{i:04d}'/'full_fit/params.npz').exists()];assert 602 in frames and len(frames)>=2,frames
 scene=SF.Scene(R/'frames/f0602/scene');idx=np.load(R/'inputs/display_sources.npz')['index'];scene.role[:]=2;scene.role[idx]=0
 base=FIT.load_params_npz(R/'frames/f0602/full_fit/params.npz');profiles=[];aligned=[];centres=[];residual_centres=[];offset=(np.arange(NS)-(NS-1)/2)/NS
 v=(np.arange(NG)-(NG-1)/2)/NS;X,Y=np.meshgrid(v,v);sigma=.75;w=np.exp(-.5*(X*X+Y*Y)/sigma**2)
 for frame in frames:
  fit=R/'frames'/f'f{frame:04d}'/'full_fit';m=json.loads((fit/'fit_meta.json').read_text());scene.use_colour_file(m['colour_file'])
  EM.set_radial_knots(m['chroma_radial_knots']);EM.set_coma_knots(m['chroma_coma_knots']);EM.set_radial_mode(m['chroma_radial_mode']);p=FIT.load_params_npz(fit/'params.npz');p['wcs_coeff']=jnp.asarray(base['wcs_coeff'],jnp.float64)
  contexts=scene.contexts(m['colour_ref'],chroma_axis=tuple(m['chroma_axis']),chroma_g8_gauge=m['chroma_g8_gauge'],chroma_g8_extras=SF.g8_extras_tuple(m['chroma_g8_extras']),delta2_mean=m['chroma_delta2_mean'],**SF.meta_colour_kwargs(m));_,ids,c=next(t for t in contexts if t[0]==0)
  mapping={int(s):j for j,s in enumerate(np.asarray(ids))};order=np.array([mapping[int(s)] for s in idx]);_,xx,yy,_=L.forward_model(p,c);star_occ=c.members.reshape(-1)[c.occ_flat_idx];ch=L._colour_slot_terms(p,c,star_occ);xe=xx.reshape(-1)+ch[1];ye=yy.reshape(-1)+ch[2]
  @jax.jit
  def render(off,centre):return L.forward_model(p,replace(c,stamp_center_x=xe+centre[:,0]+off[0],stamp_center_y=ye+centre[:,1]+off[1]))[0].reshape(len(ids),15,15)
  def sample(centre):
   ctxcentre=np.zeros_like(centre);ctxcentre[order]=centre;P=np.zeros((len(idx),NG,NG),np.float32)
   for iy,dy in enumerate(offset):
    for ix,dx in enumerate(offset):P[:,iy::NS,ix::NS]=np.asarray(render(jnp.array([dx,dy],dtype=jnp.float64),jnp.asarray(ctxcentre,dtype=jnp.float64)))[order]
   return P
  centre=np.zeros((len(idx),2));P=sample(centre);profiles.append(P.copy())
  for iteration in range(12):
   q=P*w;mass=q.sum((-2,-1));mu=np.stack([(q*X).sum((-2,-1))/mass,(q*Y).sum((-2,-1))/mass],axis=-1)
   if np.max(np.abs(mu))<2e-6:break
   M=np.zeros((len(idx),2,2));M[:,0,0]=(q*X*X).sum((-2,-1))/mass;M[:,1,1]=(q*Y*Y).sum((-2,-1))/mass;M[:,0,1]=M[:,1,0]=(q*X*Y).sum((-2,-1))/mass
   step=np.linalg.solve(np.eye(2)[None]-M/sigma**2,mu[...,None])[...,0];centre+=np.clip(step,-.2,.2);P=sample(centre)
  aligned.append(P);centres.append(centre);residual_centres.append(mu);print('profile/core origin',frame,'iterations',iteration,'max weighted residual',float(np.max(abs(mu))),flush=True)
 profiles=np.asarray(profiles);aligned=np.asarray(aligned);centres=np.asarray(centres)
 np.savez_compressed(R/('full_profile_comparison.npz' if NS==4 else f'full_profile_comparison_os{NS}.npz'),frame=frames,source_id=scene.z['source_id'][idx],profile=profiles,profile_core_aligned=aligned,core_origin=centres,core_origin_residual=np.asarray(residual_centres),core_sigma=sigma,step=1/NS,coordinate_min=-(NG-1)/(2*NS),coordinate_arithmetic='float64 for diagnostic rendering; fitted model coefficient values unchanged; not production astrometric precision',alignment='common reference WCS/spatial coordinates and explicit-colour render origin; optional Gaussian-window core origin measured from model only')
 if NS!=4:
  print('sampling-check complete',NS,frames,flush=True);return
 plot(profiles,frames,scene,idx,'full_profile_differences.png','Independent full single-FFI models: differences at a common explicit render origin')
 plot(aligned,frames,scene,idx,'full_profile_core_aligned.png','Independent full single-FFI models: differences after matching Gaussian-window core origins')
 fig,ax=plt.subplots(1,2,figsize=(12,4))
 for j,frame in enumerate(frames[1:],1):
  dc=1000*(centres[j]-centres[0]);ax[0].plot(np.arange(1,len(idx)+1),dc[:,0],'o-',label=f'FFI {frame}');ax[1].plot(np.arange(1,len(idx)+1),dc[:,1],'o-',label=f'FFI {frame}')
 for k,a in enumerate(ax):a.axhline(0,color='.5',lw=.7);a.set(xlabel='Fixed location (same order as profile rows)',ylabel=('x' if k==0 else 'y')+' core-origin change [mpx]');a.legend()
 fig.suptitle('Model centre-convention changes relative to FFI 602; σ=0.75 px Gaussian window\nNumerical quadrature/model diagnostic, not an absolute astrometric truth measurement',fontsize=11);fig.tight_layout();fig.savefig(R/'figures/profile_origin_changes.png',dpi=140);plt.close(fig)
 print('comparison complete',frames,flush=True)
if __name__=='__main__':main()
