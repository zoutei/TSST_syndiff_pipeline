"""Native-pixel local shift/shape response measurements, with no temporal model.
Coefficients are conditional linearized profile responses, not independent optical truth.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import jax
import jax.numpy as jnp
from dataclasses import replace
from pathlib import Path
import argparse,json
import numpy as np
from syndiff_pipeline.forward_model import scene_fit as SF,fit as FIT,loss as L,epsf_model as EM
from syndiff_pipeline.forward_model.crossfit.score_oof import configure_model_env
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')
OFF=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]],float)
def responses(samples,h,shift_samples=None,shift_h=None,wide_samples=None):
 p,xp,xm,yp,ym,pp,pm,mp,mm=samples
 dx=(xp-xm)/(2*h);dy=(yp-ym)/(2*h)
 if shift_samples is not None:
  dx=(shift_samples[1]-shift_samples[2])/(2*shift_h);dy=(shift_samples[3]-shift_samples[4])/(2*shift_h)
 out=np.stack([p,dx,dy,.5*(xp-2*p+xm)/h**2,.5*(yp-2*p+ym)/h**2,(pp-pm-mp+mm)/(4*h**2)],axis=-1)
 if wide_samples is not None:out[...,3:6]=(4*out[...,3:6]-responses(wide_samples,2*h)[...,3:6])/3
 return out
def solve(A,d,sigma,valid):
 w=valid&np.isfinite(d)&np.isfinite(sigma)&(sigma>0)&np.isfinite(A).all(axis=1)
 if w.sum()<3*A.shape[1]:return np.full(A.shape[1],np.nan),np.nan
 a=A[w]/sigma[w,None];b=d[w]/sigma[w];scale=np.sqrt(np.sum(a*a,axis=0));scale=np.maximum(scale,1e-30)
 x,_,rank,s=np.linalg.lstsq(a/scale,b,rcond=1e-10)
 cond=s[0]/s[-1] if s[-1]>0 else np.inf
 if rank<A.shape[1] or cond>1e8:return np.full(A.shape[1],np.nan),cond
 return x/scale,cond

def measure(frame,fit_kind='wcs_fit',analysis_subdir=''):
 root=R/'frames'/f'f{frame:04d}';out=root/analysis_subdir;meta=json.loads((root/fit_kind/'fit_meta.json').read_text());z=np.load(out/'native.npz');scene=SF.Scene(root/'scene');scene.use_colour_file(meta['colour_file'])
 configure_model_env(meta);EM.set_radial_knots(meta['chroma_radial_knots']);EM.set_coma_knots(meta['chroma_coma_knots']);EM.set_radial_mode(meta['chroma_radial_mode'])
 p=FIT.load_params_npz(root/fit_kind/'params.npz');h=.25
 ctxs=scene.contexts(meta['colour_ref'],chroma_axis=tuple(meta['chroma_axis']),chroma_g8_gauge=meta['chroma_g8_gauge'],chroma_g8_no_dil=meta.get('chroma_g8_no_dil',False),chroma_g8_extras=SF.g8_extras_tuple(meta['chroma_g8_extras']),delta2_mean=meta['chroma_delta2_mean'],**SF.meta_colour_kwargs(meta))
 S=int(z['stamp']);k=np.arange(S*S);dx=z['cx'][:,None]+k%S-S//2-z['x'][:,None];dy=z['cy'][:,None]+k//S-S//2-z['y'][:,None];rr=np.hypot(dx,dy)
 refrr=np.hypot(z['cx'][:,None]+k%S-S//2-scene.z['x0'][:,None],z['cy'][:,None]+k//S-S//2-scene.z['y0'][:,None])
 use=(z['role']!=2)&(z['tess_mag']>=8.5)&(z['tess_mag']<11.5)&(z['valid']&(refrr<3)).sum(1).__ge__(20)
 indices=np.flatnonzero(use);co=np.full((3,scene.N,9),np.nan);condition=np.full((3,scene.N),np.nan);pred=np.full((scene.N,S*S),np.nan,np.float32);held=np.full((2,scene.N),np.nan)
 for role,idx,ctx in ctxs:
  if role==2:continue
  idx=np.asarray(idx);take=np.flatnonzero(use[idx]);stars=idx[take]
  if not len(stars):continue
  @jax.jit
  def render(pp,off):
   c=replace(ctx,stamp_center_x=ctx.stamp_center_x-off[0],stamp_center_y=ctx.stamp_center_y-off[1])
   return L.forward_model(pp,c)[0].reshape(len(idx),S*S)
  samples=np.asarray([np.asarray(render(p,jnp.asarray(off*h,np.float32)))[take].astype(float) for off in OFF])
  small=np.asarray([np.asarray(render(p,jnp.asarray(off*.03,np.float32)))[take].astype(float) for off in OFF[:5]])
  wide=np.asarray([np.asarray(render(p,jnp.asarray(off*(2*h),np.float32)))[take].astype(float) for off in OFF])
  basis=responses(samples,h,small,.03,wide)
  for j,i in enumerate(stars):
   flux=float(z['flux'][i]);
   if flux<=0:continue
   A=np.column_stack([basis[j],np.ones(S*S),dx[i],dy[i]])
   valid=z['valid'][i]&(refrr[i]<=5);d=z['residual'][i]/flux;sigma=z['noise'][i]/flux
   # Detector-fixed checkerboard blocks. All results retained, failures stay NaN.
   parity=((z['cx'][i]+k%S-S//2)//2+(z['cy'][i]+k//S-S//2)//2)%2
   for half,m in enumerate([valid,valid&(parity==0),valid&(parity==1)]):co[half,i],condition[half,i]=solve(A,d,sigma,m)
   pred[i]=(A@co[0,i]*flux).astype(np.float32)
   for half in [0,1]:
    train=valid&(parity==half);test=valid&(parity!=half)
    cols=[0,1,2,6,7,8];csmall,_=solve(A[:,cols],d,sigma,train)
    if test.sum() and np.isfinite(csmall).all() and np.isfinite(co[half+1,i]).all():
     e0=np.mean(((d[test]-A[test][:,cols]@csmall)/sigma[test])**2)
     e1=np.mean(((d[test]-A[test]@co[half+1,i])/sigma[test])**2)
     held[half,i]=e1/e0 if e0>0 else np.nan
  print('response role',role,'sources',len(stars),flush=True)
 np.savez_compressed(out/'shape_response.npz',source_id=z['source_id'][indices],index=indices,coeff=co[:,indices],condition=condition[:,indices],x=z['x'][indices],y=z['y'][indices],tess_mag=z['tess_mag'][indices],colour=z['colour'][indices],h=h,shift_h=.03,predicted_residual=pred[indices],held_shape_over_shift_mse=held[:,indices])
 (out/'shape_response.json').write_text(json.dumps(dict(coefficients=['fractional amplitude','local dx px','local dy px','dSigma_xx px2','dSigma_yy px2','dSigma_xy px2','background/flux','x background gradient/flux','y background gradient/flux'],halves=['all','checkerboard0','checkerboard1'],n_selected=len(indices),n_valid=[int(np.isfinite(co[h,indices,0]).sum()) for h in range(3)],scope='Richardson finite-grid shape response h=0.25/0.5 px, position derivative h=0.03 px; local response of latest chromatic reference profile; source neighbours jointly fit beforehand; not a temporal mode or final astrometric/photometric validation'),indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('frame',type=int);a.add_argument('--fit-kind',default='wcs_fit');a.add_argument('--analysis-subdir',default='');v=a.parse_args();measure(v.frame,v.fit_kind,v.analysis_subdir)
