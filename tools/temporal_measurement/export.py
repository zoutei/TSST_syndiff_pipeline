"""Export native scene residuals using saved joint fluxes and the exact recorded colour model.
No photometric flattening. WCS positions are achromatic model origins, not absolute sky truth.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import jax
import jax.numpy as jnp
import argparse,json
from pathlib import Path
import numpy as np
from syndiff_pipeline.forward_model import scene_fit as SF,fit as FIT,loss as L,epsf_model as EM
from syndiff_pipeline.forward_model.crossfit.score_oof import configure_model_env
from syndiff_pipeline.forward_model.diagnostics import stacked_temporal_residuals as STR
R=Path('/astro/armin/koji/syndiff/dev_runs/s24_temporal_measurement_20261016')

def export(frame):
 out=R/'frames'/f'f{frame:04d}';fdir=out/'wcs_fit';meta=json.loads((fdir/'fit_meta.json').read_text())
 configure_model_env(meta);EM.set_radial_knots(meta['chroma_radial_knots']);EM.set_coma_knots(meta['chroma_coma_knots']);EM.set_radial_mode(meta['chroma_radial_mode'])
 scene=SF.Scene(out/'scene');scene.use_colour_file(meta['colour_file']);z=scene.z
 params=FIT.load_params_npz(fdir/'params.npz');saved=np.load(fdir/'flux_solved.npz');assert np.array_equal(saved['source_id'],z['source_id'])
 flux=np.asarray(saved['flux']);S=scene.S;T=np.zeros((scene.N,S*S),np.float32);x=np.zeros(scene.N);y=x.copy()
 ctxs=scene.contexts(meta['colour_ref'],chroma_axis=tuple(meta['chroma_axis']),chroma_g8_gauge=meta['chroma_g8_gauge'],chroma_g8_no_dil=meta.get('chroma_g8_no_dil',False),chroma_g8_extras=SF.g8_extras_tuple(meta['chroma_g8_extras']),delta2_mean=meta['chroma_delta2_mean'],**SF.meta_colour_kwargs(meta))
 for role,idx,ctx in ctxs:
  a,xx,yy,_=jax.jit(lambda p:L.forward_model(p,ctx))(params);i=np.asarray(idx)
  T[i]=np.asarray(a).reshape(len(i),S*S);x[i]=np.asarray(xx).reshape(-1);y[i]=np.asarray(yy).reshape(-1)
 uid=z['uid'];own=flux[:,None]*T
 union=np.bincount(uid.ravel(),weights=own.ravel(),minlength=scene.U+1);model=union[uid];res=z['data']-model
 valid=z['valid'].astype(bool)&(uid<scene.U);contrib=(z['role']!=SF.ROLE_NUISANCE)&(flux>0)
 # Catalogue-selected display cohort is fixed by reference IDs/magnitude/position, not residual quality.
 ii=np.flatnonzero(contrib);bp=np.asarray(scene.src.bp_rp)[z['star_bundle_index']]
 k=np.arange(S*S);px=z['cx'][:,None]+k%S-S//2;py=z['cy'][:,None]+k//S-S//2
 np.savez_compressed(out/'native.npz',data=z['data'],model=model.astype(np.float32),own=own.astype(np.float32),residual=res.astype(np.float32),valid=valid,flux=flux,source_id=z['source_id'],role=z['role'],tess_mag=z['tess_mag'],colour=bp,x=x,y=y,cx=z['cx'],cy=z['cy'],stamp=S,noise=z['noise'])
 # Canonical grid stores scene residual/model and BP-RP, re-centred on fitted WCS origins.
 # It is an in-sample diagnostic, not an independent shape validation.
 RR=np.full((len(ii),33,33),np.nan,np.float32);MM=RR.copy()
 for j,i in enumerate(ii):
  args=(np.array([x[i]]),np.array([y[i]]),px[i],py[i],valid[i].astype(float)[None])
  RR[j]=STR.rasterize_packed_values((res[i]/flux[i])[None],*args,oversample=4,half_width=4)[0]
  MM[j]=STR.rasterize_packed_values((model[i]/flux[i])[None],*args,oversample=4,half_width=4)[0]
 np.savez_compressed(out/'raster_res_model.npz',res=RR,model=MM,flux=flux[ii],colour=bp[ii],x=x[ii],y=y[ii],role=z['role'][ii],tess_mag=z['tess_mag'][ii],source_id=z['source_id'][ii])
 summary=dict(frame=frame,n_sources=scene.N,n_nuisance=int((z['role']==2).sum()),n_display_sources=len(ii),stage_outcomes=meta.get('stage_outcomes'),positions='achromatic fitted WCS origins; common frozen chromatic profile; not Gaussian centres',flux='saved weighted simultaneous island solution, not re-solved unweighted',shape='reference fixed; residual images measure remaining mismatch; not proof of intrinsic temporal PSF alone')
 (out/'export.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('frame',type=int);a=p.parse_args();export(a.frame)
