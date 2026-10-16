"""Apply only differential catalogue proper motion to the baked reference geometry.
Preserves the reference bundle's projection/rounding convention; never reprojects float32 RA absolutely.
"""
import json
from pathlib import Path
import jax
import numpy as np
import pandas as pd
from syndiff_pipeline.forward_model import fit_bundle as FB,cheb_wcs as CW,gaia_pm as GP
CAT=Path('/astro/armin/koji/syndiff/dev_runs/paper_dataset_20261001/data_root/s0024/c2/k2/catalogs/gaia_catalog_s0024_2_2.csv')
def build(source,scene_z,out,target_btjd):
 source=Path(source);out=Path(out);b=FB.load_fit_bundle(source)
 ref=float(b.meta['gaia_pm_propagation']['target_btjd'])
 c=pd.read_csv(CAT,dtype={'source_id':'int64'}).set_index('source_id').reindex(scene_z['source_id'])
 good=np.isfinite(c.ra)&np.isfinite(c.dec)&np.isfinite(c.pmra)&np.isfinite(c.pmdec)
 c=c[good];idx=np.asarray(scene_z['star_bundle_index'])[good];assert len(np.unique(idx))==len(idx)
 ra0,dec0=GP.propagate_ra_dec(c.ra.to_numpy(),c.dec.to_numpy(),c.pmra.to_numpy(),c.pmdec.to_numpy(),GP.btjd_to_time(ref))
 ra1,dec1=GP.propagate_ra_dec(c.ra.to_numpy(),c.dec.to_numpy(),c.pmra.to_numpy(),c.pmdec.to_numpy(),GP.btjd_to_time(target_btjd))
 with jax.enable_x64(True):
  x0,y0,a0=map(np.asarray,CW.star_basis(ra0,dec0,b.cheb_static));x1,y1,a1=map(np.asarray,CW.star_basis(ra1,dec1,b.cheb_static))
 del b
 raw=dict(np.load(source));dx=x1-x0;dy=y1-y0
 for key,delta in [('x_lin',dx),('y_lin',dy),('cheb_basis',a1-a0)]:
  v=raw[key].copy();v[idx]=(v[idx].astype(float)+delta).astype(v.dtype);raw[key]=v
 np.savez(out,**raw)
 meta=json.loads(source.with_name('fit_bundle_meta.json').read_text())
 meta['temporal_differential_pm']=dict(reference_btjd=ref,target_btjd=float(target_btjd),matched_finite_pm=len(idx),unavailable=len(good)-int(good.sum()),method='float64 delta on catalogue propagated to both epochs, added to original baked float32 projection/basis',max_linear_shift_px=float(np.max(np.hypot(dx,dy))),source=str(source))
 out.with_name('fit_bundle_meta.json').write_text(json.dumps(meta,indent=1))
 return meta['temporal_differential_pm']
