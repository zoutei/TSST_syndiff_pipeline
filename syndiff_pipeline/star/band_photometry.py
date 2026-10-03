"""Manifest-driven per-band target add-back and multi-epoch forced photometry.

Consumes transported retained components and frozen per-frame profiles. All
inputs are explicit; provisional products remain labelled in every output.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import shift
from scipy.spatial import cKDTree
from .band_joint import solve


def _npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def _image(path,hdu=1):
    p=Path(path)
    if p.suffix=='.npy':return np.load(p,allow_pickle=False)
    return np.asarray(fits.getdata(p,hdu))


def _shift(profile,x,y,cx,cy):
    return shift(profile,(int(round(y))-cy,int(round(x))-cx),order=0,mode='constant',cval=0,prefilter=False)


def run(manifest_path):
    manifest_path=Path(manifest_path).resolve();cfg=json.loads(manifest_path.read_text())
    if cfg.get('schema_version')!=1:raise ValueError('Unsupported band photometry manifest schema')
    state=cfg.get('artifact_state')
    if state not in ('provisional','validated'):raise ValueError('Declare artifact_state explicitly')
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    targets=pd.read_csv(cfg['targets_csv'],dtype={'objID':str,'gaia_source_id':str})
    if not targets.target_index.is_unique or not targets.objID.is_unique:raise ValueError('Duplicate targets')
    zp=float(cfg['flux_zero_point']);outputs=[];stamp_ids=[];stamp_frames=[];stamps=[];stamp_good=[]
    profiles_cache={};neighbours_cache={}
    for frame in cfg['frames']:
        pf=frame.get('profiles_npz',cfg['profiles_npz'])
        if pf not in profiles_cache:
            loaded=_npz(pf)
            if cfg.get('profile_normalization')=='unit_sum_full_support':
                sums=loaded['profile'].sum(axis=(1,2))
                if np.any(~np.isfinite(sums)|(sums<=0)):raise ValueError('Invalid full-profile normalization')
                loaded['profile']=loaded['profile']/sums[:,None,None]
            profiles_cache[pf]=loaded
        P=profiles_cache[pf]
        nf=frame.get('neighbour_profiles_npz',cfg.get('neighbour_profiles_npz'))
        N=None;tree=None
        if nf:
            if nf not in neighbours_cache:
                N=_npz(nf)
                if cfg.get('profile_normalization')=='unit_sum_full_support':
                    sums=N['profile'].sum(axis=(1,2))
                    if np.any(~np.isfinite(sums)|(sums<=0)):raise ValueError('Invalid neighbour-profile normalization')
                    N['profile']=N['profile']/sums[:,None,None]
                neighbours_cache[nf]=(N,cKDTree(np.c_[N['x'],N['y']]))
            N,tree=neighbours_cache[nf]
        image=_image(frame['difference_fits'],frame.get('difference_hdu',1))
        noise=_image(frame['noise_fits'],frame.get('noise_hdu',2))
        ox,oy=frame.get('noise_origin_xy',[0,0]);noise=noise[oy:oy+image.shape[0],ox:ox+image.shape[1]]
        mask=_image(frame['physical_mask_fits'],frame.get('mask_hdu',1)).astype(np.int64)
        scale=_image(frame['scale_image']) if 'scale_image' in frame else np.full(image.shape,float(frame['scale']))
        if any(a.shape!=image.shape for a in (noise,mask,scale)):raise ValueError('Mismatched frame geometry')
        allowed=0
        for bit in set(cfg.get('allowed_physical_mask_bit_indices',[5])):allowed |= 1<<int(bit)
        if allowed&2 and N is None:raise ValueError('Bright-circle pixels require joint neighbour profiles')
        raw_control=None
        if 'raw_science_fits' in frame:
            raw=_image(frame['raw_science_fits'],frame.get('raw_hdu',1))
            raw_control=np.asarray(raw[oy:oy+image.shape[0],ox:ox+image.shape[1]],float)
            if 'background_fits' in frame:raw_control-=_image(frame['background_fits'])
            if raw_control.shape!=image.shape:raise ValueError('Raw-control geometry mismatch')
        half=int(cfg.get('stamp_half_size',10));size=2*half+1
        for row in targets.itertuples():
            k=int(row.target_index)
            if str(P['objID'][k])!=row.objID:raise ValueError('Profile/target identity mismatch')
            x,y=float(P['x'][k]),float(P['y'][k])
            rec=dict(target_index=k,objID=row.objID,gaia_source_id=row.gaia_source_id,frame=frame['id'],time_btjd=frame.get('time_btjd'),artifact_state=state,reference_profile_reused=bool(frame.get('reference_profile_reused',False)),x=x,y=y)
            for name in ('tmag_ps1','tmag_ps1_stat_err','r_minus_z','ccd_cell','mag_bin'):
                if hasattr(row,name):rec[name]=getattr(row,name)
            paths=[Path(root)/f'{k:04d}.npz' for root in cfg['transport_dirs']]
            path=next((p for p in paths if p.is_file()),None)
            if path is None:rec['status']='no_transport';outputs.append(rec);continue
            try:
                t=_npz(path);x0,x1,y0,y1=map(int,t['bounds']);sl=np.s_[y0:y1,x0:x1]
                if min(x0,y0)<0 or x1>image.shape[1] or y1>image.shape[0] or (x1-x0,y1-y0)!=(size,size):raise ValueError('Invalid stamp bounds')
                cx,cy=x0+half,y0+half;p=_shift(P['profile'][k],x,y,cx,cy)
                near=[] if tree is None else tree.query_ball_point([x,y],float(cfg.get('neighbour_radius_px',14)))
                ps=[p]
                for j in near:
                    if str(N['source_id'][j])==row.gaia_source_id:continue
                    q=_shift(N['profile'][j],N['x'][j],N['y'][j],cx,cy)
                    if np.sum(np.abs(q))>.001:ps.append(q)
                good=(mask[sl]&~allowed)==0;yy,xx=np.mgrid[y0:y1,x0:x1]
                for j in near:
                    if N['tmag'][j]<cfg.get('hard_bright_tmag',9):
                        good &= (xx-N['x'][j])**2+(yy-N['y'][j])**2>float(cfg.get('hard_bright_radius_px',6))**2
                add=np.asarray(t['addback_unscaled'])*scale[sl]
                if add.shape!=p.shape or not np.isfinite(add).all():raise ValueError('Invalid transported component')
                restored=image[sl]+add;ps=np.asarray(ps)
                fit,model=solve(restored,noise[sl],ps,good)
                before,_=solve(image[sl],noise[sl],ps,good);added,_=solve(add,noise[sl],ps,good)
                flux=fit['flux'];mag=zp-2.5*np.log10(flux) if flux>0 else np.nan
                rec.update(**fit,status='ok',snr=flux/fit['flux_err'],measured_mag=mag,difference_flux=before['flux'],added_flux=added['flux'],flux_closure=flux-before['flux']-added['flux'],low_support=fit['profile_support']*float(p.sum())<.7,profile_support_total=fit['profile_support']*float(p.sum()),profile_sum=float(p.sum()),transport_path=str(path))
                if hasattr(row,'tmag_ps1'):
                    expected=10**(.4*(zp-row.tmag_ps1));rec.update(expected_flux=expected,flux_ratio=flux/expected,delta_mag=mag-row.tmag_ps1)
                raw_planes=[np.full(p.shape,np.nan)]*3
                if raw_control is not None:
                    raw_fit,raw_model=solve(raw_control[sl],noise[sl],ps,good)
                    rec.update(raw_flux=raw_fit['flux'],raw_flux_err=raw_fit['flux_err'],raw_chi2=raw_fit['chi2'],raw_minus_restored_flux=raw_fit['flux']-flux)
                    raw_planes=[raw_control[sl],raw_model,raw_control[sl]-raw_model]
                if cfg.get('write_stamps',True):
                    stamp_ids.append(k);stamp_frames.append(frame['id']);stamps.append(np.array([image[sl],add,restored,model,restored-model]+raw_planes));stamp_good.append(good)
            except (ValueError,OSError,KeyError) as exc:rec.update(status='failed',reason=str(exc))
            outputs.append(rec)
    table=pd.DataFrame(outputs);table.to_csv(out/'measurements.csv',index=False)
    lcdir=out/'lightcurves';lcdir.mkdir(exist_ok=True)
    for obj,part in table.groupby('objID',sort=False):part.sort_values('time_btjd').to_csv(lcdir/f'{obj}.csv',index=False)
    table.groupby(['objID','status'],dropna=False).size().rename('n_epochs').reset_index().to_csv(out/'batch_manifest.csv',index=False)
    if stamps:np.savez_compressed(out/'stamps.npz',target_index=stamp_ids,frame=np.array(stamp_frames,dtype='U'),stamps=np.array(stamps),good=np.array(stamp_good),planes=np.array(['difference','addback','restored','model','residual','raw','raw_model','raw_residual']))
    record=dict(schema_version=1,manifest=str(manifest_path),manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),artifact_state=state,n_targets=len(targets),n_frames=len(cfg['frames']),n_measurements=len(table),status_counts=table.status.value_counts().to_dict(),uncertainties='Conditional pixel-noise errors including joint-neighbour covariance; template/PSF/calibration uncertainty not included')
    (out/'run_record.json').write_text(json.dumps(record,indent=2));(out/'input_manifest.json').write_text(json.dumps(cfg,indent=2))
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);a=p.parse_args(argv)
    print(json.dumps(run(a.manifest),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
