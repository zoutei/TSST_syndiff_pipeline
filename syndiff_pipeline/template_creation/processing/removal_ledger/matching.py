"""Lossless catalogue identities and conservative Gaia/PS1 associations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _unit(ra,dec):
    ra,dec=np.deg2rad(ra),np.deg2rad(dec)
    return np.column_stack([np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)])


def _numeric(table,name):
    return pd.to_numeric(table[name],errors='coerce').replace(-999,np.nan) if name in table else pd.Series(np.nan,index=table.index)


def gaia_sources(table,wcs,*,epoch_year=None):
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    out=table.copy();out['gaia_id']=out.source_id.astype('string')
    out['source_key']='gaia:'+out.gaia_id
    out['entity_key']=out.source_key
    out['catalogue']='gaia_dr3'
    out['original_ra']=out.ra;out['original_dec']=out.dec
    out['position_status']='catalogue_epoch_2016'
    if epoch_year is not None:
        good=np.isfinite(out.pmra)&np.isfinite(out.pmdec)
        if good.any():
            sky=SkyCoord(ra=out.loc[good,'ra'].values*u.deg,dec=out.loc[good,'dec'].values*u.deg,
                pm_ra_cosdec=out.loc[good,'pmra'].values*u.mas/u.yr,pm_dec=out.loc[good,'pmdec'].values*u.mas/u.yr,
                obstime=Time(2016.,format='jyear'))
            moved=sky.apply_space_motion(Time(epoch_year,format='jyear'))
            out.loc[good,'ra']=moved.ra.deg;out.loc[good,'dec']=moved.dec.deg
        out['position_status']=np.where(good,'proper_motion_propagated','missing_motion_unpropagated')
    out['position_epoch_year']=2016. if epoch_year is None else epoch_year
    out['pixel_x'],out['pixel_y']=wcs.all_world2pix(out.ra.values,out.dec.values,0)
    return out


def ps1_sources(table,wcs):
    """Keep stack measurements, including split detections sharing one objID."""
    out=table.copy()
    out['ps1_obj_id']=out.objID.astype('string')
    out['ps1_detection_id']=out.uniquePspsSTid.astype('string')
    out['source_key']='ps1det:'+out.ps1_detection_id
    out['entity_key']='ps1:'+out.ps1_obj_id
    out['catalogue']='ps1_dr2_stack'
    ra,dec=_numeric(out,'raStack'),_numeric(out,'decStack')
    meanra,meandec=_numeric(out,'raMean'),_numeric(out,'decMean')
    stackok=np.isfinite(ra)&np.isfinite(dec)&(abs(dec)<=90)
    out['ra']=np.where(stackok,ra,meanra);out['dec']=np.where(stackok,dec,meandec)
    out['position_status']=np.where(stackok,'stack_measurement','mean_fallback')
    out['pixel_x'],out['pixel_y']=wcs.all_world2pix(out.ra.values,out.dec.values,0)
    out['split_object_measurements']=out.ps1_obj_id.map(out.ps1_obj_id.value_counts())
    return out


def match_gaia_ps1(gaia,ps1,*,candidate_radius_arcsec=2.0,min_calibrators=12):
    """All candidates, epoch-aware positions, and calibrated unique pairings.

    Missing motion/epoch is an explicit unconfirmed state. Catalogue identity
    links never modify the stack coordinates used to intersect deleted pixels.
    """
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    import warnings
    if candidate_radius_arcsec<=0:raise ValueError('Positive search radius required')
    g=gaia[np.isfinite(gaia.ra)&np.isfinite(gaia.dec)].reset_index(drop=True)
    p=ps1.sort_values('ps1_detection_id').drop_duplicates('ps1_obj_id').copy()
    r=_numeric(p,'raMean');d=_numeric(p,'decMean');mjd=_numeric(p,'epochMean')
    ok=np.isfinite(r)&np.isfinite(d)&(abs(d)<=90)
    p['match_ra']=np.where(ok,r,p.ra);p['match_dec']=np.where(ok,d,p.dec)
    p['match_epoch']=np.where(ok & (mjd>40000) & (mjd<80000),2000.+(mjd-51544.5)/365.25,np.nan)
    p=p[np.isfinite(p.match_ra)&np.isfinite(p.match_dec)].reset_index(drop=True)
    cols=['gaia_key','ps1_entity_key','ps1_obj_id','distance_arcsec','calibrated_distance_arcsec','epoch_propagated','status']
    if len(g)==0 or len(p)==0:return pd.DataFrame(columns=cols),dict(status='insufficient_catalogue',accepted=0)
    pmra=_numeric(g,'pmra').to_numpy();pmdec=_numeric(g,'pmdec').to_numpy()
    motion=np.isfinite(pmra)&np.isfinite(pmdec)
    ra0=g.original_ra.to_numpy() if 'original_ra' in g else g.ra.to_numpy()
    dec0=g.original_dec.to_numpy() if 'original_dec' in g else g.dec.to_numpy()
    epochs=p.match_epoch.to_numpy();finite_epochs=np.isfinite(epochs)
    epoch=float(np.median(epochs[finite_epochs])) if finite_epochs.any() else 2016.
    def propagate(indices,new_epoch):
        ri,di=ra0[indices].copy(),dec0[indices].copy()
        good=motion[indices]&np.isfinite(new_epoch)
        if good.any():
            sky=SkyCoord(ra=ri[good]*u.deg,dec=di[good]*u.deg,
                pm_ra_cosdec=pmra[indices][good]*u.mas/u.yr,pm_dec=pmdec[indices][good]*u.mas/u.yr,
                obstime=Time(2016.,format='jyear'))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                moved=sky.apply_space_motion(Time(np.asarray(new_epoch)[good],format='jyear'))
            ri[good]=moved.ra.deg;di[good]=moved.dec.deg
        return ri,di,good
    gr,gd,_=propagate(np.arange(len(g)),np.full(len(g),epoch))
    max_span=float(np.max(abs(epochs[finite_epochs]-epoch))) if finite_epochs.any() else 0.
    max_pm=float(np.max(np.hypot(pmra[motion],pmdec[motion])))/1000 if motion.any() else 0.
    search_radius=candidate_radius_arcsec+max_span*max_pm
    candidates=cKDTree(_unit(gr,gd)).query_ball_point(_unit(p.match_ra.to_numpy(),p.match_dec.to_numpy()),
        2*np.sin(np.deg2rad(search_radius/3600)/2))
    pairs=[(i,j) for i,cs in enumerate(candidates) for j in cs]
    if not pairs:return pd.DataFrame(columns=cols),dict(status='no_candidates',accepted=0)
    ii,jj=np.asarray(pairs,dtype=int).T
    pair_ra,pair_dec,propagated=propagate(jj,epochs[ii])
    # For missing epochs, preserve the candidate search reference epoch and do
    # not promote the identity to an accepted match.
    pair_ra=np.where(np.isfinite(epochs[ii]),pair_ra,gr[jj]);pair_dec=np.where(np.isfinite(epochs[ii]),pair_dec,gd[jj])
    dr=((p.match_ra.to_numpy()[ii]-pair_ra+180)%360-180)*np.cos(np.deg2rad(pair_dec))*3600
    dd=(p.match_dec.to_numpy()[ii]-pair_dec)*3600
    sep=np.hypot(dr,dd);keep=sep<=candidate_radius_arcsec
    ii,jj,dr,dd,sep,propagated=[v[keep] for v in [ii,jj,dr,dd,sep,propagated]]
    pg=np.bincount(ii,minlength=len(p));gg=np.bincount(jj,minlength=len(g))
    unique=(pg[ii]==1)&(gg[jj]==1)
    mag=_numeric(g,'phot_g_mean_mag').to_numpy()
    calibrator=unique & propagated & (mag[jj]>13)&(mag[jj]<18)
    offsets=np.column_stack([dr[calibrator],dd[calibrator]])
    calibrated=len(offsets)>=min_calibrators
    if calibrated:
        offset=np.median(offsets,axis=0);residual=np.linalg.norm(offsets-offset,axis=1)
        scale=1.4826*np.median(abs(residual-np.median(residual)))
        tolerance=min(candidate_radius_arcsec,max(.3,float(np.median(residual)+5*scale)))
    else:offset=np.zeros(2);tolerance=0.
    corrected=np.hypot(dr-offset[0],dd-offset[1]);rows=[]
    for k,(i,j) in enumerate(zip(ii,jj)):
        status='ambiguous'
        if unique[k]:status='accepted_unique_calibrated' if calibrated and propagated[k] and corrected[k]<=tolerance else 'unconfirmed_candidate'
        rows.append(dict(gaia_key=g.source_key.iloc[j],ps1_entity_key=p.entity_key.iloc[i],ps1_obj_id=p.ps1_obj_id.iloc[i],
            distance_arcsec=float(sep[k]),calibrated_distance_arcsec=float(corrected[k]),epoch_propagated=bool(propagated[k]),status=status))
    out=pd.DataFrame(rows,columns=cols)
    return out,dict(status='calibrated' if calibrated else 'insufficient_calibrators',calibrators=len(offsets),
        offset_ra_cosdec_arcsec=float(offset[0]),offset_dec_arcsec=float(offset[1]),tolerance_arcsec=tolerance,
        candidate_radius_arcsec=candidate_radius_arcsec,search_radius_arcsec=search_radius,
        accepted=int((out.status=='accepted_unique_calibrated').sum()),
        caveat='PS1 mean-position epoch used for identities; stack image positions retain their own uncertainty and may represent epoch mixtures.')


def attach_entities(sources,matches):
    out=sources.copy()
    accepted=matches[matches.status=='accepted_unique_calibrated']
    aliases=accepted.set_index('ps1_entity_key').gaia_key.to_dict()
    out['canonical_entity_key']=out.entity_key.map(aliases).fillna(out.entity_key)
    out['identity_status']=np.where(out.entity_key.isin(aliases),'gaia_ps1_candidate_identity',
                                    np.where(out.catalogue=='gaia_dr3','gaia','ps1_only_or_unresolved'))
    return out
