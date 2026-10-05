"""Calibrate catalogue-to-image offsets without modifying the image WCS."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def calibrate_image_positions(raw,quality_mask,sources,*,min_stars=12,isolation_px=25.,radius_px=8,max_offset_px=3.):
    """Centroid clean isolated Gaia stars; compare the same peaks to PS1 stacks.

    Rejected calibrators stay in the source catalogue. The calibration samples
    are returned for inspection. Insufficient/unstable calibration remains an
    explicit status and does not cause a guessed coordinate correction.
    """
    required={'source_key','canonical_entity_key','catalogue','pixel_x','pixel_y'}
    if not required<=set(sources):raise ValueError('Sources lack identity/position columns')
    s=sources.copy()
    finite=np.isfinite(s.pixel_x)&np.isfinite(s.pixel_y)
    entities=s[finite].sort_values(['catalogue','source_key']).drop_duplicates('canonical_entity_key')
    if len(entities)<2:return s,dict(status='insufficient_sources'),pd.DataFrame()
    coords=entities[['pixel_x','pixel_y']].to_numpy();tree=cKDTree(coords)
    refs=entities[(entities.catalogue=='gaia_dr3') & (entities.phot_g_mean_mag>14)&(entities.phot_g_mean_mag<17)]
    observations=[]
    for star in refs.itertuples():
        x,y=float(star.pixel_x),float(star.pixel_y)
        dist,_=tree.query([x,y],k=2)
        if dist[1]<isolation_px:continue
        ix,iy=round(x),round(y);r=int(radius_px)
        if ix-r<0 or iy-r<0 or ix+r>=raw.shape[1] or iy+r>=raw.shape[0]:continue
        sl=np.s_[iy-r:iy+r+1,ix-r:ix+r+1];image=np.asarray(raw[sl],float)
        yy,xx=np.mgrid[iy-r:iy+r+1,ix-r:ix+r+1];rr=np.hypot(xx-x,yy-y)
        core=rr<r*.7;ring=rr>=r*.8
        if not np.isfinite(image).all():continue
        if quality_mask is not None and ((quality_mask[sl][core].astype(np.int64)&0x1020)!=0).any():continue
        bg=float(np.median(image[ring]));sigma=1.4826*np.median(abs(image[ring]-bg))
        signal=np.maximum(image-bg,0)*core
        if signal.max()<10*max(sigma,1e-12) or signal.sum()<=0:continue
        xobs=float((xx*signal).sum()/signal.sum());yobs=float((yy*signal).sum()/signal.sum())
        if np.hypot(xobs-x,yobs-y)>max_offset_px:continue
        observations.append(dict(source_key=star.source_key,canonical_entity_key=star.canonical_entity_key,
            catalogue='gaia_dr3',catalogue_x=x,catalogue_y=y,observed_x=xobs,observed_y=yobs,dx=xobs-x,dy=yobs-y))
        for ps in s[(s.catalogue=='ps1_dr2_stack')&(s.canonical_entity_key==star.canonical_entity_key)].itertuples():
            if np.hypot(ps.pixel_x-xobs,ps.pixel_y-yobs)>max_offset_px:continue
            observations.append(dict(source_key=ps.source_key,canonical_entity_key=ps.canonical_entity_key,
                catalogue='ps1_dr2_stack',catalogue_x=float(ps.pixel_x),catalogue_y=float(ps.pixel_y),
                observed_x=xobs,observed_y=yobs,dx=xobs-ps.pixel_x,dy=yobs-ps.pixel_y))
    table=pd.DataFrame(observations)
    s['catalogue_pixel_x']=s.pixel_x;s['catalogue_pixel_y']=s.pixel_y
    s['image_calibration_status']='insufficient_calibrators';s['image_position_scatter_px']=np.nan
    report={}
    for cat in ['gaia_dr3','ps1_dr2_stack']:
        part=table[table.catalogue==cat] if len(table) else pd.DataFrame()
        # Multiple measurements of one star do not inflate calibrator counts.
        if len(part):part=part.groupby('canonical_entity_key')[['dx','dy']].median()
        if len(part)<min_stars:report[cat]=dict(status='insufficient_calibrators',stars=len(part));continue
        residual=part[['dx','dy']].to_numpy();shift=np.median(residual,axis=0)
        scatter=1.4826*np.median(abs(residual-shift),axis=0)
        stable=bool(np.max(scatter)<1.0 and np.linalg.norm(shift)<max_offset_px)
        report[cat]=dict(status='calibrated' if stable else 'unstable',stars=len(part),
                        dx_px=float(shift[0]),dy_px=float(shift[1]),sigma_x_px=float(scatter[0]),sigma_y_px=float(scatter[1]))
        select=s.catalogue==cat
        s.loc[select,'image_calibration_status']=report[cat]['status']
        if stable:
            s.loc[select,'pixel_x']+=shift[0];s.loc[select,'pixel_y']+=shift[1]
            s.loc[select,'image_position_scatter_px']=float(np.hypot(*scatter))
    return s,report,table
