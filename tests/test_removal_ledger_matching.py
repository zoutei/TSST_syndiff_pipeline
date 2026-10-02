import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from syndiff_pipeline.template_creation.processing.removal_ledger.matching import match_gaia_ps1,attach_entities


def catalogue_pair(n=20):
    ra=10.+np.arange(n)*.03;dec=np.full(n,30.)
    pmra=np.full(n,100.);pmdec=np.full(n,-100.)
    moved=SkyCoord(ra=ra*u.deg,dec=dec*u.deg,pm_ra_cosdec=pmra*u.mas/u.yr,pm_dec=pmdec*u.mas/u.yr,
                   obstime=Time(2016.,format='jyear')).apply_space_motion(Time(2012.,format='jyear'))
    gaia=pd.DataFrame(dict(source_key=[f'gaia:{i}' for i in range(n)],entity_key=[f'gaia:{i}' for i in range(n)],
        ra=ra,dec=dec,original_ra=ra,original_dec=dec,pmra=pmra,pmdec=pmdec,phot_g_mean_mag=np.full(n,15.)))
    ps=pd.DataFrame(dict(ps1_detection_id=[str(100+i) for i in range(n)],ps1_obj_id=[str(1000+i) for i in range(n)],
        entity_key=[f'ps1:{1000+i}' for i in range(n)],ra=moved.ra.deg,dec=moved.dec.deg,
        raMean=moved.ra.deg+.1/3600/np.cos(np.deg2rad(30)),decMean=moved.dec.deg-.1/3600,
        epochMean=np.full(n,Time(2012.,format='jyear').mjd)))
    return gaia,ps


def test_epoch_propagation_and_calibration_are_used():
    g,p=catalogue_pair();m,cal=match_gaia_ps1(g,p)
    assert len(m)==20 and (m.status=='accepted_unique_calibrated').all()
    assert m.epoch_propagated.all()
    assert abs(cal['offset_ra_cosdec_arcsec']-.1)<1e-5
    assert abs(cal['offset_dec_arcsec']+.1)<1e-5


def test_close_companion_not_forced_to_nearest():
    g,p=catalogue_pair()
    extra=p.iloc[[0]].copy();extra.ps1_obj_id='999';extra.ps1_detection_id='999';extra.entity_key='ps1:999'
    extra.raMean+=.1/3600
    m,_=match_gaia_ps1(g,pd.concat([p,extra],ignore_index=True))
    assert (m[m.gaia_key=='gaia:0'].status=='ambiguous').all()


def test_missing_motion_or_epoch_not_accepted():
    g,p=catalogue_pair();g.loc[0,'pmra']=np.nan;p.loc[1,'epochMean']=np.nan
    m,_=match_gaia_ps1(g,p)
    assert (m[m.gaia_key.isin(['gaia:0','gaia:1'])].status!='accepted_unique_calibrated').all()


def test_split_stack_measurements_keep_one_object_identity():
    g,p=catalogue_pair();extra=p.iloc[[0]].copy();extra.ps1_detection_id='duplicate-measurement'
    m,_=match_gaia_ps1(g,pd.concat([p,extra],ignore_index=True))
    assert len(m)==20
