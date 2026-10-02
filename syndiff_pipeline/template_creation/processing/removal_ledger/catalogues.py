"""Immutable, checked PS1/Gaia accounting catalogues; never deletion triggers."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from filelock import FileLock

from .cell import file_digest

PS1_ENDPOINT = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/stack"


def get_checked(session, url, params=None, timeout=90):
    last = None
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last=e
            if isinstance(e,requests.HTTPError) and e.response.status_code not in (429,500,502,503,504):raise
            if attempt < 2: time.sleep(2**attempt)
    raise last


def ps1_columns(metadata):
    available={m['name'].lower():m['name'] for m in metadata}
    wanted=['objID','uniquePspsSTid','raMean','decMean','raMeanErr','decMeanErr','epochMean',
            'raStack','decStack','raStackErr','decStackErr','primaryDetection','bestDetection',
            'nDetections','nStackDetections','objInfoFlag','qualityFlag']
    for band in 'rizy':
        wanted += [band+s for s in ['stackImageID','ra','dec','Epoch','PSFMag','PSFMagErr',
                                   'KronMag','KronRad','ApMag','infoFlag','infoFlag2','infoFlag3',
                                   'PSFFlux','PSFFluxErr','MomentXX','MomentYY','MomentXY']]
    columns=[available[w.lower()] for w in wanted if w.lower() in available]
    if not {'objID','uniquePspsSTid','raStack','decStack'} <= set(columns):
        raise ValueError("PS1 API schema missing required identity/position fields")
    return columns


def parse_count(response):
    value=response.json()
    try:n=int(value['data'][0][0])
    except (KeyError,IndexError,TypeError,ValueError) as e:raise ValueError('Invalid count response') from e
    if n<0:raise ValueError('Negative catalogue count')
    return n


def fetch_ps1_cone(root, ra, dec, radius_deg, *, session=None, page_size=5000, endpoint=PS1_ENDPOINT):
    """Fetch every stack detection in a cone, without primary/quality/mag cuts.

    Count agreement, stable detection keys and final re-count are mandatory.
    IDs are parsed as strings, never floats. A failed query cannot publish an
    empty complete catalogue. Different raw responses produce different hashes.
    """
    if not (0 <= ra < 360 and -90 <= dec <= 90 and 0 < radius_deg <= 1):
        raise ValueError('Invalid PS1 cone')
    if page_size<1:raise ValueError('Invalid page size')
    query=dict(ra=float(ra),dec=float(dec),radius=float(radius_deg))
    request=dict(endpoint=endpoint,query=query,schema=1)
    key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()[:24]
    root=Path(root);root.mkdir(parents=True,exist_ok=True);dest=root/key
    sess=session or requests.Session()
    with FileLock(str(root/(key+'.lock'))):
        if dest.exists():return read_catalogue(dest)
        temp=Path(tempfile.mkdtemp(prefix='.ps1-',dir=root))
        try:
            metadata=get_checked(sess,endpoint+'/metadata.json').json()
            columns=ps1_columns(metadata)
            (temp/'metadata.json').write_text(json.dumps(metadata,sort_keys=True))
            integer_columns={m['name']:'string' for m in metadata if m['name'] in columns and m.get('datatype')=='long'}
            expected=parse_count(get_checked(sess,endpoint+'/count.json',query))
            frames=[];keys=set();rawhash=[]
            for page in range(1,(expected+page_size-1)//page_size+1):
                params=dict(query,columns='['+','.join(columns)+']',pagesize=page_size,
                            page=page,sort_by='uniquePspsSTid.asc')
                response=get_checked(sess,endpoint+'.csv',params)
                raw=response.content
                rawhash.append(hashlib.sha256(raw).hexdigest())
                (temp/f'page_{page:05d}.csv').write_bytes(raw)
                frame=pd.read_csv(io.BytesIO(raw),dtype=integer_columns)
                if not set(columns)<=set(frame):raise ValueError('Truncated/invalid PS1 columns')
                expected_page=min(page_size,expected-(page-1)*page_size)
                if len(frame)!=expected_page:raise ValueError('PS1 page count disagrees with query count')
                if frame.uniquePspsSTid.isna().any() or frame.objID.isna().any():raise ValueError('Missing PS1 identity')
                pagekeys=frame.uniquePspsSTid.astype(str).tolist()
                if len(set(pagekeys))!=len(pagekeys) or keys.intersection(pagekeys):
                    raise ValueError('Duplicate/repeated PS1 page or nonunique detection identity')
                keys.update(pagekeys);frames.append(frame)
            final_count=parse_count(get_checked(sess,endpoint+'/count.json',query))
            if final_count!=expected:raise ValueError('PS1 catalogue changed during pagination')
            table=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame({c:pd.Series(dtype='string' if c in integer_columns else 'float64') for c in columns})
            table.to_parquet(temp/'catalogue.parquet',index=False)
            manifest=dict(request,key=key,status='complete',rows=len(table),expected_rows=expected,
                column_metadata_sha256=file_digest(temp/'metadata.json'),page_sha256=rawhash,
                catalogue_sha256=file_digest(temp/'catalogue.parquet'),
                caveat='Complete query response, not complete astrophysical sky. No primary/bestDetection filtering.')
            (temp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
            os.replace(temp,dest)
        except BaseException:shutil.rmtree(temp,ignore_errors=True);raise
    return read_catalogue(dest)


def read_catalogue(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m.get('status')!='complete':raise ValueError('Incomplete catalogue')
    if file_digest(path/'catalogue.parquet')!=m['catalogue_sha256']:raise ValueError('Corrupt catalogue')
    table=pd.read_parquet(path/'catalogue.parquet')
    if len(table)!=m['rows']:raise ValueError('Catalogue count mismatch')
    return table,m,path


def fetch_gaia_box(root, ra_min, ra_max, dec_min, dec_max, *, endpoint=None):
    """Flathub DR3 positional query, with no magnitude or RP/BP requirement.

    Flathub's array result is checked against its independent count endpoint
    before and after download. No claim about astrophysical completeness.
    """
    from ..pancakes import _fetch_flathub_numpy, _structured_array_to_gaia_dataframe, GAIA_CATALOG_COLUMNS, DEFAULT_FLATHUB_ENDPOINT
    if not (0<=ra_min<ra_max<360 and -90<=dec_min<dec_max<=90):raise ValueError('Split RA-wrap boxes before querying')
    query=dict(ra=[float(ra_min),float(ra_max)],dec=[float(dec_min),float(dec_max)])
    request=dict(endpoint=endpoint or DEFAULT_FLATHUB_ENDPOINT,query=query,schema=2,catalogue='gaiadr3',photometric_cuts=None)
    key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()[:24]
    root=Path(root);root.mkdir(parents=True,exist_ok=True);dest=root/key
    with FileLock(str(root/(key+'.lock'))):
        if dest.exists():return read_catalogue(dest)
        temp=Path(tempfile.mkdtemp(prefix='.gaia-',dir=root))
        try:
            import flathub
            filters=flathub.Filters(ra=(ra_min,ra_max),dec=(dec_min,dec_max))
            def count():
                response=requests.post(request['endpoint'].rstrip('/')+'/gaiadr3/count',json=filters.json(),timeout=90)
                response.raise_for_status()
                result=response.json()
                if isinstance(result,bool) or not isinstance(result,int) or result<0:raise ValueError('Invalid Gaia query count')
                return result
            expected=count()
            arr=_fetch_flathub_numpy('gaiadr3',list(GAIA_CATALOG_COLUMNS),endpoint=request['endpoint'],
                ra=(ra_min,ra_max),dec=(dec_min,dec_max))
            table=_structured_array_to_gaia_dataframe(arr)
            if len(table)!=expected or count()!=expected:raise ValueError('Gaia query count mismatch or changed catalogue')
            if table.source_id.duplicated().any():raise ValueError('Duplicate Gaia IDs in response')
            table['source_id']=table.source_id.astype('string')
            table.to_parquet(temp/'catalogue.parquet',index=False)
            manifest=dict(request,key=key,status='complete',rows=len(table),expected_rows=expected,
                catalogue_sha256=file_digest(temp/'catalogue.parquet'),
                caveat='Complete returned bbox query, independently count-checked before and after download. No magnitude/RP cuts; no claim about survey completeness.')
            (temp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
            os.replace(temp,dest)
        except BaseException:shutil.rmtree(temp,ignore_errors=True);raise
    return read_catalogue(dest)


def cell_query_geometry(wcs, shape, margin_px=600):
    """Dense boundary, spherical cone, and bbox covering a padded native cell."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    h,w=shape;x0=y0=-float(margin_px);x1=w-1+margin_px;y1=h-1+margin_px
    t=np.linspace(0,1,100)
    x=np.r_[x0+(x1-x0)*t,np.full_like(t,x1),x1-(x1-x0)*t,np.full_like(t,x0)]
    y=np.r_[np.full_like(t,y0),y0+(y1-y0)*t,np.full_like(t,y1),y1-(y1-y0)*t]
    ra,dec=wcs.all_pix2world(x,y,0);cra,cdec=wcs.all_pix2world((w-1)/2,(h-1)/2,0)
    sky=SkyCoord(ra*u.deg,dec*u.deg);centre=SkyCoord(float(cra)*u.deg,float(cdec)*u.deg)
    return dict(ra=float(cra),dec=float(cdec),radius_deg=float(centre.separation(sky).deg.max())+1e-6,
                bbox=[float(np.min(ra)),float(np.max(ra)),float(np.min(dec)),float(np.max(dec))],margin_px=margin_px)
