import numpy as np
import pandas as pd
from syndiff_pipeline.template_creation.processing.removal_ledger.astrometry import calibrate_image_positions


def test_known_image_offset_recovered_without_wcs_edit():
    yy,xx=np.mgrid[:250,:250];image=np.zeros((250,250),np.float32);rows=[]
    for j,y in enumerate([30,80,130,180]):
        for i,x in enumerate([30,80,130,180]):
            image+=100*np.exp(-((xx-(x+.4))**2+(yy-(y-.3))**2)/(2*1.2**2))
            key=f'gaia:{j*4+i}'
            rows.append(dict(source_key=key,canonical_entity_key=key,catalogue='gaia_dr3',pixel_x=float(x),pixel_y=float(y),phot_g_mean_mag=15.))
            rows.append(dict(source_key=key.replace('gaia:','ps1det:'),canonical_entity_key=key,catalogue='ps1_dr2_stack',pixel_x=x+.2,pixel_y=y+.1,phot_g_mean_mag=np.nan))
    sources=pd.DataFrame(rows)
    corrected,report,samples=calibrate_image_positions(image,np.zeros(image.shape,np.uint16),sources)
    assert report['gaia_dr3']['status']=='calibrated'
    assert abs(report['gaia_dr3']['dx_px']-.4)<.01
    assert abs(report['ps1_dr2_stack']['dx_px']-.2)<.01
    assert abs(report['ps1_dr2_stack']['dy_px']+.4)<.01
    assert np.allclose(corrected.catalogue_pixel_x,sources.pixel_x)
    assert len(samples)==32


def test_saturated_calibrators_do_not_force_a_solution():
    image=np.ones((50,50),np.float32)
    src=pd.DataFrame(dict(source_key=['g:1','g:2'],canonical_entity_key=['g:1','g:2'],catalogue=['gaia_dr3']*2,pixel_x=[10.,40.],pixel_y=[10.,40.],phot_g_mean_mag=[15.,15.]))
    result,report,table=calibrate_image_positions(image,np.full(image.shape,0x1020,np.uint16),src)
    assert report['gaia_dr3']['status']=='insufficient_calibrators'
    assert np.array_equal(result.pixel_x,src.pixel_x)
