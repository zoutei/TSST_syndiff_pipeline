"""Focused geometry tests: reference replay and known proper-motion direction/scale."""
import jax
import numpy as np
from astropy.time import Time
from syndiff_pipeline.forward_model import cheb_wcs as CW,gaia_pm as GP

def test_differential_projection():
 static=CW.ChebWcsStatic(150.,60.,np.array([[-1/0.005833,0],[0,1/0.005833]]),np.array([1025.,1025.]),np.array([1024.,1024.]),np.array([1024.,1024.]),2,((0,0),(1,0),(0,1),(2,0),(1,1),(0,2)))
 ra=np.array([150.,150.1]);dec=np.array([60.,60.1]);pmra=np.array([1000.,0]);pmdec=np.array([0.,1000.]);epoch=Time(2020.,format='jyear');later=Time(2020.+13.5/365.25,format='jyear')
 a,b=GP.propagate_ra_dec(ra,dec,pmra,pmdec,epoch)
 c,d=GP.propagate_ra_dec(ra,dec,pmra,pmdec,later)
 with jax.enable_x64(True):
  x0,y0,A0=map(np.asarray,CW.star_basis(a,b,static));xr,yr,Ar=map(np.asarray,CW.star_basis(a,b,static));x1,y1,A1=map(np.asarray,CW.star_basis(c,d,static))
 assert np.array_equal(x0,xr) and np.array_equal(A0,Ar)
 expected=1000*13.5/365.25/(3600*1000*.005833)
 np.testing.assert_allclose(x1[0]-x0[0],-expected,rtol=.005)
 np.testing.assert_allclose(y1[1]-y0[1],expected,rtol=.005)
 assert np.max(np.abs(A1-A0))>0
