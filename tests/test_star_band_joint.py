import numpy as np
from syndiff_pipeline.star.band_joint import solve

def test_duplicate_nuisances_do_not_destroy_identifiable_target():
 y,x=np.mgrid[-7:8,-7:8];p=np.exp(-(x*x+y*y)/2);p/=p.sum();q=np.exp(-((x-3)**2+(y+1)**2)/3);q/=q.sum()
 profiles=np.array([p,q,q,np.zeros_like(q)])
 data=-4*p+120*q+3
 r,m=solve(data,np.ones(p.shape),profiles,np.ones(p.shape,bool))
 np.testing.assert_allclose(r['flux'],-4,atol=1e-10);np.testing.assert_allclose(m,data,atol=1e-10)

def test_formal_uncertainty_and_variable_flux_recovery():
 rng=np.random.default_rng(42);y,x=np.mgrid[-7:8,-7:8];p=np.exp(-(x*x+y*y)/2);p/=p.sum();q=np.exp(-((x-2)**2+(y-1)**2)/2);q/=q.sum()
 profiles=np.array([p,q]);noise=np.full(p.shape,2.);good=np.ones(p.shape,bool);good[0,:]=False
 residual=[];fluxes=[];truth=[]
 for k in range(300):
  f=25+12*np.sin(2*np.pi*k/37);data=f*p+(70+.1*k)*q+5+rng.normal(size=p.shape)*noise
  r,_=solve(data,noise,profiles,good);residual.append((r['flux']-f)/r['flux_err']);fluxes.append(r['flux']);truth.append(f)
 assert abs(np.mean(residual))<.15
 assert .85<np.std(residual)<1.15
 # Deterministic varying target+neighbour amplitudes exactly recover without noise.
 for f in [-8,0,12,70]:
  r,_=solve(f*p+40*q+2,noise,profiles,good);np.testing.assert_allclose(r['flux'],f,atol=1e-10)
