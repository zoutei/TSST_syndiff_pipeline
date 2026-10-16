import numpy as np
from numpy.polynomial.legendre import leggauss
from response import responses,solve,OFF

def gaussian(cov,shift):
 # Independent analytic continuous Gaussian integrated by fixed quadrature.
 q,w=leggauss(8);q=q*.5;w=w*.5;yy,xx=np.mgrid[-7:8,-7:8];P=np.zeros_like(xx,float);ic=np.linalg.inv(cov)
 for i in range(8):
  for j in range(8):
   x=xx+q[i]-shift[0];y=yy+q[j]-shift[1]
   P+=w[i]*w[j]*np.exp(-.5*(ic[0,0]*x*x+2*ic[0,1]*x*y+ic[1,1]*y*y))/(2*np.pi*np.sqrt(np.linalg.det(cov)))
 return P

def test_independent_integrated_gaussian_response():
 cov=np.array([[.8**2,.12],[.12,1.2**2]]);h=.03
 samples=np.array([gaussian(cov,o*h).ravel() for o in OFF]);A=responses(samples,h)
 y,x=np.mgrid[-7:8,-7:8];A=np.column_stack([A,np.ones(225),x.ravel(),y.ravel()])
 ds=np.array([[.004,.001],[.001,.006]]);shift=[.01,-.012]
 data=1.003*gaussian(cov+ds,shift).ravel()+1e-5-samples[0]
 c,cond=solve(A,data,np.ones(225)*1e-5,np.hypot(x,y).ravel()<=5)
 np.testing.assert_allclose(c[1:3],shift,atol=1e-4,rtol=0)
 np.testing.assert_allclose(c[3:6],[.004,.006,.001],atol=2e-4,rtol=0)
 assert abs(c[0]-.003)<1e-4 and cond<1e8
