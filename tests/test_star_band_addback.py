import numpy as np
from types import SimpleNamespace
from scipy.signal import fftconvolve
from syndiff_pipeline.star.band_addback import convolve_target_patch,native_patch,restore_target,fit_flux
from syndiff_pipeline.forward_model.chain._tk import convolve_blended,block_sum


def test_patch_matches_independent_full_image_operator_across_nodes():
    rng=np.random.default_rng(19)
    g=SimpleNamespace(oversampling=4,ffi_xmin=36,ffi_ymin=-8,
        science_xmin_ffi=44,science_ymin_ffi=0,width_os=160,height_os=160)
    nx=ny=np.array([0.,10.,24.]);ks=rng.normal(size=(4,3,3,9,9))
    a=rng.uniform(size=(4,13,17));origin=(37,39)
    result,org=convolve_target_patch(a,ks,nx,ny,origin_os=origin,grid=g)
    actual=native_patch(result,org,grid=g,bounds_sci=(0,24,0,24))
    whole=np.zeros((4,160,160));whole[:,37:50,39:56]=a
    expected=sum(convolve_blended(whole[b],ks[b],g,nx,ny,workers=1) for b in range(4))
    expected=block_sum(expected,4)[8:32,8:32]
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=1e-12)
    # Independent with/without-target identity, with nonconstant output scale.
    base=rng.normal(size=actual.shape);scale=1+np.arange(24)[None,:]/24
    np.testing.assert_allclose(restore_target(base,actual,np.broadcast_to(scale,base.shape)),base+expected*scale,atol=3e-12)


def test_negative_flux_and_masked_core_preserve_full_flux_normalization():
    y,x=np.mgrid[-5:6,-5:6];p=np.exp(-(x*x+y*y)/2);p/=p.sum()
    good=np.ones(p.shape,bool);good[5,5]=False
    r=fit_flux(-7*p+3,np.ones(p.shape),p,good)
    np.testing.assert_allclose([r['flux'],r['background']],[-7,3],atol=1e-12)
