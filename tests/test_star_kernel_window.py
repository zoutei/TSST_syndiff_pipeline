import numpy as np
import pytest
from hotpants import HotpantsConfig
from hotpants.pure.convolution import apply_kernel
from syndiff_pipeline.star.kernel_window import FrameKernel


@pytest.mark.parametrize("factor", [1,2])
def test_window_vs_s1_full_grid_replay(factor):
    hp=HotpantsConfig(rkernel=2,ko=2,bgo=-1,ngauss=1,deg_fixe=[1],sigma_gauss=[1.0])
    shape=(39,48)
    rng=np.random.default_rng(45)
    sol=rng.normal(size=hp.n_comp_total+1)
    k=FrameKernel(sol,hp,shape,factor)
    scene=rng.normal(size=np.array(shape)*factor).astype('f4')
    full=apply_kernel(scene,sol,np.zeros_like(scene),np.zeros(scene.shape,'i4'),hp,k.basis,
                      oversample=factor)[0]
    guard=scene[7*factor:27*factor,12*factor:33*factor]
    np.testing.assert_allclose(k.apply(guard,12,7),full[9:25,14:31],rtol=2e-6,atol=5e-7)
