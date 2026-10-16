import numpy as np
from support import science_support
from syndiff_pipeline.forward_model import scene_export as SE

def test_bright_science_core_and_shared_bad_pixel():
 z=dict(stamp=1,cx=np.array([0,0,1]),cy=np.array([0,0,0]),uid=np.array([[0],[0],[1]]),n_union=2,valid=np.ones((3,1),bool))
 mask=np.array([[2,SE.BIT_ASTEROID]],dtype=np.int64)
 v=science_support(z,mask,np.ones((1,2),bool),np.zeros((1,2),bool))
 np.testing.assert_array_equal(v[:,0],[True,True,False])
 v=science_support(z,mask,np.array([[False,True]]),np.zeros((1,2),bool))
 assert not v.any()
