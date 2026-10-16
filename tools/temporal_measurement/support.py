"""Science-fit support: template/kernel fitting masks must not remove bright-star cores."""
import numpy as np
from syndiff_pipeline.forward_model import scene_export as SE

def science_support(reference, mask, finite, defect):
 S=int(reference['stamp']);k=np.arange(S*S);xx=reference['cx'][:,None]+k%S-S//2;yy=reference['cy'][:,None]+k//S-S//2
 inside=(xx>=0)&(xx<mask.shape[1])&(yy>=0)&(yy<mask.shape[0]);x=np.clip(xx,0,mask.shape[1]-1);y=np.clip(yy,0,mask.shape[0]-1)
 # Bright-star masking describes the removed template/kernel-fit regions, whose science light we FIT.
 forbidden=SE.BIT_EDGE|SE.BIT_PS1|SE.BIT_TNS|SE.BIT_ASTEROID
 good=inside&((mask[y,x]&forbidden)==0)&finite[y,x]&~defect[y,x]
 uid=reference['uid'];kill=np.zeros(int(reference['n_union'])+1,bool);kill[uid[~good]]=True;kill[-1]=True
 return reference['valid'].astype(bool)&~kill[uid]
