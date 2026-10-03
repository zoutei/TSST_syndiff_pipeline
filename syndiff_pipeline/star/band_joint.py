"""Signed target flux after projecting out residual neighbours and constant sky."""
import numpy as np

def solve(data,noise,profiles,good):
 p=np.asarray(profiles,float);d=np.asarray(data,float);n=np.asarray(noise,float)
 keep=np.asarray(good,bool)&np.isfinite(d)&np.isfinite(n)&(n>0)&np.all(np.isfinite(p),axis=0)
 if keep.sum()<8:raise ValueError('insufficient_joint_support')
 target=p[0][keep]/n[keep];dw=d[keep]/n[keep]
 nuisance=np.column_stack([p[1:,keep].T,np.ones(keep.sum())]);nw=nuisance/n[keep,None]
 norms=np.linalg.norm(nw,axis=0);active=norms>1e-12
 na=nw[:,active]/norms[active]
 U,s,Vt=np.linalg.svd(na,full_matrices=False);rank=s>s[0]*1e-7
 Q=U[:,rank];orth=target-Q@(Q.T@target);info=float(orth@orth)
 rawinfo=float(target@target)
 if rawinfo<1e-20 or info<1e-8*rawinfo:raise ValueError('target_unresolved_after_neighbour_projection')
 flux=float((orth@dw)/info)
 coef_active=(Vt[rank].T/s[rank])@(U[:,rank].T@(dw-flux*target))/norms[active]
 coef=np.zeros(nuisance.shape[1]);coef[active]=coef_active
 model=flux*p[0]+np.sum(coef[:-1,None,None]*p[1:],axis=0)+coef[-1]
 res=(d[keep]-model[keep])/n[keep]
 return dict(flux=flux,flux_err=float(1/np.sqrt(info)),background=float(coef[-1]),chi2=float(res@res),dof=int(keep.sum()-rank.sum()-1),n_good=int(keep.sum()),condition=float(np.sqrt(rawinfo/info)),profile_support=float(p[0][keep].sum()/p[0].sum()),n_neighbours=len(p)-1,nuisance_rank=int(rank.sum())),model
