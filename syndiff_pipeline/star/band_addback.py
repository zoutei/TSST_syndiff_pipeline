"""Coordinate-preserving target add-back for saved per-band difference images.

The caller supplies target-only band templates transported through the producing
PS1 masks, blur, seams and registration. Catalogue flux is never used as the
measured amplitude. Origins are explicit global oversampled template indices.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from syndiff_pipeline.forward_model.chain._tk import hat_weights


def convolve_target_patch(bands, kernels, node_x, node_y, *, origin_os, grid):
    """Convolve retained target patches with source-position node weights.

    Returns a full-support patch and its global OS origin. ``kernels`` has
    shape (band, node_y, node_x, ky, kx). Band weights/rescaling must already
    be included in ``bands``; no photometric scale or background is added.
    """
    a=np.asarray(bands,dtype=np.float64);k=np.asarray(kernels,dtype=np.float64)
    if a.ndim!=3 or k.ndim!=5 or a.shape[0]!=k.shape[0]:
        raise ValueError('Expected (band,y,x) patches and (band,ny,nx,ky,kx) kernels')
    if not np.isfinite(a).all() or not np.isfinite(k).all():
        raise ValueError('Explicitly resolve masks/nonfinite input before convolution')
    if k.shape[-1]%2!=1 or k.shape[-2]%2!=1:
        raise ValueError('Kernels must have odd dimensions')
    oy,ox=map(int,origin_os);f=int(grid.oversampling)
    x=grid.ffi_xmin-grid.science_xmin_ffi+(ox+np.arange(a.shape[2])+.5)/f-.5
    y=grid.ffi_ymin-grid.science_ymin_ffi+(oy+np.arange(a.shape[1])+.5)/f-.5
    hx=hat_weights(x,node_x);hy=hat_weights(y,node_y)
    if k.shape[1:3]!=(len(node_y),len(node_x)):
        raise ValueError('Kernel node shape does not match coordinate arrays')
    out=np.zeros((a.shape[1]+k.shape[-2]-1,a.shape[2]+k.shape[-1]-1))
    for iy in np.flatnonzero(hy.any(axis=1)):
        for ix in np.flatnonzero(hx.any(axis=1)):
            w=hy[iy,:,None]*hx[ix,None,:]
            for b in range(len(a)):
                out+=fftconvolve(a[b]*w,k[b,iy,ix],mode='full')
    return out,(oy-k.shape[-2]//2,ox-k.shape[-1]//2)


def native_patch(image,origin_os,*,grid,bounds_sci):
    """Block-sum a global OS patch into science-local native [x0,x1,y0,y1)."""
    x0,x1,y0,y1=map(int,bounds_sci);f=int(grid.oversampling)
    gx0=(x0+grid.science_xmin_ffi-grid.ffi_xmin)*f
    gy0=(y0+grid.science_ymin_ffi-grid.ffi_ymin)*f
    out=np.zeros(((y1-y0)*f,(x1-x0)*f),dtype=np.float64)
    oy,ox=map(int,origin_os);a=np.asarray(image)
    xx0=max(gx0,ox);xx1=min(gx0+out.shape[1],ox+a.shape[1])
    yy0=max(gy0,oy);yy1=min(gy0+out.shape[0],oy+a.shape[0])
    if xx1>xx0 and yy1>yy0:
        out[yy0-gy0:yy1-gy0,xx0-gx0:xx1-gx0]=a[yy0-oy:yy1-oy,xx0-ox:xx1-ox]
    return out.reshape(y1-y0,f,x1-x0,f).sum(axis=(1,3))


def restore_target(difference,target_model,scale):
    """D + a(x,y) M_target, applying scale at OUTPUT pixels exactly as subtraction."""
    d=np.asarray(difference);m=np.asarray(target_model);a=np.asarray(scale)
    if d.shape!=m.shape or a.ndim>0 and a.shape!=d.shape:
        raise ValueError('Difference, add-back and scale coordinates must match')
    return d+a*m


def fit_flux(stamp,noise,profile,good,*,fit_background=True):
    """Unconstrained weighted flux fit; preserves negative and low-S/N fluxes.

    Profile must carry the full-flux normalization from the renderer, never
    renormalized to the surviving pixels. Formal errors are conditional on
    fixed templates/profile and independent supplied pixel errors.
    """
    d,n,p=np.broadcast_arrays(stamp,noise,profile)
    keep=np.asarray(good,bool)&np.isfinite(d)&np.isfinite(n)&(n>0)&np.isfinite(p)
    A=p[keep,None]
    if fit_background:A=np.column_stack((A,np.ones(keep.sum())))
    if keep.sum()<=A.shape[1]+2:raise ValueError('Insufficient usable pixels')
    Aw=A/n[keep,None];dw=d[keep]/n[keep]
    normal=Aw.T@Aw
    if np.linalg.cond(normal)>1e12:raise ValueError('Ill-conditioned flux/background fit')
    cov=np.linalg.inv(normal);beta=cov@(Aw.T@dw)
    residual=(d[keep]-A@beta)/n[keep]
    return dict(flux=float(beta[0]),flux_err=float(np.sqrt(cov[0,0])),
                background=float(beta[1]) if fit_background else 0.,
                chi2=float(residual@residual),dof=int(keep.sum()-A.shape[1]),n_good=int(keep.sum()))
