"""Transport deletion flux through the selected template's actual operator.

Do not blur integer labels or add independently counted overlap catalogues.
The renderer is dependency-injected so historical v2 cannot silently use v3
ownership. Cross-projection source priority uses the production implementation.
"""
from __future__ import annotations

import numpy as np

from .. import canonical_cell as cc
from .. import padding_correction as pc


def required_inputs(cell,metadata,mapping):
    names={cell,*cc.canonical_neighbour_names(metadata,cell)}
    if cell in mapping.index:
        names.update(s['neighbor'] for s in pc.cross_projection_padding_spec(mapping.loc[cell]))
    return sorted(names)


def render_cell(cell,metadata,mapping,fetch_image,*,sigma,radius,canonical_renderer,require_complete=True):
    """Same projection plus the real signed cross-projection correction once."""
    if canonical_renderer is None:raise ValueError('Explicit ownership renderer required')
    if require_complete:
        missing=[n for n in required_inputs(cell,metadata,mapping) if fetch_image(n) is None]
        if missing:raise ValueError(f'Missing donor inputs: {missing}')
    image=canonical_renderer(cell,metadata,fetch_image,sigma,radius)
    if image is None:raise ValueError('Recipient unavailable')
    if cell not in mapping.index:raise ValueError('Recipient absent from mapping')
    spec=pc.cross_projection_padding_spec(mapping.loc[cell])
    if not spec:return image
    own=fetch_image(cell)
    total=np.zeros(image.shape,dtype=np.float64)
    for location,neighbors in pc._grouped_padding_spec(spec).items():
        total+=pc._location_correction(location=location,neighbors=neighbors,skycell=cell,
            recipient_wcs=pc._cell_wcs(mapping.loc[cell]),cell_shape=image.shape,own_combined=own,
            data_root='',skycell_df=mapping,psf_sigma=sigma,kernel_radius=radius,fetch_image=fetch_image)
    result=image.astype(np.float64)
    finite=np.isfinite(result);result[finite]+=total[finite]
    return result.astype(image.dtype)


def fixed_domain(before,after):
    if before.shape!=after.shape:raise ValueError('Paired input shapes differ')
    # Historical zeroing can turn NaN into zero. Both counterfactuals must use
    # the actual output's domain; invalid input contributes zero, never NaN flux.
    valid=np.isfinite(after)
    b=np.where(valid,np.nan_to_num(before,nan=0.,posinf=0.,neginf=0.),np.nan).astype(after.dtype)
    a=np.where(valid,after,np.nan)
    delta=np.where(valid,b-a,np.nan).astype(after.dtype)
    return b,a,delta


def transport_cell(cell,metadata,mapping,before,after,*,sigma,radius,canonical_renderer):
    names=required_inputs(cell,metadata,mapping)
    paired={}
    for name in names:
        b,a=before(name),after(name)
        if b is None or a is None:raise ValueError(f'Missing paired image: {name}')
        paired[name]=fixed_domain(b,a)
    planes=[]
    for k in range(3):
        planes.append(render_cell(cell,metadata,mapping,lambda n:paired.get(n,(None,None,None))[k],
            sigma=sigma,radius=radius,canonical_renderer=canonical_renderer))
    b,a,d=planes;valid=np.isfinite(b)&np.isfinite(a)&np.isfinite(d)
    if not (np.array_equal(np.isfinite(a),np.isfinite(b)) and np.array_equal(np.isfinite(a),np.isfinite(d))):
        raise ValueError('Transport domains changed across paired renders')
    residual=np.where(valid,b.astype(float)-a.astype(float)-d.astype(float),0.)
    # A declared arithmetic allowance, not a fit to observed residuals: four
    # float32 output-rounding units on the local result magnitude plus tiny floor.
    allowance=4*np.finfo(np.float32).eps*(abs(b.astype(float))+abs(a.astype(float))+abs(d.astype(float)))+1e-10
    passed=bool(np.all(abs(residual[valid])<=allowance[valid]))
    return dict(before=b,after=a,transported=d,residual=residual,valid=valid,
        report=dict(inputs=names,closure_passed=passed,max_absolute_residual=float(abs(residual).max()),
                    integrated_residual=float(residual.sum()),rounding_bound_sum=float(allowance[valid].sum()),
                    allowance='4*float32_eps*(abs(before)+abs(after)+abs(delta))+1e-10 per pixel'))


def bin_regmap(image,regmap,shape,*,quality_mask=None,exclude_bit=4096):
    """Exact integer assignment sum; source masks are not independently remapped."""
    if regmap.shape!=image.shape:raise ValueError('Regmap/input shape mismatch')
    valid=np.isfinite(regmap)&(regmap>=0)&(regmap<np.prod(shape))
    if np.any(regmap[valid]!=np.floor(regmap[valid])):raise ValueError('Noninteger regmap assignment')
    if quality_mask is not None:
        if quality_mask.shape!=image.shape:raise ValueError('Mask shape mismatch')
        valid&=(quality_mask.astype(np.int64)&exclude_bit)==0
    values=np.where(np.isfinite(image),image,0.)
    return np.bincount(regmap[valid].astype(np.int64),weights=values[valid],minlength=int(np.prod(shape))).reshape(shape)
