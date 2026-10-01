# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Memory-bounded forward rendering helpers for post-fit diagnostics."""

from __future__ import annotations

import gc
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp

from .. import flux_solve as FS
from .. import loss as L


def _pixel_weights(fd, start: int, stop: int):
    if bool(fd.ctx.is_packed):
        return fd.weight[start:stop] * fd.ctx.pix_valid[start:stop, None, :]
    rmask = L.radius_pixel_mask(
        fd.ctx.fit_radius[start:stop], stamp=int(fd.data.shape[-1])
    )
    return fd.weight[start:stop] * rmask[:, None, :, :]


def render_and_solve_chunks(params: dict, fd, *, group_chunk: int = 128, frame_indices=None, pedestal: bool = False):
    """Render templates and solve fluxes in bounded group chunks.

    The returned NumPy arrays have the same leading group layout as the
    monolithic path, but no JAX computation ever holds all groups at once.

    ``pedestal=True`` (task M4): also returns a fifth array, ``pedestal_out``
    ``(n_groups, n_frames)`` -- the additive per-group-per-frame background
    solved jointly with flux (see ``flux_solve.solve_group_fluxes``). Default
    False returns the original 4-tuple, unchanged.
    """
    n_groups = int(fd.ctx.members.shape[0])
    if frame_indices is not None:
        fi = np.asarray(frame_indices, dtype=np.int32)
        ctx0 = replace(
            fd.ctx,
            wcs_frame_basis=fd.ctx.wcs_frame_basis[fi],
            w_frame_basis=fd.ctx.w_frame_basis[fi],
            stamp_active=fd.ctx.stamp_active[:, fi],
        )
        fd = replace(fd, ctx=ctx0, data=fd.data[:, fi], noise=fd.noise[:, fi], weight=fd.weight[:, fi])
    chunk = max(1, int(group_chunk))
    flux_out: list[np.ndarray] = []
    model_out: list[np.ndarray] = []
    chi2_red_out: list[np.ndarray] = []
    pix_sum_out: list[np.ndarray] = []
    reduce_axes = (-1,) if bool(getattr(fd.ctx, "is_packed", False)) else (-1, -2)
    pedestal_out: list[np.ndarray] = []
    for start in range(0, n_groups, chunk):
        stop = min(n_groups, start + chunk)
        ctx = L.slice_static_context(fd.ctx, start, stop)
        templates, _, _, _ = L.forward_model(params, ctx)
        noise = fd.noise[start:stop]
        data = fd.data[start:stop]
        var = L.pixel_variance(noise)
        pw = _pixel_weights(fd, start, stop)
        weights = L.inverse_variance_weights(pw, var)
        flux = FS.solve_group_fluxes(templates, data, weights, ridge=1e-6, pedestal=pedestal)
        if pedestal:
            flux, pedestal_b = flux
            model = FS.model_stamps_with_pedestal(templates, flux, pedestal_b)
            pedestal_out.append(np.asarray(pedestal_b))
        else:
            model = FS.model_stamps(templates, flux)
        chi2 = pw * (data - model) ** 2 / var
        chi2_sum = np.sum(np.asarray(chi2), axis=reduce_axes)
        pix_sum = np.sum(np.asarray(pw), axis=reduce_axes)
        chi2_red = chi2_sum / np.maximum(pix_sum, 1e-6)
        flux_out.append(np.asarray(flux))
        model_out.append(np.asarray(model))
        chi2_red_out.append(np.asarray(chi2_red))
        pix_sum_out.append(np.asarray(pix_sum))
        del templates, flux, model, chi2, chi2_sum, ctx, pw, weights
    gc.collect()
    # Templates are only an intermediate; do not retain/concatenate them.
    if pedestal:
        return (
            np.concatenate(flux_out, axis=0),
            np.concatenate(model_out, axis=0),
            np.concatenate(chi2_red_out, axis=0),
            np.concatenate(pix_sum_out, axis=0),
            np.concatenate(pedestal_out, axis=0),
        )
    return (
        np.concatenate(flux_out, axis=0),
        np.concatenate(model_out, axis=0),
        np.concatenate(chi2_red_out, axis=0),
        np.concatenate(pix_sum_out, axis=0),
    )

