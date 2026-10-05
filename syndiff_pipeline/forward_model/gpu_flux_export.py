# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""GPU end-of-run post-fit flux, chi2_red, and preview model exporter.

Runs one jitted, forward-only (no gradient) evaluation per (K,P) bucket,
block-accumulated on-device exactly the way Adam training accumulates its
gradient (see ``fit.make_accum_step_fn``/``fit._block_num_den_fn``): each
bucket compiles ONE executable (traced block-start index, static block
size, explicit-array jit arguments -- never data closed over as compiled
constants), then that executable is called once per frame-block position in
an ordinary Python loop. Saves a compressed NPZ artifact containing solved
fluxes, chi2_red, outlier masks, and preview metadata.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import flux_solve as FS
from . import loss as L
from . import stamp_reject as SR

log = logging.getLogger(__name__)


def _build_export_block_fn(
    ctx,
    *,
    packed: bool,
    block_size: int,
    local_cache,
    n_pix: int,
    do_recenter: bool,
    recenter_n_iter: int,
    pedestal: bool = False,
):
    """Compile one forward-only, per-block flux/chi2 kernel for this bucket.

    ``ctx`` (and every other value that doesn't vary across block positions
    within this bucket -- ``packed``, ``block_size``, ``local_cache``,
    ``n_pix``, ``do_recenter``, ``recenter_n_iter``) is closed over rather
    than passed as a jit argument: ``StaticContext`` is not a registered JAX
    pytree (it mixes static Python fields with array fields), so it cannot
    flow through a jit boundary as a dynamic argument -- exactly why
    ``fit._block_num_den_fn`` closes over its per-bucket context/settings too.

    Every value that DOES vary per block call -- ``params`` and the padded
    per-bucket ``data``/``noise``/``weight``/frame-basis/gauge arrays, plus
    the traced block-start ``lo`` -- is an explicit jit argument. This is
    deliberate: it prevents XLA from baking the (potentially multi-GB) stamp
    tensors into the compiled executable as constants, which is exactly what
    ``stamp_reject._build_packed_chunked_chi2_jit`` does (and is willing to
    pay for, since that function's compile cost is amortized over many
    ``reject_every`` refreshes within one training run) -- this kernel is
    built once and used once, so there is no reuse to amortize against.

    ``lo`` is traced, not a Python int in ``static_argnums``: padding every
    bucket's frame axis to a multiple of ``block_size`` before calling (see
    ``loss.slice_static_context_frames_dynamic``'s docstring) means this
    compiles exactly once per bucket regardless of how many block positions
    are called, instead of once per distinct ``(lo, hi)`` pair.
    """

    @jax.jit
    def block_fn(
        params,
        data_padded, noise_padded, weight_padded,
        wcs_fb_padded, w_fb_padded, stamp_active_padded, w_of_t_padded,
        lo,
    ):
        ctx_blk = L.slice_static_context_frames_dynamic(
            ctx, wcs_fb_padded, w_fb_padded, stamp_active_padded, lo, block_size,
        )
        data_b = jax.lax.dynamic_slice_in_dim(data_padded, lo, block_size, axis=1)
        noise_b = jax.lax.dynamic_slice_in_dim(noise_padded, lo, block_size, axis=1)
        weight_b = jax.lax.dynamic_slice_in_dim(weight_padded, lo, block_size, axis=1)
        w_of_t_b = jax.lax.dynamic_slice_in_dim(w_of_t_padded, lo, block_size, axis=0)

        templates, _, _, _ = L.forward_model(
            params, ctx_blk,
            local_cache=local_cache,
            n_pix=n_pix,
            do_recenter=do_recenter,
            recenter_n_iter=recenter_n_iter,
            w_of_t_override=w_of_t_b,
        )
        var = L.pixel_variance(noise_b)
        if packed:
            pix_w = weight_b * ctx_blk.pix_valid[:, None, :]
        else:
            rmask = L.radius_pixel_mask(ctx_blk.fit_radius, stamp=int(data_b.shape[-1]))
            pix_w = weight_b * rmask[:, None, :, :]
        iv = L.inverse_variance_weights(pix_w, var)
        flux_b = FS.solve_group_fluxes(templates, data_b, iv, ridge=1e-6, pedestal=pedestal)
        if pedestal:
            flux_b, pedestal_b = flux_b
            model_b = FS.model_stamps_with_pedestal(templates, flux_b, pedestal_b)
        else:
            model_b = FS.model_stamps(templates, flux_b)

        reduce_axes = (-1,) if packed else (-1, -2)
        chi2 = pix_w * (data_b - model_b) ** 2 / var
        chi2_sum = jnp.sum(chi2, axis=reduce_axes)
        pix_sum_b = jnp.sum(pix_w, axis=reduce_axes)
        chi2_red_b = chi2_sum / jnp.maximum(pix_sum_b, 1e-6)
        if pedestal:
            return flux_b, chi2_red_b, pix_sum_b, pedestal_b
        return flux_b, chi2_red_b, pix_sum_b

    return block_fn


def _build_w_accum_block_fn(
    ctx,
    *,
    packed: bool,
    block_size: int,
    local_cache,
    n_pix: int,
    do_recenter: bool,
    recenter_n_iter: int,
    pedestal: bool = False,
):
    """Task PW: one jitted kernel returning this block's ``(S, rhs)`` pieces of
    the shared per-frame amplitude system (``flux_solve.profile_w_schur_pieces``).

    Same jit/closure contract as ``_build_export_block_fn`` (context closed
    over, everything block-varying passed explicitly).  ``S``/``rhs`` are
    additive over groups, so the caller sums them across every bucket before
    solving -- a per-bucket solve would give each bucket its own amplitude,
    which is exactly what the shared mode is not.
    """

    @jax.jit
    def block_fn(
        params,
        data_padded, noise_padded, weight_padded,
        wcs_fb_padded, w_fb_padded, stamp_active_padded, w_of_t_padded,
        lo,
    ):
        ctx_blk = L.slice_static_context_frames_dynamic(
            ctx, wcs_fb_padded, w_fb_padded, stamp_active_padded, lo, block_size,
        )
        data_b = jax.lax.dynamic_slice_in_dim(data_padded, lo, block_size, axis=1)
        noise_b = jax.lax.dynamic_slice_in_dim(noise_padded, lo, block_size, axis=1)
        weight_b = jax.lax.dynamic_slice_in_dim(weight_padded, lo, block_size, axis=1)
        w_of_t_b = jax.lax.dynamic_slice_in_dim(w_of_t_padded, lo, block_size, axis=0)

        templates, _, _, _, mode_templates = L.forward_model(
            params, ctx_blk,
            local_cache=local_cache,
            n_pix=n_pix,
            do_recenter=do_recenter,
            recenter_n_iter=recenter_n_iter,
            return_mode_templates=True,
        )
        var = L.pixel_variance(noise_b)
        if packed:
            pix_w = weight_b * ctx_blk.pix_valid[:, None, :]
        else:
            rmask = L.radius_pixel_mask(ctx_blk.fit_radius, stamp=int(data_b.shape[-1]))
            pix_w = weight_b * rmask[:, None, :, :]
        iv = L.inverse_variance_weights(pix_w, var)
        S, rhs, _, _ = FS.profile_w_schur_pieces(
            templates, mode_templates, data_b, iv, w_of_t_b,
            ridge=1e-6, pedestal=pedestal,
        )
        return S, rhs

    return block_fn


def _solve_profiled_w(
    params: dict,
    prepared: list,
    *,
    block: int,
    block_starts: list,
    n_padded: int,
    pedestal: bool,
    iterations: int,
    ridge_w: float = 1e-6,
) -> np.ndarray:
    """Task PW: Gauss-Newton solve of the shared ``w_of_t`` over ALL buckets.

    Each iteration walks every bucket x frame-block once, sums the additive
    ``(S, rhs)`` Schur pieces onto the padded frame axis, then solves the
    ``n_modes x n_modes`` system per frame.  ``n_modes`` comes from the
    decoded modes; the padding frames carry zero data/weight so they
    contribute a singular block, which the ridge keeps invertible and whose
    solution is discarded by the caller's ``[:n_frames_local]`` slice.
    """
    n_modes = int(np.asarray(L.decoded_epsf_modes(params)).shape[0])
    w = np.zeros((n_padded, n_modes), dtype=np.float32)
    if n_modes == 0:
        return w
    accum_fns: dict[int, object] = {}
    for it in range(max(0, int(iterations))):
        S_tot = np.zeros((n_padded, n_modes, n_modes), dtype=np.float64)
        r_tot = np.zeros((n_padded, n_modes), dtype=np.float64)
        w_j = jnp.asarray(w)
        for b, prep in enumerate(prepared):
            if prep is None:
                continue
            fn = accum_fns.get(b)
            if fn is None:
                fn = _build_w_accum_block_fn(
                    prep["ctx"],
                    packed=prep["packed"],
                    block_size=block,
                    local_cache=prep["local_cache"],
                    n_pix=prep["n_pix"],
                    do_recenter=prep["do_recenter"],
                    recenter_n_iter=prep["recenter_n_iter"],
                    pedestal=pedestal,
                )
                accum_fns[b] = fn
            for lo in block_starts:
                S_b, r_b = fn(
                    params, prep["data_padded"], prep["noise_padded"], prep["weight_padded"],
                    prep["wcs_fb_padded"], prep["w_fb_padded"], prep["stamp_active_padded"],
                    w_j, lo,
                )
                S_tot[lo:lo + block] += np.asarray(S_b, dtype=np.float64)
                r_tot[lo:lo + block] += np.asarray(r_b, dtype=np.float64)
        eye = np.eye(n_modes)[None]
        dw = np.linalg.solve(S_tot + ridge_w * eye, r_tot[..., None])[..., 0]
        w = (w + dw).astype(np.float32)
        print(
            f"  flux export: profile-w iteration {it + 1}/{iterations}: "
            f"|dw| max={np.abs(dw).max():.4e}, w rms={np.sqrt((w ** 2).mean(axis=0))}",
            flush=True,
        )
    return w


def export_gpu_flux_solution(
    out_dir: Path,
    params: dict,
    bundle,
    fds: list,
    buckets: list,
    frame_indices: np.ndarray,
    mask_active: np.ndarray,
    *,
    n_preview_frames: int = 10,
    reject_n_sigma: float = 3.0,
    frame_block: int | None = None,
    pedestal: bool = False,
    profile_w: bool = False,
    profile_w_iters: int = 2,
) -> Path:
    """Run one jitted forward solve per bucket on GPU and save flux_solved.npz.

    ``fds``/``buckets`` align 1:1 (same convention as
    ``fit.merge_fds_stamp_active``, which consumes the exact same
    ``fds, buckets, frame_indices`` triple): ``buckets[i][3]`` is bucket
    ``i``'s array of TRUE global group-row ids -- this is the source of
    truth for "which rows of ``flux_all`` does this bucket's output belong
    to", not ``frame_indices`` (a flat array of TRUE global frame columns,
    shared across every bucket, used only to map a block's LOCAL frame
    position to its GLOBAL column). Conflating the two -- pairing ``fds``
    against ``frame_indices`` instead of ``buckets`` -- was a real bug here
    that silently dropped all but one group per bucket and mislabeled the
    survivors; see ``fit.merge_fds_stamp_active`` for the proven-correct
    reference usage of this triple.

    ``frame_block``: with ``None`` each bucket's forward pass runs in one
    on-device block; otherwise the frame axis is chunked to bound peak VRAM,
    padded to a multiple of ``frame_block`` and processed by one
    jit-compiled-once-per-bucket kernel (see ``_build_export_block_fn``)
    called once per block position -- mirrors ``fit.make_accum_step_fn``'s
    proven accumulation pattern, not a plain Python-dispatched forward call
    per block (which pays a full JAX/XLA trace+dispatch round trip per call
    and left the GPU 0-10% utilized: measured 966s at T=1674/frame_block=8
    versus a full Adam step -- forward AND backward, same data -- at ~7-15s).
    """
    out_dir = Path(out_dir)
    out_path = out_dir / "flux_solved.npz"
    n_groups = int(bundle.group_set().n_groups)
    n_frames = int(bundle.n_frames)
    max_k = int(bundle.members.shape[1])
    frame_indices = np.asarray(frame_indices, dtype=np.intp)

    # 1. Determine 10 preview frame indices evenly spaced across the orbit
    n_prev = min(n_preview_frames, n_frames)
    preview_indices = np.linspace(0, n_frames - 1, n_prev, dtype=np.int32)

    flux_all = np.zeros((n_groups, n_frames, max_k), dtype=np.float32)
    chi2_red_all = np.zeros((n_groups, n_frames), dtype=np.float32)
    pix_sum_all = np.zeros((n_groups, n_frames), dtype=np.float32)
    # Task M4: one additive background level per group per frame (shared by
    # all K members), only allocated/populated/saved when pedestal=True --
    # default False writes exactly the pre-M4 file (no new key).
    pedestal_all = np.zeros((n_groups, n_frames), dtype=np.float32) if pedestal else None

    n_frames_local = int(frame_indices.shape[0])
    block = max(1, min(int(frame_block), n_frames_local)) if frame_block else n_frames_local
    n_blocks = -(-n_frames_local // block)  # ceil division
    pad_len = n_blocks * block - n_frames_local
    block_starts = [i * block for i in range(n_blocks)]

    n_buckets_active = sum(1 for fd in fds if fd is not None)
    total_blocks = n_buckets_active * n_blocks
    t_start = time.monotonic()
    t_last_print = 0.0
    blocks_done = 0
    print(
        f"  flux export: starting, {n_buckets_active} bucket(s), "
        f"{n_blocks} block(s)/bucket, frame_block={block}, "
        f"{total_blocks} block(s) total (jitted, compiled once per bucket)",
        flush=True,
    )

    # ---- per-bucket padded arrays, prepared once (task PW re-walks them) ----
    prepared: list = []
    for fd, entry in zip(fds, buckets):
        if fd is None:
            prepared.append(None)
            continue
        ctx = fd.ctx
        packed = bool(ctx.is_packed)
        local_cache = fd.local_cache if fd.use_dx_only else None
        prepared.append(dict(
            ctx=ctx,
            packed=packed,
            K_b=int(ctx.members.shape[1]),
            bidx_arr=np.asarray(entry[3], dtype=np.intp),
            local_cache=local_cache,
            n_pix=(fd.n_pix if fd.n_pix is not None
                   else (1 if packed else int(fd.data.shape[-1]))),
            do_recenter=bool(fd.do_recenter and local_cache is None),
            recenter_n_iter=int(fd.recenter_n_iter),
            wcs_fb_padded=L._pad_frame_axis(ctx.wcs_frame_basis, 0, pad_len),
            w_fb_padded=L._pad_frame_axis(ctx.w_frame_basis, 0, pad_len),
            stamp_active_padded=L._pad_frame_axis(ctx.stamp_active, 1, pad_len),
            data_padded=L._pad_frame_axis(fd.data, 1, pad_len),
            noise_padded=L._pad_frame_axis(fd.noise, 1, pad_len),
            weight_padded=L._pad_frame_axis(fd.weight, 1, pad_len),
        ))

    # ---- task PW: solve the shared per-frame amplitudes before exporting ----
    n_padded = n_blocks * block
    w_profiled = None
    if profile_w:
        w_profiled = _solve_profiled_w(
            params, prepared, block=block, block_starts=block_starts,
            n_padded=n_padded, pedestal=pedestal, iterations=int(profile_w_iters),
        )

    for b, (fd, prep) in enumerate(zip(fds, prepared)):
        if fd is None:
            continue
        ctx = prep["ctx"]
        packed = prep["packed"]
        K_b = prep["K_b"]
        bidx_arr = prep["bidx_arr"]
        local_cache = prep["local_cache"]
        n_pix = prep["n_pix"]
        do_recenter = prep["do_recenter"]
        recenter_n_iter = prep["recenter_n_iter"]
        wcs_fb_padded = prep["wcs_fb_padded"]
        w_fb_padded = prep["w_fb_padded"]
        stamp_active_padded = prep["stamp_active_padded"]
        data_padded = prep["data_padded"]
        noise_padded = prep["noise_padded"]
        weight_padded = prep["weight_padded"]

        if profile_w:
            # The amplitude is the one solved above, shared by every bucket;
            # the trained w_coeff spline is not part of the model at all.
            w_of_t_padded = jnp.asarray(w_profiled)
        else:
            # Whole-orbit gauge, computed once on the unpadded, unblocked context
            # -- the zero-time-mean gauge mean MUST be taken over the real
            # frames only (loss.whole_orbit_w_of_t's docstring; matches
            # fit.py's step()). Passing no override here (the previous
            # exporter's bug) re-derives a different gauge baseline per block,
            # silently corrupting the model/flux/chi2 whenever frame_block <
            # n_frames_local.
            w_of_t_full = L.whole_orbit_w_of_t(params, ctx)
            w_of_t_padded = L._pad_frame_axis(w_of_t_full, 0, pad_len)

        block_fn = _build_export_block_fn(
            ctx,
            packed=packed,
            block_size=block,
            local_cache=local_cache,
            n_pix=n_pix,
            do_recenter=do_recenter,
            recenter_n_iter=recenter_n_iter,
            pedestal=pedestal,
        )

        for lo in block_starts:
            out = block_fn(
                params, data_padded, noise_padded, weight_padded,
                wcs_fb_padded, w_fb_padded, stamp_active_padded, w_of_t_padded,
                lo,
            )
            if pedestal:
                flux_b, chi2_b, pix_b, pedestal_b = out
                pedestal_np = np.asarray(pedestal_b, dtype=np.float32)
                if pedestal_np.ndim == 1:
                    pedestal_np = pedestal_np[None, :]
            else:
                flux_b, chi2_b, pix_b = out

            flux_np = np.asarray(flux_b, dtype=np.float32)
            chi2_np = np.asarray(chi2_b, dtype=np.float32)
            pix_np = np.asarray(pix_b, dtype=np.float32)
            if flux_np.ndim == 2:
                flux_np = flux_np[None, :, :]
                chi2_np = chi2_np[None, :]
                pix_np = pix_np[None, :]

            hi_local = min(lo + block, n_frames_local)
            width = hi_local - lo
            global_cols = frame_indices[lo:hi_local]  # local block position -> true global frame column
            flux_np = flux_np[:, :width]
            chi2_np = chi2_np[:, :width]
            pix_np = pix_np[:, :width]
            if pedestal:
                pedestal_np = pedestal_np[:, :width]

            for i_sub, gi in enumerate(bidx_arr):  # bidx_arr: true global group-row ids for this bucket
                flux_all[int(gi), global_cols, :K_b] = flux_np[i_sub]
                chi2_red_all[int(gi), global_cols] = chi2_np[i_sub]
                pix_sum_all[int(gi), global_cols] = pix_np[i_sub]
                if pedestal:
                    pedestal_all[int(gi), global_cols] = pedestal_np[i_sub]

            blocks_done += 1
            t_now = time.monotonic()
            is_last = blocks_done == total_blocks
            if is_last or (t_now - t_last_print) >= 15.0:
                t_last_print = t_now
                print(
                    f"  flux export: bucket {b + 1}/{len(fds)} frames {lo}-{hi_local}/{n_frames_local} "
                    f"-- block {blocks_done}/{total_blocks} ({t_now - t_start:.1f}s elapsed)",
                    flush=True,
                )

    # 2. Compute residual MAD outlier rejection mask
    pix_ok = pix_sum_all > 0
    mad_mask = SR.mad_reject_mask(chi2_red_all, n_sigma=reject_n_sigma, pix_active=pix_ok)
    baseline = np.asarray(mask_active, dtype=bool)
    stamp_active_final = SR.combine_stamp_active(baseline, mad_mask)

    meta = dict(bundle.meta)
    btjd = np.asarray(meta.get("selected_frame_btjd", np.arange(n_frames)), dtype=np.float64)

    # 3. Save compressed npz
    save_kwargs = dict(
        flux=flux_all,
        chi2_red=chi2_red_all,
        pix_sum=pix_sum_all,
        stamp_active=stamp_active_final,
        preview_frame_indices=preview_indices,
        btjd=btjd,
        # Provenance: which parameters produced these fluxes. Readers must check
        # this before reusing the file (see diagnostics/flux_cache.solve_fluxes).
        params_sha=np.array(L.params_fingerprint(params)),
    )
    if pedestal:
        # Task M4: only present when the export ran with pedestal=True, so a
        # default-off export writes the exact pre-M4 key set.
        save_kwargs["pedestal"] = pedestal_all
    if profile_w:
        # Task PW: the per-frame amplitude the model actually used, mapped
        # back onto the global frame axis (zeros wherever a frame was not in
        # this run's selection). Saved RAW -- i.e. NOT re-gauged to zero time
        # mean -- because these are the amplitudes the exported fluxes were
        # solved with; ``w_of_t_time_mean`` is the constant that
        # ``flux_solve.gauge_zero_time_mean`` would remove, and
        # ``flux_solve.shift_base_by_modes(base, modes, w_of_t_time_mean)``
        # is the base that makes the gauged pair render the identical model.
        w_used = np.asarray(w_profiled, dtype=np.float32)[:n_frames_local]
        w_global = np.zeros((n_frames, w_used.shape[1]), dtype=np.float32)
        w_global[frame_indices] = w_used
        save_kwargs["w_of_t"] = w_global
        save_kwargs["w_of_t_time_mean"] = w_used.mean(axis=0).astype(np.float32)
    np.savez_compressed(out_path, **save_kwargs)
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out_path
