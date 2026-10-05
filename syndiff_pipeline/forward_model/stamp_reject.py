# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Stamp rejection: QC pre-filter, TNS/asteroid mask gate, residual MAD clip.

Pre-filter drops primary stars that fail photutils QC on too many frames.
TNS (bit 64) / asteroid (bit 128) hard-zeros any (group, frame) whose fixed
S×S stamp intersects those mask bits (asteroids are per-FFI via MaskCatalog).
Periodic residual MAD rejection hard-zeros outlier stamps and is ANDed with
the sticky mask baseline so contaminated stamps cannot re-enter.

``per_stamp_chi2_red`` is the hot path: it is called once per K/P bucket on
every ``--reject-every`` refresh, and used to run the *entire* forward model
op-by-op in eager JAX (no ``jax.jit``) -- ~160x slower than jitted on a
representative bundle (see ``clear_chi2_jit_cache``'s docstring / the module
tests). A module-level cache of compiled callables, keyed by a signature of
everything that actually enters the computation (NOT by ``id(fd.ctx)``, which
changes identity on every refresh -- see ``refresh_stamp_active``/``_multi``),
makes each bucket compile once and reuse the compiled executable thereafter.

Two rejection gates coexist here, selectable per call (neither replaces the
other so they can be A/B'd directly against the same checkpoints):

  - the original pooled-chi2 MAD gate (``refresh_stamp_active`` /
    ``refresh_stamp_active_multi``): a single threshold on the raw, linear
    ``chi2_red`` pooled across every group and frame. Measured evidence (see
    ``dev/forward_epsf_wcs/output/rejection_study/``) shows this gate is
    broken in a specific, structural way: ``chi2_red`` varies ~5 decades
    *between* groups (astrophysical/model-fit difficulty) but only CV~0.08
    *within* a group across frames (the actual per-cadence anomalies we want
    to catch). A single pooled linear threshold can only make whole-group,
    all-or-nothing cuts on the between-group spread, and empirically does --
    every group's reject fraction was measured at 0 or 1, never in between --
    while the threshold itself ran away (5935 -> 1219 over one run) because
    it was recomputed every refresh on a distribution whose tail it had just
    deleted.
  - the level-2 automatic gate (``refresh_stamp_active_v2``): per-group log
    centring removes the between-group spread, leaving only the tight,
    symmetric, MAD-valid per-cadence deviation; see that function's
    docstring for the full derivation and the anti-runaway knobs (frozen
    scale, churn cap, hysteresis) that keep it from reproducing the same
    runaway failure mode with a different score.
  - the level-1 audit (``level1_group_mismatch_table``): the between-group
    spread the level-2 gate deliberately discards is not thrown away -- it's
    surfaced as a ranked, human-reviewed table. Level 1 never cuts anything
    automatically; ``apply_group_exclusions`` is the only way a level-1
    finding turns into zeroed stamps, and it requires an explicit group list
    from a human.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import flux_solve as FS
from . import loss as L
from . import runtime as RT

MAD_SCALE = 1.4826
REJECT_WARN_FRAC = 0.5
# Stable public mask-bit assignments.  Keep these scalar values here so the
# bundle-only GPU rejection path does not import the workspace masking stack.
TNS = 64
ASTEROID = 128
TNS_ASTEROID_BITS = TNS | ASTEROID


def update_stamp_active_hysteresis(
    chi2_per_stamp: np.ndarray,
    current_active: np.ndarray,
    baseline_active: np.ndarray,
    frozen_scales: np.ndarray,
    tau_drop: float = 3.5,
    tau_keep: float = 2.2,
    max_churn_frac: float = 0.005,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply a dual-threshold, budgeted Schmitt trigger to a stamp mask.

    The group centre is deliberately recomputed from the *current* active
    population, while ``frozen_scales`` is fixed at stage start.  This avoids
    both threshold runaway and the active/rejected limit cycle of a single
    hard cut.  ``baseline_active`` is an absolute Gate-C invariant.
    """
    chi2 = np.asarray(chi2_per_stamp, dtype=np.float64)
    active = np.asarray(current_active, dtype=bool)
    baseline = np.asarray(baseline_active, dtype=bool)
    scales = np.asarray(frozen_scales, dtype=np.float64)
    if chi2.ndim != 2 or active.shape != chi2.shape or baseline.shape != chi2.shape:
        raise ValueError("chi2_per_stamp, current_active, and baseline_active must share (G, T) shape")
    if scales.shape != (chi2.shape[0],):
        raise ValueError("frozen_scales must have shape (G,)")
    if tau_keep > tau_drop:
        raise ValueError("tau_keep must be <= tau_drop")
    if max_churn_frac < 0:
        raise ValueError("max_churn_frac must be non-negative")

    log_chi2 = np.log(np.maximum(chi2, 1e-6))
    valid = active & baseline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(np.where(valid, log_chi2, np.nan), axis=1, keepdims=True)
    # A row can only be empty after a prior pathological cut.  Keeping its
    # existing state is safer than manufacturing a score from NaN.
    score = (log_chi2 - med) / np.maximum(scales[:, None], 1e-4)
    want_drop = active & baseline & np.isfinite(score) & (score > tau_drop)
    want_recover = (~active) & baseline & np.isfinite(score) & (score < tau_keep)
    max_flips = max(1, int(max_churn_frac * chi2.size))

    new_active = active.copy()
    drops = np.argwhere(want_drop)
    if len(drops):
        drops = drops[np.argsort(score[want_drop])[::-1][:max_flips]]
        new_active[drops[:, 0], drops[:, 1]] = False
    remaining = max_flips - len(drops)
    recovered = np.empty((0, 2), dtype=np.intp)
    if remaining > 0:
        recoveries = np.argwhere(want_recover)
        if len(recoveries):
            recovered = recoveries[np.argsort(score[want_recover])[:remaining]]
            new_active[recovered[:, 0], recovered[:, 1]] = True
    new_active &= baseline
    stats = {
        "n_dropped": int(np.sum(active & ~new_active)),
        "n_recovered": int(np.sum(~active & new_active)),
        "net_churn": int(np.sum(active != new_active)),
        "active_frac": float(np.mean(new_active)),
    }
    return new_active, stats


def prefilter_primaries_by_centroids_qc(
    frames: list[FrameRecord],
    fit_stars: pd.DataFrame,
    gaia_full: pd.DataFrame,
    *,
    star_cfg: StarSelectionConfig | None = None,
    min_frac: float = 0.5,
    merged_by_stem: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep primaries that pass centroids_r1 QC on a quorum of frames.

    A star is kept if it has at least one photometry row across ``frames`` and
    passes ``select_qc_stars`` on ``>= min_frac`` of the frames where it appears.
    Stars never seen in any photresults table are dropped.

    Pass ``merged_by_stem`` (from ``data.preload_merged_stars``) to avoid
    re-reading centroids CSVs.
    """
    # Workspace-only dependencies stay behind the preprocessing function so
    # importing stamp_reject on a lean --from-bundle GPU worker remains
    # jax/optax/numpy-only.
    from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import load_merged_stars, select_qc_stars  # noqa: PLC0415
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.fit_wcs_from_centroids import StarSelectionConfig  # noqa: PLC0415

    cfg = star_cfg or StarSelectionConfig()
    n = len(fit_stars)
    source_ids = fit_stars["source_id"].to_numpy()
    sid_to_idx = {int(s): i for i, s in enumerate(source_ids)}
    n_seen = np.zeros(n, dtype=np.int32)
    n_pass = np.zeros(n, dtype=np.int32)

    for frame in frames:
        if merged_by_stem is not None and frame.stem in merged_by_stem:
            merged = merged_by_stem[frame.stem]
        else:
            merged = load_merged_stars(frame, gaia_full)
        if "source_id" not in merged.columns:
            continue
        for sid in merged["source_id"].to_numpy():
            idx = sid_to_idx.get(int(sid))
            if idx is not None:
                n_seen[idx] += 1
        qc = select_qc_stars(merged, cfg)
        if "source_id" not in qc.columns:
            continue
        for sid in qc["source_id"].to_numpy():
            idx = sid_to_idx.get(int(sid))
            if idx is not None:
                n_pass[idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n_seen > 0, n_pass.astype(float) / n_seen.astype(float), 0.0)
    keep = (n_seen > 0) & (frac >= float(min_frac))
    kept = fit_stars.loc[keep].reset_index(drop=True)
    stats = {
        "n_primary_in": int(n),
        "n_primary_kept": int(keep.sum()),
        "n_never_seen": int((n_seen == 0).sum()),
        "n_below_frac": int(((n_seen > 0) & (frac < float(min_frac))).sum()),
        "qc_min_frac": float(min_frac),
        "median_pass_frac_kept": float(np.median(frac[keep])) if keep.any() else float("nan"),
    }
    return kept, stats


def prefilter_from_frame_tables(
    fit_stars: pd.DataFrame,
    frame_merged: list[pd.DataFrame],
    frame_qc: list[pd.DataFrame],
    *,
    min_frac: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Same quorum logic as ``prefilter_primaries_by_centroids_qc`` on synthetic tables.

    Used by unit tests that do not touch disk.
    """
    if len(frame_merged) != len(frame_qc):
        raise ValueError("frame_merged and frame_qc must have the same length")
    n = len(fit_stars)
    source_ids = fit_stars["source_id"].to_numpy()
    sid_to_idx = {int(s): i for i, s in enumerate(source_ids)}
    n_seen = np.zeros(n, dtype=np.int32)
    n_pass = np.zeros(n, dtype=np.int32)
    for merged, qc in zip(frame_merged, frame_qc):
        for sid in merged["source_id"].to_numpy():
            idx = sid_to_idx.get(int(sid))
            if idx is not None:
                n_seen[idx] += 1
        for sid in qc["source_id"].to_numpy():
            idx = sid_to_idx.get(int(sid))
            if idx is not None:
                n_pass[idx] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n_seen > 0, n_pass.astype(float) / n_seen.astype(float), 0.0)
    keep = (n_seen > 0) & (frac >= float(min_frac))
    kept = fit_stars.loc[keep].reset_index(drop=True)
    stats = {
        "n_primary_in": int(n),
        "n_primary_kept": int(keep.sum()),
        "n_never_seen": int((n_seen == 0).sum()),
        "n_below_frac": int(((n_seen > 0) & (frac < float(min_frac))).sum()),
        "qc_min_frac": float(min_frac),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# Task 1: compiled-callable cache for per_stamp_chi2_red.
# ---------------------------------------------------------------------------

_CHI2_JIT_CACHE_MAX_ENTRIES = 64
_CHI2_JIT_CACHE: "OrderedDict[tuple, Callable]" = OrderedDict()
_CHI2_CHUNK_CACHE: "OrderedDict[tuple, Callable]" = OrderedDict()


def _chi2_signature(fd: Any, *, packed: bool, local_cache: jnp.ndarray | None) -> tuple:
    """Cache key for ``per_stamp_chi2_red``'s compiled callable.

    Deliberately does NOT include ``id(fd.ctx)`` or ``id(fd)``: ``fd.ctx`` is
    *replaced* wholesale (a new ``StaticContext`` instance) by
    ``refresh_stamp_active``/``_multi`` on every refresh via
    ``loss.with_stamp_active``, but the only field that changes is
    ``stamp_active`` -- which ``per_stamp_chi2_red``/``forward_model`` never
    read (confirmed by inspection: ``forward_model`` assigns a local
    ``stamp_active`` default from ``ctx.stamp_active`` and never uses it
    again). So a ctx replacement must not force a recompile.

    ``id(fd.ctx.fit_radius)`` stands in for "this is still the same bucket's
    context": ``build_static_context`` allocates a fresh ``fit_radius`` array
    per call, and ``with_stamp_active``/``with_fit_radius`` never touch a
    field they don't own, so this id is stable across refreshes within one
    bucket's lifetime and (for all practical purposes) unique across
    different buckets/bundles -- it's what actually disambiguates two
    same-shaped-but-different buckets that the shape-only parts of this key
    can't tell apart.
    """
    n_pix = fd.n_pix if fd.n_pix is not None else (1 if packed else int(fd.data.shape[-1]))
    if local_cache is not None:
        lc_sig: tuple | None = (tuple(local_cache.shape), str(local_cache.dtype))
    else:
        lc_sig = None
    return (
        bool(packed),
        tuple(fd.data.shape), str(fd.data.dtype),
        tuple(fd.noise.shape), str(fd.noise.dtype),
        tuple(fd.weight.shape), str(fd.weight.dtype),
        tuple(fd.ctx.members.shape),
        int(n_pix),
        bool(fd.use_dx_only),
        local_cache is not None,
        lc_sig,
        # Defensive: shape+dtype alone can't tell apart two distinct local_cache
        # arrays of identical shape (e.g. rebuilt from different params). The
        # compiled callable bakes `local_cache`'s *values* in as a closure
        # constant (see `_build_chi2_jit`), so a shape-only match would return
        # silently-stale results if a caller ever rebuilt `fd.local_cache` for
        # the same `fd.ctx.fit_radius` identity. No current call site does
        # this (`fit.run_stage` builds it once, before `fd.ctx.fit_radius` is
        # ever assigned, and never rebuilds it for a live fd/ctx), but the id
        # costs nothing and closes the hole for any future caller that does.
        # Same id-liveness argument as `id(fd.ctx.fit_radius)` below: while a
        # cache entry is resident, `_build_chi2_jit`'s closure holds a strong
        # reference to the exact array whose id was used to build this key, so
        # that id cannot be reassigned to a different, unrelated object.
        id(local_cache) if local_cache is not None else None,
        id(fd.ctx.fit_radius),
        bool(fd.do_recenter and local_cache is None),
        int(fd.recenter_n_iter),
        float(fd.weights.ridge),
        str(fd.weights.flux_objective),
        float(fd.weights.huber_delta),
        int(fd.weights.huber_irls_iters),
    )


def _build_chi2_jit(fd: Any, *, packed: bool, local_cache: jnp.ndarray | None) -> Callable:
    """Compile ``per_stamp_chi2_red``'s body for this ``fd``.

    ``data``/``noise``/``weight`` are **JIT arguments**, not closure constants,
    for the same reason ``fit._combined_loss_fn`` passes them explicitly: baking
    the stamp tensors into the executable made XLA capture 3.01 GB of constants,
    which then overflowed protobuf's hard 2 GB limit when the CPU backend tried
    to serialize the executable for the persistent compilation cache
    (``xla.cpu.CompilationResultProto exceeded maximum protobuf size of 2GB:
    3009614057`` / ``PjRtCpuClient::SerializeExecutable proto serialization
    failed``). That made every compile uncacheable -- the ~350 s recompile each
    GPU session pays, which the run catalog had recorded as unavoidable -- and
    it also constant-folded the frame padding/blocking at compile time.

    ``fd.ctx`` and ``local_cache`` are still closed over: both are small, and the
    ``fd.ctx`` mutation argument in ``_chi2_signature``'s docstring still holds.
    Because the arrays are now arguments, one compiled executable is valid for
    any data of the same shape, so cache hits improve as a side effect.
    """
    ctx = fd.ctx
    data = fd.data
    noise = fd.noise
    weight = fd.weight
    n_pix = fd.n_pix if fd.n_pix is not None else (1 if packed else int(data.shape[-1]))
    do_recenter = fd.do_recenter and local_cache is None
    recenter_n_iter = fd.recenter_n_iter
    ridge = fd.weights.ridge
    flux_objective = fd.weights.flux_objective
    huber_delta = fd.weights.huber_delta
    huber_irls_iters = fd.weights.huber_irls_iters
    stamp_pedestal = fd.weights.stamp_pedestal
    reduce_axes = (-1,) if packed else (-1, -2)

    def _fn(params: dict[str, jnp.ndarray], data, noise, weight
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
        templates, _, _, _ = L.forward_model(
            params, ctx,
            local_cache=local_cache,
            n_pix=n_pix,
            do_recenter=do_recenter,
            recenter_n_iter=recenter_n_iter,
        )
        var = L.pixel_variance(noise)
        if packed:
            pix_w = weight * ctx.pix_valid[:, None, :]
        else:
            rmask = L.radius_pixel_mask(ctx.fit_radius, stamp=data.shape[-1])
            pix_w = weight * rmask[:, None, :, :]
        iv = L.inverse_variance_weights(pix_w, var)
        flux = FS.solve_fluxes(
            templates, data, iv,
            flux_objective=flux_objective,
            ridge=ridge,
            huber_delta=huber_delta,
            irls_iterations=huber_irls_iters,
            pedestal=stamp_pedestal,
        )
        if stamp_pedestal:
            flux, pedestal_val = flux
            model = FS.model_stamps_with_pedestal(templates, flux, pedestal_val)
        else:
            model = FS.model_stamps(templates, flux)
        chi2 = pix_w * (data - model) ** 2 / var
        chi2_sum = jnp.sum(chi2, axis=reduce_axes)
        pix_sum = jnp.sum(pix_w, axis=reduce_axes)
        chi2_red = chi2_sum / jnp.clip(pix_sum, 1e-6, None)
        return chi2_red, pix_sum

    return jax.jit(_fn)


def clear_chi2_jit_cache() -> None:
    """Drop every cached compiled callable (and the ``FitData`` arrays/ctx it
    closes over).

    The cache is FIFO-bounded at ``_CHI2_JIT_CACHE_MAX_ENTRIES`` entries so a
    single long-running training process can't grow it unboundedly, but a
    long diagnostic session that walks many *unrelated* bundles/checkpoints
    (e.g. ``diagnostics/rejection_core.py`` across several labeled
    checkpoints) can still pin more memory than necessary between them. Call
    this between unrelated sessions sharing a process to release it eagerly.
    """
    _CHI2_JIT_CACHE.clear()
    _CHI2_CHUNK_CACHE.clear()


def _chunked_chi2(params: dict[str, jnp.ndarray], fd: Any, chunk: int):
    """Score frame slices without materializing full-orbit model templates."""
    if bool(getattr(fd.ctx, "is_packed", False)):
        sig = ("packed-scan", int(chunk), _chi2_signature(fd, packed=True, local_cache=None))
        fn = _CHI2_JIT_CACHE.get(sig)
        if fn is None:
            fn = _build_packed_chunked_chi2_jit(fd, int(chunk))
            _CHI2_JIT_CACHE[sig] = fn
        else:
            _CHI2_JIT_CACHE.move_to_end(sig)
        return fn(params, fd.data, fd.noise, fd.weight)
    n_frames = int(fd.data.shape[1])
    outputs = []
    for start in range(0, n_frames, chunk):
        end = min(start + chunk, n_frames)
        key = (id(fd), start, end)
        fn = _CHI2_CHUNK_CACHE.get(key)
        if fn is None:
            ctx = replace(
                fd.ctx,
                wcs_frame_basis=fd.ctx.wcs_frame_basis[start:end],
                w_frame_basis=fd.ctx.w_frame_basis[start:end],
                stamp_active=fd.ctx.stamp_active[:, start:end],
            )
            local_cache = fd.local_cache
            if local_cache is not None:
                local_cache = local_cache[:, start:end]
            sliced = replace(
                fd, ctx=ctx, data=fd.data[:, start:end],
                noise=fd.noise[:, start:end], weight=fd.weight[:, start:end],
                mask_stamp_active=(None if fd.mask_stamp_active is None else
                                   fd.mask_stamp_active[:, start:end]),
                local_cache=local_cache, stamp_chunk=None,
            )
            packed = bool(getattr(ctx, "is_packed", False))
            if packed and local_cache is not None:
                raise NotImplementedError("dx-only local_cache is not supported on packed path")
            fn = _build_chi2_jit(
                sliced, packed=packed,
                local_cache=local_cache if sliced.use_dx_only else None,
            )
            _CHI2_CHUNK_CACHE[key] = fn
            if len(_CHI2_CHUNK_CACHE) > _CHI2_JIT_CACHE_MAX_ENTRIES:
                _CHI2_CHUNK_CACHE.popitem(last=False)
        else:
            _CHI2_CHUNK_CACHE.move_to_end(key)
        outputs.append(fn(params, fd.data[:, start:end], fd.noise[:, start:end],
                          fd.weight[:, start:end]))
    return (
        jnp.concatenate([item[0] for item in outputs], axis=1),
        jnp.concatenate([item[1] for item in outputs], axis=1),
    )


def _build_packed_chunked_chi2_jit(fd: Any, block: int) -> Callable:
    """One scan-compiled packed scorer; peak templates scale with ``block``."""
    ctx, data, noise, weight = fd.ctx, fd.data, fd.noise, fd.weight
    n_groups, K = ctx.members.shape
    n_frames = int(data.shape[1])
    block = max(1, min(int(block), n_frames))
    n_blocks = -(-n_frames // block)
    pad = n_blocks * block - n_frames
    ridge = fd.weights.ridge
    objective = fd.weights.flux_objective
    huber_delta = fd.weights.huber_delta
    irls = fd.weights.huber_irls_iters
    stamp_pedestal = fd.weights.stamp_pedestal

    def score(params, data, noise, weight):
        # data/noise/weight are arguments, not closure constants -- see
        # ``_build_chi2_jit``'s docstring for the 2 GB protobuf failure this avoids.
        x_t, y_t = L.CW.eval_all_positions(
            ctx.x_lin, ctx.y_lin, ctx.cheb_basis, params["wcs_coeff"],
            ctx.wcs_frame_basis, ctx.n_terms,
        )
        w_t = L.w_field_from_coeff(params["w_coeff"], ctx.w_frame_basis)
        flat = ctx.members.reshape(-1)
        stars = flat[ctx.occ_flat_idx]
        n_occ = int(ctx.occ_flat_idx.shape[0])
        x_occ = jnp.reshape(x_t[stars], (n_occ, n_frames))
        y_occ = jnp.reshape(y_t[stars], (n_occ, n_frames))
        base = L.decoded_epsf_base(params)
        modes = L.decoded_epsf_modes(params)

        def blocks(arr, axis):
            arr = L._pad_frame_axis(arr, axis, pad)
            return L._to_frame_blocks(arr, axis, n_blocks, block)

        xs = (blocks(x_occ, 1), blocks(y_occ, 1), blocks(w_t, 0),
              blocks(data, 1), blocks(noise, 1), blocks(weight, 1))
        pix_valid = ctx.pix_valid

        def body(_carry, vals):
            x_i, y_i, w_i, data_i, noise_i, weight_i = vals
            occ_templates = L._render_occ_templates_packed(
                ctx, x_i, y_i, w_i, base=base, modes=modes,
                do_recenter=fd.do_recenter,
                recenter_n_iter=fd.recenter_n_iter,
            )
            templates = jnp.zeros(
                (n_groups, K, block, occ_templates.shape[-1]),
                dtype=occ_templates.dtype,
            ).at[ctx.occ_group, ctx.occ_slot].set(occ_templates)
            var = L.pixel_variance(noise_i)
            pix_w = weight_i * pix_valid[:, None, :]
            iv = L.inverse_variance_weights(pix_w, var)
            flux = FS.solve_fluxes(
                templates, data_i, iv, flux_objective=objective, ridge=ridge,
                huber_delta=huber_delta, irls_iterations=irls,
                pedestal=stamp_pedestal,
            )
            if stamp_pedestal:
                flux, pedestal_val = flux
                model = FS.model_stamps_with_pedestal(templates, flux, pedestal_val)
            else:
                model = FS.model_stamps(templates, flux)
            chi2 = jnp.sum(pix_w * (data_i - model) ** 2 / var, axis=-1)
            pix_sum = jnp.sum(pix_w, axis=-1)
            return None, (chi2 / jnp.clip(pix_sum, 1e-6, None), pix_sum)

        _, (chi2_b, pix_b) = jax.lax.scan(jax.checkpoint(body), None, xs)
        chi2 = jnp.moveaxis(chi2_b, 0, 1).reshape(n_groups, n_blocks * block)[:, :n_frames]
        pix = jnp.moveaxis(pix_b, 0, 1).reshape(n_groups, n_blocks * block)[:, :n_frames]
        return chi2, pix

    return jax.jit(score)


def per_stamp_chi2_red(
    params: dict[str, jnp.ndarray],
    fd: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reduced chi2 per (group, frame) on the active pixel support.

    Returns ``(chi2_red, pix_sum)`` each shaped ``(n_groups, n_frames)``.

    Square path: weights = coverage × ``fit_radius`` disk (hp_d MASK unused).
    Packed path: weights = coverage × ``pix_valid`` (fit_radius unused; same
    convention as ``loss.total_loss``).

    Compiled and cached per bucket (see ``_chi2_signature``/``_build_chi2_jit``):
    the first call for a given (data/noise/weight shape+dtype, ctx.members
    shape, packedness, n_pix, use_dx_only, local_cache presence+shape,
    fit_radius identity, do_recenter, recenter_n_iter, ridge) signature
    traces and compiles; every later call with the same signature (e.g. every
    ``--reject-every`` refresh within a training stage) reuses the compiled
    executable and only re-runs it on the new ``params``. Measured ~160x
    faster than the previous eager (un-jitted) call on a representative
    bundle -- see ``dev/forward_epsf_wcs/tests/test_stamp_reject_v2.py`` and
    this module's docstring.
    """
    if getattr(fd, "stamp_chunk", None):
        return _chunked_chi2(params, fd, int(fd.stamp_chunk))
    packed = bool(getattr(fd.ctx, "is_packed", False))
    local_cache = fd.local_cache if fd.use_dx_only else None
    if packed and local_cache is not None:
        raise NotImplementedError("dx-only local_cache is not supported on the packed path")
    sig = _chi2_signature(fd, packed=packed, local_cache=local_cache)
    fn = _CHI2_JIT_CACHE.get(sig)
    if fn is not None:
        _CHI2_JIT_CACHE.move_to_end(sig)
    else:
        fn = _build_chi2_jit(fd, packed=packed, local_cache=local_cache)
        _CHI2_JIT_CACHE[sig] = fn
        if len(_CHI2_JIT_CACHE) > _CHI2_JIT_CACHE_MAX_ENTRIES:
            _CHI2_JIT_CACHE.popitem(last=False)
    return fn(params, fd.data, fd.noise, fd.weight)


def _mad_threshold(vals: np.ndarray, n_sigma: float) -> float:
    """``med + n_sigma * 1.4826 * MAD`` of ``vals`` (assumed already the active subset)."""
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = MAD_SCALE * mad
    return med + 1e-6 if scale < 1e-12 else med + float(n_sigma) * scale


def _pool_active_mask(
    pix_sum: np.ndarray,
    baseline: np.ndarray | None,
) -> np.ndarray:
    """``pix_ok`` (``pix_sum > 0``) ANDed with ``baseline > 0`` when given.

    Task 2 fix: stamps already hard-zeroed by the sticky TNS/asteroid
    baseline have inflated, physically-meaningless ``chi2_red`` values (the
    model was never asked to fit them well). Including them in the pool used
    to compute the MAD median/threshold inflates the MAD and loosens the
    threshold for every *other* stamp. Pooling over ``pix_ok & (baseline>0)``
    instead excludes them from the statistics while leaving them exactly as
    rejected as before (``combine_stamp_active`` still ANDs the baseline into
    the final mask regardless). Behavior is unchanged when ``baseline`` is
    None.
    """
    pix_ok = np.asarray(pix_sum) > 0
    if baseline is None:
        return pix_ok
    return pix_ok & (np.asarray(baseline) > 0)


def mad_reject_mask(
    chi2_red: np.ndarray,
    *,
    n_sigma: float = 3.0,
    pix_active: np.ndarray | None = None,
    pool_active: np.ndarray | None = None,
) -> np.ndarray:
    """Hard keep-mask from MAD clip on reduced chi2.

    Stamps with ``pix_active==False`` (or missing pixels) are always rejected
    (weight 0). Among active stamps, keep those with
    ``chi2_red <= med + n_sigma * 1.4826 * MAD``.

    ``pool_active`` (defaults to ``pix_active``) selects the subset of
    stamps whose ``chi2_red`` values are pooled to compute that median/MAD;
    pass a mask that additionally excludes stamps already hard-zeroed by a
    sticky mask baseline (see ``_pool_active_mask``) so their scores can't
    loosen the threshold for stamps that ``pix_active`` still considers
    candidates. Defaulting to ``pix_active`` reproduces the exact previous
    behavior.
    """
    chi2 = np.asarray(chi2_red, dtype=np.float64)
    if pix_active is None:
        active = np.ones(chi2.shape, dtype=bool)
    else:
        active = np.asarray(pix_active, dtype=bool)
    if pool_active is None:
        pool = active
    else:
        pool = np.asarray(pool_active, dtype=bool)
    out = np.zeros(chi2.shape, dtype=np.float32)
    vals = chi2[pool]
    if vals.size == 0:
        return out
    thresh = _mad_threshold(vals, n_sigma)
    keep = active & (chi2 <= thresh)
    out[keep] = 1.0
    return out


def combine_stamp_active(
    mad_mask: np.ndarray,
    mask_baseline: np.ndarray | None = None,
) -> np.ndarray:
    """AND residual MAD keep-mask with sticky TNS/asteroid baseline (if any)."""
    out = np.asarray(mad_mask, dtype=np.float32)
    if mask_baseline is not None:
        base = np.asarray(mask_baseline, dtype=np.float32)
        if base.shape != out.shape:
            raise ValueError(
                f"mask_baseline shape {base.shape} != mad_mask shape {out.shape}"
            )
        out = out * base
    return out


def build_tns_asteroid_stamp_active(
    catalog: Any,
    btjds: list[float] | np.ndarray,
    stamp_center_x: np.ndarray,
    stamp_center_y: np.ndarray,
    *,
    stamp: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hard keep-mask: 0 where the fixed S×S stamp hits TNS or asteroid bits.

    ``stamp_center_x/y`` are crop-local (same coords as ``MaskCatalog.mask_at``).
    Asteroid bit 128 is resolved per frame via ``catalog.mask_at(btjd)``.
    Only bits ``TNS | ASTEROID`` (64|128) reject; straps/edges/PS1/catalog
    star stamps are ignored here.
    """
    if stamp < 1 or stamp % 2 == 0:
        raise ValueError(f"stamp must be odd positive, got {stamp}")
    cx = np.asarray(stamp_center_x, dtype=np.int64)
    cy = np.asarray(stamp_center_y, dtype=np.int64)
    btjds = np.asarray(btjds, dtype=np.float64)
    n_groups = int(cx.shape[0])
    n_frames = int(btjds.shape[0])
    if cy.shape[0] != n_groups:
        raise ValueError("stamp_center_x/y length mismatch")
    half = stamp // 2
    ny, nx = catalog.shape
    keep = np.ones((n_groups, n_frames), dtype=np.float32)
    hit_tns = np.zeros((n_groups, n_frames), dtype=bool)
    hit_ast = np.zeros((n_groups, n_frames), dtype=bool)

    for fi, btjd in enumerate(btjds):
        full = np.asarray(catalog.mask_at(float(btjd), which="full"), dtype=np.int16)
        for gi in range(n_groups):
            x0 = int(cx[gi]) - half
            y0 = int(cy[gi]) - half
            x1, y1 = x0 + stamp, y0 + stamp
            if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
                # Off-array stamps are already dropped upstream; treat as reject.
                keep[gi, fi] = 0.0
                continue
            cut = full[y0:y1, x0:x1]
            tns = (cut.astype(np.int64) & TNS) != 0
            ast = (cut.astype(np.int64) & ASTEROID) != 0
            if tns.any():
                hit_tns[gi, fi] = True
            if ast.any():
                hit_ast[gi, fi] = True
            if tns.any() or ast.any():
                keep[gi, fi] = 0.0

    n_total = n_groups * n_frames
    n_tns = int(hit_tns.sum())
    n_asteroid = int(hit_ast.sum())
    n_either = int((keep == 0.0).sum())
    stats = {
        "n_groups": n_groups,
        "n_frames": n_frames,
        "n_total": n_total,
        "n_tns": n_tns,
        "n_asteroid": n_asteroid,
        "n_either": n_either,
        "n_kept": int(keep.sum()),
        "frac_tns": float(n_tns) / float(max(n_total, 1)),
        "frac_asteroid": float(n_asteroid) / float(max(n_total, 1)),
        "frac_either": float(n_either) / float(max(n_total, 1)),
        "bits": int(TNS_ASTEROID_BITS),
    }
    return keep, stats


def refresh_stamp_active(
    params: dict[str, jnp.ndarray],
    fd: Any,
    *,
    n_sigma: float = 3.0,
    mask_baseline: np.ndarray | None = None,
) -> dict[str, float]:
    """Recompute ``fd.ctx.stamp_active`` from residuals ANDed with mask baseline.

    Task 2: the MAD median/threshold is pooled over ``pix_ok & (baseline>0)``
    (see ``_pool_active_mask``) instead of plain ``pix_ok`` -- baseline-
    rejected stamps no longer inflate the MAD and loosen the threshold for
    everyone else. The candidate count / returned keep-mask are still plain
    ``pix_ok``: a baseline-rejected candidate that would pass the chi2 cut is
    still evaluated normally, and ``combine_stamp_active`` re-applies the
    baseline AND regardless -- only the threshold's own statistics change.
    Unchanged when ``mask_baseline`` (and ``fd.mask_stamp_active``) are None.
    """
    chi2_red, pix_sum = per_stamp_chi2_red(params, fd)
    chi2_np = np.asarray(chi2_red)
    pix_np = np.asarray(pix_sum)
    pix_ok = pix_np > 0
    baseline = mask_baseline
    if baseline is None:
        baseline = getattr(fd, "mask_stamp_active", None)
    pool_active = _pool_active_mask(pix_np, baseline)
    mad = mad_reject_mask(chi2_np, n_sigma=n_sigma, pix_active=pix_ok, pool_active=pool_active)
    mask = combine_stamp_active(mad, baseline)
    n_cand = int(pix_ok.sum())
    n_keep = int(mask.sum())
    n_rejected = n_cand - n_keep
    frac = float(n_rejected) / float(max(n_cand, 1))
    med = float(np.median(chi2_np[pix_ok])) if n_cand else float("nan")
    fd.ctx = L.with_stamp_active(fd.ctx, mask)
    return {
        "n_rejected": float(n_rejected),
        "n_active_cand": float(n_cand),
        "n_active_kept": float(n_keep),
        "frac_rejected": frac,
        "med_chi2_red": med,
    }


DEFAULT_STATIC_TAU = 4.0
DEFAULT_STATIC_WINDOW = 201


def static_brightness_proxy(fd) -> np.ndarray:
    """Per-group observed stamp flux, summed over pixels and frames.

    The brightness ordinate the static gate detrends against. Taken from the
    *observed* pixels (``fd.data``), so it is a fixed property of the bundle --
    computed once per stage, never per refresh -- and needs no catalog
    magnitude. ``ctx.stamp_snr_weight`` is deliberately NOT used: it is
    ``min(10**(-0.4*(mag-cap)), 1)**2`` floored at a minimum, so it saturates
    flat at 1.0 across most of the bright primary window and would give the
    detrender no ordering there at all.
    """
    data = np.asarray(fd.data, dtype=np.float64)
    valid = getattr(fd.ctx, "pix_valid", None)
    if valid is not None and np.asarray(valid).ndim == 2:
        v = np.asarray(valid, dtype=np.float64)  # (G, P)
        flux = np.einsum("gtp,gp->g", data, v) if data.ndim == 3 else np.nansum(
            data.reshape(data.shape[0], -1), axis=1
        )
    else:
        flux = np.nansum(data.reshape(data.shape[0], -1), axis=1)
    return np.asarray(flux, dtype=np.float64)


def static_rank_outliers(
    log_chi2: np.ndarray,
    brightness: np.ndarray,
    *,
    tau: float = DEFAULT_STATIC_TAU,
    window: int = DEFAULT_STATIC_WINDOW,
    usable: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided outlier mask from a brightness-rank-local median/MAD of log-chi2.

    Pure function, no model state: ``refresh_stamp_active_static``'s decision rule,
    split out so it can be tested against a known brightness trend plus known
    outliers.

    ``chi2_red`` rises steeply and smoothly with stellar brightness (model error
    scales with flux, the noise does not), spanning decades over an 8-13 mag
    sample, so a single pooled MAD would cut the bright end wholesale. Comparing
    each group against its ``window`` nearest neighbours **in brightness order**
    removes that trend without needing magnitude bins.

    Returns ``(drop, dev)``: the boolean cut and the signed local deviation in
    MAD units (NaN where not scored).
    """
    log_chi2 = np.asarray(log_chi2, dtype=np.float64)
    brightness = np.asarray(brightness, dtype=np.float64)
    if log_chi2.shape != brightness.shape:
        raise ValueError(
            f"log_chi2 {log_chi2.shape} and brightness {brightness.shape} must match"
        )
    if usable is None:
        usable = np.isfinite(log_chi2) & np.isfinite(brightness)
    else:
        usable = np.asarray(usable, dtype=bool) & np.isfinite(log_chi2) & np.isfinite(brightness)

    drop = np.zeros(log_chi2.shape, dtype=bool)
    dev = np.full(log_chi2.shape, np.nan)
    if usable.sum() < 8:
        return drop, dev

    idx = np.flatnonzero(usable)
    order = idx[np.argsort(brightness[idx], kind="stable")]
    y = log_chi2[order]
    n = y.size
    half = max(1, min(int(window) // 2, n // 2))
    med = np.empty(n)
    scale = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = y[lo:hi]
        m = np.median(w)
        med[i] = m
        scale[i] = np.median(np.abs(w - m)) * MAD_SCALE
    # A degenerate local window (identical scores) must not reject everything.
    positive = np.isfinite(scale) & (scale > 0)
    floor = float(np.median(scale[positive])) if positive.any() else 1e-4
    floor = max(1e-4, floor)
    scale = np.where(positive, scale, floor)
    d = (y - med) / scale
    dev[order] = d
    drop[order] = d > float(tau)
    return drop, dev


def refresh_stamp_active_static(
    params,
    fds,
    *,
    brightness: list[np.ndarray],
    tau: float = DEFAULT_STATIC_TAU,
    window: int = DEFAULT_STATIC_WINDOW,
    mask_baselines: list | None = None,
) -> dict[str, float]:
    """Whole-star outlier gate for the static (single-FFI) fit.

    Every other mode in this module centres ``log chi2`` **per group across
    frames** (``axis=1``). With one frame that deviation is identically zero, so
    those modes silently reject nothing while still logging as if they ran --
    which is why they are refused outright for a 1-frame bundle rather than
    left to no-op.

    With one frame the only meaningful comparison is *between* stars, and a
    rejection is necessarily a whole-star cut: exactly the intent here (blends,
    variables, asteroid/cosmic hits, background artefacts). A global pooled
    ``log chi2`` MAD cannot be used directly, because ``chi2_red`` rises steeply
    with stellar brightness -- model error scales with flux while the noise does
    not -- spanning several decades over an 8-13 mag sample. So the scale is
    estimated *locally in brightness rank*: groups from every bucket are pooled,
    sorted by ``brightness``, and each group is compared against the median and
    MAD of its ``window`` nearest neighbours in that ordering. That detrends the
    brightness dependence without needing magnitude bins or catalog magnitudes.

    Cuts are one-sided (only implausibly *large* residuals) and are ANDed with
    the physical Gate-C baseline, which is never overridden.
    """
    if not fds:
        return {"n_rejected": 0.0, "n_active_cand": 0.0, "n_active_kept": 0.0,
                "frac_rejected": 0.0, "med_chi2_red": float("nan"),
                "n_downweighted": 0.0, "frac_downweighted": 0.0}
    if len(brightness) != len(fds):
        raise ValueError(
            f"brightness has {len(brightness)} entries for {len(fds)} buckets"
        )
    if mask_baselines is None:
        mask_baselines = [getattr(fd, "mask_stamp_active", None) for fd in fds]

    chi2_all, pix_all, bright_all, owner = [], [], [], []
    for bi, fd_i in enumerate(fds):
        chi2, pix = per_stamp_chi2_red(params, fd_i)
        chi2 = np.asarray(chi2, dtype=np.float64)
        pix = np.asarray(pix)
        # (G, T) -> one score per group: with T == 1 this is the frame itself; with
        # a few frames under a static model it is the group's own median.
        with np.errstate(invalid="ignore"):
            cov = pix > 0
            score = np.where(cov.any(axis=1),
                             np.nanmedian(np.where(cov, chi2, np.nan), axis=1),
                             np.nan)
        chi2_all.append(score)
        pix_all.append(cov.any(axis=1))
        b = np.asarray(brightness[bi], dtype=np.float64)
        if b.shape != score.shape:
            raise ValueError(
                f"bucket {bi}: brightness shape {b.shape} != n_groups {score.shape}"
            )
        bright_all.append(b)
        owner.append(np.full(score.shape, bi, dtype=np.int64))

    score = np.concatenate(chi2_all)
    covered = np.concatenate(pix_all)
    bright = np.concatenate(bright_all)
    log_chi2 = np.log(np.maximum(score, 1e-6))

    usable = covered & np.isfinite(log_chi2) & np.isfinite(bright)
    drop, dev = static_rank_outliers(
        log_chi2, bright, tau=tau, window=window, usable=usable,
    )

    n_cand = 0
    n_keep = 0
    n_dropped = 0
    off = 0
    for fd_i, baseline in zip(fds, mask_baselines):
        act = np.asarray(fd_i.ctx.stamp_active)
        g, t = act.shape
        sel = slice(off, off + g)
        off += g
        base = (np.ones((g, t), dtype=bool) if baseline is None
                else np.asarray(baseline, dtype=bool))
        keep_group = covered[sel] & ~drop[sel]
        new = base & keep_group[:, None]
        fd_i.ctx = L.with_stamp_active(fd_i.ctx, new.astype(np.float32))
        n_cand += int((base & covered[sel][:, None]).sum())
        n_keep += int(new.sum())
        n_dropped += int((drop[sel] & covered[sel]).sum())

    med_chi2_red = (float(np.median(score[covered & np.isfinite(score)]))
                    if np.any(covered & np.isfinite(score)) else float("nan"))
    return {
        "n_rejected": float(n_cand - n_keep),
        "n_active_cand": float(n_cand),
        "n_active_kept": float(n_keep),
        "frac_rejected": float(n_cand - n_keep) / float(max(n_cand, 1)),
        "med_chi2_red": med_chi2_red,
        "n_groups_dropped": float(n_dropped),
        "n_groups_scored": float(int(usable.sum())),
        "max_dev": float(np.nanmax(dev)) if np.any(np.isfinite(dev)) else float("nan"),
        "n_downweighted": 0.0,
        "frac_downweighted": 0.0,
    }


def refresh_stamp_active_multi(
    params: dict[str, jnp.ndarray],
    fds: list[Any],
    *,
    n_sigma: float = 3.0,
    mask_baselines: list[np.ndarray | None] | None = None,
) -> dict[str, float]:
    """Multi-bucket generalization of ``refresh_stamp_active`` (K-bucketed groups,
    see ``groups.bucket_groups_by_size``): one ``fd`` per K tier instead of one
    for the whole region.

    The MAD median/threshold is computed by *pooling* every bucket's active
    chi2 values together first (matching what a single, non-bucketed
    ``refresh_stamp_active`` call would have computed over the same region),
    then applied back to each bucket's own residuals -- bucketing must not
    change *which* stamps get rejected, only how the compute is batched.

    Task 2: "active" for the pooling step means ``pix_ok & (baseline>0)`` per
    bucket (see ``_pool_active_mask``), not plain ``pix_ok`` -- see
    ``refresh_stamp_active``'s docstring for the rationale. The per-bucket
    candidate/keep counts and the applied threshold test are still plain
    ``pix_ok``, unchanged.
    """
    if mask_baselines is None:
        mask_baselines = [getattr(fd, "mask_stamp_active", None) for fd in fds]
    per_bucket = [per_stamp_chi2_red(params, fd) for fd in fds]

    chi2_all = np.concatenate([np.asarray(c, dtype=np.float64).ravel() for c, _ in per_bucket])
    pool_all = np.concatenate([
        _pool_active_mask(np.asarray(p), b).ravel()
        for (_, p), b in zip(per_bucket, mask_baselines)
    ])
    vals = chi2_all[pool_all]
    med = float(np.median(vals)) if vals.size else float("nan")
    thresh = _mad_threshold(vals, n_sigma) if vals.size else float("inf")

    n_cand_total = 0
    n_keep_total = 0
    for (chi2_red, pix_sum), fd, baseline in zip(per_bucket, fds, mask_baselines):
        chi2_np = np.asarray(chi2_red, dtype=np.float64)
        pix_ok = np.asarray(pix_sum) > 0
        mad = np.zeros(chi2_np.shape, dtype=np.float32)
        mad[pix_ok & (chi2_np <= thresh)] = 1.0
        mask = combine_stamp_active(mad, baseline)
        n_cand_total += int(pix_ok.sum())
        n_keep_total += int(mask.sum())
        fd.ctx = L.with_stamp_active(fd.ctx, mask)

    n_rejected = n_cand_total - n_keep_total
    frac = float(n_rejected) / float(max(n_cand_total, 1))
    return {
        "n_rejected": float(n_rejected),
        "n_active_cand": float(n_cand_total),
        "n_active_kept": float(n_keep_total),
        "frac_rejected": frac,
        "med_chi2_red": med,
    }


# ---------------------------------------------------------------------------
# Task 3: two-level replacement -- level-2 automatic per-cadence gate, level-1
# audit-only per-group mismatch table. New functions ALONGSIDE the pooled-chi2
# gate above; nothing above is deleted, so the two remain directly A/B-able.
# ---------------------------------------------------------------------------


def _safe_log_chi2(chi2_red: np.ndarray) -> np.ndarray:
    """``log(chi2_red)``, NaN wherever ``chi2_red`` is non-positive or non-finite.

    ``per_stamp_chi2_red`` floors its denominator (``pix_sum``) but not its
    numerator: a stamp with zero pixel weight everywhere (fully masked) or a
    (numerically vanishing) perfect fit can legitimately produce
    ``chi2_red <= 0``, and ``log`` of that is ``-inf``/``nan``. Guarded here
    once so every downstream consumer (``per_group_log_chi2_baseline``,
    ``per_cadence_log_deviation``, ``level1_group_mismatch_table``) sees a
    clean NaN instead of an inf that would otherwise poison medians/MAD.
    """
    chi2 = np.asarray(chi2_red, dtype=np.float64)
    safe = np.where(np.isfinite(chi2) & (chi2 > 0), chi2, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(safe)


def per_group_log_chi2_baseline(
    chi2_red: np.ndarray,
    *,
    pool_active: np.ndarray,
) -> np.ndarray:
    """``b_g = median_t( log(chi2_red[g,:]) )`` over ``pool_active`` cadences.

    This is the per-group constant the level-2 gate subtracts out (and what
    the level-1 audit table ranks groups by -- level 1 surfaces exactly what
    level 2 discards). Groups with no active cadence get NaN.
    """
    log_chi2 = _safe_log_chi2(chi2_red)
    pool = np.asarray(pool_active, dtype=bool)
    masked = np.where(pool, log_chi2, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN group row
        b_g = np.nanmedian(masked, axis=1)
    return b_g


def per_cadence_log_deviation(
    chi2_red: np.ndarray,
    *,
    pool_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group-centred log-chi2 deviation ``d[g,t]`` and the per-group baseline ``b_g``.

        d[g,t] = log(chi2_red[g,t]) - median_t( log(chi2_red[g,:]) )

    Per-group centring removes the ~5-decade between-group spread in raw
    ``chi2_red`` that forces the pooled linear-domain gate
    (``refresh_stamp_active_multi``) into whole-group, all-or-nothing cuts;
    the log handles the multiplicative, right-skewed tail (chi2 is
    model-error dominated here, not Gaussian noise -- median chi2_red ~ 300
    on the reference bundle). What's left, ``d``, is the CV~0.08 per-cadence
    deviation of each group's own residual level: tight and symmetric enough
    for a MAD-based scale to actually mean something (see
    ``level2_mad_scale``).

    Returns ``(d, b_g)``, both computed relative to ``pool_active`` cadences
    (typically ``pix_ok & (baseline>0)``, see ``_pool_active_mask``); ``d``
    is defined for every ``(g, t)`` cell (including inactive ones, which the
    caller is expected to mask out downstream), ``b_g`` is ``(n_groups,)``.
    """
    log_chi2 = _safe_log_chi2(chi2_red)
    b_g = per_group_log_chi2_baseline(chi2_red, pool_active=pool_active)
    d = log_chi2 - b_g[:, None]
    return d, b_g


def level2_mad_scale(d: np.ndarray, pool_active: np.ndarray) -> float:
    """``1.4826 * MAD(d)`` pooled over ``pool_active`` cells (NaN-safe)."""
    pool = np.asarray(pool_active, dtype=bool)
    d = np.asarray(d, dtype=np.float64)
    finite = pool & np.isfinite(d)
    vals = d[finite]
    if vals.size == 0:
        return float("nan")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return MAD_SCALE * mad


DEFAULT_L2_CHURN_CAP_FRAC = 0.05
DEFAULT_L2_HYSTERESIS_N = 2

# The standardized gate deliberately lives alongside ``refresh_stamp_active_v2``.
# Keeping v2 intact makes old checkpoints/runs exactly reproducible while the
# scientifically safer, continuous-weight gate can be selected explicitly.
DEFAULT_SCALE_SHRINKAGE_FRAMES = 20.0
DEFAULT_STANDARDIZED_MIN_WEIGHT = 0.05


@dataclass
class _BucketGateState:
    """Per-bucket hysteresis state for one ``Level2GateState``.

    Shapes lock in on the bucket's first refresh through this state; a later
    call with a different ``(n_groups, n_frames)`` shape for the same bucket
    index resets the streak/committed arrays (treated as a fresh bucket).
    """

    streak: np.ndarray | None = None
    committed_reject: np.ndarray | None = None


@dataclass
class Level2GateState:
    """Mutable state threaded across ``refresh_stamp_active_v2`` calls within
    one training stage.

    Own one instance per stage (a fresh instance at each stage boundary,
    mirroring ``fit.run_stage``'s per-stage optimizer reset) and pass the
    *same* instance back in on every refresh within that stage -- it is
    mutated in place, never replaced, so holding a reference is enough.

    ``scale``: the frozen/EMA level-2 MAD scale (``None`` until the first
    refresh sets it; see ``refresh_stamp_active_v2``'s ``freeze_scale_after_first``
    / ``scale_ema_alpha``).
    ``n_refreshes``: how many refreshes this state has been used for.
    """

    scale: float | None = None
    n_refreshes: int = 0
    _buckets: dict[int, _BucketGateState] = field(default_factory=dict)

    def bucket(self, i: int) -> _BucketGateState:
        b = self._buckets.get(i)
        if b is None:
            b = _BucketGateState()
            self._buckets[i] = b
        return b


@dataclass
class _BucketStandardizedState:
    """State for one bucket of the continuous standardized gate."""

    weights: np.ndarray | None = None
    streak: np.ndarray | None = None
    scales: np.ndarray | None = None


@dataclass
class StandardizedGateState:
    """Persistent continuous-weight gate state for one training stage.

    Reuse this object across residual refreshes to obtain hysteresis and
    re-entry.  Construct a new state when the frame geometry changes.  A
    bucket whose shape changes is reset automatically, which also makes a
    curriculum-stage boundary safe even if the caller accidentally reuses
    the containing object.
    """

    n_refreshes: int = 0
    _buckets: dict[int, _BucketStandardizedState] = field(default_factory=dict)

    def bucket(self, i: int) -> _BucketStandardizedState:
        b = self._buckets.get(i)
        if b is None:
            b = _BucketStandardizedState()
            self._buckets[i] = b
        return b


def hierarchical_group_log_scales(
    d: np.ndarray,
    pool_active: np.ndarray,
    *,
    shrinkage_frames: float = DEFAULT_SCALE_SHRINKAGE_FRAMES,
    scale_floor: float = 1e-6,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return per-group robust log-deviation scales shrunk to a pooled scale.

    For group ``g``, ``w_g = n_g / (n_g + shrinkage_frames)`` and the
    variance estimate is ``w_g*s_g**2 + (1-w_g)*s_pool**2``.  Thus short or
    degenerate light curves borrow strength from the population, while long
    light curves retain their measured temporal scatter.  Calibration uses
    every finite ``pool_active`` cell; callers must pass the physical
    baseline mask, not the previous residual weights.
    """
    values = np.asarray(d, dtype=np.float64)
    pool = np.asarray(pool_active, dtype=bool)
    if values.ndim != 2 or pool.shape != values.shape:
        raise ValueError("d and pool_active must be same-shape 2-D arrays")
    finite = pool & np.isfinite(values)
    counts = finite.sum(axis=1).astype(np.int64)
    pooled = level2_mad_scale(values, finite)
    if not np.isfinite(pooled) or pooled < scale_floor:
        pooled = float(scale_floor)
    raw = np.full(values.shape[0], np.nan, dtype=np.float64)
    for gi in range(values.shape[0]):
        vals = values[gi, finite[gi]]
        if vals.size:
            med = np.median(vals)
            raw[gi] = MAD_SCALE * np.median(np.abs(vals - med))
    raw = np.where(np.isfinite(raw), np.maximum(raw, scale_floor), pooled)
    strength = max(float(shrinkage_frames), 0.0)
    frac = counts / (counts + strength) if strength else np.ones_like(counts, dtype=float)
    scales = np.sqrt(frac * raw**2 + (1.0 - frac) * pooled**2)
    return np.maximum(scales, scale_floor), float(pooled), counts


def one_sided_robust_stamp_weights(
    d: np.ndarray,
    group_scales: np.ndarray,
    *,
    n_sigma: float = 3.0,
    min_weight: float = DEFAULT_STANDARDIZED_MIN_WEIGHT,
) -> np.ndarray:
    """Continuous Huber influence weights for positive temporal excursions.

    Negative/ordinary deviations retain weight one.  Above ``n_sigma`` the
    weight decays as ``n_sigma / z`` and is floored strictly above zero;
    therefore residual evidence alone never hard-deletes a stamp.  Nonfinite
    scores receive the floor, while physical-mask baselines are applied by
    the refresh function and remain exact hard zeros.
    """
    values = np.asarray(d, dtype=np.float64)
    scales = np.asarray(group_scales, dtype=np.float64)
    if values.ndim != 2 or scales.shape != (values.shape[0],):
        raise ValueError("group_scales must have one value per row of d")
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive")
    if not 0.0 < min_weight <= 1.0:
        raise ValueError("min_weight must be in (0, 1]")
    z = np.maximum(values, 0.0) / np.maximum(scales[:, None], 1e-12)
    out = np.ones_like(values, dtype=np.float64)
    tail = np.isfinite(z) & (z > n_sigma)
    out[tail] = n_sigma / z[tail]
    out[~np.isfinite(z)] = min_weight
    return np.clip(out, min_weight, 1.0).astype(np.float32)


def level1_group_mismatch_table(
    chi2_red: np.ndarray,
    *,
    pool_active: np.ndarray,
    group_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Ranked per-group persistent-mismatch AUDIT table. Never cuts anything.

    ``baseline_log_chi2`` is ``b_g = median_t( log(chi2_red[g,:]) )`` -- the
    same per-group constant the level-2 gate centres out. A large positive
    value means "this group is persistently a much worse fit than the
    typical group, on every cadence" -- exactly the between-group spread
    level 2 is designed to ignore (it's a model/astrophysics mismatch, not a
    transient per-cadence anomaly). Sorted worst first.

    This function only reports; it never zeros a stamp. Use
    ``apply_group_exclusions`` with an explicit, human-reviewed group list to
    actually act on a finding here -- level 1 has no automatic cut path by
    design (see the module docstring and Task 3's rationale: whole-group
    persistent mismatch needs a human judgment call, not a threshold).
    """
    import pandas as pd  # noqa: PLC0415  (diagnostic compatibility only)

    return pd.DataFrame(level1_group_mismatch_rows(
        chi2_red, pool_active=pool_active, group_ids=group_ids,
    )).sort_values(
        "baseline_log_chi2", ascending=False, na_position="last",
    ).reset_index(drop=True)


def level1_group_mismatch_rows(
    chi2_red: np.ndarray,
    *,
    pool_active: np.ndarray,
    group_ids: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Lean equivalent of :func:`level1_group_mismatch_table`.

    Returns ordinary dictionaries so the GPU training path can write its audit
    CSV with the standard library and never import pandas.
    """
    chi2_red = np.asarray(chi2_red, dtype=np.float64)
    pool = np.asarray(pool_active, dtype=bool)
    b_g = per_group_log_chi2_baseline(chi2_red, pool_active=pool)
    n_active = pool.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med_chi2 = np.nanmedian(np.where(pool, chi2_red, np.nan), axis=1)
    n_groups, n_frames = chi2_red.shape
    gid = np.arange(n_groups) if group_ids is None else np.asarray(group_ids)
    rows = [
        {
            "group_id": int(gid[i]),
            "baseline_log_chi2": float(b_g[i]),
            "median_chi2_red": float(med_chi2[i]),
            "n_active_frames": int(n_active[i]),
            "n_frames": int(n_frames),
        }
        for i in range(n_groups)
    ]
    return sorted(
        rows,
        key=lambda row: (
            not np.isfinite(row["baseline_log_chi2"]),
            -row["baseline_log_chi2"] if np.isfinite(row["baseline_log_chi2"]) else 0.0,
        ),
    )


def apply_group_exclusions(
    stamp_active: np.ndarray,
    excluded_group_ids: list[int] | np.ndarray | None,
    *,
    group_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Hard-zero every frame for explicitly human-excluded groups.

    This is level 1's *only* effect path -- it is never invoked
    automatically by ``refresh_stamp_active_v2``. ``group_ids`` maps
    ``stamp_active``'s row index to a (possibly non-contiguous/bucket-local)
    group id; defaults to the row index itself.
    """
    out = np.asarray(stamp_active, dtype=np.float32).copy()
    excl_list = list(excluded_group_ids) if excluded_group_ids else []
    if not excl_list:
        return out
    excl = {int(g) for g in excl_list}
    gid = np.arange(out.shape[0]) if group_ids is None else np.asarray(group_ids)
    rows = np.array([i for i, g in enumerate(gid) if int(g) in excl], dtype=int)
    if rows.size:
        out[rows, :] = 0.0
    return out


def refresh_stamp_active_v2(
    params: dict[str, jnp.ndarray],
    fds: list[Any],
    *,
    n_sigma: float = 3.0,
    mask_baselines: list[np.ndarray | None] | None = None,
    state: Level2GateState | None = None,
    freeze_scale_after_first: bool = True,
    scale_ema_alpha: float = 1.0,
    churn_cap_frac: float = DEFAULT_L2_CHURN_CAP_FRAC,
    hysteresis_n: int = DEFAULT_L2_HYSTERESIS_N,
) -> dict[str, float]:
    """Level-2 automatic gate: per-cadence log-deviation MAD.

    Drop-in alternative to ``refresh_stamp_active_multi`` (same ``params``/
    ``fds``/``mask_baselines`` calling convention, same K-bucketed pooling
    structure) that replaces the pooled *linear* chi2 MAD with a per-group
    centred, log-domain, per-cadence MAD::

        d[g,t] = log(chi2_red[g,t]) - median_t(log(chi2_red[g,:]))
        scale  = 1.4826 * MAD(d), pooled over active cells (pix_ok & baseline)
        reject where d > n_sigma * scale

    Rationale (measured on ``dev/forward_epsf_wcs/output/rejection_study/``;
    see the module docstring): per-group centring removes the ~5-decade
    between-group spread in raw ``chi2_red`` that forced the old pooled gate
    into whole-group, all-or-nothing cuts (every group's reject fraction was
    measured at exactly 0 or 1). The log handles chi2's multiplicative,
    right-skewed tail -- this is model-error dominated (median chi2_red ~
    300, i.e. |chi| ~ 17/pixel), not Gaussian noise, so a linear "3 sigma"
    clip on the raw value is meaningless. What survives centring is the
    CV~0.08 temporal deviation of each group's own residual level: the only
    quantity here tight and symmetric enough for MAD/1.4826 to be a valid
    spread estimate.

    Anti-runaway knobs (optional, conservative defaults -- the old gate's
    failure mode was partly a *runaway*: its threshold moved 5935 -> 1219
    because it was recomputed every refresh on a distribution whose tail it
    had just deleted; these knobs stop level 2 from doing the same thing with
    a different score):
      - ``freeze_scale_after_first`` (default True): after this ``state``'s
        first-ever refresh sets ``state.scale``, later refreshes reuse it
        verbatim rather than recomputing on a self-selected sample.
      - ``scale_ema_alpha`` (default 1.0, i.e. no smoothing): only consulted
        when ``freeze_scale_after_first`` is False; EMA-shrinks scale
        updates (``new = alpha*raw + (1-alpha)*old``).
      - ``churn_cap_frac`` (default 0.05 = 5% of a bucket's active
        candidates): caps how many cells may change committed reject state
        per refresh; on overflow, only the highest-confidence flips
        (largest ``|d - n_sigma*scale|``) are applied and a loud
        ``runtime.log`` warning is emitted -- the rest simply carry their
        hysteresis evidence into the next refresh.
      - ``hysteresis_n`` (default 2): a cell only flips its *committed*
        state after this many consecutive refreshes of same-direction raw
        evidence; a single noisy refresh can never flip anything.

    ``state`` (a ``Level2GateState``) is mutated in place and must be reused
    across every refresh within one training stage for the freeze/EMA/
    hysteresis machinery to mean anything; pass ``None`` to get a fresh,
    single-call state -- ``hysteresis_n`` is forced to 1 for that call (a raw
    reject/keep is immediately decisive, since there is no prior streak to
    accumulate against), which is the literal "disabling hysteresis for that
    one call" this always claimed; scale freezing is a no-op on a fresh state
    regardless (nothing to freeze against yet). Still churn-capped.

    Returns a stats dict that is a strict superset of
    ``refresh_stamp_active_multi``'s keys (``n_rejected``, ``n_active_cand``,
    ``n_active_kept``, ``frac_rejected``, ``med_chi2_red``) so
    ``fit.run_stage``'s existing logging works unchanged if this function is
    swapped in for it; adds ``scale``, ``raw_scale``, ``n_flipped``,
    ``n_churn_capped``, ``n_refreshes``.
    """
    if mask_baselines is None:
        mask_baselines = [getattr(fd, "mask_stamp_active", None) for fd in fds]
    # Bug fix (audit): a caller passing state=None gets a brand-new
    # Level2GateState with all-zero streaks. On that single call, a cell whose
    # very first raw_reject sets its streak to 1 (see the `np.where(streak > 0,
    # streak + 1, 1)` below) can never satisfy `new_streak >= hysteresis_n` for
    # the default hysteresis_n=2 -- so nothing is EVER rejected, silently,
    # contradicting this function's own docstring ("equivalent to disabling
    # ... hysteresis for that one call"). Effective hysteresis_n=1 for a
    # caller-anonymous (state=None) single-shot call restores that documented
    # behavior: one refresh's raw evidence is immediately decisive, with no
    # prior state to have "frozen" or accumulated a streak against. A caller
    # that threads its own persistent state across refreshes (the normal
    # training-loop usage) is unaffected -- it always gets the real
    # `hysteresis_n`.
    state_was_none = state is None
    if state is None:
        state = Level2GateState()
    hysteresis_n_eff = 1 if state_was_none else int(hysteresis_n)

    per_bucket = [per_stamp_chi2_red(params, fd) for fd in fds]
    chi2_nps = [np.asarray(c, dtype=np.float64) for c, _ in per_bucket]
    pix_sums = [np.asarray(p) for _, p in per_bucket]
    pool_masks = [_pool_active_mask(p, b) for p, b in zip(pix_sums, mask_baselines)]
    pix_oks = [p > 0 for p in pix_sums]

    d_list: list[np.ndarray] = []
    for chi2_np, pool in zip(chi2_nps, pool_masks):
        d, _b_g = per_cadence_log_deviation(chi2_np, pool_active=pool)
        d_list.append(d)

    d_all = np.concatenate([d.ravel() for d in d_list]) if d_list else np.zeros((0,))
    pool_all = np.concatenate([m.ravel() for m in pool_masks]) if pool_masks else np.zeros((0,), dtype=bool)
    raw_scale = level2_mad_scale(d_all, pool_all)

    if state.scale is None or not np.isfinite(state.scale):
        state.scale = raw_scale
    elif not freeze_scale_after_first:
        alpha = float(scale_ema_alpha)
        state.scale = alpha * raw_scale + (1.0 - alpha) * state.scale
    # else: freeze_scale_after_first and state.scale already set -> keep it.
    state.n_refreshes += 1
    scale = state.scale if state.scale is not None and np.isfinite(state.scale) else raw_scale

    n_cand_total = 0
    n_keep_total = 0
    n_flipped_total = 0
    n_capped_total = 0
    med_vals: list[np.ndarray] = []

    for bi, (chi2_np, pix_ok, pool, d, fd, baseline) in enumerate(
        zip(chi2_nps, pix_oks, pool_masks, d_list, fds, mask_baselines),
    ):
        bstate = state.bucket(bi)
        shape = chi2_np.shape
        if bstate.streak is None or bstate.streak.shape != shape:
            bstate.streak = np.zeros(shape, dtype=np.int32)
            bstate.committed_reject = np.zeros(shape, dtype=bool)

        raw_reject = pix_ok & np.isfinite(d) & (d > n_sigma * scale)
        raw_keep = pix_ok & np.isfinite(d) & ~raw_reject

        streak = bstate.streak
        new_streak = streak.copy()
        new_streak = np.where(raw_reject, np.where(streak > 0, streak + 1, 1), new_streak)
        new_streak = np.where(raw_keep, np.where(streak < 0, streak - 1, -1), new_streak)
        bstate.streak = new_streak

        committed = bstate.committed_reject
        desired = committed.copy()
        flip_to_reject = (~committed) & (new_streak >= hysteresis_n_eff)
        flip_to_keep = committed & (new_streak <= -hysteresis_n_eff)
        desired[flip_to_reject] = True
        desired[flip_to_keep] = False

        candidate_flip = (desired != committed) & pix_ok
        n_active = int(pix_ok.sum())
        n_flip = int(candidate_flip.sum())
        max_flips = max(1, int(round(churn_cap_frac * n_active))) if n_active else 0
        if n_flip > max_flips:
            margin = np.abs(d - n_sigma * scale)
            margin = np.where(candidate_flip, margin, -np.inf)
            flat_order = np.argsort(margin, axis=None)[::-1]
            keep_flip = np.zeros(int(np.prod(shape)), dtype=bool)
            keep_flip[flat_order[:max_flips]] = True
            candidate_flip = keep_flip.reshape(shape)
            n_capped = n_flip - max_flips
            n_capped_total += n_capped
            RT.log(
                f"stamp_reject L2 gate: churn cap hit on bucket {bi} "
                f"({n_flip} candidate flips > cap {max_flips} = "
                f"{churn_cap_frac:.1%} of {n_active} active cells); applying "
                f"{max_flips}, deferring {n_capped} to the next refresh"
            )

        committed_new = committed.copy()
        committed_new[candidate_flip] = desired[candidate_flip]
        bstate.committed_reject = committed_new
        n_flipped_total += int(candidate_flip.sum())

        mad = np.zeros(shape, dtype=np.float32)
        mad[pix_ok & ~committed_new] = 1.0
        mask = combine_stamp_active(mad, baseline)
        fd.ctx = L.with_stamp_active(fd.ctx, mask)

        n_cand_total += int(pix_ok.sum())
        n_keep_total += int(mask.sum())
        if pix_ok.any():
            med_vals.append(chi2_np[pix_ok])

    n_rejected = n_cand_total - n_keep_total
    frac = float(n_rejected) / float(max(n_cand_total, 1))
    med = float(np.median(np.concatenate(med_vals))) if med_vals else float("nan")

    return {
        "n_rejected": float(n_rejected),
        "n_active_cand": float(n_cand_total),
        "n_active_kept": float(n_keep_total),
        "frac_rejected": frac,
        "med_chi2_red": med,
        "scale": float(scale) if scale is not None else float("nan"),
        "raw_scale": float(raw_scale),
        "n_flipped": float(n_flipped_total),
        "n_churn_capped": float(n_capped_total),
        "n_refreshes": float(state.n_refreshes),
    }


def refresh_stamp_active_standardized(
    params: dict[str, jnp.ndarray],
    fds: list[Any],
    *,
    n_sigma: float = 3.0,
    mask_baselines: list[np.ndarray | None] | None = None,
    state: StandardizedGateState | None = None,
    shrinkage_frames: float = DEFAULT_SCALE_SHRINKAGE_FRAMES,
    min_weight: float = DEFAULT_STANDARDIZED_MIN_WEIGHT,
    hysteresis_n: int = DEFAULT_L2_HYSTERESIS_N,
    freeze_scales_after_first: bool = False,
) -> dict[str, float]:
    """Apply hierarchically standardized, continuous residual weights.

    This is the integration entry point for the ``standardized`` rejection
    mode.  Each group's log-deviation MAD is shrunk toward the pooled MAD as
    a function of its number of baseline-valid frames.  Ordinary positive
    excursions receive continuous Huber influence weights; only a supplied
    physical baseline can produce zero.

    Calibration is intentionally independent of prior residual decisions:
    it uses ``pix_ok & baseline`` and never reads ``ctx.stamp_active``.  With
    persistent ``state``, a downweight or recovery must be supported on
    ``hysteresis_n`` consecutive refreshes.  Recovery is symmetric and so a
    previously downweighted cadence can re-enter.  Passing ``state=None`` is
    a one-shot audit and applies the current continuous targets immediately.
    """
    if hysteresis_n < 1:
        raise ValueError("hysteresis_n must be >= 1")
    if mask_baselines is None:
        mask_baselines = [getattr(fd, "mask_stamp_active", None) for fd in fds]
    if len(mask_baselines) != len(fds):
        raise ValueError("mask_baselines must have one entry per FitData bucket")
    anonymous = state is None
    if state is None:
        state = StandardizedGateState()
    hysteresis_eff = 1 if anonymous else int(hysteresis_n)

    per_bucket = [per_stamp_chi2_red(params, fd) for fd in fds]
    n_cand = 0
    n_kept = 0
    n_downweighted = 0
    n_changed = 0
    pooled_scales: list[float] = []
    group_scales_all: list[np.ndarray] = []
    med_vals: list[np.ndarray] = []

    for bi, ((chi2, pix_sum), fd, baseline) in enumerate(zip(per_bucket, fds, mask_baselines)):
        chi2_np = np.asarray(chi2, dtype=np.float64)
        pix_np = np.asarray(pix_sum)
        pix_ok = pix_np > 0
        # _pool_active_mask depends only on pixel support and the sticky
        # physical baseline.  It deliberately ignores prior residual weights.
        calibration = _pool_active_mask(pix_np, baseline)
        d, _ = per_cadence_log_deviation(chi2_np, pool_active=calibration)
        scales, pooled, _counts = hierarchical_group_log_scales(
            d, calibration, shrinkage_frames=shrinkage_frames,
        )
        bstate = state.bucket(bi)
        shape = chi2_np.shape
        if bstate.weights is None or bstate.weights.shape != shape:
            bstate.weights = np.ones(shape, dtype=np.float32)
            bstate.streak = np.zeros(shape, dtype=np.int32)
            bstate.scales = None
        if freeze_scales_after_first and bstate.scales is not None:
            scales = bstate.scales
        else:
            bstate.scales = scales.copy()

        target = one_sided_robust_stamp_weights(
            d, scales, n_sigma=n_sigma, min_weight=min_weight,
        )
        # Cells lacking pixel support do not participate in residual state.
        target = np.where(pix_ok, target, 1.0).astype(np.float32)
        current = bstate.weights
        eps = 1e-7
        wants_down = target < current - eps
        wants_up = target > current + eps
        streak = bstate.streak
        streak = np.where(wants_down, np.where(streak > 0, streak + 1, 1), streak)
        streak = np.where(wants_up, np.where(streak < 0, streak - 1, -1), streak)
        streak = np.where(~(wants_down | wants_up), 0, streak)
        eligible = (wants_down & (streak >= hysteresis_eff)) | (wants_up & (streak <= -hysteresis_eff))
        updated = current.copy()
        updated[eligible] = target[eligible]
        # Once admitted, follow a continuing tail smoothly; the initial
        # transition (and direction reversal/re-entry) remains hysteretic.
        same_side_tail = (current < 1.0 - eps) & (target < 1.0 - eps) & ~wants_up
        updated[same_side_tail] = target[same_side_tail]
        bstate.weights = updated
        bstate.streak = streak
        n_changed += int(np.count_nonzero(np.abs(updated - current) > eps))

        baseline_np = np.ones(shape, dtype=np.float32) if baseline is None else np.asarray(baseline, dtype=np.float32)
        applied = updated * (baseline_np > 0).astype(np.float32)
        fd.ctx = L.with_stamp_active(fd.ctx, applied)

        n_cand += int(pix_ok.sum())
        n_kept += int(np.count_nonzero(applied > 0))
        n_downweighted += int(np.count_nonzero(pix_ok & (applied > 0) & (applied < 1.0 - eps)))
        pooled_scales.append(pooled)
        group_scales_all.append(scales)
        if pix_ok.any():
            med_vals.append(chi2_np[pix_ok])

    state.n_refreshes += 1
    n_hard_rejected = n_cand - n_kept
    all_scales = np.concatenate(group_scales_all) if group_scales_all else np.array([], dtype=float)
    return {
        "n_rejected": float(n_hard_rejected),
        "n_active_cand": float(n_cand),
        "n_active_kept": float(n_kept),
        "frac_rejected": float(n_hard_rejected) / float(max(n_cand, 1)),
        "med_chi2_red": float(np.median(np.concatenate(med_vals))) if med_vals else float("nan"),
        "n_downweighted": float(n_downweighted),
        "frac_downweighted": float(n_downweighted) / float(max(n_cand, 1)),
        "n_weight_changed": float(n_changed),
        "pooled_scale": float(np.median(pooled_scales)) if pooled_scales else float("nan"),
        "median_group_scale": float(np.median(all_scales)) if all_scales.size else float("nan"),
        "n_refreshes": float(state.n_refreshes),
    }
