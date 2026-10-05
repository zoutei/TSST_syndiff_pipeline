# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Adam (via optax) + staged unfreeze + warm-start + progress logging.

Stage schedule:
    0: warm-start only
    1: WCS only (dx-only forward with cached local ePSF by default)
    2: + epsf_base_raw (optional soft freeze of WCS for first N steps)
    3: + epsf_modes, w_coeff

Performance notes:
- ``stamp_active`` is a dynamic step argument (no re-JIT on reject refresh).
- Occupied slots are packed in ``forward_model``; stage 1 skips blend/recenter.
"""

from __future__ import annotations

import json
import os
import time
import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import _bootstrap  # noqa: F401
from . import cheb_wcs as CW
from . import epsf_model as EM
from . import loss as L
from . import runtime as RT
from .loss import LossWeights, StaticContext, total_loss

# pandas / data_io / fit_wcs_from_centroids are warmstart-only (lazy in
# warmstart_wcs_coeff). stamp_reject is imported only when reject_every > 0.

def _log(msg: str) -> None:
    RT.log(msg)


STAGE_LEAVES = ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff")

# 1 = WCS, 2 = + ePSF base, 3 = + modes/w/parametric chroma, 4 = + the free
# colour image. Stage 4 exists so the free image trains only once everything it
# could otherwise absorb has settled; a 3-entry --steps-per-stage still means
# "no stage 4", bit-identically to before it existed.
N_STAGES = 4

# ``w_coeff``'s learning-rate scale, relative to the stage learning rate.
# w(t) is measured in units where the fitted values are ~1e-5 and the sky's
# true temporal width signal is ~2e-6 (0.08% PSF width), so the default puts
# one Adam step at roughly 1% of the signal instead of 5x the parameter.
W_LR_SCALE_DEFAULT = 2e-4
# ``chroma_image``'s learning-rate scale, relative to the stage learning rate,
# and its own optimizer bucket. Measured 2026-09-16 on the s24 single-FFI bundle:
# sharing the parametric ``train_chroma`` bucket at scale 1.0 took the loss from
# 68.03 to 104.37 in ONE step. Two separate reasons, both already documented for
# ``w_coeff`` in ``_leaf_labels``:
#   1. UNITS. The image lives in the ePSF's flux-fraction units (its meaningful
#      amplitude is ~4e-4, i.e. a couple of percent of a node's 0.02 peak), while
#      chroma_shift is in pixels and chroma_dilation is a fractional scale. One
#      Adam step at the shared rate is many times the whole quantity.
#   2. THE CLIP. ``optax.multi_transform`` masks each bucket, so
#      ``clip_by_global_norm`` is evaluated per bucket. 3364 image parameters
#      dominate the norm of a bucket that also holds 180 parametric numbers, so
#      the clip factor chosen for the image rescales the parametric leaves too.
CHROMA_IMAGE_LR_SCALE_DEFAULT = 1e-2
# Present only when the chromatic term is enabled. Kept out of STAGE_LEAVES so every
# pre-chroma checkpoint still loads: those are REQUIRED, these are carried when found.
# Deliberately left as exactly the original pair -- an existing test
# (test_end_to_end_chroma_trains_and_survives_the_stage_handoff) asserts every key in
# this tuple is present in the stage-1 checkpoint whenever --chroma alone is passed,
# which is correct for shift+dilation but would be WRONG for the C1 colour-affine
# leaves below: those are default-off behind --chroma-affine/--chroma-kurt, so a
# plain --chroma run must not be required to carry them. See CHROMA_AFFINE_LEAVES.
OPTIONAL_LEAVES = ("chroma_shift", "chroma_dilation")
# C1 colour-affine extension (default-off; see --chroma-affine) and the optional
# flux-neutral kurtosis leaf (default-off; see --chroma-kurt). Layered on top of
# OPTIONAL_LEAVES, never required by it -- kept as separate tuples for exactly the
# reason in the comment above.
CHROMA_AFFINE_LEAVES = ("chroma_aniso", "chroma_shear")
CHROMA_KURT_LEAVES = ("chroma_kurt",)
# Additive chromatic halo on a fixed r^-index profile (default-off behind
# --chroma-halo). Unfrozen at stage 3 with the rest of the colour family; it needs no
# prerequisite leaf because it is additive rather than a reshaping of the base.
CHROMA_HALO_LEAVES = ("chroma_halo",)
# The free dP/dcolour image (default-off behind --chroma-image), unfrozen at
# STAGE 4 only -- after the WCS, the ePSF base and the parametric colour terms
# have all converged. See _leaf_labels.
CHROMA_IMAGE_LEAVES = ("chroma_image",)
# Every leaf that should be carried through checkpoints/bootstrap/labeling when
# present, regardless of which of the flags above produced it. Use this (not the
# narrower OPTIONAL_LEAVES) for any "carry whichever optional leaves both sides have"
# loop -- see checkpoint_history.save_params_leaves_npz, load_params_npz, and the
# --init-params merge in train_loop.py/run_fit.py.
CHROMA_G8_LEAVES = ("chroma_g8",)  # global 8-parameter colour model (loss.CHROMA_G8_LEAVES)
# Brightness-width blur coefficient (bright_width.BRIGHT_WIDTH_LEAVES): trains in the chroma
# bucket at stage 3 like chroma_g8 (scene_fit can hold it fixed instead).
BRIGHT_WIDTH_LEAVES = ("bright_width",)
ALL_OPTIONAL_LEAVES = (
    OPTIONAL_LEAVES + CHROMA_AFFINE_LEAVES + CHROMA_KURT_LEAVES
    + CHROMA_HALO_LEAVES + CHROMA_IMAGE_LEAVES + CHROMA_G8_LEAVES + BRIGHT_WIDTH_LEAVES
)
ALL_LEAVES = STAGE_LEAVES + ALL_OPTIONAL_LEAVES


def save_params_npz(path: Path, params: dict, *, stamp_active: np.ndarray | None = None) -> None:
    """Atomically write fit params (raw + decoded ePSF fields) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save = dict(params)
    save["epsf_base"] = EM.decode_epsf_base(params["epsf_base_raw"])
    save["epsf_modes_decoded"] = EM.decode_epsf_modes(params["epsf_modes"], save["epsf_base"])
    if "chroma_image" in params:
        # The gauged image is what the model renders; post-fit reads this, not the
        # raw leaf, exactly as it reads epsf_modes_decoded rather than epsf_modes.
        save["chroma_image_decoded"] = EM.decode_chroma_image(
            params["chroma_image"], save["epsf_base"]
        )
    if stamp_active is not None:
        save["stamp_active"] = np.asarray(stamp_active, dtype=np.float32)
    save["epsf_repr"] = EM.EPSF_REPR
    arrays = {k: np.asarray(v) for k, v in save.items()}
    tmp = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def save_stamp_active_npz(path: Path, stamp_active: np.ndarray) -> None:
    """Atomically write the full (n_groups, n_frames) keep mask (1=active, 0=rejected)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(stamp_active, dtype=np.float32)
    tmp = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez(tmp, stamp_active=arr)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def load_stamp_active_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "stamp_active" not in data.files:
            raise KeyError(f"{path}: missing stamp_active array")
        return np.asarray(data["stamp_active"], dtype=np.float32)


def merge_fds_stamp_active(
    mask_active: np.ndarray,
    fds: list[FitData],
    buckets: list,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Copy per-bucket ``ctx.stamp_active`` into a global (G, T) mask."""
    mask = np.asarray(mask_active, dtype=np.float32)
    fidx = np.asarray(frame_indices, dtype=np.intp)
    for fd_i, entry in zip(fds, buckets):
        bidx = np.asarray(entry[3], dtype=np.intp)
        mask[np.ix_(bidx, fidx)] = np.asarray(fd_i.ctx.stamp_active, dtype=np.float32)
    return mask




def load_params_npz(path: Path) -> dict[str, jnp.ndarray]:
    """Load optimizer leaves from a ``save_params_npz`` checkpoint.

    Decoded convenience arrays (``epsf_base``, ``epsf_modes_decoded``) are ignored.
    """
    path = Path(path)
    loaded = dict(np.load(path))
    missing = [k for k in STAGE_LEAVES if k not in loaded]
    if missing:
        raise KeyError(f"{path} missing required leaves: {missing}")
    out = {k: jnp.asarray(loaded[k]) for k in STAGE_LEAVES}
    # Optional leaves are carried when present and simply absent otherwise, so a
    # pre-chroma checkpoint keeps loading and a chroma (or chroma-affine/-kurt)
    # checkpoint round-trips.
    for k in ALL_OPTIONAL_LEAVES:
        if k in loaded:
            out[k] = jnp.asarray(loaded[k])
    # Legacy (pre-representation-change, tag absent) checkpoints stored
    # 58-grid raw leaves -- detect + convert (one warning), never silently
    # mis-render a sub-pixel grid as pixel-integrated.
    out = EM.convert_legacy_params_raw(out, name=str(path))
    return out


def _leaf_labels(
    stage: int, *, freeze_wcs: bool, freeze_modes: bool = False,
    freeze_w: bool = False,
    freeze_others_stage4: bool = False,
    param_keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Map param keys → optax multi_transform labels.

    Two consumers with different needs. ``optax.multi_transform`` requires the label
    tree to match the params tree EXACTLY, so ``make_stage_optimizer`` always passes
    the real keys. ``stop_grad_frozen_params`` only looks labels up by key, so a
    superset is harmless there -- and a superset is what it must get, or an optional
    leaf that should be frozen in stages 1-2 would silently stay trainable. Hence the
    default is the superset, not the required set.
    """
    keys = tuple(ALL_LEAVES if param_keys is None else param_keys)
    labels = {k: "frozen" for k in keys}
    if stage >= 1 and not freeze_wcs:
        labels["wcs_coeff"] = "train_wcs"
    if stage >= 2:
        labels["epsf_base_raw"] = "train_epsf"
    if stage >= 3 and not freeze_modes:
        # ``freeze_modes``: default off. Added 2026-09-07 for a K=2 warm start
        # (iso_defocus + kurt) whose SHAPE comes from a validated, gauged
        # finite-difference init and is not meant to keep training -- only its
        # per-frame amplitude (``w_coeff``, its own bucket regardless of this
        # flag) should move. Mirrors ``freeze_wcs`` exactly: same "frozen"
        # label + ``optax.set_to_zero()`` bucket, same stop_grad_frozen_params
        # path, no new machinery.
        labels["epsf_modes"] = "train_epsf"
    if stage >= 3 and not freeze_w:
        # ``freeze_w``: default off. Set by ``--profile-w`` (task PW), which
        # solves the per-frame mode amplitudes in closed form inside the flux
        # solve instead of reading them off this spline. Under PW the loss
        # never calls ``w_field_from_coeff`` at all (``loss.total_loss``
        # zeroes ``w_of_t``), so ``w_coeff`` has an identically-zero gradient
        # and training it would be a pure random walk on an unused leaf.
        #
        # ``w_coeff`` gets its OWN bucket, for the same reason the chromatic leaves
        # do: it is in different units from the rest of the ePSF group. Its natural
        # scale is ~5e-5 against ``epsf_base_raw``'s ~14, so one Adam step at the
        # shared rate moved it by 5.2x its own value and it could never converge --
        # it random-walked at the step scale and left a spurious ~1.4% global width
        # ramp across the orbit (docs/TEMPORAL_RESIDUAL_ROOT_CAUSE_20260906.md).
        #
        # Splitting the bucket also fixes the gradient clip, which matters MORE.
        # ``optax.multi_transform`` masks each bucket, so ``clip_by_global_norm``
        # is evaluated per bucket. With w_coeff inside ``train_epsf`` its gradient
        # norm (~1.1e4) was 100.0000% of that bucket's norm, so the clip rescaled
        # epsf_base_raw and epsf_modes by ~9e-5 and pinned them at Adam's epsilon
        # (modes 1.2x eps, base 19x eps). Seven numbers were throttling 26,912.
        labels["w_coeff"] = "train_w"
        # Chroma unfreezes at stage 3 only. Training it earlier, while the WCS and
        # the base are still moving, would let it absorb WCS residuals: the colour
        # term is strongly correlated with the WCS, and only the star-to-star colour
        # spread separates them. Same bucket for the C1 colour-affine/kurt leaves --
        # they are unfrozen at stage 3 exactly when the shift/dilation pair is.
        for k in ALL_OPTIONAL_LEAVES:
            if k in labels and k not in CHROMA_IMAGE_LEAVES:
                labels[k] = "train_chroma"
    if stage >= 4:
        # The free colour image is 3364 parameters that can mimic a width error, a
        # flux error or a base defect. It is unfrozen only here, a whole stage after
        # the parametric colour terms, so that what it ends up holding is what those
        # 200-odd parametric numbers could NOT represent -- which is the entire
        # question it exists to answer.
        for k in CHROMA_IMAGE_LEAVES:
            if k in labels:
                labels[k] = "train_chroma_image"
        if freeze_others_stage4:
            for k in ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff",
                      *OPTIONAL_LEAVES, *CHROMA_AFFINE_LEAVES, *CHROMA_KURT_LEAVES,
                      *CHROMA_HALO_LEAVES):
                if k in labels:
                    labels[k] = "frozen"
    return labels


def stop_grad_frozen_params(
    params: dict[str, jnp.ndarray],
    labels: dict[str, str],
) -> dict[str, jnp.ndarray]:
    """Apply ``stop_gradient`` to leaves labeled frozen so AD skips them."""
    return {
        k: (jax.lax.stop_gradient(v) if labels.get(k) == "frozen" else v)
        for k, v in params.items()
    }


def make_stage_optimizer(
    stage: int,
    lr: float,
    *,
    epsf_lr_scale: float = 0.4,
    freeze_wcs: bool = False,
    freeze_modes: bool = False,
    freeze_w: bool = False,
    grad_clip: float = 1.0,
    param_keys: Iterable[str] | None = None,
    chroma_lr_scale: float = 1.0,
    w_lr_scale: float = W_LR_SCALE_DEFAULT,
    chroma_image_lr_scale: float = CHROMA_IMAGE_LR_SCALE_DEFAULT,
    freeze_others_stage4: bool = False,
) -> optax.GradientTransformation:
    keys = tuple(param_keys) if param_keys is not None else STAGE_LEAVES
    labels = _leaf_labels(
        stage, freeze_wcs=freeze_wcs, freeze_modes=freeze_modes,
        freeze_w=freeze_w, freeze_others_stage4=freeze_others_stage4,
        param_keys=keys,
    )
    clip = optax.clip_by_global_norm(grad_clip)
    transforms = {
        "train_wcs": optax.chain(clip, optax.adam(lr)),
        "train_epsf": optax.chain(clip, optax.adam(lr * epsf_lr_scale)),
        # The chromatic leaves are twelve numbers against ~1e9 pixels, and they are
        # in physical units (px, and a fractional scale) rather than the ePSF's
        # flux-fraction units, so they get their own learning-rate bucket.
        "train_chroma": optax.chain(clip, optax.adam(lr * chroma_lr_scale)),
        # Seven numbers whose natural scale is ~1e-5 to 1e-6, i.e. nine orders below
        # epsf_base_raw. Sharing a learning rate with it is what stopped w(t) from
        # ever converging; see ``_leaf_labels``. Each bucket is clipped separately
        # because multi_transform masks the tree, so this also stops w_coeff's
        # enormous gradient from setting the clip factor for the ePSF leaves.
        "train_w": optax.chain(clip, optax.adam(lr * w_lr_scale)),
        # The free colour image: its own bucket, own clip, own rate. See
        # CHROMA_IMAGE_LR_SCALE_DEFAULT for the measurement that forced this.
        "train_chroma_image": optax.chain(clip, optax.adam(lr * chroma_image_lr_scale)),
        "frozen": optax.set_to_zero(),
    }
    return optax.multi_transform(transforms, labels)


@dataclass
class FitData:
    ctx: StaticContext
    data: jnp.ndarray
    noise: jnp.ndarray
    weight: jnp.ndarray
    wcs_second_diff: jnp.ndarray
    w_second_diff: jnp.ndarray
    epsf_modes_init: jnp.ndarray
    weights: LossWeights = field(default_factory=LossWeights)
    local_cache: jnp.ndarray | None = None
    use_dx_only: bool = False
    do_recenter: bool = True
    recenter_n_iter: int = EM.HOTPATH_RECENTER_N_ITER
    n_pix: int | None = None
    # Minibatch fractions (1.0 = full batch). Resampled each step when < 1.
    group_frac: float = 1.0
    frame_frac: float = 1.0
    rng_seed: int = 0
    # Sticky TNS/asteroid keep-mask (G, T); MAD reject ANDs with this.
    mask_stamp_active: np.ndarray | None = None
    # Frames per lax.scan+jax.checkpoint block in loss.total_loss's data-term
    # (None/0 = unchunked). See loss._chunked_data_term.
    stamp_chunk: int | None = None
    # >0 splits the frame axis into blocks and accumulates gradients across
    # them (see make_accum_step_fn), bounding peak memory by block size rather
    # than by total frame count. 0/None keeps the single-shot step.
    frame_block: int | None = None
    # Optional global group id per local row (bucket-local index -> caller's
    # own numbering, e.g. train_loop's `bidx`). Purely cosmetic: only used to
    # label rows in the level-1 audit table (stamp_reject.level1_group_mismatch_table)
    # written by run_stage; None falls back to the local row index.
    group_ids: np.ndarray | None = None


def _reg_only_loss_fn(fds: list[FitData], labels: dict[str, str]):
    """The parameter-only half of the loss: everything except the data term.

    ``_combined_loss_fn`` takes these once from bucket 0 (they don't depend on
    which stars are in a bucket). Frame-block accumulation needs them
    separately, added once after the per-block data-term gradients, so they are
    not multiplied by the number of blocks.
    """
    fd0 = fds[0]

    def reg_fn(params):
        params_sg = stop_grad_frozen_params(params, labels)
        w = fd0.weights
        smooth_wcs = L.spline_smoothness_penalty(params_sg["wcs_coeff"], fd0.wcs_second_diff)
        smooth_w = L.spline_smoothness_penalty(params_sg["w_coeff"], fd0.w_second_diff)
        flux_neutral = L.flux_neutral_penalty(params_sg)
        base = L.decoded_epsf_base(params_sg)
        lap = L.node_smoothness_penalty(base[None])
        fine_nbr = L.fine_nbr_penalty(base[None], w.fine_nbr_mode)
        pixel_lap = L.pixel_laplacian_penalty(base)
        mode_prior = L.mode_prior_penalty(params_sg, fd0.epsf_modes_init)
        # Mirrors loss.total_loss exactly. A penalty added to only one of these two
        # is silently dropped whenever --frame-block > 0 (see CHROMATIC_TERM.md S3).
        chroma_lap = L.chroma_image_penalty(params_sg)
        reg = (
            w.lambda_smooth_wcs * smooth_wcs
            + w.lambda_smooth_w * smooth_w
            + w.lambda_flux * flux_neutral
            + w.lambda_lap * lap
            + w.lambda_fine_nbr * fine_nbr
            + w.lambda_pixel * pixel_lap
            + w.lambda_mode_prior * mode_prior
            + w.lambda_chroma_lap * chroma_lap
        )
        # Matches total_loss: the hard core-centroid gauge already pins the
        # origin, so the penalty is reported but kept out of the gradient
        # unless centroid_in_grad is set.
        w_of_t = L.whole_orbit_w_of_t(params_sg, fd0.ctx)
        centroid = L.centroid_penalty(params_sg, w_of_t)
        if w.centroid_in_grad:
            reg = reg + w.lambda_centroid * centroid
        else:
            centroid = jax.lax.stop_gradient(centroid)
        metrics = {
            "smooth_wcs": smooth_wcs, "smooth_w": smooth_w,
            "flux_neutral": flux_neutral, "lap": lap, "fine_nbr": fine_nbr,
            "pixel_lap": pixel_lap, "mode_prior": mode_prior,
            "chroma_lap": chroma_lap,
            "centroid": centroid,
            "mode_base_overlap": L.mode_base_overlap_metric(
                L.decoded_epsf_modes(params_sg), base,
            ),
            # Metric only -- decoded_epsf_base already enforces the flux rule
            # (EM.enforce_phase_flux_rule); logged as a sanity check on the
            # hard gauge (CONTRACT_pixel_integrated_epsf.md).
            "phase_flux_rms": EM.phase_flux_rms(base),
        }
        return reg, metrics

    return reg_fn


def _block_num_den_fn(fds: list[FitData], labels: dict[str, str], local_caches, n_pixes, block_size: int):
    """Unnormalized data-term numerator/denominator for one frame block.

    The pooled data term is ``sum_b sum_buckets num / sum_b sum_buckets den``.
    ``den`` (the stamp-weight sum) depends only on ``stamp_snr_weight``,
    ``stamp_active`` and the pixel weights -- never on ``params`` -- so the
    gradient of the pooled mean is ``(sum_b grad num_b) / den_total``. That is
    what makes block accumulation *exact* rather than an approximation.

    ``block_size`` is a static Python int, closed over rather than passed as a
    traced value; ``lo`` (this block's start) IS traced, via
    ``jax.lax.dynamic_slice_in_dim``. Padding the frame axis to a multiple of
    ``block_size`` before calling (see ``make_accum_step_fn``) means every
    block, including the last, has this same static size -- so the caller's
    jit compiles exactly once regardless of block count. An earlier version
    passed ``(lo, hi)`` as ``static_argnums`` with Python ``[lo:hi]`` slicing:
    correct math, but JAX's jit cache keys on static arg *values*, so every
    distinct block position triggered its own full XLA compile -- ~30 of them
    at 1864 frames/block 64, which exhausted the CPU backend's LLVM codegen
    memory (``LLVM ERROR: Unable to allocate section memory!``) despite
    per-block runtime memory being small and correct.
    """

    def num_den_fn(params, stamp_actives, stamp_arrays, frame_basis_padded, w_of_t_full, lo):
        params_sg = stop_grad_frozen_params(params, labels)
        num_total = 0.0
        den_total = 0.0
        den_bucket0 = 0.0
        for i, (fd, stamp_active, arrays, fb_p, local_cache, n_pix) in enumerate(zip(
            fds, stamp_actives, stamp_arrays, frame_basis_padded, local_caches, n_pixes,
        )):
            data, noise, weight = arrays
            wcs_fb_p, w_fb_p = fb_p
            ctx_blk = L.slice_static_context_frames_dynamic(
                fd.ctx, wcs_fb_p, w_fb_p, stamp_active, lo, block_size,
            )
            data_blk = jax.lax.dynamic_slice_in_dim(data, lo, block_size, axis=1)
            noise_blk = jax.lax.dynamic_slice_in_dim(noise, lo, block_size, axis=1)
            weight_blk = jax.lax.dynamic_slice_in_dim(weight, lo, block_size, axis=1)
            sa_blk = jax.lax.dynamic_slice_in_dim(stamp_active, lo, block_size, axis=1)
            w_of_t_blk = jax.lax.dynamic_slice_in_dim(w_of_t_full, lo, block_size, axis=0)
            _loss_i, metrics_i = L.total_loss(
                params_sg, ctx_blk,
                data_blk, noise_blk, weight_blk,
                fd.wcs_second_diff, fd.w_second_diff,
                epsf_modes_init=fd.epsf_modes_init,
                weights=fd.weights,
                stamp_active=sa_blk,
                local_cache=local_cache,
                n_pix=n_pix,
                do_recenter=fd.do_recenter and local_cache is None,
                recenter_n_iter=fd.recenter_n_iter,
                stamp_chunk=None,  # already at block scale; no nested chunking
                w_of_t_override=w_of_t_blk,
            )
            den_i = metrics_i["stamp_weight_sum"]
            num_total = num_total + metrics_i["data_term"] * den_i
            den_total = den_total + den_i
            if i == 0:
                den_bucket0 = den_bucket0 + den_i
        # `stamp_weight_sum` is reported bucket-0-only to match
        # _combined_loss_fn's convention (it forwards metrics_list[0]); the
        # pooled `den_total` used for normalization is the all-bucket sum.
        return num_total, (den_total, den_bucket0)

    return num_den_fn


def _combined_loss_fn(fds: list[FitData], labels: dict[str, str]):
    """Build a loss function over one or more bucketed contexts.

    ``stamp_actives`` and ``stamp_arrays`` are tuples with one entry per
    bucket.  Keeping data/noise/weight as explicit JIT arguments prevents XLA
    from baking the large stamp tensors into the executable and constant-
    folding the frame-padding/blocking operations at compile time.  The arrays
    are already device arrays in ``FitData``, so repeated calls do not imply a
    host-to-device copy.

    Combines via the *exact* pooled-weighted-mean identity (see
    ``loss.total_loss``'s ``stamp_weight_sum`` and
    ``test_bucketed_total_loss_recombination_matches_unbucketed``): the
    parameter-only regularization terms (smoothness/flux-neutral/mode-prior/
    centroid) don't depend on which stars are in a bucket, so they're taken
    once from bucket 0, not summed across buckets (summing would multiply
    them by len(fds)). For a single bucket this reduces exactly to the
    original single-context loss (W_0*dt_0/W_0 + (loss_0-dt_0) == loss_0).
    """
    local_caches = [fd.local_cache if fd.use_dx_only else None for fd in fds]
    n_pixes = [
        fd.n_pix if fd.n_pix is not None
        else (1 if fd.ctx.is_packed else int(fd.data.shape[-1]))
        for fd in fds
    ]

    def loss_fn(params, stamp_actives, stamp_arrays):
        params_sg = stop_grad_frozen_params(params, labels)
        losses = []
        metrics_list = []
        for fd, stamp_active, arrays, local_cache, n_pix in zip(
            fds, stamp_actives, stamp_arrays, local_caches, n_pixes,
        ):
            data, noise, weight = arrays
            loss_i, metrics_i = total_loss(
                params_sg, fd.ctx, data, noise, weight,
                fd.wcs_second_diff, fd.w_second_diff,
                epsf_modes_init=fd.epsf_modes_init,
                weights=fd.weights,
                stamp_active=stamp_active,
                local_cache=local_cache,
                n_pix=n_pix,
                do_recenter=fd.do_recenter and local_cache is None,
                recenter_n_iter=fd.recenter_n_iter,
                stamp_chunk=fd.stamp_chunk,
            )
            losses.append(loss_i)
            metrics_list.append(metrics_i)

        w = jnp.stack([m["stamp_weight_sum"] for m in metrics_list])
        dt = jnp.stack([m["data_term"] for m in metrics_list])
        combined_data_term = jnp.sum(w * dt) / jnp.clip(jnp.sum(w), 1e-12, None)
        # The fine-neighbour prior (fine_nbr_sigma mode) is normalised by the POOLED
        # denominator, not bucket 0's: swap bucket 0's prior term for sum/W_total.
        m0 = metrics_list[0]
        w_total = jnp.clip(jnp.sum(w), 1e-12, None)
        prior_term = m0["fine_nbr_prior_sum"] / w_total
        reg_terms = losses[0] - m0["data_term"] - m0["fine_nbr_prior_term"] + prior_term
        loss = combined_data_term + reg_terms
        metrics = {**m0, "data_term": combined_data_term, "loss": loss,
                   "fine_nbr_prior_term": prior_term}
        if fds[0].weights.fine_nbr_sigma > 0.0:
            metrics["lambda_fine_nbr_eff"] = m0["lambda_fine_nbr_eff"] * m0["stamp_weight_sum"] / w_total
        return loss, metrics

    return loss_fn


def make_step_fn(
    fd: FitData | list[FitData],
    tx: optax.GradientTransformation,
    *,
    stage: int,
    freeze_wcs: bool = False,
    freeze_others_stage4: bool = False,
):
    fds = [fd] if isinstance(fd, FitData) else list(fd)
    # Task PW: under --profile-w the amplitude spline is not part of the model,
    # so it must also be stop_gradient'ed here (it already has an identically
    # zero gradient; labelling it frozen keeps the two paths in agreement).
    labels = _leaf_labels(
        stage, freeze_wcs=freeze_wcs, freeze_w=bool(fds[0].weights.profile_w),
        freeze_others_stage4=freeze_others_stage4,
    )
    group_frac = float(fds[0].group_frac)
    frame_frac = float(fds[0].frame_frac)
    use_minibatch = group_frac < 1.0 - 1e-12 or frame_frac < 1.0 - 1e-12
    if use_minibatch:
        raise NotImplementedError(
            "--group-frac/--frame-frac minibatching is not implemented: the old "
            "single-bucket path sampled indices but discarded them, silently running "
            "the full batch. Use group_frac=frame_frac=1.0."
        )

    combined_loss_fn = _combined_loss_fn(fds, labels)

    @jax.jit
    def step(params, opt_state, stamp_actives, stamp_arrays):
        (loss, metrics), grads = jax.value_and_grad(combined_loss_fn, has_aux=True)(
            params, stamp_actives, stamp_arrays,
        )
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        grad_rms = {
            f"grad_rms_{k}": jnp.sqrt(jnp.mean(jnp.square(grads[k])))
            for k in STAGE_LEAVES if k in grads
        }
        metrics = {**metrics, **grad_rms}
        return new_params, new_opt_state, metrics

    return step, use_minibatch


def make_accum_step_fn(
    fd: FitData | list[FitData],
    tx: optax.GradientTransformation,
    *,
    stage: int,
    freeze_wcs: bool = False,
    freeze_others_stage4: bool = False,
    frame_block: int,
):
    """Frame-block gradient-accumulation step (memory-bounded equivalent of ``make_step_fn``).

    ``make_step_fn`` puts every K/P bucket and every frame into a single
    ``jax.value_and_grad``, so peak memory grows linearly with the frame count
    (measured on this bundle: ~0.17 GB/frame, i.e. ~320 GB at 1864 frames).
    ``--stamp-chunk`` does not bound it: it shrinks the per-block activations
    inside the scan, but reverse-mode still keeps per-iteration residuals whose
    total scales with T.

    Here the frame axis is split into blocks of ``frame_block``; each block's
    gradient is computed and accumulated in a Python loop, so a block's buffers
    are freed before the next is traced and peak memory scales with
    ``frame_block`` instead of T.

    Exactness: the pooled data term is ``N/D`` with ``D`` parameter-independent
    (see ``_block_num_den_fn``), so ``grad(N/D) = (sum_b grad N_b)/D``.
    Parameter-only regularization is added once at the end, and the ``w_of_t``
    zero-time-mean gauge is computed on the *full* frame set and sliced per
    block, so the model is identical to the unchunked path -- only the
    summation order differs (fp32 roundoff).
    """
    fds = [fd] if isinstance(fd, FitData) else list(fd)
    labels = _leaf_labels(
        stage, freeze_wcs=freeze_wcs, freeze_w=bool(fds[0].weights.profile_w),
        freeze_others_stage4=freeze_others_stage4,
    )
    if fds[0].group_frac < 1.0 - 1e-12 or fds[0].frame_frac < 1.0 - 1e-12:
        raise NotImplementedError("frame-block accumulation requires group_frac=frame_frac=1.0")

    local_caches = [fd_.local_cache if fd_.use_dx_only else None for fd_ in fds]
    n_pixes = [
        fd_.n_pix if fd_.n_pix is not None
        else (1 if fd_.ctx.is_packed else int(fd_.data.shape[-1]))
        for fd_ in fds
    ]

    n_frames = int(fds[0].ctx.wcs_frame_basis.shape[0])
    block = max(1, min(int(frame_block), n_frames))
    n_blocks = -(-n_frames // block)  # ceil division
    pad_len = n_blocks * block - n_frames
    block_starts = [i * block for i in range(n_blocks)]  # Python ints; lo is traced below

    num_den_fn = _block_num_den_fn(fds, labels, local_caches, n_pixes, block)
    reg_fn = _reg_only_loss_fn(fds, labels)

    # Frame-basis arrays are static per fd (not step inputs), so pad once here.
    wcs_fb_padded = [L._pad_frame_axis(fd_.ctx.wcs_frame_basis, 0, pad_len) for fd_ in fds]
    w_fb_padded = [L._pad_frame_axis(fd_.ctx.w_frame_basis, 0, pad_len) for fd_ in fds]
    frame_basis_padded = tuple(zip(wcs_fb_padded, w_fb_padded))
    real_mask = (
        jnp.concatenate([jnp.ones((n_frames,), jnp.float32), jnp.zeros((pad_len,), jnp.float32)])
        if pad_len > 0 else None
    )

    def _pad_frames(stamp_actives, stamp_arrays):
        """Zero-pad the frame axis of the per-step data to a multiple of block.

        stamp_active padding is forced to 0 in the pad region (belt-and-
        suspenders alongside the zero data/weight padding) so padded frames
        contribute exactly 0 to num/den -- same convention as
        loss._chunked_data_term.
        """
        if pad_len == 0:
            return stamp_actives, stamp_arrays
        sa_p = tuple(L._pad_frame_axis(sa, 1, pad_len) * real_mask[None, :] for sa in stamp_actives)
        arr_p = tuple(
            (L._pad_frame_axis(d, 1, pad_len), L._pad_frame_axis(n, 1, pad_len), L._pad_frame_axis(w, 1, pad_len))
            for (d, n, w) in stamp_arrays
        )
        return sa_p, arr_p

    # block_size is closed over (static); lo is a regular traced argument, so
    # this compiles once total, not once per block position (see
    # _block_num_den_fn's docstring for why that distinction matters).
    @jax.jit
    def block_grad(params, stamp_actives, stamp_arrays, w_of_t_full, lo):
        (num_b, (den_b, den0_b)), g = jax.value_and_grad(num_den_fn, has_aux=True)(
            params, stamp_actives, stamp_arrays, frame_basis_padded, w_of_t_full, lo,
        )
        return num_b, den_b, den0_b, g

    @jax.jit
    def reg_grad(params):
        (reg, reg_metrics), g = jax.value_and_grad(reg_fn, has_aux=True)(params)
        return reg, reg_metrics, g

    sigma_prior = float(fds[0].weights.fine_nbr_sigma)

    @jax.jit
    def prior_grad(params):
        # Normalised by the pooled den_total below, like the data term.
        return jax.value_and_grad(
            lambda p: L.fine_nbr_prior_sum(stop_grad_frozen_params(p, labels), sigma_prior,
                                           fds[0].weights.fine_nbr_mode)
        )(params)

    @jax.jit
    def apply_update(params, opt_state, grads):
        updates, new_opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state

    def step(params, opt_state, stamp_actives, stamp_arrays):
        # Whole-orbit gauge, computed once on the unsliced (unpadded) context
        # -- the gauge mean MUST be taken over the real frames only.
        w_of_t_full = L.whole_orbit_w_of_t(stop_grad_frozen_params(params, labels), fds[0].ctx)
        if pad_len > 0:
            w_of_t_full = L._pad_frame_axis(w_of_t_full, 0, pad_len)
        stamp_actives_p, stamp_arrays_p = _pad_frames(stamp_actives, stamp_arrays)

        num_total = jnp.zeros((), dtype=jnp.float32)
        den_total = jnp.zeros((), dtype=jnp.float32)
        den0_total = jnp.zeros((), dtype=jnp.float32)
        grads_num = None
        for lo in block_starts:
            num_b, den_b, den0_b, g_b = block_grad(
                params, stamp_actives_p, stamp_arrays_p, w_of_t_full, lo,
            )
            num_total = num_total + num_b
            den_total = den_total + den_b
            den0_total = den0_total + den0_b
            grads_num = g_b if grads_num is None else jax.tree.map(jnp.add, grads_num, g_b)
            del g_b  # release this block's buffers before tracing the next

        inv_den = 1.0 / jnp.clip(den_total, 1e-12, None)
        data_term = num_total * inv_den
        grads = jax.tree.map(lambda a: a * inv_den, grads_num)

        reg, reg_metrics, reg_g = reg_grad(params)
        grads = jax.tree.map(jnp.add, grads, reg_g)
        prior_term = jnp.zeros((), dtype=jnp.float32)
        lambda_eff = jnp.asarray(fds[0].weights.lambda_fine_nbr, dtype=jnp.float32)
        if sigma_prior > 0.0:
            prior_sum, prior_g = prior_grad(params)
            grads = jax.tree.map(lambda a, b: a + b * inv_den, grads, prior_g)
            prior_term = prior_sum * inv_den
            base_shape = params["epsf_base_raw"].shape
            lambda_eff = (L.fine_neighbour_count((1,) + tuple(base_shape))
                          / (2.0 * sigma_prior ** 2)) * inv_den

        new_params, new_opt_state = apply_update(params, opt_state, grads)
        lambda_centroid = fds[0].weights.lambda_centroid
        metrics = {
            **reg_metrics,
            "data_term": data_term,
            "stamp_weight_sum": den0_total,
            "loss": data_term + reg + prior_term,
            "fine_nbr_prior_term": prior_term,
            "lambda_fine_nbr_eff": lambda_eff,
            "ratio_centroid": (lambda_centroid * reg_metrics["centroid"])
            / jnp.clip(data_term, 1e-12, None),
            **{
                f"grad_rms_{k}": jnp.sqrt(jnp.mean(jnp.square(grads[k])))
                for k in STAGE_LEAVES if k in grads
            },
        }
        return new_params, new_opt_state, metrics

    return step, False


def _append_history_jsonl(path: Path | None, record: dict) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def run_stage(
    params: dict[str, jnp.ndarray],
    fd: FitData | list[FitData],
    *,
    stage: int,
    n_steps: int,
    lr: float,
    log_every: int = 20,
    epsf_lr_scale: float = 0.4,
    # Stage 4 only: hold the WCS, the ePSF base and the parametric colour terms
    # fixed while the free colour image trains. Run both ways -- frozen says what
    # the parametric family could not represent, unfrozen says whether the free
    # image merely repaints the base.
    freeze_others_stage4: bool = False,
    chroma_image_lr_scale: float = CHROMA_IMAGE_LR_SCALE_DEFAULT,
    chroma_lr_scale: float = 1.0,
    w_lr_scale: float = W_LR_SCALE_DEFAULT,
    freeze_wcs_steps: int = 0,
    freeze_modes: bool = False,
    freeze_w: bool = False,
    grad_clip: float = 1.0,
    history_path: Path | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int | None = None,
    reject_every: int = 20,
    reject_burn_in: int = 50,
    reject_n_sigma: float = 3.0,
    reject_mode: str = "hysteresis",
    reject_tau_drop: float = 3.5,
    reject_tau_keep: float = 2.2,
    reject_max_churn: float = 0.005,
    l2_churn_cap_frac: float | None = None,
    l2_hysteresis_n: int | None = None,
    l2_freeze_scale_after_first: bool = True,
    l2_scale_ema_alpha: float = 1.0,
    standardized_shrinkage_frames: float = 20.0,
    standardized_min_weight: float = 0.05,
    early_stop_patience: int | None = None,
    early_stop_tol: float = 1e-5,
    early_stop_tol_coarse: float = 1e-4,
    resume_phase: str | None = None,
    resume_next_step: int = 0,
    resume_opt_leaves: list[np.ndarray] | None = None,
    state_callback=None,
    resume_gate_state=None,
    stop_file: Path | None = None,
    advance_file: Path | None = None,
) -> tuple[dict[str, jnp.ndarray], list[dict]]:
    """``fd`` may be one ``FitData`` (whole region, one shared K) or a list --
    one per K-bucket tier (see ``groups.bucket_groups_by_size``). A list of
    one behaves identically to passing that one ``FitData`` directly (the
    bucketed loss combination is an exact identity in that case).

    ``reject_mode`` selects residual policy. ``audit`` records scores and the
    level-1 table but applies only physical masks. ``standardized`` uses
    hierarchically per-group-standardized continuous weights. ``legacy`` and
    ``two-level`` preserve the historical hard gates for reproduction.
    """
    if reject_mode not in ("hysteresis", "audit", "standardized", "legacy", "two-level", "static"):
        raise ValueError(
            "reject_mode must be hysteresis, audit, standardized, legacy, two-level "
            f"or static, got {reject_mode!r}"
        )
    fds = [fd] if isinstance(fd, FitData) else list(fd)

    # Stage 1 is intentionally a stationary Gate-C-only WCS fit.  In
    # particular, a legacy CLI reject setting must not alter this invariant.
    if stage == 1:
        for fd_i in fds:
            baseline = fd_i.mask_stamp_active
            if baseline is not None:
                fd_i.ctx = L.with_stamp_active(fd_i.ctx, np.asarray(baseline, dtype=np.float32))

    history: list[dict] = []
    ckpt_every = int(checkpoint_every) if checkpoint_every is not None else int(log_every)
    rej_every = int(reject_every)
    last_reject_stats: dict[str, float] = {}
    SR = None
    l2_state = None
    standardized_state = None
    static_brightness: list[np.ndarray] | None = None
    l2_churn_cap_frac_eff = l2_churn_cap_frac
    l2_hysteresis_n_eff = l2_hysteresis_n
    hysteresis_scales: list[np.ndarray] | None = None
    mask_frozen = stage == 1
    phase_losses: list[float] = []
    n_frames_fit = int(np.asarray(fds[0].ctx.stamp_active).shape[1]) if fds else 0
    if rej_every > 0 and n_frames_fit < 2 and reject_mode not in ("audit", "static"):
        # Every frame-axis mode centres log-chi2 per group over frames, so with one
        # frame the deviation is identically zero: it would reject nothing while
        # logging as though it had run. Refuse instead of no-op'ing.
        raise ValueError(
            f"reject_mode={reject_mode!r} centres residual scores per group ACROSS "
            f"frames, but this fit has {n_frames_fit} frame(s), so it can never "
            "reject anything. Use --reject-mode static (brightness-detrended "
            "whole-star gate) or audit (physical masks only) for a single-FFI fit."
        )
    if stage >= 2 and rej_every > 0:
        from . import stamp_reject as SR  # noqa: PLC0415  (heavy; optional for Colab bench)
        if l2_churn_cap_frac_eff is None:
            l2_churn_cap_frac_eff = SR.DEFAULT_L2_CHURN_CAP_FRAC
        if l2_hysteresis_n_eff is None:
            l2_hysteresis_n_eff = SR.DEFAULT_L2_HYSTERESIS_N
        if reject_mode == "two-level":
            # One state per run_stage call (== one stage): fresh streaks/scale
            # at each stage boundary, reused across every refresh within it --
            # see Level2GateState's docstring.
            l2_state = SR.Level2GateState()
            if resume_gate_state is not None:
                l2_state = resume_gate_state
            _log(
                f"stage {stage}: reject_mode=two-level "
                f"(n_sigma={reject_n_sigma}, churn_cap_frac={l2_churn_cap_frac_eff}, "
                f"hysteresis_n={l2_hysteresis_n_eff}, "
                f"freeze_scale_after_first={l2_freeze_scale_after_first}, "
                f"scale_ema_alpha={l2_scale_ema_alpha})"
            )
        elif reject_mode == "standardized":
            standardized_state = SR.StandardizedGateState()
            _log(
                f"stage {stage}: reject_mode=standardized "
                f"(n_sigma={reject_n_sigma}, hysteresis_n={l2_hysteresis_n_eff}, "
                f"shrinkage_frames={standardized_shrinkage_frames}, "
                f"min_weight={standardized_min_weight})"
            )
        elif reject_mode == "audit":
            _log(
                f"stage {stage}: reject_mode=audit "
                "(physical masks only; residual scores report-only)"
            )
        elif reject_mode == "static":
            static_brightness = [SR.static_brightness_proxy(fd_i) for fd_i in fds]
            _log(
                f"stage {stage}: reject_mode=static (tau={reject_tau_drop}, "
                f"window={SR.DEFAULT_STATIC_WINDOW}, "
                f"{sum(int(b.size) for b in static_brightness)} groups ranked by "
                "observed stamp flux)"
            )

    if stage >= 2 and rej_every > 0 and reject_mode == "hysteresis":
        # Calculate once at stage start over the active (and physical-baseline)
        # stamps.  Per-group MAD is intentionally never recomputed on refresh.
        hysteresis_scales = []
        for fd_i in fds:
            chi2, _pix = SR.per_stamp_chi2_red(params, fd_i)
            log_chi2 = np.log(np.maximum(np.asarray(chi2, dtype=np.float64), 1e-6))
            base = (np.ones_like(log_chi2, dtype=bool) if fd_i.mask_stamp_active is None
                    else np.asarray(fd_i.mask_stamp_active, dtype=bool))
            active = np.asarray(fd_i.ctx.stamp_active, dtype=bool) & base
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(np.where(active, log_chi2, np.nan), axis=1, keepdims=True)
                scale = np.nanmedian(np.abs(np.where(active, log_chi2, np.nan) - med), axis=1) * SR.MAD_SCALE
            hysteresis_scales.append(np.where(np.isfinite(scale), scale, 1e-4))
        _log(f"stage {stage}: hysteresis rejection enabled (burn_in={reject_burn_in}, "
             f"tau_drop={reject_tau_drop}, tau_keep={reject_tau_keep}, max_churn={reject_max_churn})")

    # Stage-1 dx-only: build each bucket's local cache once (frozen ePSF).
    for bi, fd_i in enumerate(fds):
        if stage == 1 and fd_i.use_dx_only and fd_i.local_cache is None:
            _log(f"stage 1: building dx-only local ePSF cache (blend+recenter once), bucket {bi}/{len(fds)}…")
            t_cache = time.time()
            for _ck in ALL_OPTIONAL_LEAVES:
                _cv = params.get(_ck)
                if _cv is not None and float(np.abs(np.asarray(_cv)).max()) > 0.0:
                    raise ValueError(
                        f"stage 1's dx-only local cache cannot represent {_ck}, and it "
                        "is non-zero. The cache is pre-blended and pre-recentered, so "
                        "the chromatic dilation would be silently dropped. Re-run "
                        "stage 1 with --no-dx-only, or start from stage 2."
                    )
            fd_i.local_cache = L.build_dx_only_local_cache(params, fd_i.ctx)
            _log(f"  local_cache shape={tuple(np.asarray(fd_i.local_cache).shape)} in {time.time()-t_cache:.2f}s")

    def _refresh_reject(step_i: int, *, freeze_wcs: bool) -> None:
        nonlocal last_reject_stats
        if rej_every <= 0 or stage == 1 or mask_frozen:
            return
        if reject_mode == "hysteresis":
            total = {"n_dropped": 0, "n_recovered": 0, "net_churn": 0, "n_active": 0, "n_total": 0}
            # med_chi2_red: same convention as the `audit`/`legacy` branches below
            # (median chi2_red over stamps with actual pixel coverage, pix_sum > 0,
            # not restricted to `stamp_active` -- a candidate stamp with no coverage
            # this step has no chi2 to contribute). Previously hardcoded to NaN here
            # (this branch is the training default, `--reject-mode hysteresis`, so
            # every logged history.jsonl row's med_chi2_red was NaN); `chi2`/`pix`
            # were already computed per bucket for the reject decision itself, so
            # this reuses them rather than adding a second per-stamp pass.
            med_chi2_values: list[np.ndarray] = []
            for fd_i, scale in zip(fds, hysteresis_scales or []):
                chi2, pix = SR.per_stamp_chi2_red(params, fd_i)
                chi2_native = np.asarray(chi2)  # unchanged from before: native dtype, for the reject decision
                pix_ok = np.asarray(pix) > 0
                if pix_ok.any():
                    med_chi2_values.append(chi2_native.astype(np.float64)[pix_ok])
                old = np.asarray(fd_i.ctx.stamp_active, dtype=bool)
                baseline = (np.ones_like(old, dtype=bool) if fd_i.mask_stamp_active is None
                            else np.asarray(fd_i.mask_stamp_active, dtype=bool))
                new, stats = SR.update_stamp_active_hysteresis(
                    chi2_native, old, baseline, scale,
                    tau_drop=reject_tau_drop, tau_keep=reject_tau_keep,
                    max_churn_frac=reject_max_churn,
                )
                fd_i.ctx = L.with_stamp_active(fd_i.ctx, new.astype(np.float32))
                for key in ("n_dropped", "n_recovered", "net_churn"):
                    total[key] += int(stats[key])
                total["n_active"] += int(new.sum())
                total["n_total"] += int(new.size)
            med_chi2_red = (
                float(np.median(np.concatenate(med_chi2_values)))
                if med_chi2_values else float("nan")
            )
            stats = {**total, "active_frac": total["n_active"] / max(total["n_total"], 1),
                     "n_rejected": total["n_total"] - total["n_active"],
                     "n_active_cand": total["n_total"], "n_active_kept": total["n_active"],
                     "frac_rejected": 1 - total["n_active"] / max(total["n_total"], 1),
                     "med_chi2_red": med_chi2_red, "n_downweighted": 0.0, "frac_downweighted": 0.0}
            last_reject_stats = stats
            _log(f"stage {stage} rejection @ step {step_i}: dropped={total['n_dropped']}, "
                 f"recovered={total['n_recovered']}, active_frac={stats['active_frac']:.3%}")
            return stats
        mask_baselines = [fd_i.mask_stamp_active for fd_i in fds]
        if reject_mode == "two-level":
            stats = SR.refresh_stamp_active_v2(
                params, fds, n_sigma=reject_n_sigma,
                mask_baselines=mask_baselines,
                state=l2_state,
                freeze_scale_after_first=l2_freeze_scale_after_first,
                scale_ema_alpha=l2_scale_ema_alpha,
                churn_cap_frac=l2_churn_cap_frac_eff,
                hysteresis_n=l2_hysteresis_n_eff,
            )
        elif reject_mode == "standardized":
            stats = SR.refresh_stamp_active_standardized(
                params, fds, n_sigma=reject_n_sigma,
                mask_baselines=mask_baselines,
                state=standardized_state,
                shrinkage_frames=standardized_shrinkage_frames,
                min_weight=standardized_min_weight,
                hysteresis_n=l2_hysteresis_n_eff,
                freeze_scales_after_first=False,
            )
        elif reject_mode == "legacy":
            stats = SR.refresh_stamp_active_multi(
                params, fds, n_sigma=reject_n_sigma,
                mask_baselines=mask_baselines,
            )
        elif reject_mode == "static":
            stats = SR.refresh_stamp_active_static(
                params, fds,
                brightness=static_brightness or [],
                tau=reject_tau_drop,
                mask_baselines=mask_baselines,
            )
        else:  # audit: score and report, but apply only the physical baseline.
            per_bucket = [SR.per_stamp_chi2_red(params, fd_i) for fd_i in fds]
            med_values = []
            n_cand = 0
            n_keep = 0
            for (chi2, pix_sum), fd_i, baseline in zip(per_bucket, fds, mask_baselines):
                chi2_np = np.asarray(chi2, dtype=np.float64)
                pix_ok = np.asarray(pix_sum) > 0
                base = (
                    np.ones_like(chi2_np, dtype=np.float32)
                    if baseline is None else np.asarray(baseline, dtype=np.float32)
                )
                applied = pix_ok.astype(np.float32) * (base > 0).astype(np.float32)
                fd_i.ctx = L.with_stamp_active(fd_i.ctx, applied)
                n_cand += int(pix_ok.sum())
                n_keep += int(applied.sum())
                if pix_ok.any():
                    med_values.append(chi2_np[pix_ok])
            stats = {
                "n_rejected": float(n_cand - n_keep),
                "n_active_cand": float(n_cand),
                "n_active_kept": float(n_keep),
                "frac_rejected": float(n_cand - n_keep) / float(max(n_cand, 1)),
                "med_chi2_red": (
                    float(np.median(np.concatenate(med_values)))
                    if med_values else float("nan")
                ),
                "n_downweighted": 0.0,
                "frac_downweighted": 0.0,
            }
        last_reject_stats = stats
        frac = float(stats["frac_rejected"])
        _log(
            f"stage {stage} reject refresh @ step {step_i} [{reject_mode}]: "
            f"rejected={int(stats['n_rejected'])}/{int(stats['n_active_cand'])} "
            f"({frac:.1%}), downweighted={int(stats.get('n_downweighted', 0))}, "
            f"med_chi2_red={stats['med_chi2_red']:.3f}"
        )
        if frac > SR.REJECT_WARN_FRAC:
            _log(
                f"  WARNING: frac_rejected={frac:.1%} > {SR.REJECT_WARN_FRAC:.0%}; "
                "continuing anyway"
            )
        rec = {
            "step": step_i,
            "stage": stage,
            "freeze_wcs": freeze_wcs,
            "event": "stamp_reject",
            "reject_mode": reject_mode,
            **{k: float(v) for k, v in stats.items()},
        }
        history.append(rec)
        _append_history_jsonl(history_path, rec)
        return stats

    def _run_block(n: int, *, freeze_wcs: bool, label: str) -> None:
        nonlocal params, history, mask_frozen
        if n <= 0:
            return
        # Soft WCS freeze in stage 2 disables dx-only (ePSF is training).
        tx = make_stage_optimizer(
            stage, lr, epsf_lr_scale=epsf_lr_scale, freeze_wcs=freeze_wcs, grad_clip=grad_clip,
            param_keys=tuple(params.keys()), chroma_lr_scale=chroma_lr_scale,
            w_lr_scale=w_lr_scale, freeze_modes=freeze_modes, freeze_w=freeze_w,
            chroma_image_lr_scale=chroma_image_lr_scale,
            freeze_others_stage4=freeze_others_stage4,
        )
        opt_state = tx.init(params)
        start_i = 0
        if resume_phase == label:
            from . import training_state as TS
            opt_state = TS.restore_optimizer(opt_state, resume_opt_leaves or [])
            start_i = int(resume_next_step)
            _log(f"stage {stage} {label}: exact resume at next_step={start_i}")
        elif resume_phase is not None:
            # A later phase checkpoint means this phase was durably completed.
            phase_order = {"soft(epsf-only)": 0, "joint": 1, "main": 0}
            if phase_order.get(label, 0) < phase_order.get(resume_phase, 0):
                return
        fb = int(getattr(fds[0], "frame_block", 0) or 0)
        if fb > 0:
            step, use_minibatch = make_accum_step_fn(
                fds, tx, stage=stage, freeze_wcs=freeze_wcs,
                freeze_others_stage4=freeze_others_stage4, frame_block=fb,
            )
        else:
            step, use_minibatch = make_step_fn(
                fds, tx, stage=stage, freeze_wcs=freeze_wcs,
                freeze_others_stage4=freeze_others_stage4,
            )
        stamp_actives = tuple(jnp.asarray(fd_i.ctx.stamp_active, dtype=jnp.float32) for fd_i in fds)
        stamp_arrays = tuple((fd_i.data, fd_i.noise, fd_i.weight) for fd_i in fds)
        k_tiers = [int(fd_i.ctx.members.shape[1]) for fd_i in fds]
        _log(
            f"stage {stage} {label}: {n} steps, lr={lr}, epsf_lr_scale={epsf_lr_scale}, "
            f"w_lr_scale={w_lr_scale}, "
            f"freeze_wcs={freeze_wcs}, reject_every={rej_every}, "
            f"dx_only={fds[0].use_dx_only and stage == 1}, "
            f"n_pix={fds[0].n_pix or fds[0].data.shape[-1]}, "
            f"minibatch(g={fds[0].group_frac:.2f},f={fds[0].frame_frac:.2f}), "
            f"k_buckets={k_tiers} "
            f"(JIT once per bucket-set shape; stamp_active is dynamic)"
        )
        t0 = time.time()
        # Early stop: track loss at each logged (log_every-spaced) checkpoint; if it
        # hasn't moved by more than early_stop_tol (relative) over the last
        # early_stop_patience checks, the stage is converged -- stop before n_steps
        # rather than always burning the full fixed budget. Uses |change|, not signed
        # change, so it also catches "stuck oscillating" as converged, not just "still
        # improving too slowly". Disabled (None) by default -- unchanged behavior.
        stop_early = False
        for i in range(start_i, n):
            if (stage >= 2 and not mask_frozen and rej_every > 0
                    and i >= reject_burn_in and i % rej_every == 0):
                refresh_stats = _refresh_reject(i, freeze_wcs=freeze_wcs)
                # Do NOT remake step — same shapes, no recompile.
                stamp_actives = tuple(jnp.asarray(fd_i.ctx.stamp_active, dtype=jnp.float32) for fd_i in fds)
                if reject_mode == "hysteresis" and refresh_stats and refresh_stats["net_churn"] == 0:
                    mask_frozen = True
                    phase_losses.clear()
                    _log(f"stage {stage} step {i}: zero mask churn; locking mask for fine polish")
            log_this_step = i % log_every == 0 or i == n - 1
            if i == 0 and start_i == 0:
                _log(f"stage {stage} step 0 starting (may JIT)…")
            elif log_this_step:
                _log(f"stage {stage} step {i:4d}/{n}  starting…")
            step_t0 = time.time()
            params, opt_state, metrics = step(
                params, opt_state, stamp_actives, stamp_arrays,
            )
            if log_this_step:
                for key in ("loss", "data_term", "centroid"):
                    if key in metrics:
                        jax.block_until_ready(metrics[key])
                step_wall_s = time.time() - step_t0
                m = {k: float(v) for k, v in metrics.items()}
                m["step"] = i
                m["elapsed_s"] = time.time() - t0
                m["stage"] = stage
                m["freeze_wcs"] = freeze_wcs
                m["step_wall_s"] = step_wall_s
                if last_reject_stats:
                    m["frac_rejected"] = float(last_reject_stats.get("frac_rejected", float("nan")))
                    m["n_rejected"] = float(last_reject_stats.get("n_rejected", float("nan")))
                    m["med_chi2_red"] = float(last_reject_stats.get("med_chi2_red", float("nan")))
                history.append(m)
                _append_history_jsonl(history_path, m)
                _log(
                    f"stage {stage} step {i:4d}/{n}  loss={m['loss']:.4f}  "
                    f"data={m['data_term']:.4f}  centroid={m['centroid']:.4f}  "
                    f"ratio_c={m.get('ratio_centroid', float('nan')):.4e}  "
                    f"({m['elapsed_s']:.1f}s, step {step_wall_s:.2f}s)"
                )
                if early_stop_patience is not None and early_stop_patience > 0:
                    phase_losses.append(m["loss"])
                    if len(phase_losses) > early_stop_patience + 1:
                        phase_losses.pop(0)
                    if len(phase_losses) == early_stop_patience + 1:
                        loss_then, loss_now = phase_losses[0], phase_losses[-1]
                        rel_change = abs(loss_then - loss_now) / max(abs(loss_then), 1e-12)
                        if (stage >= 2 and reject_mode == "hysteresis" and not mask_frozen
                                and i >= reject_burn_in and rel_change < early_stop_tol_coarse):
                            mask_frozen = True
                            phase_losses.clear()
                            _log(f"stage {stage} step {i}: coarse tolerance reached "
                                 f"({rel_change:.2e} < {early_stop_tol_coarse:.2e}); locking mask")
                        elif (stage == 1 or mask_frozen) and rel_change < early_stop_tol:
                            _log(
                                f"stage {stage} {label}: converged at step {i}/{n} "
                                f"(|Δloss|/|loss| = {rel_change:.2e} < tol={early_stop_tol:.2e} "
                                f"over the last {early_stop_patience} checks, "
                                f"loss {loss_then:.4f} -> {loss_now:.4f}); "
                                f"stopping {n - 1 - i} steps early"
                            )
                            stop_early = True
            checkpoint_due = checkpoint_path is not None and ckpt_every > 0 and (
                i % ckpt_every == 0 or i == n - 1
            )
            if checkpoint_due:
                save_params_npz(checkpoint_path, params)
                from . import checkpoint_history as CH

                CH.save_history_checkpoint(
                    checkpoint_path.parent,
                    params,
                    stage=stage,
                    step=i,
                    phase_label=label,
                    elapsed_s=time.time() - t0,
                )
                if state_callback is not None:
                    state_callback(params, opt_state, gate_state=l2_state,
                                   stage=stage, phase=label, next_step=i + 1)
            advance_requested = advance_file is not None and advance_file.is_file()
            if (stop_file is not None and stop_file.is_file()) or advance_requested:
                # Control requests are cooperative: persist exact parameters and optimizer
                # state after this completed step before returning to the caller.
                if checkpoint_path is not None and not checkpoint_due:
                    save_params_npz(checkpoint_path, params)
                    from . import checkpoint_history as CH

                    CH.save_history_checkpoint(
                        checkpoint_path.parent,
                        params,
                        stage=stage,
                        step=i,
                        phase_label=label,
                        elapsed_s=time.time() - t0,
                    )
                if state_callback is not None and not checkpoint_due:
                    state_callback(params, opt_state, gate_state=l2_state,
                                   stage=stage, phase=label, next_step=i + 1)
                stop_rec = {
                    "event": "stage_advance_requested" if advance_requested else "stop_requested",
                    "stage": stage,
                    "phase": label,
                    "step": i,
                    "next_step": i + 1,
                    "elapsed_s": time.time() - t0,
                }
                history.append(stop_rec)
                _append_history_jsonl(history_path, stop_rec)
                _log(
                    f"stage {stage} {label}: cooperative "
                    f"{'advance' if advance_requested else 'stop'} after completed step "
                    f"{i}; checkpoint flushed"
                )
                break
            if stop_early:
                break
        _log(f"stage {stage} {label} done in {time.time() - t0:.1f}s" + (" (converged early)" if stop_early else ""))

    def _write_level1_audit_csv() -> None:
        """Ranked per-group persistent-mismatch AUDIT table (never cuts
        anything -- see stamp_reject.level1_group_mismatch_table), written
        once at the end of this stage from final params. One row per group
        per bucket; ``group_id`` is ``fd.group_ids`` when the caller set it
        (train_loop populates it with each bucket's global group indices),
        else falls back to the bucket-local row index."""
        if history_path is not None:
            out_path = Path(history_path).parent / f"level1_audit_stage{stage}.csv"
        elif checkpoint_path is not None:
            out_path = Path(checkpoint_path).parent / f"level1_audit_stage{stage}.csv"
        else:
            _log(f"stage {stage}: no history_path/checkpoint_path given -- skipping level-1 audit CSV")
            return
        rows = []
        for bi, fd_i in enumerate(fds):
            chi2_red, pix_sum = SR.per_stamp_chi2_red(params, fd_i)
            chi2_np = np.asarray(chi2_red, dtype=np.float64)
            pix_np = np.asarray(pix_sum)
            pool = SR._pool_active_mask(pix_np, fd_i.mask_stamp_active)
            gid = fd_i.group_ids if fd_i.group_ids is not None else np.arange(chi2_np.shape[0])
            for row in SR.level1_group_mismatch_rows(
                chi2_np, pool_active=pool, group_ids=gid,
            ):
                rows.append({"bucket": bi, **row})
        if not rows:
            return
        rows.sort(key=lambda row: (
            not np.isfinite(row["baseline_log_chi2"]),
            -row["baseline_log_chi2"]
            if np.isfinite(row["baseline_log_chi2"]) else 0.0,
        ))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _log(f"stage {stage}: wrote level-1 audit table ({len(rows)} rows) -> {out_path}")

    if stage == 2 and freeze_wcs_steps > 0:
        n_soft = min(freeze_wcs_steps, n_steps)
        n_joint = n_steps - n_soft
        _run_block(n_soft, freeze_wcs=True, label="soft(epsf-only)")
        _run_block(n_joint, freeze_wcs=False, label="joint")
    else:
        _run_block(n_steps, freeze_wcs=False, label="main")

    if stage >= 2 and rej_every > 0 and SR is not None and n_steps > 0:
        _write_level1_audit_csv()

    return params, history


def warmstart_wcs_coeff(
    frames: list,
    gaia_full,
    cheb_static: CW.ChebWcsStatic,
    wcs_frame_basis: np.ndarray,
    *,
    star_cfg=None,
    merged_by_stem: dict | None = None,
    mad_clip_k: float | None = 3.0,
    mad_clip_max_iters: int = 5,
) -> jnp.ndarray:
    from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import load_merged_stars, select_qc_stars  # noqa: E402

    n_terms = cheb_static.n_terms
    coeff_x_track = np.zeros((len(frames), n_terms))
    coeff_y_track = np.zeros((len(frames), n_terms))
    for fi, frame in enumerate(frames):
        if merged_by_stem is not None and frame.stem in merged_by_stem:
            merged = merged_by_stem[frame.stem]
        elif merged_by_stem is not None:
            continue
        else:
            merged = load_merged_stars(frame, gaia_full)
        qc = select_qc_stars(merged, star_cfg)
        ra = qc["ra"].to_numpy(dtype=float)
        dec = qc["dec"].to_numpy(dtype=float)
        x_lin, y_lin = CW.linear_predict(jnp.asarray(ra), jnp.asarray(dec), cheb_static)
        x_lin, y_lin = np.asarray(x_lin), np.asarray(y_lin)
        x_obs = qc["x_fit"].to_numpy(dtype=float)
        y_obs = qc["y_fit"].to_numpy(dtype=float)
        cx, cy, _mask = CW.fit_frame_cheb_warmstart(x_lin, y_lin, x_obs, y_obs, cheb_static)
        coeff_x_track[fi] = cx
        coeff_y_track[fi] = cy

    basis = np.asarray(wcs_frame_basis)
    if mad_clip_k is None:
        coeff_x_spline, *_ = np.linalg.lstsq(basis, coeff_x_track, rcond=None)
        coeff_y_spline, *_ = np.linalg.lstsq(basis, coeff_y_track, rcond=None)
        coeff_matrix = np.concatenate([coeff_x_spline.T, coeff_y_spline.T], axis=0)
    else:
        from syndiff_pipeline.forward_model.init_study.mad_bspline_clip import fit_coeff_tracks_mad

        cx_mat, _, _ = fit_coeff_tracks_mad(
            basis, coeff_x_track, k=float(mad_clip_k), max_iters=int(mad_clip_max_iters)
        )
        cy_mat, _, _ = fit_coeff_tracks_mad(
            basis, coeff_y_track, k=float(mad_clip_k), max_iters=int(mad_clip_max_iters)
        )
        coeff_matrix = np.concatenate([cx_mat, cy_mat], axis=0)
    return jnp.asarray(coeff_matrix, dtype=jnp.float32)
