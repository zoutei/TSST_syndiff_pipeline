# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Adam stage loop driven from a ``FitBundle`` (no workspace I/O)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from . import epsf_model as EM
from . import fit as FIT
from . import groups as G
from . import loss as L
from . import packed_support as PS
from . import runtime as RT
from . import temporal as T
from .fit_bundle import FitBundle, params0_as_jnp


def _log(msg: str) -> None:
    RT.log(msg)


def _write_params_provenance(
    out_dir: Path,
    *,
    stage: int,
    step: int,
    reason: str,
) -> None:
    payload = {
        "artifact": "params_latest.npz",
        "stage": int(stage),
        "step": int(step),
        "reason": reason,
        "timestamp": time.time(),
    }
    path = out_dir / "params_latest_meta.json"
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_stages_from_bundle(
    bundle: FitBundle,
    *,
    out_dir: Path,
    stage: int = 3,
    start_stage: int = 1,
    steps_per_stage: list[int] | None = None,
    lr_per_stage: list[float] | None = None,
    epsf_lr_scale: float = 0.4,
    w_lr_scale: float = FIT.W_LR_SCALE_DEFAULT,
    stage2_freeze_wcs_steps: int = 20,
    freeze_modes: bool = False,
    grad_clip: float = 1.0,
    chroma: bool = False,
    chroma_lr_scale: float = 1.0,
    chroma_affine: bool = False,
    chroma_kurt: bool = False,
    chroma_halo: bool = False,
    chroma_halo_index: float | None = None,
    chroma_halo_core_px: float | None = None,
    chroma_halo_flux_neutral: bool = False,
    chroma_image: bool = False,
    chroma_image_lr_scale: float = 1e-2,
    stage4_freeze_others: bool = False,
    w_spatial: bool = False,
    log_every: int = 20,
    checkpoint_every: int | None = None,
    reject_every: int = 50,
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
    stage1_core_stamp: int = L.STAGE1_CORE_STAMP,
    no_stage1_dx_only: bool = False,
    group_frac: float = 1.0,
    frame_frac: float = 1.0,
    recenter_n_iter: int = EM.HOTPATH_RECENTER_N_ITER,
    huber_delta: float = L.HUBER_DELTA_DEFAULT,
    lambda_pixel_lap: float = 1e-2,
    lambda_lap: float = 1e-3,
    lambda_fine_nbr: float = 0.0,
    fine_nbr_sigma: float = 0.0,
    fine_nbr_mode: str = L.FINE_NBR_MODE_DEFAULT,
    lambda_chroma_lap: float = 0.0,
    lambda_smooth_wcs: float = 1e-4,
    lambda_smooth_w: float = 1e-3,
    flux_objective: str = "l2",
    huber_irls_iters: int = 2,
    support_size_weight_power: float = 0.0,
    stamp_pedestal: bool = False,
    profile_w: bool = False,
    profile_w_iters: int = 2,
    lambda_centroid: float = 1000.0,
    centroid_in_grad: bool = False,
    n_frames_per_stage: list[int] | None = None,
    init_params: Path | None = None,
    resume_state: Path | None = None,
    configuration_fingerprint: str | None = None,
    stamp_chunk: int | None = None,
    frame_block: int | None = None,
    gpu_flux_frame_block: int = 8,
    fit_radius_min: float = 0.0,
    stop_file: Path | None = None,
    advance_file: Path | None = None,
) -> dict[str, Any]:
    """Run Adam stages 1..``stage`` from a prepared bundle. Returns updated meta.

    ``fit_radius_min``: floor (``np.maximum``) the bundle's already-baked
    ``fit_radius_stage1/23`` at this value. A bundle only stores the per-group
    radius computed from mag at export time (not the raw mag), so this is the
    knob for widening an already-exported bundle's NLL radius mask without a
    re-export -- see ``loss.fit_radius_tiers_default`` for why the *new*
    export-time default no longer needs this in the first place.
    """
    if reject_mode not in ("hysteresis", "audit", "standardized", "legacy", "two-level", "static"):
        raise SystemExit(
            "reject_mode must be hysteresis, audit, standardized, legacy, two-level "
            f"or static, got {reject_mode!r}"
        )
    l2_churn_cap_frac_eff = l2_churn_cap_frac
    l2_hysteresis_n_eff = l2_hysteresis_n
    if reject_every > 0:
        from . import stamp_reject as SR  # noqa: PLC0415  (heavy; optional for Colab bench)
        if l2_churn_cap_frac_eff is None:
            l2_churn_cap_frac_eff = SR.DEFAULT_L2_CHURN_CAP_FRAC
        if l2_hysteresis_n_eff is None:
            l2_hysteresis_n_eff = SR.DEFAULT_L2_HYSTERESIS_N

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_jsonl = out_dir / "history.jsonl"
    if int(start_stage) <= 1 and resume_state is None:
        history_jsonl.write_text("")
    elif not history_jsonl.exists():
        history_jsonl.write_text("")

    stages_steps = steps_per_stage or [30, 50, 80]
    stages_lr = lr_per_stage or [1e-2, 3e-4, 1e-4]
    stamp_physical = int(bundle.stamp_physical)
    stage1_core = int(stage1_core_stamp)
    if stage1_core < 0 or (stage1_core > 0 and stage1_core % 2 == 0):
        raise SystemExit(f"stage1_core_stamp must be 0 or odd positive, got {stage1_core}")

    groups = bundle.group_set()
    k_tiers = bundle.k_tiers if bundle.k_tiers else (groups.max_group_size,)
    packed = bool(bundle.is_packed)
    # Tier-segmented (bundle_version=2) bundles carry their own per-(K,P)
    # bucket plan and PackedTier stamp arrays -- use those directly and never
    # touch bundle.data/.pix_x/etc (which would force a dense, globally
    # padded reconstruction defeating the whole point of tiered storage).
    tier_native = packed and bundle.packed_tiers is not None
    pix_x_full = pix_y_full = pix_valid_full = None
    # Packed: (K,P) tiers so flux-solve / render pad at true support size.
    # Square: K-tiers only (existing path).
    if tier_native:
        buckets = list(bundle.packed_bucket_plan())
        tier_sources = list(bundle.packed_tiers)
        _log(
            f"  packed (K,P)-bucketed [tier-native] into {len(buckets)} tier(s): "
            + ", ".join(
                f"K={kt}/P={pt}(n={bg.n_groups})" for bg, _, _, _, kt, pt in buckets
            )
        )
    elif packed:
        pix_x_full = np.asarray(bundle.pix_x)
        pix_y_full = np.asarray(bundle.pix_y)
        pix_valid_full = np.asarray(bundle.pix_valid)
        p_tiers = (
            bundle.p_tiers if bundle.p_tiers
            else (int(np.asarray(bundle.pix_x).shape[-1]),)
        )
        kp_buckets = PS.bucket_packed_by_kp(
            groups, bundle.stamp_center_x, bundle.stamp_center_y, pix_valid_full,
            k_tiers=k_tiers, p_tiers=p_tiers,
        )
        buckets = list(kp_buckets)
        tier_sources = [None] * len(buckets)
        _log(
            f"  packed (K,P)-bucketed into {len(buckets)} tier(s): "
            + ", ".join(
                f"K={kt}/P={pt}(n={bg.n_groups})" for bg, _, _, _, kt, pt in buckets
            )
        )
    else:
        k_only = G.bucket_groups_by_size(
            groups, bundle.stamp_center_x, bundle.stamp_center_y, tiers=k_tiers,
        )
        buckets = [
            (bg, bcx, bcy, bidx, bg.max_group_size, 0)
            for bg, bcx, bcy, bidx in k_only
        ]
        tier_sources = [None] * len(buckets)
        _log(
            f"  K-bucketed into {len(buckets)} tier(s): "
            + ", ".join(
                f"K={bg.max_group_size}(n={bg.n_groups})"
                for bg, _, _, _, _, _ in buckets
            )
        )

    # Stop-gradient routing is a per-group invariant.  Make each execution
    # batch homogeneous here, including the K=1/P=64 tier where compact faint
    # anchors can otherwise share a bucket with a tiny bright support.  This
    # keeps the renderer single-pass per batch instead of paying for a mixed
    # bright/faint conditional render.
    contributor = np.asarray(bundle.is_epsf_contributor, dtype=bool)
    split_buckets = []
    split_sources = []
    for source, (bg, bcx, bcy, bidx, kt, pt) in zip(tier_sources, buckets):
        flags = contributor[np.asarray(bidx, dtype=int)]
        for flag in (True, False):
            rows = np.flatnonzero(flags == flag)
            if not rows.size:
                continue
            if rows.size == len(flags):
                sub_bg, sub_cx, sub_cy = bg, bcx, bcy
                sub_source = source
            else:
                members = np.asarray(bg.members)[rows]
                valid = np.asarray(bg.valid)[rows]
                kept = np.zeros_like(bg.kept_star_mask, dtype=bool)
                for member in members[valid]:
                    if member >= 0:
                        kept[int(member)] = True
                sub_bg = G.GroupSet(
                    n_groups=int(rows.size), max_group_size=bg.max_group_size,
                    members=members, valid=valid, kept_star_mask=kept,
                    dropped_oversized=0,
                )
                sub_cx, sub_cy = np.asarray(bcx)[rows], np.asarray(bcy)[rows]
                if source is None:
                    sub_source = None
                else:
                    sub_source = replace(
                        source,
                        data=source.data[rows], noise=source.noise[rows],
                        weight_u8=source.weight_u8[rows], pix_x=source.pix_x[rows],
                        pix_y=source.pix_y[rows], pix_valid=source.pix_valid[rows],
                        group_idx=source.group_idx[rows],
                        is_epsf_contributor=np.full(rows.size, flag, dtype=bool),
                    )
            split_buckets.append((sub_bg, sub_cx, sub_cy, np.asarray(bidx)[rows], kt, pt))
            split_sources.append(sub_source)
    buckets, tier_sources = split_buckets, split_sources

    r_stage1 = bundle.fit_radius_stage1
    if fit_radius_min > 0:
        r_stage1 = np.maximum(r_stage1, np.float32(fit_radius_min))
    stamp_w = bundle.stamp_snr_weight

    # Reference colour for the chromatic term. Computed ONCE over the whole fitting
    # population, before the bucket loop, and weighted by the same per-group weight
    # the loss uses, so that the population mean of (c - c_ref) is zero.
    #
    # This gauge is not cosmetic. eps * <delta> * D[P] is a star-independent shape
    # perturbation, which lies exactly in the tangent space of the free per-node
    # ePSF base: it is a genuine first-order null direction. Centring delta removes
    # it. (The chromatic SHIFT is only strongly CORRELATED with wcs_coeff, not
    # degenerate with it, because delta varies star to star at fixed position.)
    bp_rp_all = getattr(bundle, "bp_rp", None)
    colour_ref = L.colour_ref_from_bundle(bundle)
    if bp_rp_all is not None:
        _c = np.asarray(bp_rp_all, dtype=np.float64)
        _log(
            f"chroma: {int(np.isfinite(_c).sum())}/{_c.size} stars have BP-RP, "
            f"weighted c_ref={colour_ref:.4f}"
        )

    ctxs = []
    for tier, (bg, bcx, bcy, bidx, _kt, pt) in zip(tier_sources, buckets):
        kw = dict(
            cheb_static=bundle.cheb_static,
            wcs_frame_basis=bundle.wcs_frame_basis,
            w_frame_basis=bundle.w_frame_basis,
            epsf_grid=bundle.epsf_grid,
            groups=bg,
            ra=bundle.ra,
            dec=bundle.dec,
            stamp_center_x=bcx,
            stamp_center_y=bcy,
            t_exp_sec=bundle.t_exp_sec,
            stamp_snr_weight=stamp_w[bidx],
            fit_radius=r_stage1[bidx],
            x_lin=bundle.x_lin,
            y_lin=bundle.y_lin,
            cheb_basis=bundle.cheb_basis,
            is_epsf_contributor=contributor[bidx],
            bp_rp=bp_rp_all,
            colour_ref=colour_ref,
        )
        if tier is not None:
            kw["pix_x"] = tier.pix_x
            kw["pix_y"] = tier.pix_y
            kw["pix_valid"] = tier.pix_valid
        elif packed:
            kw["pix_x"] = pix_x_full[bidx, :pt]
            kw["pix_y"] = pix_y_full[bidx, :pt]
            kw["pix_valid"] = pix_valid_full[bidx, :pt]
        ctxs.append(L.build_static_context(**kw))

    params = params0_as_jnp(bundle)
    if w_spatial and params["w_coeff"].ndim == 2:
        # T3: give the temporal mode amplitude a per-node field instead of one
        # global w_k(t). Bundle params0 is always built 2-D (bundle export
        # never knows about this flag); upgrade it here, at the same point
        # the chroma leaves get added below, so a fresh (no --init-params) run
        # starts from an all-zero spatial field (bit-identical predictions to
        # the 2-D case at step 0, since every node's curve is exactly zero).
        _base = np.asarray(bundle.epsf_base)
        _nr, _nc = int(_base.shape[0]), int(_base.shape[1])
        _n_modes, _n_wbasis = (int(s) for s in params["w_coeff"].shape)
        params["w_coeff"] = jnp.zeros((_n_modes, _nr, _nc, _n_wbasis), dtype=jnp.float32)
        _log(
            f"w_coeff: spatial ({_nr}x{_nc} nodes), K={_n_modes}, "
            f"n_basis={_n_wbasis} ({_n_modes * _nr * _nc * _n_wbasis} scalars, "
            f"was {_n_modes * _n_wbasis})"
        )
    if chroma_affine and not chroma:
        raise SystemExit("--chroma-affine requires --chroma")
    if chroma_kurt and not chroma_affine:
        raise SystemExit("--chroma-kurt requires --chroma-affine")
    if chroma_image and not chroma:
        raise SystemExit("--chroma-image requires --chroma")
    if chroma_halo and not chroma:
        raise SystemExit("--chroma-halo requires --chroma")
    if (chroma_halo_index is not None or chroma_halo_core_px is not None) and not chroma_halo:
        raise SystemExit(
            "--chroma-halo-index/--chroma-halo-core-px only mean anything with --chroma-halo"
        )
    if chroma_halo:
        # Set BEFORE any params are built or any forward is traced: the profile is
        # read inside epsf_model.chroma_halo_field at call time, so a later override
        # would silently apply to only part of a run.
        if chroma_halo_index is not None:
            EM.CHROMA_HALO_INDEX = float(chroma_halo_index)
        if chroma_halo_core_px is not None:
            EM.CHROMA_HALO_CORE_PX = float(chroma_halo_core_px)
        EM.CHROMA_HALO_FLUX_NEUTRAL = bool(chroma_halo_flux_neutral)
    if stage4_freeze_others and not chroma_image:
        raise SystemExit(
            "--stage4-freeze-others only has an effect with --chroma-image "
            "(stage 4 trains nothing else)"
        )
    if chroma:
        if bp_rp_all is None:
            raise SystemExit(
                "--chroma requires a bundle carrying per-star bp_rp; re-export the "
                "bundle with a Gaia catalog that has phot_bp/rp_mean_mag"
            )
        _base = np.asarray(bundle.epsf_base)
        _nr, _nc = int(_base.shape[0]), int(_base.shape[1])
        _chroma_leaf_shapes = [("chroma_shift", (2, _nr, _nc)), ("chroma_dilation", (_nr, _nc))]
        # C1 colour-affine extension, default-off: a checkpoint that already carries
        # these leaves (e.g. --init-params a warm-started params_warm_C1.npz) picks
        # them up regardless of this flag via the merge below; --chroma-affine only
        # controls whether a FRESH (no --init-params) run starts training them.
        if chroma_affine:
            _chroma_leaf_shapes += [("chroma_aniso", (_nr, _nc)), ("chroma_shear", (_nr, _nc))]
        if chroma_kurt:
            _chroma_leaf_shapes.append(("chroma_kurt", (_nr, _nc)))
        if chroma_halo:
            _chroma_leaf_shapes.append(("chroma_halo", (_nr, _nc)))
        # The free colour image is ONE global (G, G) image, not a per-node field:
        # 36 free images on a 6x6 grid would be ~121k parameters against a
        # ~0.3-mag colour lever on a single frame.
        _g_size = int(_base.shape[-1])
        _image_shape = (_g_size, _g_size)
        if chroma_image:
            _chroma_leaf_shapes.append(("chroma_image", _image_shape))
        for _k, _shape in _chroma_leaf_shapes:
            if _k not in params:
                params[_k] = jnp.zeros(_shape, dtype=jnp.float32)
        _n_chroma = sum(
            int(np.prod(_shape)) for _k, _shape in _chroma_leaf_shapes
            if _k != "chroma_image"
        )
        _log(
            f"chroma: enabled, {_nr}x{_nc} node grid, "
            f"{_n_chroma} trainable parameters (unfrozen at stage 3)"
            + (", affine" if chroma_affine else "")
            + (", kurt" if chroma_kurt else "")
            + (f", halo r^-{EM.CHROMA_HALO_INDEX:g} (core {EM.CHROMA_HALO_CORE_PX:g} px"
               + (", FLUX-NEUTRAL" if EM.CHROMA_HALO_FLUX_NEUTRAL else "") + ")"
               if chroma_halo else "")
        )
        if chroma_image:
            _log(
                f"chroma: free colour image enabled, one global {_g_size}x{_g_size} "
                f"dP/dcolour image ({_g_size * _g_size} parameters, unfrozen at "
                f"stage 4 only), others_frozen_in_stage4={bool(stage4_freeze_others)}, "
                f"lr_scale={chroma_image_lr_scale} (its OWN optax bucket: shared "
                f"with the parametric leaves at 1.0 the loss rose 68.0->104.4 in one step)"
            )
    epsf_modes_init = jnp.asarray(bundle.epsf_modes)

    resume_opt_leaves = None
    resume_gate_state = None
    resume_meta = {}
    if resume_state is not None:
        from . import training_state as TS
        resume_params, resume_opt_leaves, resume_meta = TS.load(resume_state)
        resume_gate_state = TS.decode_two_level_gate(
            resume_meta.get("two_level_gate", {}), TS.load_extras(resume_state),
        )
        expected = configuration_fingerprint
        if not expected:
            raise SystemExit("resume requires a bundle/configuration fingerprint")
        TS.validate(resume_meta, expected_fingerprint=expected, reject_every=reject_every)
        for key, value in resume_params.items():
            if key not in params or tuple(value.shape) != tuple(params[key].shape):
                raise SystemExit(f"resume parameter {key} is incompatible with this bundle")
            params[key] = jnp.asarray(value)
        start_stage = int(resume_meta["stage"])
        _log(f"  loaded training state from {resume_state} at stage={start_stage} next_step={resume_meta.get('next_step')}")

    start_stage = int(start_stage)
    if start_stage > int(stage):
        raise SystemExit(f"--start-stage {start_stage} exceeds --stage {stage}")
    if start_stage > 1 and init_params is None:
        # Allow resume from baked params0 (stage-0 warmstart already in bundle).
        _log("  start_stage>1 with no --init-params: using bundle params0")

    inherited_stamp_active = None
    if init_params is not None:
        loaded_params = FIT.load_params_npz(init_params)
        with np.load(init_params, allow_pickle=False) as init_payload:
            if "stamp_active" in init_payload.files:
                inherited_stamp_active = np.asarray(init_payload["stamp_active"], dtype=np.float32)
        merged = dict(params)
        # Auto-adopt any optional leaf the CHECKPOINT carries but this run has not
        # (yet) initialized -- e.g. resuming into a C1 colour-affine checkpoint
        # (``params_warm_C1.npz``) without having to also pass --chroma-affine on
        # this exact invocation: seed a zero array of the checkpoint's own shape so
        # the generic merge loop below (data-driven, not flag-driven) picks it up.
        # A run that never enables --chroma at all never reaches this branch's
        # loaded_params containing chroma_* keys in the first place (the bundle/
        # export path does not produce them), so this cannot smuggle chroma into a
        # non-chromatic run.
        for _k in FIT.ALL_OPTIONAL_LEAVES:
            if _k in loaded_params and _k not in merged:
                merged[_k] = jnp.zeros_like(jnp.asarray(loaded_params[_k]))
        # Iterate the REQUIRED leaves plus whichever optional ones both sides have.
        # This loop is the only channel between isolated stages, which run in fresh
        # processes and hand parameters over solely through --init-params. A
        # hardcoded four-key list here silently resets the chromatic leaves to zero
        # at every stage boundary, which looks exactly like "chroma did not train".
        _merge_keys = list(FIT.STAGE_LEAVES) + [
            k for k in FIT.ALL_OPTIONAL_LEAVES if k in loaded_params and k in merged
        ]
        for key in _merge_keys:
            if tuple(loaded_params[key].shape) != tuple(merged[key].shape):
                raise SystemExit(
                    f"--init-params {init_params} {key} shape "
                    f"{tuple(loaded_params[key].shape)} != bundle "
                    f"{tuple(merged[key].shape)}"
                )
            merged[key] = loaded_params[key]
        params = merged
        _log(f"  loaded init params from {init_params} (exact leaf match)")

    checkpoint_every = checkpoint_every if checkpoint_every is not None else log_every
    params_latest = out_dir / "params_latest.npz"
    if start_stage <= 1:
        FIT.save_params_npz(out_dir / "params_stage0.npz", params)
        _log(f"  wrote {out_dir / 'params_stage0.npz'}")
    else:
        _log(f"  resume: start_stage={start_stage}")
    FIT.save_params_npz(params_latest, params)
    _write_params_provenance(out_dir, stage=0, step=-1, reason="initial")
    _log(f"  wrote {params_latest}")

    frames_per_stage = n_frames_per_stage or [bundle.n_frames] * 3
    meta = dict(bundle.meta)
    meta.update({
        "from_bundle": True,
        "n_groups": bundle.n_groups,
        "n_frames": bundle.n_frames,
        "stamp_physical": stamp_physical,
        "stage1_core_stamp": stage1_core,
        "stage1_dx_only": not bool(no_stage1_dx_only),
        "n_frames_per_stage": frames_per_stage,
        "group_frac": group_frac,
        "frame_frac": frame_frac,
        "stamp_chunk": stamp_chunk,
        "frame_block": frame_block,
        "epsf_lr_scale": epsf_lr_scale,
        "w_lr_scale": w_lr_scale,
        "w_spatial": bool(w_spatial),
        "stage2_freeze_wcs_steps": stage2_freeze_wcs_steps,
        "grad_clip": grad_clip,
        "stage": stage,
        "start_stage": start_stage,
        "init_params": str(init_params) if init_params else None,
        "steps_per_stage": stages_steps,
        "lr_per_stage": stages_lr,
        "log_every": log_every,
        "checkpoint_every": checkpoint_every,
        "fit_complete": False,
        "k_tiers": list(k_tiers),
        "k_buckets": [
            {"K": kt, "P": pt, "n_groups": bg.n_groups}
            for bg, _, _, _, kt, pt in buckets
        ],
        "packed": packed,
        "p_tiers": list(bundle.p_tiers) if bundle.p_tiers else [],
        "reject_every": reject_every,
        "reject_burn_in": reject_burn_in,
        "reject_n_sigma": reject_n_sigma,
        "reject_mode": reject_mode,
        "reject_tau_drop": reject_tau_drop,
        "reject_tau_keep": reject_tau_keep,
        "reject_max_churn": reject_max_churn,
        "early_stop_tol_coarse": early_stop_tol_coarse,
        "early_stop_tol_fine": early_stop_tol,
        "l2_churn_cap_frac": l2_churn_cap_frac_eff,
        "l2_hysteresis_n": l2_hysteresis_n_eff,
        "l2_freeze_scale_after_first": l2_freeze_scale_after_first,
        "l2_scale_ema_alpha": l2_scale_ema_alpha,
        "standardized_shrinkage_frames": float(standardized_shrinkage_frames),
        "standardized_min_weight": float(standardized_min_weight),
        "flux_objective": flux_objective,
        "huber_irls_iters": int(huber_irls_iters),
        "support_size_weight_power": float(support_size_weight_power),
        "lambda_pixel_lap": float(lambda_pixel_lap),
        "lambda_lap": float(lambda_lap),
        "lambda_fine_nbr": float(lambda_fine_nbr),
        "fine_nbr_sigma": float(fine_nbr_sigma),
        "fine_nbr_mode": str(fine_nbr_mode),
        "lambda_chroma_lap": float(lambda_chroma_lap),
        "lambda_smooth_wcs": float(lambda_smooth_wcs),
        "lambda_smooth_w": float(lambda_smooth_w),
        "profile_w": bool(profile_w),
        "profile_w_iters": int(profile_w_iters),
    })
    (out_dir / "fit_meta.json").write_text(json.dumps(meta, indent=2))

    if profile_w and w_spatial:
        raise NotImplementedError(
            "--profile-w solves ONE amplitude per mode per frame, shared by the "
            "whole field; --w-spatial makes that amplitude a per-node field. "
            "The two parameterisations are mutually exclusive."
        )
    if float(fine_nbr_sigma) > 0.0 and float(lambda_fine_nbr) > 0.0:
        raise ValueError("--fine-nbr-sigma and --lambda-fine-nbr are mutually exclusive")
    if float(fine_nbr_sigma) > 0.0 and float(support_size_weight_power) != 1.0:
        _log("  WARNING: --fine-nbr-sigma is only a physical prior width with "
             "--support-size-weight-power 1 (per-pixel likelihood); got "
             f"{support_size_weight_power}")
    loss_weights = L.LossWeights(
        huber_delta=huber_delta,
        flux_objective=flux_objective,
        huber_irls_iters=int(huber_irls_iters),
        support_size_weight_power=float(support_size_weight_power),
        lambda_centroid=lambda_centroid,
        lambda_pixel=float(lambda_pixel_lap),
        lambda_lap=float(lambda_lap),
        lambda_fine_nbr=float(lambda_fine_nbr),
        fine_nbr_sigma=float(fine_nbr_sigma),
        fine_nbr_mode=str(fine_nbr_mode),
        lambda_chroma_lap=float(lambda_chroma_lap),
        lambda_smooth_wcs=float(lambda_smooth_wcs),
        lambda_smooth_w=float(lambda_smooth_w),
        centroid_in_grad=bool(centroid_in_grad),
        stamp_pedestal=bool(stamp_pedestal),
        profile_w=bool(profile_w),
        profile_w_iters=int(profile_w_iters),
    )
    _log(
        f"  loss weights: lambda_centroid={lambda_centroid}, "
        f"centroid_in_grad={bool(centroid_in_grad)}, "
        f"lambda_lap={lambda_lap}, lambda_fine_nbr={lambda_fine_nbr}, fine_nbr_sigma={fine_nbr_sigma}, fine_nbr_mode={fine_nbr_mode}, lambda_pixel={lambda_pixel_lap}, "
        f"lambda_chroma_lap={lambda_chroma_lap}, "
        f"flux_objective={flux_objective}, huber_irls_iters={huber_irls_iters}, "
        f"support_size_weight_power={support_size_weight_power}, "
        f"stamp_pedestal={bool(stamp_pedestal)}, "
        f"profile_w={bool(profile_w)} (iters={int(profile_w_iters)})"
    )
    try:
        import jax
        _log(f"  jax backend: {jax.default_backend()}")
    except Exception as exc:
        _log(f"  jax backend: unavailable ({exc})")

    all_history: list = []
    if resume_state is not None and history_jsonl.exists():
        for line in history_jsonl.read_text().splitlines():
            try:
                all_history.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    mask_active = np.asarray(bundle.mask_active, dtype=np.float32)
    if start_stage >= 3 and inherited_stamp_active is not None:
        if inherited_stamp_active.shape != mask_active.shape:
            raise SystemExit(f"--init-params stamp_active shape {inherited_stamp_active.shape} != bundle {mask_active.shape}")
        mask_active[:] = inherited_stamp_active
        _log("  stage 3 initialized stamp_active from init checkpoint")
    n_frames_loaded = bundle.n_frames

    # Per-bucket stamp arrays, sliced to each bucket's own group rows (and,
    # for packed, its own P tier) exactly once -- then the (potentially
    # huge, ~90% padding) full dense arrays are dropped instead of being
    # held alive for the entire run. Tier-segmented (bundle_version=2)
    # bundles never touch bundle.data/.noise/.weight at all: their
    # PackedTier arrays are already sized to true per-tier occupancy, so
    # they're used directly. Packed weight may be uint8 (tier-native) or
    # float32 (legacy/square); cast to float32 only at the jnp boundary
    # below, per-stage, on the already-small per-bucket slice.
    if tier_native:
        bucket_data = [t.data for t in tier_sources]
        bucket_noise = [t.noise for t in tier_sources]
        bucket_weight = [t.weight_u8 for t in tier_sources]
    else:
        full_data = np.asarray(bundle.data)
        full_noise = np.asarray(bundle.noise)
        full_weight = np.asarray(bundle.weight)
        bucket_data = []
        bucket_noise = []
        bucket_weight = []
        for _bg, _bcx, _bcy, bidx, _kt, pt in buckets:
            d = full_data[bidx]
            n = full_noise[bidx]
            w = full_weight[bidx]
            if packed:
                d = d[:, :, :pt]
                n = n[:, :, :pt]
                w = w[:, :, :pt]
            bucket_data.append(d)
            bucket_noise.append(n)
            bucket_weight.append(w)
        del full_data, full_noise, full_weight

    r_stage23 = bundle.fit_radius_stage23
    if fit_radius_min > 0:
        r_stage23 = np.maximum(r_stage23, np.float32(fit_radius_min))
        _log(f"  fit_radius_min={fit_radius_min}: floored bundle's baked fit_radius_stage23 "
             f"(orig med={float(np.median(bundle.fit_radius_stage23)):.2f} -> "
             f"new med={float(np.median(r_stage23)):.2f})")
    wcs_n_basis = int(bundle.wcs_frame_basis.shape[1])
    w_n_basis = int(bundle.w_frame_basis.shape[1])
    wcs_second_diff = T.second_difference_matrix(wcs_n_basis)
    w_second_diff = T.second_difference_matrix(w_n_basis)

    def _middle_frame_index(n_want: int) -> np.ndarray:
        n_want = int(max(1, min(n_want, n_frames_loaded)))
        start = max(0, (n_frames_loaded - n_want) // 2)
        return np.arange(start, start + n_want, dtype=np.int32)

    if gpu_flux_frame_block < 1:
        raise ValueError("gpu_flux_frame_block must be >= 1")
    fds = []
    stopped = False
    for stage_i in range(start_stage, stage + 1):
        if stop_file is not None and stop_file.is_file():
            meta["fit_complete"] = False
            meta["stop_requested"] = True
            meta["stop_stage"] = stage_i
            stopped = True
            _log("  stop requested before stage launch; preserving existing checkpoint")
            break
        n_steps = stages_steps[stage_i - 1] if stage_i - 1 < len(stages_steps) else stages_steps[-1]
        lr = stages_lr[stage_i - 1] if stage_i - 1 < len(stages_lr) else stages_lr[-1]
        fit_r = r_stage1 if stage_i <= 1 else r_stage23
        n_want = frames_per_stage[stage_i - 1] if stage_i - 1 < len(frames_per_stage) else frames_per_stage[-1]
        fidx = _middle_frame_index(n_want)
        use_core = (not packed) and stage_i == 1 and stage1_core > 0 and stage1_core < stamp_physical
        n_pix = stage1_core if use_core else stamp_physical
        freeze_wcs = stage2_freeze_wcs_steps if stage_i == 2 else 0
        # Keep the rolling full snapshot available even when slim checkpoint
        # history is disabled. The isolated-stage handoff remains params_stageN.
        ckpt_path = params_latest
        use_dx_only = (not packed) and (stage_i == 1 and not bool(no_stage1_dx_only))
        stamp_chunk_eff = stamp_chunk

        fds = []
        for i, (ctx, tier, (bg, bcx, bcy, bidx, _kt, pt)) in enumerate(
            zip(ctxs, tier_sources, buckets),
        ):
            data_s = bucket_data[i][:, fidx]
            noise_s = bucket_noise[i][:, fidx]
            weight_s = bucket_weight[i][:, fidx]  # uint8 (tiered) or float32
            if use_core:
                data_s = L.central_crop_stamps(data_s, stage1_core)
                noise_s = L.central_crop_stamps(noise_s, stage1_core)
                weight_s = L.central_crop_stamps(weight_s, stage1_core)
            ctx_stage = L.with_fit_radius(ctx, fit_r[bidx])
            mask_s = np.asarray(mask_active[bidx][:, fidx], dtype=np.float32)
            ctx_kw = dict(
                wcs_frame_basis=jnp.asarray(np.asarray(bundle.wcs_frame_basis)[fidx], dtype=jnp.float32),
                w_frame_basis=jnp.asarray(np.asarray(bundle.w_frame_basis)[fidx], dtype=jnp.float32),
                stamp_active=jnp.asarray(mask_s, dtype=jnp.float32),
            )
            if tier is not None:
                ctx_kw["pix_x"] = jnp.asarray(tier.pix_x, dtype=jnp.float32)
                ctx_kw["pix_y"] = jnp.asarray(tier.pix_y, dtype=jnp.float32)
                ctx_kw["pix_valid"] = jnp.asarray(tier.pix_valid, dtype=jnp.float32)
            elif packed:
                ctx_kw["pix_x"] = jnp.asarray(pix_x_full[bidx, :pt], dtype=jnp.float32)
                ctx_kw["pix_y"] = jnp.asarray(pix_y_full[bidx, :pt], dtype=jnp.float32)
                ctx_kw["pix_valid"] = jnp.asarray(pix_valid_full[bidx, :pt], dtype=jnp.float32)
            ctx_stage = replace(ctx_stage, **ctx_kw)
            fds.append(FIT.FitData(
                ctx=ctx_stage,
                data=jnp.asarray(data_s),
                noise=jnp.asarray(noise_s),
                weight=jnp.asarray(weight_s, dtype=jnp.float32),
                wcs_second_diff=wcs_second_diff,
                w_second_diff=w_second_diff,
                epsf_modes_init=epsf_modes_init,
                weights=loss_weights,
                use_dx_only=use_dx_only,
                do_recenter=True,
                recenter_n_iter=int(recenter_n_iter),
                n_pix=n_pix,
                group_frac=float(group_frac),
                frame_frac=float(frame_frac),
                mask_stamp_active=mask_s,
                stamp_chunk=stamp_chunk_eff,
                frame_block=frame_block,
                group_ids=np.asarray(bidx),
            ))
        _log(
            f"--- stage {stage_i}: {n_steps} steps, lr={lr}, "
            f"fit_radius_med={float(np.median(fit_r)):.2f}, "
            f"n_frames={len(fidx)}, n_pix={n_pix}, dx_only={fds[0].use_dx_only}, "
            f"packed={packed}, stamp_chunk={stamp_chunk_eff}, frame_block={frame_block}, "
            f"kp_buckets={[(kt, pt) for _, _, _, _, kt, pt in buckets]} ---"
        )
        t_stage = time.time()
        def _save_state(current_params, current_opt_state, gate_state=None, **position):
            from . import training_state as TS
            gate_meta, gate_arrays = TS.encode_two_level_gate(gate_state)
            mask_active[:] = FIT.merge_fds_stamp_active(mask_active, fds, buckets, fidx)
            FIT.save_stamp_active_npz(out_dir / "stamp_active_latest.npz", mask_active)
            FIT.save_params_npz(params_latest, current_params, stamp_active=mask_active)
            TS.save(out_dir / "training_state_latest.npz", params=current_params,
                    opt_state=current_opt_state, metadata={**position,
                    "fingerprint": configuration_fingerprint, "schedule": stages_steps,
                    "two_level_gate": gate_meta,
                    "optimizer": {"lr_per_stage": stages_lr,
                                  "epsf_lr_scale": epsf_lr_scale,
                                  "w_lr_scale": w_lr_scale,
                                  "grad_clip": grad_clip},
                    "history_lines": sum(1 for _ in history_jsonl.open())},
                    extra_arrays=gate_arrays)

        params, history = FIT.run_stage(
            params, fds, stage=stage_i, n_steps=n_steps, lr=lr,
            log_every=log_every,
            epsf_lr_scale=epsf_lr_scale,
            chroma_lr_scale=chroma_lr_scale,
            w_lr_scale=w_lr_scale,
            freeze_wcs_steps=freeze_wcs,
            freeze_modes=freeze_modes,
            # Task PW: w_coeff is not in the model when the amplitude is
            # profiled out, so it must not be an Adam leaf either.
            freeze_w=bool(profile_w),
            freeze_others_stage4=bool(stage4_freeze_others),
            chroma_image_lr_scale=float(chroma_image_lr_scale),
            grad_clip=grad_clip,
            history_path=history_jsonl,
            checkpoint_path=ckpt_path,
            checkpoint_every=checkpoint_every if checkpoint_every != 0 else None,
            reject_every=reject_every,
            reject_burn_in=reject_burn_in,
            reject_n_sigma=reject_n_sigma,
            reject_mode=reject_mode,
            reject_tau_drop=reject_tau_drop,
            reject_tau_keep=reject_tau_keep,
            reject_max_churn=reject_max_churn,
            l2_churn_cap_frac=l2_churn_cap_frac_eff,
            l2_hysteresis_n=l2_hysteresis_n_eff,
            l2_freeze_scale_after_first=l2_freeze_scale_after_first,
            l2_scale_ema_alpha=l2_scale_ema_alpha,
            standardized_shrinkage_frames=standardized_shrinkage_frames,
            standardized_min_weight=standardized_min_weight,
            early_stop_patience=early_stop_patience,
            early_stop_tol=early_stop_tol,
            early_stop_tol_coarse=early_stop_tol_coarse,
            resume_phase=resume_meta.get("phase") if stage_i == start_stage else None,
            resume_next_step=int(resume_meta.get("next_step", 0)) if stage_i == start_stage else 0,
            resume_opt_leaves=resume_opt_leaves if stage_i == start_stage else None,
            state_callback=_save_state,
            resume_gate_state=resume_gate_state if stage_i == start_stage else None,
            stop_file=stop_file,
            advance_file=advance_file,
        )
        _log(f"  stage {stage_i} wall time: {time.time() - t_stage:.1f}s")
        all_history.extend(history)
        mask_active[:] = FIT.merge_fds_stamp_active(mask_active, fds, buckets, fidx)
        FIT.save_stamp_active_npz(out_dir / f"stamp_active_stage{stage_i}.npz", mask_active)
        FIT.save_stamp_active_npz(out_dir / "stamp_active_latest.npz", mask_active)
        stage_path = out_dir / f"params_stage{stage_i}.npz"
        FIT.save_params_npz(stage_path, params, stamp_active=mask_active)
        FIT.save_params_npz(params_latest, params, stamp_active=mask_active)
        completed_step = int(
            max(
                (
                    row.get("step", -1)
                    for row in history
                    if row.get("stage") == stage_i and "step" in row
                ),
                default=-1,
            )
        )
        advance_requested = any(
            row.get("event") == "stage_advance_requested" for row in history
        )
        _write_params_provenance(
            out_dir,
            stage=stage_i,
            step=completed_step,
            reason=("stage_advance_requested" if advance_requested else
                    ("stop_requested" if stop_file is not None and stop_file.is_file()
                     else "stage_complete")),
        )
        _log(f"  wrote {stage_path} (+ stamp_active_stage{stage_i}.npz)")
        if advance_requested:
            try:
                advance_file.unlink()
            except OSError:
                pass
            _log(f"  stage {stage_i}: advance request consumed; continuing to next stage")
        if stop_file is not None and stop_file.is_file():
            meta["fit_complete"] = False
            meta["stop_requested"] = True
            meta["stop_stage"] = stage_i
            meta["stop_step"] = completed_step
            stopped = True
            _log("  stop requested: checkpoint saved; exporting final GPU flux solution")
            break

    FIT.save_params_npz(out_dir / "params.npz", params, stamp_active=mask_active)
    FIT.save_stamp_active_npz(out_dir / "stamp_active.npz", mask_active)
    (out_dir / "history.json").write_text(json.dumps(all_history, indent=2))
    if stage >= 1 and fds:
        actives = [np.asarray(fd_i.ctx.stamp_active) for fd_i in fds]
        n_active = sum(int(a.sum()) for a in actives)
        n_total = sum(int(a.size) for a in actives)
    if fds:
        try:
            from .gpu_flux_export import export_gpu_flux_solution
            export_gpu_flux_solution(
                out_dir, params, bundle, fds, buckets, fidx, mask_active,
                frame_block=gpu_flux_frame_block,
                pedestal=loss_weights.stamp_pedestal,
                profile_w=loss_weights.profile_w,
                profile_w_iters=loss_weights.profile_w_iters,
            )
            meta["gpu_flux_exported"] = True
            meta["gpu_flux_frame_block"] = int(gpu_flux_frame_block)
        except Exception as exc:
            meta["gpu_flux_exported"] = False
            _log(f"  warning: export_gpu_flux_solution failed: {exc}")
    meta["fit_complete"] = not stopped
    (out_dir / "fit_meta.json").write_text(json.dumps(meta, indent=2))
    _log(f"wrote {out_dir}")
    return meta
