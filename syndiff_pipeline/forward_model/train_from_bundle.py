# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Lean GPU/CPU entry point for training an already-prepared FitBundle.

This module intentionally has no workspace-preparation branch.  Its runtime
dependencies are the Python standard library, NumPy, JAX, and Optax through the
training modules it imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-bundle", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--stage", type=int, choices=[0, 1, 2, 3, 4], default=3)
    p.add_argument("--start-stage", type=int, choices=[1, 2, 3, 4], default=1)
    p.add_argument("--init-params", type=Path, default=None)
    p.add_argument("--resume-state", type=Path, default=None)
    p.add_argument("--stop-file", type=Path, default=None,
                   help="Cooperative stop sentinel checked after completed steps")
    p.add_argument("--advance-file", type=Path, default=None,
                   help="Cooperative stage-advance sentinel; checkpoint then continue")
    p.add_argument("--steps-per-stage", default="30,50,80")
    p.add_argument("--lr-per-stage", default="1e-2,3e-4,1e-4")
    p.add_argument("--n-frames-per-stage", default=None)
    p.add_argument("--epsf-lr-scale", type=float, default=0.4)
    p.add_argument("--stage2-freeze-wcs-steps", type=int, default=20)
    p.add_argument(
        "--freeze-epsf-modes", action="store_true",
        help="stage 3: label epsf_modes 'frozen' (optax.set_to_zero(), same "
             "mechanism as --stage2-freeze-wcs-steps uses for wcs_coeff) so its "
             "SHAPE does not keep training. w_coeff (the per-frame amplitude of "
             "that shape) still trains normally in its own bucket. Default off.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--early-stop-tol", "--early-stop-tol-fine", dest="early_stop_tol", type=float, default=1e-5)
    p.add_argument("--early-stop-tol-coarse", type=float, default=1e-4)
    p.add_argument("--stage1-core-stamp", type=int, default=7)
    p.add_argument("--no-stage1-dx-only", action="store_true")
    p.add_argument("--group-frac", type=float, default=1.0)
    p.add_argument("--frame-frac", type=float, default=1.0)
    p.add_argument("--stamp-chunk", type=int, default=0)
    p.add_argument(
        "--frame-block", type=int, default=0,
        help="Split the frame axis into blocks of this many frames and accumulate "
             "gradients across them (exact; see fit.make_accum_step_fn). Bounds peak "
             "memory by block size instead of total frames. 0 disables.",
    )
    p.add_argument(
        "--gpu-flux-frame-block", type=int, default=8,
        help="Frame block for the final GPU solved-flux export.  A positive bounded "
             "value is required so a graceful STOP can export light curves without "
             "materializing the full orbit on VRAM (default: 8).",
    )
    p.add_argument("--fit-radius-min", type=float, default=0.0)
    p.add_argument("--recenter-n-iter", type=int, default=1)
    p.add_argument("--huber-delta", type=float, default=3.0)
    p.add_argument("--lambda-pixel-lap", type=float, default=1e-2)
    p.add_argument("--lambda-lap", type=float, default=1e-3,
                    help="Adjacent-node P_base smoothness (LossWeights.lambda_lap)")
    p.add_argument("--lambda-fine-nbr", type=float, default=0.0,
                    help="Adjacent-node smoothness of the fine (<~half-pixel) ePSF part only "
                         "(LossWeights.lambda_fine_nbr); smooth field variation stays free")
    p.add_argument("--fine-nbr-sigma", type=float, default=0.0,
                    help="Same coupling as a Gaussian prior of this width (flux fraction per "
                         "node-grid sample), normalised by the pooled likelihood denominator, "
                         "so it does not need re-tuning per scene. Use with "
                         "--support-size-weight-power 1; excludes --lambda-fine-nbr")
    p.add_argument("--fine-nbr-mode", choices=("moment_blind", "moment_blind_pair", "plain"), default="moment_blind",  # = loss.FINE_NBR_MODES
                    help="coupling used by --lambda-fine-nbr/--fine-nbr-sigma: moment_blind (default) "
                         "ignores node-to-node flux/shift/width differences; plain biases core width")
    p.add_argument("--lambda-smooth-wcs", type=float, default=1e-4)
    p.add_argument("--lambda-smooth-w", type=float, default=1e-3)
    p.add_argument("--flux-objective", choices=["l2", "huber-irls"], default="l2")
    p.add_argument("--huber-irls-iters", type=int, default=2)
    p.add_argument("--support-size-weight-power", type=float, default=0.0)
    p.add_argument(
        "--stamp-pedestal", action="store_true",
        help="Task M4: solve one additive per-group-per-frame pedestal jointly "
             "with flux (K+1-unknown weighted LS). Default off, bit-identical.",
    )
    p.add_argument(
        "--profile-w", action="store_true",
        help="Task PW: profile the temporal ePSF mode amplitudes w_k(t) out in closed form, per frame, jointly with the per-stamp fluxes (and pedestal), instead of training the w_coeff spline. Freezes w_coeff (it leaves the model entirely) and costs (n_modes+1)x the render. Default off, bit-identical.",
    )
    p.add_argument("--profile-w-iters", type=int, default=2,
                   help="Gauss-Newton iterations of the joint flux/amplitude solve.")
    p.add_argument("--lambda-centroid", type=float, default=1000.0)
    p.add_argument("--centroid-in-grad", action="store_true")
    p.add_argument("--reject-every", type=int, default=50)
    p.add_argument("--reject-burn-in", type=int, default=50)
    p.add_argument("--reject-n-sigma", type=float, default=3.0)
    p.add_argument(
        "--reject-mode", choices=["hysteresis", "audit", "standardized", "legacy", "two-level", "static"],
        default="hysteresis",
    )
    p.add_argument("--reject-tau-drop", type=float, default=3.5)
    p.add_argument("--reject-tau-keep", type=float, default=2.2)
    p.add_argument("--reject-max-churn", type=float, default=0.005)
    p.add_argument("--l2-churn-cap-frac", type=float, default=None)
    p.add_argument("--l2-hysteresis-n", type=int, default=None)
    p.add_argument("--l2-no-freeze-scale", action="store_true")
    p.add_argument("--l2-scale-ema-alpha", type=float, default=1.0)
    p.add_argument("--standardized-shrinkage-frames", type=float, default=20.0)
    p.add_argument("--standardized-min-weight", type=float, default=0.05)
    p.add_argument(
        "--chroma", action="store_true",
        help="enable the chromatic PSF term (per-star Gaia BP-RP colour drives a "
             "bilinear shift field plus a bilinear dilation field on the ePSF node "
             "grid; 12 params on a 2x2 grid, unfrozen at stage 3 only)",
    )
    p.add_argument(
        "--chroma-lr-scale", type=float, default=1.0,
        help="learning-rate multiplier for the chromatic leaves, relative to the "
             "stage learning rate",
    )
    p.add_argument(
        "--chroma-affine", action="store_true",
        help="C1: extend --chroma with per-node anisotropic-stretch and 45-degree-"
             "shear leaves (chroma_aniso/chroma_shear, +8 params on a 2x2 grid, same "
             "train_chroma bucket/lr, unfrozen at stage 3 only). Requires --chroma. "
             "Only needed to start these leaves from scratch (zero) with no "
             "checkpoint; resuming from a checkpoint that already carries them "
             "(e.g. --init-params params_warm_C1.npz) does not require this flag.",
    )
    p.add_argument(
        "--chroma-kurt", action="store_true",
        help="C1 optional: also add the flux-neutral kurtosis leaf chroma_kurt "
             "(+4 params on a 2x2 grid). Requires --chroma-affine.",
    )
    p.add_argument(
        "--chroma-halo", action="store_true",
        help="Add the additive chromatic HALO leaf chroma_halo: one amplitude per ePSF "
             "node (36 on a 6x6 grid) multiplying a FIXED r^-index radial profile. "
             "Unlike every other colour leaf this is not a reshaping of the base -- the "
             "measured colour-dependent excess follows r^-2 while the ePSF and every "
             "generator derived from it falls as r^-4, so no existing leaf can represent "
             "it. Requires --chroma. Identifiable only when the stamp support is wide "
             "enough for r^-index to look different from a flat per-stamp pedestal: see "
             "run_fit.py --irregular-enclose-pad-px and "
             "diagnostics/chroma_halo_profile.py.",
    )
    p.add_argument(
        "--chroma-halo-index", type=float, default=None,
        help="Radial index of the halo profile (default: epsf_model.CHROMA_HALO_INDEX "
             "= 2.0, the measured value). A physical constant, not a tuning knob -- set "
             "it from a profile measurement and it is recorded in the run meta.",
    )
    p.add_argument(
        "--chroma-halo-flux-neutral", action="store_true",
        help="Additionally impose the zero-sum gauge on the halo, i.e. make it a genuine "
             "core-to-wing TRANSFER at fixed total flux instead of an addition. Expected "
             "to fit WORSE: subtracting the mean drives the profile negative beyond ~3 px "
             "while the measured profile is positive out to 6 px, because an r^-2 halo's "
             "compensating deficit lies mostly outside the stamp. Exposed so that is a "
             "measurement, not an assertion.",
    )
    p.add_argument(
        "--chroma-halo-core-px", type=float, default=None,
        help="Radius inside which the halo profile is flattened, in physical px "
             "(default: epsf_model.CHROMA_HALO_CORE_PX = 1.0). Keeps the profile finite "
             "at the origin; the core carries little weight after the base-orthogonal "
             "gauge.",
    )
    p.add_argument(
        "--chroma-image", action="store_true",
        help="Add the free dP/dcolour IMAGE leaf: one global (G, G) image (3364 "
             "numbers at stamp_physical=13) carrying the chromatic PSF SHAPE change, "
             "gauged flux-neutral / base-orthogonal / core-dipole-free like an ePSF "
             "mode. Unfrozen at STAGE 4 only, i.e. after the parametric colour terms "
             "have converged, so what it holds is what they could not represent. "
             "The chromatic displacement stays in chroma_shift -- a dipole in this "
             "image is removed per star by the core-centroid gauge. Requires --chroma; "
             "pair with --lambda-chroma-lap and a 4-value --steps-per-stage.",
    )
    p.add_argument(
        "--chroma-image-lr-scale", type=float, default=None,
        help="learning-rate scale for the free colour image ONLY, relative to the "
             "stage lr, in its own optax bucket. Default: "
             "fit.CHROMA_IMAGE_LR_SCALE_DEFAULT (1e-2), resolved after import so "
             "argparse stays JAX-free. The image is in the ePSF's flux-fraction "
             "units (~4e-4) while the parametric colour leaves are px and a "
             "fractional scale; sharing their bucket at 1.0 took the loss from "
             "68.03 to 104.37 in one step.",
    )
    p.add_argument(
        "--stage4-freeze-others", action="store_true",
        help="Stage 4: hold wcs_coeff, epsf_base_raw and the parametric chroma leaves "
             "frozen while the free colour image trains. Run both ways -- frozen says "
             "what the parametric family missed, unfrozen says whether the free image "
             "just repaints the base.",
    )
    p.add_argument(
        "--lambda-chroma-lap", type=float, default=0.0,
        help="Pixel-Laplacian smoothness weight on the GAUGED free colour image "
             "(LossWeights.lambda_chroma_lap). 0 = off; 3364 free parameters on one "
             "frame will otherwise fit per-pixel noise.",
    )
    p.add_argument(
        "--w-lr-scale", type=float, default=None,
        help="learning-rate scale for w_coeff, relative to the stage lr. w(t) is "
             "~1e-5 in these units while epsf_base_raw is ~14; sharing the ePSF rate "
             "moved w_coeff by 5x its own value per step so it never converged "
             "(docs/TEMPORAL_RESIDUAL_ROOT_CAUSE_20260906.md). Default: "
             "fit.W_LR_SCALE_DEFAULT, resolved after import so argparse stays "
             "JAX-free.")
    p.add_argument(
        "--w-spatial", action="store_true",
        help="T3: give the temporal ePSF mode's amplitude w_k(t) a per-node "
             "field w_k(t, x, y) instead of one value shared by the whole "
             "field (leaf grows from (K, n_basis) to (K, n_rows, n_cols, "
             "n_basis), bilinear-blended to each star's position with the "
             "same node blend the ePSF base/modes already use). Default off, "
             "bit-identical to the pre-T3 global-w model. A fresh (no "
             "--init-params) run starts every node's curve at zero; resuming "
             "from a checkpoint that already carries a 4-D w_coeff (e.g. "
             "params_warm_T3.npz) requires this flag so the freshly-built "
             "params tree's shape matches it.",
    )
    p.add_argument("--fit-threads", type=int, default=None)
    p.add_argument("--jax-cache-dir", default=None)
    p.add_argument(
        "--allow-cpu", action="store_true",
        help="Permit CPU execution for local smoke tests; rented GPU runs fail closed by default",
    )
    return p.parse_args(argv)



def _append_run_provenance(out_dir: Path, bundle_path: Path, argv) -> None:
    """Append one JSON line to ``out_dir/run_provenance.jsonl``: which bundle (path +
    sha256), which code (git commit + dirty file count of the imported package), where
    and how this invocation ran. fit_meta.json records hyper-parameters but not these.
    Never raises -- provenance must not be able to kill a fit.
    """
    import socket
    import subprocess
    import sys

    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": socket.gethostname(),
           "argv": list(argv) if argv is not None else sys.argv[1:],
           "bundle_path": str(Path(bundle_path).resolve())}
    try:
        h = hashlib.sha256()
        with Path(bundle_path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        rec["bundle_sha256"] = h.hexdigest()
    except OSError as exc:
        rec["bundle_sha256"] = f"unreadable: {exc}"
    pkg = Path(__file__).resolve().parent
    rec["code_path"] = str(pkg)
    try:
        git = ["git", "-C", str(pkg)]
        rec["git_commit"] = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
        rec["git_branch"] = subprocess.run(git + ["rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(git + ["status", "--porcelain", "-uno", "--", "."], capture_output=True, text=True, timeout=30).stdout.splitlines()
        rec["git_dirty_files"] = [line[3:] for line in dirty]
    except Exception as exc:  # noqa: BLE001
        rec["git_commit"] = f"unavailable: {exc}"
    try:
        from . import epsf_model as _EM
        rec["epsf_repr"] = getattr(_EM, "EPSF_REPR", "subpixel_v0")
    except Exception:  # noqa: BLE001
        pass
    try:
        with (out_dir / "run_provenance.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass

def _triple(text: str, cast, flag: str) -> list:
    """Per-stage list: 3 or 4 values (stage 4 = the free colour image).

    A 3-value list is padded by repeating the last entry, so an existing
    3-value --lr-per-stage keeps working and a 3-value --steps-per-stage means
    "0 steps in stage 4" only because ``--stage`` still defaults to 3. Callers
    that must distinguish pass 4 values explicitly.
    """
    values = [cast(v) for v in text.split(",")]
    if len(values) not in (3, 4):
        raise SystemExit(f"{flag} must contain 3 or 4 comma-separated values")
    if len(values) == 3:
        values = values + [values[-1]]
    return values


def main(argv=None) -> None:
    args = parse_args(argv)
    # Configure the accelerator before importing anything that imports JAX.
    # XLA reads these values while the backend is initialized; setting them
    # through runtime.py afterward is too late.
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=1")
    cache_dir = args.jax_cache_dir or os.environ.get(
        "JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".syndiff" / "jax_cache")
    )
    # Namespaced per CPU capability set -- $HOME is shared NFS across nodes that do
    # NOT all share instruction sets, and a flat cache lets one node load another's
    # AOT executable ("...doesn't match the machine type for execution ... could
    # lead to execution errors such as SIGILL"). See runtime.configure_jax_cache.
    from . import runtime as _RT  # noqa: PLC0415 -- env-var only, no jax import
    cache_dir = str(Path(cache_dir) / _RT.cpu_cache_tag())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    if args.fit_threads is not None:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(key, str(args.fit_threads))

    import jax  # noqa: PLC0415
    from . import fit_bundle as FB  # noqa: PLC0415
    from . import runtime as RT  # noqa: PLC0415
    from . import train_loop as TL  # noqa: PLC0415

    if jax.default_backend() != "gpu" and not args.allow_cpu:
        raise SystemExit(
            f"GPU REQUIRED: JAX backend is {jax.default_backend()!r}; "
            "fix the CUDA/JAX installation or use --allow-cpu only for a local smoke test"
        )
    if args.group_frac != 1.0 or args.frame_frac != 1.0:
        raise SystemExit("minibatching is not implemented; use --group-frac 1 --frame-frac 1")
    if args.stamp_chunk < 0:
        raise SystemExit("--stamp-chunk must be >= 0")
    if args.gpu_flux_frame_block < 1:
        raise SystemExit("--gpu-flux-frame-block must be >= 1")
    if args.huber_irls_iters < 1:
        raise SystemExit("--huber-irls-iters must be >= 1")
    if args.support_size_weight_power < 0:
        raise SystemExit("--support-size-weight-power must be >= 0")
    if args.standardized_shrinkage_frames < 0:
        raise SystemExit("--standardized-shrinkage-frames must be >= 0")
    if not 0 < args.standardized_min_weight <= 1:
        raise SystemExit("--standardized-min-weight must be in (0, 1]")

    out_dir = args.out_dir or (
        _require_out_dir()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    thread_cfg = {
        "backend": jax.default_backend(),
        "device_count": str(len(jax.devices())),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        "JAX_COMPILATION_CACHE_DIR": cache_dir,
    }
    RT.write_thread_report(out_dir / "thread_env.txt", thread_cfg)
    _append_run_provenance(out_dir, args.from_bundle, argv)

    bundle = FB.load_fit_bundle(args.from_bundle)
    RT.log(
        f"loaded prepared bundle: G={bundle.n_groups} T={bundle.n_frames} "
        f"packed={bundle.is_packed} tiers={len(bundle.packed_tiers or ())}"
    )
    frames = (
        _triple(args.n_frames_per_stage, int, "--n-frames-per-stage")
        if args.n_frames_per_stage else None
    )
    fingerprint_config = {
        "steps_per_stage": _triple(args.steps_per_stage, int, "--steps-per-stage"),
        "lr_per_stage": _triple(args.lr_per_stage, float, "--lr-per-stage"),
        "epsf_lr_scale": args.epsf_lr_scale, "stage2_freeze_wcs_steps": args.stage2_freeze_wcs_steps,
        "grad_clip": args.grad_clip, "reject_every": args.reject_every,
        "stamp_chunk": args.stamp_chunk or None, "frame_block": args.frame_block or None,
        "gpu_flux_frame_block": args.gpu_flux_frame_block, "flux_objective": args.flux_objective,
        "huber_delta": args.huber_delta, "support_size_weight_power": args.support_size_weight_power,
    }
    digest = hashlib.sha256()
    with args.from_bundle.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    digest.update(json.dumps(fingerprint_config, sort_keys=True, separators=(",", ":")).encode())
    configuration_fingerprint = digest.hexdigest()
    # Resolved here, not in argparse: this module keeps its parser JAX-free so
    # --help and env setup run before the first JAX import.
    w_lr_scale = (TL.FIT.W_LR_SCALE_DEFAULT if args.w_lr_scale is None
                  else float(args.w_lr_scale))
    TL.run_stages_from_bundle(
        bundle,
        out_dir=out_dir,
        stage=args.stage,
        start_stage=args.start_stage,
        steps_per_stage=_triple(args.steps_per_stage, int, "--steps-per-stage"),
        lr_per_stage=_triple(args.lr_per_stage, float, "--lr-per-stage"),
        epsf_lr_scale=args.epsf_lr_scale,
        stage2_freeze_wcs_steps=args.stage2_freeze_wcs_steps,
        freeze_modes=bool(args.freeze_epsf_modes),
        grad_clip=args.grad_clip,
        chroma=bool(args.chroma),
        chroma_lr_scale=float(args.chroma_lr_scale),
        chroma_affine=bool(args.chroma_affine),
        chroma_kurt=bool(args.chroma_kurt),
        chroma_halo=bool(args.chroma_halo),
        chroma_halo_index=args.chroma_halo_index,
        chroma_halo_core_px=args.chroma_halo_core_px,
        chroma_halo_flux_neutral=bool(args.chroma_halo_flux_neutral),
        chroma_image=bool(args.chroma_image),
        chroma_image_lr_scale=(
            TL.FIT.CHROMA_IMAGE_LR_SCALE_DEFAULT if args.chroma_image_lr_scale is None
            else float(args.chroma_image_lr_scale)
        ),
        stage4_freeze_others=bool(args.stage4_freeze_others),
        lambda_chroma_lap=float(args.lambda_chroma_lap),
        w_lr_scale=w_lr_scale,
        w_spatial=bool(args.w_spatial),
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        reject_every=args.reject_every,
        reject_burn_in=args.reject_burn_in,
        reject_n_sigma=args.reject_n_sigma,
        reject_mode=args.reject_mode,
        reject_tau_drop=args.reject_tau_drop,
        reject_tau_keep=args.reject_tau_keep,
        reject_max_churn=args.reject_max_churn,
        l2_churn_cap_frac=args.l2_churn_cap_frac,
        l2_hysteresis_n=args.l2_hysteresis_n,
        l2_freeze_scale_after_first=not args.l2_no_freeze_scale,
        l2_scale_ema_alpha=args.l2_scale_ema_alpha,
        standardized_shrinkage_frames=args.standardized_shrinkage_frames,
        standardized_min_weight=args.standardized_min_weight,
        early_stop_patience=args.early_stop_patience,
        early_stop_tol=args.early_stop_tol,
        early_stop_tol_coarse=args.early_stop_tol_coarse,
        stage1_core_stamp=args.stage1_core_stamp,
        no_stage1_dx_only=args.no_stage1_dx_only,
        group_frac=args.group_frac,
        frame_frac=args.frame_frac,
        recenter_n_iter=args.recenter_n_iter,
        huber_delta=args.huber_delta,
        lambda_pixel_lap=args.lambda_pixel_lap,
        lambda_lap=args.lambda_lap,
        lambda_fine_nbr=args.lambda_fine_nbr,
        fine_nbr_sigma=args.fine_nbr_sigma,
        fine_nbr_mode=args.fine_nbr_mode,
        lambda_smooth_wcs=args.lambda_smooth_wcs,
        lambda_smooth_w=args.lambda_smooth_w,
        flux_objective=args.flux_objective,
        huber_irls_iters=args.huber_irls_iters,
        support_size_weight_power=args.support_size_weight_power,
        stamp_pedestal=bool(args.stamp_pedestal),
        profile_w=bool(args.profile_w),
        profile_w_iters=int(args.profile_w_iters),
        lambda_centroid=args.lambda_centroid,
        centroid_in_grad=args.centroid_in_grad,
        n_frames_per_stage=frames,
        init_params=args.init_params,
        resume_state=args.resume_state,
        configuration_fingerprint=configuration_fingerprint,
        stamp_chunk=args.stamp_chunk or None,
        frame_block=args.frame_block or None,
        gpu_flux_frame_block=args.gpu_flux_frame_block,
        fit_radius_min=args.fit_radius_min,
        stop_file=args.stop_file,
        advance_file=args.advance_file,
    )


if __name__ == "__main__":
    main()


def _require_out_dir():
    """Migration: the dev default wrote runs into the package directory (the /home checkout)."""
    raise SystemExit("pass --out-dir (run outputs belong under /astro/armin/koji/syndiff/, never /home)")
