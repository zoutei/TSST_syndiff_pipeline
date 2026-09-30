# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""CLI: fit the forward-modeled ePSF + Chebyshev/B-spline WCS on a region/frame window.

Training never opens hp_d/centroids when given ``--from-bundle`` (preferred).
Prep still lives here behind ``--workspace`` + ``--export-bundle`` (or use
``python -m syndiff_pipeline.forward_model.export_fit_bundle``).

Examples:

    # Prep once (cluster)
    python -m syndiff_pipeline.forward_model.export_fit_bundle \\
        --workspace data/data/s0020/c3/k3/diff_linear --sector 20 --orbit-index 1 \\
        --region 1536,1536,2048,2048 --n-frames 295 --tess-mag 7,10 \\
        --out-dir dev/forward_epsf_wcs/output/bundles/orbit1_half_mag710

    # Train (cluster or Colab) — no hp_d
    python -m syndiff_pipeline.forward_model.run_fit \\
        --from-bundle .../fit_bundle.npz --stage 2 --start-stage 2 \\
        --steps-per-stage 0,5,0 --log-every 1 --out-dir .../speed_bench
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import _bootstrap  # noqa: F401
from . import epsf_model as EM
from . import fit_bundle as FB
from . import loss as L
from . import runtime as RT
from . import train_loop as TL


def _log(msg: str) -> None:
    RT.log(msg)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-bundle", type=Path, default=None,
        help="Load prepared fit_bundle.npz and run Adam only (no hp_d/centroids/Gaia/PRF)",
    )
    p.add_argument(
        "--export-bundle", type=Path, default=None,
        help="After workspace prep, write fit_bundle.npz to this path/dir",
    )
    p.add_argument(
        "--export-only", action="store_true",
        help="With --workspace prep: write --export-bundle and exit (no Adam)",
    )
    p.add_argument("--workspace", type=Path, default=None)
    p.add_argument("--sector", type=int, default=None)
    p.add_argument("--camera", type=int, default=3)
    p.add_argument("--ccd", type=int, default=3)
    p.add_argument("--orbit-index", type=int, default=1)
    p.add_argument("--region", type=str, default=None, help="x_min,y_min,x_max,y_max")
    p.add_argument(
        "--temporal-wcs-root", type=Path, default=None,
        help="Published temporal Chebyshev WCS store.  When set, build the crop WCS "
             "and its initial coefficient tracks from a temporal-WCS grid, never centroids_r1.",
    )
    p.add_argument(
        "--temporal-wcs-grid-size", type=int, default=100,
        help="Samples per axis for --temporal-wcs-root crop fitting (default: 100).",
    )
    p.add_argument("--region-margin-px", type=float, default=8.0)
    p.add_argument("--epsf-grid", type=str, default="2x2", help="n_rows x n_cols")
    p.add_argument(
        "--epsf-grid-placement", type=str, default="center", choices=["center", "edge"],
        help="Node placement within the region: 'center' (default, unchanged) puts nodes "
             "at cell centers of an even n_rows x n_cols split; 'edge' puts the outermost "
             "nodes at the region bounds, with the rest evenly spaced between (see "
             "epsf_model.EpsfGridStatic.from_region docstring).",
    )
    p.add_argument(
        "--mode-init", type=str, default="iso_defocus",
        help="comma-separated subset of epsf_model.FD_MODE_NAMES; sets both the stage-3 "
             "mode count K and its FD init (default: single iso-defocus mode). "
             "Dipole modes x_smear/y_smear are rejected (joint WCS+ePSF null space).",
    )
    p.add_argument(
        "--init-epsf-base", type=Path, default=None,
        help="seed P_base from a previous run's converged epsf_base (a saved "
             "params*.npz with an 'epsf_base' array, same --epsf-grid/--stamp-physical) "
             "instead of resampling the TESS SPOC PRF. P_k are still (re-)derived from "
             "this base via --mode-init's finite differences. Independent of "
             "--init-params: this seeds only the base map on a fresh optimizer/WCS/modes "
             "init, not a full checkpoint resume.",
    )
    p.add_argument(
        "--init-wcs-coeff", type=Path, default=None,
        help="Load centroids_r1-derived wcs_coeff.npz (init-study warmstart) instead "
             "of re-fitting stage-0 warmstart from hp_d. When --init-study-stamp-reject "
             "is set and this is omitted, defaults to study_dir/wcs_warmstart/wcs_coeff.npz.",
    )
    p.add_argument("--cheb-degree", type=int, default=3)
    p.add_argument("--n-frames", type=int, default=100)
    p.add_argument(
        "--n-frames-per-stage", type=str, default=None,
        help="Comma list of frame counts for stages 1,2,3 (curriculum). "
             "Uses the middle N of the loaded window. Default: all stages use --n-frames",
    )
    p.add_argument("--frame-offset", choices=["start", "middle"], default="middle")
    p.add_argument(
        "--frame-stem", type=str, default=None,
        help="Select exactly the named FFI stem(s) (comma list) from the orbit "
             "instead of an --n-frames window. Overrides --n-frames/--frame-offset. "
             "The single-FFI static fit's way of naming which image it fitted.",
    )
    p.add_argument("--tess-mag", type=str, default="7,11",
                   help="Bright ePSF-primary magnitude interval (inclusive; default 7,11)")
    p.add_argument("--max-sep-px", type=float, default=7.0)
    p.add_argument("--max-group-size", type=int, default=4)
    p.add_argument(
        "--irregular-stamps", action=argparse.BooleanOptionalAction, default=True,
        help="Build packed irregular supports via irregular_stamps.build_epsf_support_stamps "
             "(segment membership + ePSF-sized pixel lists) instead of square mag-bin attach. "
             "Exports a packed FitBundle (pix_x/y/valid). Enabled by default; use "
             "--no-irregular-stamps only for legacy square-stamp experiments.",
    )
    p.add_argument(
        "--irregular-from-masks", type=Path, default=None,
        help="Skip detection; load mask_primary_*.npz from a segmentation export dir "
             "(member_indices/pix_x/pix_y must index the same expanded star table as this run).",
    )
    p.add_argument(
        "--min-sep-px", type=float, default=6.0,
        help="Isolation radius for irregular-stamp primaries (default 6)",
    )
    p.add_argument(
        "--faint-wcs-mag", type=str, default="11.0,13.0",
        help="Faint isolated WCS-anchor magnitude interval (lo,hi], default 11,13",
    )
    p.add_argument(
        "--max-stars", type=int, default=3000,
        help="Safety cap on total stars (bright primaries + faint WCS anchors), "
             "default 3000. Above this, the faintest faint-anchor candidates are "
             "dropped (bright irregular-group primaries/companions are never "
             "trimmed -- they keep full contamination treatment) until under the "
             "cap, tightening the effective faint_wcs_mag upper bound. Sized with "
             "headroom over the validated sector-20 reference (~2193 groups: 400 "
             "bright + 1793 faint), which dense CVZ fields can exceed by 2-3x at "
             "identical mag cuts. <= 0 disables the cap.",
    )
    p.add_argument(
        "--faint-stamp-size", type=int, default=7,
        help="Odd compact square footprint for faint WCS anchors (default 7; max 7)",
    )
    p.add_argument(
        "--faint-iso-radius-px", type=float, default=5.0,
        help="Strict Euclidean catalog-isolation radius for faint anchors (default 5)",
    )
    p.add_argument(
        "--irregular-n-sigma", type=float, default=5.0,
        help="|hp_d|/NOISE detection threshold for irregular stamps (default 5)",
    )
    p.add_argument(
        "--irregular-erode-px", type=int, default=1,
        help="Binary erosion iterations after detection (default 1)",
    )
    p.add_argument(
        "--stamp-min", type=int, default=5,
        help="Minimum odd enclosing square for irregular member stamps (default 5)",
    )
    p.add_argument(
        "--irregular-enclose-pad-px", type=int, default=0,
        help="Grow every irregular member square by this many pixels per side after "
             "the tight enclose, then clamp to --stamp-physical (13). Default 0 is "
             "bit-identical to the historical behaviour. The pad widens the PAINTED "
             "support, i.e. it adds genuinely fitted pixels, which is how a "
             "colour-dependent halo far flatter than the PSF (measured r^-1.4 vs the "
             "PSF's r^-4) becomes separable from a flat per-stamp pedestal: at r<3.5 px "
             "the two are nearly parallel. It does NOT change segmentation or group "
             "membership. Cost: more pixels per step, and fewer faint anchors survive "
             "the pixel-exclusion test against the widened bright supports.",
    )
    p.add_argument(
        "--p-tiers", type=str, default="64,128,256,512",
        help="Comma-separated P tiers for packed irregular supports (default 64,128,256,512)",
    )
    p.add_argument(
        "--k-tiers", type=str, default=None,
        help="Comma-separated K tiers to bucket groups into (see "
             "groups.bucket_groups_by_size) -- e.g. '1,2,4,8'. Removes the "
             "K^3 flux-solve padding waste a single global max_group_size pays "
             "for every group, including isolated 1-star ones, whenever any "
             "group anywhere in the region is crowded. Must cover the "
             "region's actual max group size or this errors. Default: no "
             "bucketing (single tier at the region's own max group size, "
             "identical to the pre-bucketing behavior) -- opt in explicitly. "
             "With --irregular-stamps default becomes 1,2,4,8.",
    )
    p.add_argument("--companion-mag-max", type=float, default=13.0)
    p.add_argument(
        "--stamp-physical", type=int, default=EM.STAMP_PHYSICAL,
        help="Odd fixed stamp size S for all groups (default 13); attach radii scale with S",
    )
    p.add_argument(
        "--stage1-core-stamp", type=int, default=L.STAGE1_CORE_STAMP,
        help="Odd central crop for stage-1 render/NLL (default 7); 0=use full --stamp-physical",
    )
    p.add_argument(
        "--no-stage1-dx-only", action="store_true",
        help="Disable stage-1 cached local ePSF (dx-only) fast path",
    )
    p.add_argument(
        "--group-frac", type=float, default=1.0,
        help="Per-step random group minibatch fraction (1=full batch)",
    )
    p.add_argument(
        "--frame-frac", type=float, default=1.0,
        help="Per-step random frame minibatch fraction (1=full batch)",
    )
    p.add_argument(
        "--stamp-chunk", type=int, default=0,
        help="Frames per lax.scan+jax.checkpoint block in the loss data-term "
             "(0/unset=disabled, current unchunked behaviour). Chunking makes "
             "peak fit memory scale with --stamp-chunk instead of the total "
             "frame count -- required for large group counts / full-CCD runs "
             "-- at the cost of ~1.3x more FLOPs (remat recomputes each "
             "block's forward pass on the backward pass). Bit-exact (to fp32 "
             "roundoff) vs. unchunked regardless of the value chosen, "
             "including block counts that don't evenly divide --n-frames. "
             "Not supported together with --group-frac/--frame-frac < 1.",
    )
    p.add_argument(
        "--fit-threads", type=int, default=None,
        help="Cap XLA/OMP threads (default: SYNDIFF_FIT_THREADS or ~1 NUMA node)",
    )
    p.add_argument(
        "--jax-cache-dir", type=str, default=None,
        help="Shared persistent JAX compilation cache dir (default: "
             "JAX_COMPILATION_CACHE_DIR env var, or ~/.syndiff/jax_cache). "
             "Point at shared NFS storage when running many similarly-shaped "
             "shard jobs so only the first pays JIT-compile cost.",
    )
    p.add_argument(
        "--io-workers", type=int, default=16,
        help="Thread pool size for hp_d + centroids preload",
    )
    p.add_argument(
        "--export-memory-gb", type=float, default=0.0,
        help="For export-only irregular bundles, cache the full cropped frame "
             "stack in RAM (threaded I/O) when >0; refuse if the measured "
             "estimate exceeds this limit. This avoids reopening FITS files "
             "for each tier and is faster than the bounded streaming path.",
    )
    p.add_argument(
        "--companion-radius-px", type=float, default=None,
        help="Flat L∞ attach radius override (debug). Default: mag-dependent "
             "r_attach(mag; S) with S=--stamp-physical",
    )
    p.add_argument(
        "--max-member-radius-px", type=float, default=0.0,
        help="Euclidean trim from stamp center; 0=off (default). Mag-dependent "
             "attach is preferred over this post-filter",
    )
    p.add_argument(
        "--static-basis", action="store_true",
        help="Time-independent model: frame_basis = ones((n_frames, 1)) for both "
             "the WCS coefficients and w_k, instead of a B-spline in time. This is "
             "the single-FFI fit (use with --n-frames 1 / --frame-stem). Every knot "
             "flag is meaningless under it and is rejected. Pair with --mode-init '' "
             "(K=0): with one frame the zero-temporal-mean gauge makes w_k identically "
             "zero, so the temporal ePSF modes are unidentifiable by construction.",
    )
    p.add_argument("--wcs-n-interior-knots", type=int, default=10)
    p.add_argument("--w-n-interior-knots", type=int, default=3)
    p.add_argument("--uniform-knots", action="store_true", default=True,
                   help="Uniform interior knots (default on for mid-orbit windows)")
    p.add_argument("--edge-densify-knots", action="store_true",
                   help="Use edge-densified knots instead of uniform")
    p.add_argument(
        "--knots-anchor",
        choices=["window", "full-orbit"],
        default="window",
        help="window: build knots on the selected-frame BTJD span (default). "
             "full-orbit: edge-densify on the full orbit, keep interiors with "
             "tau < n_fit/n_orbit, evaluate Phi on the selected frames only "
             "(requires --edge-densify-knots).",
    )
    p.add_argument(
        "--edge-frac", type=float, default=0.12,
        help="Edge band width in tau for edge-densified knots (default 0.12)",
    )
    p.add_argument(
        "--edge-interior-split", type=str, default=None,
        help="WCS knot band counts start,mid,end (e.g. 2,1,2); must sum to "
             "--wcs-n-interior-knots. w_k uses auto 1/3 split.",
    )
    p.add_argument("--stage", type=int, default=3, choices=[0, 1, 2, 3, 4])
    p.add_argument(
        "--start-stage", type=int, default=1, choices=[1, 2, 3],
        help="First optimization stage to run (skip earlier stages; use with --init-params)",
    )
    p.add_argument(
        "--init-params", type=Path, default=None,
        help="Load optimizer leaves from a params_stageN.npz before the stage loop",
    )
    p.add_argument("--steps-per-stage", type=str, default="30,50,80")
    p.add_argument("--lr-per-stage", type=str, default="1e-2,3e-4,1e-4")
    p.add_argument("--epsf-lr-scale", type=float, default=0.4)
    p.add_argument("--stage2-freeze-wcs-steps", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--recenter-n-iter", type=int, default=EM.HOTPATH_RECENTER_N_ITER,
        help="COM-recenter shift+renorm iterations on the per-slot render hot path "
             "(default 1; decode uses 2). More iterations reduce boundary flux loss "
             "for wing-heavy composites at extra compute cost.",
    )
    p.add_argument("--snr-cap-mag", type=float, default=L.SNR_CAP_MAG_DEFAULT)
    p.add_argument("--stamp-weight-floor", type=float, default=L.STAMP_WEIGHT_FLOOR)
    p.add_argument(
        "--fit-radius-tiers", type=str, default=None,
        help="Comma-separated 'mag_upper_bound:radius_px' tiers for the stage2+ "
             "NLL radius mask (loss.radius_pixel_mask), e.g. '9:6.0,11:3.5,inf:2.5'. "
             "A star gets the radius of the first tier whose mag bound it is below "
             "('inf' as the last bound covers everyone fainter than the previous "
             "tier). Default: loss.fit_radius_tiers_default(--stamp-physical) -- "
             "every tier covers the whole stamp (corner radius "
             "sqrt(2)*(stamp_physical//2)); a gradient-coverage diagnostic "
             "(dev_grad_coverage.py) found the old hardcoded '9:6.0,11:3.5,inf:2.5' "
             "gave epsf_base_raw structurally zero gradient beyond each tier's "
             "radius, for every node in the outer ePSF grid, for the mag9-11 "
             "majority population most of all -- see loss.fit_radius_tiers_default.",
    )
    p.add_argument(
        "--fit-radius-min", type=float, default=0.0,
        help="--from-bundle only: floor every group's already-baked "
             "fit_radius_stage1/23 at this value (np.maximum), without needing "
             "to re-export the bundle from --fit-radius-tiers. A bundle only "
             "stores per-group radii computed from the mag at export time, not "
             "the raw mag itself, so this is the knob for widening an "
             "already-exported bundle's radius for a quick test. Default 0.0 "
             "(no-op; use the bundle's stored radii unchanged).",
    )
    p.add_argument("--huber-delta", type=float, default=L.HUBER_DELTA_DEFAULT)
    p.add_argument(
        "--flux-objective", choices=["l2", "huber-irls"], default="l2",
        help="Analytic per-stamp flux objective. l2 preserves the existing weighted "
             "least-squares solve; huber-irls aligns flux profiling with the Huber "
             "pixel loss.",
    )
    p.add_argument(
        "--huber-irls-iters", type=int, default=2,
        help="Fixed differentiable IRLS iterations for --flux-objective huber-irls",
    )
    p.add_argument(
        "--support-size-weight-power", type=float, default=0.0,
        help="Multiply per-stamp weight by effective_pixel_count**power. "
             "Default 0 preserves equal-per-stamp weighting; use only for controlled ablations.",
    )
    p.add_argument(
        "--stamp-pedestal", action="store_true",
        help="Task M4: solve one additive per-group-per-frame pedestal jointly "
             "with flux (K+1-unknown weighted LS), shared by every K member of "
             "the group. Default off, bit-identical to pre-M4 behavior.",
    )
    p.add_argument(
        "--profile-w", action="store_true",
        help="Task PW: profile the temporal ePSF mode amplitudes w_k(t) out in closed form, per frame, jointly with the per-stamp fluxes (and pedestal), instead of training the w_coeff spline. Freezes w_coeff (it leaves the model entirely) and costs (n_modes+1)x the render. Default off, bit-identical.",
    )
    p.add_argument(
        "--profile-w-iters", type=int, default=2,
        help="Gauss-Newton iterations of the joint flux/amplitude solve (task PW).",
    )
    p.add_argument(
        "--w-spatial", action="store_true",
        help="T3: give the temporal ePSF mode's amplitude w_k(t) a per-node "
             "field w_k(t, x, y) instead of one value shared by the whole "
             "field. Default off, bit-identical to the pre-T3 global-w model.",
    )
    p.add_argument(
        "--lambda-centroid", type=float, default=1000.0,
        help="Centroid-penalty weight; target ratio_centroid~0.01-0.1 at init "
             "(legacy λ=1 gave ~1e-5 with per-stamp NLL)",
    )
    p.add_argument("--qc-min-frac", type=float, default=0.5,
                   help="Keep primaries that pass centroids_r1 QC on >= this fraction of frames")
    p.add_argument("--no-prefilter-qc", action="store_true",
                   help="Skip centroids_r1 QC pre-filter on primaries")
    p.add_argument("--reject-every", type=int, default=50,
                   help="Refresh residual stamp reject mask every N Adam steps (0=disable)")
    p.add_argument("--reject-n-sigma", type=float, default=3.0,
                   help="MAD n-sigma threshold for per-(group,frame) stamp rejection")
    p.add_argument(
        "--reject-mode",
        choices=["hysteresis", "audit", "standardized", "legacy", "two-level", "static"],
        default="hysteresis",
        help="Residual policy: audit reports scores but applies only physical "
             "masks; standardized applies per-group-shrunk continuous weights; "
             "legacy and two-level preserve historical hard gates; static is the "
             "single-FFI gate (brightness-detrended whole-star cut -- the others "
             "centre per group ACROSS frames and so reject nothing at 1 frame).",
    )
    p.add_argument("--reject-burn-in", type=int, default=50)
    p.add_argument("--reject-tau-drop", type=float, default=3.5)
    p.add_argument("--reject-tau-keep", type=float, default=2.2)
    p.add_argument("--reject-max-churn", type=float, default=0.005)
    p.add_argument(
        "--l2-churn-cap-frac", type=float, default=None,
        help="two-level only: max fraction of a bucket's active candidates that may "
             "flip committed reject-state per refresh (default: stamp_reject."
             "DEFAULT_L2_CHURN_CAP_FRAC)",
    )
    p.add_argument(
        "--l2-hysteresis-n", type=int, default=None,
        help="two-level only: consecutive same-direction refreshes required before a "
             "cell's committed reject-state flips (default: stamp_reject."
             "DEFAULT_L2_HYSTERESIS_N)",
    )
    p.add_argument(
        "--l2-no-freeze-scale", action="store_true",
        help="two-level only: recompute (EMA-blend via --l2-scale-ema-alpha) the "
             "level-2 MAD scale every refresh instead of freezing it after the first "
             "(default: frozen, to avoid the runaway-threshold failure mode measured "
             "on the legacy gate)",
    )
    p.add_argument(
        "--l2-scale-ema-alpha", type=float, default=1.0,
        help="two-level only, with --l2-no-freeze-scale: EMA blend weight for the "
             "raw per-refresh scale update (1.0=no smoothing)",
    )
    p.add_argument(
        "--standardized-shrinkage-frames", type=float, default=20.0,
        help="Prior effective-frame count shrinking per-group residual scales "
             "toward the pooled scale",
    )
    p.add_argument(
        "--standardized-min-weight", type=float, default=0.05,
        help="Positive floor for standardized residual weights; physical masks "
             "remain the only automatic hard zero",
    )
    p.add_argument(
        "--data-root", type=Path, default=None,
        help="SCC data_root for shared_mask + asteroid sidecars "
             "(default: infer from --workspace …/sSSSS/cC/kK/diff_*)",
    )
    p.add_argument(
        "--no-mask-reject", action="store_true",
        help="Disable TNS/asteroid stamp rejection (default: reject stamps "
             "that intersect bit 64 or per-FFI bit 128)",
    )
    p.add_argument(
        "--init-study-stamp-reject",
        type=Path,
        default=None,
        help="Init study config.yaml (or study dir): run/load Hotpants gridded-PSF "
             "stamp QA and AND per-(primary,frame) keep into bundle mask_active at export.",
    )
    p.add_argument(
        "--stamp-reject-workers",
        type=int,
        default=0,
        help="Parallel loky workers for --init-study-stamp-reject frame QA "
             "(0=auto: half CPU, cap 16)",
    )
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument(
        "--checkpoint-every", type=int, default=None,
        help="Overwrite params_latest.npz every N steps "
             "(default: same as --log-every; 0=disable during profiling)",
    )
    p.add_argument(
        "--early-stop-patience", type=int, default=None,
        help="Stop a stage/sub-phase once |loss change| stays below "
             "--early-stop-tol (relative) over this many consecutive --log-every "
             "checks, instead of always running its full step budget. "
             "Default: disabled (always run the fixed --steps-per-stage count).",
    )
    p.add_argument(
        "--early-stop-tol", "--early-stop-tol-fine", dest="early_stop_tol", type=float, default=1e-5,
        help="Relative loss-change tolerance for --early-stop-patience (default 1e-4)",
    )
    p.add_argument("--early-stop-tol-coarse", type=float, default=1e-4)
    p.add_argument(
        "--centroid-in-grad", action="store_true",
        help="Include soft centroid penalty in the Adam loss (default: off tape)",
    )
    p.add_argument("--prf-localdatadir", type=str, default=EM.PRF_ROOT_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def _group_brightest_primary_mag(
    groups: G.GroupSet,
    expanded_mags: np.ndarray,
    primary_index_set: set[int],
) -> np.ndarray:
    """Per-group tess_mag of the brightest primary member (fallback: any member)."""
    out = np.full(groups.n_groups, 12.0, dtype=float)
    for gi in range(groups.n_groups):
        idx = groups.members[gi][groups.valid[gi]]
        prim = [i for i in idx if int(i) in primary_index_set]
        use = prim if prim else list(idx)
        if use:
            out[gi] = float(np.min(expanded_mags[use]))
    return out


def infer_data_root_from_workspace(workspace: Path) -> Path | None:
    """Infer ``data_root`` from ``…/sSSSS/cC/kK/diff_*`` workspace path."""
    ws = Path(workspace).resolve()
    # parents: diff_lane, kK, cC, sSSSS, data_root
    if len(ws.parts) < 4:
        return None
    k_dir, c_dir, s_dir = ws.parent.name, ws.parent.parent.name, ws.parent.parent.parent.name
    if (
        k_dir.startswith("k")
        and c_dir.startswith("c")
        and s_dir.startswith("s")
        and ws.name.startswith("diff")
    ):
        return ws.parent.parent.parent.parent
    return None


def load_mask_catalog_for_fit(
    workspace: Path,
    *,
    data_root: Path | None,
    sector: int,
    camera: int,
    ccd: int,
):
    """Load MaskCatalog from SCC lane; return None on missing artifacts."""
    from syndiff_pipeline.difference_imaging.masking.ffi_mask import (
        load_catalog_for_scc_lane,
    )

    root = data_root if data_root is not None else infer_data_root_from_workspace(workspace)
    if root is None:
        _log(
            "  WARNING: could not infer --data-root from workspace; "
            "skipping TNS/asteroid stamp reject"
        )
        return None
    try:
        return load_catalog_for_scc_lane(
            workspace,
            data_root=root,
            sector=sector,
            camera=camera,
            ccd=ccd,
        )
    except FileNotFoundError as exc:
        _log(f"  WARNING: MaskCatalog unavailable ({exc}); skipping TNS/asteroid stamp reject")
        return None
    except Exception as exc:
        _log(f"  WARNING: MaskCatalog load failed ({exc}); skipping TNS/asteroid stamp reject")
        return None


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args(argv)
    if args.group_frac < 1.0 or args.frame_frac < 1.0:
        raise SystemExit(
            "--group-frac/--frame-frac minibatching is not implemented; previous "
            "single-bucket behavior silently ignored sampled indices. Use 1.0."
        )
    if args.group_frac > 1.0 or args.frame_frac > 1.0:
        raise SystemExit("--group-frac and --frame-frac must be <= 1.0")
    if args.huber_irls_iters < 1:
        raise SystemExit("--huber-irls-iters must be >= 1")
    if args.support_size_weight_power < 0:
        raise SystemExit("--support-size-weight-power must be >= 0")
    if args.standardized_shrinkage_frames < 0:
        raise SystemExit("--standardized-shrinkage-frames must be >= 0")
    if not 0 < args.standardized_min_weight <= 1:
        raise SystemExit("--standardized-min-weight must be in (0, 1]")
    if args.from_bundle is not None and args.workspace is not None:
        raise SystemExit("pass either --from-bundle or --workspace, not both")
    if args.static_basis:
        # A constant basis has no knots. Silently ignoring a knot flag here would
        # let a command line claim a knot layout the fit does not have.
        bad = [
            name for name, on in (
                ("--knots-anchor full-orbit", str(args.knots_anchor) == "full-orbit"),
                ("--edge-densify-knots", bool(args.edge_densify_knots)),
                ("--edge-interior-split", args.edge_interior_split is not None),
            ) if on
        ]
        if bad:
            raise SystemExit(
                f"--static-basis has no knots; remove {', '.join(bad)}"
            )
    if args.from_bundle is None and args.workspace is None:
        raise SystemExit("require --from-bundle PATH  or  --workspace (prep / export)")
    if args.export_only and args.export_bundle is None:
        raise SystemExit("--export-only requires --export-bundle")
    if args.from_bundle is None:
        if args.sector is None or args.region is None:
            raise SystemExit("--workspace prep requires --sector and --region")

    thread_cfg = RT.configure_cpu_threads(args.fit_threads)
    jax_cache_dir = RT.configure_jax_cache(args.jax_cache_dir)

    out_dir = args.out_dir or (_require_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    RT.write_thread_report(out_dir / "thread_env.txt", thread_cfg)
    _log(f"output -> {out_dir}")
    _log(f"CPU threads: effective={thread_cfg.get('effective_threads')} "
         f"(OMP={thread_cfg.get('OMP_NUM_THREADS')}, prefer_gpu={thread_cfg.get('prefer_gpu')})")
    _log(f"JAX compilation cache -> {jax_cache_dir}")

    if args.from_bundle is not None:
        bundle = FB.load_fit_bundle(args.from_bundle)
        _log(f"loaded fit bundle from {args.from_bundle} "
             f"(G={bundle.n_groups}, T={bundle.n_frames}, S={bundle.stamp_physical})")
        stages_steps = [int(v) for v in args.steps_per_stage.split(",")]
        stages_lr = [float(v) for v in args.lr_per_stage.split(",")]
        frames_per_stage = (
            [int(v) for v in args.n_frames_per_stage.split(",")]
            if args.n_frames_per_stage else None
        )
        TL.run_stages_from_bundle(
            bundle,
            out_dir=out_dir,
            stage=int(args.stage),
            start_stage=int(args.start_stage),
            steps_per_stage=stages_steps,
            lr_per_stage=stages_lr,
            epsf_lr_scale=float(args.epsf_lr_scale),
            stage2_freeze_wcs_steps=int(args.stage2_freeze_wcs_steps),
            grad_clip=float(args.grad_clip),
            log_every=int(args.log_every),
            checkpoint_every=args.checkpoint_every,
            reject_every=int(args.reject_every),
            reject_burn_in=int(args.reject_burn_in),
            reject_n_sigma=float(args.reject_n_sigma),
            reject_mode=str(args.reject_mode),
            reject_tau_drop=float(args.reject_tau_drop),
            reject_tau_keep=float(args.reject_tau_keep),
            reject_max_churn=float(args.reject_max_churn),
            l2_churn_cap_frac=args.l2_churn_cap_frac,
            l2_hysteresis_n=args.l2_hysteresis_n,
            l2_freeze_scale_after_first=not bool(args.l2_no_freeze_scale),
            l2_scale_ema_alpha=float(args.l2_scale_ema_alpha),
            standardized_shrinkage_frames=float(args.standardized_shrinkage_frames),
            standardized_min_weight=float(args.standardized_min_weight),
            early_stop_patience=args.early_stop_patience,
            early_stop_tol=float(args.early_stop_tol),
            early_stop_tol_coarse=float(args.early_stop_tol_coarse),
            stage1_core_stamp=int(args.stage1_core_stamp),
            no_stage1_dx_only=bool(args.no_stage1_dx_only),
            group_frac=float(args.group_frac),
            frame_frac=float(args.frame_frac),
            recenter_n_iter=int(args.recenter_n_iter),
            huber_delta=float(args.huber_delta),
            flux_objective=str(args.flux_objective),
            huber_irls_iters=int(args.huber_irls_iters),
            support_size_weight_power=float(args.support_size_weight_power),
            stamp_pedestal=bool(args.stamp_pedestal),
            profile_w=bool(args.profile_w),
            profile_w_iters=int(args.profile_w_iters),
            w_spatial=bool(args.w_spatial),
            lambda_centroid=float(args.lambda_centroid),
            centroid_in_grad=bool(args.centroid_in_grad),
            n_frames_per_stage=frames_per_stage,
            init_params=args.init_params,
            stamp_chunk=int(args.stamp_chunk) or None,
            fit_radius_min=float(args.fit_radius_min),
        )
        return

    # ---- workspace prep path (hp_d / centroids / Gaia / PRF) ----
    from . import cheb_wcs as CW
    from . import gaia_pm as GP
    from . import fit as FIT
    from . import groups as G
    from . import irregular_stamps as IS
    from . import packed_support as PS
    from . import stamp_reject as SR
    from . import temporal as T
    from .data import (
        RegionSpec,
        companion_candidates_by_mag,
        filter_stars_by_xy,
        list_orbit_frames,
        list_orbit_frames_from_temporal_wcs,
        load_frame_region,
        load_region_stack,
        merge_star_tables,
        preload_merged_stars,
        primary_candidates_by_mag,
        primary_to_expanded_index_map,
        select_middle_frames,
        xy_in_region_mask,
    )
    from .shared_wcs import fit_region_shared_wcs
    from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import load_gaia_catalog  # noqa: E402

    use_irregular = bool(args.irregular_stamps) or args.irregular_from_masks is not None
    if args.irregular_from_masks is not None and not args.irregular_stamps:
        # Loading prebuilt masks implies packed irregular path.
        use_irregular = True
    if int(args.stamp_min) < 1 or int(args.stamp_min) % 2 == 0:
        raise SystemExit(f"--stamp-min must be an odd integer >= 1, got {args.stamp_min}")
    if int(args.stamp_min) > int(args.stamp_physical):
        raise SystemExit("--stamp-min cannot exceed --stamp-physical")

    if int(args.stamp_physical) < 3 or int(args.stamp_physical) % 2 == 0:
        raise SystemExit(f"--stamp-physical must be an odd integer >= 3, got {args.stamp_physical}")
    stage1_core = int(args.stage1_core_stamp)
    if stage1_core < 0 or (stage1_core > 0 and stage1_core % 2 == 0):
        raise SystemExit(f"--stage1-core-stamp must be 0 or odd positive, got {stage1_core}")
    stamp_physical = int(args.stamp_physical)
    ws = args.workspace
    region = RegionSpec.parse(args.region)
    margin = args.region_margin_px
    # Clamp the lower bound at 0: load_frame_region slices the loaded array with
    # slice(region.y_min, region.y_max) / slice(region.x_min, region.x_max), and a
    # negative start wraps (Python slice semantics) instead of clipping -- e.g. for
    # a region touching the true CCD edge (x_min=0/y_min=0, as in a full-CCD run),
    # x_min - margin goes negative and silently loads a tiny wrong corner of the
    # array instead of the intended margin-padded crop, which then drops every
    # group as "off-array" (array_origin below is derived from this same value, so
    # it must match what's actually loaded). The upper bound needs no such clamp:
    # a slice stop past the array end already saturates correctly, and
    # ``array_shape`` downstream is read from the loaded array's true shape.
    region_margin = RegionSpec(
        max(0, int(region.x_min - margin)), max(0, int(region.y_min - margin)),
        int(region.x_max + margin), int(region.y_max + margin),
    )
    n_rows, n_cols = (int(v) for v in args.epsf_grid.lower().split("x"))
    mag_lo, mag_hi = (float(v) for v in args.tess_mag.split(","))
    faint_mag_lo, faint_mag_hi = (float(v) for v in args.faint_wcs_mag.split(","))
    if not (mag_lo <= mag_hi <= faint_mag_lo < faint_mag_hi <= 13.0):
        raise SystemExit(
            "require bright --tess-mag [lo,hi] ending at or below --faint-wcs-mag "
            "(lo,hi] with hi <= 13"
        )
    if int(args.faint_stamp_size) < 1 or int(args.faint_stamp_size) % 2 == 0 or int(args.faint_stamp_size) > 7:
        raise SystemExit("--faint-stamp-size must be an odd integer in [1, 7]")
    if float(args.faint_iso_radius_px) <= 0:
        raise SystemExit("--faint-iso-radius-px must be positive")
    if not use_irregular and faint_mag_hi > faint_mag_lo:
        raise SystemExit("faint WCS anchors require --irregular-stamps (the default)")
    stages_steps = [int(v) for v in args.steps_per_stage.split(",")]
    stages_lr = [float(v) for v in args.lr_per_stage.split(",")]
    uniform_knots = bool(args.uniform_knots) and not bool(args.edge_densify_knots)

    history_jsonl = out_dir / "history.jsonl"
    if int(args.start_stage) <= 1:
        history_jsonl.write_text("")  # truncate fresh run
    elif not history_jsonl.exists():
        history_jsonl.write_text("")

    _log("listing frames...")
    if args.temporal_wcs_root is not None:
        frames_all, (btjd0, btjd1, orbit_num) = list_orbit_frames_from_temporal_wcs(
            ws, args.temporal_wcs_root, sector=args.sector, orbit_index=args.orbit_index,
        )
        _log(f"listing frames from published temporal WCS: {args.temporal_wcs_root}")
    else:
        frames_all, (btjd0, btjd1, orbit_num) = list_orbit_frames(
            ws, sector=args.sector, orbit_index=args.orbit_index,
        )
    if args.frame_stem:
        wanted = [t.strip() for t in str(args.frame_stem).split(",") if t.strip()]
        by_stem = {f.stem: f for f in frames_all}
        missing = [t for t in wanted if t not in by_stem]
        if missing:
            raise SystemExit(
                f"--frame-stem: {missing} not in orbit {args.orbit_index} of sector "
                f"{args.sector} ({len(frames_all)} frames). Pass stems as they appear "
                f"in the temporal-WCS cadence table, e.g. {frames_all[0].stem}."
            )
        frames = [by_stem[t] for t in wanted]
        frames.sort(key=lambda f: f.btjd)
    elif args.frame_offset == "middle":
        frames = select_middle_frames(frames_all, args.n_frames)
    else:
        frames = frames_all[: args.n_frames]
    btjd = np.array([f.btjd for f in frames])
    if len(frames) < 2 and not args.static_basis:
        raise SystemExit(
            f"{len(frames)} frame(s) selected: a B-spline temporal basis needs at "
            "least 2. Pass --static-basis for the time-independent (single-FFI) "
            "model, together with --mode-init '' (K=0)."
        )
    _log(
        f"orbit {orbit_num}: using {len(frames)}/{len(frames_all)} frames, "
        f"selected btjd [{float(btjd[0]):.3f},{float(btjd[-1]):.3f}] "
        f"(orbit span [{btjd0:.3f},{btjd1:.3f}])"
    )

    gaia_full = load_gaia_catalog(ws)
    target_btjd = float(np.median(btjd))
    gaia_full, pm_stats = GP.apply_pm_to_dataframe(gaia_full, target_btjd)
    if pm_stats.get("applied"):
        _log(
            f"  Gaia PM propagated to mid-orbit btjd={target_btjd:.3f} "
            f"(Gaia epoch J{pm_stats['ref_epoch_jyear']:.1f}); "
            f"max shift {pm_stats['shift_arcsec_max']:.3f} arcsec, "
            f"median {pm_stats['shift_arcsec_median']:.4f} arcsec "
            f"({pm_stats['n_pm_finite']}/{pm_stats['n_rows']} with PM)"
        )
    else:
        _log(f"  Gaia PM not applied ({pm_stats.get('reason', 'unknown')})")
    temporal_grid_init = None
    if args.temporal_wcs_root is None:
        _log("fitting shared linear WCS...")
        wcs, region_qc = fit_region_shared_wcs(frames[0], gaia_full, region, margin_px=margin)
        _log(f"  {len(region_qc)} QC stars in region; CRVAL={wcs.wcs.crval}")
        cheb_static = CW.ChebWcsStatic.from_wcs(wcs, region, poly_degree=args.cheb_degree)

    wcs_edge_split: tuple[int, int, int] | None = None
    if args.edge_interior_split is not None:
        parts = [int(v) for v in str(args.edge_interior_split).split(",")]
        if len(parts) != 3 or any(p < 1 for p in parts):
            raise SystemExit("--edge-interior-split must be three positive integers, e.g. 2,1,2")
        if sum(parts) != int(args.wcs_n_interior_knots):
            raise SystemExit(
                f"--edge-interior-split {parts} must sum to "
                f"--wcs-n-interior-knots={args.wcs_n_interior_knots}"
            )
        wcs_edge_split = (parts[0], parts[1], parts[2])

    edge_frac = float(args.edge_frac)
    knots_anchor = str(args.knots_anchor)
    if args.static_basis:
        wcs_tb = T.build_static_basis(btjd)
        w_tb = T.build_static_basis(btjd)
        tau_cut = None
        _log(
            f"  static basis: wcs n_basis={wcs_tb.n_basis}, w n_basis={w_tb.n_basis} "
            f"(time-independent model over {len(frames)} frame(s))"
        )
    elif knots_anchor == "full-orbit":
        if uniform_knots:
            raise SystemExit("--knots-anchor full-orbit requires --edge-densify-knots")
        btjd_full = np.array([f.btjd for f in frames_all], dtype=float)
        tau_cut = float(len(frames) / len(frames_all))
        wcs_tb = T.build_temporal_basis_orbit_fraction(
            btjd, btjd_full,
            n_interior=args.wcs_n_interior_knots,
            edge_frac=edge_frac,
            edge_interior_split=wcs_edge_split,
            tau_cut=tau_cut,
        )
        w_tb = T.build_temporal_basis_orbit_fraction(
            btjd, btjd_full,
            n_interior=args.w_n_interior_knots,
            edge_frac=edge_frac,
            tau_cut=tau_cut,
        )
        wcs_kept = int(T.interior_knots(wcs_tb.knot_vector, wcs_tb.degree).size)
        w_kept = int(T.interior_knots(w_tb.knot_vector, w_tb.degree).size)
        _log(
            f"  temporal basis (full-orbit anchor): tau_cut={tau_cut:.4f}, "
            f"wcs n_interior_kept={wcs_kept} n_basis={wcs_tb.n_basis}, "
            f"w n_interior_kept={w_kept} n_basis={w_tb.n_basis}, "
            f"uniform_knots=False"
        )
    else:
        wcs_tb = T.build_temporal_basis(
            btjd, n_interior=args.wcs_n_interior_knots, uniform_knots=uniform_knots,
            edge_frac=edge_frac, edge_interior_split=wcs_edge_split,
        )
        w_tb = T.build_temporal_basis(
            btjd, n_interior=args.w_n_interior_knots, uniform_knots=uniform_knots,
            edge_frac=edge_frac,
        )
        tau_cut = None
        _log(
            f"  temporal basis: wcs n_basis={wcs_tb.n_basis}, w n_basis={w_tb.n_basis}, "
            f"uniform_knots={uniform_knots}"
        )

    if args.temporal_wcs_root is not None:
        from .temporal_wcs_init import build_temporal_grid_init

        _log(
            f"stage 0: fitting crop WCS from {args.temporal_wcs_grid_size}x"
            f"{args.temporal_wcs_grid_size} published temporal-WCS grid (no centroids_r1)..."
        )
        t0 = time.time()
        temporal_grid_init = build_temporal_grid_init(
            args.temporal_wcs_root, frames, region, np.asarray(wcs_tb.frame_basis),
            cheb_degree=args.cheb_degree, grid_size=args.temporal_wcs_grid_size,
        )
        wcs = temporal_grid_init.reference_wcs
        cheb_static = CW.ChebWcsStatic.from_wcs(wcs, region, poly_degree=args.cheb_degree)
        wcs_coeff_ws = temporal_grid_init.wcs_coeff
        _log(
            f"  temporal grid init done in {time.time()-t0:.1f}s; "
            f"median direct-grid RMS={temporal_grid_init.metadata['direct_grid_rms_px_median']:.4g} px, "
            f"max={temporal_grid_init.metadata['direct_grid_max_px']:.4g} px"
        )

    merged_by_stem: dict | None = None
    fit_stars_from_snapshot = False
    init_wcs_meta: dict = {"enabled": False}
    study_cfg_for_init = None

    if args.init_study_stamp_reject is not None:
        from syndiff_pipeline.forward_model.init_study.stamp_reject_export import resolve_study_config

        study_cfg_for_init = resolve_study_config(args.init_study_stamp_reject)

    init_wcs_path = args.init_wcs_coeff
    if init_wcs_path is None and study_cfg_for_init is not None:
        candidate = study_cfg_for_init.study_dir / "wcs_warmstart" / "wcs_coeff.npz"
        if candidate.is_file():
            init_wcs_path = candidate

    if temporal_grid_init is not None:
        init_wcs_meta = dict(temporal_grid_init.metadata)
    elif init_wcs_path is not None:
        from syndiff_pipeline.forward_model.init_study.init_wcs_export import (
            load_init_wcs_coeff,
            try_load_fit_stars_snapshot,
        )

        _log(f"stage 0: loading init WCS coefficients from {init_wcs_path}...")
        t0 = time.time()
        wcs_coeff_ws, init_wcs_meta = load_init_wcs_coeff(
            init_wcs_path, frames, wcs_tb, cheb_degree=args.cheb_degree,
        )
        init_wcs_meta["enabled"] = True
        init_wcs_meta["path"] = str(init_wcs_path)
        _log(
            f"  loaded wcs_coeff {tuple(wcs_coeff_ws.shape)} in {time.time()-t0:.1f}s "
            f"(skipped centroids_r1 warmstart fit)"
        )
        if study_cfg_for_init is not None:
            snap = try_load_fit_stars_snapshot(
                study_cfg_for_init.study_dir,
                frame_stems=[f.stem for f in frames],
                region=region,
                tess_mag=(mag_lo, mag_hi),
                qc_min_frac=float(args.qc_min_frac),
                no_prefilter_qc=bool(args.no_prefilter_qc),
            )
            if snap is not None:
                fit_stars = snap
                fit_stars_from_snapshot = True
                _log(f"  using cached fit_stars snapshot ({len(fit_stars)} primaries)")
    else:
        _log("stage 0: warm-start WCS coefficients from centroids_r1...")
        t0 = time.time()
        _log(f"  preloading centroids_r1 for {len(frames)} frames (workers={args.io_workers})...")
        merged_by_stem = preload_merged_stars(frames, gaia_full, n_workers=args.io_workers)
        _log(f"  centroids preload done in {time.time()-t0:.1f}s ({len(merged_by_stem)} tables)")
        t0 = time.time()
        wcs_coeff_ws = FIT.warmstart_wcs_coeff(
            frames, gaia_full, cheb_static, np.asarray(wcs_tb.frame_basis),
            merged_by_stem=merged_by_stem,
        )
        _log(f"  warm-start done in {time.time()-t0:.1f}s")

    ref_frame_index = len(frames) // 2

    prefilter_stats: dict = {"enabled": False}
    # The faint-anchor isolation catalogue is fixed by the difference-image
    # processing boundary, not by the legacy companion modelling knob.
    companion_cand = companion_candidates_by_mag(gaia_full, tess_mag_max=13.0)

    if not fit_stars_from_snapshot:
        # Mag-filter without catalog xy; membership uses reference-frame warmstart.
        primary_cand = primary_candidates_by_mag(gaia_full, tess_mag_range=(mag_lo, mag_hi))

        ra_pc = primary_cand["ra"].to_numpy(dtype=float)
        dec_pc = primary_cand["dec"].to_numpy(dtype=float)
        x_ws_pc, y_ws_pc = CW.eval_positions_at_frame_index(
            ra_pc, dec_pc, wcs_coeff_ws, cheb_static, wcs_tb.frame_basis, ref_frame_index,
        )
        catalog_primary_in = int(xy_in_region_mask(
            primary_cand["x"].to_numpy(dtype=float), primary_cand["y"].to_numpy(dtype=float),
            region, margin_px=margin,
        ).sum())
        gaia_fit, x_ws_fit, y_ws_fit = filter_stars_by_xy(
            primary_cand, x_ws_pc, y_ws_pc, region, margin_px=margin,
        )
        fit_stars = gaia_fit.loc[~gaia_fit.too_bright & ~gaia_fit.too_faint].reset_index(drop=True)
        _log(
            f"  primary membership (warmstart xy): {len(fit_stars)} in region "
            f"(catalog-xy would keep {catalog_primary_in} mag-pass candidates before QC)"
        )

        if args.temporal_wcs_root is not None:
            prefilter_stats = {
                "enabled": False,
                "skipped": "temporal-WCS initialization explicitly avoids centroids_r1",
            }
            _log("  skipping centroids_r1 QC prefilter (temporal-WCS mode)")
        elif not args.no_prefilter_qc:
            _log(
                f"pre-filtering {len(fit_stars)} primaries by centroids_r1 QC "
                f"(min_frac={args.qc_min_frac})..."
            )
            t0 = time.time()
            if merged_by_stem is None:
                _log(
                    f"  preloading centroids_r1 for QC prefilter "
                    f"(workers={args.io_workers})..."
                )
                merged_by_stem = preload_merged_stars(
                    frames, gaia_full, n_workers=args.io_workers,
                )
            fit_stars, prefilter_stats = SR.prefilter_primaries_by_centroids_qc(
                frames, fit_stars, gaia_full, min_frac=args.qc_min_frac,
                merged_by_stem=merged_by_stem,
            )
            prefilter_stats["enabled"] = True
            _log(
                f"  QC prefilter: {prefilter_stats['n_primary_in']} -> "
                f"{prefilter_stats['n_primary_kept']} kept "
                f"(never_seen={prefilter_stats['n_never_seen']}, "
                f"below_frac={prefilter_stats['n_below_frac']}) "
                f"in {time.time()-t0:.1f}s"
            )
            if len(fit_stars) == 0:
                raise RuntimeError("QC prefilter removed all primary stars; try --no-prefilter-qc")

        if study_cfg_for_init is not None and init_wcs_meta.get("enabled"):
            from syndiff_pipeline.forward_model.init_study.init_wcs_export import save_fit_stars_snapshot

            snap_path = save_fit_stars_snapshot(
                study_cfg_for_init.study_dir,
                fit_stars,
                frame_stems=[f.stem for f in frames],
                region=region,
                tess_mag=(mag_lo, mag_hi),
                qc_min_frac=float(args.qc_min_frac),
                no_prefilter_qc=bool(args.no_prefilter_qc),
            )
            _log(f"  saved fit_stars snapshot -> {snap_path}")

    ra_cc = companion_cand["ra"].to_numpy(dtype=float)
    dec_cc = companion_cand["dec"].to_numpy(dtype=float)
    x_ws_cc, y_ws_cc = CW.eval_positions_at_frame_index(
        ra_cc, dec_cc, wcs_coeff_ws, cheb_static, wcs_tb.frame_basis, ref_frame_index,
    )
    catalog_comp_in = int(xy_in_region_mask(
        companion_cand["x"].to_numpy(dtype=float), companion_cand["y"].to_numpy(dtype=float),
        region, margin_px=margin,
    ).sum())
    companion_pool, _, _ = filter_stars_by_xy(
        companion_cand, x_ws_cc, y_ws_cc, region, margin_px=margin,
    )
    expanded_stars = merge_star_tables(fit_stars, companion_pool)
    _log(
        f"  {len(fit_stars)} primary stars (tess_mag {mag_lo}-{mag_hi}), "
        f"{len(companion_pool)} catalog pool (tess_mag <= 13.0; "
        f"catalog-xy would keep {catalog_comp_in}), "
        f"{len(expanded_stars)} unique expanded"
    )

    ra = expanded_stars["ra"].to_numpy(dtype=float)
    dec = expanded_stars["dec"].to_numpy(dtype=float)
    mags = expanded_stars["tess_mag"].to_numpy(dtype=float)
    # Per-star Gaia colour for the chromatic term. The catalog carries the two
    # magnitudes, not the colour, and either can be missing for bright or blended
    # sources; NaN is kept here and turned into a zero colour offset at context
    # build, so a missing colour means "treat as the reference colour" rather than
    # poisoning the gradient.
    if {"phot_bp_mean_mag", "phot_rp_mean_mag"} <= set(expanded_stars.columns):
        _bp = expanded_stars["phot_bp_mean_mag"].to_numpy(dtype=float)
        _rp = expanded_stars["phot_rp_mean_mag"].to_numpy(dtype=float)
        bp_rp = np.where(np.isfinite(_bp) & np.isfinite(_rp), _bp - _rp, np.nan)
        _log(
            f"bp_rp: {int(np.isfinite(bp_rp).sum())}/{len(bp_rp)} stars have a colour, "
            f"median {np.nanmedian(bp_rp):.3f}"
            if np.any(np.isfinite(bp_rp)) else "bp_rp: no finite colours"
        )
    else:
        bp_rp = None
        _log("bp_rp: catalog lacks phot_bp/rp_mean_mag; chromatic term unavailable")
    x_ws, y_ws = CW.eval_positions_at_frame_index(
        ra, dec, wcs_coeff_ws, cheb_static, wcs_tb.frame_basis, ref_frame_index,
    )
    primary_remap = primary_to_expanded_index_map(fit_stars, expanded_stars)
    primary_index_set = set(int(i) for i in primary_remap)

    if use_irregular and args.export_memory_gb <= 0:
        # Full-orbit packed export must never retain 1,674 image crops in
        # RAM.  This sequence loads exactly one hp_d crop per access; packed
        # arrays themselves are disk-backed below.
        class _StreamingRegionFrames:
            def __len__(self):
                return len(frames)

            def __getitem__(self, index):
                if isinstance(index, slice):
                    return [self[i] for i in range(*index.indices(len(self)))]
                return load_frame_region(frames[int(index)], region_margin)

        frame_imgs = _StreamingRegionFrames()
        _log(f"streaming {len(frames)} hp_d frames one at a time (no image stack retained)")
    elif use_irregular:
        _log(
            f"caching {len(frames)} hp_d crops in RAM for fast export "
            f"(workers={args.io_workers}, limit={args.export_memory_gb:.1f} GB)..."
        )
        t0 = time.time()
        frame_imgs = load_region_stack(frames, region_margin, n_workers=args.io_workers)
        measured_gb = sum(
            (int(f.cal.nbytes) + int(f.noise.nbytes) + int(f.bad.nbytes))
            for f in frame_imgs
        ) / (1024 ** 3)
        _log(f"  cached frame stack: {measured_gb:.2f} GB in {time.time()-t0:.1f}s")
        if measured_gb > float(args.export_memory_gb):
            raise MemoryError(
                f"export frame cache requires {measured_gb:.2f} GB, exceeds "
                f"--export-memory-gb={args.export_memory_gb:.2f}"
            )
    else:
        _log(f"loading {len(frames)} hp_d frames (workers={args.io_workers})...")
        t0 = time.time()
        frame_imgs = load_region_stack(frames, region_margin, n_workers=args.io_workers)
        _log(f"  done in {time.time()-t0:.1f}s")

    irregular_meta: dict = {"enabled": False}
    is_epsf_contributor = None
    if use_irregular:
        p_tiers = tuple(int(v) for v in str(args.p_tiers).split(",") if v.strip())
        if args.k_tiers:
            k_tiers_irr = tuple(int(v) for v in args.k_tiers.split(",") if v.strip())
        else:
            k_tiers_irr = PS.DEFAULT_K_TIERS

        if args.irregular_from_masks is not None:
            _log(f"irregular stamps: loading masks from {args.irregular_from_masks}")
            irr_stamps = PS.irregular_stamps_from_mask_dir(args.irregular_from_masks)
            irregular_meta = {
                "enabled": True,
                "source": "mask_dir",
                "mask_dir": str(args.irregular_from_masks),
                "n_stamps": len(irr_stamps),
            }
        else:
            # Preserve the established irregular ePSF selection exactly:
            # bright primaries are screened against every catalog companion
            # through the legacy min-separation rule before segmentation.
            # Faint stars landing in an accepted irregular segment remain
            # joint-fit companions (and therefore ePSF contributors); only
            # standalone faint square anchors use the stop-gradient route.
            isolated = IS.select_isolated_primaries(
                x_ws, y_ws, mags,
                mag_lo=mag_lo, mag_hi=mag_hi,
                bright_mag_max=float(args.companion_mag_max),
                min_sep_px=float(args.min_sep_px),
                x_min=float(region.x_min), x_max=float(region.x_max),
                y_min=float(region.y_min), y_max=float(region.y_max),
                edge_margin_px=float(stamp_physical) / 2.0 + 1.0,
            )
            primary_indices = np.asarray(
                [int(i) for i in isolated if int(i) in primary_index_set],
                dtype=int,
            )
            _log(
                f"irregular stamps: {len(primary_indices)} isolated primaries "
                f"(min_sep={args.min_sep_px}, of {len(primary_index_set)} QC-kept)"
            )
            if len(primary_indices) == 0:
                raise RuntimeError("irregular stamps: no isolated primaries left")

            ref_img = frame_imgs[ref_frame_index]
            t_seg = time.time()
            assignments = IS.build_epsf_support_stamps(
                primary_indices, x_ws, y_ws, mags,
                cal=ref_img.cal, noise=ref_img.noise,
                stamp_physical=stamp_physical,
                stamp_min=int(args.stamp_min),
                # Preserve all legacy <13-mag joint-fit companions in the
                # irregular ePSF groups.  A faint source is WCS-only only
                # when it is accepted later as a standalone square anchor.
                bright_mag_max=float(args.companion_mag_max),
                max_group_size=int(args.max_group_size),
                region_x_min=int(region_margin.x_min),
                region_y_min=int(region_margin.y_min),
                n_sigma=float(args.irregular_n_sigma),
                erode_px=int(args.irregular_erode_px),
                enclose_pad_px=int(args.irregular_enclose_pad_px),
            )
            # Claim every scored bright-support pixel before considering faint
            # anchors.  This is a complete pixel-level exclusion, not merely
            # a center-distance heuristic.
            claimed_pixels = {
                (int(round(px)), int(round(py)))
                for assignment in assignments
                for px, py in zip(np.asarray(assignment.pix_x), np.asarray(assignment.pix_y))
            }
            faint_assignments, faint_anchor_stats = IS.select_isolated_faint_anchor_stamps(
                x_ws, y_ws, mags,
                faint_mag_range=(faint_mag_lo, faint_mag_hi),
                catalog_mag_max=13.0,
                isolation_radius_px=float(args.faint_iso_radius_px),
                stamp_size=int(args.faint_stamp_size),
                claimed_pixels=claimed_pixels,
                bounds=(
                    float(region_margin.x_min), float(region_margin.y_min),
                    float(region_margin.x_min + frame_imgs[0].cal.shape[1]),
                    float(region_margin.y_min + frame_imgs[0].cal.shape[0]),
                ),
            )
            n_bright = len(primary_indices)
            n_faint_before = len(faint_assignments)
            faint_assignments, cap_stats = IS.cap_faint_anchors_by_mag(
                faint_assignments, mags, n_bright=n_bright, max_stars=int(args.max_stars),
            )
            if cap_stats["capped"]:
                _log(
                    f"  --max-stars={args.max_stars}: {n_bright} bright + {n_faint_before} "
                    f"faint = {n_bright + n_faint_before} exceeds cap; dropped "
                    f"{cap_stats['dropped']} faintest faint-anchors, keeping "
                    f"{len(faint_assignments)} (effective faint_wcs_mag hi "
                    f"{faint_mag_hi:.2f} -> {cap_stats['effective_mag_hi']})"
                )
            faint_anchor_stats["retained"] = len(faint_assignments)
            faint_anchor_stats["capped"] = cap_stats["capped"]
            faint_anchor_stats["capped_dropped"] = cap_stats["dropped"]
            faint_anchor_stats["capped_effective_mag_hi"] = cap_stats["effective_mag_hi"]

            irr_stamps = PS.irregular_stamps_from_assignments(assignments + faint_assignments)
            n_mem = np.array([len(s.member_star_idx) for s in irr_stamps], dtype=int)
            n_pix = np.array([len(s.pix_x) for s in irr_stamps], dtype=int)
            n_ov = len(IS.find_overlapping_stamp_pairs(assignments))
            _log(
                f"  build_epsf_support_stamps: {len(irr_stamps)} stamps in "
                f"{time.time()-t_seg:.2f}s  "
                f"K=[{n_mem.min() if len(n_mem) else 0},{n_mem.max() if len(n_mem) else 0}]  "
                f"P=[{n_pix.min() if len(n_pix) else 0},{n_pix.max() if len(n_pix) else 0}]  "
                f"residual_overlap_pairs={n_ov}"
            )
            irregular_meta = {
                "enabled": True,
                "source": "build_epsf_support_stamps",
                "n_stamps": len(irr_stamps),
                # Keep the legacy keys/values so the bright-support
                # provenance remains directly comparable to the init-study
                # bundle.  The new fields below describe only appended faint
                # anchors.
                "n_isolated_primaries": int(len(primary_indices)),
                "min_sep_px": float(args.min_sep_px),
                "n_sigma": float(args.irregular_n_sigma),
                "erode_px": int(args.irregular_erode_px),
                "stamp_min": int(args.stamp_min),
                "k_min": int(n_mem.min()) if len(n_mem) else 0,
                "k_max": int(n_mem.max()) if len(n_mem) else 0,
                "p_min": int(n_pix.min()) if len(n_pix) else 0,
                "p_max": int(n_pix.max()) if len(n_pix) else 0,
                "n_bright_epsf_groups": int(len(assignments)),
                "n_faint_wcs_anchors": int(faint_anchor_stats["retained"]),
                "faint_anchor": {
                    "mag_range": [float(faint_mag_lo), float(faint_mag_hi)],
                    "stamp_size": int(args.faint_stamp_size),
                    "isolation_radius_px": float(args.faint_iso_radius_px),
                    **faint_anchor_stats,
                },
            }

        if len(irr_stamps) == 0:
            raise RuntimeError("irregular stamps: empty stamp list")

        # Packing pads to discrete P tiers; merged irregular supports can exceed
        # the CLI default max (512). Grow the ladder and freeze the effective
        # tiers into the bundle so --from-bundle rebucketing cannot undershoot.
        max_stamp_p = max((len(s.pix_x) for s in irr_stamps), default=0)
        max_stamp_k = max((len(s.member_star_idx) for s in irr_stamps), default=0)
        p_tiers_req = p_tiers
        k_tiers_req = k_tiers_irr
        p_tiers = PS.ensure_tiers_cover(max_stamp_p, p_tiers)
        k_tiers_irr = PS.ensure_tiers_cover(max_stamp_k, k_tiers_irr)
        if p_tiers != p_tiers_req or k_tiers_irr != k_tiers_req:
            _log(
                f"  tier ladder extended for max stamp K={max_stamp_k} P={max_stamp_p}: "
                f"k {list(k_tiers_req)}->{list(k_tiers_irr)}, "
                f"p {list(p_tiers_req)}->{list(p_tiers)}"
            )

        stamp_cx, stamp_cy = PS.stamp_centers_from_irregular(irr_stamps, x_ws, y_ws)
        t_pack = time.time()
        pack_scratch_dir = None
        if args.export_only and args.export_bundle is not None:
            export_target = Path(args.export_bundle)
            pack_scratch_dir = (
                export_target.parent / f".{export_target.stem}_packing_scratch"
                if export_target.suffix == ".npz"
                else export_target / ".packing_scratch"
            )
            _log(f"  packed gather uses disk-backed frame chunks -> {pack_scratch_dir}")

        def _pack_progress(bucket_i: int, done: int, total: int, pixel_cells: int) -> None:
            if done == total or done % 256 == 0:
                label = "single-pass" if bucket_i < 0 else f"tier {bucket_i}"
                _log(
                    f"  packed gather {label}: {done}/{total} frames "
                    f"({pixel_cells} group-pixels/frame)"
                )

        # Gather all packed tiers in one frame pass. The legacy helper reads
        # every FITS frame once per tier; this path opens each hp_d crop once
        # and scatters it into all tier-native outputs.
        gather_fn = (
            PS.build_packed_stamp_batches
            if args.export_memory_gb > 0
            else PS.build_packed_stamp_batches_single_pass
        )
        packed_batches = gather_fn(
            irr_stamps, frame_imgs,
            k_tiers=k_tiers_irr,
            p_tiers=p_tiers,
            array_origin=(region_margin.x_min, region_margin.y_min),
            n_stars=len(expanded_stars),
            x_ref=x_ws, y_ref=y_ws,
            drop_invalid_peak=True,
            scratch_dir=pack_scratch_dir,
            frame_chunk=32,
            progress=_pack_progress,
        )
        if not packed_batches:
            raise RuntimeError(
                "irregular stamps: all stamps dropped by peak-in-support check "
                "or empty after tier packing"
            )
        # Batches are sorted by (K,P), not input order.  Carry source-row
        # identity through that reordering before building the global flag.
        #
        # Do *not* call concat_packed_batches here.  That routine is a legacy
        # dense compatibility view which pads every native P tier to P_max;
        # for a full orbit this was a second multi-tens-of-GB allocation and
        # caused the local exporter to be OOM-killed.  Keep the tier-native
        # rows all the way through bundle serialization instead.
        packed_orig = np.concatenate([b.orig_stamp_idx for b in packed_batches])
        if args.irregular_from_masks is None:
            is_epsf_contributor = (packed_orig < len(assignments))
        else:
            is_epsf_contributor = np.ones(packed_orig.shape[0], dtype=bool)
        k_max = max(int(b.k_tier) for b in packed_batches)
        pmembers = np.full((packed_orig.size, k_max), -1, dtype=np.int32)
        pvalid = np.zeros((packed_orig.size, k_max), dtype=bool)
        row0 = 0
        for b in packed_batches:
            row1 = row0 + int(b.members.shape[0])
            pmembers[row0:row1, :b.k_tier] = b.members
            pvalid[row0:row1, :b.k_tier] = b.valid
            row0 = row1
        stamp_cx = np.asarray(stamp_cx, dtype=np.float32)[packed_orig]
        stamp_cy = np.asarray(stamp_cy, dtype=np.float32)[packed_orig]
        kept_mask = np.zeros((len(expanded_stars),), dtype=bool)
        kept_members = pmembers[pvalid]
        kept_members = kept_members[(kept_members >= 0) & (kept_members < len(expanded_stars))]
        kept_mask[kept_members] = True
        groups = G.GroupSet(
            n_groups=int(pmembers.shape[0]),
            max_group_size=int(k_max),
            members=pmembers,
            valid=pvalid,
            kept_star_mask=kept_mask,
            dropped_oversized=0,
        )
        if is_epsf_contributor.shape != (groups.n_groups,):
            raise RuntimeError("packed ePSF-contributor flags lost alignment with group rows")
        # Fake StampBatch-like namespace for the shared downstream code.
        class _PackedStampView:
            pass
        stamp_batch = _PackedStampView()
        stamp_batch.stamp_center_x = stamp_cx
        stamp_batch.stamp_center_y = stamp_cy
        packed_pix = True
        p_tiers_used = p_tiers
        augment_stats = {
            "companions_added": 0,
            "groups_grown": 0,
            "dropped_oversized": 0,
        }
        edge_stats = {"dropped_groups": 0, "trimmed_members": 0, "n_groups_kept": groups.n_groups}
        off_stats = {"dropped_groups": 0, "n_groups_kept": groups.n_groups}
        _log(
            f"  packed gather: {groups.n_groups} groups x {len(frames)} frames, "
            f"P<={max(int(b.p_tier) for b in packed_batches)}, K={k_max}, tiers={len(packed_batches)} "
            f"in {time.time()-t_pack:.2f}s"
        )
        irregular_meta["n_groups_packed"] = int(groups.n_groups)
        irregular_meta["p_max"] = int(max(int(b.p_tier) for b in packed_batches))
        irregular_meta["k_tiers"] = list(k_tiers_irr)
        irregular_meta["p_tiers"] = list(p_tiers)
    else:
        packed_pix = None
        p_tiers_used = ()
        groups = G.build_groups(
            x_ws_fit, y_ws_fit, max_sep_px=args.max_sep_px, max_group_size=args.max_group_size,
        )
        primary_members_by_group = [
            groups.members[gi][groups.valid[gi]].copy() for gi in range(groups.n_groups)
        ]
        _log(f"  {groups.n_groups} primary groups, sizes: {np.bincount(groups.valid.sum(axis=1))}")

        primary_members_expanded = [primary_remap[members] for members in primary_members_by_group]
        neighbor_indices = np.array(
            [i for i in range(len(expanded_stars)) if i not in primary_index_set],
            dtype=int,
        )
        t_augment = time.time()
        groups, augment_stats, stamp_cx, stamp_cy = G.augment_groups_with_stamp_neighbors(
            groups, x_ws, y_ws, neighbor_indices,
            primary_members_by_group=primary_members_expanded,
            neighbor_mags=mags,
            stamp_physical=stamp_physical,
            companion_radius_px=args.companion_radius_px,
            max_group_size=args.max_group_size,
            mags=mags,
        )
        dt_augment = time.time() - t_augment
        attach_desc = (
            f"flat r={args.companion_radius_px}"
            if args.companion_radius_px is not None
            else f"mag-dependent (S={stamp_physical})"
        )
        _log(
            f"  companions added: {augment_stats['companions_added']} ({attach_desc}), "
            f"groups grown: {augment_stats['groups_grown']}, "
            f"pruned oversized: {augment_stats['dropped_oversized']}, "
            f"max_group_size now {groups.max_group_size} "
            f"(augment_groups_with_stamp_neighbors: {dt_augment:.2f}s, "
            f"{len(neighbor_indices)} candidates x {groups.n_groups} groups -- "
            f"tiered cKDTree query_ball_point)"
        )

        groups, stamp_cx, stamp_cy, edge_stats = G.filter_groups_by_member_radius(
            groups, x_ws, y_ws, stamp_cx, stamp_cy,
            max_member_radius_px=args.max_member_radius_px,
        )
        if args.max_member_radius_px and args.max_member_radius_px > 0:
            _log(
                f"  edge filter: dropped_groups={edge_stats['dropped_groups']}, "
                f"trimmed_members={edge_stats['trimmed_members']}, "
                f"n_groups={edge_stats['n_groups_kept']}"
            )
        else:
            _log("  edge filter: off (--max-member-radius-px<=0)")
        _log(f"  {groups.n_groups} groups after filter, sizes: {np.bincount(groups.valid.sum(axis=1))}")

        groups, stamp_cx, stamp_cy, off_stats = G.drop_groups_off_array(
            groups, stamp_cx, stamp_cy,
            array_origin=(region_margin.x_min, region_margin.y_min),
            array_shape=frame_imgs[0].cal.shape,
            stamp=stamp_physical,
            n_stars=len(expanded_stars),
        )
        _log(
            f"  off-array stamp drop: dropped_groups={off_stats['dropped_groups']}, "
            f"n_groups_kept={off_stats['n_groups_kept']}"
        )
        if groups.n_groups == 0:
            raise RuntimeError("all groups dropped: stamps fall outside loaded region crop")

        t_extract = time.time()
        stamp_batch = G.extract_stamps(
            groups, frame_imgs, x_ws, y_ws,
            array_origin=(region_margin.x_min, region_margin.y_min),
            stamp=stamp_physical,
            stamp_center_x=stamp_cx, stamp_center_y=stamp_cy,
        )
        _log(
            f"  extract_stamps: {time.time()-t_extract:.2f}s "
            f"({groups.n_groups} groups x {len(frames)} frames)"
        )

    if groups.n_groups == 0:
        raise RuntimeError("no groups/stamps available for fit")
    if is_epsf_contributor is None:
        is_epsf_contributor = np.ones(groups.n_groups, dtype=bool)
    if np.asarray(is_epsf_contributor).shape != (groups.n_groups,):
        raise RuntimeError("is_epsf_contributor does not match final group axis")

    group_mag = _group_brightest_primary_mag(groups, mags, primary_index_set)
    stamp_w = L.soft_snr_stamp_weights(
        group_mag, snr_cap_mag=args.snr_cap_mag, w_min=args.stamp_weight_floor,
    )
    fit_radius_tiers = (
        L.parse_fit_radius_tiers(args.fit_radius_tiers) if args.fit_radius_tiers else None
    )
    r_stage1 = L.fit_radius_from_mag(
        group_mag, stage=1, tiers=fit_radius_tiers, stamp_physical=stamp_physical,
    )
    r_stage23 = L.fit_radius_from_mag(
        group_mag, stage=2, tiers=fit_radius_tiers, stamp_physical=stamp_physical,
    )
    _log(
        f"  stamp weights: min={stamp_w.min():.3f} med={np.median(stamp_w):.3f} max={stamp_w.max():.3f}; "
        f"r_fit stage1 med={np.median(r_stage1):.2f} stage2+ med={np.median(r_stage23):.2f}"
    )

    if use_irregular:
        n_dropped = int(sum(
            (np.asarray(batch.weight).sum(axis=(1, 2)) == 0).sum()
            for batch in packed_batches
        ))
        _log(
            f"  tier-native stamp batch: {groups.n_groups} groups x {len(frames)} frames, "
            f"{n_dropped} groups fully off-edge/masked"
        )
    else:
        n_dropped = int((stamp_batch.weight.sum(axis=(1, 2, 3)) == 0).sum())
        _log(f"  stamp batch {stamp_batch.data.shape}, {n_dropped} groups fully off-edge/masked")

    # Native packed tiers intentionally have no global dense stamp tensor.
    n_g, n_t = groups.n_groups, len(frames)
    mask_reject_stats: dict | None = None
    if bool(args.no_mask_reject):
        mask_active = np.ones((n_g, n_t), dtype=np.float32)
        _log("  TNS/asteroid stamp reject: disabled (--no-mask-reject)")
    else:
        catalog = load_mask_catalog_for_fit(
            ws,
            data_root=args.data_root,
            sector=args.sector,
            camera=args.camera,
            ccd=args.ccd,
        )
        if catalog is None:
            mask_active = np.ones((n_g, n_t), dtype=np.float32)
            mask_reject_stats = {"enabled": False, "reason": "catalog_unavailable"}
        else:
            btjds = [float(f.btjd) for f in frames]
            mask_active, mask_reject_stats = SR.build_tns_asteroid_stamp_active(
                catalog, btjds,
                stamp_batch.stamp_center_x, stamp_batch.stamp_center_y,
                stamp=stamp_physical,
            )
            mask_reject_stats["enabled"] = True
            mask_reject_stats["has_temporal"] = bool(catalog.has_temporal())
            _log(
                f"  TNS/asteroid stamp reject: kept={int(mask_active.sum())}/{mask_active.size} "
                f"(tns={mask_reject_stats['n_tns']}, asteroid={mask_reject_stats['n_asteroid']}, "
                f"either={mask_reject_stats['n_either']})"
            )
            if float(mask_reject_stats["frac_either"]) > SR.REJECT_WARN_FRAC:
                _log(
                    f"  WARNING: mask frac_either={mask_reject_stats['frac_either']:.1%} "
                    f"> {SR.REJECT_WARN_FRAC:.0%}"
                )

    init_study_stamp_stats: dict | None = None
    if args.init_study_stamp_reject is not None:
        from syndiff_pipeline.forward_model.init_study.stamp_reject_export import (
            DEFAULT_WORKERS as STAMP_REJECT_DEFAULT_WORKERS,
            apply_stamp_keep_mask,
            load_or_build_stamp_keep,
            resolve_study_config,
        )

        study_cfg = resolve_study_config(args.init_study_stamp_reject)
        n_stamp_workers = int(args.stamp_reject_workers)
        if n_stamp_workers <= 0:
            n_stamp_workers = STAMP_REJECT_DEFAULT_WORKERS
        _log(
            f"init-study stamp reject: building/loading keep table for "
            f"{len(fit_stars)} primaries x {len(frames)} frames (workers={n_stamp_workers})..."
        )
        t_stamp = time.time()
        keep_df = load_or_build_stamp_keep(
            study_cfg, frames, fit_stars, n_workers=n_stamp_workers,
        )
        mask_active, init_study_stamp_stats = apply_stamp_keep_mask(
            mask_active,
            keep_df,
            groups=groups,
            expanded_stars=expanded_stars,
            primary_index_set=primary_index_set,
            frames=frames,
        )
        init_study_stamp_stats["enabled"] = True
        init_study_stamp_stats["table_rows"] = int(len(keep_df))
        _log(
            f"  init-study stamp reject: clipped={init_study_stamp_stats['n_cells_clipped']} "
            f"cells, active {init_study_stamp_stats['n_cells_before']} -> "
            f"{init_study_stamp_stats['n_cells_after']} "
            f"({init_study_stamp_stats['frac_keep_after']:.1%} kept) "
            f"in {time.time()-t_stamp:.1f}s"
        )

    mode_names = tuple(name.strip() for name in args.mode_init.split(",") if name.strip())
    try:
        mode_names = EM.validate_mode_names(mode_names)
    except ValueError as exc:
        raise SystemExit(f"--mode-init: {exc}") from exc
    epsf_grid = EM.EpsfGridStatic.from_region(
        region, n_rows=n_rows, n_cols=n_cols, crop_origin=(44, 0),
        placement=args.epsf_grid_placement,
    )
    if args.init_epsf_base is not None:
        loaded_base = dict(np.load(args.init_epsf_base)).get("epsf_base")
        if loaded_base is None:
            raise SystemExit(
                f"--init-epsf-base {args.init_epsf_base} has no 'epsf_base' array "
                f"(expected a params*.npz written by fit.save_params_npz)"
            )
        # Legacy (pre-representation-change) checkpoints stored a 58-grid
        # epsf_base -- detect + convert (one warning) before the shape check
        # below, which compares against the current (pixel-integrated) size.
        loaded_base = np.asarray(
            EM.convert_legacy_epsf_array(loaded_base, name=f"{args.init_epsf_base} epsf_base")
        )
        _, node_size, _ = EM.node_geometry(stamp_physical)
        expected_shape = (n_rows, n_cols, node_size, node_size)
        if tuple(loaded_base.shape) != expected_shape:
            raise SystemExit(
                f"--init-epsf-base {args.init_epsf_base} epsf_base shape "
                f"{loaded_base.shape} != expected {expected_shape} "
                f"(--epsf-grid={n_rows}x{n_cols}, --stamp-physical={stamp_physical}); "
                f"seed checkpoint must match this run's grid/stamp geometry"
            )
        _log(f"  seeding epsf_base from {args.init_epsf_base} (skipping PRF init)")
        epsf_params0 = EM.init_epsf_from_base(loaded_base, mode_names=mode_names)
    else:
        epsf_params0 = EM.init_epsf_from_prf(
            camera=args.camera, ccd=args.ccd, sector=args.sector,
            grid=epsf_grid, localdatadir=args.prf_localdatadir,
            stamp_physical=stamp_physical, mode_names=mode_names,
        )

    if use_irregular:
        k_tiers = (
            tuple(int(v) for v in args.k_tiers.split(",") if v.strip())
            if args.k_tiers else PS.DEFAULT_K_TIERS
        )
        if max(k_tiers) < groups.max_group_size:
            raise SystemExit(
                f"--k-tiers={k_tiers} does not cover max_group_size={groups.max_group_size}"
            )
    else:
        k_tiers = (
            tuple(int(v) for v in args.k_tiers.split(",") if v.strip())
            if args.k_tiers else (groups.max_group_size,)
        )
    t_exp_sec = float(frame_imgs[0].exposure_days * 86400.0)
    params = L.init_params(
        cheb_static, epsf_params0, n_wcs_basis=wcs_tb.n_basis, n_w_basis=w_tb.n_basis,
        w_spatial=bool(args.w_spatial),
    )
    params["wcs_coeff"] = wcs_coeff_ws

    start_stage = int(args.start_stage)
    if start_stage > int(args.stage):
        raise SystemExit(f"--start-stage {start_stage} exceeds --stage {args.stage}")

    if args.init_params is not None:
        loaded_params = FIT.load_params_npz(args.init_params)
        n_modes_loaded = int(loaded_params["epsf_modes"].shape[0])
        n_modes_requested = len(mode_names)
        if n_modes_loaded != n_modes_requested:
            raise SystemExit(
                f"--init-params {args.init_params} has K={n_modes_loaded} modes but "
                f"--mode-init {args.mode_init!r} requests K={n_modes_requested}"
            )
        merged = dict(params)
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
                    f"--init-params {args.init_params} {key} shape "
                    f"{tuple(loaded_params[key].shape)} != current "
                    f"{tuple(merged[key].shape)}"
                )
            merged[key] = loaded_params[key]
        params = merged
        _log(f"  loaded init params from {args.init_params} (exact leaf match)")

    if args.n_frames_per_stage:
        frames_per_stage = [int(v) for v in args.n_frames_per_stage.split(",")]
    else:
        frames_per_stage = [len(frames)] * 3

    meta = {
        "workspace": str(ws), "sector": args.sector, "camera": args.camera, "ccd": args.ccd,
        "orbit_index": args.orbit_index, "orbit_num": orbit_num,
        "region": [region.x_min, region.y_min, region.x_max, region.y_max],
        "epsf_grid": [n_rows, n_cols], "epsf_grid_placement": args.epsf_grid_placement,
        "cheb_degree": args.cheb_degree,
        "n_frames": len(frames), "n_groups": groups.n_groups,
        "frame_offset": args.frame_offset,
        "frame_stem_request": args.frame_stem,
        "static_basis": bool(args.static_basis),
        "irregular_enclose_pad_px": int(args.irregular_enclose_pad_px),
        "irregular_n_sigma": float(args.irregular_n_sigma),
        "selected_frame_stems": [f.stem for f in frames],
        "selected_frame_btjd": [float(f.btjd) for f in frames],
        "tess_mag": [float(mag_lo), float(mag_hi)],
        "n_fit_stars": len(fit_stars), "n_expanded_stars": len(expanded_stars),
        "n_companion_pool": len(companion_pool),
        "companions_added": augment_stats["companions_added"],
        "groups_grown": augment_stats["groups_grown"],
        "companions_dropped_oversized": augment_stats["dropped_oversized"],
        "edge_filter": edge_stats,
        "companion_mag_max": args.companion_mag_max,
        "companion_radius_px": args.companion_radius_px,
        "stamp_physical": stamp_physical,
        "fit_radius_tiers": list(fit_radius_tiers) if fit_radius_tiers else list(
            L.fit_radius_tiers_default(stamp_physical)
        ),
        "stage1_core_stamp": stage1_core,
        "stage1_dx_only": not bool(args.no_stage1_dx_only),
        "n_frames_per_stage": frames_per_stage,
        "group_frac": args.group_frac,
        "frame_frac": args.frame_frac,
        "stamp_chunk": int(args.stamp_chunk) or None,
        "fit_threads": thread_cfg.get("effective_threads"),
        "max_member_radius_px": args.max_member_radius_px,
        "ref_frame_index": ref_frame_index,
        "gaia_pm_propagation": pm_stats,
        "max_group_size": groups.max_group_size,
        "n_dropped_edge_groups": n_dropped,
        "off_array_stamp_drop": off_stats,
        "membership": "warmstart_xy" if not use_irregular else "irregular_segment",
        "irregular_stamps": irregular_meta,
        "n_bright_epsf_groups": int(np.count_nonzero(is_epsf_contributor)),
        "n_faint_wcs_anchors": int(np.count_nonzero(~is_epsf_contributor)),
        "mode_names": list(mode_names),
        "init_epsf_base": str(args.init_epsf_base) if args.init_epsf_base is not None else None,
        "init_wcs_coeff": init_wcs_meta,
        "wcs_n_basis": wcs_tb.n_basis, "w_n_basis": w_tb.n_basis,
        "t_exp_sec": t_exp_sec,
        "uniform_knots": uniform_knots,
        "knots_anchor": knots_anchor,
        "tau_cut": tau_cut,
        "edge_frac": edge_frac,
        "wcs_edge_interior_split": list(wcs_edge_split) if wcs_edge_split else None,
        "wcs_n_interior_knots_request": int(args.wcs_n_interior_knots),
        "w_n_interior_knots_request": int(args.w_n_interior_knots),
        "snr_cap_mag": args.snr_cap_mag,
        "stamp_weight_floor": args.stamp_weight_floor,
        "huber_delta": args.huber_delta,
        "flux_objective": args.flux_objective,
        "huber_irls_iters": args.huber_irls_iters,
        "support_size_weight_power": args.support_size_weight_power,
        "stamp_pedestal": bool(args.stamp_pedestal),
        "profile_w": bool(args.profile_w),
        "profile_w_iters": int(args.profile_w_iters),
        "w_spatial": bool(args.w_spatial),
        "lambda_centroid": args.lambda_centroid,
        "centroid_in_grad": bool(args.centroid_in_grad),
        "qc_min_frac": args.qc_min_frac,
        "prefilter_qc": prefilter_stats,
        "reject_every": args.reject_every,
        "reject_n_sigma": args.reject_n_sigma,
        "reject_mode": args.reject_mode,
        "l2_churn_cap_frac": args.l2_churn_cap_frac,
        "l2_hysteresis_n": args.l2_hysteresis_n,
        "l2_freeze_scale_after_first": not bool(args.l2_no_freeze_scale),
        "l2_scale_ema_alpha": args.l2_scale_ema_alpha,
        "standardized_shrinkage_frames": args.standardized_shrinkage_frames,
        "standardized_min_weight": args.standardized_min_weight,
        "mask_reject": False if args.no_mask_reject else True,
        "mask_reject_stats": mask_reject_stats,
        "init_study_stamp_reject": str(args.init_study_stamp_reject)
        if args.init_study_stamp_reject is not None
        else None,
        "init_study_stamp_reject_stats": init_study_stamp_stats,
        "data_root": str(args.data_root) if args.data_root is not None else (
            str(infer_data_root_from_workspace(ws))
            if infer_data_root_from_workspace(ws) is not None else None
        ),
        "epsf_lr_scale": args.epsf_lr_scale,
        "stage2_freeze_wcs_steps": args.stage2_freeze_wcs_steps,
        "grad_clip": args.grad_clip,
        "stage": args.stage,
        "start_stage": start_stage,
        "init_params": str(args.init_params) if args.init_params else None,
        "steps_per_stage": stages_steps,
        "lr_per_stage": stages_lr,
        "log_every": args.log_every,
        "k_tiers": list(k_tiers),
    }

    x_lin_b, y_lin_b, cheb_basis_b = CW.star_basis(
        np.asarray(ra, dtype=np.float32),
        np.asarray(dec, dtype=np.float32),
        cheb_static,
    )

    bundle = FB.FitBundle(
        data=(None if use_irregular else np.asarray(stamp_batch.data, dtype=np.float32)),
        noise=(None if use_irregular else np.asarray(stamp_batch.noise, dtype=np.float32)),
        weight=(None if use_irregular else np.asarray(stamp_batch.weight, dtype=np.float32)),
        stamp_center_x=np.asarray(stamp_batch.stamp_center_x),
        stamp_center_y=np.asarray(stamp_batch.stamp_center_y),
        mask_active=np.asarray(mask_active, dtype=np.float32),
        ra=np.asarray(ra, dtype=np.float64),
        dec=np.asarray(dec, dtype=np.float64),
        x_lin=np.asarray(x_lin_b, dtype=np.float32),
        y_lin=np.asarray(y_lin_b, dtype=np.float32),
        cheb_basis=np.asarray(cheb_basis_b, dtype=np.float32),
        members=np.asarray(groups.members, dtype=np.int32),
        valid=np.asarray(groups.valid, dtype=bool),
        is_epsf_contributor=np.asarray(is_epsf_contributor, dtype=bool),
        bp_rp=(None if bp_rp is None else np.asarray(bp_rp, dtype=np.float64)),
        kept_star_mask=np.asarray(groups.kept_star_mask, dtype=bool),
        max_group_size=int(groups.max_group_size),
        stamp_snr_weight=np.asarray(stamp_w, dtype=np.float32),
        fit_radius_stage1=np.asarray(r_stage1, dtype=np.float32),
        fit_radius_stage23=np.asarray(r_stage23, dtype=np.float32),
        cheb_static=cheb_static,
        epsf_grid=epsf_grid,
        wcs_frame_basis=np.asarray(wcs_tb.frame_basis, dtype=np.float32),
        w_frame_basis=np.asarray(w_tb.frame_basis, dtype=np.float32),
        epsf_base=np.asarray(epsf_params0.base, dtype=np.float32),
        epsf_modes=np.asarray(epsf_params0.modes, dtype=np.float32),
        params0={k: np.asarray(v) for k, v in params.items()},
        t_exp_sec=float(t_exp_sec),
        stamp_physical=stamp_physical,
        k_tiers=k_tiers,
        meta=meta,
        packed_tiers=(PS.packed_tiers_from_batches(packed_batches) if use_irregular else None),
    )
    if packed_pix is not None:
        _log(
            f"  packed FitBundle: G={bundle.n_groups} T={bundle.n_frames} "
            f"P<={max(int(t.p_tier) for t in bundle.packed_tiers)} K={bundle.max_group_size}"
        )

    if args.export_bundle is not None:
        # Preserve the native K/P tiers on disk.  Slicing the globally padded
        # training view would recreate the large v1 representation at export.
        export_bundle = bundle
        bundle_path = FB.save_fit_bundle(args.export_bundle, export_bundle)
        _log(f"exported fit bundle -> {bundle_path} "
             f"({export_bundle.n_groups} groups x {export_bundle.n_frames} frames)")
        if args.export_only:
            _log("export-only: skipping Adam stages")
            return

    TL.run_stages_from_bundle(
        bundle,
        out_dir=out_dir,
        stage=int(args.stage),
        start_stage=start_stage,
        steps_per_stage=stages_steps,
        lr_per_stage=stages_lr,
        epsf_lr_scale=float(args.epsf_lr_scale),
        stage2_freeze_wcs_steps=int(args.stage2_freeze_wcs_steps),
        grad_clip=float(args.grad_clip),
        log_every=int(args.log_every),
        checkpoint_every=args.checkpoint_every,
        reject_every=int(args.reject_every),
        reject_n_sigma=float(args.reject_n_sigma),
        reject_mode=str(args.reject_mode),
        l2_churn_cap_frac=args.l2_churn_cap_frac,
        l2_hysteresis_n=args.l2_hysteresis_n,
        l2_freeze_scale_after_first=not bool(args.l2_no_freeze_scale),
        l2_scale_ema_alpha=float(args.l2_scale_ema_alpha),
        standardized_shrinkage_frames=float(args.standardized_shrinkage_frames),
        standardized_min_weight=float(args.standardized_min_weight),
        early_stop_patience=args.early_stop_patience,
        early_stop_tol=float(args.early_stop_tol),
        stage1_core_stamp=stage1_core,
        no_stage1_dx_only=bool(args.no_stage1_dx_only),
        group_frac=float(args.group_frac),
        frame_frac=float(args.frame_frac),
        recenter_n_iter=int(args.recenter_n_iter),
        huber_delta=float(args.huber_delta),
        flux_objective=str(args.flux_objective),
        huber_irls_iters=int(args.huber_irls_iters),
        support_size_weight_power=float(args.support_size_weight_power),
        stamp_pedestal=bool(args.stamp_pedestal),
        profile_w=bool(args.profile_w),
        profile_w_iters=int(args.profile_w_iters),
        w_spatial=bool(args.w_spatial),
        lambda_centroid=float(args.lambda_centroid),
        centroid_in_grad=bool(args.centroid_in_grad),
        n_frames_per_stage=frames_per_stage,
        init_params=None if args.init_params is None else args.init_params,
        stamp_chunk=int(args.stamp_chunk) or None,
    )


if __name__ == "__main__":
    main()


def _require_out_dir():
    """Migration: the dev default wrote runs into the package directory (the /home checkout)."""
    raise SystemExit("pass --out-dir (run outputs belong under /astro/armin/koji/syndiff/, never /home)")
