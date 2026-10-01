# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""B-spline temporal basis, lifted from dev/temporal_wcs_poly/temporal_model.py.

Cubic (degree 3), edge-densified knots — validated there to reach the
single-FFI residual floor on a whole TESS orbit (0.024/0.018 px on s20 orbit
1) while avoiding the monomial temporal basis's Runge-type blowup at orbit
boundaries. Every WCS coefficient and every ePSF ``w_k(t)`` curve in this
prototype shares this construction (only ``n_basis`` differs).

The knot/basis functions are imported directly from ``temporal_model.py``
(via ``_bootstrap``) rather than copied, so the corner-test basis is bit-for-bit
what the validated whole-orbit runs used.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from . import _bootstrap  # noqa: F401

# scipy BSpline + temporal_model are prep-only (build_temporal_basis*).
# second_difference_matrix is pure numpy and is all Adam needs from here.

__all__ = [
    "cap_spline_interior_knots",
    "interior_knots",
    "make_edge_weighted_knot_vector",
    "make_uniform_knot_vector",
    "n_spline_coeffs",
    "TemporalBasis",
    "build_temporal_basis",
    "build_static_basis",
    "build_temporal_basis_orbit_fraction",
    "build_temporal_basis_gap_aware",
    "second_difference_matrix",
    "override_w_frame_basis",
    "save_w_frame_basis_sidecar",
]


def cap_spline_interior_knots(*args, **kwargs):
    from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.temporal_model import cap_spline_interior_knots as _fn  # noqa: E402
    return _fn(*args, **kwargs)


def interior_knots(*args, **kwargs):
    from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.temporal_model import interior_knots as _fn  # noqa: E402
    return _fn(*args, **kwargs)


def make_edge_weighted_knot_vector(
    tau_min: float = 0.0,
    tau_max: float = 1.0,
    n_interior: int = 10,
    degree: int = 3,
    edge_frac: float = 0.12,
    *,
    edge_interior_split: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Edge-densified knot vector; optional explicit start/mid/end interior counts."""
    if edge_interior_split is None:
        from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.temporal_model import make_edge_weighted_knot_vector as _fn  # noqa: E402
        return _fn(
            tau_min=tau_min, tau_max=tau_max,
            n_interior=n_interior, degree=degree, edge_frac=edge_frac,
        )
    n_start, n_mid, n_end = (int(v) for v in edge_interior_split)
    if n_start < 1 or n_mid < 1 or n_end < 1:
        raise ValueError(
            f"edge_interior_split counts must be >= 1, got {edge_interior_split}"
        )
    if n_start + n_mid + n_end != int(n_interior):
        raise ValueError(
            f"edge_interior_split {edge_interior_split} must sum to n_interior={n_interior}"
        )
    edge_frac = float(np.clip(edge_frac, 0.02, 0.45))
    start_knots = np.linspace(tau_min, tau_min + edge_frac, n_start + 2)[1:-1]
    mid_knots = np.linspace(tau_min + edge_frac, tau_max - edge_frac, n_mid + 2)[1:-1]
    end_knots = np.linspace(tau_max - edge_frac, tau_max, n_end + 2)[1:-1]
    interior = np.sort(np.unique(np.concatenate([start_knots, mid_knots, end_knots])))
    return np.concatenate([
        np.full(degree + 1, tau_min),
        interior,
        np.full(degree + 1, tau_max),
    ])


def n_spline_coeffs(*args, **kwargs):
    from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.temporal_model import n_spline_coeffs as _fn  # noqa: E402
    return _fn(*args, **kwargs)


@dataclass(frozen=True)
class TemporalBasis:
    btjd_ref: float
    btjd_scale: float
    knot_vector: np.ndarray  # (n_knots,)
    degree: int
    frame_basis: jnp.ndarray  # (n_frames, n_basis), frozen constant

    @property
    def n_basis(self) -> int:
        return n_spline_coeffs(self.knot_vector, self.degree)


def _tau(btjd: np.ndarray, btjd_ref: float, btjd_scale: float) -> np.ndarray:
    t = (np.asarray(btjd, dtype=float) - btjd_ref) / btjd_scale
    return np.clip(t, 0.0, 1.0)


def make_uniform_knot_vector(
    tau_min: float = 0.0,
    tau_max: float = 1.0,
    n_interior: int = 10,
    degree: int = 3,
) -> np.ndarray:
    """Clamped knot vector with uniformly spaced interior knots (for mid-orbit subsets)."""
    n_interior = max(0, int(n_interior))
    if n_interior == 0:
        interior = np.array([], dtype=float)
    else:
        interior = np.linspace(tau_min, tau_max, n_interior + 2)[1:-1]
    return np.concatenate([
        np.full(degree + 1, tau_min),
        interior,
        np.full(degree + 1, tau_max),
    ])


def build_temporal_basis(
    btjd: np.ndarray,
    *,
    degree: int = 3,
    n_interior: int = 10,
    edge_frac: float = 0.12,
    edge_interior_split: tuple[int, int, int] | None = None,
    uniform_knots: bool = False,
) -> TemporalBasis:
    """Build the frozen (n_frames, n_basis) B-spline design matrix for ``btjd``.

    ``btjd_ref``/``btjd_scale`` are the first/last frame in the (assumed
    BTJD-sorted) input, matching ``fit_temporal_wcs.py``'s convention.
    ``n_interior`` is capped via ``cap_spline_interior_knots`` so the basis
    stays identifiable at small frame counts (e.g. the 100-frame corner test).

    Use ``uniform_knots=True`` for mid-orbit frame windows (avoids edge-densify
    ringing at the subset boundaries). Full-orbit fits keep edge densification.
    """
    from scipy.interpolate import BSpline

    btjd = np.asarray(btjd, dtype=float)
    if len(btjd) < 2:
        raise ValueError("need at least 2 frames to build a temporal basis")
    btjd_ref = float(btjd[0])
    btjd_scale = float(btjd[-1] - btjd[0])
    if btjd_scale <= 0.0:
        raise ValueError("btjd must be increasing")

    n_interior_capped = cap_spline_interior_knots(n_interior, len(btjd), degree)
    if uniform_knots:
        knot_vector = make_uniform_knot_vector(
            n_interior=n_interior_capped, degree=degree,
        )
    else:
        knot_vector = make_edge_weighted_knot_vector(
            n_interior=n_interior_capped, degree=degree, edge_frac=edge_frac,
            edge_interior_split=edge_interior_split,
        )
    tau = _tau(btjd, btjd_ref, btjd_scale)
    design = BSpline.design_matrix(tau, knot_vector, degree).toarray()
    return TemporalBasis(
        btjd_ref=btjd_ref,
        btjd_scale=btjd_scale,
        knot_vector=knot_vector,
        degree=degree,
        frame_basis=jnp.asarray(design, dtype=jnp.float32),
    )


def build_static_basis(btjd: np.ndarray) -> TemporalBasis:
    """Constant (time-independent) design matrix: ``frame_basis = ones((n, 1))``.

    The single-FFI / static fit.  Every WCS coefficient and every ``w_k`` becomes
    one number instead of a spline track, which is what "ignore the temporal
    change" means in this model.  Nothing else downstream needs a special case:
    ``n_basis = 1`` already flows through ``eval_all_positions``,
    ``w_field_from_coeff`` and the loss, and ``second_difference_matrix(1)``
    returns an empty operator so the spline smoothness penalties evaluate to a
    clean zero rather than erroring.

    ``n_frames >= 1`` is accepted on purpose.  With one frame this is the
    single-FFI fit; with several it is "one static model fitted jointly to N
    frames", which is how the single-frame noise floor is cross-checked.

    ``btjd_scale`` is 1.0, not the frame span: tau is never evaluated (the basis
    is constant), and a zero span -- the single-frame case -- would otherwise
    have to be special-cased or divide by zero.  ``btjd_ref`` keeps the first
    frame's BTJD so consumers that only want a reference epoch still get one.
    """
    btjd = np.asarray(btjd, dtype=float)
    if btjd.size < 1:
        raise ValueError("need at least 1 frame to build a static basis")
    return TemporalBasis(
        btjd_ref=float(btjd[0]),
        btjd_scale=1.0,
        knot_vector=np.array([0.0, 1.0], dtype=float),
        degree=0,
        frame_basis=jnp.ones((int(btjd.size), 1), dtype=jnp.float32),
    )


def build_temporal_basis_orbit_fraction(
    btjd_fit: np.ndarray,
    btjd_full: np.ndarray,
    *,
    degree: int = 3,
    n_interior: int = 10,
    edge_frac: float = 0.12,
    edge_interior_split: tuple[int, int, int] | None = None,
    tau_cut: float | None = None,
) -> TemporalBasis:
    """Edge-densified full-orbit knot profile, truncated to ``[0, tau_cut]``.

    Builds the validated full-orbit edge-weighted interiors on the full-orbit
    BTJD span, keeps only interiors with ``tau < tau_cut``, and clamps the
    B-spline on ``[0, tau_cut]``. ``frame_basis`` is evaluated for ``btjd_fit``
    only (typically the first fraction of the orbit).

    ``n_interior`` is the *full-orbit profile request*, not the final basis
    size. Default ``n_interior=10`` with ``tau_cut=0.5`` keeps 5 interiors →
    ``n_basis=9`` (vs 14 for the untruncated full orbit).

    If ``tau_cut`` is ``None``, it defaults to
    ``(btjd_fit[-1] - btjd_full[0]) / (btjd_full[-1] - btjd_full[0])``.
    """
    from scipy.interpolate import BSpline

    btjd_fit = np.asarray(btjd_fit, dtype=float)
    btjd_full = np.asarray(btjd_full, dtype=float)
    if len(btjd_fit) < 2:
        raise ValueError("need at least 2 fit frames to build a temporal basis")
    if len(btjd_full) < 2:
        raise ValueError("need at least 2 full-orbit frames for knot anchoring")
    btjd_ref = float(btjd_full[0])
    btjd_scale = float(btjd_full[-1] - btjd_full[0])
    if btjd_scale <= 0.0:
        raise ValueError("btjd_full must be increasing")

    if tau_cut is None:
        tau_cut = float((btjd_fit[-1] - btjd_ref) / btjd_scale)
    tau_cut = float(tau_cut)
    if not (0.0 < tau_cut <= 1.0):
        raise ValueError(f"tau_cut must be in (0, 1], got {tau_cut}")

    # Cap against full-orbit length so the *profile* stays identifiable if the
    # orbit itself is short; truncation then further reduces the basis.
    n_interior_capped = cap_spline_interior_knots(n_interior, len(btjd_full), degree)
    full_kv = make_edge_weighted_knot_vector(
        n_interior=n_interior_capped, degree=degree, edge_frac=edge_frac,
        edge_interior_split=edge_interior_split,
    )
    interiors = np.asarray(interior_knots(full_kv, degree), dtype=float)
    kept = np.unique(interiors[interiors < tau_cut])
    if kept.size < 1:
        raise ValueError(
            f"no full-orbit interiors fall below tau_cut={tau_cut}; "
            f"increase n_interior or tau_cut (interiors={interiors})"
        )

    knot_vector = np.concatenate([
        np.full(degree + 1, 0.0),
        kept,
        np.full(degree + 1, tau_cut),
    ])
    n_basis = int(n_spline_coeffs(knot_vector, degree))
    if len(btjd_fit) <= n_basis:
        raise ValueError(
            f"truncated basis not identifiable: n_frames={len(btjd_fit)} <= "
            f"n_basis={n_basis} (kept {kept.size} interiors below tau_cut={tau_cut})"
        )

    tau = np.clip((btjd_fit - btjd_ref) / btjd_scale, 0.0, tau_cut)
    design = BSpline.design_matrix(tau, knot_vector, degree).toarray()
    return TemporalBasis(
        btjd_ref=btjd_ref,
        btjd_scale=btjd_scale,
        knot_vector=knot_vector,
        degree=degree,
        frame_basis=jnp.asarray(design, dtype=jnp.float32),
    )


def build_temporal_basis_gap_aware(
    btjd: np.ndarray,
    *,
    degree: int = 3,
    dense_start_days: float = 1.5,
    n_dense_start: int = 5,
    n_mid: int = 4,
    n_end: int = 2,
    edge_frac: float = 0.08,
    gap_btjd: tuple[float, float] | None = None,
    gap_knot_multiplicity: int = 2,
) -> TemporalBasis:
    """Cubic B-spline knot layout with denser start knots and an optional
    interior knot cluster at a within-orbit gap (task T1-2, see
    ``docs/TEMPORAL_RESIDUAL_ROOT_CAUSE_20260906.md`` sec 6: a 7-knot
    edge-weighted spline cannot represent the fast first-day settling or the
    momentum-dump-gap step). Opt-in only -- ``build_temporal_basis``'s
    default construction and every existing caller are unchanged.

    Interior knots (in ``tau = (btjd - btjd_ref) / btjd_scale``), all
    concatenated and sorted (duplicates from the gap cluster are kept, not
    deduplicated -- that is what buys the extra local flexibility):

    - ``n_dense_start`` knots uniformly spaced over
      ``[0, dense_start_days / btjd_scale]`` (fast early-orbit settling).
    - ``n_mid`` knots uniformly spaced over
      ``[dense_start_days / btjd_scale, 1 - edge_frac]`` (baseline coverage
      spanning the gap, same role as the default construction's mid knots).
    - if ``gap_btjd`` is given, ``gap_knot_multiplicity`` *coincident*
      knots at ``tau_gap = mean(gap_btjd)`` mapped through the same
      ``btjd_ref``/``btjd_scale``. A repeated interior knot lowers the
      spline's continuity there (multiplicity ``m`` -> ``C^{degree-m}``),
      giving the two sides of the gap nearly independent coefficients
      without needing any data to actually fall inside it (there is
      none -- it is a genuine downlink gap). ``gap_knot_multiplicity``
      may go up to ``degree + 1`` (a full ``C^{-1}`` value discontinuity):
      unlike the usual reason to avoid that (a rank-deficient design
      matrix without data bracketing the knot on both sides), real frames
      exist right up to each edge of this gap, so the two independent
      cubic segments are each well-determined; measured condition number
      stays O(10) at every multiplicity 1-4 on the real S52 cadence.
      Default is 2 (partial, conservative) -- pass ``degree + 1`` for the
      full step this task's gap needs.
    - ``n_end`` knots densified over the tail ``[1 - edge_frac, 1]``, as
      in ``make_edge_weighted_knot_vector``.

    Raises if the resulting basis is not identifiable
    (``n_frames <= n_basis``), same convention as
    ``build_temporal_basis_orbit_fraction``.
    """
    btjd = np.asarray(btjd, dtype=float)
    if len(btjd) < 2:
        raise ValueError("need at least 2 frames to build a temporal basis")
    if not (1 <= int(gap_knot_multiplicity) <= degree + 1):
        raise ValueError(
            f"gap_knot_multiplicity must be in [1, degree+1={degree + 1}], "
            f"got {gap_knot_multiplicity}"
        )
    btjd_ref = float(btjd[0])
    btjd_scale = float(btjd[-1] - btjd[0])
    if btjd_scale <= 0.0:
        raise ValueError("btjd must be increasing")

    tau_dense_end = float(np.clip(dense_start_days / btjd_scale, 1e-6, 1.0 - edge_frac))
    tau_tail = 1.0 - float(edge_frac)
    if tau_tail <= tau_dense_end:
        raise ValueError(
            f"edge_frac={edge_frac} leaves no room after dense_start_days "
            f"({dense_start_days} d = tau {tau_dense_end:.4f})"
        )

    start_knots = np.linspace(0.0, tau_dense_end, int(n_dense_start) + 2)[1:-1]
    mid_knots = np.linspace(tau_dense_end, tau_tail, int(n_mid) + 2)[1:-1]
    end_knots = np.linspace(tau_tail, 1.0, int(n_end) + 2)[1:-1]

    pieces = [start_knots, mid_knots, end_knots]
    if gap_btjd is not None:
        gap_mean = 0.5 * (float(gap_btjd[0]) + float(gap_btjd[1]))
        tau_gap = float(np.clip((gap_mean - btjd_ref) / btjd_scale, tau_dense_end, tau_tail))
        pieces.append(np.full(int(gap_knot_multiplicity), tau_gap))

    interior = np.sort(np.concatenate(pieces))
    knot_vector = np.concatenate([
        np.full(degree + 1, 0.0),
        interior,
        np.full(degree + 1, 1.0),
    ])
    n_basis = int(n_spline_coeffs(knot_vector, degree))
    if len(btjd) <= n_basis:
        raise ValueError(
            f"basis not identifiable: n_frames={len(btjd)} <= n_basis={n_basis}"
        )

    from scipy.interpolate import BSpline

    tau = np.clip((btjd - btjd_ref) / btjd_scale, 0.0, 1.0)
    design = BSpline.design_matrix(tau, knot_vector, degree).toarray()
    return TemporalBasis(
        btjd_ref=btjd_ref,
        btjd_scale=btjd_scale,
        knot_vector=knot_vector,
        degree=degree,
        frame_basis=jnp.asarray(design, dtype=jnp.float32),
    )


def second_difference_matrix(n_basis: int) -> jnp.ndarray:
    """(n_basis-2, n_basis) second-difference operator for a smoothness penalty.

    Applied per coefficient row: ``smoothness = sum((D @ coeff_row) ** 2)``
    penalizes curvature of the fitted curve across spline coefficients, which
    is a reasonable proxy for "no non-physical high-frequency change" given
    the basis functions are already smooth and overlapping.
    """
    if n_basis < 3:
        return jnp.zeros((0, n_basis), dtype=jnp.float32)
    D = np.zeros((n_basis - 2, n_basis), dtype=np.float32)
    for i in range(n_basis - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return jnp.asarray(D)


def override_w_frame_basis(bundle, new_basis: np.ndarray):
    """Replace an already-loaded ``FitBundle``'s ``w_frame_basis`` in place
    (task T1-2's "sidecar / override option that replaces w_frame_basis at
    load time", implemented as a post-load monkeypatch rather than a
    ``fit_bundle.py``/``train_from_bundle.py`` edit -- both are under heavy
    concurrent edit by another session as of 2026-09-06 for an unrelated
    colour-affine change; every consumer of ``w_frame_basis`` already sizes
    itself from ``bundle.w_frame_basis.shape[1]`` (``train_loop.py``'s
    ``w_n_basis = int(bundle.w_frame_basis.shape[1])``,
    ``second_difference_matrix(w_n_basis)``, ``loss.zero_params``'s
    ``n_w_basis`` argument), so no other module needs to change for a wider
    basis to work -- see ``docs/T1_TEMPORAL_MODE_NOTES.md``).

    ``bundle`` is any object with a mutable ``w_frame_basis`` attribute whose
    first axis is the frame count (``FitBundle`` qualifies: it is a plain
    class, not a frozen dataclass). Returns ``bundle`` for chaining. Does
    NOT touch ``bundle.params0["w_coeff"]``; callers that resume training
    from ``bundle.params0`` rather than an explicit ``--init-params``
    checkpoint must also resize/re-zero that leaf to
    ``(n_modes, new_basis.shape[1])`` themselves -- this function only owns
    the frame-basis leaf, which is the one every deliverable here writes to
    a fresh checkpoint via ``--init-params`` instead.
    """
    new_basis = np.asarray(new_basis, dtype=np.float32)
    if new_basis.ndim != 2:
        raise ValueError(f"new_basis must be 2-D (T, n_w), got shape {new_basis.shape}")
    old_frames = int(np.asarray(bundle.w_frame_basis).shape[0])
    if new_basis.shape[0] != old_frames:
        raise ValueError(
            f"new_basis has {new_basis.shape[0]} frames, bundle has {old_frames}; "
            "the frame axis (and its BTJD ordering) must match exactly"
        )
    bundle.w_frame_basis = new_basis
    return bundle


def save_w_frame_basis_sidecar(path, basis: TemporalBasis) -> None:
    """Write a small sidecar npz (just the new ``w_frame_basis`` + provenance)
    rather than a copy of the 2GB bundle. Load with
    ``np.load(path)["w_frame_basis"]`` and pass to ``override_w_frame_basis``.
    """
    np.savez(
        path,
        w_frame_basis=np.asarray(basis.frame_basis, dtype=np.float32),
        btjd_ref=np.float64(basis.btjd_ref),
        btjd_scale=np.float64(basis.btjd_scale),
        knot_vector=np.asarray(basis.knot_vector, dtype=np.float64),
        degree=np.int64(basis.degree),
    )
