# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Batched differentiable flux solves (the profile-likelihood step).

Given the current WCS + ePSF parameters, the rendered per-member unit-flux
templates make the model linear in flux: ``model = sum_j f_j * template_j``.
For each (group, frame) we solve the small weighted normal equations
``(M^T W M + lambda I) f = M^T W d`` directly with ``jnp.linalg.solve``, which
JAX differentiates through cleanly -- so gradient descent only ever touches
WCS coefficients, ePSF grids, and the temporal ``w_k`` curves, never flux.

``weight`` is any per-pixel weight. Callers should pass inverse-variance
``iv = coverage / (NOISE**2 + eps)`` for a correct profile likelihood (this
prototype uses all-ones coverage; hp_d MASK is not used).

Templates may be square stamps ``(n_groups, K, n_frames, S, S)`` or packed
1D supports ``(n_groups, K, n_frames, P)``.
"""

from __future__ import annotations

import jax.numpy as jnp


FLUX_OBJECTIVES = ("l2", "huber_irls")


def _as_matrix(
    templates: jnp.ndarray,
    data: jnp.ndarray,
    weight: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Return ``(M, d, w, K)`` with ``M (n_groups, n_frames, P, K)``."""
    if templates.ndim == 5:
        n_groups, K, n_frames, S, _ = templates.shape
        P = S * S
        M = jnp.moveaxis(templates.reshape(n_groups, K, n_frames, P), 1, -1)
        w = weight.reshape(n_groups, n_frames, P)
        d = data.reshape(n_groups, n_frames, P)
    elif templates.ndim == 4:
        n_groups, K, n_frames, P = templates.shape
        M = jnp.moveaxis(templates, 1, -1)
        w = weight
        d = data
    else:
        raise ValueError(f"templates ndim must be 4 or 5, got {templates.ndim}")
    return M, d, w, K


def _normal_equations(
    templates: jnp.ndarray,
    data: jnp.ndarray,
    weight: jnp.ndarray,
    *,
    ridge: float = 1e-6,
    pedestal: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(MtWM, MtWd, M)`` for the batched flux solve.

    ``pedestal=True`` (task M4) augments the design with one extra constant
    column of ones over the footprint -- a single additive background level
    per group per frame, shared by all K members of the group (a per-stamp
    pedestal, not per-member) -- turning the K-unknown weighted LS into a
    K+1-unknown one solved by the same normal equations. Default False keeps
    ``M``/``K`` exactly as before (the concatenate never runs), so the
    default path is bit-identical to pre-M4 behavior.
    """
    M, d, w, K = _as_matrix(templates, data, weight)
    if pedestal:
        M = jnp.concatenate([M, jnp.ones_like(M[..., :1])], axis=-1)
        K = K + 1
    MtW = M * w[..., None]
    MtWM = jnp.einsum("gfpk,gfpl->gfkl", MtW, M)
    MtWd = jnp.einsum("gfpk,gfp->gfk", MtW, d)
    eye = jnp.eye(K, dtype=templates.dtype)
    MtWM = MtWM + ridge * eye
    return MtWM, MtWd, M


def solve_group_fluxes(
    templates: jnp.ndarray,
    data: jnp.ndarray,
    weight: jnp.ndarray,
    *,
    ridge: float = 1e-6,
    pedestal: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """
    templates: (n_groups, K, n_frames, S, S) or (n_groups, K, n_frames, P)
               unit-flux model per member slot (padding slots all-zero).
    data / weight: matching spatial layout (square or packed P).

    Default (``pedestal=False``): returns flux ``(n_groups, n_frames, K)``,
    exactly as before M4.

    ``pedestal=True`` (task M4): returns ``(flux, pedestal)`` where
    ``pedestal`` is ``(n_groups, n_frames)`` -- one additive background level
    per group per frame, solved jointly with flux in the same K+1-unknown
    weighted LS (ridge 1e-6 on the augmented normal equations, same as flux).
    Differentiable/gradient-safe: this is still ``jnp.linalg.solve`` on a
    small dense system, nothing about the call changes structurally.
    """
    MtWM, MtWd, _ = _normal_equations(templates, data, weight, ridge=ridge, pedestal=pedestal)
    sol = jnp.linalg.solve(MtWM, MtWd[..., None]).squeeze(-1)
    if not pedestal:
        return sol
    return sol[..., :-1], sol[..., -1]


def solve_group_fluxes_huber_irls(
    templates: jnp.ndarray,
    data: jnp.ndarray,
    weight: jnp.ndarray,
    *,
    ridge: float = 1e-6,
    huber_delta: float = 1.0,
    iterations: int = 2,
    pedestal: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Solve group fluxes with a fixed number of Huber IRLS iterations.

    ``weight`` is interpreted as inverse variance, so the Huber transition is
    applied to the standardized residual ``sqrt(weight) * (data - model)``.
    Starting from the exact weighted-L2 solution, each iteration solves normal
    equations with the usual Huber influence weight
    ``min(1, huber_delta / abs(standardized_residual))``.  The loop is unrolled
    at trace time and consists only of JAX operations, allowing gradients to
    propagate through both the templates and the iteratively reweighted solve.

    ``pedestal=True`` (task M4) carries the joint flux+pedestal solve through
    every IRLS iteration: the standardized residual used to build the Huber
    influence weight is computed against the pedestal-included model
    (``model_stamps(...) + pedestal[..., None]``), matching the K+1-unknown
    solve in ``solve_group_fluxes``. Default False is untouched (no pedestal
    term is ever formed), so the default path is bit-identical.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    if huber_delta <= 0:
        raise ValueError(f"huber_delta must be positive, got {huber_delta}")

    result = solve_group_fluxes(templates, data, weight, ridge=ridge, pedestal=pedestal)
    flux, b = result if pedestal else (result, None)
    _, data_flat, weight_flat, _ = _as_matrix(templates, data, weight)
    for _ in range(iterations):
        model = model_stamps(templates, flux).reshape(data_flat.shape)
        if pedestal:
            model = model + b[..., None]
        standardized = jnp.sqrt(jnp.maximum(weight_flat, 0.0)) * (data_flat - model)
        abs_standardized = jnp.abs(standardized)
        robust_weight = jnp.minimum(
            1.0,
            huber_delta / jnp.maximum(abs_standardized, jnp.finfo(templates.dtype).tiny),
        )
        effective_weight = weight_flat * robust_weight
        result = solve_group_fluxes(
            templates,
            data,
            effective_weight.reshape(weight.shape),
            ridge=ridge,
            pedestal=pedestal,
        )
        flux, b = result if pedestal else (result, None)
    if pedestal:
        return flux, b
    return flux


def solve_fluxes(
    templates: jnp.ndarray,
    data: jnp.ndarray,
    weight: jnp.ndarray,
    *,
    flux_objective: str = "l2",
    ridge: float = 1e-6,
    huber_delta: float = 1.0,
    irls_iterations: int = 2,
    pedestal: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Dispatch to the requested differentiable profile-flux objective.

    The default is deliberately the historical weighted-L2 solve.  Objective
    selection is a Python-level/static choice suitable for use outside or as a
    static argument to a jitted loss function.

    ``pedestal=True`` (task M4): returns ``(flux, pedestal)`` instead of just
    ``flux`` -- see ``solve_group_fluxes``. Default False is unaffected.
    """
    # The public CLI uses a hyphen; normalize while this remains a static
    # Python string (before any traced JAX work).
    flux_objective = flux_objective.replace("-", "_")
    if flux_objective == "l2":
        return solve_group_fluxes(templates, data, weight, ridge=ridge, pedestal=pedestal)
    if flux_objective == "huber_irls":
        return solve_group_fluxes_huber_irls(
            templates,
            data,
            weight,
            ridge=ridge,
            huber_delta=huber_delta,
            iterations=irls_iterations,
            pedestal=pedestal,
        )
    raise ValueError(
        f"unknown flux_objective {flux_objective!r}; expected one of {FLUX_OBJECTIVES}"
    )


def solve_group_fluxes_with_err(
    templates: jnp.ndarray,
    data: jnp.ndarray,
    weight: jnp.ndarray,
    *,
    ridge: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forced-PSF fluxes and 1-sigma errors from ``diag((M^T W M)^{-1})``."""
    MtWM, MtWd, _ = _normal_equations(templates, data, weight, ridge=ridge)
    flux = jnp.linalg.solve(MtWM, MtWd[..., None]).squeeze(-1)
    cov = jnp.linalg.inv(MtWM)
    sigma_f = jnp.sqrt(jnp.clip(jnp.diagonal(cov, axis1=-2, axis2=-1), 0.0))
    return flux, sigma_f


def enclosed_flux_renormalize(
    flux: jnp.ndarray,
    template_footprint_sum: jnp.ndarray,
    base_footprint_sum: jnp.ndarray,
) -> jnp.ndarray:
    """Enclosed-flux-preserving post-hoc flux renormalisation (task T1-3).

    A star with constant TRUE total flux can still show a drifting
    profile-fit ``flux`` (``f_s(t)``, from ``solve_group_fluxes*`` above) if
    the per-frame, wobbling unit-flux template's footprint-enclosed sum
    drifts relative to the static (``w_coeff=0``) ``base``'s -- S2's
    "raw aperture sum / normalization drift" mechanism (A) in
    ``diagnostics/wfix_mode_footprint_flux.py`` / ``S2_mode_footprint``,
    which the profile fit only partially self-corrects against (mechanism B).
    This renormalises the *already solved* flux by the ratio of the two
    footprint-enclosed sums, pinning the footprint normalisation to the
    static base at every frame -- the opposite end of the honesty spectrum
    from trusting the profile fit's partial self-correction as-is:

        f_renorm(t) = f_s(t) * sum_footprint(T_s(t)) / sum_footprint(base_s)

    All three arguments must already be restricted to the SAME footprint
    (e.g. the packed-tier pixels within some radius R of the star) and the
    same units (unit-flux templates, i.e. ``template_footprint_sum`` and
    ``base_footprint_sum`` are dimensionless sums of a flux-fraction map, not
    counts). ``base_footprint_sum`` may be per-(group,) (a single static
    number) or per-(group, frame) (broadcastable) -- passing a per-frame
    array with the SAME value at every frame is equivalent to the static
    case and is not required.

    This is a diagnostic/reporting option (see
    ``diagnostics/t1_footprint_gauge.py`` findings for which convention is
    right for a constant-flux star), not wired into any production export
    path by this change.
    """
    return flux * (template_footprint_sum / base_footprint_sum)


def model_stamps(templates: jnp.ndarray, flux: jnp.ndarray) -> jnp.ndarray:
    """templates (...,K,...,spatial) × flux (...,K) → model with spatial layout of templates.

    Square: templates (n_groups,K,n_frames,S,S), flux (n_groups,n_frames,K)
        → (n_groups,n_frames,S,S)
    Packed: templates (n_groups,K,n_frames,P), flux (n_groups,n_frames,K)
        → (n_groups,n_frames,P)
    """
    if templates.ndim == 5:
        return jnp.einsum("gkfxy,gfk->gfxy", templates, flux)
    if templates.ndim == 4:
        return jnp.einsum("gkfp,gfk->gfp", templates, flux)
    raise ValueError(f"templates ndim must be 4 or 5, got {templates.ndim}")


def model_stamps_with_pedestal(
    templates: jnp.ndarray, flux: jnp.ndarray, pedestal: jnp.ndarray,
) -> jnp.ndarray:
    """``model_stamps(...)`` plus the per-group-per-frame additive pedestal
    (task M4), broadcast over the trailing spatial axis/axes.

    ``pedestal``: (n_groups, n_frames), the second element of the tuple
    ``solve_group_fluxes*(..., pedestal=True)`` returns -- one background
    level per group per frame, shared by every K member (same value added
    to every pixel of the stamp, not per-member).
    """
    model = model_stamps(templates, flux)
    if templates.ndim == 5:
        return model + pedestal[..., None, None]
    if templates.ndim == 4:
        return model + pedestal[..., None]
    raise ValueError(f"templates ndim must be 4 or 5, got {templates.ndim}")


# ---------------------------------------------------------------------------
# Task PW: profile the temporal ePSF mode amplitude out in closed form,
# per frame, jointly with the per-stamp fluxes.
# ---------------------------------------------------------------------------
#
# The model's temporal term is ``field(t) = base + sum_k w_k(t) * mode_k`` and
# the rendered unit-flux template is LINEAR in the per-frame scalars ``w_k(t)``
# to ~1e-5 (the only nonlinearity is the core-recenter's centroid ratio and
# ``renorm_scalar``; verified in ``.../investigation_20260906/T2_w_warm_k2/``
# and ``diagnostics/warm_start_w.py``).  So, writing ``A`` for the template
# rendered at ``w = 0`` and ``B_m = d(template)/d(w_m)``,
#
#     T_s(t; w) = A_s(t) + sum_m w_m(t) * B_{s,m}(t)
#
# and the per-(group, frame) model is
#
#     model_g(t) = sum_j f_{gj}(t) * T_{gj}(t; w(t))  [+ b_g(t)]
#
# which is BILINEAR in (f, w): the fluxes are per stamp, the ``w`` are shared
# by every stamp in the frame.  The unknowns per frame are therefore
# ``{f_s} u {b_s} u {w_m}`` -- a bordered system, block-diagonal in the stamps
# with a dense K_modes-wide border.  ``solve_group_fluxes_profile_w`` below
# solves it by Gauss-Newton: at each iteration the per-stamp blocks (1x1, 2x2,
# or (K_members+1)^2 with the pedestal) are eliminated first and only the small
# K_modes x K_modes Schur complement is formed and inverted per frame.  Two
# iterations suffice because the neglected term is second order (dF * dW).
#
# Why jointly and not "solve w freely, then re-solve flux": the mode is gauged
# flux-neutral on the FULL ePSF grid but carries net flux inside the ~4 px
# packed footprints where the flux is actually solved
# (``.../S2_mode_footprint/findings.md``), so a freely fitted amplitude can
# masquerade as a flux change.  The Schur complement is exactly the statement
# "the part of the mode that the per-stamp fluxes cannot already absorb",
# which removes that degeneracy by construction.


def _as_mode_matrix(mode_templates: jnp.ndarray) -> jnp.ndarray:
    """``(n_modes, n_groups, K, n_frames, *spatial)`` -> ``(n_modes, n_groups, n_frames, P, K)``.

    The per-mode analogue of ``_as_matrix``'s template reshape, so square
    ``(..., S, S)`` and packed ``(..., P)`` derivative stacks flatten the same
    way the templates they differentiate do.
    """
    if mode_templates.ndim == 6:
        n_modes, n_groups, K, n_frames, S, _ = mode_templates.shape
        flat = mode_templates.reshape(n_modes, n_groups, K, n_frames, S * S)
        return jnp.moveaxis(flat, 2, -1)
    if mode_templates.ndim == 5:
        return jnp.moveaxis(mode_templates, 2, -1)
    raise ValueError(
        f"mode_templates ndim must be 5 (packed) or 6 (square), got {mode_templates.ndim}"
    )


def templates_at_w(
    templates: jnp.ndarray,
    mode_templates: jnp.ndarray,
    w_of_t: jnp.ndarray,
) -> jnp.ndarray:
    """``A + sum_m w_m(t) B_m`` in the templates' own layout.

    ``templates`` ``(n_groups, K, n_frames, *spatial)`` is the ``w = 0``
    render; ``mode_templates`` ``(n_modes, n_groups, K, n_frames, *spatial)``
    the per-mode derivative; ``w_of_t`` ``(n_frames, n_modes)``.
    """
    if mode_templates.shape[0] == 0:
        return templates
    return templates + jnp.einsum("fm,mgkf...->gkf...", w_of_t, mode_templates)


def _huber_effective_weight(
    data_flat: jnp.ndarray,
    model_flat: jnp.ndarray,
    weight_flat: jnp.ndarray,
    huber_delta: float,
) -> jnp.ndarray:
    """The IRLS influence weight used by ``solve_group_fluxes_huber_irls``.

    Factored out verbatim (same standardized residual, same
    ``min(1, delta/|chi|)`` clip, same ``tiny`` guard) so the profiled-w solve
    reweights its bordered system exactly the way the flux-only solve
    reweights its normal equations.
    """
    standardized = jnp.sqrt(jnp.maximum(weight_flat, 0.0)) * (data_flat - model_flat)
    robust = jnp.minimum(
        1.0,
        huber_delta / jnp.maximum(jnp.abs(standardized), jnp.finfo(weight_flat.dtype).tiny),
    )
    return weight_flat * robust


def solve_group_fluxes_profile_w(
    templates: jnp.ndarray,
    mode_templates: jnp.ndarray,
    data: jnp.ndarray,
    weight: jnp.ndarray,
    *,
    ridge: float = 1e-6,
    ridge_w: float | None = None,
    pedestal: bool = False,
    w_init: jnp.ndarray | None = None,
    iterations: int = 2,
    flux_objective: str = "l2",
    huber_delta: float = 1.0,
    irls_iterations: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
    """Solve, per FRAME, the ``n_modes`` amplitudes jointly with every stamp's
    flux (and pedestal) in closed form.  Returns ``(flux, pedestal, w_of_t)``
    with ``pedestal`` ``None`` when ``pedestal=False``.

    Arguments mirror ``solve_group_fluxes`` plus:

    ``mode_templates``
        ``(n_modes, n_groups, K, n_frames, *spatial)`` -- the derivative of the
        unit-flux template with respect to each ``w_m``, i.e. the render at
        ``w = e_m`` minus the render at ``w = 0`` (see ``loss.forward_model``'s
        ``return_mode_templates``).  ``n_modes == 0`` short-circuits to the
        plain flux solve.
    ``w_init``
        ``(n_frames, n_modes)`` starting point (default zeros, i.e. start from
        the static template).  Passing the trained spline's ``w_of_t`` here
        only changes the starting iterate, not the fixed point.
    ``iterations``
        Gauss-Newton iterations on the bilinear system.  Each costs one exact
        conditional flux solve plus one bordered/Schur solve; the final flux
        returned is always an exact conditional solve at the final ``w``, so
        ``iterations=0`` reproduces ``solve_fluxes`` at ``w_init`` exactly.
    ``ridge_w``
        Diagonal ridge on the ``n_modes x n_modes`` Schur complement.  Default
        ``None`` means "the same ridge as the per-stamp blocks".

    The per-frame system is bordered:

        [ H_11        0     H_1w ] [ df_1 ]   [ g_1 ]
        [    .        .        . ] [   .  ] = [  .  ]
        [    0     H_GG   H_Gw   ] [ df_G ]   [ g_G ]
        [ H_1w^T .. H_Gw^T H_ww  ] [ dw   ]   [ g_w ]

    with ``H_gg`` of size ``(K_members + pedestal)^2`` and ``H_ww`` of size
    ``n_modes^2``.  Eliminating the block-diagonal part gives the Schur
    complement ``S = H_ww - sum_g H_gw^T H_gg^{-1} H_gw`` -- a single
    ``n_modes x n_modes`` solve per frame -- and then ``df_g`` by back
    substitution.  Nothing of size ``(n_stamps * K + n_modes)^2`` is ever
    formed.

    GAUGE: ``w_of_t`` is solved FREELY here; it is NOT forced to zero time
    mean.  The constant part is a genuine gauge freedom of the model, exactly
    absorbable into the base (``base' = base + sum_k c_k mode_k``,
    ``w'(t) = w(t) - c`` renders an identical field at every frame), so the
    profiled loss is exactly invariant along it and there is nothing to
    constrain inside a per-frame solve.  Call ``gauge_zero_time_mean`` on the
    result -- together with ``shift_base_by_modes`` on the ePSF base -- to
    restore the ``w_field_from_coeff`` convention for reporting/checkpointing.

    Gradients: this differentiates THROUGH the solve (every step is a jnp op
    on small dense systems, as in ``solve_group_fluxes``), not via the
    envelope theorem.  That is the correct choice here because a fixed,
    finite ``iterations`` makes the profiled ``w`` a deterministic function of
    the parameters rather than an exact argmin -- the envelope theorem's
    ``d/dtheta L(theta, w*(theta)) = partial_theta L`` only holds at the exact
    stationary point, and would silently drop a real term at ``iterations=1``.
    """
    A = templates
    if mode_templates.shape[0] == 0:
        result = solve_fluxes(
            A, data, weight,
            flux_objective=flux_objective, ridge=ridge,
            huber_delta=huber_delta, irls_iterations=irls_iterations,
            pedestal=pedestal,
        )
        n_frames = A.shape[2]
        w_out = jnp.zeros((n_frames, 0), dtype=A.dtype)
        if pedestal:
            return result[0], result[1], w_out
        return result, None, w_out
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")

    Amat, d, wgt, n_members = _as_matrix(A, data, weight)      # (G,F,P,K), (G,F,P), (G,F,P)
    Bmat = _as_mode_matrix(mode_templates)                      # (M,G,F,P,K)
    n_modes = int(Bmat.shape[0])
    n_frames = int(Amat.shape[1])
    dtype = Amat.dtype
    ridge_w_eff = ridge if ridge_w is None else ridge_w

    if w_init is None:
        w = jnp.zeros((n_frames, n_modes), dtype=dtype)
    else:
        w = jnp.asarray(w_init, dtype=dtype).reshape(n_frames, n_modes)

    eye_m = jnp.eye(n_modes, dtype=dtype)

    for _ in range(iterations):
        S, rhs, _, _ = profile_w_schur_pieces(
            A, mode_templates, data, weight, w,
            ridge=ridge, pedestal=pedestal, flux_objective=flux_objective,
            huber_delta=huber_delta, irls_iterations=irls_iterations,
        )
        dw = jnp.linalg.solve(S + ridge_w_eff * eye_m, rhs[..., None])[..., 0]
        w = w + dw

    T = templates_at_w(A, mode_templates, w)
    result = solve_fluxes(
        T, data, weight,
        flux_objective=flux_objective, ridge=ridge,
        huber_delta=huber_delta, irls_iterations=irls_iterations,
        pedestal=pedestal,
    )
    if pedestal:
        return result[0], result[1], w
    return result, None, w


def profile_w_schur_pieces(
    templates: jnp.ndarray,
    mode_templates: jnp.ndarray,
    data: jnp.ndarray,
    weight: jnp.ndarray,
    w_of_t: jnp.ndarray,
    *,
    ridge: float = 1e-6,
    pedestal: bool = False,
    flux_objective: str = "l2",
    huber_delta: float = 1.0,
    irls_iterations: int = 2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """One Gauss-Newton step's bordered-system pieces, summed over the groups
    given here.  Returns ``(S, rhs, flux, pedestal)`` with

      ``S``    ``(n_frames, n_modes, n_modes)`` -- the Schur complement
               ``sum_g [C_g^T W C_g - H_gw^T H_gg^{-1} H_gw]``
      ``rhs``  ``(n_frames, n_modes)`` -- the matching right-hand side
      ``flux`` the exact conditional flux at ``w_of_t`` (and ``pedestal``)

    so that the amplitude increment is ``dw = solve(S + ridge_w I, rhs)``.

    Both ``S`` and ``rhs`` are ADDITIVE over groups, which is the whole point
    of exposing this: the amplitudes are shared by every stamp in a frame, but
    a full-population solve never has all the stamps in memory at once (the
    packed tiers/buckets are processed a chunk at a time).  Callers accumulate
    ``S``/``rhs`` across chunks, solve the small system once, and only then
    re-render at the updated ``w`` -- see ``gpu_flux_export`` and
    ``diagnostics/pw_solve_w.py``.  Solving per chunk and averaging would NOT
    be the same thing.
    """
    A = templates
    Amat, d, wgt, _ = _as_matrix(A, data, weight)
    Bmat = _as_mode_matrix(mode_templates)
    dtype = Amat.dtype
    n_modes = int(Bmat.shape[0])

    T = templates_at_w(A, mode_templates, w_of_t)
    result = solve_fluxes(
        T, data, weight,
        flux_objective=flux_objective, ridge=ridge,
        huber_delta=huber_delta, irls_iterations=irls_iterations,
        pedestal=pedestal,
    )
    flux, ped = result if pedestal else (result, None)

    Tmat = jnp.moveaxis(T.reshape(T.shape[0], T.shape[1], T.shape[2], -1), 1, -1)  # (G,F,P,K)
    model = jnp.einsum("gfpk,gfk->gfp", Tmat, flux)
    if pedestal:
        model = model + ped[..., None]
    if flux_objective.replace("-", "_") == "huber_irls":
        W = _huber_effective_weight(d, model, wgt, huber_delta)
    else:
        W = wgt
    resid = d - model

    # Per-stamp design: the templates themselves, plus the constant pedestal
    # column when it is part of the solve.
    if pedestal:
        X = jnp.concatenate([Tmat, jnp.ones_like(Tmat[..., :1])], axis=-1)
    else:
        X = Tmat
    n_unknown = X.shape[-1]
    # Border design: d(model)/d(w_m) = sum_j f_j B_{j,m}.
    C = jnp.einsum("gfk,mgfpk->gfpm", flux, Bmat)

    XW = X * W[..., None]
    CW = C * W[..., None]
    H_gg = jnp.einsum("gfpk,gfpl->gfkl", XW, X) + ridge * jnp.eye(n_unknown, dtype=dtype)
    H_gw = jnp.einsum("gfpk,gfpm->gfkm", XW, C)
    H_ww = jnp.einsum("gfpm,gfpn->fmn", CW, C)
    g_g = jnp.einsum("gfpk,gfp->gfk", XW, resid)
    g_w = jnp.einsum("gfpm,gfp->fm", CW, resid)

    Z = jnp.linalg.solve(H_gg, H_gw)                    # (G,F,n_unknown,M)
    z = jnp.linalg.solve(H_gg, g_g[..., None])[..., 0]  # (G,F,n_unknown)
    S = H_ww - jnp.einsum("gfkm,gfkn->fmn", H_gw, Z)
    rhs = g_w - jnp.einsum("gfkm,gfk->fm", H_gw, z)
    del n_modes
    return S, rhs, flux, ped


def gauge_zero_time_mean(
    w_of_t: jnp.ndarray,
    frame_weight: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Restore the ``w_field_from_coeff`` zero-time-mean gauge on a freely
    solved ``w_of_t`` ``(n_frames, n_modes)``.

    Returns ``(w_gauged, const)`` with ``w_gauged = w_of_t - const`` and
    ``const`` the (optionally ``frame_weight``-weighted) time mean.

    WHAT HAPPENS TO THE CONSTANT: it goes into the ePSF base, exactly and
    losslessly.  The rendered field is
    ``field(t) = base + sum_k w_k(t) mode_k``, so

        base + sum_k w_k(t) mode_k == (base + sum_k c_k mode_k)
                                     + sum_k (w_k(t) - c_k) mode_k

    for any constant ``c``.  Use ``shift_base_by_modes(base, modes, const)``
    to build the matching base; the pair ``(base', w_gauged)`` renders a
    bit-identical field at every frame (up to fp roundoff), which is why the
    profiled loss cannot see this direction at all.  The constant is NOT
    discarded and NOT absorbed by the fluxes.
    """
    if frame_weight is None:
        const = jnp.mean(w_of_t, axis=0)
    else:
        fw = jnp.asarray(frame_weight, dtype=w_of_t.dtype).reshape(-1, 1)
        const = jnp.sum(w_of_t * fw, axis=0) / jnp.clip(jnp.sum(fw), 1e-30, None)
    return w_of_t - const, const


def shift_base_by_modes(
    base: jnp.ndarray,
    modes: jnp.ndarray,
    const: jnp.ndarray,
) -> jnp.ndarray:
    """``base + sum_k const_k * modes_k`` -- the other half of
    ``gauge_zero_time_mean`` (see its docstring).

    ``base`` ``(n_rows, n_cols, G, G)``, ``modes`` ``(n_modes, n_rows,
    n_cols, G, G)``, ``const`` ``(n_modes,)``.
    """
    if modes.shape[0] == 0:
        return base
    return base + jnp.einsum("k,kijxy->ijxy", jnp.asarray(const, dtype=base.dtype), modes)
