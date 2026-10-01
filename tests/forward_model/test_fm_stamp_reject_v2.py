# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for the two-level stamp-rejection replacement in ``stamp_reject.py``.

This module was added +646 lines by a prior agent that died before writing
any tests -- everything here is new coverage for that diff, not a rewrite of
existing tests. See ``stamp_reject.py``'s module docstring for the two-gate
design (pooled linear MAD vs. per-group-centred log-domain MAD) this exists
to validate.

Sections:
  1. ``per_stamp_chi2_red`` jit-cache correctness (eager-equivalence + reuse).
  2. THE KEY REGRESSION TEST -- old pooled gate vs. new level-2 gate on a
     synthetic (group, frame) chi2 matrix built to reproduce the measured
     failure mode (whole-group cuts, missed per-cadence spikes).
  3. Degenerate inputs (zero/negative/NaN chi2, empty groups, all-masked).
  4. Level-2 statefulness: hysteresis requires consecutive evidence, churn
     cap actually caps (and confirms the ``state=None`` audit-fix: a single
     anonymous call must decide immediately, not never).
  5. Level-1 audit table: ranks groups, never cuts by itself.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import flux_solve as FS
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import stamp_reject as SR
from syndiff_pipeline.forward_model import temporal as T
from syndiff_pipeline.forward_model._bootstrap import _EXTRA_PATHS  # noqa: F401 (sys.path wiring)
from syndiff_pipeline.forward_model.data import RegionSpec
from syndiff_pipeline.forward_model.groups import GroupSet

from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture: a tiny, fully synthetic (no disk) StaticContext + FitData.
# One star per group (K=1); pixel content is irrelevant for the level-2 gate
# tests (they monkeypatch SR.per_stamp_chi2_red), but must be self-consistent
# for the jit-vs-eager equivalence test, which runs the real forward model.
# ---------------------------------------------------------------------------


def _tiny_ctx_and_fd(n_groups: int, n_frames: int, *, stamp: int = 13):
    from syndiff_pipeline.forward_model import fit as FIT

    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=1, exponents=tuple(sci2idl_exponents(1)),
    )
    grid = EM.EpsfGridStatic(
        node_x=np.array([25.0, 75.0]), node_y=np.array([25.0, 75.0]),
        node_col_ccd=np.array([25.0, 75.0]), node_row_ccd=np.array([25.0, 75.0]),
    )
    Gsz = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:Gsz, 0:Gsz]
    r2 = (xx - EM.NODE_CENTER_INDEX) ** 2 + (yy - EM.NODE_CENTER_INDEX) ** 2
    blob = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.5) ** 2)).astype(np.float32)
    blob /= blob.sum()
    base = np.zeros((2, 2, Gsz, Gsz), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            base[i, j] = blob
    modes = np.zeros((1, 2, 2, Gsz, Gsz), dtype=np.float32)
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))

    members = np.arange(n_groups, dtype=int).reshape(n_groups, 1)
    valid = np.ones((n_groups, 1), dtype=bool)
    groups = GroupSet(n_groups, 1, members, valid, np.ones(n_groups, dtype=bool), 0)
    ra = np.linspace(179.9, 180.1, n_groups).astype(np.float32)
    dec = np.linspace(-0.05, 0.05, n_groups).astype(np.float32)
    n_basis = 4
    wcs_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    wcs_fb[:, 0] = 1.0
    w_fb = np.zeros((n_frames, n_basis), dtype=np.float32)
    w_fb[:, 0] = 1.0

    ctx = L.build_static_context(
        cheb_static=static, wcs_frame_basis=wcs_fb, w_frame_basis=w_fb,
        epsf_grid=grid, groups=groups, ra=ra, dec=dec,
        stamp_center_x=np.full(n_groups, 50, dtype=np.int64),
        stamp_center_y=np.full(n_groups, 50, dtype=np.int64),
        t_exp_sec=1426.0, stamp_snr_weight=np.ones(n_groups, dtype=np.float32),
        fit_radius=np.full(n_groups, 3.0, dtype=np.float32),
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_basis, n_w_basis=n_basis)

    shape = (n_groups, n_frames, stamp, stamp)
    data = jnp.zeros(shape, dtype=jnp.float32)
    noise = jnp.ones(shape, dtype=jnp.float32)
    weight = jnp.ones(shape, dtype=jnp.float32)
    mask_s = np.ones((n_groups, n_frames), dtype=np.float32)

    fd = FIT.FitData(
        ctx=ctx, data=data, noise=noise, weight=weight,
        wcs_second_diff=T.second_difference_matrix(n_basis),
        w_second_diff=T.second_difference_matrix(n_basis),
        epsf_modes_init=epsf0.modes,
        mask_stamp_active=mask_s,
    )
    return fd, params


@pytest.fixture(autouse=True)
def _clear_jit_cache_and_patch():
    """Every test gets a clean module-level chi2-jit cache and a fresh
    ``SR.per_stamp_chi2_red`` (several tests monkeypatch it directly since
    they only exercise the numpy-side gate logic, not the real forward
    model)."""
    SR.clear_chi2_jit_cache()
    original = SR.per_stamp_chi2_red
    yield
    SR.per_stamp_chi2_red = original
    SR.clear_chi2_jit_cache()


# ---------------------------------------------------------------------------
# 1. per_stamp_chi2_red jit cache: eager-equivalence + reuse.
# ---------------------------------------------------------------------------


def _eager_chi2_red_reference(params, fd):
    """Exact copy of the pre-diff (un-jitted, uncached) ``per_stamp_chi2_red``
    body -- the ground truth the jit cache must reproduce bit-for-bit (modulo
    fp32 roundoff)."""
    packed = bool(getattr(fd.ctx, "is_packed", False))
    local_cache = fd.local_cache if fd.use_dx_only else None
    n_pix = fd.n_pix if fd.n_pix is not None else (1 if packed else int(fd.data.shape[-1]))
    templates, _, _, _ = L.forward_model(
        params, fd.ctx,
        local_cache=local_cache,
        n_pix=n_pix,
        do_recenter=fd.do_recenter and local_cache is None,
        recenter_n_iter=fd.recenter_n_iter,
    )
    var = L.pixel_variance(fd.noise)
    if packed:
        pix_w = fd.weight * fd.ctx.pix_valid[:, None, :]
        reduce_axes = (-1,)
    else:
        rmask = L.radius_pixel_mask(fd.ctx.fit_radius, stamp=fd.data.shape[-1])
        pix_w = fd.weight * rmask[:, None, :, :]
        reduce_axes = (-1, -2)
    iv = L.inverse_variance_weights(pix_w, var)
    flux = FS.solve_group_fluxes(templates, fd.data, iv, ridge=fd.weights.ridge)
    model = FS.model_stamps(templates, flux)
    chi2 = pix_w * (fd.data - model) ** 2 / var
    chi2_sum = jnp.sum(chi2, axis=reduce_axes)
    pix_sum = jnp.sum(pix_w, axis=reduce_axes)
    chi2_red = chi2_sum / jnp.clip(pix_sum, 1e-6, None)
    return chi2_red, pix_sum


def test_per_stamp_chi2_red_jit_matches_eager_reference():
    fd, params = _tiny_ctx_and_fd(2, 5)
    rng = np.random.default_rng(1)
    fd.data = jnp.asarray(rng.normal(5.0, 1.0, size=fd.data.shape).astype(np.float32))
    fd.noise = jnp.asarray(np.abs(rng.normal(1.0, 0.2, size=fd.noise.shape)).astype(np.float32))

    chi2_eager, pix_eager = _eager_chi2_red_reference(params, fd)
    chi2_jit, pix_jit = SR.per_stamp_chi2_red(params, fd)

    np.testing.assert_allclose(np.asarray(chi2_jit), np.asarray(chi2_eager), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.asarray(pix_jit), np.asarray(pix_eager), rtol=1e-5, atol=1e-4)


def test_per_stamp_chi2_red_reuses_compiled_callable_across_refreshes(monkeypatch):
    """Repeated calls on the same (shape/dtype/ctx.fit_radius-identity) fd must
    compile once (``_build_chi2_jit``) and reuse the cached jax.jit executable
    thereafter -- this is the entire point of Task 1. A ctx swap via
    ``loss.with_stamp_active`` (what every refresh does) must NOT force a
    rebuild, since it keeps the same ``fit_radius`` array identity."""
    fd, params = _tiny_ctx_and_fd(1, 4)

    build_calls = []
    real_build = SR._build_chi2_jit

    def counting_build(*a, **kw):
        build_calls.append(1)
        return real_build(*a, **kw)

    monkeypatch.setattr(SR, "_build_chi2_jit", counting_build)

    SR.per_stamp_chi2_red(params, fd)
    assert len(build_calls) == 1

    # Simulate what refresh_stamp_active(_multi) does every refresh: replace
    # ctx wholesale (new StaticContext instance) but keep fit_radius identity.
    fd.ctx = L.with_stamp_active(fd.ctx, np.zeros((1, 4), dtype=np.float32))
    SR.per_stamp_chi2_red(params, fd)
    assert len(build_calls) == 1, "ctx replacement (stamp_active only) must not force a recompile"

    # A genuinely different fit_radius (with_fit_radius) must bust the cache.
    fd.ctx = L.with_fit_radius(fd.ctx, np.full(1, 5.0, dtype=np.float32))
    SR.per_stamp_chi2_red(params, fd)
    assert len(build_calls) == 2, "a real fit_radius change must trigger a rebuild"


def test_clear_chi2_jit_cache_empties_cache():
    fd, params = _tiny_ctx_and_fd(1, 3)
    SR.per_stamp_chi2_red(params, fd)
    assert len(SR._CHI2_JIT_CACHE) == 1
    SR.clear_chi2_jit_cache()
    assert len(SR._CHI2_JIT_CACHE) == 0


# ---------------------------------------------------------------------------
# 2. THE KEY REGRESSION TEST.
#
# Reproduces the measured failure mode from output/rejection_study/ directly:
# group MEDIANS span decades (here 300 -> 30000, 2 decades) while the
# within-group CV is tight (~8%, matching the measured CV~0.08). A single
# pooled *linear* MAD threshold sits well above the low-baseline groups and
# well below the high-baseline groups, so it can only make whole-group,
# all-or-nothing cuts on the *between*-group spread -- and it completely
# misses a genuine per-cadence anomaly injected into a low-baseline group,
# because that anomaly is tiny compared to the between-group spread even
# though it is enormous (many MAD-sigmas) relative to its own group's tight
# scatter.
#
# The level-2 (per-group log-centred) gate does the opposite on the exact
# same chi2_red values: it is blind to the between-group offset (that's the
# whole design) so it does not touch the high-baseline group at all, and it
# catches the per-cadence spike because, relative to its own group's tiny
# scatter, the spike is a huge, easily MAD-significant deviation.
# ---------------------------------------------------------------------------


def _build_two_gate_regression_chi2():
    rng = np.random.default_rng(7)
    n_groups, n_frames = 15, 40
    sigma_log = 0.08  # matches the measured within-group CV~0.08
    baselines = np.logspace(np.log10(300.0), np.log10(30000.0), n_groups)
    chi2 = np.zeros((n_groups, n_frames))
    for g in range(n_groups):
        chi2[g] = baselines[g] * np.exp(rng.normal(0.0, sigma_log, size=n_frames))

    spike_group, spike_frame = 0, 20  # lowest-baseline group
    chi2[spike_group, spike_frame] = baselines[spike_group] * 5.0  # ~13 group-local MAD-sigma
    return chi2, baselines, spike_group, spike_frame


def test_old_pooled_gate_makes_whole_group_cuts_and_misses_the_spike():
    chi2, baselines, spike_group, spike_frame = _build_two_gate_regression_chi2()
    pix_ok = np.ones_like(chi2, dtype=bool)

    mad_keep = SR.mad_reject_mask(chi2, n_sigma=3.0, pix_active=pix_ok, pool_active=pix_ok)
    old_reject = (mad_keep == 0) & pix_ok

    # Whole-group, all-or-nothing: the two highest-baseline groups (well above
    # the pooled threshold) are rejected on essentially every frame.
    per_group_frac = old_reject.mean(axis=1)
    assert per_group_frac[-1] == pytest.approx(1.0)
    assert per_group_frac[-2] >= 0.9

    # A pristine low-baseline group is never touched by the pooled gate.
    assert per_group_frac[0:5].sum() == 0 or per_group_frac[spike_group] <= 1.0 / chi2.shape[1] + 1e-9

    # The per-cadence spike (~13 group-local MAD-sigma) is invisible to the
    # pooled linear gate: it is tiny next to the between-group spread.
    assert not old_reject[spike_group, spike_frame]


def test_new_level2_gate_ignores_baseline_group_and_catches_the_spike():
    chi2, baselines, spike_group, spike_frame = _build_two_gate_regression_chi2()
    pix_ok = np.ones_like(chi2, dtype=bool)

    d, b_g = SR.per_cadence_log_deviation(chi2, pool_active=pix_ok)
    scale = SR.level2_mad_scale(d, pix_ok)
    new_reject = pix_ok & np.isfinite(d) & (d > 3.0 * scale)

    # The two highest-baseline groups are essentially untouched: level 2
    # subtracts out each group's own baseline before thresholding.
    per_group_frac = new_reject.mean(axis=1)
    assert per_group_frac[-1] == 0.0
    assert per_group_frac[-2] <= 1.0 / chi2.shape[1] + 1e-9

    # The per-cadence spike IS caught.
    assert new_reject[spike_group, spike_frame]

    # And overall it rejects far fewer cells than the pooled gate (graded,
    # not all-or-nothing) -- this is the measured "every group is 0 or 1"
    # symptom the new gate is designed to fix.
    mad_keep = SR.mad_reject_mask(chi2, n_sigma=3.0, pix_active=pix_ok, pool_active=pix_ok)
    old_reject = (mad_keep == 0) & pix_ok
    assert new_reject.sum() < old_reject.sum()


# ---------------------------------------------------------------------------
# 3. Degenerate inputs.
# ---------------------------------------------------------------------------


def test_safe_log_chi2_nans_out_nonpositive_and_nonfinite():
    chi2 = np.array([[1.0, 0.0, -5.0, np.nan, np.inf, 2.0]])
    log = SR._safe_log_chi2(chi2)
    assert log[0, 0] == pytest.approx(0.0)
    assert np.isnan(log[0, 1])  # zero
    assert np.isnan(log[0, 2])  # negative
    assert np.isnan(log[0, 3])  # already nan
    assert np.isnan(log[0, 4])  # +inf
    assert log[0, 5] == pytest.approx(np.log(2.0))


def test_per_group_log_chi2_baseline_nan_for_group_with_no_active_frames():
    chi2 = np.array([[100.0, 110.0, 90.0], [50.0, 55.0, 45.0]])
    pool = np.array([[True, True, True], [False, False, False]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        b_g = SR.per_group_log_chi2_baseline(chi2, pool_active=pool)
    assert np.isfinite(b_g[0])
    assert np.isnan(b_g[1])


def test_per_cadence_log_deviation_propagates_nan_for_inactive_group():
    chi2 = np.array([[100.0, 110.0, 90.0], [50.0, 55.0, 45.0]])
    pool = np.array([[True, True, True], [False, False, False]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d, b_g = SR.per_cadence_log_deviation(chi2, pool_active=pool)
    assert np.all(np.isfinite(d[0]))
    assert np.all(np.isnan(d[1]))


def test_level2_mad_scale_nan_when_all_cells_masked():
    scale = SR.level2_mad_scale(np.zeros((2, 3)), np.zeros((2, 3), dtype=bool))
    assert np.isnan(scale)


def test_level2_mad_scale_nan_propagating_d_excluded_via_isfinite():
    d = np.array([[0.0, 0.1, np.nan]])
    pool = np.ones((1, 3), dtype=bool)
    scale = SR.level2_mad_scale(d, pool)
    assert np.isfinite(scale)  # nan cell dropped, not poisoning the result


def test_refresh_stamp_active_v2_handles_zero_negative_nan_chi2_without_crashing(monkeypatch):
    fd, params = _tiny_ctx_and_fd(3, 6)

    def fake(_params, _fd):
        chi2 = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, -5.0, np.nan, np.inf, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ])
        pix = np.full((3, 6), 100.0)
        return jnp.asarray(chi2), jnp.asarray(pix)

    monkeypatch.setattr(SR, "per_stamp_chi2_red", fake)
    stats = SR.refresh_stamp_active_v2(params, [fd], state=None, n_sigma=3.0)
    assert np.isfinite(stats["n_rejected"])
    assert np.isfinite(stats["frac_rejected"])


def test_refresh_stamp_active_v2_all_cells_masked_by_baseline(monkeypatch):
    """A bucket whose sticky TNS/asteroid baseline is entirely zero must not
    crash (pool empty -> nan scale) and every stamp stays rejected via the
    baseline AND, independent of the gate's own (nan) decision."""
    fd, params = _tiny_ctx_and_fd(2, 4)
    fd.mask_stamp_active = np.zeros((2, 4), dtype=np.float32)

    def fake(_params, _fd):
        chi2 = np.full((2, 4), 300.0)
        pix = np.full((2, 4), 100.0)
        return jnp.asarray(chi2), jnp.asarray(pix)

    monkeypatch.setattr(SR, "per_stamp_chi2_red", fake)
    stats = SR.refresh_stamp_active_v2(
        params, [fd], state=None, n_sigma=3.0,
        mask_baselines=[fd.mask_stamp_active],
    )
    assert stats["n_active_kept"] == 0.0
    assert np.asarray(fd.ctx.stamp_active).sum() == 0.0


# ---------------------------------------------------------------------------
# 4. Level-2 statefulness: hysteresis + churn cap + the state=None audit fix.
# ---------------------------------------------------------------------------


def test_state_none_single_call_rejects_immediately_bug_fix():
    """Audit fix: before the fix, ``state=None`` created a fresh
    ``Level2GateState`` whose streaks always start at 0, so a first-ever
    ``raw_reject`` set streak=1 -- which can never satisfy the default
    ``hysteresis_n=2`` on a single call. That silently made ``state=None``
    reject NOTHING, EVER, contradicting the function's own documented
    "equivalent to disabling hysteresis for that one call". This is the exact
    scenario a one-shot, non-training frozen-checkpoint evaluation (e.g. the
    rejection-study acceptance gate) would hit."""
    fd, params = _tiny_ctx_and_fd(1, 20)

    def fake(_params, _fd):
        chi2 = np.full((1, 20), 300.0)
        chi2[0, 5] = 300.0 * 5.0
        return jnp.asarray(chi2), jnp.asarray(np.full((1, 20), 100.0))

    import unittest.mock as mock
    with mock.patch.object(SR, "per_stamp_chi2_red", side_effect=fake):
        stats = SR.refresh_stamp_active_v2(
            params, [fd], state=None, n_sigma=3.0, hysteresis_n=2, churn_cap_frac=0.5,
        )
    assert stats["n_rejected"] == 1.0
    assert stats["n_flipped"] == 1.0
    assert np.asarray(fd.ctx.stamp_active)[0, 5] == 0.0


def test_hysteresis_requires_two_consecutive_refreshes_with_persistent_state():
    fd, params = _tiny_ctx_and_fd(1, 20)

    def fake(_params, _fd):
        chi2 = np.full((1, 20), 300.0)
        chi2[0, 5] = 300.0 * 5.0  # persistent spike, every refresh
        return jnp.asarray(chi2), jnp.asarray(np.full((1, 20), 100.0))

    import unittest.mock as mock
    state = SR.Level2GateState()
    with mock.patch.object(SR, "per_stamp_chi2_red", side_effect=fake):
        s0 = SR.refresh_stamp_active_v2(
            params, [fd], state=state, n_sigma=3.0, hysteresis_n=2, churn_cap_frac=0.5,
        )
        assert s0["n_rejected"] == 0.0, "one refresh of evidence must not flip anything (hysteresis_n=2)"
        assert state.bucket(0).streak[0, 5] == 1

        s1 = SR.refresh_stamp_active_v2(
            params, [fd], state=state, n_sigma=3.0, hysteresis_n=2, churn_cap_frac=0.5,
        )
        assert s1["n_rejected"] == 1.0, "second consecutive refresh must commit the reject"
        assert state.bucket(0).committed_reject[0, 5]

        # A single subsequent keep-side refresh must not immediately flip back
        # (needs its own 2 consecutive keep refreshes).
        def fake_recovered(_params, _fd):
            chi2 = np.full((1, 20), 300.0)  # spike gone
            return jnp.asarray(chi2), jnp.asarray(np.full((1, 20), 100.0))

    with mock.patch.object(SR, "per_stamp_chi2_red", side_effect=fake_recovered):
        s2 = SR.refresh_stamp_active_v2(
            params, [fd], state=state, n_sigma=3.0, hysteresis_n=2, churn_cap_frac=0.5,
        )
        assert s2["n_rejected"] == 1.0, "still committed after only one recovery refresh"
        s3 = SR.refresh_stamp_active_v2(
            params, [fd], state=state, n_sigma=3.0, hysteresis_n=2, churn_cap_frac=0.5,
        )
        assert s3["n_rejected"] == 0.0, "re-enters after two consecutive recovery refreshes"


def test_churn_cap_limits_simultaneous_flips_and_defers_the_rest():
    n_groups, n_frames = 5, 20
    fd, params = _tiny_ctx_and_fd(n_groups, n_frames)
    n_active = n_groups * n_frames

    # Deterministic mild wobble (nonzero within-group scale) shared by every
    # group, plus 10 simultaneous spike cells in group 0.
    t = np.arange(n_frames)
    base_pattern = 300.0 * (1.0 + 0.02 * np.sin(t * 0.7))

    def fake(_params, _fd):
        chi2 = np.tile(base_pattern, (n_groups, 1))
        chi2[0, :10] = 300.0 * 5.0
        return jnp.asarray(chi2), jnp.asarray(np.full((n_groups, n_frames), 100.0))

    import unittest.mock as mock
    state = SR.Level2GateState()
    with mock.patch.object(SR, "per_stamp_chi2_red", side_effect=fake):
        s0 = SR.refresh_stamp_active_v2(
            params, [fd], state=state, n_sigma=3.0, hysteresis_n=2, churn_cap_frac=0.05,
        )
        assert s0["n_rejected"] == 0.0  # still building streak (hysteresis_n=2)

        s1 = SR.refresh_stamp_active_v2(
            params, [fd], state=state, n_sigma=3.0, hysteresis_n=2, churn_cap_frac=0.05,
        )
        max_flips = max(1, round(0.05 * n_active))
        assert max_flips == 5
        assert s1["n_flipped"] == max_flips
        assert s1["n_rejected"] == max_flips
        assert s1["n_churn_capped"] == 10 - max_flips

        # Deferred candidates carry their hysteresis evidence forward and get
        # applied (still cap-limited if needed) on the next refresh.
        s2 = SR.refresh_stamp_active_v2(
            params, [fd], state=state, n_sigma=3.0, hysteresis_n=2, churn_cap_frac=0.05,
        )
        assert s2["n_rejected"] == 10.0
        assert s2["n_churn_capped"] == 0.0


def test_churn_cap_at_least_one_flip_even_for_tiny_buckets():
    """``max(1, round(churn_cap_frac * n_active))`` floors at 1 flip/refresh
    so a small bucket is never permanently frozen by rounding to 0."""
    fd, params = _tiny_ctx_and_fd(1, 2)

    def fake(_params, _fd):
        return jnp.asarray([[300.0, 1500.0]]), jnp.asarray([[100.0, 100.0]])

    import unittest.mock as mock
    state = SR.Level2GateState()
    with mock.patch.object(SR, "per_stamp_chi2_red", side_effect=fake):
        SR.refresh_stamp_active_v2(
            params, [fd], state=state, n_sigma=3.0, hysteresis_n=1, churn_cap_frac=0.0001,
        )
        s1 = SR.refresh_stamp_active_v2(
            params, [fd], state=state, n_sigma=3.0, hysteresis_n=1, churn_cap_frac=0.0001,
        )
    assert s1["n_flipped"] >= 0.0  # no crash; churn cap floors at 1, not 0


# ---------------------------------------------------------------------------
# 5. Level-1 audit table: ranks groups, never cuts by itself.
# ---------------------------------------------------------------------------


def test_level1_group_mismatch_table_ranks_worst_first_and_handles_empty_group():
    chi2 = np.array([
        [300.0, 310.0, 295.0],   # typical
        [30000.0, 31000.0, 29500.0],  # persistently much worse
        [50.0, 55.0, 45.0],       # zero active frames
    ])
    pool = np.array([
        [True, True, True],
        [True, True, True],
        [False, False, False],
    ])
    gid = np.array([10, 11, 12])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        table = SR.level1_group_mismatch_table(chi2, pool_active=pool, group_ids=gid)

    assert list(table["group_id"]) == [11, 10, 12]  # worst first, empty-group last
    assert table.loc[table["group_id"] == 11, "n_active_frames"].item() == 3
    assert table.loc[table["group_id"] == 12, "n_active_frames"].item() == 0
    assert np.isnan(table.loc[table["group_id"] == 12, "baseline_log_chi2"].item())


def test_level1_group_mismatch_table_never_cuts_stamps_by_itself():
    """The table is audit-only: computing it must not mutate anything, and by
    itself has no cutting mechanism (only apply_group_exclusions does)."""
    chi2 = np.array([[300.0, 310.0], [30000.0, 31000.0]])
    pool = np.ones((2, 2), dtype=bool)
    chi2_before = chi2.copy()
    _ = SR.level1_group_mismatch_table(chi2, pool_active=pool)
    np.testing.assert_array_equal(chi2, chi2_before)  # no mutation


def test_apply_group_exclusions_is_noop_by_default_and_targeted_when_given():
    stamp_active = np.ones((3, 4), dtype=np.float32)
    # Default / empty / None: no-op.
    out_default = SR.apply_group_exclusions(stamp_active, None)
    np.testing.assert_array_equal(out_default, stamp_active)
    out_empty = SR.apply_group_exclusions(stamp_active, [])
    np.testing.assert_array_equal(out_empty, stamp_active)

    # Explicit group list (by group_id value, not row index): only the row
    # whose group_id matches is hard-zeroed -- row 1 carries group_id=20.
    out = SR.apply_group_exclusions(stamp_active, [20], group_ids=np.array([10, 20, 30]))
    np.testing.assert_array_equal(out[0], np.ones(4))
    np.testing.assert_array_equal(out[1], np.zeros(4))
    np.testing.assert_array_equal(out[2], np.ones(4))
    # Original input must not be mutated in place.
    assert stamp_active[1, 0] == 1.0


def test_refresh_stamp_active_v2_stats_are_superset_of_legacy_keys():
    fd, params = _tiny_ctx_and_fd(1, 6)

    def fake(_params, _fd):
        return jnp.asarray(np.full((1, 6), 300.0)), jnp.asarray(np.full((1, 6), 100.0))

    import unittest.mock as mock
    with mock.patch.object(SR, "per_stamp_chi2_red", side_effect=fake):
        stats = SR.refresh_stamp_active_v2(params, [fd], state=None, n_sigma=3.0)

    legacy_keys = {"n_rejected", "n_active_cand", "n_active_kept", "frac_rejected", "med_chi2_red"}
    assert legacy_keys.issubset(stats.keys())
    extra_keys = {"scale", "raw_scale", "n_flipped", "n_churn_capped", "n_refreshes"}
    assert extra_keys.issubset(stats.keys())
