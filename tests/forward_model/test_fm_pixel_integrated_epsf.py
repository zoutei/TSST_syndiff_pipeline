# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Pixel-integrated (Anderson & King) ePSF representation -- required tests.

See CONTRACT_pixel_integrated_epsf.md. Every render path must reproduce the
LEGACY sub-pixel-grid renderer (``EM._legacy_subpixel_render_stamps``, kept
private in ``epsf_model.py`` for exactly this purpose) once the legacy grid
is converted via ``EM.to_pixel_integrated`` -- to fp64 rtol 1e-12, on a grid
whose outer PAD_SAMPLES-sample ring is zero (the documented equivalence
precondition: the conversion is only exact when no signal sits in the ring
the box-sum can't fully cover on both representations' boundary handling).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _enable_x64_scoped():
    """Enable fp64 for exactness assertions in THIS module only.

    A bare module-level ``jax.config.update("jax_enable_x64", True)`` runs
    during pytest COLLECTION (before any test executes) and is a process-
    global flag -- it would leak into every other test file's session,
    silently flipping default float dtypes suite-wide (this was caught: it
    broke unrelated ``jax.lax.scan``-based chunked-loss tests elsewhere with
    a float32-vs-float64 carry-type mismatch). Scope it to each test here
    instead, restoring the prior value immediately after.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)

from syndiff_pipeline.forward_model import epsf_model as EM  # noqa: E402


def _zero_ring_legacy_grid(seed: int = 0, *, asymmetric: bool = True) -> np.ndarray:
    """A compact, PSF-like legacy (sub-pixel ``P``, size 58) grid with its
    outer ``PAD_SAMPLES``-sample ring exactly zero (equivalence precondition).
    """
    rng = np.random.default_rng(seed)
    G = EM.SUBPIXEL_GRID_SIZE
    c = EM.SUBPIXEL_CENTER_INDEX
    idx = np.arange(G, dtype=np.float64)
    r2 = (idx[:, None] - c) ** 2 + (idx[None, :] - c) ** 2
    core = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.3) ** 2))
    grid = core.copy()
    if asymmetric:
        wing_r2 = (idx[:, None] - c - 2.0) ** 2 + (idx[None, :] - c + 1.5) ** 2
        grid = grid + 0.25 * np.exp(-wing_r2 / (2 * (EM.OVERSAMPLE * 2.0) ** 2))
        grid = grid + 0.02 * rng.standard_normal(grid.shape)
    grid = np.clip(grid, 0.0, None)
    P = EM.PAD_SAMPLES
    grid[:P, :] = 0.0
    grid[-P:, :] = 0.0
    grid[:, :P] = 0.0
    grid[:, -P:] = 0.0
    return grid / grid.sum()


_PHASES = (
    0.0, 0.05, -0.05, 0.13, -0.13, 0.25, -0.25, 0.3, -0.3, 0.42, -0.42,
    0.49, -0.49, 0.499999, -0.499999,
)


# ---------------------------------------------------------------------------
# Test 1: equivalence of every render path vs. the private legacy reference
# ---------------------------------------------------------------------------


def test_equivalence_render_stamps():
    P = _zero_ring_legacy_grid(0)
    E = np.asarray(EM.to_pixel_integrated(jnp.asarray(P)))
    assert E.shape == (EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)

    worst = 0.0
    for dx in _PHASES:
        for dy in _PHASES:
            legacy = np.asarray(
                EM._legacy_subpixel_render_stamps(
                    jnp.asarray(P)[None, None], jnp.asarray([[dx]]), jnp.asarray([[dy]])
                )
            )[0, 0]
            new = np.asarray(
                EM.render_stamps(
                    jnp.asarray(E)[None, None], jnp.asarray([[dx]]), jnp.asarray([[dy]])
                )
            )[0, 0]
            scale = np.abs(legacy).max() + 1e-300
            worst = max(worst, float(np.abs(legacy - new).max() / scale))
    assert worst < 1e-12, f"render_stamps equivalence worst rel err {worst}"


def test_equivalence_axis_band_einsum():
    """``render_stamps`` reconstructed via the closed-form ``axis_band``
    matrices (the einsum identity ``axis_band`` documents) against the
    legacy reference."""
    P = _zero_ring_legacy_grid(1)
    E = np.asarray(EM.to_pixel_integrated(jnp.asarray(P)))
    worst = 0.0
    for dx in _PHASES:
        for dy in _PHASES:
            legacy = np.asarray(
                EM._legacy_subpixel_render_stamps(
                    jnp.asarray(P)[None, None], jnp.asarray([[dx]]), jnp.asarray([[dy]])
                )
            )[0, 0]
            Ay = EM.axis_band(jnp.asarray(dy))
            Ax = EM.axis_band(jnp.asarray(dx))
            got = np.asarray(jnp.einsum("sg,gh,th->st", Ay, jnp.asarray(E), Ax))
            scale = np.abs(legacy).max() + 1e-300
            worst = max(worst, float(np.abs(legacy - got).max() / scale))
    assert worst < 1e-12, f"axis_band equivalence worst rel err {worst}"


def test_equivalence_render_band_folded_com_shift():
    """``render_band`` (the folded-COM-shift render path) two ways:

    (1) Cross-representation equivalence at ``off_com=0`` (fold degenerates
    to a no-op, so this must agree with the legacy renderer exactly like
    ``axis_band`` does -- redundant with the ``axis_band`` test, but
    ``render_band`` is listed as its own render path in the contract).

    (2) The fold's OWN defining identity, checked self-consistently within
    the pixel-integrated (``E``) representation alone (there is no
    well-posed cross-representation check at nonzero COM: a windowed,
    discretely-sampled core centroid computed on two different samplings --
    2-tap-native vs 4-sample-block -- of the same underlying continuum need
    not agree bit-for-bit, so folding an *equal* COM value into each
    representation's own render kernel is not expected to produce
    numerically-equal stamps; ``test_render_band_matches_recentered_render_stamps``
    in test_forward_epsf_wcs.py already covers exactly this identity, now
    reproduced here directly on a zero-ring grid for the fp64 tolerance).
    """
    P = _zero_ring_legacy_grid(2)
    E = np.asarray(EM.to_pixel_integrated(jnp.asarray(P)))
    E_j = jnp.asarray(E)

    # (1) off_com=0 cross-representation check.
    worst = 0.0
    for dx in _PHASES[:8]:
        for dy in _PHASES[:8]:
            legacy = np.asarray(
                EM._legacy_subpixel_render_stamps(
                    jnp.asarray(P)[None, None], jnp.array([[dx]]), jnp.array([[dy]])
                )
            )[0, 0]
            Ay = EM.render_band(jnp.asarray([dy]), jnp.asarray([0.0]))
            Ax = EM.render_band(jnp.asarray([dx]), jnp.asarray([0.0]))
            got = np.asarray(jnp.einsum("nsg,ngh,nth->nst", Ay, E_j[None], Ax)[0])
            scale = np.abs(legacy).max() + 1e-300
            worst = max(worst, float(np.abs(legacy - got).max() / scale))
    assert worst < 1e-12, f"render_band (com=0) equivalence worst rel err {worst}"

    # (2) Fold identity, self-consistent within E (UNNORMALIZED on both
    # sides -- renorm_scalar's own identity with bilinear_shift_physical's
    # total mass is already covered by test_renorm_scalar_matches_bilinear_shift_sum):
    # render_band(dx, com) einsum'd against E == render_stamps applied to E
    # pre-shifted by the SAME com, for an arbitrary (not necessarily the
    # grid's own centroid) com.
    worst2 = 0.0
    for dx in _PHASES[:6]:
        for dy in _PHASES[:6]:
            for cx0, cy0 in ((0.02, -0.015), (-0.3, 0.41), (0.49, -0.49)):
                cx, cy = jnp.asarray(cx0), jnp.asarray(cy0)
                shifted = EM.bilinear_shift_physical(E_j, -cx, -cy)[None, None]
                ref = np.asarray(
                    EM.render_stamps(shifted, jnp.array([[dx]]), jnp.array([[dy]]))
                )[0, 0]
                Ay = EM.render_band(jnp.asarray([dy]), cy[None])
                Ax = EM.render_band(jnp.asarray([dx]), cx[None])
                got = np.asarray(jnp.einsum("nsg,ngh,nth->nst", Ay, E_j[None], Ax)[0])
                scale = np.abs(ref).max() + 1e-300
                worst2 = max(worst2, float(np.abs(ref - got).max() / scale))
    assert worst2 < 1e-10, f"render_band fold self-consistency worst rel err {worst2}"


def test_equivalence_render_physical_pixels_blocksum():
    P = _zero_ring_legacy_grid(3)
    E = np.asarray(EM.to_pixel_integrated(jnp.asarray(P)))
    S = EM.STAMP_PHYSICAL
    pidx = np.arange(S) - (S - 1) / 2.0
    py, px = np.meshgrid(pidx, pidx, indexing="ij")

    worst = 0.0
    for dx in _PHASES[:8]:
        for dy in _PHASES[:8]:
            legacy = np.asarray(
                EM._legacy_subpixel_render_stamps(
                    jnp.asarray(P)[None, None], jnp.asarray([[dx]]), jnp.asarray([[dy]])
                )
            )[0, 0]
            ox = jnp.asarray((px - dx).reshape(1, -1))
            oy = jnp.asarray((py - dy).reshape(1, -1))
            got = np.asarray(
                EM.render_physical_pixels_blocksum(jnp.asarray(E)[None], ox, oy)
            )[0].reshape(S, S)
            scale = np.abs(legacy).max() + 1e-300
            worst = max(worst, float(np.abs(legacy - got).max() / scale))
    assert worst < 1e-12, f"render_physical_pixels_blocksum equivalence worst rel err {worst}"


def test_equivalence_render_packed_pixels_banded():
    P = _zero_ring_legacy_grid(4)
    E = np.asarray(EM.to_pixel_integrated(jnp.asarray(P)))
    E_j = jnp.asarray(E)
    S = EM.STAMP_PHYSICAL
    pidx = np.arange(S) - (S - 1) / 2.0
    py, px = np.meshgrid(pidx, pidx, indexing="ij")
    pix_x_index = jnp.asarray(px.reshape(1, -1).astype(np.int64))
    pix_y_index = jnp.asarray(py.reshape(1, -1).astype(np.int64))
    com0 = jnp.asarray([0.0])

    worst = 0.0
    for dx in _PHASES[:8]:
        for dy in _PHASES[:8]:
            legacy = np.asarray(
                EM._legacy_subpixel_render_stamps(
                    jnp.asarray(P)[None, None], jnp.asarray([[dx]]), jnp.asarray([[dy]])
                )
            )[0, 0]
            got = np.asarray(
                EM.render_packed_pixels_banded(
                    E_j[None], pix_x_index, pix_y_index,
                    jnp.asarray([dx]), jnp.asarray([dy]), com0, com0,
                )
            )[0].reshape(S, S)
            scale = np.abs(legacy).max() + 1e-300
            worst = max(worst, float(np.abs(legacy - got).max() / scale))
    assert worst < 1e-12, f"render_packed_pixels_banded equivalence worst rel err {worst}"


# ---------------------------------------------------------------------------
# Test 2: no invisible directions
# ---------------------------------------------------------------------------


def _legacy_axis_matrix_reference(off: float) -> np.ndarray:
    S, os_, G, PAD = EM.STAMP_PHYSICAL, EM.OVERSAMPLE, EM.SUBPIXEL_GRID_SIZE, EM.PAD_SAMPLES
    m = np.arange(S * os_)
    g = m - off * os_ + PAD
    inb = (g >= 0.0) & (g <= float(G - 1))
    x0 = np.clip(np.floor(g).astype(int), 0, G - 2)
    f = np.clip(g - x0, 0.0, 1.0)
    W = np.zeros((S * os_, G))
    W[m, x0] += (1 - f) * inb
    W[m, x0 + 1] += f * inb
    return W.reshape(S, os_, G).sum(axis=1)


def test_no_invisible_directions_full_rank_on_reachable_columns():
    """Stack the per-axis render matrix over 200 phases spanning the entire
    physically valid primary-star offset range ``[-0.5, 0.5)``.

    Both the legacy and pixel-integrated grids have a handful of columns
    that are the OUTERMOST pad sample on each side, which no primary-star
    ``dx`` in this range ever addresses with nonzero weight (a plain
    boundary/reach artifact of the fixed padding vs. a bilinear 2-tap -- or
    5-tap -- kernel, identical in kind for both representations, and
    unrelated to the aliasing bug; those columns are only ever populated by
    far-separation companions through the packed/irregular render path).
    Excluding those trivial all-zero columns, the LEGACY sub-pixel grid's
    remaining (populated) columns are still rank-DEFICIENT by 3 (the "5
    invisible directions per axis" claim = 2 boundary + 3 genuine period-4
    aliasing directions) -- the new pixel-integrated grid's populated
    columns are FULL rank: zero genuine invisible directions left.
    """
    phases = np.linspace(-0.5, 0.5, 200, endpoint=False)

    legacy_mats = [_legacy_axis_matrix_reference(float(p)) for p in phases]
    legacy_stack = np.concatenate(legacy_mats, axis=0)
    legacy_colsum = np.abs(legacy_stack).max(axis=0)
    legacy_populated = np.where(legacy_colsum > 1e-9)[0]
    legacy_rank = np.linalg.matrix_rank(legacy_stack[:, legacy_populated])
    assert legacy_stack.shape[1] == EM.SUBPIXEL_GRID_SIZE
    assert len(legacy_populated) == EM.SUBPIXEL_GRID_SIZE - 2  # 2 boundary columns
    assert legacy_rank == len(legacy_populated) - 3  # genuine period-4 deficiency
    assert np.linalg.matrix_rank(legacy_stack) == 53  # matches the measured "53 of 58"

    new_mats = [np.asarray(EM.axis_band(jnp.asarray(float(p)))) for p in phases]
    new_stack = np.concatenate(new_mats, axis=0)
    assert new_stack.shape[1] == EM.NODE_GRID_SIZE
    new_colsum = np.abs(new_stack).max(axis=0)
    new_populated = np.where(new_colsum > 1e-9)[0]
    assert len(new_populated) == EM.NODE_GRID_SIZE - 2  # same trivial boundary artifact
    new_rank = np.linalg.matrix_rank(new_stack[:, new_populated])
    assert new_rank == len(new_populated), (
        "pixel-integrated axis_band must have NO invisible directions among "
        f"its reachable columns (got rank {new_rank} of {len(new_populated)})"
    )

    # A wide sweep (far-separation companion offsets) reaches every column of
    # both grids and both then trivially attain full rank -- confirming the
    # 2 "boundary" columns above are a reach artifact, not a true null
    # direction of either representation.
    wide_phases = np.linspace(-3.0, 3.0, 400, endpoint=False)
    legacy_wide = np.concatenate(
        [_legacy_axis_matrix_reference(float(p)) for p in wide_phases], axis=0
    )
    new_wide = np.concatenate(
        [np.asarray(EM.axis_band(jnp.asarray(float(p)))) for p in wide_phases], axis=0
    )
    assert np.linalg.matrix_rank(legacy_wide) == EM.SUBPIXEL_GRID_SIZE
    assert np.linalg.matrix_rank(new_wide) == EM.NODE_GRID_SIZE


# ---------------------------------------------------------------------------
# Test 3: flux rule
# ---------------------------------------------------------------------------


def _compact_raw_base(seed: int = 0) -> jnp.ndarray:
    """A raw (pre-decode) leaf whose DECODED grid is a compact, already
    near-centered PSF-like shape -- representative of a real fitted ePSF
    (unlike white noise, whose huge core-centroid pulls a large recenter
    shift that genuinely perturbs phase-class sums away from the lattice;
    see CONTRACT_pixel_integrated_epsf.md -- recentering only "preserves
    equal class sums away from the edges" for an already-small shift).
    """
    G = EM.NODE_GRID_SIZE
    c = EM.NODE_CENTER_INDEX
    idx = np.arange(G, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r2 = (idx[:, None] - c) ** 2 + (idx[None, :] - c) ** 2
    gauss = np.exp(-r2 / (2 * (EM.OVERSAMPLE * 1.4) ** 2))
    gauss = gauss + 0.01 * rng.standard_normal(gauss.shape) * gauss  # small asymmetric texture
    gauss = np.clip(gauss, 1e-9, None)
    gauss = gauss / gauss.sum()
    return jnp.asarray(EM.encode_epsf_base(jnp.asarray(gauss)))


def test_flux_rule_phase_class_sums_and_phase_independent_total_flux():
    raw = _compact_raw_base(0)[None, None]  # (1, 1, G, G)
    base = EM.decode_epsf_base(raw)

    pcs = np.asarray(EM.phase_class_sums(base))
    assert np.allclose(pcs, 1.0, atol=1e-6), f"phase_class_sums max dev {np.abs(pcs - 1).max()}"
    rms = float(EM.phase_flux_rms(base))
    assert rms < 1e-6

    # Rendered total flux must not depend on the star's sub-pixel phase.
    fluxes = []
    for dx in (0.0, 0.1, 0.25, -0.25, 0.4, -0.4, 0.499, -0.499):
        for dy in (0.0, -0.13, 0.3):
            stamp = EM.render_stamps(base, jnp.array([[dx]]), jnp.array([[dy]]))
            fluxes.append(float(jnp.sum(stamp)))
    fluxes = np.asarray(fluxes)
    assert np.allclose(fluxes, fluxes[0], atol=2e-4), (
        f"flux varies with phase: min={fluxes.min()} max={fluxes.max()}"
    )


def test_phase_class_sums_shape_and_scale():
    O = EM.OVERSAMPLE
    rng = np.random.default_rng(7)
    grid = rng.random((2, 3, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE))
    grid = grid / grid.sum(axis=(-2, -1), keepdims=True)
    pcs = np.asarray(EM.phase_class_sums(jnp.asarray(grid)))
    assert pcs.shape == (2, 3, O, O)
    # Sums of all O*O class sums / O**2 must equal the grid's total flux (~1).
    total_from_classes = pcs.sum(axis=(-2, -1)) / (O ** 2)
    assert np.allclose(total_from_classes, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: legacy load conversion
# ---------------------------------------------------------------------------


def test_legacy_load_conversion_matches_legacy_render():
    """A legacy (58-grid) params dict/bundle converts, re-decodes, and
    renders the same stamps as the legacy renderer on the legacy params, to
    ~1e-5 relative."""
    raw_legacy = np.asarray(
        _legacy_raw_base_from_decoded(_zero_ring_legacy_grid(5, asymmetric=False))
    )[None, None]

    # "Legacy world": decode + render with the OLD (pre-representation-change)
    # semantics -- softplus/normalize/recenter directly on P, block-sum render.
    pos = np.asarray(jax.nn.softplus(jnp.asarray(raw_legacy))) + 1e-8
    legacy_base = pos / pos.sum(axis=(-2, -1), keepdims=True)
    legacy_base = np.asarray(
        EM.recenter_grid_core(jnp.asarray(legacy_base), clip_nonneg=True)
    )
    legacy_stamp = np.asarray(
        EM._legacy_subpixel_render_stamps(
            jnp.asarray(legacy_base), jnp.array([[0.13]]), jnp.array([[-0.22]])
        )
    )[0, 0]

    # New world: detect + convert via the shared load helper, then decode +
    # render with the new pixel-integrated semantics.
    assert EM.is_subpixel_grid(raw_legacy.shape[-1])
    converted_decoded = EM.convert_legacy_epsf_array(legacy_base, name="test")
    assert converted_decoded.shape[-1] == EM.NODE_GRID_SIZE
    # Re-encode + re-decode (as a real load site does for a raw leaf) to
    # confirm round-tripping through encode/decode doesn't drift.
    reencoded = EM.encode_epsf_base(converted_decoded)
    redecoded = EM.decode_epsf_base(reencoded[None, None])
    new_stamp = np.asarray(
        EM.render_stamps(redecoded, jnp.array([[0.13]]), jnp.array([[-0.22]]))
    )[0, 0]

    scale = np.abs(legacy_stamp).max() + 1e-300
    rel = np.abs(legacy_stamp - new_stamp).max() / scale
    # CONTRACT: "~1e-5 relative" -- a second decode pass (phase-flux-rule +
    # core recenter, both absent from the old legacy decode) legitimately
    # perturbs the grid at roughly this level; this is not the fp64 render
    # equivalence (that's rtol 1e-12, tested above on raw arrays with no
    # second decode in between).
    assert rel < 5e-5, f"legacy-load conversion relative diff {rel}"


def _legacy_raw_base_from_decoded(decoded_p: np.ndarray) -> np.ndarray:
    """Inverse softplus (legacy, no phase-flux-rule) so re-decoding recovers
    (approximately) the same array before recenter."""
    clipped = np.clip(decoded_p, 1e-8, None)
    return clipped + np.log(-np.expm1(-clipped))


def test_init_epsf_from_base_converts_legacy_grid():
    legacy_decoded = _zero_ring_legacy_grid(6, asymmetric=False)
    legacy_decoded = np.asarray(
        EM.recenter_grid_core(jnp.asarray(legacy_decoded)[None, None], clip_nonneg=True)
    )  # (1, 1, G_P, G_P), satisfies decode_epsf_base's gauges in the legacy world
    params = EM.init_epsf_from_base(legacy_decoded, mode_names=("iso_defocus",))
    assert params.base.shape[-1] == EM.NODE_GRID_SIZE
    assert params.modes.shape[-1] == EM.NODE_GRID_SIZE


def test_is_subpixel_grid_and_stamp_physical_roundtrip():
    assert EM.is_subpixel_grid(EM.SUBPIXEL_GRID_SIZE)
    assert not EM.is_subpixel_grid(EM.NODE_GRID_SIZE)
    for S in (9, 11, 13, 15, 17):
        _, node_e, _ = EM.node_geometry(S)
        _, node_p, _ = EM.node_geometry(S, legacy=True)
        assert EM.stamp_physical_from_node_size(node_e) == S
        assert EM.stamp_physical_from_node_size(node_p) == S
        assert EM.is_subpixel_grid(node_p)
        assert not EM.is_subpixel_grid(node_e)


def test_to_pixel_integrated_conserves_flux_and_shape():
    P = _zero_ring_legacy_grid(9)
    E = np.asarray(EM.to_pixel_integrated(jnp.asarray(P, dtype=jnp.float64)))
    assert E.shape == (EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)
    assert abs(float(E.sum()) - float(P.sum())) < 1e-12

    # Linearity: converting a batch or a single mode (possibly zero-sum)
    # behaves the same as converting each slice independently.
    stack = np.stack([P, 2.0 * P, -P], axis=0)
    batch = np.asarray(EM.to_pixel_integrated(jnp.asarray(stack)))
    for k, scale in enumerate((1.0, 2.0, -1.0)):
        assert np.allclose(batch[k], scale * E, atol=1e-12)
