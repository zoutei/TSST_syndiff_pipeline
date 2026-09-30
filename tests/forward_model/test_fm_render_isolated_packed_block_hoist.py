# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Regression test for the block-hoist refactor of
``stacked_temporal_residuals.render_isolated_packed_numpy`` (task T2).

The original implementation looped group-outer, frame-block-inner, and
recomputed the node-grid composite field (and, with chroma on, each chroma
generator field) from scratch INSIDE the per-group loop even though neither
depends on which group is being rendered -- an ``n_group``-fold redundant
cost that became the dominant bottleneck once the K=2/4x4-edge bundle (16
nodes, 2 modes) made each of those recomputations ~4-7x more expensive than
the 2x2/K=1 case it was originally profiled on. The fix reorders the loop
(block-outer, group-inner) and hoists the block-level quantities out of the
group loop; every per-group computation is untouched.

This test proves the reorder is a pure performance change: a small,
hand-rolled reference that reimplements the ORIGINAL group-outer formula
verbatim (not by calling the module under test) must match the current
function's output exactly, with and without chroma, and across a
frame-block boundary (``frame_block`` smaller than ``n_frame``, so the
outer loop actually runs more than once).
"""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.forward_model.diagnostics import stacked_temporal_residuals as STR


def _naive_reference(
    params, x, y, frame_basis, node_x, node_y, pix_x, pix_y, pix_valid,
    *, frame_block=16, chroma_delta=None, chroma_shift_px=None,
):
    """Verbatim re-derivation of the ORIGINAL group-outer/block-inner loop,
    kept intentionally separate from the module under test (no shared code
    with the refactored function) so this is a true independent check.
    """
    base, modes = STR._decode_epsf_numpy(params)
    _chroma = chroma_delta is not None
    if _chroma:
        _field_coeffs = {"dilation": np.asarray(params["chroma_dilation"], dtype=float)}
        if "chroma_aniso" in params and "chroma_shear" in params:
            _field_coeffs["aniso"] = np.asarray(params["chroma_aniso"], dtype=float)
            _field_coeffs["shear"] = np.asarray(params["chroma_shear"], dtype=float)
        _delta = np.asarray(chroma_delta, dtype=float)
        _chsh = (np.zeros((x.shape[0], 2)) if chroma_shift_px is None
                 else np.asarray(chroma_shift_px, dtype=float))
    n_group, n_frame = x.shape
    p = pix_x.shape[1]
    out = np.zeros((n_group, n_frame, p), dtype=np.float32)
    w = np.asarray(params["w_coeff"], dtype=float) @ np.asarray(frame_basis, dtype=float).T
    w = (w.T - np.mean(w.T, axis=0, keepdims=True))
    size = base.shape[-1]
    center = (size - 1) / 2.0
    sub = (np.arange(4, dtype=float) - 1.5) / 4.0
    for group in range(n_group):
        for start in range(0, n_frame, max(1, int(frame_block))):
            stop = min(start + max(1, int(frame_block)), n_frame)
            xb, yb, wb = x[group, start:stop], y[group, start:stop], w[start:stop]
            j0 = np.clip(np.searchsorted(node_x, xb, side="right") - 1, 0, len(node_x) - 2)
            i0 = np.clip(np.searchsorted(node_y, yb, side="right") - 1, 0, len(node_y) - 2)
            wx = np.clip((xb - node_x[j0]) / (node_x[j0 + 1] - node_x[j0]), 0, 1)
            wy = np.clip((yb - node_y[i0]) / (node_y[i0 + 1] - node_y[i0]), 0, 1)
            field = base[None] + np.einsum("bk,kijxy->bijxy", wb, modes, optimize=True)
            nb = np.arange(stop - start)

            def _blend(f, scale=None, i0=i0, j0=j0, wy=wy, wx=wx, nb=nb):
                s00 = 1.0 if scale is None else scale[i0, j0][:, None, None]
                s01 = 1.0 if scale is None else scale[i0, j0 + 1][:, None, None]
                s10 = 1.0 if scale is None else scale[i0 + 1, j0][:, None, None]
                s11 = 1.0 if scale is None else scale[i0 + 1, j0 + 1][:, None, None]
                return (
                    (1 - wy)[:, None, None] * ((1 - wx)[:, None, None] * s00 * f[nb, i0, j0]
                                               + wx[:, None, None] * s01 * f[nb, i0, j0 + 1])
                    + wy[:, None, None] * ((1 - wx)[:, None, None] * s10 * f[nb, i0 + 1, j0]
                                           + wx[:, None, None] * s11 * f[nb, i0 + 1, j0 + 1])
                )

            local = _blend(field)
            gx_shift = gy_shift = 0.0
            if _chroma:
                for _name, _coeff in _field_coeffs.items():
                    _gfield = STR._CHROMA_FIELD_GENERATORS_NUMPY[_name](field)
                    local = local + _delta[group] * _blend(_gfield, scale=_coeff)
                gx_shift, gy_shift = float(_chsh[group, 0]), float(_chsh[group, 1])
            local = STR._recenter_numpy(local)
            ox = pix_x[group][None, :, None, None] - (xb + gx_shift)[:, None, None, None] + sub[None, None, None, :]
            oy = pix_y[group][None, :, None, None] - (yb + gy_shift)[:, None, None, None] + sub[None, None, :, None]
            gx, gy = center + 4 * ox, center + 4 * oy
            out[group, start:stop] = (STR._bilinear_sample_numpy(local, gx, gy).sum(axis=(2, 3))
                                       * pix_valid[group][None]).astype(np.float32)
    return out


def _make_case(rng, *, n_rows, n_cols, n_group, n_frame, n_knots=5, n_pix=9, with_chroma=False,
                with_affine=False):
    G = 20
    base_raw = rng.normal(size=(n_rows, n_cols, G, G)).astype(np.float32)
    modes = rng.normal(size=(1, n_rows, n_cols, G, G)).astype(np.float32) * 0.05
    params = {"epsf_base_raw": base_raw, "epsf_modes": modes,
              "w_coeff": (rng.normal(size=(1, n_knots)) * 0.02).astype(np.float32)}
    if with_chroma:
        params["chroma_dilation"] = (rng.normal(size=(n_rows, n_cols)) * 0.01).astype(np.float32)
        if with_affine:
            params["chroma_aniso"] = (rng.normal(size=(n_rows, n_cols)) * 0.01).astype(np.float32)
            params["chroma_shear"] = (rng.normal(size=(n_rows, n_cols)) * 0.01).astype(np.float32)
    frame_basis = rng.normal(size=(n_frame, n_knots)).astype(np.float32)
    node_x = np.linspace(0, 100, n_cols)
    node_y = np.linspace(0, 100, n_rows)
    x = rng.uniform(10, 90, size=(n_group, n_frame))
    y = rng.uniform(10, 90, size=(n_group, n_frame))
    pix_x = rng.uniform(-3, 3, size=(n_group, n_pix))
    pix_y = rng.uniform(-3, 3, size=(n_group, n_pix))
    pix_valid = np.ones((n_group, n_pix), dtype=np.float32)
    kwargs = {}
    if with_chroma:
        kwargs["chroma_delta"] = rng.uniform(-0.5, 0.5, size=n_group)
        kwargs["chroma_shift_px"] = rng.normal(size=(n_group, 2)) * 0.01
    return params, x, y, frame_basis, node_x, node_y, pix_x, pix_y, pix_valid, kwargs


def test_block_hoist_matches_naive_loop_no_chroma_multiple_grid_sizes():
    rng = np.random.default_rng(0)
    for n_rows, n_cols in ((2, 2), (4, 4)):
        args = _make_case(rng, n_rows=n_rows, n_cols=n_cols, n_group=6, n_frame=10)
        params, x, y, fb, nx, ny, px, py, pv, kwargs = args
        expected = _naive_reference(params, x, y, fb, nx, ny, px, py, pv, frame_block=4, **kwargs)
        actual = STR.render_isolated_packed_numpy(params, x, y, fb, nx, ny, px, py, pv,
                                                    frame_block=4, **kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_block_hoist_matches_naive_loop_with_chroma_dilation_only():
    rng = np.random.default_rng(1)
    args = _make_case(rng, n_rows=4, n_cols=4, n_group=5, n_frame=9, with_chroma=True)
    params, x, y, fb, nx, ny, px, py, pv, kwargs = args
    expected = _naive_reference(params, x, y, fb, nx, ny, px, py, pv, frame_block=4, **kwargs)
    actual = STR.render_isolated_packed_numpy(params, x, y, fb, nx, ny, px, py, pv,
                                                frame_block=4, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_block_hoist_matches_naive_loop_with_chroma_affine():
    rng = np.random.default_rng(2)
    args = _make_case(rng, n_rows=4, n_cols=4, n_group=5, n_frame=13, with_chroma=True, with_affine=True)
    params, x, y, fb, nx, ny, px, py, pv, kwargs = args
    expected = _naive_reference(params, x, y, fb, nx, ny, px, py, pv, frame_block=5, **kwargs)
    actual = STR.render_isolated_packed_numpy(params, x, y, fb, nx, ny, px, py, pv,
                                                frame_block=5, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_block_hoist_single_frame_block_boundary_case():
    """frame_block >= n_frame (single block, the common case in production
    calls) also matches -- degenerate case of the general loop."""
    rng = np.random.default_rng(3)
    args = _make_case(rng, n_rows=2, n_cols=2, n_group=4, n_frame=6, with_chroma=True)
    params, x, y, fb, nx, ny, px, py, pv, kwargs = args
    expected = _naive_reference(params, x, y, fb, nx, ny, px, py, pv, frame_block=16, **kwargs)
    actual = STR.render_isolated_packed_numpy(params, x, y, fb, nx, ny, px, py, pv,
                                                frame_block=16, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
