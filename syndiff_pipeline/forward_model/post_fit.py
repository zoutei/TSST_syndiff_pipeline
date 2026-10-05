# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Post-fit diagnostics: position residuals vs centroids, w_k(t) curves, chi2.

Kept deliberately small -- returns arrays/DataFrames for the caller (notebook)
to plot, rather than owning a plotting stack.

Also hosts thin loss-landscape helpers (param directions + metric eval) used by
``notebooks/loss_landscape_smoke.ipynb``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.ndimage import shift as ndi_shift

from . import cheb_wcs as CW
from . import epsf_model as EM
from . import fit as FIT
from . import loss as L
from .groups import GroupSet


def fitted_vs_warmstart_positions(
    params: dict,
    ctx: L.StaticContext,
) -> tuple[np.ndarray, np.ndarray]:
    """(n_groups, K, n_frames) fitted x, y for every (group, slot, frame)."""
    templates, x_t, y_t, _ = L.forward_model(params, ctx)
    return np.asarray(x_t), np.asarray(y_t)


def position_residuals_vs_centroids(
    x_t: np.ndarray,
    y_t: np.ndarray,
    groups: GroupSet,
    stems: list[str],
    fit_stars: pd.DataFrame,
    centroid_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Per-frame median |du|, |dv| (px) between fitted positions and
    ``centroids_r1`` PSFPhotometry x_fit/y_fit -- plan §Verification gate 3/6.

    ``centroid_tables[stem]`` must have ``source_id, x_fit, y_fit`` columns
    (i.e. the QC-joined per-frame star table, see ``shared_wcs.fit_region_shared_wcs``).
    """
    rows = []
    n_groups, K, n_frames = x_t.shape
    source_ids = fit_stars["source_id"].to_numpy()
    for fi, stem in enumerate(stems):
        cen = centroid_tables.get(stem)
        if cen is None or cen.empty:
            continue
        cen_map = dict(zip(cen["source_id"], zip(cen["x_fit"], cen["y_fit"])))
        du_list, dv_list = [], []
        for gi in range(n_groups):
            for k in range(K):
                if not groups.valid[gi, k]:
                    continue
                sid = source_ids[groups.members[gi, k]]
                hit = cen_map.get(sid)
                if hit is None:
                    continue
                du_list.append(x_t[gi, k, fi] - hit[0])
                dv_list.append(y_t[gi, k, fi] - hit[1])
        if not du_list:
            continue
        du = np.abs(np.asarray(du_list))
        dv = np.abs(np.asarray(dv_list))
        rows.append({"stem": stem, "n_stars": len(du_list), "med_abs_du": np.median(du), "med_abs_dv": np.median(dv)})
    return pd.DataFrame(rows)


def epsf_base_health(params: dict) -> pd.DataFrame:
    """Per-node positivity / flux-sum checks on the decoded ``epsf_base`` grid."""
    if "epsf_base" in params:
        base = np.asarray(
            EM.convert_legacy_epsf_array(params["epsf_base"], name="params['epsf_base']")
        )
    elif "epsf_base_raw" in params:
        base = np.asarray(EM.decode_epsf_base(params["epsf_base_raw"]))
    else:
        raise KeyError("params must contain epsf_base or epsf_base_raw")

    rows = []
    n_rows, n_cols = base.shape[:2]
    for i in range(n_rows):
        for j in range(n_cols):
            node = base[i, j]
            rows.append(
                {
                    "node_i": i,
                    "node_j": j,
                    "min": float(node.min()),
                    "frac_negative": float((node < 0).mean()),
                    "sum": float(node.sum()),
                }
            )
    return pd.DataFrame(rows)


def w_curves(params: dict, w_frame_basis, btjd: np.ndarray) -> pd.DataFrame:
    """w_k(t), one column per configured mode (K = params['w_coeff'].shape[0]).

    Named by ``epsf_model.FD_MODE_NAMES`` only for the legacy full 5-mode case,
    where the canonical order is unambiguous; generic ``mode_k`` otherwise, since
    which physical FD vector(s) a given checkpoint's modes were initialized from
    isn't recoverable from ``params`` alone (see ``EM.init_epsf_from_prf``'s
    ``mode_names``).
    """
    import jax.numpy as jnp

    w_of_t = np.asarray((params["w_coeff"] @ jnp.asarray(w_frame_basis).T).T)  # (n_frames, K)
    n_modes = w_of_t.shape[1]
    if n_modes == len(EM.FD_MODE_NAMES):
        names = list(EM.FD_MODE_NAMES)
    else:
        names = [f"mode_{k}" for k in range(n_modes)]
    df = pd.DataFrame(w_of_t, columns=names)
    df.insert(0, "btjd", btjd)
    return df


def chi2_per_frame(
    data: np.ndarray, model: np.ndarray, noise: np.ndarray, weight: np.ndarray, *, t_exp_sec: float
) -> np.ndarray:
    """Reduced chi2 per frame (sum over groups/pixels / active pixel count).

    Uses ``σ² = NOISE² + ε`` (same as the fit). ``t_exp_sec`` is kept for
    call-site compatibility but is unused.
    """
    del t_exp_sec
    var = noise**2 + 1e-6
    chi2 = weight * (data - model) ** 2 / var
    axes = tuple(a for a in range(data.ndim) if a != 1)  # keep frame axis (1)
    n_active = np.sum(weight, axis=axes)
    return np.sum(chi2, axis=axes) / np.clip(n_active, 1.0, None)


# ---------------------------------------------------------------------------
# Loss-landscape helpers
# ---------------------------------------------------------------------------


def copy_params(params: dict[str, Any]) -> dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in params.items()}


def eval_loss_metrics(
    params: dict[str, jnp.ndarray],
    fd: FIT.FitData,
    *,
    weights: L.LossWeights | None = None,
) -> dict[str, float]:
    """Evaluate ``total_loss`` once; return plain float metrics."""
    w = fd.weights if weights is None else weights
    _, metrics = L.total_loss(
        params,
        fd.ctx,
        fd.data,
        fd.noise,
        fd.weight,
        fd.wcs_second_diff,
        fd.w_second_diff,
        epsf_modes_init=fd.epsf_modes_init,
        weights=w,
    )
    return {k: float(v) for k, v in metrics.items()}


def wcs_translation_direction(
    wcs_coeff: np.ndarray | jnp.ndarray,
    *,
    n_terms: int,
    axis: str = "x",
) -> jnp.ndarray:
    """Direction in ``wcs_coeff`` that adds ~1 px translation on ``axis``.

    Uses the Chebyshev constant term (Sci2Idl index 0) and the B-spline
    partition of unity: setting every temporal coeff of that term to 1 yields
    a frame-independent +1 px residual for all stars.
    """
    axis = axis.lower()
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    d = np.zeros(np.asarray(wcs_coeff).shape, dtype=np.float32)
    row = 0 if axis == "x" else int(n_terms)
    d[row, :] = 1.0
    return jnp.asarray(d)


def epsf_flux_centroid_xy(base: np.ndarray | jnp.ndarray) -> tuple[float, float]:
    """Flux-weighted centroid (physical px) of a decoded ``epsf_base`` field."""
    base_np = np.asarray(base, dtype=np.float64)
    coord = np.asarray(L.node_coord_grid(), dtype=np.float64)
    total = base_np.sum(axis=(-1, -2), keepdims=True) + 1e-12
    cx = (base_np * coord[None, None, None, :]).sum(axis=(-1, -2)) / total[..., 0, 0]
    cy = (base_np * coord[None, None, :, None]).sum(axis=(-1, -2)) / total[..., 0, 0]
    return float(np.mean(cx)), float(np.mean(cy))


def shift_epsf_base_physical(
    base: np.ndarray | jnp.ndarray,
    *,
    dx_phys: float = 1.0,
    dy_phys: float = 0.0,
) -> jnp.ndarray:
    """Shift decoded ePSF content by ``(dx, dy)`` physical pixels (all nodes).

    Positive ``dx_phys`` moves flux toward higher detector-x (higher array x
    index), matching ``centroid_penalty`` / ``node_coord_grid`` sign.
    """
    base_np = np.asarray(base, dtype=np.float64)
    # ndi_shift: shift along axes (..., y, x); positive shift moves values
    # toward higher indices when order uses the destination convention...
    # scipy.ndimage.shift: The input is shifted so feature at position p moves
    # to p + shift. We want content moved +dx in x → shift=(..., dy*os, dx*os).
    os_ = float(EM.OVERSAMPLE)
    out = np.empty_like(base_np, dtype=np.float64)
    for i in range(base_np.shape[0]):
        for j in range(base_np.shape[1]):
            moved = ndi_shift(
                base_np[i, j],
                shift=(dy_phys * os_, dx_phys * os_),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            moved = np.clip(moved, 0.0, None)
            s = moved.sum()
            if s > 0:
                moved /= s
            else:
                moved = base_np[i, j]
            out[i, j] = moved
    return jnp.asarray(out, dtype=jnp.float32)


def epsf_centroid_shift_raw_direction(
    epsf_base_raw: np.ndarray | jnp.ndarray,
    *,
    dx_phys: float = 1.0,
    dy_phys: float = 0.0,
) -> jnp.ndarray:
    """Raw-parameter direction whose +1 step ≈ +``dx/dy`` px ePSF centroid move."""
    raw0 = jnp.asarray(epsf_base_raw)
    base0 = EM.decode_epsf_base(raw0)
    base1 = shift_epsf_base_physical(base0, dx_phys=dx_phys, dy_phys=dy_phys)
    raw1 = EM.encode_epsf_base(base1)
    return raw1 - raw0


def apply_leaf_step(
    params: dict[str, jnp.ndarray],
    leaf: str,
    direction: jnp.ndarray,
    alpha: float,
) -> dict[str, jnp.ndarray]:
    """Return a shallow copy of ``params`` with ``leaf += alpha * direction``."""
    out = copy_params(params)
    out[leaf] = out[leaf] + jnp.asarray(alpha, dtype=out[leaf].dtype) * direction
    return out


def apply_wcs_epsf_plane_step(
    params: dict[str, jnp.ndarray],
    *,
    wcs_dir: jnp.ndarray,
    epsf_dir: jnp.ndarray,
    alpha: float,
    beta: float,
) -> dict[str, jnp.ndarray]:
    """Perturb WCS by ``alpha`` and ``epsf_base_raw`` by ``beta`` along unit dirs."""
    out = copy_params(params)
    out["wcs_coeff"] = out["wcs_coeff"] + jnp.asarray(alpha, dtype=out["wcs_coeff"].dtype) * wcs_dir
    out["epsf_base_raw"] = (
        out["epsf_base_raw"] + jnp.asarray(beta, dtype=out["epsf_base_raw"].dtype) * epsf_dir
    )
    return out


def unit_direction_like(arr: jnp.ndarray, rng: np.random.Generator | None = None) -> jnp.ndarray:
    """Gaussian random direction with RMS 1 (Frobenius)."""
    rng = rng or np.random.default_rng(0)
    v = rng.normal(size=np.asarray(arr).shape).astype(np.float32)
    n = float(np.linalg.norm(v.ravel()) + 1e-12)
    return jnp.asarray(v / n)


def normalize_direction(arr: jnp.ndarray) -> jnp.ndarray:
    v = np.asarray(arr, dtype=np.float32)
    n = float(np.linalg.norm(v.ravel()) + 1e-12)
    return jnp.asarray(v / n)


def grad_leaf_direction(
    params: dict[str, jnp.ndarray],
    fd: FIT.FitData,
    leaf: str,
    *,
    weights: L.LossWeights | None = None,
) -> jnp.ndarray:
    """Unit-Frobenius gradient direction for one parameter leaf."""
    w = fd.weights if weights is None else weights

    def loss_fn(p):
        loss, _ = L.total_loss(
            p,
            fd.ctx,
            fd.data,
            fd.noise,
            fd.weight,
            fd.wcs_second_diff,
            fd.w_second_diff,
            epsf_modes_init=fd.epsf_modes_init,
            weights=w,
        )
        return loss

    grads = jax.grad(loss_fn)(params)
    return normalize_direction(grads[leaf])


def slice_loss_1d(
    params: dict[str, jnp.ndarray],
    fd: FIT.FitData,
    leaf: str,
    direction: jnp.ndarray,
    alphas: np.ndarray,
    *,
    weights: L.LossWeights | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate loss / data_term along ``params[leaf] + alpha * direction``."""
    w = fd.weights if weights is None else weights
    alphas = np.asarray(alphas, dtype=np.float32)
    direction = jnp.asarray(direction)

    def one(alpha):
        p = apply_leaf_step(params, leaf, direction, alpha)
        loss, metrics = L.total_loss(
            p,
            fd.ctx,
            fd.data,
            fd.noise,
            fd.weight,
            fd.wcs_second_diff,
            fd.w_second_diff,
            epsf_modes_init=fd.epsf_modes_init,
            weights=w,
        )
        return loss, metrics["data_term"]

    losses, data_terms = jax.vmap(one)(jnp.asarray(alphas))
    return {
        "alpha": alphas,
        "loss": np.asarray(losses),
        "data_term": np.asarray(data_terms),
    }


def loss_on_wcs_epsf_plane(
    params: dict[str, jnp.ndarray],
    fd: FIT.FitData,
    *,
    wcs_dir: jnp.ndarray,
    epsf_dir: jnp.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    weights: L.LossWeights | None = None,
) -> dict[str, np.ndarray]:
    """2D grid of ``loss`` / ``data_term`` / ``centroid`` on the degeneracy plane."""
    w = fd.weights if weights is None else weights
    alphas = np.asarray(alphas, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32)
    aa, bb = np.meshgrid(alphas, betas, indexing="xy")
    flat_a = aa.ravel()
    flat_b = bb.ravel()

    def one(alpha, beta):
        p = apply_wcs_epsf_plane_step(
            params, wcs_dir=wcs_dir, epsf_dir=epsf_dir, alpha=alpha, beta=beta,
        )
        loss, metrics = L.total_loss(
            p,
            fd.ctx,
            fd.data,
            fd.noise,
            fd.weight,
            fd.wcs_second_diff,
            fd.w_second_diff,
            epsf_modes_init=fd.epsf_modes_init,
            weights=w,
        )
        return loss, metrics["data_term"], metrics["centroid"]

    loss_f, data_f, cen_f = jax.vmap(one)(jnp.asarray(flat_a), jnp.asarray(flat_b))
    shape = aa.shape
    return {
        "alphas": alphas,
        "betas": betas,
        "loss": np.asarray(loss_f).reshape(shape),
        "data_term": np.asarray(data_f).reshape(shape),
        "centroid": np.asarray(cen_f).reshape(shape),
    }


def run_stage1_adam(
    params: dict[str, jnp.ndarray],
    fd: FIT.FitData,
    *,
    n_steps: int,
    lr: float,
    grad_clip: float = 1.0,
) -> tuple[dict[str, jnp.ndarray], list[dict]]:
    """Stage-1 Adam (WCS only), logging every step."""
    return FIT.run_stage(
        copy_params(params),
        fd,
        stage=1,
        n_steps=n_steps,
        lr=lr,
        log_every=1,
        grad_clip=grad_clip,
        history_path=None,
    )


def run_stage1_lbfgs(
    params: dict[str, jnp.ndarray],
    fd: FIT.FitData,
    *,
    n_steps: int,
    memory_size: int = 10,
) -> tuple[dict[str, jnp.ndarray], list[dict]]:
    """Stage-1 L-BFGS on ``wcs_coeff`` only (other leaves held fixed)."""
    import optax
    import time

    fixed = copy_params(params)
    wcs0 = fixed["wcs_coeff"]
    w = fd.weights

    def loss_from_wcs(wcs_coeff):
        p = {**fixed, "wcs_coeff": wcs_coeff}
        loss, metrics = L.total_loss(
            p,
            fd.ctx,
            fd.data,
            fd.noise,
            fd.weight,
            fd.wcs_second_diff,
            fd.w_second_diff,
            epsf_modes_init=fd.epsf_modes_init,
            weights=w,
        )
        return loss, metrics

    def value_fn(wcs_coeff):
        loss, _ = loss_from_wcs(wcs_coeff)
        return loss

    tx = optax.lbfgs(memory_size=memory_size)
    opt_state = tx.init(wcs0)
    history: list[dict] = []
    wcs = wcs0
    t0 = time.time()

    @jax.jit
    def step(wcs_coeff, state):
        (loss, metrics), grad = jax.value_and_grad(loss_from_wcs, has_aux=True)(wcs_coeff)
        updates, new_state = tx.update(
            grad,
            state,
            wcs_coeff,
            value=loss,
            grad=grad,
            value_fn=value_fn,
        )
        new_wcs = optax.apply_updates(wcs_coeff, updates)
        return new_wcs, new_state, loss, metrics

    for i in range(n_steps):
        wcs, opt_state, loss, metrics = step(wcs, opt_state)
        m = {k: float(v) for k, v in metrics.items()}
        m["step"] = i
        m["elapsed_s"] = time.time() - t0
        m["stage"] = 1
        m["solver"] = "lbfgs"
        history.append(m)

    out = copy_params(params)
    out["wcs_coeff"] = wcs
    return out, history


def write_gaia_membership_regions(
    path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    region_x_min: int,
    region_y_min: int,
    pool_indices: np.ndarray,
    primary_indices: np.ndarray,
    companion_indices: np.ndarray,
    rejected_primary_indices: np.ndarray | None = None,
    tess_mag: np.ndarray | None = None,
    include_pool: bool = True,
    radius_px: float = 2.5,
) -> None:
    """Write a DS9 region file for Gaia membership on a region mosaic FITS.

    Coordinates are DS9 **image** (1-based) relative to the region mosaic:
    ``x_img = x - region_x_min + 1``, same for y (``x,y`` are crop-local like
    the WCS model).

    **One color per star** (highest wins):
    rejected primary → magenta; companion → cyan; primary kept → red;
    remaining pool → green (omit pool with ``include_pool=False``).

    When ``tess_mag`` is given, non-pool stars are labeled with ``Tmag`` to
    two decimals (no role text). Pool stars stay unlabeled if drawn.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rejected = {int(i) for i in (rejected_primary_indices if rejected_primary_indices is not None else [])}
    primary_set = {int(i) for i in primary_indices}
    companion_set = {int(i) for i in companion_indices}
    pool_set = {int(i) for i in pool_indices}
    mags = None if tess_mag is None else np.asarray(tess_mag, dtype=float)

    def _label(i: int, role: str) -> str:
        if role == "pool" or mags is None:
            return ""
        return f"{float(mags[i]):.2f}"

    def _circle(i: int, color: str, role: str) -> str:
        xi = float(x[i]) - float(region_x_min) + 1.0
        yi = float(y[i]) - float(region_y_min) + 1.0
        lab = _label(i, role)
        if lab:
            return f"circle({xi:.3f},{yi:.3f},{radius_px}) # color={color} width=2 text={{{lab}}}"
        return f"circle({xi:.3f},{yi:.3f},{radius_px}) # color={color} width=2"

    # Exclusive assignment: rejected > companion > primary > pool
    role: dict[int, str] = {}
    if include_pool:
        for i in pool_set:
            role[i] = "pool"
    for i in primary_set:
        role[i] = "primary"
    for i in companion_set:
        role[i] = "companion"
    for i in rejected:
        if i in primary_set:
            role[i] = "primary_rej"

    color_for = {
        "pool": "green",
        "primary": "red",
        "companion": "cyan",
        "primary_rej": "magenta",
    }

    lines = [
        "# Region file format: DS9 version 4.1",
        "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "image",
    ]
    order = ("pool", "primary", "companion", "primary_rej")
    by_role: dict[str, list[int]] = {r: [] for r in order}
    for i, r in role.items():
        by_role[r].append(i)
    for r in order:
        for i in sorted(by_role[r]):
            lines.append(_circle(i, color_for[r], r))

    path.write_text("\n".join(lines) + "\n")
