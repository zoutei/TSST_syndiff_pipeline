# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Brightness-dependent PSF width ("bright width", 2026-09-30).

Physics (measured in dev_runs/bfwidth_20260929, README "Conclusions"): bright TESS stars are
broader than faint ones. The best out-of-fold law is ONE global parameter, an isotropic
Gaussian blur proportional to the star's peak charge per 2-s read ``q``::

    P_s = P + 1/2 b (q_s - q_ref) (P_xx + P_yy),   b ~ 0.80e-3 px^2 (DeltaSigma) per 1e4 e-

where DeltaSigma is the extra blur covariance PER AXIS (px^2) and

    q_s = 2.0 s * f_s [e-/s] * (peak-pixel fraction of the star's rendered, pixel-integrated,
                               unit-flux template).

Implementation. The term rides on the renderer's existing per-slot colour machinery: it is an
extra per-slot weight on the raw-P-gauged blur generator ``blur_raw``. Under the A3 default
(``ctx.chroma_g8_gauge == "raw"``) that is exactly the colour model's blur field and the weight
is summed into it; otherwise (legacy mean gauge, node colour model, no colour model) a
``blur_raw`` field is created. The mean-gauge ``blur`` generator is deliberately NOT used:
its flat sheet alpha*mean(P) dominates the second moments (measured stamp-level
DeltaSigma/(2w) = -6.4 at sigma 0.8 px, -3.1 at 1.1 px; node grid -5.6), so it is not a
calibrated blur. ``merge_slot_terms`` is the only
hook; it is called once, in ``loss.chroma_slot_terms``. Without the ``bright_width`` leaf it
returns its input object untouched, so a run without the term traces the identical program.

Leaf. ``params["bright_width"]`` has shape (1,) and is stored in units of
``LEAF_UNIT`` = 1e-3 px^2 (DeltaSigma) per 1e4 e- per read, so the bfwidth value is a leaf of
~0.8 (an O(1) number for Adam). The CLI (``scene_fit --bright-width-init``) takes the
coefficient in px^2 per 1e4 e- (e.g. 0.8e-3) and divides by LEAF_UNIT.

Generator -> DeltaSigma conversion (``GEN_PER_DSIGMA`` = 0.5). ``EM.blur_generator`` is the
Laplacian of the node field in physical px^-2 (central differences of central differences,
i.e. a second difference with spacing 2h, h = 1/OVERSAMPLE px). For any field,
sum x^2 * D2[P] = 2 sum P exactly for such a stencil (it is exact on quadratics; summation by
parts), so P + w * Lap(P) has second moments Sigma + 2 w per axis: DeltaSigma = 2 w, i.e.
w = 0.5 * DeltaSigma. The raw-P gauge (``_gauge_off_raw_base``) removes a multiple of P,
which changes the normalised second moment only at second order in w
(Sigma' = Sigma + 2w / (1 - w alpha)). Measured (``tests/test_bright_width.py``): node grid,
Gaussian sigma 1.1 px, ungauged 1.0000, raw-gauged 0.9968 of the predicted 2w; end to end
through ``loss.forward_model`` on a centred Gaussian ePSF (13-px stamp, DeltaSigma = 8e-3 px^2)
within 1% (a star 3 px off the stamp centre reads a few % low in x only: truncation of the
term's r^2-weighted wings by the stamp, a measurement artefact, not the conversion).

Per-star q is STATIC (computed once at setup from the warm-start fluxes and the warm-start
rendered template peak; see ``scene_fit``) and stored star-indexed on the context
(``ctx.bright_q``, like ``x_lin``) together with ``ctx.bright_q_ref``. It applies to every
rendered star (contributors, anchors, nuisance): it is detector physics.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

BRIGHT_WIDTH_LEAVES = ("bright_width",)
FORMS = ("none", "lin")
LEAF_UNIT = 1e-3          # leaf 1.0 == 1e-3 px^2 DeltaSigma per Q_UNIT e- per read
Q_UNIT = 1e4              # e- per 2-s read
GEN_PER_DSIGMA = 0.5      # blur-generator weight per px^2 of DeltaSigma (per axis)
READ_TIME_S = 2.0         # TESS 2-s exposures co-added into every FFI
Q_MAX_DEFAULT = 2.0e5     # cap on q (e- per read): ~TESS pixel full well; saturated stars bleed
TESS_ZP_E_PER_S_T10 = 15000.0   # e-/s at Tmag = 10 (TESS Instrument Handbook), catalogue fallback only


def has_bright_width(params) -> bool:
    return "bright_width" in params


def blur_field_name(ctx) -> str:
    """The generator the brightness blur rides on: always ``blur_raw`` (see module docstring)."""
    return generator_for_gauge(getattr(ctx, "chroma_g8_gauge", "mean"))


def generator_for_gauge(gauge: str) -> str:
    """``blur_raw`` for either colour gauge. Under A3 (raw) this IS the colour model's blur
    field and the weight is summed into it; under the legacy mean gauge the colour model's
    ``blur`` is NOT a calibrated blur (its flat sheet flips the sign of the stamp-level
    second-moment change, measured -3..-7x), so the brightness term keeps its own
    ``blur_raw`` field beside it."""
    if gauge not in ("mean", "raw"):
        raise ValueError(f"unknown chroma_g8_gauge {gauge!r}")
    return "blur_raw"


def slot_weight(params, ctx, star_occ):
    """Per-slot blur-generator weight ``0.5 * b * (q - q_ref) / 1e4`` (b in px^2 per 1e4 e-)."""
    q = getattr(ctx, "bright_q", None)
    if q is None:
        raise ValueError(
            "params carry the bright_width leaf but the context has no bright_q; build the "
            "context with per-star peak charges (scene_fit does this). Refusing to default it "
            "to zero, which would silently disagree with the fitted model.")
    b = params["bright_width"][0] * LEAF_UNIT
    return (GEN_PER_DSIGMA * b / Q_UNIT) * (q[star_occ] - ctx.bright_q_ref)


def merge_slot_terms(params, ctx, star_occ, chroma):
    """Fold the brightness blur into the colour model's per-slot terms.

    ``chroma`` is what the colour model returned (None or ``(delta, sx, sy, fields)``). Without
    the leaf it is returned unchanged (same object). Otherwise the per-slot weight is added to
    the gauge's blur field (created if absent; with no colour model at all the colour delta
    and shifts are zeros, which every consumer treats as "no colour term").
    """
    if not has_bright_width(params):
        return chroma
    w = slot_weight(params, ctx, star_occ)
    name = blur_field_name(ctx)
    if chroma is None:
        z = jnp.zeros_like(w)
        return z, z, z, {name: w}
    delta, sx, sy, fields = chroma
    fields = dict(fields)
    if name in fields:
        if fields[name].ndim != 1:
            raise ValueError(f"colour field {name!r} is not a per-slot weight; cannot merge bright_width")
        fields[name] = fields[name] + w
    else:
        fields[name] = w
    return delta, sx, sy, fields


# ---------------------------------------------------------------- setup helpers (numpy)

def peak_charge(flux_e_per_s, peak_fraction, read_time_s: float = READ_TIME_S) -> np.ndarray:
    """q = read_time * max(f, 0) * peak fraction (e- per read); non-finite -> NaN."""
    f = np.asarray(flux_e_per_s, dtype=np.float64)
    pk = np.asarray(peak_fraction, dtype=np.float64)
    return read_time_s * np.clip(f, 0.0, None) * pk


def catalogue_flux(tess_mag) -> np.ndarray:
    return TESS_ZP_E_PER_S_T10 * 10.0 ** (-0.4 * (np.asarray(tess_mag, dtype=np.float64) - 10.0))


def weighted_q_ref(q, weight, mask) -> float:
    """Loss-weighted mean q over ``mask`` (the trained stars)."""
    q = np.asarray(q, dtype=np.float64)
    w = np.where(np.asarray(mask, bool) & np.isfinite(q), np.asarray(weight, dtype=np.float64), 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        return float(np.nanmean(q[np.asarray(mask, bool)])) if np.any(mask) else 0.0
    return float(np.sum(w * np.nan_to_num(q)) / np.sum(w))


def set_bright_width(params: dict, form: str, init_leaf: float | None) -> tuple[dict, str]:
    """Add/drop the leaf for ``form``; returns (params, provenance of the initial value).

    ``init_leaf`` (leaf units) wins; else an incoming leaf is kept; else zeros.
    """
    if form not in FORMS:
        raise ValueError(f"unknown bright-width form {form!r}")
    p = {k: v for k, v in params.items() if k not in BRIGHT_WIDTH_LEAVES}
    if form == "none":
        return p, ("dropped warm-start leaf" if has_bright_width(params) else "absent")
    if init_leaf is not None:
        p["bright_width"] = jnp.asarray([float(init_leaf)], jnp.float32)
        return p, "cli"
    if has_bright_width(params):
        p["bright_width"] = jnp.asarray(params["bright_width"], jnp.float32).reshape(1)
        return p, "warm_start"
    p["bright_width"] = jnp.zeros((1,), jnp.float32)
    return p, "zero"


def leaf_to_b(leaf) -> float:
    """Leaf value -> b in px^2 DeltaSigma per 1e4 e- per read."""
    return float(np.asarray(leaf).reshape(-1)[0]) * LEAF_UNIT
