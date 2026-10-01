# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Memory-bounded representative-stamp counterfactuals.

The full S52 bundle stores its packed stamp cubes as compressed, group-major
``.npy`` members inside ``fit_bundle.npz``.  Loading the largest data/noise
members together costs more than 2 GiB.  This module instead streams each
member once and retains only preselected group rows, then renders the frozen
model for those rows on CPU.

The analysis is deliberately diagnostic: no optimizer state or frozen input is
modified.  The empirical-per-FFI WCS branch is recorded as unavailable when the
only pointing parquet is the model self-refit produced by ``bundle_pointing``.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
from types import SimpleNamespace

import numpy as np
import pandas as pd
from numpy.lib import format as npformat

from .. import fit as FIT
from .. import fit_bundle as FB
from .. import loss as L
from ..groups import GroupSet
from .residual_basis_decomp import weighted_lstsq


MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class SelectedTier:
    tier: int
    global_groups: np.ndarray
    local_rows: np.ndarray
    target_slots: np.ndarray
    star_rows: np.ndarray
    data: np.ndarray
    noise: np.ndarray
    weight: np.ndarray
    pix_x: np.ndarray
    pix_y: np.ndarray
    pix_valid: np.ndarray
    members: np.ndarray
    valid: np.ndarray
    context: L.StaticContext


def _robust_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    return float(MAD_TO_SIGMA * np.nanmedian(np.abs(values - med)))


def select_representative_stars(
    candidates: pd.DataFrame,
    *,
    max_stars: int = 10,
) -> pd.DataFrame:
    """Greedy deterministic coverage of magnitude, quadrant, crowding and PC sign."""

    required = {
        "star_row", "group_index", "slot_index", "tess_mag", "mag_bin",
        "x", "y", "group_size", "pc1_loading", "active_fraction",
    }
    missing = required - set(candidates)
    if missing:
        raise KeyError(f"representative-star table missing {sorted(missing)}")
    work = candidates.copy()
    work = work[np.isfinite(work.tess_mag) & np.isfinite(work.x) & np.isfinite(work.y)]
    work = work[work.active_fraction >= 0.20]
    work = work.sort_values(
        ["active_fraction", "tess_mag", "group_index"],
        ascending=[False, True, True],
    ).drop_duplicates("group_index", keep="first")
    if work.empty:
        raise ValueError("no candidate group has >=20% active coverage")
    xm, ym = work.x.median(), work.y.median()
    work["quadrant"] = [f"Q{1 + int(x >= xm) + 2 * int(y >= ym)}" for x, y in zip(work.x, work.y)]
    work["crowding_class"] = np.where(work.group_size <= 1, "isolated", "blended")
    work["loading_sign"] = np.where(work.pc1_loading >= 0, "positive", "negative")
    work["loading_abs"] = np.abs(work.pc1_loading)

    chosen: list[int] = []

    def add_best(group: pd.DataFrame) -> None:
        if len(chosen) >= max_stars or group.empty:
            return
        group = group.loc[~group.index.isin(chosen)]
        if group.empty:
            return
        chosen.append(int(group.sort_values(["loading_abs", "active_fraction"], ascending=False).index[0]))

    for _, group in work.groupby("mag_bin", sort=True):
        add_best(group)
    for _, group in work.groupby("quadrant", sort=True):
        add_best(group)
    for _, group in work.groupby(["crowding_class", "loading_sign"], sort=True):
        add_best(group)
    add_best(work[work.loading_abs <= work.loading_abs.median()])
    while len(chosen) < min(max_stars, len(work)):
        add_best(work)
    out = work.loc[chosen].copy().reset_index(drop=True)
    out.insert(0, "selection_index", np.arange(len(out), dtype=int))
    return out


def _npy_header(stream) -> tuple[tuple[int, ...], np.dtype, int]:
    version = npformat.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = npformat.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        reader = npformat.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
    else:
        raise ValueError(f"unsupported npy version {version}")
    if fortran:
        raise ValueError("streamed packed arrays must be C-contiguous")
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError("object arrays are not supported")
    return tuple(int(v) for v in shape), dtype, int(stream.tell())


def _discard(stream, n_bytes: int, *, chunk: int = 8 * 1024 * 1024) -> None:
    remaining = int(n_bytes)
    while remaining:
        block = stream.read(min(chunk, remaining))
        if not block:
            raise EOFError(f"unexpected EOF while discarding {remaining} bytes")
        remaining -= len(block)


def _read_exact(stream, n_bytes: int) -> bytes:
    parts: list[bytes] = []
    remaining = int(n_bytes)
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise EOFError(f"unexpected EOF with {remaining} bytes remaining")
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def read_selected_npy_rows(npz_path: Path, member: str, rows: Sequence[int]) -> np.ndarray:
    """Read group-major rows from one compressed npz member without materializing it."""

    requested = np.asarray(rows, dtype=int)
    if requested.ndim != 1 or len(np.unique(requested)) != len(requested):
        raise ValueError("rows must be a one-dimensional unique index list")
    order = np.argsort(requested)
    sorted_rows = requested[order]
    with zipfile.ZipFile(npz_path) as archive, archive.open(member) as stream:
        shape, dtype, _ = _npy_header(stream)
        if len(shape) < 1 or np.any(sorted_rows < 0) or np.any(sorted_rows >= shape[0]):
            raise IndexError(f"{member}: row outside shape {shape}")
        row_shape = shape[1:]
        row_bytes = int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize
        sorted_out = np.empty((len(sorted_rows), *row_shape), dtype=dtype)
        cursor = 0
        for out_index, row in enumerate(sorted_rows):
            _discard(stream, (int(row) - cursor) * row_bytes)
            raw = _read_exact(stream, row_bytes)
            sorted_out[out_index] = np.frombuffer(raw, dtype=dtype).reshape(row_shape)
            cursor = int(row) + 1
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return sorted_out[inverse]


def _params_path(artifact_dir: Path) -> Path:
    for name in ("params_stage3.npz", "params.npz", "params_latest.npz"):
        path = Path(artifact_dir) / name
        if path.is_file():
            return path
    raise FileNotFoundError("no stage-3 parameter export")


def load_selected_tiers(
    bundle_path: Path,
    artifact_dir: Path,
    selection: pd.DataFrame,
) -> tuple[list[SelectedTier], dict[str, jnp.ndarray], np.ndarray]:
    """Stream selected packed groups and construct one compact context per tier."""

    bundle_path = Path(bundle_path)
    params = FIT.load_params_npz(_params_path(Path(artifact_dir)))
    with np.load(bundle_path, allow_pickle=False) as raw:
        if int(np.asarray(raw["n_packed_tiers"])) <= 0:
            raise ValueError("representative counterfactuals currently require tier-packed stamps")
        n_tiers = int(np.asarray(raw["n_packed_tiers"]))
        small = {
            key: np.asarray(raw[key])
            for key in (
                "stamp_center_x", "stamp_center_y", "mask_active", "ra", "dec",
                "x_lin", "y_lin", "cheb_basis", "members", "valid",
                "kept_star_mask", "stamp_snr_weight", "fit_radius_stage23",
                "wcs_frame_basis", "w_frame_basis", "is_epsf_contributor",
                "t_exp_sec", "cheb_ra0_deg", "cheb_dec0_deg", "cheb_cd_inv",
                "cheb_crpix", "cheb_center", "cheb_half_extents",
                "cheb_poly_degree", "cheb_exponents", "epsf_node_x",
                "epsf_node_y", "epsf_node_col_ccd", "epsf_node_row_ccd",
            )
        }
        cheb = FB._cheb_from_arrays(small)
        epsf_grid = FB._epsf_grid_from_arrays(small)
        tier_maps = [np.asarray(raw[f"pt{i}_group_idx"], dtype=int) for i in range(n_tiers)]
        tier_static = [
            {
                "pix_x": np.asarray(raw[f"pt{i}_pix_x"], dtype=np.float32),
                "pix_y": np.asarray(raw[f"pt{i}_pix_y"], dtype=np.float32),
                "pix_valid": np.asarray(raw[f"pt{i}_pix_valid"], dtype=np.float32),
                "k": int(np.asarray(raw[f"pt{i}_k_tier"])),
            }
            for i in range(n_tiers)
        ]
    with np.load(Path(artifact_dir) / "flux_solved.npz", allow_pickle=False) as solved:
        active_all = np.asarray(solved["stamp_active"], dtype=bool)
        btjd = np.asarray(solved["btjd"], dtype=float)

    outputs: list[SelectedTier] = []
    for tier, global_map in enumerate(tier_maps):
        reverse = {int(group): row for row, group in enumerate(global_map)}
        selected_rows = selection[selection.group_index.isin(reverse)].copy()
        if selected_rows.empty:
            continue
        local_rows = np.asarray([reverse[int(g)] for g in selected_rows.group_index], dtype=int)
        global_groups = selected_rows.group_index.to_numpy(dtype=int)
        data = read_selected_npy_rows(bundle_path, f"pt{tier}_data.npy", local_rows).astype(np.float32)
        noise = read_selected_npy_rows(bundle_path, f"pt{tier}_noise.npy", local_rows).astype(np.float32)
        weight = read_selected_npy_rows(bundle_path, f"pt{tier}_weight_u8.npy", local_rows).astype(np.float32)
        static = tier_static[tier]
        pix_x = static["pix_x"][local_rows]
        pix_y = static["pix_y"][local_rows]
        pix_valid = static["pix_valid"][local_rows]
        k = int(static["k"])
        members_global = small["members"][global_groups, :k].astype(np.int32)
        valid = small["valid"][global_groups, :k].astype(bool)
        used_stars = np.unique(members_global[valid])
        star_remap = {int(star): index for index, star in enumerate(used_stars)}
        members = np.zeros_like(members_global)
        for group_row, slot in np.argwhere(valid):
            members[group_row, slot] = star_remap[int(members_global[group_row, slot])]
        groups = GroupSet(
            len(global_groups), k, members, valid,
            np.ones(len(used_stars), dtype=bool), 0,
        )
        # Colour, for a run fitted with --chroma. Without it ``forward_model``
        # renders a DIFFERENT model than the one that was fitted (or refuses to,
        # depending on the guard), so the counterfactual would be measured against
        # the wrong baseline. ``bp_rp`` is per STAR, on the same axis as ra/x_lin.
        bp_rp_all = (np.asarray(raw["bp_rp"]) if "bp_rp" in raw.files else None)
        # Reuse THE gauge rather than recomputing it: ``colour_ref_from_bundle`` is
        # duck-typed on these four arrays, and its docstring is explicit that every
        # consumer must get the identical number.
        colour_ref = (
            L.colour_ref_from_bundle(SimpleNamespace(
                bp_rp=bp_rp_all,
                members=np.asarray(raw["members"]),
                valid=np.asarray(raw["valid"]),
                stamp_snr_weight=np.asarray(raw["stamp_snr_weight"]),
            ))
            if bp_rp_all is not None else None
        )

        ctx = L.build_static_context(
            bp_rp=(None if bp_rp_all is None else bp_rp_all[used_stars]),
            colour_ref=colour_ref,
            cheb_static=cheb,
            wcs_frame_basis=small["wcs_frame_basis"],
            w_frame_basis=small["w_frame_basis"],
            epsf_grid=epsf_grid,
            groups=groups,
            ra=small["ra"][used_stars], dec=small["dec"][used_stars],
            stamp_center_x=small["stamp_center_x"][global_groups],
            stamp_center_y=small["stamp_center_y"][global_groups],
            t_exp_sec=float(np.asarray(small["t_exp_sec"])),
            stamp_snr_weight=small["stamp_snr_weight"][global_groups],
            fit_radius=small["fit_radius_stage23"][global_groups],
            stamp_active=active_all[global_groups],
            x_lin=small["x_lin"][used_stars], y_lin=small["y_lin"][used_stars],
            cheb_basis=small["cheb_basis"][used_stars],
            pix_x=pix_x, pix_y=pix_y, pix_valid=pix_valid,
            is_epsf_contributor=small["is_epsf_contributor"][global_groups],
        )
        outputs.append(SelectedTier(
            tier=tier, global_groups=global_groups, local_rows=local_rows,
            target_slots=selected_rows.slot_index.to_numpy(dtype=int),
            star_rows=selected_rows.star_row.to_numpy(dtype=int),
            data=data, noise=noise, weight=weight,
            pix_x=pix_x, pix_y=pix_y, pix_valid=pix_valid,
            members=members, valid=valid, context=ctx,
        ))
    if not outputs:
        raise ValueError("none of the selected groups was found in packed tiers")
    return outputs, params, btjd


def _render(
    params: Mapping[str, jnp.ndarray],
    ctx: L.StaticContext,
    *,
    w_override=None,
    frame_block: int = 64,
):
    """Render in short frame blocks to keep CPU XLA programs and memory bounded."""

    n_frames = int(ctx.wcs_frame_basis.shape[0])
    if w_override is None:
        w_full = np.asarray(params["w_coeff"]) @ np.asarray(ctx.w_frame_basis).T
        w_full = w_full.T
        w_full -= np.mean(w_full, axis=0, keepdims=True)
    else:
        w_full = np.asarray(w_override, dtype=np.float32)
        if w_full.shape[0] != n_frames:
            raise ValueError("w_override must contain every context frame")
    templates_out, x_out, y_out = [], [], []
    block = max(1, int(frame_block))
    for start in range(0, n_frames, block):
        stop = min(n_frames, start + block)
        block_ctx = replace(
            ctx,
            wcs_frame_basis=ctx.wcs_frame_basis[start:stop],
            w_frame_basis=ctx.w_frame_basis[start:stop],
            stamp_active=ctx.stamp_active[:, start:stop],
        )
        templates, x, y, _ = L.forward_model(
            dict(params), block_ctx,
            w_of_t_override=jnp.asarray(w_full[start:stop]),
        )
        templates_out.append(np.asarray(templates))
        x_out.append(np.asarray(x))
        y_out.append(np.asarray(y))
    return (
        np.concatenate(templates_out, axis=2),
        np.concatenate(x_out, axis=2),
        np.concatenate(y_out, axis=2),
        w_full,
    )


def _inverse_variance(tier: SelectedTier) -> np.ndarray:
    return tier.weight * tier.pix_valid[:, None, :] / (tier.noise**2 + L.VARIANCE_FLOOR)


def solve_profiles(
    templates: np.ndarray,
    data: np.ndarray,
    ivar: np.ndarray,
    coverage: np.ndarray,
    *,
    background: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched weighted profile solve, optionally with one constant offset."""

    design = np.moveaxis(np.asarray(templates, dtype=float), 1, -1)  # G,T,P,K
    k = design.shape[-1]
    if background:
        design = np.concatenate([design, np.ones((*design.shape[:-1], 1))], axis=-1)
    normal = np.einsum("gtpi,gtp,gtpj->gtij", design, ivar, design, optimize=True)
    rhs = np.einsum("gtpi,gtp,gtp->gti", design, ivar, data, optimize=True)
    diag = np.arange(normal.shape[-1])
    normal[..., diag, diag] += 1e-6
    coef = np.linalg.solve(normal, rhs[..., None])[..., 0]
    model = np.einsum("gtpi,gti->gtp", design, coef, optimize=True)
    resid = data - model
    chi2 = np.sum(ivar * resid**2, axis=-1) / np.maximum(np.sum(coverage, axis=-1), 1.0)
    bg = coef[..., -1] if background else np.zeros(coef.shape[:2], dtype=float)
    return coef[..., :k], bg, model, chi2


def fixed_aperture_flux(
    data: np.ndarray,
    pix_x: np.ndarray,
    pix_y: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    coverage: np.ndarray,
    *,
    radius: float,
    annulus: tuple[float, float] = (3.0, 4.5),
) -> np.ndarray:
    """Simple packed-support circular aperture with a per-frame annulus median."""

    dx = pix_x[:, None, :] - x_target[:, :, None]
    dy = pix_y[:, None, :] - y_target[:, :, None]
    rr = np.sqrt(dx**2 + dy**2)
    valid = coverage > 0
    ann = valid & (rr >= annulus[0]) & (rr < annulus[1])
    bg = np.nanmedian(np.where(ann, data, np.nan), axis=-1)
    bg = np.where(np.isfinite(bg), bg, 0.0)
    aperture = valid & (rr <= float(radius))
    return np.sum(np.where(aperture, data - bg[..., None], 0.0), axis=-1)


def _normalise_curves(curves: np.ndarray, active: np.ndarray) -> np.ndarray:
    masked = np.where(active & np.isfinite(curves), curves, np.nan)
    med = np.nanmedian(masked, axis=1)
    return np.where(active, curves / med[:, None] - 1.0, np.nan)


def _method_metrics(
    name: str,
    curves: np.ndarray,
    active: np.ndarray,
    btjd: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    chi2: np.ndarray | None,
    *,
    status: str = "performed",
) -> dict:
    if status != "performed":
        return {"method": name, "status": status}
    frac = _normalise_curves(curves, active)
    ensemble = np.nanmedian(frac, axis=0)
    star_rms = np.asarray([_robust_rms(row) for row in frac])
    gap = np.r_[False, np.diff(btjd) < 1.5 * np.nanmedian(np.diff(btjd))]
    hf = []
    for row, ok in zip(frac, active):
        pair = ok & np.r_[False, ok[:-1]] & gap & np.isfinite(row) & np.r_[False, np.isfinite(row[:-1])]
        diff = row[pair] - row[np.flatnonzero(pair) - 1]
        hf.append(_robust_rms(diff) / np.sqrt(2.0) if len(diff) else np.nan)
    load = np.full(len(frac), np.nan)
    denom = np.nansum(ensemble**2)
    if denom > 0:
        for i, row in enumerate(frac):
            ok = np.isfinite(row) & np.isfinite(ensemble)
            if ok.sum() > 10:
                load[i] = np.sum(row[ok] * ensemble[ok]) / np.sum(ensemble[ok] ** 2)
    spatial_r2 = np.nan
    ok = np.isfinite(load) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 8:
        xx = (x[ok] - np.mean(x[ok])) / max(np.std(x[ok]), 1e-9)
        yy = (y[ok] - np.mean(y[ok])) / max(np.std(y[ok]), 1e-9)
        design = np.column_stack([np.ones(ok.sum()), xx, yy, xx * yy, xx**2, yy**2])
        pred = design @ np.linalg.lstsq(design, load[ok], rcond=None)[0]
        spatial_r2 = 1 - np.sum((load[ok] - pred) ** 2) / max(np.sum((load[ok] - np.mean(load[ok])) ** 2), 1e-20)
    return {
        "method": name, "status": status, "n_stars": int(len(curves)),
        "common_mode_robust_rms": _robust_rms(ensemble),
        "common_mode_range_5_95": float(np.nanpercentile(ensemble, 95) - np.nanpercentile(ensemble, 5)),
        "median_star_robust_rms": float(np.nanmedian(star_rms)),
        "median_high_frequency_scatter": float(np.nanmedian(hf)),
        "median_chi2_red": float(np.nanmedian(np.where(active, chi2, np.nan))) if chi2 is not None else np.nan,
        "spatial_loading_quadratic_r2": spatial_r2,
    }


def _pixel_basis_rows(
    tier: SelectedTier,
    templates: np.ndarray,
    model: np.ndarray,
    flux: np.ndarray,
    shifted: Mapping[str, np.ndarray],
    btjd: np.ndarray,
    strict: np.ndarray,
) -> list[dict]:
    ivar = _inverse_variance(tier)
    resid = tier.data - model
    bases = {
        "flux": model,
        "x_shift": np.einsum("gktp,gtk->gtp", shifted["dx"], flux, optimize=True),
        "y_shift": np.einsum("gktp,gtk->gtp", shifted["dy"], flux, optimize=True),
        "defocus": np.einsum("gktp,gtk->gtp", shifted["dw"], flux, optimize=True),
        "background": np.ones_like(model),
    }
    rows: list[dict] = []
    for gi, group in enumerate(tier.global_groups):
        for fi in range(len(btjd)):
            basis = {name: values[gi, fi] for name, values in bases.items()}
            fit = weighted_lstsq(resid[gi, fi], basis, ivar[gi, fi])
            for name in bases:
                rows.append({
                    "group_index": int(group), "frame_index": fi, "btjd": float(btjd[fi]),
                    "basis": name, "coefficient": fit.get(name, np.nan),
                    "unique_variance_fraction": fit.get(f"frac_{name}", np.nan),
                    "joint_variance_fraction": fit.get("frac_explained", np.nan),
                    "residual_rms_before": fit.get("resid_rms_before", np.nan),
                    "residual_rms_after": fit.get("resid_rms_after", np.nan),
                    "strict_cadence": bool(strict[fi]),
                })
    return rows


def _signal(btjd: np.ndarray, kind: str, amplitude: float, period: float | None) -> np.ndarray:
    if kind == "constant":
        return np.zeros_like(btjd)
    assert period is not None
    phase = np.mod(btjd - btjd[0], period) / period
    if kind == "sinusoid":
        return amplitude * np.sin(2 * np.pi * phase)
    if kind == "transit":
        in_transit = np.abs(((phase + 0.5) % 1.0) - 0.5) < 0.05
        return -amplitude * in_transit.astype(float)
    raise ValueError(kind)


def _recovery(signal: np.ndarray, recovered: np.ndarray, btjd: np.ndarray, kind: str, amplitude: float, period: float | None) -> dict:
    med = np.nanmedian(recovered)
    frac = recovered / med - 1.0 if np.isfinite(med) and med != 0 else np.full_like(recovered, np.nan)
    ok = np.isfinite(frac)
    if kind == "constant":
        false_rms = _robust_rms(frac[ok])
        return {
            "recovered_amplitude": false_rms, "amplitude_bias_fraction": np.nan,
            "timing_error_days": np.nan, "false_detection_rms": false_rms,
            "passes_amplitude": True, "passes_timing": True,
            "passes_false_detection": bool(false_rms < 5e-4),
        }
    assert period is not None
    if kind == "sinusoid":
        omega = 2 * np.pi / period
        design = np.column_stack([np.ones(ok.sum()), np.sin(omega * (btjd[ok] - btjd[0])), np.cos(omega * (btjd[ok] - btjd[0]))])
        coef = np.linalg.lstsq(design, frac[ok], rcond=None)[0]
        recovered_amp = float(np.hypot(coef[1], coef[2]))
        timing = float(abs(np.arctan2(coef[2], coef[1]) / omega))
    else:
        expected = signal / amplitude
        best_shift, best_sse, best_depth = 0, np.inf, np.nan
        for shift in range(-3, 4):
            template = np.roll(expected, shift)
            design = np.column_stack([np.ones(ok.sum()), template[ok]])
            coef = np.linalg.lstsq(design, frac[ok], rcond=None)[0]
            sse = np.sum((frac[ok] - design @ coef) ** 2)
            if sse < best_sse:
                best_shift, best_sse, best_depth = shift, sse, float(coef[1])
        recovered_amp = abs(best_depth)
        timing = abs(best_shift) * float(np.nanmedian(np.diff(btjd)))
    bias = recovered_amp / amplitude - 1.0
    cadence = float(np.nanmedian(np.diff(btjd)))
    return {
        "recovered_amplitude": recovered_amp, "amplitude_bias_fraction": bias,
        "timing_error_days": timing, "false_detection_rms": np.nan,
        "passes_amplitude": bool(abs(bias) < 0.05),
        "passes_timing": bool(timing <= cadence + 1e-12),
        "passes_false_detection": True,
    }


def run_stamp_counterfactuals(
    *,
    bundle_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    candidates: pd.DataFrame,
    strict_cadence_mask: np.ndarray,
    aperture_by_star_row: np.ndarray | None = None,
    max_stars: int = 10,
    allow_cpu_renderer: bool = False,
) -> dict:
    """Run the feasible frozen-stamp causal tests and write tables/figures."""

    import jax

    if jax.default_backend() == "cpu" and not allow_cpu_renderer:
        raise RuntimeError(
            "packed representative-stamp rendering is disabled on CPU: the "
            "2026-08-27 S52 smoke (one tier-1 group, two frames, both banded "
            "and reference renderers) SIGSEGVed inside XLA compilation; use "
            "run_preview_basis_decomposition for the safe three-mosaic "
            "descriptive fallback"
        )
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    selection = select_representative_stars(candidates, max_stars=max_stars)
    selection.to_csv(output_dir / "representative_stamps.csv", index=False)
    tiers, params, btjd = load_selected_tiers(bundle_path, artifact_dir, selection)
    strict = np.asarray(strict_cadence_mask, dtype=bool)
    if strict.shape != btjd.shape:
        raise ValueError("strict cadence mask and bundle BTJD length disagree")

    basis_rows: list[dict] = []
    counter_curves: dict[str, list[np.ndarray]] = {}
    counter_active: list[np.ndarray] = []
    counter_x: list[float] = []
    counter_y: list[float] = []
    counter_chi2: dict[str, list[np.ndarray]] = {}
    injection_rows: list[dict] = []

    for tier in tiers:
        templates, x_slot, y_slot, w_t = _render(params, tier.context)
        mean_params = dict(params)
        mean_params["w_coeff"] = jnp.zeros_like(mean_params["w_coeff"])
        mean_templates, _, _, _ = _render(mean_params, tier.context)
        ivar = _inverse_variance(tier)
        coverage = tier.weight * tier.pix_valid[:, None, :]
        flux, _, model, chi2 = solve_profiles(templates, tier.data, ivar, coverage)
        mean_flux, _, _, mean_chi2 = solve_profiles(mean_templates, tier.data, ivar, coverage)
        bg_flux, _, _, bg_chi2 = solve_profiles(templates, tier.data, ivar, coverage, background=True)

        eps = 0.02
        txp, *_ = _render(params, replace(tier.context, x_lin=tier.context.x_lin + eps))
        txm, *_ = _render(params, replace(tier.context, x_lin=tier.context.x_lin - eps))
        typ, *_ = _render(params, replace(tier.context, y_lin=tier.context.y_lin + eps))
        tym, *_ = _render(params, replace(tier.context, y_lin=tier.context.y_lin - eps))
        w_delta = max(float(np.nanstd(w_t)) * 0.1, 0.01)
        twp, *_ = _render(params, tier.context, w_override=jnp.asarray(w_t + w_delta))
        twm, *_ = _render(params, tier.context, w_override=jnp.asarray(w_t - w_delta))
        basis_rows.extend(_pixel_basis_rows(
            tier, templates, model, flux,
            {"dx": (txp - txm) / (2 * eps), "dy": (typ - tym) / (2 * eps), "dw": (twp - twm) / (2 * w_delta)},
            btjd, strict,
        ))

        target = np.arange(len(tier.global_groups))
        slots = tier.target_slots
        active = np.asarray(tier.context.stamp_active, dtype=bool) & strict[None, :]
        target_x = x_slot[target, slots]
        target_y = y_slot[target, slots]
        methods = {
            "current_psf": flux[target, :, slots],
            "mean_epsf_psf": mean_flux[target, :, slots],
            "background_offset_psf": bg_flux[target, :, slots],
        }
        for radius in (1.5, 2.0, 2.5):
            methods[f"fixed_aperture_r{radius:.1f}"] = fixed_aperture_flux(
                tier.data, tier.pix_x, tier.pix_y, target_x, target_y, coverage, radius=radius,
            )
        if aperture_by_star_row is not None:
            methods["current_psf_aperture_corrected"] = methods["current_psf"] / aperture_by_star_row[tier.star_rows]
        for name, curves in methods.items():
            counter_curves.setdefault(name, []).extend(list(curves))
        counter_chi2.setdefault("current_psf", []).extend(list(chi2))
        counter_chi2.setdefault("mean_epsf_psf", []).extend(list(mean_chi2))
        counter_chi2.setdefault("background_offset_psf", []).extend(list(bg_chi2))
        counter_active.extend(list(active))
        counter_x.extend([float(np.nanmedian(v)) for v in target_x])
        counter_y.extend([float(np.nanmedian(v)) for v in target_y])

        isolated = np.flatnonzero(np.sum(tier.valid, axis=1) == 1)[:4]
        span = float(btjd[-1] - btjd[0])
        periods = {"slow": max(0.8 * span, 1.0), "separated": max(0.5, 10 * np.nanmedian(np.diff(btjd)))}
        configs = [("constant", 0.0, "none", None)]
        for kind in ("sinusoid", "transit"):
            for amp in (0.001, 0.005, 0.01):
                configs.extend((kind, amp, label, period) for label, period in periods.items())
        for gi in isolated:
            slot = int(slots[gi])
            f0 = float(np.nanmedian(flux[gi, active[gi], slot]))
            if not np.isfinite(f0) or f0 <= 0:
                continue
            method_templates = {
                "current_psf": templates[gi:gi + 1],
                "mean_epsf_psf": mean_templates[gi:gi + 1],
            }
            for kind, amp, period_label, period in configs:
                sig = _signal(btjd, kind, amp, period)
                injection = templates[gi, slot] * (f0 * (1.0 + sig))[:, None]
                injected = tier.data[gi:gi + 1] + injection[None, ...]
                recovered: dict[str, np.ndarray] = {}
                for name, tmpl in method_templates.items():
                    inc, *_ = solve_profiles(tmpl, injection[None, ...], ivar[gi:gi + 1], coverage[gi:gi + 1])
                    recovered[name] = inc[0, :, slot]
                inc_bg, *_ = solve_profiles(
                    templates[gi:gi + 1], injection[None, ...], ivar[gi:gi + 1], coverage[gi:gi + 1], background=True,
                )
                recovered["background_offset_psf"] = inc_bg[0, :, slot]
                base_ap = {}
                inj_ap = {}
                for radius in (1.5, 2.0, 2.5):
                    name = f"fixed_aperture_r{radius:.1f}"
                    base_ap[name] = fixed_aperture_flux(
                        tier.data[gi:gi + 1], tier.pix_x[gi:gi + 1], tier.pix_y[gi:gi + 1],
                        target_x[gi:gi + 1], target_y[gi:gi + 1], coverage[gi:gi + 1], radius=radius,
                    )[0]
                    inj_ap[name] = fixed_aperture_flux(
                        injected, tier.pix_x[gi:gi + 1], tier.pix_y[gi:gi + 1],
                        target_x[gi:gi + 1], target_y[gi:gi + 1], coverage[gi:gi + 1], radius=radius,
                    )[0]
                    recovered[name] = inj_ap[name] - base_ap[name]
                if aperture_by_star_row is not None:
                    recovered["current_psf_aperture_corrected"] = recovered["current_psf"] / aperture_by_star_row[tier.star_rows[gi]]
                for method, curve in recovered.items():
                    facts = _recovery(sig[active[gi]], curve[active[gi]], btjd[active[gi]], kind, amp, period)
                    injection_rows.append({
                        "group_index": int(tier.global_groups[gi]), "star_row": int(tier.star_rows[gi]),
                        "method": method, "signal_kind": kind, "injected_amplitude": amp,
                        "period_class": period_label, "period_days": period,
                        **facts,
                    })

    basis = pd.DataFrame(basis_rows)
    basis.to_csv(output_dir / "pixel_basis_decomposition.csv", index=False)
    active_arr = np.asarray(counter_active, dtype=bool)
    x_arr, y_arr = np.asarray(counter_x), np.asarray(counter_y)
    counter_rows = []
    for method, rows in counter_curves.items():
        curves = np.asarray(rows, dtype=float)
        chi = np.asarray(counter_chi2[method], dtype=float) if method in counter_chi2 else None
        counter_rows.append(_method_metrics(method, curves, active_arr, btjd, x_arr, y_arr, chi))
    counter_rows.append(_method_metrics("empirical_per_ffi_wcs", np.empty((0, len(btjd))), np.empty((0, len(btjd)), bool), btjd, np.array([]), np.array([]), None, status="unavailable_model_self_refit_only"))
    counter = pd.DataFrame(counter_rows)
    baseline = float(counter.loc[counter.method == "current_psf", "common_mode_robust_rms"].iloc[0])
    counter["common_mode_improvement_fraction"] = 1.0 - counter.common_mode_robust_rms / baseline
    baseline_hf = float(counter.loc[counter.method == "current_psf", "median_high_frequency_scatter"].iloc[0])
    counter["high_frequency_scatter_change_fraction"] = counter.median_high_frequency_scatter / baseline_hf - 1.0
    counter.to_csv(output_dir / "counterfactual_comparisons.csv", index=False)
    injections = pd.DataFrame(injection_rows)
    injections.to_csv(output_dir / "injection_recovery.csv", index=False)

    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    clean_basis = basis[basis.strict_cadence]
    for ax, name in zip(axes, ("flux", "x_shift", "y_shift", "defocus", "background")):
        sub = clean_basis[clean_basis.basis == name]
        med = sub.groupby("frame_index").coefficient.median()
        ax.plot(btjd[med.index.to_numpy(dtype=int)], med, lw=0.7)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("BTJD")
    fig.savefig(fig_dir / "pixel_basis_decomposition.png", dpi=150)
    plt.close(fig)

    performed = counter[counter.status == "performed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(performed.method, performed.common_mode_robust_rms)
    axes[1].bar(performed.method, performed.median_high_frequency_scatter)
    for ax in axes:
        ax.tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("common-mode robust RMS")
    axes[1].set_ylabel("median high-frequency scatter")
    fig.savefig(fig_dir / "counterfactual_comparisons.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    if len(injections):
        show = injections[injections.signal_kind != "constant"].copy()
        show["abs_bias"] = np.abs(show.amplitude_bias_fraction)
        med = show.groupby(["method", "signal_kind"]).abs_bias.median().unstack()
        med.plot.bar(ax=ax)
        ax.axhline(0.05, color="r", ls="--", lw=1)
    ax.set_ylabel("median |amplitude/depth bias|")
    fig.savefig(fig_dir / "injection_recovery.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_selected_stars": int(len(selection)),
        "n_selected_tiers": int(len(tiers)),
        "pixel_basis_rows": int(len(basis)),
        "counterfactual_methods_performed": performed.method.tolist(),
        "empirical_wcs_status": "unavailable_model_self_refit_only",
        "injection_rows": int(len(injections)),
        "all_injection_amplitude_pass": bool(injections.passes_amplitude.all()) if len(injections) else False,
        "all_injection_timing_pass": bool(injections.passes_timing.all()) if len(injections) else False,
        "all_constant_false_detection_pass": bool(injections[injections.signal_kind == "constant"].passes_false_detection.all()) if len(injections) else False,
    }
    (output_dir / "stamp_counterfactual_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_preview_basis_decomposition(artifact_dir: Path, output_dir: Path) -> dict:
    """Decompose the three saved model/residual mosaics without any rendering.

    This is a descriptive whole-mosaic projection only.  The FITS mosaics are
    mean-pasted stamp footprints, not independent representative stamps, and
    only three cadences were exported.  Consequently these rows must never be
    used as a causal counterfactual or an injection-recovery gate.
    """

    from astropy.io import fits
    from .residual_basis_decomp import build_basis

    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    root = artifact_dir / "plots" / "fits_export"
    residual_paths = sorted(root.glob("residual_*.fits"))
    rows: list[dict] = []
    for residual_path in residual_paths:
        stem = residual_path.stem.removeprefix("residual_")
        model_path = root / f"model_{stem}.fits"
        if not model_path.is_file():
            continue
        with fits.open(residual_path, memmap=True) as hdul:
            residual = np.asarray(hdul[0].data, dtype=float)
            header = hdul[0].header.copy()
        with fits.open(model_path, memmap=True) as hdul:
            model = np.asarray(hdul[0].data, dtype=float)
        finite = np.isfinite(residual) & np.isfinite(model)
        weight = finite.astype(float)
        basis = build_basis(np.where(finite, model, 0.0), 0.0, 0.0)
        basis["background"] = np.ones_like(model)
        fit = weighted_lstsq(np.where(finite, residual, 0.0), basis, weight)
        rows.append({
            "stem": stem,
            "frame_index": int(header.get("FRAMEIDX", -1)),
            "btjd": float(header.get("BTJD", np.nan)),
            "n_finite_pixels": int(finite.sum()),
            "scope": "three_preview_mosaics_not_representative_stamp_causal",
            **fit,
        })
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "pixel_basis_preview.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    if len(table):
        labels = [f"frame {v}" for v in table.frame_index]
        x = np.arange(len(table))
        width = 0.16
        for offset, name in enumerate(("frac_flux", "frac_x", "frac_y", "frac_width", "frac_background")):
            axes[0].bar(x + (offset - 2) * width, table[name], width=width, label=name.removeprefix("frac_"))
        axes[0].set_xticks(x, labels)
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].plot(x, table.resid_rms_before, "o-", label="before")
        axes[1].plot(x, table.resid_rms_after, "o-", label="after")
        axes[1].set_xticks(x, labels)
        axes[1].legend()
    axes[0].set_ylabel("unique weighted variance fraction")
    axes[1].set_ylabel("weighted residual RMS")
    fig.suptitle("Three exported preview mosaics — descriptive, not causal")
    fig.savefig(fig_dir / "pixel_basis_preview.png", dpi=150)
    plt.close(fig)
    summary = {
        "status": "descriptive_only",
        "n_preview_mosaics": int(len(table)),
        "scope": "three_preview_mosaics_not_representative_stamp_causal",
        "representative_stamp_renderer": "blocked_cpu_xla_sigsegv",
    }
    (output_dir / "pixel_basis_preview_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
