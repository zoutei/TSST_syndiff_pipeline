# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Lean semantic preflight for a prepared GPU FitBundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from . import fit_bundle as FB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--expect-frames", type=int, default=None)
    p.add_argument("--expect-groups", type=int, default=None)
    p.add_argument("--expect-tiers", type=int, default=None)
    args = p.parse_args(argv)

    bundle = FB.load_fit_bundle(args.bundle)
    errors: list[str] = []
    if args.expect_frames is not None and bundle.n_frames != args.expect_frames:
        errors.append(f"frames {bundle.n_frames} != expected {args.expect_frames}")
    if args.expect_groups is not None and bundle.n_groups != args.expect_groups:
        errors.append(f"groups {bundle.n_groups} != expected {args.expect_groups}")
    tiers = bundle.packed_tiers or ()
    if args.expect_tiers is not None and len(tiers) != args.expect_tiers:
        errors.append(f"tiers {len(tiers)} != expected {args.expect_tiers}")
    if not bundle.is_packed or not tiers:
        errors.append("GPU full-CCD artifact must be a native tiered packed bundle")

    for name, value in bundle.params0.items():
        if not np.isfinite(np.asarray(value)).all():
            errors.append(f"params0[{name}] contains non-finite values")
    for name in ("x_lin", "y_lin", "cheb_basis", "wcs_frame_basis", "w_frame_basis"):
        if not np.isfinite(np.asarray(getattr(bundle, name))).all():
            errors.append(f"{name} contains non-finite values")
    mask = np.asarray(bundle.mask_active)
    if mask.shape != (bundle.n_groups, bundle.n_frames):
        errors.append(f"mask_active has wrong shape {mask.shape}")
    if not np.isin(mask, (0.0, 1.0)).all():
        errors.append("physical mask_active is not binary")

    seen: list[np.ndarray] = []
    for i, tier in enumerate(tiers):
        prefix = f"tier[{i}] K={tier.k_tier} P={tier.p_tier}"
        group_idx = np.asarray(tier.group_idx)
        seen.append(group_idx)
        valid = np.asarray(tier.pix_valid, dtype=bool)
        weight = np.asarray(tier.weight_u8)
        if not np.isfinite(np.asarray(tier.data)).all():
            errors.append(f"{prefix}: data contains non-finite values")
        noise = np.asarray(tier.noise)
        if not np.isfinite(noise).all():
            errors.append(f"{prefix}: noise contains non-finite values")
        active = (weight > 0) & valid[:, None, :]
        if active.any() and not (noise[active] > 0).all():
            errors.append(f"{prefix}: active pixels contain non-positive noise")
        if np.any(weight & (~valid[:, None, :])):
            errors.append(f"{prefix}: invalid support pixels have nonzero weight")
        member_count = np.asarray(bundle.valid)[group_idx].sum(axis=1)
        if np.any(member_count < 1) or np.any(member_count > tier.k_tier):
            errors.append(f"{prefix}: member occupancy is outside [1,K]")
        pixel_count = valid.sum(axis=1)
        if np.any(pixel_count < 1) or np.any(pixel_count > tier.p_tier):
            errors.append(f"{prefix}: pixel occupancy is outside [1,P]")

    if seen:
        indices = np.concatenate(seen)
        if not np.array_equal(np.sort(indices), np.arange(bundle.n_groups)):
            errors.append("tier group indices do not cover every global group exactly once")

    if not bundle.meta.get("gaia_pm_propagation", {}).get("applied", False):
        errors.append("bundle does not attest that Gaia proper motion was applied")
    if errors:
        raise SystemExit("GPU PREFLIGHT FAILED:\n- " + "\n- ".join(errors))

    tier_bytes = sum(
        t.data.nbytes + t.noise.nbytes + t.weight_u8.nbytes
        + t.pix_x.nbytes + t.pix_y.nbytes + t.pix_valid.nbytes
        for t in tiers
    )
    print("GPU PREFLIGHT PASSED")
    print(f"sha256={_sha256(args.bundle)}")
    print(
        f"G={bundle.n_groups} T={bundle.n_frames} tiers={len(tiers)} "
        f"host_tier_arrays={tier_bytes / 1e9:.3f} GB physical_masked={int((mask == 0).sum())}"
    )


if __name__ == "__main__":
    main()
