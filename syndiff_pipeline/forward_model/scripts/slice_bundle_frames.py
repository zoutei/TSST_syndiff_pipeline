# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
#!/usr/bin/env python3
"""Subset a packed fit_bundle.npz along its frame axis (stride or head).

Cheap alternative to re-running ``export_fit_bundle`` (~85 min on a
1864-frame orbit, dominated by NFS reads of centroids_r1 + hp_d). Slicing
the finished bundle takes seconds and is exact: every frame-dependent array
is sliced on its frame axis, and the temporal B-spline design matrices
(``wcs_frame_basis``/``w_frame_basis``) are already *evaluated per frame*,
so taking their rows is identical to having evaluated the same basis at the
kept frames.

Two uses:
  * ``--stride N``  keep every Nth frame -- preserves the full orbit time
    span (what the temporal spline cares about) at 1/N the frames.
  * ``--head N``    keep the first N frames -- truncates the time span; use
    only for memory/scaling experiments, not for science.

Usage:
    python -m syndiff_pipeline.forward_model.scripts.slice_bundle_frames \\
        --in  dev/forward_epsf_wcs/output/bundles/s50_orbit1/fit_bundle.npz \\
        --out dev/forward_epsf_wcs/output/bundles/s50_orbit1_s3/fit_bundle.npz \\
        --stride 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Frame axis per array-name pattern. Anything not listed is copied verbatim.
FRAME_AXIS_EXACT = {
    "mask_active": 1,
    "w_frame_basis": 0,
    "wcs_frame_basis": 0,
}
FRAME_AXIS_TIER_SUFFIX = {  # pt{i}_<suffix>
    "data": 1,
    "noise": 1,
    "weight_u8": 1,
}


def frame_axis_for(key: str, shape: tuple[int, ...], n_frames: int) -> int | None:
    """Frame axis of *key*, or None if it carries no frame axis."""
    if key in FRAME_AXIS_EXACT:
        return FRAME_AXIS_EXACT[key]
    if key.startswith("pt") and "_" in key:
        suffix = key.split("_", 1)[1]
        if suffix in FRAME_AXIS_TIER_SUFFIX:
            return FRAME_AXIS_TIER_SUFFIX[suffix]
    return None


def infer_n_frames(z) -> int:
    """Frame count from wcs_frame_basis, the one unambiguous (T, n_basis) array."""
    return int(np.asarray(z["wcs_frame_basis"]).shape[0])


def slice_bundle(in_path: Path, out_path: Path, *, stride: int | None, head: int | None) -> None:
    z = np.load(in_path, allow_pickle=True)
    n_frames = infer_n_frames(z)

    if stride is not None:
        idx = np.arange(0, n_frames, int(stride))
    elif head is not None:
        idx = np.arange(0, min(int(head), n_frames))
    else:
        raise ValueError("pass --stride or --head")
    print(f"input frames: {n_frames} -> keeping {len(idx)}")

    out: dict[str, np.ndarray] = {}
    n_sliced = 0
    for key in z.files:
        arr = z[key]
        axis = frame_axis_for(key, getattr(arr, "shape", ()), n_frames)
        if axis is None:
            out[key] = arr
            continue
        if arr.shape[axis] != n_frames:
            raise ValueError(
                f"{key}: expected frame axis {axis} to have length {n_frames}, "
                f"got shape {arr.shape}"
            )
        out[key] = np.take(arr, idx, axis=axis)
        n_sliced += 1
        print(f"  sliced {key:28s} {str(arr.shape):22s} -> {out[key].shape}")

    if n_sliced == 0:
        raise RuntimeError("no frame-axis arrays were sliced -- layout changed?")

    # Guard: nothing frame-shaped should survive at the original length.
    for key, arr in out.items():
        if hasattr(arr, "shape") and n_frames in arr.shape and len(idx) != n_frames:
            raise RuntimeError(f"{key} still has a {n_frames}-length axis after slicing: {arr.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    size_gb = out_path.stat().st_size / 1e9
    print(f"\nsliced {n_sliced} frame arrays; wrote {out_path} ({size_gb:.2f} GB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stride", type=int, help="keep every Nth frame (preserves time span)")
    g.add_argument("--head", type=int, help="keep first N frames (truncates span; experiments only)")
    args = p.parse_args()
    slice_bundle(args.in_path, args.out_path, stride=args.stride, head=args.head)


if __name__ == "__main__":
    main()
