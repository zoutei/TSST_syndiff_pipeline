"""Convert legacy dense (v1) L4b rim caches to the sparse (v2) layout.

The v1 layout stored two PS1-shaped int32 arrays per pair-state
(``exact_tid_lo``/``exact_tid_hi``, ~315 MB decompressed) of which only ~0.3-1.3%
is valid and only one side is ever read. Downsample paid that decompression once
per (composite key, neighbour), which dominated the stage. v2 stores flat
``(idx, val)`` pairs instead -- same values, ~45x cheaper to read.

Readers accept both layouts, so this migration is optional and can run while
other work proceeds; it is idempotent and safe to re-run or interrupt.

Usage:
    python -m syndiff_pipeline.template_creation.processing.migrate_l4b_rim_store \\
        --data-root /path/to/data --sector 20 --camera 3 --ccd 3 \\
        --oversampling 1 [--jobs 32] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from syndiff_pipeline.template_creation.processing.field_abutting import (
    L4B_RIM_FORMAT_VERSION,
    l4b_rim_is_sparse,
    sparsify_l4b_rim_payload,
    write_l4b_rim_cache,
)

log = logging.getLogger(__name__)

__all__ = [
    "convert_l4b_rim_file",
    "iter_l4b_rim_files",
    "migrate_l4b_rim_store",
]


def iter_l4b_rim_files(l4b_root: Path | str):
    """Yield every ``*_rim.npz`` under an ``exact_cache_l4b/`` tree."""
    return sorted(Path(l4b_root).rglob("*_rim.npz"))


def _scalar(z, key: str, default: int = 0) -> int:
    try:
        return int(z[key])
    except Exception:
        return int(default)


def convert_l4b_rim_file(path: Path | str, *, verify: bool = True) -> str:
    """Convert one rim NPZ in place (via temp + rename). Returns a status string.

    ``"skip"`` when already sparse, ``"convert"`` on success. Verification
    reconstructs the dense arrays from the sparse payload and requires exact
    equality before the original is replaced.
    """
    p = Path(path)
    with np.load(p) as z:
        if l4b_rim_is_sparse(z):
            return "skip"
        dense_lo = np.asarray(z["exact_tid_lo"], dtype=np.int32)
        dense_hi = np.asarray(z["exact_tid_hi"], dtype=np.int32)
        meta = {
            "id_lo": _scalar(z, "id_lo"),
            "id_hi": _scalar(z, "id_hi"),
            "sx_lo": _scalar(z, "sx_lo"),
            "sy_lo": _scalar(z, "sy_lo"),
            "sx_hi": _scalar(z, "sx_hi"),
            "sy_hi": _scalar(z, "sy_hi"),
            "pair_epoch_id": _scalar(z, "pair_epoch_id"),
            "rep_frame_index": _scalar(z, "rep_frame_index"),
        }

    if verify:
        for dense in (dense_lo, dense_hi):
            if dense.size == 0:
                continue
            idx, val = sparsify_l4b_rim_payload(dense)
            rebuilt = np.full(dense.size, -1, dtype=np.int32)
            rebuilt[idx] = val
            # Values below zero are all sentinels; only >= 0 pixels are ever read.
            if not np.array_equal(
                np.maximum(rebuilt, -1),
                np.maximum(dense.ravel(), -1),
            ):
                raise RuntimeError(f"sparse round-trip mismatch for {p}")

    shape = None
    for dense in (dense_lo, dense_hi):
        if dense.ndim == 2:
            shape = (int(dense.shape[0]), int(dense.shape[1]))
            break

    write_l4b_rim_cache(p, exact_tid_lo=dense_lo, exact_tid_hi=dense_hi, ps1_shape=shape, **meta)
    return "convert"


def _convert_one(path: str) -> tuple[str, str, int, int]:
    p = Path(path)
    before = p.stat().st_size
    try:
        status = convert_l4b_rim_file(p)
    except Exception as exc:  # noqa: BLE001 - reported per file, run continues
        log.warning("L4b rim conversion failed for %s: %s", p, exc)
        return (path, "error", before, before)
    return (path, status, before, p.stat().st_size)


def migrate_l4b_rim_store(
    l4b_root: Path | str,
    *,
    jobs: int = 1,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert every dense rim cache under ``l4b_root`` to the sparse layout."""
    root = Path(l4b_root)
    if not root.is_dir():
        raise FileNotFoundError(f"L4b cache root not found: {root}")

    files = iter_l4b_rim_files(root)
    if limit is not None:
        files = files[: int(limit)]
    log.info("Found %d L4b rim caches under %s", len(files), root)
    if dry_run:
        n_dense = 0
        for p in files[:200]:
            with np.load(p) as z:
                n_dense += 0 if l4b_rim_is_sparse(z) else 1
        return {
            "l4b_root": str(root),
            "n_files": len(files),
            "dry_run": True,
            "dense_in_first_200": n_dense,
        }

    t0 = time.perf_counter()
    counts = {"convert": 0, "skip": 0, "error": 0}
    bytes_before = bytes_after = 0

    if int(jobs) > 1 and files:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=int(jobs), prefer="processes")(
            delayed(_convert_one)(str(p)) for p in files
        )
    else:
        results = [_convert_one(str(p)) for p in files]

    for _path, status, before, after in results:
        counts[status] = counts.get(status, 0) + 1
        bytes_before += before
        bytes_after += after

    elapsed = time.perf_counter() - t0
    out = {
        "l4b_root": str(root),
        "format_version": L4B_RIM_FORMAT_VERSION,
        "n_files": len(files),
        "n_converted": counts["convert"],
        "n_already_sparse": counts["skip"],
        "n_errors": counts["error"],
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "elapsed_s": round(elapsed, 1),
    }
    log.info(
        "Converted %d/%d rim caches (%d already sparse, %d errors) in %.1fs",
        counts["convert"],
        len(files),
        counts["skip"],
        counts["error"],
        elapsed,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l4b-root", help="Path to an exact_cache_l4b/ directory")
    ap.add_argument("--data-root")
    ap.add_argument("--sector", type=int)
    ap.add_argument("--camera", type=int)
    ap.add_argument("--ccd", type=int)
    ap.add_argument("--oversampling", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.l4b_root:
        root = Path(args.l4b_root)
    else:
        if not all(
            v is not None for v in (args.data_root, args.sector, args.camera, args.ccd)
        ):
            ap.error("provide --l4b-root, or --data-root/--sector/--camera/--ccd")
        root = (
            Path(args.data_root)
            / f"s{int(args.sector):04d}"
            / f"c{int(args.camera)}"
            / f"k{int(args.ccd)}"
            / "remap"
            / f"oversampling_{int(args.oversampling)}"
            / "exact_cache_l4b"
        )

    result = migrate_l4b_rim_store(
        root, jobs=args.jobs, dry_run=args.dry_run, limit=args.limit
    )
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0 if int(result.get("n_errors", 0)) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
