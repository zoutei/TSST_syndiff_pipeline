# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Transcode a legacy dense packed ``fit_bundle.npz`` (bundle_version=1) into
tier-segmented storage (bundle_version=2, ``FitBundle.packed_tiers``).

Packed bundles pad every group's stamp/pixel arrays to the *global* max
``P`` across all ``(K,P)`` tiers, even though most groups belong to a much
smaller tier -- see ``dev/forward_epsf_wcs/README.md`` section 6/13 and
``fit_bundle.py``'s module docstring. On a crowded full-CCD bundle this is
~90% padding: ``fullccd_mag710_irreg_590`` decompresses to 11.15 GB of
``data``/``noise``/``weight`` for only 160,211 real pixels out of 1,570,816
padded slots.

This script re-buckets an existing v1 bundle's packed arrays by ``(K,P)``
tier (the same bucketing ``train_loop.run_stages_from_bundle`` /
``packed_support.bucket_packed_by_kp`` already do at train time) and writes
each bucket as its own ``PackedTier`` -- sized to its own true occupancy,
weight stored as uint8 -- instead of one globally-padded array. The output
is a *new* bundle directory; the source bundle is never modified.

Why transcode instead of re-export: the original workspace (FFI/PS1/Gaia
prep) that ``export_fit_bundle`` reads from may not be reachable from where
this runs (e.g. a Colab session with only the bundle files synced down).
Transcoding needs nothing but the existing ``fit_bundle.npz``.

Usage::

    python -m syndiff_pipeline.forward_model.scripts.transcode_bundle_tiered \\
        --bundle dev/forward_epsf_wcs/output/bundles/fullccd_mag710_irreg_590 \\
        --out-dir dev/forward_epsf_wcs/output/bundles/fullccd_mag710_irreg_590_tiered

Optionally pass ``--k-tiers``/``--p-tiers`` to re-bucket at a different
granularity than the bundle's own baked tiers (default: reuse
``bundle.k_tiers`` / ``bundle.p_tiers``, falling back to
``packed_support`` defaults).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .. import _bootstrap  # noqa: F401
from .. import packed_support as PS
from .. import fit_bundle as FB


def _dir_size_bytes(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def transcode(
    bundle: FB.FitBundle,
    *,
    k_tiers: tuple[int, ...] | None = None,
    p_tiers: tuple[int, ...] | None = None,
    log_fn=print,
) -> FB.FitBundle:
    """Return a new tier-segmented (bundle_version=2) copy of ``bundle``.

    ``bundle`` must be an existing packed bundle (dense v1 or, harmlessly,
    already tiered -- in which case its tiers are simply re-bucketed).
    """
    if not bundle.is_packed:
        raise ValueError("transcode_bundle_tiered: bundle is not packed (no pix_x)")

    kt = tuple(int(v) for v in (k_tiers or bundle.k_tiers or PS.DEFAULT_K_TIERS))
    pt = tuple(int(v) for v in (p_tiers or bundle.p_tiers or PS.DEFAULT_P_TIERS))

    if bundle.packed_tiers is not None:
        # Re-bucket tier-native rows.  In particular, never touch the lazy
        # dense properties: that defeats transcoding large v2 bundles.
        sizes_k = np.asarray(bundle.valid).sum(axis=1).astype(int)
        source: dict[int, tuple[FB.PackedTier, int]] = {}
        for tier in bundle.packed_tiers:
            for row, gi in enumerate(np.asarray(tier.group_idx, dtype=int)):
                source[int(gi)] = (tier, row)
        pt_eff = PS.ensure_tiers_cover(
            max((int(np.asarray(t.pix_valid).sum(axis=1).max()) for t in bundle.packed_tiers), default=0), pt
        )
        kt_eff = PS.ensure_tiers_cover(int(sizes_k.max()), kt)
        bins: dict[tuple[int, int], list[int]] = {}
        for gi in range(bundle.n_groups):
            src, row = source[gi]
            npix = int(np.count_nonzero(src.pix_valid[row] > .5))
            key = (PS._smallest_tier(int(sizes_k[gi]), kt_eff),
                   PS._smallest_tier(max(npix, 1), pt_eff))
            bins.setdefault(key, []).append(gi)
        new_tiers = []
        for (bkt, bpt), indices in sorted(bins.items()):
            ng, nf = len(indices), bundle.n_frames
            data = np.zeros((ng, nf, bpt), np.float32)
            noise = np.ones((ng, nf, bpt), np.float32)
            weight = np.zeros((ng, nf, bpt), np.uint8)
            px = np.zeros((ng, bpt), np.float32)
            py = np.zeros((ng, bpt), np.float32)
            pv = np.zeros((ng, bpt), np.float32)
            for out_row, gi in enumerate(indices):
                src, row = source[gi]
                take = min(int(src.p_tier), bpt)
                data[out_row, :, :take] = src.data[row, :, :take]
                noise[out_row, :, :take] = src.noise[row, :, :take]
                weight[out_row, :, :take] = src.weight_u8[row, :, :take]
                px[out_row, :take] = src.pix_x[row, :take]
                py[out_row, :take] = src.pix_y[row, :take]
                pv[out_row, :take] = src.pix_valid[row, :take]
            log_fn(f"  tier K={bkt} P={bpt}: n_groups={ng}")
            new_tiers.append(FB.PackedTier(
                data=data, noise=noise, weight_u8=weight, pix_x=px, pix_y=py,
                pix_valid=pv, group_idx=np.asarray(indices, np.int32),
                k_tier=bkt, p_tier=bpt,
            ))
        return FB.bundle_from_tiers(bundle, new_tiers)

    groups = bundle.group_set()
    pix_valid_full = np.asarray(bundle.pix_valid)
    buckets = PS.bucket_packed_by_kp(
        groups, bundle.stamp_center_x, bundle.stamp_center_y, pix_valid_full,
        k_tiers=kt, p_tiers=pt,
    )
    if not buckets:
        raise ValueError("transcode_bundle_tiered: no non-empty (K,P) buckets")

    data_full = np.asarray(bundle.data)
    noise_full = np.asarray(bundle.noise)
    weight_full = np.asarray(bundle.weight)
    pix_x_full = np.asarray(bundle.pix_x)
    pix_y_full = np.asarray(bundle.pix_y)

    bad = np.asarray(weight_full)
    off01 = ~np.isin(bad, (0.0, 1.0))
    if off01.any():
        n_bad = int(off01.sum())
        raise ValueError(
            f"transcode_bundle_tiered: {n_bad} weight values are not exactly "
            f"0.0/1.0 -- uint8 storage would be lossy for this bundle "
            f"(expected coverage * pix_valid, both binary)"
        )

    packed_tiers: list[FB.PackedTier] = []
    for bg, _cx, _cy, bidx, bkt, bpt in buckets:
        log_fn(f"  tier K={bkt} P={bpt}: n_groups={bg.n_groups}")
        packed_tiers.append(FB.PackedTier(
            data=np.ascontiguousarray(data_full[bidx][:, :, :bpt], dtype=np.float32),
            noise=np.ascontiguousarray(noise_full[bidx][:, :, :bpt], dtype=np.float32),
            weight_u8=np.ascontiguousarray(weight_full[bidx][:, :, :bpt], dtype=np.uint8),
            pix_x=np.ascontiguousarray(pix_x_full[bidx, :bpt], dtype=np.float32),
            pix_y=np.ascontiguousarray(pix_y_full[bidx, :bpt], dtype=np.float32),
            pix_valid=np.ascontiguousarray(pix_valid_full[bidx, :bpt], dtype=np.float32),
            group_idx=np.asarray(bidx, dtype=np.int32),
            k_tier=int(bkt),
            p_tier=int(bpt),
        ))

    return FB.bundle_from_tiers(bundle, packed_tiers)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="existing (v1) packed bundle dir/path")
    p.add_argument("--out-dir", required=True, help="output dir for the tiered (v2) bundle")
    p.add_argument("--k-tiers", default=None, help="comma list; default bundle.k_tiers")
    p.add_argument("--p-tiers", default=None, help="comma list; default bundle.p_tiers")
    args = p.parse_args(argv)

    src = Path(args.bundle)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir {out_dir} already exists and is non-empty; refusing to overwrite")

    k_tiers = tuple(int(v) for v in args.k_tiers.split(",")) if args.k_tiers else None
    p_tiers = tuple(int(v) for v in args.p_tiers.split(",")) if args.p_tiers else None

    print(f"loading {src} ...", flush=True)
    t0 = time.time()
    bundle = FB.load_fit_bundle(src)
    print(f"  loaded in {time.time() - t0:.1f}s: G={bundle.n_groups} T={bundle.n_frames} "
          f"packed={bundle.is_packed} already_tiered={bundle.packed_tiers is not None}", flush=True)

    src_dir = src if src.is_dir() else src.parent
    src_bytes = _dir_size_bytes(src_dir)
    if bundle.packed_tiers is None:
        data_gb = np.asarray(bundle.data).nbytes / 1e9
        noise_gb = np.asarray(bundle.noise).nbytes / 1e9
        weight_gb = np.asarray(bundle.weight).nbytes / 1e9
        n_valid = int(np.asarray(bundle.pix_valid).sum())
        n_padded = int(np.asarray(bundle.pix_valid).size)
    else:
        data_gb = sum(t.data.nbytes for t in bundle.packed_tiers) / 1e9
        noise_gb = sum(t.noise.nbytes for t in bundle.packed_tiers) / 1e9
        weight_gb = sum(t.weight_u8.nbytes for t in bundle.packed_tiers) / 1e9
        n_valid = sum(int(t.pix_valid.sum()) for t in bundle.packed_tiers)
        n_padded = sum(int(t.pix_valid.size) for t in bundle.packed_tiers)
    print(f"  in-memory source stamp arrays: data={data_gb:.3f} GB "
          f"noise={noise_gb:.3f} GB weight={weight_gb:.3f} GB "
          f"total={data_gb + noise_gb + weight_gb:.3f} GB", flush=True)
    if n_valid is not None:
        print(f"  valid pixels: {n_valid} of {n_padded} -> "
              f"{100 * (1 - n_valid / max(n_padded, 1)):.1f}% waste", flush=True)

    print("transcoding ...", flush=True)
    t0 = time.time()
    tiered = transcode(bundle, k_tiers=k_tiers, p_tiers=p_tiers)
    print(f"  transcoded in {time.time() - t0:.1f}s into {len(tiered.packed_tiers)} tier(s)", flush=True)

    tier_gb = sum(
        t.data.nbytes + t.noise.nbytes + t.weight_u8.nbytes
        + t.pix_x.nbytes + t.pix_y.nbytes + t.pix_valid.nbytes
        for t in tiered.packed_tiers
    ) / 1e9
    print(f"  in-memory (v2, tier-segmented, weight=uint8): total={tier_gb:.3f} GB", flush=True)

    print(f"saving -> {out_dir} ...", flush=True)
    t0 = time.time()
    out_path = FB.save_fit_bundle(out_dir, tiered)
    print(f"  saved in {time.time() - t0:.1f}s -> {out_path}", flush=True)

    out_bytes = _dir_size_bytes(out_dir)
    print(
        f"\non-disk (compressed): src={src_bytes / 1e6:.1f} MB  "
        f"new={out_bytes / 1e6:.1f} MB  "
        f"ratio={out_bytes / max(src_bytes, 1):.3f}",
        flush=True,
    )
    print(
        f"in-memory (decompressed, stamp arrays only): "
        f"v1={data_gb + noise_gb + weight_gb:.3f} GB -> v2={tier_gb:.3f} GB  "
        f"({100 * (1 - tier_gb / max(data_gb + noise_gb + weight_gb, 1e-9)):.1f}% reduction)",
        flush=True,
    )

    provenance = {
        "source_bundle": str(src),
        "k_tiers": list(k_tiers) if k_tiers else list(bundle.k_tiers),
        "p_tiers": list(p_tiers) if p_tiers else list(bundle.p_tiers),
        "n_packed_tiers": len(tiered.packed_tiers),
        "v1_stamp_arrays_gb": data_gb + noise_gb + weight_gb,
        "v2_stamp_arrays_gb": tier_gb,
    }
    (out_dir / "transcode_provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
