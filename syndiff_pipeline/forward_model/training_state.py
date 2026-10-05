# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Atomic, non-pickle training-state checkpoints for exact Adam continuation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def fingerprint(bundle_path: Path, config: dict) -> str:
    digest = hashlib.sha256()
    with Path(bundle_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def save(path: Path, *, params: dict, opt_state, metadata: dict,
         extra_arrays: dict[str, np.ndarray] | None = None) -> None:
    """Write arrays atomically; optimizer structure is reconstructed from its template."""
    leaves, _ = jax.tree_util.tree_flatten(opt_state)
    arrays = {f"param__{name}": np.asarray(value) for name, value in params.items()}
    arrays.update({f"opt__{i:04d}": np.asarray(value) for i, value in enumerate(leaves)})
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    arrays.update({f"extra__{k}": np.asarray(v) for k, v in (extra_arrays or {}).items()})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    with tmp.open("rb") as stream: os.fsync(stream.fileno())
    os.replace(tmp, path)


def load(path: Path) -> tuple[dict, list[np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as data:
        params = {k[7:]: np.asarray(data[k]) for k in data.files if k.startswith("param__")}
        opt = [np.asarray(data[k]) for k in sorted(k for k in data.files if k.startswith("opt__"))]
        metadata = json.loads(str(data["metadata_json"]))
    # State-exact resume restores raw params AND the optimizer's per-leaf
    # Adam moments (`opt__*`, matched purely by tree position/shape in
    # `restore_optimizer`) bit-for-bit. A legacy (58-grid) `epsf_base_raw`/
    # `epsf_modes` here means the accompanying `opt__*` moment buffers are
    # ALSO 58-sized -- there is no valid conversion of those (a per-element
    # Adam accumulator has no meaning after a representation change that
    # redefines what each grid cell means), so this deliberately does NOT
    # auto-convert like the other load sites; it fails loudly instead of
    # silently resuming Adam moments against the wrong grid.
    from . import epsf_model as EM  # noqa: E402  (local: avoid a module-load cycle)
    base_raw = params.get("epsf_base_raw")
    if base_raw is not None and EM.is_subpixel_grid(int(np.asarray(base_raw).shape[-1])):
        raise ValueError(
            f"{path}: state-exact resume checkpoint predates the pixel-integrated "
            "ePSF representation (epsf_base_raw is a legacy 58-grid). The optimizer's "
            "Adam moment buffers for this leaf are also 58-sized and cannot be "
            "meaningfully converted -- resume from a fresh init (e.g. --init-epsf-base "
            "with an already-converted params*.npz) instead of --resume-state."
        )
    return params, opt, metadata


def load_extras(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k[7:]: np.asarray(data[k]) for k in data.files if k.startswith("extra__")}


def encode_two_level_gate(state) -> tuple[dict, dict[str, np.ndarray]]:
    if state is None:
        return {}, {}
    meta = {"scale": state.scale, "n_refreshes": state.n_refreshes,
            "bucket_ids": sorted(state._buckets)}
    arrays = {}
    for bucket_id, bucket in state._buckets.items():
        if bucket.streak is not None: arrays[f"gate_{bucket_id}_streak"] = bucket.streak
        if bucket.committed_reject is not None:
            arrays[f"gate_{bucket_id}_committed"] = bucket.committed_reject
    return meta, arrays


def decode_two_level_gate(meta: dict, arrays: dict[str, np.ndarray]):
    if not meta:
        return None
    from .stamp_reject import Level2GateState
    state = Level2GateState(scale=meta.get("scale"), n_refreshes=int(meta.get("n_refreshes", 0)))
    for bucket_id in meta.get("bucket_ids", []):
        bucket = state.bucket(int(bucket_id))
        bucket.streak = arrays.get(f"gate_{bucket_id}_streak")
        committed = arrays.get(f"gate_{bucket_id}_committed")
        bucket.committed_reject = None if committed is None else committed.astype(bool)
    return state


def restore_optimizer(template, leaves: list[np.ndarray]):
    current, treedef = jax.tree_util.tree_flatten(template)
    if len(current) != len(leaves):
        raise ValueError(f"optimizer leaf count differs: checkpoint={len(leaves)} runtime={len(current)}")
    restored = []
    for expected, value in zip(current, leaves):
        if np.shape(expected) != np.shape(value):
            raise ValueError(f"optimizer leaf shape differs: {np.shape(value)} != {np.shape(expected)}")
        restored.append(jnp.asarray(value, dtype=getattr(expected, "dtype", None)))
    return jax.tree_util.tree_unflatten(treedef, restored)


def validate(metadata: dict, *, expected_fingerprint: str, reject_every: int) -> None:
    if reject_every != 0:
        raise ValueError("state-exact resume requires --reject-every 0")
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError("resume state bundle/configuration fingerprint is incompatible")
