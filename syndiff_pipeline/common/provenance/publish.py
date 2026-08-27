"""
publish.py
==========
Producer-side atomic publish + lock-free sidecar transport (§10).

Two shapes of artifact:

- **Directory-shaped** (:func:`publish_dir`): combined/convolved skycell
  cells, remap stores, mapping dirs -- anything with multiple files under one
  fingerprinted directory. A ``_provenance.json`` is written *inside* the
  directory so the store is self-describing for :mod:`reindex` even with an
  empty/stale DB.
- **File-shaped** (:func:`publish_record`): a single fingerprinted file (e.g.
  one per-FFI diff FITS living inside a shared per-recipe directory, decision
  #12 -- individual files do not get their own directory or sidecar file;
  their provenance is carried entirely by the spool record).

Both follow the same publish protocol: write under a ``_tmp_{fp}_{pid}``
sibling in the destination's parent directory, then a single atomic
``os.replace`` onto the fingerprinted key. A crash mid-write leaves only the
``_tmp_*`` orphan -- never a partial that looks complete (§5 invariants,
§17 failure matrix).

After the rename, one JSON line describing the publish is appended to
``bookkeeping/spool/{host}.{pid}.jsonl`` opened with ``O_APPEND`` (lock-free;
each producer process owns its own spool file, so no two writers ever share
an fd). The supervisor is the sole reader/drainer (:mod:`ingest`).

``try_publish_dir`` / ``try_publish_record`` are best-effort wrappers that
never raise into the caller's compute path -- publish failures are logged
and swallowed, matching the "Try/except-guarded, non-fatal" pattern used
elsewhere for checkpoint sidecars (§11).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

log = logging.getLogger(__name__)

__all__ = [
    "PROVENANCE_JSON_BASENAME",
    "host_pid_tag",
    "git_sha",
    "default_spool_path",
    "build_record",
    "append_spool_record",
    "publish_dir",
    "publish_record",
    "try_publish_dir",
    "try_publish_record",
]

PROVENANCE_JSON_BASENAME = "_provenance.json"


def host_pid_tag() -> str:
    """``{host}.{pid}`` tag used for spool filenames and tmp-dir uniqueness."""
    return f"{socket.gethostname()}.{os.getpid()}"


# Sentinel distinguishing "not resolved yet" from a legitimately cached
# ``None`` (e.g. running from a tarball with no ``.git``) -- see git_sha().
_UNRESOLVED = object()
_git_sha_cache: Any = _UNRESOLVED


def git_sha() -> Optional[str]:
    """Git SHA of the running checkout, or ``None``. Cached per process; never raises.

    Resolved once via ``git rev-parse HEAD`` run with ``cwd`` set to this
    module's own directory (inside the package, so ``git`` finds the
    enclosing repo by walking up from there regardless of where the process
    itself was launched from). Because there is no deploy step in this
    project -- ``pip install -e .`` means the daemon and every re-executed
    Condor job import live source straight off disk, sometimes hours after
    submission (see CLAUDE.md invariant 14) -- this is often the only way to
    tell which code version actually produced a given artifact, so the full
    40-character SHA is stored (unambiguous for ``git show``/``git log``),
    not an abbreviated form.

    Never raises and never blocks past the 5s subprocess timeout: a missing
    ``git`` binary, a checkout with no ``.git`` (e.g. an extracted tarball),
    a non-zero exit, or a timeout all resolve to ``None`` -- which is then
    cached just like a real SHA, so a broken environment is not retried on
    every subsequent call.
    """
    global _git_sha_cache
    if _git_sha_cache is _UNRESOLVED:
        _git_sha_cache = _resolve_git_sha()
    return _git_sha_cache


def _resolve_git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    except Exception:  # noqa: BLE001 - this helper must never raise
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def default_spool_path(spool_dir: str | Path) -> Path:
    """Default per-process spool file: ``{spool_dir}/{host}.{pid}.jsonl``."""
    return Path(spool_dir) / f"{host_pid_tag()}.jsonl"


def _utcnow_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_record(
    fingerprint: str,
    kind: str,
    spatial_key: Mapping[str, Any],
    recipe_id: str,
    code_version: int,
    input_fps: Iterable[str],
    location: str,
    *,
    recipe_params: Optional[Mapping[str, Any]] = None,
    state: str = "complete",
    bytes_: Optional[int] = None,
    wall_time_s: Optional[float] = None,
    produced_by: Optional[str] = None,
    created_at: Optional[str] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> dict:
    """
    Build the JSON-serializable record shared by ``_provenance.json`` and the
    spool line. Both carry the full recipe (not just its id) so ``ingest``
    can populate the ``recipes`` table without a second lookup.

    ``git_sha`` (this process's :func:`git_sha`, cached) is always stamped
    onto the record alongside the recipe, purely as descriptive metadata --
    it is never an input to ``recipe_id``/``fingerprint`` (see
    :mod:`fingerprint`), so its value can never change an artifact's or
    recipe's identity.
    """
    record: dict[str, Any] = {
        "fingerprint": str(fingerprint),
        "kind": str(kind),
        "spatial_key": dict(spatial_key),
        "recipe_id": str(recipe_id),
        "code_version": int(code_version),
        "inputs": sorted(str(fp) for fp in input_fps),
        "location": str(location),
        "state": str(state),
        "bytes": bytes_,
        "wall_time_s": wall_time_s,
        "produced_by": produced_by or host_pid_tag(),
        "created_at": created_at or _utcnow_iso(),
        "git_sha": git_sha(),
    }
    if recipe_params is not None:
        record["recipe_params"] = dict(recipe_params)
    if meta:
        record["meta"] = dict(meta)
    return record


def append_spool_record(spool_dir: str | Path, record: Mapping[str, Any]) -> Path:
    """
    Append one JSON line to this process's spool file with ``O_APPEND``.

    Lock-free: each producer process writes only to its own
    ``{host}.{pid}.jsonl``, and POSIX guarantees ``O_APPEND`` writes below
    ``PIPE_BUF`` are atomic w.r.t. other appenders to the same file -- which
    never happens here since the filename already encodes the pid.
    """
    spool_dir = Path(spool_dir)
    spool_dir.mkdir(parents=True, exist_ok=True)
    path = default_spool_path(spool_dir)
    line = (json.dumps(dict(record), sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return path


def _unique_tmp_dir(parent: Path, fp: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    # Include a uuid4 suffix so two threads in the same process (same pid)
    # never collide on the same tmp name.
    return parent / f"_tmp_{fp}_{host_pid_tag()}_{uuid.uuid4().hex[:8]}"


def _cleanup_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:  # pragma: no cover - best-effort cleanup
        pass


def _cleanup_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:  # pragma: no cover
        pass


def publish_dir(
    dest_root: str | Path,
    fingerprint: str,
    kind: str,
    spatial_key: Mapping[str, Any],
    recipe_id: str,
    code_version: int,
    input_fps: Iterable[str],
    write_payload: Callable[[Path], None],
    *,
    recipe_params: Optional[Mapping[str, Any]] = None,
    spool_dir: Optional[str | Path] = None,
    produced_by: Optional[str] = None,
    wall_time_s: Optional[float] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Atomically publish a directory-shaped artifact at ``dest_root/{fp}/``.

    *write_payload(tmp_dir)* must populate *tmp_dir* with the artifact's
    files; a ``_provenance.json`` sidecar is then written inside it before
    the single atomic rename onto the fingerprinted key.

    Idempotent under concurrent publishers of the same fingerprint: if
    another process's rename already landed the final directory first,
    ``os.replace`` fails with ``ENOTEMPTY``/``EEXIST``; this is detected and
    treated as success (content-addressed, so the bytes are the same) rather
    than raised -- both writers' tmp dirs are cleaned up either way (§17:
    "both rename to same key; identical bytes; idempotent sidecars").
    """
    dest_root = Path(dest_root)
    final_dir = dest_root / str(fingerprint)
    tmp_dir = _unique_tmp_dir(dest_root, str(fingerprint))
    tmp_dir.mkdir(parents=True, exist_ok=False)

    record = build_record(
        fingerprint,
        kind,
        spatial_key,
        recipe_id,
        code_version,
        input_fps,
        str(final_dir),
        recipe_params=recipe_params,
        wall_time_s=wall_time_s,
        produced_by=produced_by,
        meta=meta,
    )

    try:
        write_payload(tmp_dir)
        (tmp_dir / PROVENANCE_JSON_BASENAME).write_text(
            json.dumps(record, sort_keys=True, indent=2), encoding="utf-8"
        )
        try:
            os.replace(tmp_dir, final_dir)
        except OSError:
            if final_dir.is_dir():
                # Another publisher already landed this fingerprint first.
                _cleanup_dir(tmp_dir)
            else:
                raise
    except BaseException:
        _cleanup_dir(tmp_dir)
        raise

    if spool_dir is not None:
        append_spool_record(spool_dir, record)

    return final_dir


def publish_record(
    dest_path: str | Path,
    fingerprint: str,
    kind: str,
    spatial_key: Mapping[str, Any],
    recipe_id: str,
    code_version: int,
    input_fps: Iterable[str],
    write_payload: Callable[[Path], None],
    *,
    recipe_params: Optional[Mapping[str, Any]] = None,
    spool_dir: Optional[str | Path] = None,
    produced_by: Optional[str] = None,
    wall_time_s: Optional[float] = None,
    bytes_: Optional[int] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Atomically publish a single fingerprinted file at *dest_path*.

    *write_payload(tmp_path)* must write the file's bytes to *tmp_path*
    (e.g. a FITS write, or ``fits_io.write_image_fits`` targeting the tmp
    name). File replace is always atomic on POSIX regardless of whether
    *dest_path* pre-existed, so no idempotency race handling is needed here
    (unlike directories).
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.parent / f"_tmp_{fingerprint}_{host_pid_tag()}_{uuid.uuid4().hex[:8]}{dest_path.suffix}"

    record = build_record(
        fingerprint,
        kind,
        spatial_key,
        recipe_id,
        code_version,
        input_fps,
        str(dest_path),
        recipe_params=recipe_params,
        wall_time_s=wall_time_s,
        produced_by=produced_by,
        bytes_=bytes_,
        meta=meta,
    )

    try:
        write_payload(tmp_path)
        os.replace(tmp_path, dest_path)
    except BaseException:
        _cleanup_file(tmp_path)
        raise

    if spool_dir is not None:
        append_spool_record(spool_dir, record)

    return dest_path


def try_publish_dir(*args: Any, **kwargs: Any) -> Optional[Path]:
    """Best-effort :func:`publish_dir`: logs and swallows any exception."""
    try:
        return publish_dir(*args, **kwargs)
    except Exception:  # noqa: BLE001 - provenance publish must never break compute
        log.warning("provenance publish_dir failed (non-fatal)", exc_info=True)
        return None


def try_publish_record(*args: Any, **kwargs: Any) -> Optional[Path]:
    """Best-effort :func:`publish_record`: logs and swallows any exception."""
    try:
        return publish_record(*args, **kwargs)
    except Exception:  # noqa: BLE001
        log.warning("provenance publish_record failed (non-fatal)", exc_info=True)
        return None
