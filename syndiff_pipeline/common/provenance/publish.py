"""Worker-side publish: atomic artifact finalization + sidecar emission.

Producers build an artifact's bytes into a temp location, then call
:func:`publish_record` to (1) write a self-describing ``_provenance.json`` into
the artifact directory, (2) atomically rename it to its fingerprinted final
location, and (3) append a one-line sidecar record to the per-host spool that
the supervisor later drains into ``provenance.db`` (see ``ingest.py``).

Two invariants (plan §5, §10):

* **Content authority + self-describing store.** The finalized bytes at the
  fingerprinted key are the truth, and each artifact dir carries its own
  ``_provenance.json`` so ``reindex`` can rebuild the whole index from disk
  without recomputing recipes.
* **Atomic publish.** Build into ``tmp`` → ``os.replace`` to the final key. A
  crash leaves only an orphan temp, never a half-written artifact that looks
  present.

Dependency-free (stdlib only). The actual array writing (zarr) is done by the
caller's build step; this module only handles finalization + bookkeeping, so it
stays importable in light contexts.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, Dict

from syndiff_pipeline.common.scc_paths import provenance_spool_file

STORE_PROVENANCE_BASENAME = "_provenance.json"


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write *obj* as JSON to *path* atomically (tmp file + fsync + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_store_provenance(artifact_dir: str | Path, record: Dict[str, Any]) -> Path:
    """Write ``_provenance.json`` (the full ingest record) into *artifact_dir*."""
    dest = Path(artifact_dir) / STORE_PROVENANCE_BASENAME
    _atomic_write_json(dest, record)
    return dest


def read_store_provenance(artifact_dir: str | Path) -> Dict[str, Any] | None:
    """Read ``_provenance.json`` from *artifact_dir*, or None if absent/malformed."""
    path = Path(artifact_dir) / STORE_PROVENANCE_BASENAME
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def emit_sidecar_to(record: Dict[str, Any], spool_file: str | Path) -> None:
    """Append *record* as one JSON line to *spool_file* (``O_APPEND``, lock-free).

    Open/append/close per record so a concurrent supervisor rotate never races a
    held file descriptor: at any instant the file is closed between writes.
    """
    path = Path(spool_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def emit_sidecar(
    record: Dict[str, Any],
    data_root: str | Path,
    *,
    host: str | None = None,
    pid: int | None = None,
) -> Path:
    """Append *record* to this host/pid's spool file under *data_root*.

    Returns the spool file path written.
    """
    host = host or socket.gethostname()
    pid = os.getpid() if pid is None else pid
    spool_file = provenance_spool_file(data_root, host, pid)
    emit_sidecar_to(record, spool_file)
    return spool_file


def publish_record(
    record: Dict[str, Any],
    *,
    tmp_dir: str | Path,
    final_dir: str | Path,
    data_root: str | Path,
    emit: bool = True,
) -> Path:
    """Finalize an already-built artifact directory and record its provenance.

    Steps: write ``_provenance.json`` into *tmp_dir*; atomically rename *tmp_dir*
    to *final_dir* (its fingerprinted key); optionally emit the sidecar. The
    caller is responsible for having written the artifact's array bytes into
    *tmp_dir* first.

    Idempotent-ish: if *final_dir* already exists (another worker won the race),
    the temp is discarded and the existing artifact is kept — identical bytes by
    construction (same fingerprint).

    Returns the final directory path.
    """
    tmp_path = Path(tmp_dir)
    final_path = Path(final_dir)
    write_store_provenance(tmp_path, record)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        # Lost the race; our identical build is redundant. Clean up the temp.
        _rmtree_quiet(tmp_path)
    else:
        try:
            os.replace(tmp_path, final_path)
        except OSError:
            # Concurrent winner between the check and the rename: keep theirs.
            if final_path.exists():
                _rmtree_quiet(tmp_path)
            else:
                raise
    if emit:
        emit_sidecar(record, data_root)
    return final_path


def _rmtree_quiet(path: Path) -> None:
    """Best-effort recursive delete of a temp dir; never raises."""
    import shutil

    try:
        shutil.rmtree(path)
    except OSError:  # pragma: no cover - defensive
        pass
