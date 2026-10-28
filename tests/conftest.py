"""Suite-wide test isolation.

Scratch files must not live on NFS.  The project rules keep ``TMPDIR`` off ``/tmp`` (4.9 GB, shared) and on
``/astro`` (NFS), but many orchestration tests create SQLite state DBs, lease files and logs inside a
``TemporaryDirectory`` / ``tmp_path`` and then delete the tree while a handle is still open.  On NFS that leaves
``.nfsXXXX`` silly-rename files, so ``rmtree`` fails with ``OSError: [Errno 39] Directory not empty``; mtime
granularity and attribute caching also make lease/heartbeat staleness checks flip.  The failing set then changes
from run to run (2026-10-27/28 investigation, ``tests/CLAUDE`` notes in the commit message).

So: when the effective temp directory is on NFS, redirect Python's ``tempfile`` (and pytest's ``basetemp``) to a
per-process directory under ``/dev/shm`` (RAM, not ``/tmp``) for the duration of the run.  Set
``SYNDIFF_TEST_ALLOW_NFS_TMP=1`` to keep the NFS location.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_SHM = Path("/dev/shm")
_redirect_dirs: list[Path] = []


def _fstype(path: Path) -> str:
    """Filesystem type of the longest mount point containing ``path`` ('' when unknown)."""
    try:
        real = os.path.realpath(path)
        best, best_type = "", ""
        with open("/proc/self/mountinfo", encoding="utf-8") as fh:
            for line in fh:
                left, _, right = line.partition(" - ")
                fields = left.split()
                mount_point = fields[4].replace("\\040", " ")
                if (real == mount_point or real.startswith(mount_point.rstrip("/") + "/")) and len(mount_point) >= len(best):
                    best, best_type = mount_point, right.split()[0]
        return best_type
    except (OSError, IndexError):
        return ""


def _on_nfs(path: Path) -> bool:
    return _fstype(path).startswith("nfs")


def _redirect_root() -> Path | None:
    if os.environ.get("SYNDIFF_TEST_ALLOW_NFS_TMP") == "1" or not _SHM.is_dir() or not os.access(_SHM, os.W_OK):
        return None
    if not _on_nfs(Path(tempfile.gettempdir())):
        return None
    root = _SHM / f"syndiff-tests-{os.getuid()}" / str(os.getpid())
    root.mkdir(parents=True, exist_ok=True)
    _redirect_dirs.append(root)
    return root


# Runs at conftest import, i.e. in the main process and in every xdist worker, before any test module is imported.
_root = _redirect_root()
if _root is not None:
    tempfile.tempdir = str(_root)
    os.environ["TMPDIR"] = str(_root)


def pytest_configure(config):
    base = getattr(config.option, "basetemp", None)
    if base is not None and _root is not None and _on_nfs(Path(base).parent):
        config.option.basetemp = str(_root / "basetemp")


def pytest_unconfigure(config):
    for d in _redirect_dirs:
        shutil.rmtree(d, ignore_errors=True)
