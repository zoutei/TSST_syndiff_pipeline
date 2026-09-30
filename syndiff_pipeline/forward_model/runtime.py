# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""CPU thread / XLA hygiene for forward_epsf_wcs Adam steps.

Call ``configure_cpu_threads`` once at process start (before heavy JAX work)
so Eigen/XLA/OpenMP do not oversubscribe a contended multi-socket host.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def log(msg: str) -> None:
    """Print a flushed log line with a local timestamp prefix."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def default_thread_count() -> int:
    """Sweet-spot default: one NUMA node worth of cores, capped at 16.

    The Adam step is gather/bandwidth-bound; 64 threads thrash DRAM. Prefer
    ``SYNDIFF_FIT_THREADS`` or ``OMP_NUM_THREADS`` when already set.
    """
    for key in ("SYNDIFF_FIT_THREADS", "OMP_NUM_THREADS", "XLA_NUM_THREADS"):
        raw = os.environ.get(key)
        if raw and raw.strip().isdigit() and int(raw) > 0:
            return int(raw)
    try:
        n = os.cpu_count() or 8
    except Exception:
        n = 8
    # 4-socket hosts in this lab have 16 cores/socket; stay on one node.
    return max(1, min(16, n // 4 if n >= 32 else min(16, n)))


def configure_cpu_threads(n_threads: int | None = None, *, force: bool = False) -> dict[str, str]:
    """Set process env for XLA/OpenMP/BLAS if unset (or always when ``force``).

    On GPU backends, skip ``xla_force_host_platform_device_count`` so CUDA JAX
    is not forced onto the host CPU.

    Returns the effective env mapping. Safe to call multiple times.
    """
    n = int(n_threads) if n_threads is not None else default_thread_count()
    n = max(1, n)

    prefer_gpu = False
    try:
        # May run before jax is imported elsewhere; safe no-op if unavailable.
        import jax  # noqa: WPS433

        prefer_gpu = str(jax.default_backend()).lower() == "gpu"
    except Exception:
        prefer_gpu = os.environ.get("JAX_PLATFORMS", "").lower().startswith("cuda") or (
            "gpu" in os.environ.get("JAX_PLATFORMS", "").lower()
        )

    if prefer_gpu:
        xla_flags = "--xla_gpu_autotune_level=1"
    else:
        xla_flags = (
            "--xla_cpu_multi_thread_eigen=true "
            "--xla_force_host_platform_device_count=1"
        )

    keys = {
        "OMP_NUM_THREADS": str(n),
        "OPENBLAS_NUM_THREADS": str(n),
        "MKL_NUM_THREADS": str(n),
        "NUMEXPR_NUM_THREADS": str(n),
        "XLA_FLAGS": xla_flags,
    }
    # Prefer an explicit intra-op thread cap when supported via env.
    if "XLA_NUM_THREADS" not in os.environ or force:
        keys["XLA_NUM_THREADS"] = str(n)
    applied = {}
    for k, v in keys.items():
        if force or k not in os.environ or (
            k == "XLA_FLAGS" and not prefer_gpu and "xla_cpu" not in os.environ.get(k, "")
        ):
            if k == "XLA_FLAGS" and os.environ.get(k) and not force:
                # Append rather than clobber user flags.
                existing = os.environ[k].strip()
                if prefer_gpu:
                    applied[k] = existing
                else:
                    extra = "--xla_cpu_multi_thread_eigen=true"
                    if extra not in existing:
                        os.environ[k] = f"{existing} {extra}".strip()
                    applied[k] = os.environ[k]
            else:
                if force or k not in os.environ:
                    os.environ[k] = v
                    applied[k] = v
                else:
                    applied[k] = os.environ[k]
        else:
            applied[k] = os.environ[k]
    # Always record the resolved worker count.
    os.environ.setdefault("SYNDIFF_FIT_THREADS", str(n))
    applied["SYNDIFF_FIT_THREADS"] = os.environ["SYNDIFF_FIT_THREADS"]
    applied["effective_threads"] = str(n)
    applied["prefer_gpu"] = str(prefer_gpu)
    return applied


DEFAULT_JAX_CACHE_DIR = str(Path.home() / ".syndiff" / "jax_cache")


def cpu_cache_tag() -> str:
    """Short tag identifying this host's CPU instruction-set capabilities.

    Two nodes with identical flags share a cache entry; two that differ never do.
    """
    import hashlib  # noqa: PLC0415
    import platform  # noqa: PLC0415

    flags = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("flags"):
                    flags = " ".join(sorted(line.split(":", 1)[1].split()))
                    break
    except OSError:
        flags = ""
    if not flags:  # non-Linux or unreadable: fall back to the node name
        return platform.node().split(".")[0] or "unknown"
    return f"{platform.machine()}-{hashlib.sha1(flags.encode()).hexdigest()[:10]}"


def configure_jax_cache(cache_dir: str | None = None) -> str:
    """Point JAX's persistent compilation cache at a shared (NFS) directory.

    Many shard jobs across the cluster compile largely identical program shapes
    (same stage structure, same stamp size, same K-bucket tiers); a shared cache
    means only the first job per (stage, shape) pays JIT-compile cost.

    **The cache is namespaced per CPU capability set, and must stay that way.**
    ``$HOME`` is shared NFS across the plscience nodes, which do NOT all have the
    same instruction sets. JAX's cache key does not fully separate target machine
    features, so a flat shared directory lets one node load an AOT executable
    another node compiled, which XLA reports as::

        Loading XLA:CPU AOT result. Target machine feature +prefer-no-gather is
        not supported on the host machine. Machine type used for XLA:CPU
        compilation doesn't match the machine type for execution.
        This could lead to execution errors such as SIGILL.

    That is the most likely source of the long-standing "JAX on CPU segfaults on
    this bundle" behaviour, which blocked CPU diagnostics for months. Appending
    ``cpu_cache_tag()`` keeps the cross-node sharing benefit between *identical*
    nodes while making a mismatched load impossible.

    Env-var-only (no ``import jax`` here), matching ``configure_cpu_threads``'s
    pattern -- must be set before JAX's backend initializes, so call this early in
    ``main()``. Safe to call multiple times / already-set.
    """
    path = cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR") or DEFAULT_JAX_CACHE_DIR
    path = str(Path(path) / cpu_cache_tag())
    Path(path).mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = path
    return path


def write_thread_report(path: Path, applied: dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in sorted(applied.items())]
    path.write_text("\n".join(lines) + "\n")
