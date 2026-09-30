# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
#!/usr/bin/env python3
"""Standard-library remote process supervisor used by ``colab_job.py``."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

TERMINAL = {"completed", "failed", "stopped"}
NOTIFY_TERMINAL = {"completed", "failed"}
EMERGENCY_TERMINAL = {"completed", "failed", "stopped"}
HISTORY_RE = re.compile(r"stage\s+(\d+)\s+step\s+(\d+).*?loss=([+\-\d.eE]+)")
STEP_START_RE = re.compile(r"stage\s+(\d+)\s+step\s+(\d+).*starting")
TRAIN_LOG_STALL_WARN_S = 90.0
STOP_DRAIN_GRACE_S = 120.0
_DISCORD_MAX_CONTENT = 2000
_WEBHOOK_TIMEOUT_S = 5.0


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def parse_meminfo(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        match = re.match(r"(MemTotal|MemAvailable):\s+(\d+)\s+kB", line)
        if match:
            values[{"MemTotal": "ram_total_bytes", "MemAvailable": "ram_available_bytes"}[match.group(1)]] = int(match.group(2)) * 1024
    return values


def parse_nvidia_smi(text: str) -> dict[str, float | int | str]:
    """Parse the CSV emitted by :func:`gpu_metrics`; tolerant of N/A fields."""
    row = next((line for line in text.splitlines() if line.strip()), "")
    fields = [part.strip() for part in row.split(",")]
    names = ("gpu_name", "vram_total_mb", "vram_used_mb", "vram_free_mb",
             "gpu_util_pct", "gpu_temperature_c", "gpu_power_w")
    out: dict[str, float | int | str] = {}
    for name, raw in zip(names, fields):
        if name == "gpu_name":
            out[name] = raw
            continue
        try:
            value = float(raw)
            out[name] = int(value) if name.endswith(("_mb", "_pct", "_c")) else value
        except ValueError:
            pass
    return out


def process_tree_rss(root_pid: int) -> tuple[int, list[int]]:
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        ppid = re.search(r"^PPid:\s+(\d+)", status, re.M)
        vmrss = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.M)
        pid = int(entry.name)
        parents[pid] = int(ppid.group(1)) if ppid else 0
        rss[pid] = (int(vmrss.group(1)) * 1024) if vmrss else 0
    tree = {root_pid}
    changed = True
    while changed:
        before = len(tree)
        tree.update(pid for pid, parent in parents.items() if parent in tree)
        changed = len(tree) != before
    return sum(rss.get(pid, 0) for pid in tree), sorted(tree)


def gpu_metrics() -> dict[str, object]:
    query = "name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return cast(dict[str, object], parse_nvidia_smi(result.stdout))
    except (OSError, subprocess.SubprocessError):
        return {}


def train_log_stats(log: Path, now: float) -> dict[str, object]:
    """Parse tail of ``train.log`` plus file age/size for telemetry and stall detection."""
    out: dict[str, object] = {}
    try:
        stat = log.stat()
        out["train_log_size_bytes"] = stat.st_size
        out["train_log_mtime_age_s"] = now - stat.st_mtime
    except FileNotFoundError:
        return out
    except OSError:
        return out
    try:
        data = log.read_bytes()[-131072:].decode(errors="replace")
    except OSError:
        return out
    completes = list(HISTORY_RE.finditer(data))
    starts = list(STEP_START_RE.finditer(data))
    if completes:
        match = completes[-1]
        out.update(
            stage=int(match.group(1)),
            step=int(match.group(2)),
            loss=float(match.group(3)),
        )
    if starts:
        start_match = starts[-1]
        out["trainer_stage_start"] = int(start_match.group(1))
        out["trainer_step_start"] = int(start_match.group(2))
    if completes and starts:
        last_complete_pos = completes[-1].start()
        last_start_pos = starts[-1].start()
        start_step = int(starts[-1].group(2))
        complete_step = int(completes[-1].group(2))
        if start_step > complete_step or (start_step == complete_step and last_start_pos > last_complete_pos):
            out["trainer_step_in_flight"] = start_step
    return out


def checkpoint_age(job_dir: Path, now: float) -> float | None:
    candidates = list(job_dir.glob("*.npz")) + list(job_dir.glob("checkpoints/*.npz"))
    return now - max(p.stat().st_mtime for p in candidates) if candidates else None


def artifact_manifest(job_dir: Path) -> dict[str, dict[str, object]]:
    from .colab_emergency_upload import is_private_artifact

    result = {}
    for path in sorted(job_dir.rglob("*")):
        if not path.is_file() or ".tmp." in path.name or path.name == "artifact_manifest.json":
            continue
        rel = str(path.relative_to(job_dir))
        if is_private_artifact(rel):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[rel] = {"size": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_discord_message(state: dict) -> str:
    run_id = str(state.get("run_id", "?"))
    terminal = str(state.get("state", "?"))
    lines = [f"[Colab ePSF] {run_id} {terminal}"]
    started = state.get("started_at")
    ended = state.get("ended_at")
    if started is not None and ended is not None:
        lines.append(
            f"duration: {_format_duration(float(ended) - float(started))} | exit={state.get('exit_code', '?')}"
        )
    stage = state.get("stage")
    step = state.get("step")
    loss = state.get("loss")
    if stage is not None and step is not None:
        tail = f"stage {stage} step {step}"
        if loss is not None:
            tail += f" | loss={loss:g}"
        lines.append(tail)
    gpu_name = state.get("gpu_name")
    peak_vram = state.get("peak_vram_used_mb")
    if gpu_name or peak_vram:
        gpu_bits = []
        if gpu_name:
            gpu_bits.append(str(gpu_name))
        if peak_vram:
            gpu_bits.append(f"peak VRAM {int(peak_vram)} MB")
        lines.append("gpu: " + " | ".join(gpu_bits))
    return "\n".join(lines)


def post_discord_webhook(url: str, content: str) -> None:
    payload = json.dumps({"content": content[:_DISCORD_MAX_CONTENT]}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "syndiff-colab",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_WEBHOOK_TIMEOUT_S) as resp:
        resp.read()


def resolve_discord_webhook_url(cli_url: str | None) -> str | None:
    if cli_url:
        return cli_url.strip() or None
    env_url = os.environ.get("SYNDIFF_DISCORD_WEBHOOK_URL", "").strip()
    return env_url or None


def maybe_emergency_drive_release(
    job_dir: Path,
    run_id: str,
    state: dict,
    manifest: dict[str, dict[str, object]],
    status_path: Path,
    *,
    grace_s: float,
    enabled: bool,
    max_upload_attempts: int = 3,
    poll_s: float = 15.0,
) -> None:
    if not enabled or grace_s <= 0:
        return
    terminal = state.get("state")
    if terminal not in EMERGENCY_TERMINAL:
        return
    ended_at = float(state.get("ended_at", time.time()))
    grace_until = ended_at + grace_s
    state = dict(state)
    state["emergency_grace_until"] = grace_until
    atomic_json(status_path, state)
    while time.time() < grace_until:
        time.sleep(min(poll_s, max(0.0, grace_until - time.time())))
    from . import colab_emergency_upload as emergency  # noqa: PLC0415

    file_id, error = emergency.emergency_upload_and_release(
        job_dir,
        run_id,
        cast(dict[str, object], manifest),
        max_upload_attempts=max_upload_attempts,
    )
    state = dict(state)
    state.pop("emergency_drive_upload_error", None)
    if file_id:
        state["emergency_drive_file_id"] = file_id
        state["emergency_drive_uploaded_at"] = time.time()
    else:
        state["emergency_drive_upload_error"] = error or "emergency upload failed"
    state["heartbeat_at"] = time.time()
    atomic_json(status_path, state)


def maybe_notify_discord(state: dict, webhook_url: str | None) -> None:
    if not webhook_url or state.get("state") not in NOTIFY_TERMINAL:
        return
    try:
        post_discord_webhook(webhook_url, format_discord_message(state))
    except (OSError, urllib.error.URLError, ValueError) as exc:
        print(f"discord notification failed: {exc}", file=sys.stderr, flush=True)


def run(args: argparse.Namespace) -> int:
    job_dir = args.job_dir.resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path, telemetry_path, log_path = job_dir / "status.json", job_dir / "telemetry.jsonl", job_dir / "train.log"
    started = time.time()
    state = {"run_id": args.run_id, "state": "starting", "started_at": started,
             "heartbeat_at": started, "peak_process_rss_bytes": 0, "peak_vram_used_mb": 0}
    atomic_json(status_path, state)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(args.command, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        state.update(state="running", pid=process.pid)
        atomic_json(status_path, state)
        requested_stop = False
        stop_requested_at: float | None = None
        stop_fallback_sent = False
        try:
            while process.poll() is None:
                now = time.time()
                if (job_dir / "STOP").exists():
                    if not requested_stop:
                        requested_stop = True
                        stop_requested_at = now
                        print(
                            f"cooperative stop requested; allowing up to "
                            f"{args.stop_drain_grace_s:.0f}s for checkpoint flush",
                            flush=True,
                        )
                    elif (
                        not stop_fallback_sent
                        and stop_requested_at is not None
                        and now - stop_requested_at >= args.stop_drain_grace_s
                    ):
                        stop_fallback_sent = True
                        print("cooperative stop grace expired; sending SIGTERM fallback", flush=True)
                        os.killpg(process.pid, signal.SIGTERM)
                try:
                    mem = parse_meminfo(Path("/proc/meminfo").read_text())
                except OSError:
                    mem = {}
                rss, pids = process_tree_rss(process.pid)
                disk = shutil.disk_usage(job_dir)
                sample = {"timestamp": now, "elapsed_s": now - started, **mem,
                          "process_rss_bytes": rss, "process_pids": pids,
                          "disk_free_bytes": disk.free, **gpu_metrics(), **train_log_stats(log_path, now)}
                sample["checkpoint_age_s"] = checkpoint_age(job_dir, now)
                log_age = float(sample.get("train_log_mtime_age_s", 0.0) or 0.0)
                if log_age > TRAIN_LOG_STALL_WARN_S:
                    inflight = sample.get("trainer_step_in_flight")
                    print(
                        f"WARN train.log quiet {log_age:.0f}s "
                        f"gpu_util={sample.get('gpu_util_pct')}% "
                        f"in_flight_step={inflight} last_completed_step={sample.get('step')}",
                        flush=True,
                    )
                with telemetry_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample, sort_keys=True) + "\n")
                    stream.flush()
                state.update(sample, heartbeat_at=now,
                             peak_process_rss_bytes=max(int(state["peak_process_rss_bytes"]), rss),
                             peak_vram_used_mb=max(int(state["peak_vram_used_mb"]), int(sample.get("vram_used_mb", 0))))
                atomic_json(status_path, state)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            requested_stop = True
            os.killpg(process.pid, signal.SIGTERM)
        exit_code = process.wait()
    ended = time.time()
    state.update(state="stopped" if requested_stop else ("completed" if exit_code == 0 else "failed"),
                 exit_code=exit_code, ended_at=ended, heartbeat_at=ended)
    atomic_json(status_path, state)
    maybe_notify_discord(state, resolve_discord_webhook_url(args.discord_webhook_url))
    manifest = artifact_manifest(job_dir)
    atomic_json(job_dir / "artifact_manifest.json", manifest)
    maybe_emergency_drive_release(
        job_dir,
        args.run_id,
        state,
        manifest,
        status_path,
        grace_s=float(args.post_terminal_grace_s),
        enabled=not args.no_emergency_drive_upload,
        max_upload_attempts=int(args.emergency_upload_attempts),
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path("/content"))
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--discord-webhook-url", default=None,
                        help="Discord incoming webhook (default: SYNDIFF_DISCORD_WEBHOOK_URL)")
    parser.add_argument("--post-terminal-grace-s", type=float, default=300.0,
                        help="seconds to wait after terminal state before emergency Drive upload")
    parser.add_argument("--stop-drain-grace-s", type=float, default=STOP_DRAIN_GRACE_S,
                        help="seconds to allow cooperative checkpoint flush before SIGTERM fallback")
    parser.add_argument("--no-emergency-drive-upload", action="store_true",
                        help="disable post-terminal emergency Drive upload + unassign")
    parser.add_argument("--emergency-upload-attempts", type=int, default=3,
                        help="Drive upload retries for emergency backup")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
