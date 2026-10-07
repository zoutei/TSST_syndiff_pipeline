"""Durable, billing-safe Colab submit/provision helpers for ``scripts/colab_job.py``."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

PROVISION_PHASES = (
    "job_dir",
    "oauth",
    "bootstrap",
    "runtime",
    "bundle",
    "preflight",
    "runner",
)
CONTROLLER_PHASES = (
    "validate_local",
    "allocate",
    "readiness",
    "provision_launch",
    "provision_monitor",
    "runner_proof",
    "supervisor",
    "done",
)
READINESS_PROBE_TIMEOUT = 30
READINESS_MAX_ATTEMPTS = 12
READINESS_SHELL_PROBE = "printf 'ready_ok\\n'"
PROVISION_LAUNCH_TIMEOUT = 60
PROVISION_LAUNCH_RETRY_TIMEOUT = 45
PROVISION_POLL_INTERVAL_S = 5.0
PROVISION_MONITOR_TIMEOUT_S = 1800.0
PROVISION_WATCHDOG_DEFAULT_S = 2400.0
RUNNER_PROOF_MAX_WAIT_S = 180.0
RUNNER_HEARTBEAT_MAX_AGE_S = 90.0
SESSION_EXEC_LOCK_TIMEOUT_S = 300.0
CLEANUP_DIAG_TIMEOUT_S = 15.0
CLEANUP_LOG_TIMEOUT_S = 20.0
GDOWN_FAILURE_MARKERS = ("gdown failed", "gdown_failed")
SESSION_LINE_RE = re.compile(
    r"^\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)\s+\|"
    r"\s*Hardware:\s*(?P<hardware>[^|]+)\|"
    r"\s*Shape:\s*(?P<shape>[^|]+)\|"
    r"\s*Variant:\s*(?P<variant>[^|]+)"
    r"(?:\s*\|\s*Status:\s*(?P<status>.*))?$"
)
SECRET_PATTERNS = (
    re.compile(r"https?://[^\s\"']+", re.I),
    re.compile(r"(?i)(token|secret|password|webhook|api[_-]?key)\s*[:=]\s*\S+"),
    re.compile(r"Bearer\s+\S+", re.I),
)
KNOWN_PROCESS_ROLES = (
    "gpu_job_runner",
    "isolated_stage_runner",
    "train_from_bundle",
    "provision_worker",
)
RESUME_CLEAR_SENTINELS = ("runner",)
RESUME_CLEAR_FILES = (
    "status.json",
    "provision_status.json",
    "provision_metadata.json",
    ".provision_worker.lock",
    ".provision_worker.pid",
    ".provision_launch.token",
)
SUBMIT_CRASH_GRACE_S = 45.0
TRAIN_LOG_JAX_MARKER = "jax backend:"


@dataclass
class AssignmentRecord:
    name: str
    endpoint: str
    accelerator: str
    variant: str
    machine_shape: str
    status: str | None = None
    orphan: bool = False
    raw_line: str = ""

    def __post_init__(self) -> None:
        # The server does not return the friendly local name.  ``?`` means an
        # assignment exists without a local owner and must block new billing.
        self.orphan = self.orphan or self.name == "?"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionInventory:
    health: str  # ok | unknown | error
    assignments: list[AssignmentRecord] = field(default_factory=list)
    error: str | None = None
    cli_rc: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "health": self.health,
            "error": self.error,
            "cli_rc": self.cli_rc,
            "assignments": [a.to_dict() for a in self.assignments],
        }

    def names(self) -> list[str]:
        return [a.name for a in self.assignments if a.name != "?"]

    def find_by_name(self, name: str) -> AssignmentRecord | None:
        for record in self.assignments:
            if record.name == name:
                return record
        return None

    def find_by_endpoint(self, endpoint: str) -> AssignmentRecord | None:
        for record in self.assignments:
            if record.endpoint == endpoint:
                return record
        return None


def redact_secrets(text: str) -> str:
    out = text
    for pattern in SECRET_PATTERNS:
        out = pattern.sub("<redacted>", out)
    return out


def redact_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_obj(v) for v in value]
    if isinstance(value, str):
        return redact_secrets(value)
    return value


def parse_session_line(line: str) -> AssignmentRecord | None:
    line = line.strip()
    if not line.startswith("["):
        return None
    match = SESSION_LINE_RE.match(line)
    if not match:
        end = line.find("]")
        name = line[1:end] if end > 1 else "?"
        parts = [p.strip() for p in line.split("|")]
        endpoint = parts[0].split("]", 1)[-1].strip() if parts else ""
        return AssignmentRecord(
            name=name,
            endpoint=endpoint,
            accelerator="?",
            variant="?",
            machine_shape="?",
            orphan=name == "?",
            raw_line=line,
        )
    name = match.group("name")
    return AssignmentRecord(
        name=name,
        endpoint=match.group("endpoint"),
        accelerator=match.group("hardware").strip(),
        variant=match.group("variant").strip(),
        machine_shape=match.group("shape").strip(),
        status=(match.group("status") or "").strip() or None,
        orphan=name == "?",
        raw_line=line,
    )


def lookup_local_session_endpoint(session: str) -> str | None:
    """Read endpoint from default ``colab_cli`` StateStore immediately after ``colab new``."""
    try:
        from colab_cli.common import state as colab_state

        record = colab_state.store.get(session)
        if record is not None and getattr(record, "endpoint", None):
            return str(record.endpoint)
    except Exception:
        pass
    return None


def fetch_session_inventory(
    *,
    colab_binary: Callable[[], str],
    timeout: int = 60,
) -> SessionInventory:
    try:
        result = subprocess.run(
            [colab_binary(), "sessions"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SessionInventory(health="unknown", error="colab sessions timed out")
    except OSError as exc:
        return SessionInventory(health="error", error=str(exc))
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        return SessionInventory(
            health="unknown",
            error=err or f"colab sessions exit {result.returncode}",
            cli_rc=result.returncode,
        )
    assignments: list[AssignmentRecord] = []
    for line in result.stdout.splitlines():
        record = parse_session_line(line)
        if record is not None:
            assignments.append(record)
    return SessionInventory(health="ok", assignments=assignments, cli_rc=result.returncode)


def resolve_session_endpoint(
    session: str,
    *,
    inventory: SessionInventory | None = None,
) -> str | None:
    endpoint = lookup_local_session_endpoint(session)
    if endpoint:
        return endpoint
    if inventory is not None and inventory.health == "ok":
        record = inventory.find_by_name(session)
        if record is not None:
            return record.endpoint
    return None


def list_server_endpoints() -> tuple[list[str], str | None]:
    try:
        from colab_cli.common import state as colab_state

        endpoints = [a.endpoint for a in colab_state.client.list_assignments()]
        return endpoints, None
    except Exception as exc:
        return [], str(exc)


def verify_endpoint_stopped(endpoint: str | None) -> bool:
    if not endpoint:
        return True
    endpoints, _err = list_server_endpoints()
    if endpoints == [] and _err:
        return False
    return endpoint not in endpoints


def unassign_endpoint(endpoint: str) -> tuple[bool, str | None]:
    try:
        from colab_cli.common import state as colab_state

        colab_state.client.unassign(endpoint)
        return True, None
    except Exception as exc:
        return False, str(exc)


def normalize_subprocess_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def merge_provision_metadata(existing: dict | None, **extra: Any) -> dict:
    payload = dict(existing or {})
    payload.update(extra)
    payload["updated_at"] = time.time()
    return payload


def parse_preflight_markers(text: str) -> tuple[str | None, str | None]:
    backend: str | None = None
    gpu_name: str | None = None
    for line in text.splitlines():
        if line.startswith("SYNDIFF_BACKEND="):
            backend = line.split("=", 1)[1].strip() or None
        elif line.startswith("SYNDIFF_DEVICE_KIND="):
            gpu_name = line.split("=", 1)[1].strip() or None
    return backend, gpu_name


def provision_status_accepts(
    status: dict | None,
    launch_token: str,
    *,
    min_updated_at: float | None = None,
    allow_done: bool = True,
) -> bool:
    if not status or status.get("launch_token") != launch_token:
        return False
    updated = status.get("updated_at")
    if min_updated_at is not None:
        if updated is None or float(updated) < float(min_updated_at):
            return False
    phase = status.get("phase")
    ok = status.get("ok")
    if ok is False:
        return True
    if phase == "done" and ok is True:
        return allow_done
    if ok is None or ok is True:
        return True
    return False


def clear_remote_provision_status_script(job_dir: str) -> str:
    q = shlex.quote(job_dir)
    return f"rm -f {q}/provision_status.json {q}/.provision_worker.lock {q}/.provision_worker.pid"


def stop_assignment_verified(
    session: str,
    *,
    endpoint: str | None,
    colab_binary: Callable[[], str],
    verify_session_stopped: Callable[..., bool],
    retries: int = 3,
    force_release: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Stop by local name and only force-unassign with explicit authority."""
    diagnostics: dict[str, Any] = {"session": session, "endpoint": endpoint}
    endpoint_gone = verify_endpoint_stopped(endpoint) if endpoint else True
    name_gone = verify_session_stopped(session, endpoint=endpoint)
    if endpoint_gone and name_gone:
        diagnostics["verified"] = True
        return True, diagnostics
    for _attempt in range(retries):
        endpoint_gone = verify_endpoint_stopped(endpoint) if endpoint else True
        name_gone = verify_session_stopped(session, endpoint=endpoint)
        if endpoint_gone and name_gone:
            diagnostics["verified"] = True
            return True, diagnostics
        if session:
            try:
                result = subprocess.run(
                    [colab_binary(), "stop", "-s", session],
                    text=True,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode != 0:
                    err = redact_secrets((result.stderr or result.stdout or "").strip())
                    if err:
                        diagnostics.setdefault("stop_errors", []).append(err)
            except subprocess.TimeoutExpired:
                diagnostics.setdefault("stop_errors", []).append("colab stop timed out")
        time.sleep(2)
    if force_release and endpoint and not endpoint_gone:
        ok, err = unassign_endpoint(endpoint)
        diagnostics["orphan_unassign"] = ok
        if err:
            diagnostics.setdefault("stop_errors", []).append(redact_secrets(err))
        time.sleep(2)
    endpoint_gone = verify_endpoint_stopped(endpoint) if endpoint else True
    name_gone = verify_session_stopped(session, endpoint=endpoint)
    verified = endpoint_gone and name_gone
    diagnostics["verified"] = verified
    diagnostics["endpoint_gone"] = endpoint_gone
    diagnostics["name_gone"] = name_gone
    if not verified and endpoint and not force_release:
        diagnostics["release_required"] = True
    return verified, diagnostics


class SubmitCleanupLock:
    """Serialize submit cleanup between parent submit and crash/watchdog paths."""

    def __init__(self, run_dir: Path, *, timeout_s: float = 60.0):
        self.path = run_dir / ".submit_cleanup.lock"
        self.timeout_s = timeout_s
        self._fh: int | None = None

    def __enter__(self) -> SubmitCleanupLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.time() + self.timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fh = fd
                return self
            except BlockingIOError:
                if time.time() >= deadline:
                    os.close(fd)
                    raise TimeoutError(f"submit cleanup lock timeout: {self.path}")
                time.sleep(0.05)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            os.close(self._fh)
            self._fh = None


class SessionExecLock:
    """Serialize ``colab exec`` traffic for one session."""

    def __init__(self, session: str, *, timeout_s: float = SESSION_EXEC_LOCK_TIMEOUT_S):
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in session)
        self.path = Path(tempfile.gettempdir()) / f"colab-exec-{safe}.lock"
        self.timeout_s = timeout_s
        self._fh: int | None = None

    def __enter__(self) -> SessionExecLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.time() + self.timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fh = fd
                return self
            except BlockingIOError:
                if time.time() >= deadline:
                    os.close(fd)
                    raise TimeoutError(f"session exec lock timeout for {self.path.name}")
                time.sleep(0.25)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            os.close(self._fh)
            self._fh = None


def config_fingerprint(state: dict) -> str:
    payload = {
        k: state.get(k)
        for k in (
            "mode", "gpu", "cpu", "expect_frames", "steps_per_stage", "first_stage",
            "epsf_lr_scale", "lr_per_stage", "stage2_freeze_wcs_steps", "stamp_chunk",
            "isolate_stages", "reject_mode", "reject_every", "early_stop_patience",
            "early_stop_tol", "huber_delta", "lambda_pixel_lap", "lambda_lap",
            "lambda_smooth_wcs", "lambda_smooth_w", "bundle_sha256", "archive_name",
            "bootstrap_sha256", "post_terminal_grace_s", "emergency_upload_attempts",
            "discord_webhook_configured",
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode(),
    ).hexdigest()


def validate_resume_config(prior_state: dict, new_fingerprint: str) -> None:
    prior = prior_state.get("config_fingerprint")
    if prior and prior != new_fingerprint:
        raise SystemExit(
            f"--resume refused: config fingerprint changed ({prior[:12]} -> {new_fingerprint[:12]}); "
            "submit flags differ from the saved run"
        )


def validate_submit_local(
    args: Any,
    *,
    jobs_root: Path,
    manifest: dict,
    colab_binary: Callable[[], str],
    resolve_submit_bundle_path: Callable[[Any, dict], Path | None],
    bundle_path_usable: Callable[[Path], bool],
    root: Path,
    min_disk_gb: float,
    prior_state: dict | None = None,
    new_fingerprint: str | None = None,
    session_inventory: SessionInventory | None = None,
) -> dict[str, Any]:
    """Validate every local input before ``colab new``. Raises SystemExit on failure."""
    errors: list[str] = []
    run_id = args.run_id or ""
    if not run_id.strip():
        errors.append("run id is required (or omit for auto timestamp)")
    if args.resume:
        controller = jobs_root / run_id / "controller.json"
        if not controller.is_file():
            errors.append(f"--resume: no controller for run id {run_id!r}")
        elif new_fingerprint and prior_state:
            try:
                validate_resume_config(prior_state, new_fingerprint)
            except SystemExit as exc:
                errors.append(str(exc))
    elif (jobs_root / run_id / "controller.json").is_file():
        errors.append(f"run id already exists: {run_id}")
    archive = Path(args.archive)
    if not archive.is_file():
        errors.append(f"--archive not found: {archive}")
    entry = manifest.get("bundles", {}).get(archive.name)
    if not entry:
        errors.append("archive is not published; run publish first")
    if entry and not entry.get("file_id"):
        errors.append("published archive missing Drive file_id")
    try:
        colab_binary()
    except Exception as exc:
        errors.append(str(exc))
    if session_inventory is not None and session_inventory.health == "ok":
        orphans = [record.endpoint for record in session_inventory.assignments if record.orphan]
        if orphans:
            errors.append(
                "server-side Colab orphan assignment(s) still active: " + ", ".join(orphans)
            )
    usage = __import__("shutil").disk_usage(root)
    if usage.free < min_disk_gb * 1024**3:
        errors.append(f"only {usage.free / 1024**3:.1f} GiB local disk free")
    bundle_path = resolve_submit_bundle_path(args, entry or {})
    if getattr(args, "bundle", None):
        bundle = Path(args.bundle).resolve()
        if not bundle_path_usable(bundle):
            errors.append(f"--bundle not usable (need file or dir/fit_bundle.npz): {bundle}")
    bootstrap = getattr(args, "bootstrap_init_params", None)
    if bootstrap:
        bp = Path(bootstrap).resolve()
        if not bp.is_file():
            errors.append(f"--bootstrap-init-params not found: {bp}")
    if errors:
        raise SystemExit("submit validation failed:\n- " + "\n- ".join(errors))
    return {
        "archive_entry": entry,
        "bundle_path": bundle_path,
        "run_id": run_id or None,
    }


def update_controller_phase(
    path: Path,
    state: dict,
    phase: str,
    *,
    atomic_json: Callable[[Path, object], None],
    extra: dict | None = None,
) -> dict:
    state = dict(state)
    state["phase"] = phase
    state["phase_updated_at"] = time.time()
    if extra:
        state.update(extra)
    atomic_json(path, state)
    return state


def readiness_handshake(
    session: str,
    *,
    remote_exec: Callable[..., str],
    inventory: Callable[[], SessionInventory],
    shell_probe: str = READINESS_SHELL_PROBE,
) -> tuple[bool, str]:
    """Treat Session READY as assignment-only; verify kernel exec with a bash probe."""
    last_err = ""
    for attempt in range(READINESS_MAX_ATTEMPTS):
        inv = inventory()
        if inv.health != "ok":
            return False, f"session inventory {inv.health}: {inv.error or 'unknown'}"
        record = inv.find_by_name(session)
        if record is None:
            return False, f"session {session!r} missing from colab sessions"
        if record.status and "BUSY" in record.status.upper():
            if "console" in (record.status or "").lower():
                return False, "colab console is attached; detach before submit"
        try:
            out = remote_exec(session, shell_probe, timeout=READINESS_PROBE_TIMEOUT)
            if "ready_ok" in out:
                return True, ""
            last_err = f"unexpected probe output: {out[:120]!r}"
        except subprocess.CalledProcessError as exc:
            last_err = redact_secrets((exc.stderr or exc.stdout or str(exc)).strip())
        except TimeoutError as exc:
            last_err = str(exc)
        time.sleep(min(5 + attempt, 15))
    return False, last_err or "readiness probe failed"


def build_resume_cleanup_script(run_id: str) -> str:
    q = shlex.quote(run_id)
    clear_sentinels = " ".join(f'rm -f "$RUN_DIR/.phase_{p}.done"' for p in RESUME_CLEAR_SENTINELS)
    clear_files = " ".join(f'rm -f "$RUN_DIR/{name}"' for name in RESUME_CLEAR_FILES)
    return f"""set -euo pipefail
RUN_DIR=/content/jobs/{q}
CANCEL="$RUN_DIR/.provisioning_cancel"
cleanup() {{
  rm -f "$CANCEL" "$RUN_DIR/STOP" 2>/dev/null || true
}}
trap cleanup EXIT
rm -f "$RUN_DIR/STOP" 2>/dev/null || true
{clear_sentinels}
{clear_files}
touch "$CANCEL" 2>/dev/null || true
pkill -f "/content/jobs/{q}/provision_worker.py" 2>/dev/null || true
pkill -f "gpu_job_runner --run-id {q}" 2>/dev/null || true
pkill -f "isolated_stage_runner --from-bundle /content/syndiff/bundle/fit_bundle.npz --out-dir /content/jobs/{q}" 2>/dev/null || true
pkill -f "train_from_bundle --out-dir /content/jobs/{q}" 2>/dev/null || true
sleep 2
rm -f "$CANCEL" "$RUN_DIR/STOP" 2>/dev/null || true
for _ in 1 2 3 4 5; do
  pgrep -f "/content/jobs/{q}/provision_worker.py" >/dev/null 2>&1 && sleep 1 || break
  pgrep -f "gpu_job_runner --run-id {q}" >/dev/null 2>&1 && sleep 1 || break
done
! pgrep -f "gpu_job_runner --run-id {q}" >/dev/null 2>&1
test ! -f "$RUN_DIR/STOP"
"""


def remote_job_dir(run_id: str) -> str:
    return f"/content/jobs/{run_id}"


def remote_paths(run_id: str) -> dict[str, str]:
    base = remote_job_dir(run_id)
    return {
        "job_dir": base,
        "provision_log": f"{base}/provision.log",
        "provision_status": f"{base}/provision_status.json",
        "provision_metadata": f"{base}/provision_metadata.json",
        "runner_log": f"{base}/runner.log",
        "train_log": f"{base}/train.log",
        "status": f"{base}/status.json",
        "stop": f"{base}/STOP",
        "worker_lock": f"{base}/.provision_worker.lock",
        "launch_token": f"{base}/.provision_launch.token",
    }


def local_run_paths(run_dir: Path, run_id: str) -> dict[str, Path]:
    return {
        "provision_status": run_dir / "remote_provision_status.json",
        "provision_metadata": run_dir / "remote_provision_metadata.json",
        "runner_status": run_dir / "remote_runner_status.json",
        "train_log": run_dir / "remote_train.log",
    }


def is_gdown_failure(error: str | None) -> bool:
    if not error:
        return False
    low = error.lower()
    return any(marker in low for marker in GDOWN_FAILURE_MARKERS)


def build_provision_worker_source(spec: dict[str, Any]) -> str:
    """Return remote ``provision_worker.py`` source (idempotent phases)."""
    spec_json = json.dumps(spec)
    return f'''#!/usr/bin/env python3
import fcntl, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

SPEC = json.loads({spec_json!r})
RUN_ID = SPEC["run_id"]
JOB = Path(SPEC["job_dir"])
PHASES = {json.dumps(list(PROVISION_PHASES))}
LOG = JOB / "provision.log"
STATUS = JOB / "provision_status.json"
METADATA = JOB / "provision_metadata.json"
LOCK = JOB / ".provision_worker.lock"
LAUNCH_TOKEN = SPEC.get("launch_token", "")


def log(msg):
    JOB.mkdir(parents=True, exist_ok=True)
    line = f"[{{time.strftime('%Y-%m-%d %H:%M:%S')}}] {{msg}}\\n"
    with LOG.open("a", encoding="utf-8") as s:
        s.write(line)


def write_metadata(**extra):
    existing = {{}}
    if METADATA.is_file():
        try:
            existing = json.loads(METADATA.read_text())
        except Exception:
            existing = {{}}
    payload = {{
        "run_id": RUN_ID,
        "launch_token": LAUNCH_TOKEN,
        "config_fingerprint": SPEC.get("config_fingerprint"),
        "bundle_sha256": SPEC.get("bundle_sha256"),
        "expected_gpu": SPEC.get("expected_gpu"),
        "use_cpu": bool(SPEC.get("use_cpu")),
        "updated_at": time.time(),
    }}
    payload.update(existing)
    payload.update(extra)
    payload["updated_at"] = time.time()
    tmp = METADATA.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, METADATA)


def set_status(phase, *, ok=True, error=None, extra=None):
    payload = {{
        "phase": phase,
        "ok": ok,
        "error": error,
        "launch_token": LAUNCH_TOKEN,
        "updated_at": time.time(),
    }}
    if extra:
        payload.update(extra)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, STATUS)


def sentinel(phase):
    return JOB / f".phase_{{phase}}.done"


def done(phase):
    return sentinel(phase).is_file()


def mark(phase):
    sentinel(phase).write_text(str(time.time()))


def unmark(phase):
    sentinel(phase).unlink(missing_ok=True)


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, check=False, **kw)


def acquire_worker_lock():
    JOB.mkdir(parents=True, exist_ok=True)
    token_path = JOB / ".provision_launch.token"
    if token_path.is_file() and token_path.read_text().strip() != LAUNCH_TOKEN:
        raise RuntimeError("launch token mismatch")
    token_path.write_text(LAUNCH_TOKEN)
    lock_fd = os.open(str(LOCK), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        return False, None
    os.write(lock_fd, str(os.getpid()).encode())
    os.ftruncate(lock_fd, len(str(os.getpid())))
    (JOB / ".provision_worker.pid").write_text(str(os.getpid()))
    return True, lock_fd


def release_worker_lock(lock_fd):
    if lock_fd is not None:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    LOCK.unlink(missing_ok=True)


def run_logged_phase(name, cmd, *, cwd=None):
    proc = run(cmd, cwd=cwd)
    if proc.stdout:
        log(f"{{name}} stdout: {{proc.stdout[-2000:]}}")
    if proc.stderr:
        log(f"{{name}} stderr: {{proc.stderr[-2000:]}}")
    if proc.returncode != 0:
        raise RuntimeError(f"{{name}} failed rc={{proc.returncode}}")
    return proc


def phase_job_dir():
    if done("job_dir"):
        return
    JOB.mkdir(parents=True, exist_ok=True)
    Path("/content/syndiff").mkdir(parents=True, exist_ok=True)
    mark("job_dir")
    log("phase job_dir complete")


def phase_oauth():
    if done("oauth"):
        return
    oauth_dir = JOB / ".gdrive_oauth"
    log("oauth dir present" if oauth_dir.is_dir() else "oauth dir missing (best-effort)")
    mark("oauth")


def phase_bootstrap():
    if done("bootstrap") or not SPEC.get("bootstrap_remote"):
        if not SPEC.get("bootstrap_remote"):
            mark("bootstrap")
        return
    remote = Path(SPEC["bootstrap_remote"])
    if not remote.is_file():
        raise RuntimeError(f"bootstrap missing: {{remote}}")
    mark("bootstrap")
    log("bootstrap ready")


def phase_runtime():
    if done("runtime"):
        return
    packages = list(SPEC.get("pip_packages") or [])
    run_logged_phase("pip", [sys.executable, "-m", "pip", "install", "-q", *packages])
    mark("runtime")
    log("runtime install complete")


def phase_bundle():
    if done("bundle"):
        return
    bundle_zip = Path("/content/bundle.zip")
    if not bundle_zip.is_file():
        rc = run(["gdown", SPEC["bundle_file_id"], "-O", str(bundle_zip)]).returncode
        if rc != 0:
            set_status("bundle", ok=False, error="gdown_failed", extra={{"needs_bundle_upload": True}})
            raise RuntimeError("gdown failed")
    digest = hashlib.sha256(bundle_zip.read_bytes()).hexdigest()
    if digest != SPEC["bundle_sha256"]:
        raise RuntimeError("bundle sha256 mismatch")
    with zipfile.ZipFile(bundle_zip) as zf:
        zf.extractall("/content/syndiff")
    mark("bundle")
    log("bundle unpacked")


def phase_preflight():
    if done("preflight"):
        return
    preflight_script = JOB / "_preflight_probe.py"
    preflight_script.write_text(SPEC["preflight_py"], encoding="utf-8")
    result = run([sys.executable, str(preflight_script)], cwd="/content/syndiff")
    combined = ((result.stdout or "") + (result.stderr or "")).strip()
    (JOB / "_preflight_probe.out").write_text(combined, encoding="utf-8")
    if result.returncode != 0:
        log(f"preflight_jax stderr: {{(result.stderr or '')[-2000:]}}")
        raise RuntimeError(f"preflight_jax failed rc={{result.returncode}}")
    if result.stdout:
        log(f"preflight_jax stdout: {{result.stdout[-2000:]}}")
    run_logged_phase(
        "gpu_preflight",
        [
            sys.executable, "-m", "syndiff_pipeline.forward_model.gpu_preflight",
            "--bundle", "/content/syndiff/bundle/fit_bundle.npz",
            "--expect-frames", str(SPEC["expect_frames"]),
        ],
        cwd="/content/syndiff",
    )
    probe_out = (JOB / "_preflight_probe.out").read_text(errors="replace") if (JOB / "_preflight_probe.out").is_file() else ""
    backend = ""
    gpu_name = ""
    for line in probe_out.splitlines():
        if line.startswith("SYNDIFF_BACKEND="):
            backend = line.split("=", 1)[1].strip()
        elif line.startswith("SYNDIFF_DEVICE_KIND="):
            gpu_name = line.split("=", 1)[1].strip()
    if not backend:
        raise RuntimeError("preflight missing SYNDIFF_BACKEND marker")
    if not SPEC.get("use_cpu") and not gpu_name:
        raise RuntimeError("preflight missing SYNDIFF_DEVICE_KIND marker")
    write_metadata(
        preflight_backend=backend,
        preflight_gpu_name=gpu_name,
        preflight_completed_at=time.time(),
    )
    preflight_script.unlink(missing_ok=True)
    mark("preflight")
    log("preflight complete")


def phase_runner():
    if done("runner"):
        return
    if (JOB / "STOP").exists():
        raise RuntimeError("STOP present before runner launch")
    if (JOB / "status.json").is_file():
        try:
            st = json.loads((JOB / "status.json").read_text())
            if st.get("state") in ("starting", "running"):
                hb = st.get("heartbeat_at") or st.get("timestamp")
                if hb and (time.time() - float(hb)) <= 90:
                    mark("runner")
                    log("runner already active")
                    return
        except Exception:
            pass
    runner_log = JOB / "runner.log"
    runner_env = {{**os.environ, "PYTHONUNBUFFERED": "1"}}
    secret_path = SPEC.get("discord_secret_path")
    if secret_path:
        secret_file = Path(secret_path)
        if secret_file.is_file():
            runner_env["SYNDIFF_DISCORD_WEBHOOK_URL"] = secret_file.read_text(encoding="utf-8").strip()
    with runner_log.open("ab", buffering=0) as logf:
        proc = subprocess.Popen(
            SPEC["runner_cmd"],
            cwd="/content/syndiff",
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=runner_env,
        )
    time.sleep(2)
    if proc.poll() is not None:
        tail = runner_log.read_text(errors="replace")[-4000:] if runner_log.is_file() else ""
        raise RuntimeError(f"runner exited early rc={{proc.returncode}}: {{tail}}")
    mark("runner")
    write_metadata(runner_pid=proc.pid, runner_launched_at=time.time())
    log(f"runner launched pid={{proc.pid}}")


HANDLERS = {{
    "job_dir": phase_job_dir,
    "oauth": phase_oauth,
    "bootstrap": phase_bootstrap,
    "runtime": phase_runtime,
    "bundle": phase_bundle,
    "preflight": phase_preflight,
    "runner": phase_runner,
}}

def main():
    started, lock_fd = acquire_worker_lock()
    if not started:
        log("duplicate worker; exiting without status update")
        return 0
    try:
        JOB.mkdir(parents=True, exist_ok=True)
        write_metadata(worker_started_at=time.time())
        for phase in PHASES:
            if done(phase):
                set_status(phase, ok=True, extra={{"skipped": True}})
                continue
            set_status(phase, ok=None)
            try:
                HANDLERS[phase]()
                set_status(phase, ok=True)
            except Exception as exc:
                set_status(phase, ok=False, error=str(exc))
                log(f"phase {{phase}} failed: {{exc}}")
                return 1
        set_status("done", ok=True)
        return 0
    finally:
        release_worker_lock(lock_fd)

if __name__ == "__main__":
    raise SystemExit(main())
'''


def monitor_provision(
    session: str,
    run_id: str,
    *,
    launch_token: str,
    colab_download: Callable[..., bool],
    paths: dict[str, str],
    local_paths: dict[str, Path],
    timeout_s: float = PROVISION_MONITOR_TIMEOUT_S,
    poll_s: float = PROVISION_POLL_INTERVAL_S,
    min_updated_at: float | None = None,
) -> tuple[dict | None, str | None]:
    """Poll remote ``provision_status.json`` via download API."""
    local_status = local_paths["provision_status"]
    deadline = time.time() + timeout_s
    last: dict | None = None
    while time.time() < deadline:
        if colab_download(session, paths["provision_status"], local_status, timeout=60):
            try:
                candidate = json.loads(local_status.read_text())
            except json.JSONDecodeError:
                candidate = None
            else:
                if provision_status_accepts(
                    candidate,
                    launch_token,
                    min_updated_at=min_updated_at,
                ):
                    last = candidate
                    phase = last.get("phase")
                    ok = last.get("ok")
                    if phase == "done" and ok is True:
                        return last, None
                    if ok is False:
                        return last, last.get("error") or f"provision failed at {phase}"
        time.sleep(poll_s)
    return last, "provision monitor timed out"


def load_provision_metadata(
    session: str,
    run_id: str,
    *,
    colab_download: Callable[..., bool],
    paths: dict[str, str],
    local_paths: dict[str, Path],
) -> dict:
    local_meta = local_paths["provision_metadata"]
    if colab_download(session, paths["provision_metadata"], local_meta, timeout=60):
        return load_json_file(local_meta, {})
    return {}


def load_json_file(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def validate_runner_proof(
    session: str,
    run_id: str,
    state: dict,
    *,
    colab_download: Callable[..., bool],
    remote_exec: Callable[..., str],
    paths: dict[str, str],
    local_paths: dict[str, Path],
    max_wait_s: float = RUNNER_PROOF_MAX_WAIT_S,
) -> tuple[bool, dict[str, Any]]:
    """Require fresh heartbeat, live runner, GPU/backend, bundle hash, and config fingerprint."""
    proof: dict[str, Any] = {"checks": {}}
    local_status = local_paths["runner_status"]
    local_train = local_paths.get("train_log") or (local_paths["provision_status"].parent / "remote_train.log")
    deadline = time.time() + max_wait_s
    expected_gpu = str(state.get("gpu", "")).upper()
    expected_fp = state.get("config_fingerprint") or config_fingerprint(state)
    expected_bundle = str(state.get("bundle_sha256", ""))
    use_cpu = bool(state.get("cpu"))
    metadata = load_provision_metadata(
        session, run_id, colab_download=colab_download, paths=paths, local_paths=local_paths,
    )
    while time.time() < deadline:
        remote: dict = {}
        if colab_download(session, paths["status"], local_status, timeout=60):
            try:
                remote = json.loads(local_status.read_text())
            except json.JSONDecodeError:
                remote = {}
        if not metadata:
            metadata = load_provision_metadata(
                session, run_id, colab_download=colab_download, paths=paths, local_paths=local_paths,
            )
        hb = remote.get("heartbeat_at") or remote.get("timestamp")
        remote_state = remote.get("state")
        hb_ok = hb is not None and (time.time() - float(hb)) <= RUNNER_HEARTBEAT_MAX_AGE_S
        state_ok = remote_state in ("starting", "running")
        fp_ok = metadata.get("config_fingerprint") == expected_fp
        bundle_ok = metadata.get("bundle_sha256") == expected_bundle
        backend_ok = metadata.get("preflight_backend") == ("cpu" if use_cpu else "gpu")
        gpu_name = str(remote.get("gpu_name") or metadata.get("preflight_gpu_name") or "")
        gpu_ok = True
        if not use_cpu:
            gpu_ok = bool(gpu_name.strip()) and expected_gpu.lower() in gpu_name.lower()
        train_log_ok = False
        if colab_download(session, paths["train_log"], local_train, timeout=60):
            if local_train.is_file():
                tail = local_train.read_text(errors="replace")[-16384:].lower()
                train_log_ok = TRAIN_LOG_JAX_MARKER in tail
        proof["checks"].update(
            heartbeat_age_s=(time.time() - float(hb)) if hb else None,
            remote_state=remote_state,
            heartbeat_ok=hb_ok,
            state_ok=state_ok,
            config_fingerprint_ok=fp_ok,
            bundle_sha256_ok=bundle_ok,
            backend_ok=backend_ok,
            gpu_ok=gpu_ok,
            gpu_name=gpu_name,
            train_log_ok=train_log_ok,
        )
        if remote_state in ("failed", "stopped", "completed"):
            proof["failure"] = f"remote state {remote_state!r}"
            return False, proof
        proc_ok = False
        try:
            ps = remote_exec(
                session,
                f"ps -eo pid=,args= | grep -F {shlex.quote(run_id)} | grep -E "
                "'gpu_job_runner|isolated_stage_runner|train_from_bundle' | grep -v grep || true",
                timeout=30,
            )
            proc_ok = bool(ps.strip())
            proof["process_count"] = len([ln for ln in ps.strip().splitlines() if ln.strip()])
        except Exception as exc:
            proof["process_error"] = str(exc)
        if (
            hb_ok and state_ok and proc_ok and fp_ok and bundle_ok
            and backend_ok and gpu_ok and train_log_ok
        ):
            proof["ok"] = True
            proof["remote"] = redact_obj(remote)
            proof["metadata"] = redact_obj(metadata)
            return True, proof
        time.sleep(5)
    proof["failure"] = "runner proof timed out"
    return False, proof


REMOTE_DIAGNOSE_SCRIPT = r'''
import json, os, re
from pathlib import Path

RUN_ID = os.environ["RUN_ID"]
JOB = Path(f"/content/jobs/{RUN_ID}")
SYNDIFF = Path("/content/syndiff")
SECRET_RE = [
    re.compile(r"https?://[^\s\"']+", re.I),
    re.compile(r"(?i)(token|secret|password|webhook|api[_-]?key)\s*[:=]\s*\S+"),
]

def redact(text):
    out = str(text)
    for pat in SECRET_RE:
        out = pat.sub("<redacted>", out)
    return out

def tail(path, n=40):
    p = Path(path)
    if not p.is_file():
        return ""
    lines = p.read_text(errors="replace").splitlines()
    return redact("\n".join(lines[-n:]))

def dir_entries(root):
    p = Path(root)
    if not p.is_dir():
        return []
    out = []
    for child in sorted(p.iterdir()):
        try:
            st = child.stat()
            out.append({"name": child.name, "size": st.st_size, "mtime": st.st_mtime})
        except OSError:
            out.append({"name": child.name, "error": "stat_failed"})
    return out

ROLE_MAP = {
    "gpu_job_runner": "runner",
    "isolated_stage_runner": "trainer",
    "train_from_bundle": "trainer",
    "provision_worker.py": "provision_worker",
}

OUT = {}
OUT["job_dir_entries"] = dir_entries(JOB)
OUT["syndiff_ready"] = (SYNDIFF / "bundle" / "fit_bundle.npz").is_file()
OUT["bundle_zip"] = Path("/content/bundle.zip").is_file()
OUT["stop_present"] = (JOB / "STOP").is_file()
status_path = JOB / "status.json"
if status_path.is_file():
    try:
        st = json.loads(status_path.read_text())
        OUT["status"] = {k: st.get(k) for k in ("state", "stage", "step", "gpu_name", "heartbeat_at")}
    except Exception as exc:
        OUT["status_error"] = redact(str(exc))
else:
    OUT["status"] = None
prov = JOB / "provision_status.json"
if prov.is_file():
    try:
        ps = json.loads(prov.read_text())
        OUT["provision_status"] = {k: ps.get(k) for k in ("phase", "ok", "error", "updated_at")}
    except Exception as exc:
        OUT["provision_status_error"] = redact(str(exc))
meta = JOB / "provision_metadata.json"
if meta.is_file():
    try:
        md = json.loads(meta.read_text())
        OUT["provision_metadata"] = {
            k: md.get(k)
            for k in ("config_fingerprint", "bundle_sha256", "preflight_backend", "preflight_gpu_name")
        }
    except Exception as exc:
        OUT["provision_metadata_error"] = redact(str(exc))

procs = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        continue
    if not cmd or RUN_ID not in cmd:
        continue
    role = None
    for needle, label in ROLE_MAP.items():
        if needle in cmd:
            role = label
            break
    if role is None:
        continue
    procs.append({"pid": int(entry.name), "role": role})
OUT["processes"] = procs
OUT["log_tails"] = {
    "provision.log": tail(JOB / "provision.log"),
    "runner.log": tail(JOB / "runner.log"),
    "train.log": tail(JOB / "train.log"),
}
print(json.dumps(OUT, sort_keys=True))
'''


def collect_remote_diagnosis(
    session: str,
    run_id: str,
    *,
    remote_exec: Callable[..., str],
    timeout: int = CLEANUP_DIAG_TIMEOUT_S,
) -> dict[str, Any]:
    script = (
        f"export RUN_ID={shlex.quote(run_id)}\n"
        "python3 - <<'PY'\n"
        + REMOTE_DIAGNOSE_SCRIPT
        + "\nPY\n"
    )
    try:
        raw = remote_exec(session, script, timeout=timeout)
        data = json.loads(raw.strip().splitlines()[-1])
        return redact_obj(data)
    except Exception as exc:
        return {"error": redact_secrets(str(exc))}


def persist_failure_diagnostics(
    directory: Path,
    state: dict,
    diagnosis: dict,
    *,
    atomic_json: Callable[[Path, object], None],
) -> dict:
    state = dict(state)
    state["submit_diagnostics"] = diagnosis
    state["submit_diagnostics_at"] = time.time()
    diag_path = directory / "submit_diagnostics.json"
    atomic_json(diag_path, diagnosis)
    atomic_json(directory / "controller.json", state)
    return state


def capture_remote_logs(
    session: str,
    run_id: str,
    directory: Path,
    *,
    colab_download: Callable[..., bool],
    timeout: int = CLEANUP_LOG_TIMEOUT_S,
) -> list[str]:
    paths = remote_paths(run_id)
    saved: list[str] = []
    for key, remote in (
        ("provision.log", paths["provision_log"]),
        ("runner.log", paths["runner_log"]),
        ("status.json", paths["status"]),
    ):
        local = directory / "artifacts" / f"remote_{key.replace('/', '_')}"
        if colab_download(session, remote, local, timeout=timeout):
            saved.append(key)
    return saved


def build_worker_launch_script(
    worker_remote: str,
    provision_log: str,
    launch_token: str,
    job_dir: str,
) -> str:
    token_path = f"{job_dir}/.provision_launch.token"
    return (
        f"echo {shlex.quote(launch_token)} > {shlex.quote(token_path)}; "
        f"nohup python3 {shlex.quote(worker_remote)} >>{shlex.quote(provision_log)} 2>&1 & echo launched"
    )


def _maybe_gdown_fallback(
    session: str,
    archive: Path,
    prov_status: dict | None,
    *,
    colab_binary: Callable[[], str],
    run_cmd: Callable[..., subprocess.CompletedProcess],
    relaunch: Callable[[], None],
    clear_remote_status: Callable[[], None] | None = None,
    launch_token: str | None = None,
) -> bool:
    err = (prov_status or {}).get("error") if prov_status else None
    token_ok = (
        launch_token is None
        or (prov_status or {}).get("launch_token") == launch_token
    )
    if not token_ok:
        return False
    if not is_gdown_failure(err) and not (prov_status or {}).get("needs_bundle_upload"):
        return False
    if clear_remote_status:
        clear_remote_status()
    run_cmd([colab_binary(), "upload", "-s", session, str(archive.resolve()), "/content/bundle.zip"])
    relaunch()
    return True


def render_diagnose_human(diagnosis: dict, state: dict) -> str:
    lines = [
        f"run_id: {state.get('run_id', '?')}",
        f"controller state: {state.get('state', '?')} phase: {state.get('phase', '?')}",
        f"session: {state.get('session', '?')} endpoint: {state.get('endpoint', '?')}",
    ]
    if diagnosis.get("error"):
        lines.append(f"diagnose error: {diagnosis['error']}")
    if "stop_present" in diagnosis:
        lines.append(f"STOP present: {diagnosis['stop_present']}")
    if diagnosis.get("status"):
        st = diagnosis["status"]
        lines.append(
            f"remote status: {st.get('state')} stage={st.get('stage')} step={st.get('step')} "
            f"gpu={st.get('gpu_name')}"
        )
    if diagnosis.get("provision_status"):
        ps = diagnosis["provision_status"]
        lines.append(f"provision: phase={ps.get('phase')} ok={ps.get('ok')} err={ps.get('error')}")
    if diagnosis.get("provision_metadata"):
        md = diagnosis["provision_metadata"]
        lines.append(
            f"metadata: fp={str(md.get('config_fingerprint', ''))[:12]} "
            f"bundle={str(md.get('bundle_sha256', ''))[:12]} backend={md.get('preflight_backend')}"
        )
    for proc in diagnosis.get("processes", [])[:8]:
        lines.append(f"proc {proc.get('pid')}: role={proc.get('role', '?')}")
    tails = diagnosis.get("log_tails") or {}
    for name, text in tails.items():
        if text:
            lines.append(f"--- {name} (tail) ---")
            lines.append(text[-2000:])
    return "\n".join(lines)
