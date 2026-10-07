#!/usr/bin/env python3
"""Submit, monitor, synchronize, and recover forward-ePSF Colab GPU jobs."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import termios
import time
import tty
import zipfile
import pwd
from pathlib import Path

from syndiff_pipeline.forward_model.colab_submit import (
    SessionExecLock,
    SessionInventory,
    SubmitCleanupLock,
    READINESS_SHELL_PROBE,
    build_provision_worker_source,
    build_resume_cleanup_script,
    build_worker_launch_script,
    capture_remote_logs,
    clear_remote_provision_status_script,
    collect_remote_diagnosis,
    config_fingerprint,
    fetch_session_inventory,
    is_gdown_failure,
    local_run_paths,
    lookup_local_session_endpoint,
    monitor_provision,
    normalize_subprocess_output,
    persist_failure_diagnostics,
    provision_status_accepts,
    readiness_handshake,
    redact_obj,
    redact_secrets,
    remote_paths,
    render_diagnose_human,
    resolve_session_endpoint,
    stop_assignment_verified,
    update_controller_phase,
    validate_runner_proof,
    validate_submit_local,
    verify_endpoint_stopped,
    _maybe_gdown_fallback,
    PROVISION_LAUNCH_TIMEOUT,
    PROVISION_LAUNCH_RETRY_TIMEOUT,
    PROVISION_WATCHDOG_DEFAULT_S,
    SUBMIT_CRASH_GRACE_S,
    CLEANUP_DIAG_TIMEOUT_S,
)

ROOT = Path(__file__).resolve().parents[1]
# Experiment area: Colab outputs, default bundles, helper scripts, plotting diagnostics.
PROJECT = ROOT / "dev/forward_epsf_wcs"
# GPU-side training code shipped to Colab and run there: the merged fitter package
# (until 2026-10-07 the dev/forward_epsf_wcs copy was shipped instead).
GPU_PACKAGE = ROOT / "syndiff_pipeline/forward_model"
DEFAULT_DEPLOYMENT_DIR = ROOT / "config"
# Durable Colab job state + bundle exports live on shared astro storage (not the
# git checkout).  Override with SYNDIFF_COLAB_STORAGE for tests / other hosts.
COLAB_STORAGE = Path(os.environ.get(
    "SYNDIFF_COLAB_STORAGE",
    "/astro/armin/koji/syndiff/output/colab",
))
JOBS = COLAB_STORAGE / "jobs"
COLAB_OUTPUT = PROJECT / "output/colab"
MANIFEST = COLAB_OUTPUT / "gdrive_manifest.json"
DEFAULT_BUNDLE = PROJECT / "output/bundles/fullccd_mag711_irreg_590_tiered"
DEFAULT_ZIP = COLAB_OUTPUT / "colab_fullccd_mag711_irreg_590_tiered.zip"
ORBIT1_BUNDLE = PROJECT / "output/bundles/orbit1_midhalf_mag813_center1k"
ORBIT1_ZIP = COLAB_OUTPUT / "colab_orbit1_midhalf_mag811_center1k.zip"
RUNTIME = {"numpy": "2.4.6", "jax": "0.9.2", "jaxlib": "0.9.2", "optax": "0.2.8"}
TERMINAL = {"completed", "failed", "stopped"}
WATCH_LIVE_PLOT_REL = "plots/watch_live.png"
WATCH_LIVE_LOG_PLOT_REL = "plots/watch_live_log.png"
WATCH_LIVE_WCS_PLOT_REL = "plots/watch_live_wcs.png"
WATCH_LIVE_EPSF_PLOT_REL = "plots/watch_live_epsf.png"
WATCH_LIVE_CHROMA_PLOT_REL = "plots/watch_live_chroma.png"
WATCH_LIVE_EPSF_STAGE2_PLOT_REL = "plots/watch_live_epsf_stage2.png"
WATCH_LIVE_AC_W0_PLOT_REL = "plots/watch_live_ac_w0.png"
WATCH_LIVE_LC_GRID_PLOT_REL = "plots/watch_live_lc_grid.png"
WATCH_STATUS_STALE_S = 120.0
WATCH_ACTIVITY_FRESH_S = 90.0
# Supervisor: no train.log growth while remote still says "running".
SUPERVISOR_TRAIN_LOG_STALL_S = 600.0
# Consecutive failed remote status fetches before treating the VM as lost (~2 min).
SUPERVISOR_REMOTE_FAIL_LIMIT = 8
# After requesting STOP, wait this long before forcing abnormal shutdown.
SUPERVISOR_HANG_STOP_GRACE_S = 120.0


def _ensure_repo_on_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def maybe_refresh_watch_plots(
    run_id: str,
    state: dict,
    remote: dict | None = None,
) -> bool:
    """Regenerate ``watch_live.png`` (linear y) and ``watch_live_log.png`` (log y)."""
    _ensure_repo_on_path()
    from dev.forward_epsf_wcs.diagnostics.watch_plot import (
        render_watch_plot,
        watch_live_log_plot_path,
        watch_live_plot_path,
    )

    directory = JOBS / run_id
    history = directory / "artifacts" / "history.jsonl"
    if not history.is_file():
        return False
    remote = remote or load_json(directory / "remote_status.json", {})
    train_log = directory / "artifacts" / "train.log"
    train_log_path = train_log if train_log.is_file() else None
    updated = False

    for log_y, path_fn, at_key, path_key, err_key in (
        (False, watch_live_plot_path, "watch_plot_updated_at", "watch_plot_path", "watch_plot_error"),
        (True, watch_live_log_plot_path, "watch_log_plot_updated_at", "watch_log_plot_path", "watch_log_plot_error"),
    ):
        out_path = path_fn(directory / "artifacts")
        try:
            written = render_watch_plot(
                history,
                out_path=out_path,
                run_id=run_id,
                controller=state,
                remote=remote,
                train_log_path=train_log_path,
                log_y=log_y,
            )
        except Exception as exc:
            state[err_key] = str(exc)
            continue
        if written is None:
            continue
        state[at_key] = time.time()
        state[path_key] = str(written)
        state.pop(err_key, None)
        updated = True
    return updated


def maybe_refresh_watch_plot(
    run_id: str,
    state: dict,
    remote: dict | None = None,
) -> bool:
    """Backward-compatible wrapper: refresh both live plot PNGs."""
    return maybe_refresh_watch_plots(run_id, state, remote)


def maybe_refresh_params_plots(
    run_id: str,
    state: dict,
    remote: dict | None = None,
) -> bool:
    """Regenerate ``watch_live_wcs.png`` and/or ``watch_live_epsf.png`` from params."""
    _ensure_repo_on_path()
    from dev.forward_epsf_wcs.diagnostics.watch_params_plot import (
        maybe_refresh_params_plots as refresh_params_plots,
    )

    directory = JOBS / run_id
    bundle_path = ensure_job_bundle_path(run_id, state)
    if bundle_path is None:
        return False
    remote = remote or load_json(directory / "remote_status.json", {})
    return refresh_params_plots(
        directory,
        run_id=run_id,
        bundle_path=bundle_path,
        remote=remote,
        state=state,
    )


def maybe_refresh_lc_plots(
    run_id: str,
    state: dict,
    remote: dict | None = None,
) -> bool:
    """Regenerate ``watch_live_ac_w0.png`` and ``watch_live_lc_grid.png``."""
    _ensure_repo_on_path()
    from dev.forward_epsf_wcs.diagnostics.watch_lc_plot import (
        maybe_refresh_lc_plots as refresh_lc_plots,
    )

    directory = JOBS / run_id
    bundle_path = ensure_job_bundle_path(run_id, state)
    if bundle_path is None:
        return False
    remote = remote or load_json(directory / "remote_status.json", {})
    return refresh_lc_plots(
        directory,
        run_id=run_id,
        bundle_path=bundle_path,
        remote=remote,
        state=state,
    )


def _live_params_plot_unit(run_id: str) -> str:
    return "syndiff-colab-live-params-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)[:180]


def launch_live_params_plot_worker(run_id: str, state: dict) -> bool:
    """Start one capped WCS/ePSF refresh outside the supervisor process."""
    unit = _live_params_plot_unit(run_id)
    if shutil.which("systemctl"):
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit], check=False,
        )
        if active.returncode == 0:
            return False
    systemd = shutil.which("systemd-run")
    if not systemd:
        state["live_params_plot_error"] = "systemd-run unavailable; worker not launched"
        return False
    command = [
        systemd, "--user", "--unit", unit, "--collect",
        "--working-directory", str(ROOT),
        "--setenv=PYTHONUNBUFFERED=1",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--property=MemoryHigh=12G",
        "--property=MemoryMax=16G",
        sys.executable, __file__, "_live_params_plot", run_id,
    ]
    started = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if started.returncode:
        state["live_params_plot_error"] = started.stderr.strip() or "systemd-run failed"
        return False
    state.update(live_params_plot_unit=unit, live_params_plot_started_at=time.time())
    state.pop("live_params_plot_error", None)
    return True


def live_params_plot_worker(args) -> int:
    """Detached, memory-capped WCS/ePSF plot refresh for a synced checkpoint."""
    directory, state = job_state(args.run_id)
    remote = load_json(directory / "remote_status.json", {})
    try:
        maybe_refresh_params_plots(args.run_id, state, remote)
    except Exception as exc:
        state["live_params_plot_error"] = str(exc)
        atomic_json(directory / "controller.json", state)
        raise
    state["live_params_plot_finished_at"] = time.time()
    state.pop("live_params_plot_error", None)
    atomic_json(directory / "controller.json", state)
    return 0


def _params_plots_need_refresh(changed_paths: list[str]) -> bool:
    if "params_latest.npz" in changed_paths:
        return True
    return any(p.startswith("checkpoints/") and p.endswith(".npz") for p in changed_paths)


def _lc_plots_need_refresh(changed_paths: list[str]) -> bool:
    if _params_plots_need_refresh(changed_paths):
        return True
    return "stamp_active_latest.npz" in changed_paths


def refresh_local_plots(
    run_id: str,
    state: dict,
    remote: dict | None = None,
) -> bool:
    """Regenerate all watch PNGs from local synced artifacts (no remote fetch)."""
    directory = JOBS / run_id
    remote = remote or load_json(directory / "remote_status.json", {})
    updated = maybe_refresh_watch_plots(run_id, state, remote)
    if (directory / "artifacts" / "params_latest.npz").is_file() or any(
        (directory / "artifacts" / "checkpoints").glob("params_s*_step*.npz")
    ):
        updated = maybe_refresh_params_plots(run_id, state, remote) or updated
        remote_stage = int(remote.get("stage", 0) or 0)
        if remote_stage >= 2:
            updated = maybe_refresh_lc_plots(run_id, state, remote) or updated
    return updated


def _local_artifact_ages(directory: Path, now: float) -> dict[str, float | None]:
    artifacts = directory / "artifacts"
    ages: dict[str, float | None] = {}
    for name in ("train.log", "history.jsonl"):
        path = artifacts / name
        ages[name] = (now - path.stat().st_mtime) if path.is_file() else None
    return ages


def enrich_remote_from_local(directory: Path, remote: dict | None, now: float) -> dict:
    """Fill missing/stale remote progress from the locally synced ``train.log``.

    A fresh remote ``status.json`` is authoritative.  The local log can lag by
    minutes while checkpoint/artifact synchronization is in progress, so it
    must never overwrite newer remote stage/step/loss telemetry.
    """
    remote = dict(remote or {})
    ages = _local_artifact_ages(directory, now)
    remote["_local_train_log_age_s"] = ages.get("train.log")
    remote["_local_history_age_s"] = ages.get("history.jsonl")
    train_log = directory / "artifacts" / "train.log"
    if train_log.is_file():
        _ensure_repo_on_path()
        from syndiff_pipeline.forward_model.training_history import parse_train_log_metrics

        rows = parse_train_log_metrics(train_log.read_bytes()[-131072:].decode(errors="replace"))
        remote_at = remote.get("timestamp") or remote.get("heartbeat_at")
        remote_is_fresh = (
            remote_at is not None
            and now - float(remote_at) < WATCH_STATUS_STALE_S
        )
        if rows and not remote_is_fresh:
            last = rows[-1]
            # Remote status is stale/unavailable, so the local log is the
            # appropriate fallback.  A fresh remote status never reaches this
            # branch, even if the local log is newer on the filesystem.
            remote["stage"] = int(last["stage"])
            remote["step"] = int(last["step"])
            remote["loss"] = float(last["loss"])
            remote["_train_log_stage"] = int(last["stage"])
            remote["_train_log_step"] = int(last["step"])
    return remote


def remote_status_stale(remote: dict, directory: Path, now: float) -> bool:
    """True when remote ``status.json`` is old and local training logs are also quiet."""
    status_at = remote.get("timestamp") or remote.get("heartbeat_at")
    if status_at is None:
        return False
    status_age = now - float(status_at)
    ages = _local_artifact_ages(directory, now)
    activity_age = ages.get("train.log")
    if activity_age is None:
        activity_age = ages.get("history.jsonl")
    if activity_age is not None and activity_age < WATCH_ACTIVITY_FRESH_S:
        return False
    return status_age > WATCH_STATUS_STALE_S


def _poll_watch_key(timeout: float) -> str | None:
    """Wait up to ``timeout`` seconds; return a single keypress or ``None``."""
    if timeout <= 0 or not sys.stdin.isatty():
        time.sleep(max(timeout, 0))
        return None
    deadline = time.time() + timeout
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([sys.stdin], [], [], min(remaining, 0.2))
            if ready:
                ch = sys.stdin.read(1)
                return ch.lower() if ch else None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

def _progress_bar(fraction: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, fraction))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _ansi(code: str, text: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def render_watch_dashboard(
    *,
    run_id: str,
    controller: dict,
    remote: dict,
    history_rows: list[dict],
    term_width: int,
    now: float,
    flash: str | None = None,
) -> str:
    _ensure_repo_on_path()
    from syndiff_pipeline.forward_model.training_history import (
        chi2_series,
        format_age,
        format_duration,
        pct_change,
        rows_for_stage,
        sparkline,
        stage_budgets,
        overall_progress,
        training_rows,
    )

    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    width = max(72, min(term_width, 120))
    inner = width - 4
    lines: list[str] = []

    state = remote.get("state", controller.get("state", "?"))
    stage = int(remote.get("stage", 0) or 0)
    step = int(remote.get("step", 0) or 0)
    loss = remote.get("loss")
    status_at = remote.get("timestamp") or remote.get("heartbeat_at")
    status_age = (now - float(status_at)) if status_at else None
    activity_age = remote.get("_local_train_log_age_s") or remote.get("_local_history_age_s")
    stale = remote_status_stale(remote, JOBS / run_id, now)
    budgets = stage_budgets(controller)
    stage_total = next((n for st, n in budgets if st == stage), 0)
    stage_frac = (step + 1) / stage_total if stage_total else 0.0
    overall_frac, done_steps, total_steps = overall_progress(stage, step, budgets)

    train = training_rows(history_rows)
    stage_rows = rows_for_stage(train, stage) if stage else train
    if stage_rows and loss is not None:
        last = stage_rows[-1]
        if int(last.get("step", -1)) != step:
            stage_rows = stage_rows + [{"stage": stage, "step": step, "loss": float(loss)}]

    loss_vals = [float(r["loss"]) for r in stage_rows]
    loss_full = sparkline(loss_vals, width=min(50, inner - 20))
    loss_zoom = sparkline(loss_vals[-40:], width=min(40, inner - 24), log_scale=False)

    loss_delta = None
    loss_pct = None
    if len(loss_vals) >= 2:
        loss_delta = loss_vals[-1] - loss_vals[-min(51, len(loss_vals))]
    if len(loss_vals) >= 2 and loss_vals[0]:
        loss_pct = pct_change(loss_vals[0], loss_vals[-1])

    chi2_pts = chi2_series(history_rows, stage if stage else None)
    chi2_vals = [v for _st, _step, v in chi2_pts]
    chi2_full = sparkline(chi2_vals, width=min(50, inner - 20), log_scale=True)
    chi2_delta = None
    if len(chi2_vals) >= 2:
        chi2_delta = chi2_vals[-1] - chi2_vals[-2]

    elapsed = remote.get("elapsed_s")
    eta_s = None
    if elapsed and done_steps > 0 and total_steps > done_steps:
        eta_s = float(elapsed) * (total_steps - done_steps) / done_steps

    def box_top(title: str) -> None:
        pad = max(0, inner - len(title) - 3)
        lines.append(f"╭ {title}{' ' * pad}─╮")

    def box_mid() -> None:
        lines.append(f"├{'─' * inner}┤")

    def box_bot() -> None:
        lines.append(f"╰{'─' * inner}╯")

    def row(text: str) -> None:
        lines.append(f"│ {text[:inner]:<{inner}} │")

    box_top(f"{run_id}  {state}")
    if budgets and stage:
        row(
            f"Stage {stage}/{budgets[-1][0]}  step {step:4d} / {stage_total:<4d}  "
            f"[{_progress_bar(stage_frac, 18)}] {100 * stage_frac:4.1f}%"
        )
    if total_steps:
        eta = f"  ETA ~{format_duration(eta_s)}" if eta_s else ""
        row(
            f"Overall [{_progress_bar(overall_frac, 26)}] {100 * overall_frac:4.1f}%{eta}"
        )
    box_mid()

    if loss is not None:
        arrow = ""
        if loss_delta is not None:
            sym = "▼" if loss_delta < 0 else ("▲" if loss_delta > 0 else "─")
            color = "32" if loss_delta < 0 else ("31" if loss_delta > 0 else "0")
            arrow = _ansi(color, f" {sym} {abs(loss_delta):.3f}", enabled=use_color)
        pct_txt = f"  ({loss_pct:+.1f}% stage)" if loss_pct is not None else ""
        row(f"Loss     {float(loss):8.4f}{arrow}{pct_txt}")
        if loss_full:
            row(f"         {loss_full}  stage")
        if loss_zoom:
            row(f"         {loss_zoom}  last 40")
    if chi2_vals:
        arrow = ""
        if chi2_delta is not None:
            sym = "▼" if chi2_delta < 0 else ("▲" if chi2_delta > 0 else "─")
            color = "32" if chi2_delta < 0 else ("31" if chi2_delta > 0 else "0")
            arrow = _ansi(color, f" {sym} {abs(chi2_delta):.2f}", enabled=use_color)
        frac_rej = stage_rows[-1].get("frac_rejected") if stage_rows else None
        rej_txt = f"  reject {100 * float(frac_rej):.1f}%" if frac_rej is not None else ""
        row(f"χ²_red  {chi2_vals[-1]:8.2f}{arrow}{rej_txt}")
        if chi2_full:
            row(f"         {chi2_full}  reject refreshes")
    box_mid()

    gpu = remote.get("gpu_name", controller.get("gpu", "?"))
    util = remote.get("gpu_util_pct")
    vram_u = remote.get("vram_used_mb")
    vram_t = remote.get("vram_total_mb")
    temp = remote.get("gpu_temperature_c")
    power = remote.get("gpu_power_w")
    gpu_bits = [str(gpu)]
    if util is not None:
        gpu_bits.append(f"util {util}%")
    if vram_u is not None and vram_t is not None:
        gpu_bits.append(f"VRAM {vram_u}/{vram_t} MB")
    if temp is not None:
        gpu_bits.append(f"{temp}°C")
    if power is not None:
        gpu_bits.append(f"{power:.0f}W")
    hb_txt = format_age(status_age)
    # Prefer the VM's train-log age.  The login-host copy can legitimately lag
    # while a checkpoint sync is in progress and must not make a healthy remote
    # trainer look stale in the dashboard.
    log_age = remote.get("train_log_mtime_age_s")
    if log_age is None:
        log_age = remote.get("_local_train_log_age_s")
    if log_age is not None:
        hb_txt = f"{hb_txt}  log {format_age(log_age)}"
    if (
        log_age is not None
        and log_age < WATCH_ACTIVITY_FRESH_S
        and status_age is not None
        and status_age > 45
    ):
        hb_txt = f"{hb_txt} status lag"
    elif stale:
        hb_txt = _ansi("33", f"{hb_txt} STALE", enabled=use_color)
    row("  ".join(gpu_bits) + f"  {hb_txt}")

    def _plot_row(label: str, path_key: str, at_key: str, err_key: str, default_rel: str) -> None:
        plot_at = controller.get(at_key)
        plot_age = (now - plot_at) if plot_at else None
        plot_path = controller.get(path_key) or str((JOBS / run_id / "artifacts" / default_rel))
        try:
            plot_rel = str(Path(plot_path).relative_to(JOBS / run_id))
        except ValueError:
            plot_rel = plot_path
        if plot_at:
            plot_ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(plot_at))
            row(f"{label:<5} {plot_rel}  updated {format_age(plot_age)} ({plot_ts})")
        else:
            err = controller.get(err_key)
            if err:
                row(f"{label:<5} pending ({err[:36]})")
            else:
                row(f"{label:<5} pending (after next sync)")

    sync_age = now - controller.get("last_sync_at", 0) if controller.get("last_sync_at") else None
    row(f"sync {format_age(sync_age)}")
    _plot_row("linear", "watch_plot_path", "watch_plot_updated_at", "watch_plot_error", WATCH_LIVE_PLOT_REL)
    _plot_row("log-y", "watch_log_plot_path", "watch_log_plot_updated_at", "watch_log_plot_error", WATCH_LIVE_LOG_PLOT_REL)
    _plot_row("wcs", "watch_wcs_plot_path", "watch_wcs_plot_updated_at", "watch_wcs_plot_error", WATCH_LIVE_WCS_PLOT_REL)
    _plot_row("epsf", "watch_epsf_plot_path", "watch_epsf_plot_updated_at", "watch_epsf_plot_error", WATCH_LIVE_EPSF_PLOT_REL)
    _plot_row("chroma", "watch_chroma_plot_path", "watch_chroma_plot_updated_at", "watch_chroma_plot_error", WATCH_LIVE_CHROMA_PLOT_REL)
    _plot_row("s2", "watch_epsf_stage2_plot_path", "watch_epsf_stage2_plot_updated_at", "watch_epsf_stage2_plot_error", WATCH_LIVE_EPSF_STAGE2_PLOT_REL)
    if int(remote.get("stage", 0) or 0) >= 2:
        _plot_row("ac/w0", "watch_ac_w0_plot_path", "watch_ac_w0_plot_updated_at", "watch_ac_w0_plot_error", WATCH_LIVE_AC_W0_PLOT_REL)
        _plot_row("lc", "watch_lc_grid_plot_path", "watch_lc_grid_plot_updated_at", "watch_lc_grid_plot_error", WATCH_LIVE_LC_GRID_PLOT_REL)
    freeze = stage_rows[-1].get("freeze_wcs") if stage_rows else None
    if freeze is not None:
        row(f"history {len(train)} rows" + ("  WCS frozen" if freeze else ""))
    box_bot()
    if flash:
        lines.append(f"  {flash}")
    lines.append("  p=plots  s=sync  Ctrl+C=exit (job keeps running)")
    return "\033[H\033[2J" + "\n".join(lines) + "\n"


# Append-only logs that grow during training; strict manifest hash checks race and
# leave stale local copies forever while the remote file keeps growing.
STREAMING_ARTIFACTS = frozenset({
    "train.log", "runner.log", "history.jsonl", "telemetry.jsonl",
    "params_latest.npz",
})
# Kill the local colab CLI wrapper if the remote exec hangs (e.g. open console).
EXEC_TIMEOUT_PAD = 30
# Fallback downloads when remote manifest exec is blocked or times out.
ESSENTIAL_ARTIFACTS = (
    "status.json", "train.log", "history.jsonl", "history.json", "fit_meta.json",
    "params_latest.npz", "params_latest_meta.json", "params_stage3.npz", "params_stage2.npz", "params_stage1.npz",
    "params_stage0.npz", "training_state_latest.npz", "telemetry.jsonl", "runner.log",
    "params.npz", "checkpoints/checkpoint_index.jsonl",
    "thread_env.txt", "level1_audit_stage1.csv", "level1_audit_stage2.csv", "level1_audit_stage3.csv",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def last_json_object(text: str, default=None):
    decoder = json.JSONDecoder()
    found = default
    found_span = -1
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, span = decoder.raw_decode(text[index:])
            if isinstance(value, dict) and span > found_span:
                found = value
                found_span = span
        except json.JSONDecodeError:
            pass
    if found is not default:
        return found
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict): return value
        except json.JSONDecodeError:
            continue
    return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_package_files() -> list[Path]:
    """Every ``.py`` file of the fitter package, recursively (``_vendor`` included), in a stable order.

    The whole package is shipped so the bundle cannot go stale when a module gains a new import
    (a fixed module list missed ``bright_width`` and ``_vendor`` after the 10-01 migration).
    """
    return sorted(p for p in GPU_PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def source_fingerprint(bundle_dir: Path) -> str:
    paths = gpu_package_files() + [bundle_dir / "fit_bundle.npz", bundle_dir / "fit_bundle_meta.json"]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        # Bundles may live on the durable /astro filesystem; retain a stable
        # path label without requiring them to be copied into the repo.
        try:
            label = path.relative_to(ROOT)
        except ValueError:
            label = path.resolve()
        digest.update(str(label).encode()); digest.update(sha256(path).encode())
    return digest.hexdigest()


def build_archive(bundle_dir: Path, output: Path, fingerprint: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f".tmp.{os.getpid()}")
    # The 1.2 GB FitBundle is already an internally compressed NPZ. Storing it
    # avoids wasting CPU and also avoids a Python 3.13/zlib crash seen on the
    # login host while recompressing that member.
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("syndiff_pipeline/__init__.py", "")
        for path in gpu_package_files():
            archive.write(path, f"syndiff_pipeline/forward_model/{path.relative_to(GPU_PACKAGE).as_posix()}")
        for name in ("fit_bundle.npz", "fit_bundle_meta.json", "release_provenance.json"):
            if (bundle_dir / name).is_file(): archive.write(bundle_dir / name, f"bundle/{name}")
        requirements = "\n".join(f"{k}=={v}" for k, v in RUNTIME.items()) + "\n"
        archive.writestr("RUNTIME_REQUIREMENTS.txt", requirements)
        archive.writestr("BUNDLE_FINGERPRINT", fingerprint + "\n")
    os.replace(tmp, output)


def run_cmd(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, check=True, **kwargs)


def colab_binary() -> str:
    configured = os.environ.get("COLAB_BINARY")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"COLAB_BINARY is not executable: {path}")
        return str(path)
    # Keep the Colab client coupled to the Python environment running this
    # module.  ``~/.local/bin/colab`` may be a different uv installation with
    # incompatible commands/API behavior; the syndiff environment ships its
    # tested client beside its interpreter.
    env_colab = Path(sys.executable).resolve().parent / "colab"
    if env_colab.is_file() and os.access(env_colab, os.X_OK):
        return str(env_colab)
    found = shutil.which("colab")
    if not found:
        raise RuntimeError("colab CLI not found; run doctor --install")
    return found


def real_home() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def drive_auth_paths() -> tuple[str, str]:
    home = real_home()
    return str(home / "google_oauth_credentials.json"), str(home / "token.json")


def remote_gdrive_oauth_dir(run_id: str) -> str:
    return f"/content/jobs/{run_id}/.gdrive_oauth"


def ship_gdrive_oauth_to_session(session: str, run_id: str) -> bool:
    """Upload login-host OAuth files for headless emergency Drive upload on the VM."""
    credentials_file, token_file = drive_auth_paths()
    token_path = Path(token_file)
    if not token_path.is_file():
        print(f"warning: OAuth token missing at {token_file}; emergency Drive upload will fail", file=sys.stderr)
        return False
    remote_dir = remote_gdrive_oauth_dir(run_id)
    remote_exec(session, f"mkdir -p {shlex.quote(remote_dir)}", timeout=60)
    run_cmd([
        colab_binary(), "upload", "-s", session,
        str(token_path), f"{remote_dir}/token.json",
    ])
    creds_path = Path(credentials_file)
    if creds_path.is_file():
        run_cmd([
            colab_binary(), "upload", "-s", session,
            str(creds_path), f"{remote_dir}/credentials.json",
        ])
    return True


def purge_private_artifacts(directory: Path) -> list[str]:
    """Remove OAuth and other private files that must never live on shared storage."""
    sys.path.insert(0, str(ROOT))
    from syndiff_pipeline.forward_model.colab_emergency_upload import is_private_artifact, PRIVATE_ARTIFACT_DIRS

    removed: list[str] = []
    artifacts = directory / "artifacts"
    for name in PRIVATE_ARTIFACT_DIRS:
        path = artifacts / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name)
    # Do not retain VM control markers or webhook secrets on shared storage.
    if artifacts.is_dir():
        from syndiff_pipeline.forward_model.colab_emergency_upload import is_private_artifact

        for path in artifacts.rglob("*"):
            if path.is_file() and is_private_artifact(str(path.relative_to(artifacts))):
                path.unlink(missing_ok=True)
                removed.append(str(path.relative_to(artifacts)))
    cache_path = directory / "sync_manifest.json"
    cache = load_json(cache_path, {})
    dirty = False
    for rel in list(cache):
        if is_private_artifact(rel):
            cache.pop(rel, None)
            removed.append(rel)
            dirty = True
    if dirty:
        atomic_json(cache_path, cache)
    return removed


def colab_new_command(session: str, args) -> list[str]:
    command = [colab_binary(), "new", "-s", session]
    if not getattr(args, "cpu", False):
        command.extend(["--gpu", args.gpu.upper()])
        if args.high_mem and args.gpu.upper() == "T4":
            command.append("--high-mem")
    return command


def session_name(run_id: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_id)
    prefix = "syndiff-"
    return prefix + clean[-(50 - len(prefix)):]


def remote_exec(session: str, script: str, *, timeout: int = 120) -> str:
    payload = base64.b64encode(script.encode()).decode()
    python_source = (
        "import base64, subprocess\n"
        f"_p = subprocess.run(['bash','-lc',base64.b64decode('{payload}').decode()], "
        "text=True, capture_output=True)\n"
        "print(_p.stdout, end='')\n"
        "print(_p.stderr, end='', file=__import__('sys').stderr)\n"
        "if _p.returncode: raise subprocess.CalledProcessError(_p.returncode, _p.args)\n"
    )
    with SessionExecLock(session):
        try:
            result = subprocess.run(
                [colab_binary(), "exec", "-s", session, "--timeout", str(timeout), "-f", "/dev/stdin"],
                input=python_source, text=True, capture_output=True,
                timeout=timeout + EXEC_TIMEOUT_PAD, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise subprocess.CalledProcessError(
                124,
                exc.cmd,
                normalize_subprocess_output(exc.stdout),
                normalize_subprocess_output(exc.stderr),
            ) from exc
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr,
        )
    return result.stdout


def colab_download(session: str, remote_path: str, local_path: Path, *, timeout: int = 180) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_name(local_path.name + f".tmp.{os.getpid()}")
    try:
        result = subprocess.run(
            [colab_binary(), "download", "-s", session, remote_path, str(tmp)],
            text=True, capture_output=True, timeout=timeout, check=False,
        )
        if result.returncode != 0 or not tmp.is_file():
            return False
        os.replace(tmp, local_path)
        return True
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False
    finally:
        if tmp.exists() and not local_path.exists():
            tmp.unlink(missing_ok=True)


def get_session_inventory() -> SessionInventory:
    return fetch_session_inventory(colab_binary=colab_binary)


def endpoint_assignment_state(endpoint: str | None) -> tuple[str, SessionInventory]:
    """Classify a controller-owned endpoint without trusting its CLI session name.

    A runtime proxy 404 and a released VM are different failures.  The server
    inventory is the billing authority; the local CLI registry is only a
    convenience mapping and may have been pruned by an older CLI.
    """
    inventory = get_session_inventory()
    if inventory.health != "ok":
        return "unknown", inventory
    if endpoint and inventory.find_by_endpoint(endpoint) is not None:
        return "assigned", inventory
    return "absent", inventory


def mark_runtime_unreachable(
    directory: Path,
    state: dict,
    *,
    failures: int,
    error: str | None = None,
) -> bool:
    """Persist an orphaned-runtime state when billing still owns the endpoint.

    Returns true only when the server confirms the exact controller endpoint
    remains assigned.  This deliberately does *not* stop or unassign anything.
    """
    assignment, inventory = endpoint_assignment_state(state.get("endpoint"))
    if assignment != "assigned":
        return False
    state.update(
        state="orphaned_runtime",
        runtime_state="unreachable",
        runtime_unreachable_at=time.time(),
        runtime_unreachable_failures=failures,
        runtime_inventory_health=inventory.health,
        runtime_recovery_action=(
            f"runtime unavailable while endpoint {state.get('endpoint')} remains assigned; "
            f"inspect local artifacts, then use `python scripts/colab_job.py stop {state['run_id']} --force` "
            "only with explicit approval to release it"
        ),
    )
    if error:
        state["runtime_unreachable_error"] = redact_secrets(error)
    atomic_json(directory / "controller.json", state)
    return True


def controller_owned_orphans(inventory: SessionInventory) -> list[dict]:
    """Return active endpoints belonging to controllers already marked orphaned."""
    if inventory.health != "ok" or not JOBS.exists():
        return []
    active: list[dict] = []
    for path in JOBS.glob("*/controller.json"):
        controller = load_json(path, {})
        endpoint = controller.get("endpoint")
        if controller.get("state") == "orphaned_runtime" and endpoint:
            if inventory.find_by_endpoint(str(endpoint)) is not None:
                active.append({"run_id": controller.get("run_id", path.parent.name), "endpoint": endpoint})
    return active


def list_colab_sessions() -> list[str]:
    """Return session names from ``colab sessions`` (empty when inventory is unknown)."""
    inventory = get_session_inventory()
    if inventory.health != "ok":
        return []
    return inventory.names()


def session_is_active(session: str) -> bool:
    inventory = get_session_inventory()
    if inventory.health != "ok":
        raise RuntimeError(
            f"session inventory {inventory.health}: {inventory.error or 'cannot verify sessions'}"
        )
    return inventory.find_by_name(session) is not None


def verify_session_stopped(session: str, *, endpoint: str | None = None) -> bool:
    """Confirm session name and optional endpoint are no longer assigned."""
    if endpoint and not verify_endpoint_stopped(endpoint):
        return False
    inventory = get_session_inventory()
    if inventory.health != "ok":
        return False
    return inventory.find_by_name(session) is None


def stop_colab_session(
    session: str,
    *,
    endpoint: str | None = None,
    retries: int = 3,
) -> bool:
    """Stop session and verify the server endpoint is unassigned."""
    ok, _diag = stop_assignment_verified(
        session,
        endpoint=endpoint,
        colab_binary=colab_binary,
        verify_session_stopped=lambda name, endpoint=None: verify_session_stopped(name),
        retries=retries,
        force_release=True,
    )
    return ok


def _controller_path(run_id: str) -> Path:
    return JOBS / run_id / "controller.json"


def _load_controller(run_id: str, default: dict | None = None) -> dict:
    return load_json(_controller_path(run_id), default or {})


def _save_controller(run_id: str, state: dict, phase: str | None = None, **extra) -> dict:
    if phase is not None:
        state = update_controller_phase(_controller_path(run_id), state, phase, atomic_json=atomic_json, extra=extra)
    elif extra:
        state = {**state, **extra}
        atomic_json(_controller_path(run_id), state)
    else:
        atomic_json(_controller_path(run_id), state)
    return state


def _submit_cleanup(
    run_id: str,
    state: dict,
    *,
    reason: str,
    session: str | None = None,
    endpoint: str | None = None,
    diagnose: bool = True,
) -> None:
    directory = JOBS / run_id
    with SubmitCleanupLock(directory):
        session = session or state.get("session")
        endpoint = endpoint or state.get("endpoint")
        if not endpoint and session:
            endpoint = lookup_local_session_endpoint(session) or endpoint
        state = dict(_load_controller(run_id, state))
        state.update(
            state="failed",
            submit_error=reason,
            failed_at=time.time(),
            cleanup_started_at=time.time(),
        )
        atomic_json(_controller_path(run_id), state)

        diagnosis: dict | None = None
        if diagnose and session:
            try:
                diagnosis = collect_remote_diagnosis(
                    session, run_id, remote_exec=remote_exec, timeout=CLEANUP_DIAG_TIMEOUT_S,
                )
                capture_remote_logs(session, run_id, directory, colab_download=colab_download)
                state = persist_failure_diagnostics(directory, state, diagnosis, atomic_json=atomic_json)
            except Exception as exc:
                state = _load_controller(run_id, state)
                state["submit_diagnostics_error"] = redact_secrets(str(exc))
                atomic_json(_controller_path(run_id), state)

        stop_ok = False
        stop_diag: dict = {}
        if session or endpoint:
            stop_ok, stop_diag = stop_assignment_verified(
                session or state.get("session", ""),
                endpoint=endpoint,
                colab_binary=colab_binary,
                verify_session_stopped=lambda name, endpoint=None: verify_session_stopped(
                    name, endpoint=endpoint if endpoint is not None else state.get("endpoint"),
                ),
            )
        state = _load_controller(run_id, state)
        state["session_stop_diagnostics"] = redact_obj(stop_diag)
        if stop_ok:
            state["session_stopped"] = True
            state.pop("session_stop_error", None)
            state.pop("cleanup_unassign_failed", None)
        else:
            state["cleanup_unassign_failed"] = True
            state["session_stop_error"] = (
                stop_diag.get("stop_errors") or ["verified stop failed during submit cleanup"]
            )
        atomic_json(_controller_path(run_id), state)


def _launch_provision_worker(
    session: str,
    run_id: str,
    *,
    worker_remote: str,
    paths: dict[str, str],
    launch_token: str,
    timeout: int = PROVISION_LAUNCH_TIMEOUT,
    launch_started_at: float | None = None,
) -> str:
    launch_started_at = launch_started_at or time.time()
    launch_script = build_worker_launch_script(
        worker_remote, paths["provision_log"], launch_token, paths["job_dir"],
    )
    local_paths = local_run_paths(JOBS / run_id, run_id)
    last_out = ""
    launch_seen = False
    for _attempt in range(3):
        try:
            last_out = remote_exec(session, launch_script, timeout=timeout)
        except subprocess.CalledProcessError:
            last_out = ""
        launch_seen = launch_seen or "launched" in last_out
        if colab_download(session, paths["provision_status"], local_paths["provision_status"], timeout=30):
            remote = load_json(local_paths["provision_status"], {})
            if provision_status_accepts(
                remote,
                launch_token,
                min_updated_at=launch_started_at,
                allow_done=False,
            ) and remote.get("phase") and remote.get("ok") is not False:
                return "launched_and_verified" if launch_seen else "recovered_from_status"
        time.sleep(2)
    raise RuntimeError(f"provision worker produced no fresh status after launch: {last_out[:200]}")


def _start_provisioning_watchdog(run_id: str, *, deadline_s: float = PROVISION_WATCHDOG_DEFAULT_S) -> int:
    log = (JOBS / run_id / "provision_watchdog.log").open("ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, __file__, "_provision_watchdog", run_id, str(deadline_s)],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc.pid


def _start_submit_crash_watchdog(run_id: str) -> int:
    log = (JOBS / run_id / "submit_crash_watchdog.log").open("ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, __file__, "_submit_crash_watchdog", run_id],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc.pid


def submit_crash_watchdog(args) -> int:
    run_id = args.run_id
    deadline = time.time() + PROVISION_WATCHDOG_DEFAULT_S
    while time.time() < deadline:
        state = _load_controller(run_id, {})
        if state.get("state") != "provisioning":
            return 0
        submit_pid = state.get("submit_pid")
        if submit_pid:
            try:
                os.kill(int(submit_pid), 0)
            except (OSError, ValueError):
                time.sleep(SUBMIT_CRASH_GRACE_S)
                state = _load_controller(run_id, {})
                if state.get("state") != "provisioning":
                    return 0
                try:
                    os.kill(int(submit_pid), 0)
                except (OSError, ValueError):
                    reason = "submit parent died during provisioning"
                    print(f"crash-watchdog: {reason}", file=sys.stderr, flush=True)
                    _submit_cleanup(
                        run_id,
                        state,
                        reason=reason,
                        session=state.get("session"),
                        endpoint=state.get("endpoint"),
                    )
                    return 2
        time.sleep(5)
    return 0


def provision_watchdog(args) -> int:
    run_id = args.run_id
    deadline = time.time() + float(args.deadline_s)
    while time.time() < deadline:
        state = _load_controller(run_id, {})
        if state.get("state") != "provisioning":
            return 0
        time.sleep(15)
    state = _load_controller(run_id, {})
    if state.get("state") != "provisioning":
        return 0
    reason = "provisioning watchdog deadline exceeded"
    print(f"watchdog: {reason}", file=sys.stderr, flush=True)
    _submit_cleanup(
        run_id,
        state,
        reason=reason,
        session=state.get("session"),
        endpoint=state.get("endpoint"),
    )
    return 2


def diagnose_job(args) -> int:
    directory, state = job_state(args.run_id)
    session = state.get("session")
    if not session:
        raise SystemExit("controller missing session name")
    diagnosis = collect_remote_diagnosis(session, args.run_id, remote_exec=remote_exec)
    inventory = get_session_inventory()
    payload = {
        "run_id": args.run_id,
        "controller": redact_obj(state),
        "inventory": inventory.to_dict(),
        "remote": diagnosis,
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_diagnose_human(diagnosis, state))
        print("\n--- inventory ---")
        for record in inventory.assignments:
            print(record.raw_line or f"[{record.name}] {record.endpoint}")
    return 0


def doctor(args) -> int:
    problems = []
    if not shutil.which("colab"):
        if args.install:
            if not shutil.which("uv"): problems.append("uv is required to install google-colab-cli")
            else: run_cmd(["uv", "tool", "install", "google-colab-cli"])
        else: problems.append("google-colab-cli missing (use --install)")
    for tool in ("zip", "unzip"):
        if not shutil.which(tool): problems.append(f"required tool missing: {tool}")
    usage = shutil.disk_usage(ROOT)
    if usage.free < args.min_disk_gb * 1024**3: problems.append(f"only {usage.free/1024**3:.1f} GiB local disk free")
    try:
        if shutil.which("colab"): run_cmd([colab_binary(), "status"], capture_output=True)
    except subprocess.CalledProcessError as exc: problems.append(f"Colab authentication/status failed: {exc}")
    try:
        sys.path.insert(0, str(PROJECT / "scripts"))
        from upload_to_gdrive import get_gdrive_service
        creds, token = drive_auth_paths()
        get_gdrive_service(creds, token).about().get(fields="user").execute()
    except Exception as exc: problems.append(f"Drive authentication failed: {exc}")
    if DEFAULT_BUNDLE.exists():
        try: source_fingerprint(DEFAULT_BUNDLE)
        except Exception as exc: problems.append(f"bundle invalid: {exc}")
    else: problems.append(f"production bundle missing: {DEFAULT_BUNDLE}")
    if problems:
        print("doctor: FAILED\n- " + "\n- ".join(problems), file=sys.stderr); return 1
    print("doctor: Colab, Drive, bundle, disk, and local tools are ready"); return 0


def publish(args) -> int:
    bundle, output = args.bundle.resolve(), args.archive.resolve()
    fingerprint = source_fingerprint(bundle)
    manifest = load_json(MANIFEST, {"bundles": {}})
    old = manifest.get("bundles", {}).get(output.name, {})
    if old.get("source_fingerprint") != fingerprint or not output.is_file():
        build_archive(bundle, output, fingerprint)
    archive_sha = sha256(output)
    if old.get("sha256") == archive_sha and old.get("file_id"):
        print(f"unchanged: {output} (Drive id={old['file_id']})"); return 0
    sys.path.insert(0, str(PROJECT / "scripts"))
    from upload_to_gdrive import upload_file_to_gdrive
    creds, token = drive_auth_paths()
    response = upload_file_to_gdrive(output, credentials_file=creds, token_file=token)
    manifest.setdefault("bundles", {})[output.name] = {
        "file_id": response["id"], "sha256": archive_sha, "size": output.stat().st_size,
        "source_fingerprint": fingerprint, "published_at": time.time(), "path": str(output),
        "bundle_path": str(bundle),
    }
    atomic_json(MANIFEST, manifest)
    print(f"published {output.name}: id={response['id']} sha256={archive_sha}"); return 0


def job_state(run_id: str) -> tuple[Path, dict]:
    directory = JOBS / run_id
    state = load_json(directory / "controller.json")
    if not state: raise SystemExit(f"unknown run id: {run_id}")
    return directory, state


def list_run_ids() -> list[str]:
    if not JOBS.is_dir():
        return []
    return sorted(
        p.name for p in JOBS.iterdir()
        if p.is_dir() and (p / "controller.json").is_file()
    )


def _resolve_run_sort_key(directory: Path, state: dict, *, now: float) -> tuple:
    """Rank jobs for default ``run_id`` selection (higher sorts first)."""
    remote = load_json(directory / "remote_status.json", {})
    remote_state = remote.get("state") or state.get("state")
    terminal = remote_state in TERMINAL or state.get("state") in TERMINAL

    heartbeat_at = remote.get("heartbeat_at") or remote.get("timestamp")
    heartbeat = float(heartbeat_at) if heartbeat_at else 0.0
    heartbeat_fresh = heartbeat > 0 and (now - heartbeat) < WATCH_STATUS_STALE_S

    created = float(state.get("created_at") or state.get("submitted_at") or 0.0)
    if not created:
        created = float((directory / "controller.json").stat().st_mtime)
    sync = float(state.get("last_sync_at") or created)

    if terminal:
        active_rank = 0
    elif heartbeat_fresh:
        active_rank = 2
    else:
        active_rank = 1
    return (active_rank, heartbeat, created, sync)


def resolve_run_id(run_id: str | None, *, announce: bool = True, now: float | None = None) -> str:
    """Return ``run_id`` or pick the best default job to watch.

    When ``run_id`` is omitted, prefer non-terminal jobs with a fresh remote
    heartbeat, then other non-terminal jobs (by ``created_at``), then the most
    recently synced terminal job.  ``last_sync_at`` alone is not used for live
    jobs because supervisors keep syncing stale/zombie runs.
    """
    if run_id:
        return run_id
    now = time.time() if now is None else now
    candidates: list[tuple[tuple, str]] = []
    for name in list_run_ids():
        directory = JOBS / name
        state = load_json(directory / "controller.json", {})
        candidates.append((_resolve_run_sort_key(directory, state, now=now), name))
    if not candidates:
        raise SystemExit(f"no colab jobs found under {JOBS}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = candidates[0][1]
    if announce:
        print(f"(most recent run: {chosen})", file=sys.stderr)
    return chosen


def _supervisor_pid_path(directory: Path) -> Path:
    return directory / "supervisor.pid"


def _supervisor_print(message: str, *, error: bool = False) -> None:
    """Write a timestamped supervisor event to the supervisor log."""
    stream = sys.stderr if error else sys.stdout
    print(
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}",
        file=stream,
        flush=True,
    )


def _supervisor_running(directory: Path) -> int | None:
    path = _supervisor_pid_path(directory)
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        path.unlink(missing_ok=True)
        return None
    return pid


def _start_supervisor(run_id: str) -> int | None:
    directory = JOBS / run_id
    existing = _supervisor_running(directory)
    if existing is not None:
        return existing
    log_path = directory / "supervisor.log"
    unit = "syndiff-colab-supervisor-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)[:180]
    command = [sys.executable, __file__, "_supervise", run_id]

    # A detached child of the AI shell is reaped when the sandbox wrapper exits.
    # Prefer a user service (Restart=on-failure) so the supervisor has a host-
    # owned lifetime.  This also makes ordinary terminal launches durable.
    systemd = shutil.which("systemd-run")
    if systemd:
        service_path = os.pathsep.join(dict.fromkeys([
            str(Path(sys.executable).parent),
            str(real_home() / ".local" / "bin"),
            os.environ.get("PATH", ""),
        ]))
        systemd_cmd = [
            systemd, "--user", "--unit", unit, "--collect",
            "--property=Restart=on-failure", "--property=RestartSec=10s",
            "--working-directory", str(ROOT),
            "--setenv=PYTHONUNBUFFERED=1",
            f"--setenv=PATH={service_path}",
            f"--setenv=HOME={real_home()}",
            *command,
        ]
        probe = subprocess.run(
            systemd_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False,
        )
        if probe.returncode == 0:
            # systemd owns/restarts the process; the service itself writes the
            # authoritative supervisor.pid once it enters supervise().
            time.sleep(0.5)
            pid = _supervisor_running(directory)
            if pid is not None:
                return pid
            # The unit may still be starting; retain a diagnostic instead of
            # silently claiming a normal detached process is durable.
            (directory / "supervisor.launch.log").write_text(
                f"systemd unit={unit} started but PID not visible yet\n"
            )
            return None

    # Fallback for hosts without a user systemd bus.  tmux owns a detached
    # server outside the terminal PTY and is durable when launched from a real
    # login shell.  In the AI sandbox it may still be reaped; report that fact.
    tmux = shutil.which("tmux")
    if tmux:
        session = unit[:180]
        shell_cmd = " ".join(shlex.quote(x) for x in command)
        launch = f"exec {shell_cmd} >> {shlex.quote(str(log_path))} 2>&1"
        probe = subprocess.run(
            [tmux, "new-session", "-d", "-s", session, launch],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False,
        )
        if probe.returncode == 0:
            time.sleep(0.5)
            pid = _supervisor_running(directory)
            if pid is not None:
                return pid
            (directory / "supervisor.launch.log").write_text(
                f"tmux session={session} started but PID not visible yet\n"
            )
            return None

    # Last resort: useful from a normal terminal, but deliberately record that
    # no persistent host owner was available.
    log = log_path.open("ab", buffering=0)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,
    )
    _supervisor_pid_path(directory).write_text(str(proc.pid))
    (directory / "supervisor.launch.log").write_text(
        "WARNING: supervisor launched without systemd/tmux persistence\n"
    )
    return proc.pid


def _clear_supervisor_pid(directory: Path) -> None:
    _supervisor_pid_path(directory).unlink(missing_ok=True)


def _gpu_assert(gpu: str) -> str:
    gpu = gpu.upper()
    if gpu == "T4":
        return (
            "gpu_name = str(getattr(jax.devices()[0], 'device_kind', '')).lower()\n"
            "assert 't4' in gpu_name, f'not a T4-compatible device: {{gpu_name}}'"
        )
    if gpu == "L4":
        return (
            "gpu_name = str(getattr(jax.devices()[0], 'device_kind', '')).lower()\n"
            "assert 'l4' in gpu_name, f'not an L4 device: {{gpu_name}}'"
        )
    raise ValueError(f"unsupported --gpu {gpu!r}")


def resolve_submit_discord_webhook(args) -> str | None:
    if getattr(args, "no_discord", False):
        return None
    explicit = getattr(args, "discord_webhook_url", None)
    if explicit:
        return str(explicit).strip() or None
    deployment_dir = Path(getattr(args, "deployment", DEFAULT_DEPLOYMENT_DIR))
    from syndiff_pipeline.common.orchestration.notifications import load_webhook_url
    return load_webhook_url(deployment_dir / "pipeline.yaml", "deployment.yaml")


def discord_runner_env_prefix(webhook_url: str | None) -> str:
    if not webhook_url:
        return ""
    return f"env SYNDIFF_DISCORD_WEBHOOK_URL={shlex.quote(webhook_url)} "


def resolve_submit_bundle_path(args, manifest_entry: dict) -> Path | None:
    if getattr(args, "bundle", None):
        bundle = Path(args.bundle).resolve()
        if not _bundle_path_usable(bundle):
            raise SystemExit(
                f"--bundle not usable (need file or directory with fit_bundle.npz): {bundle}"
            )
        return bundle
    bundle_raw = manifest_entry.get("bundle_path")
    if bundle_raw:
        bundle = Path(bundle_raw).resolve()
        if bundle.is_file() or (bundle.is_dir() and (bundle / "fit_bundle.npz").is_file()):
            return bundle
    return None


def _bundle_path_usable(bundle: Path) -> bool:
    bundle = Path(bundle).resolve()
    return bundle.is_file() or (bundle.is_dir() and (bundle / "fit_bundle.npz").is_file())


def resolve_job_bundle_path(state: dict) -> Path | None:
    """Local FitBundle for a job: ``controller.json`` bundle_path or gdrive manifest."""
    bundle_raw = state.get("bundle_path")
    if bundle_raw and _bundle_path_usable(Path(bundle_raw)):
        return Path(bundle_raw).resolve()
    manifest = load_json(MANIFEST, {"bundles": {}})
    bundles = manifest.get("bundles", {})
    archive = state.get("archive_name")
    if archive and archive in bundles:
        entry = bundles[archive]
        bp = entry.get("bundle_path")
        if bp and _bundle_path_usable(Path(bp)):
            return Path(bp).resolve()
    sha = state.get("bundle_sha256")
    if sha:
        for entry in bundles.values():
            if entry.get("sha256") == sha:
                bp = entry.get("bundle_path")
                if bp and _bundle_path_usable(Path(bp)):
                    return Path(bp).resolve()
    return None


def ensure_job_bundle_path(run_id: str, state: dict) -> Path | None:
    """Resolve bundle path and persist to ``controller.json`` when inferred from manifest."""
    bundle = resolve_job_bundle_path(state)
    if bundle is None:
        return None
    bundle_str = str(bundle)
    if state.get("bundle_path") != bundle_str:
        state["bundle_path"] = bundle_str
        atomic_json(JOBS / run_id / "controller.json", state)
    return bundle


# Never selected as the post-fit parameter set: the bootstrap is the PREVIOUS
# run's output, copied in before training starts, and the "nochroma" file is a
# hand-made ablation. Both are frequently the newest file on disk after a sync.
POSTFIT_PARAMS_NEVER = ("params_bootstrap.npz", "params_stage3_nochroma.npz")

# Preference order for the post-fit parameter set, best first.
POSTFIT_PARAMS_PREFERRED = (
    "params_stage3.npz",
    "params_stage2.npz",
    "params_stage1.npz",
    "params_latest.npz",
    "params.npz",
)


def resolve_postfit_params(artifacts_dir: Path) -> str | None:
    """Return the parameter artifact post-fit should analyse.

    Selection is by NAME, in a fixed preference order, and never by filesystem
    mtime.  Choosing by mtime is what this used to do, and it silently picked
    ``params_bootstrap.npz`` -- the *previous* run's trained parameters, copied
    in before this run started -- in every one of r9, chroma-v1 and chroma-v2,
    because syncing from the VM does not preserve write order.  Every post-fit
    product from those runs therefore describes the previous generation of the
    model rather than the run that produced it
    (``dev/forward_epsf_wcs/docs/TEMPORAL_RESIDUAL_ROOT_CAUSE_20260906.md`` 6c).

    Falls back to the newest *stage* checkpoint under ``checkpoints/`` only when
    no top-level trained artifact exists, and never returns a file in
    ``POSTFIT_PARAMS_NEVER``.
    """
    for name in POSTFIT_PARAMS_PREFERRED:
        candidate = artifacts_dir / name
        if candidate.is_file():
            return name
    checkpoints = artifacts_dir / "checkpoints"
    if checkpoints.is_dir():
        stepped = [
            path
            for path in checkpoints.glob("params_s*_step*.npz")
            if path.is_file() and path.name not in POSTFIT_PARAMS_NEVER
        ]
        if stepped:
            # Highest (stage, step) by name, which is zero-padded and therefore
            # sorts correctly -- again avoiding mtime.
            newest = max(stepped, key=lambda path: path.name)
            return newest.relative_to(artifacts_dir).as_posix()
    return None


def postfit_frame_indices(expect_frames: int | None) -> str:
    if not expect_frames or expect_frames <= 1:
        return "0,147,294"
    last = expect_frames - 1
    return f"0,{last // 2},{last}"


def launch_postfit_diagnostics(run_id: str, state: dict, *, force: bool = False) -> int | None:
    """Launch ``run_postfit`` on synced artifacts; return PID when available."""
    # ``--force`` is the explicit operator override for a real run that was
    # submitted with post-fit disabled (for example, an older submission).
    if state.get("no_postfit") and not force:
        return None
    if state.get("postfit_launched") and not force:
        return state.get("postfit_pid")
    directory = JOBS / run_id
    artifacts = directory / "artifacts"
    bundle = ensure_job_bundle_path(run_id, state)
    if bundle is None:
        _supervisor_print("supervise: skip postfit — bundle_path missing from controller.json")
        return None
    # Controllers conventionally store the bundle directory, while
    # ``run_postfit`` takes the concrete NPZ path.
    if bundle.is_dir():
        bundle = bundle / "fit_bundle.npz"
    if not bundle.is_file():
        _supervisor_print(f"supervise: skip postfit — bundle file missing: {bundle}")
        return None
    params_name = resolve_postfit_params(artifacts)
    if not params_name:
        _supervisor_print("supervise: skip postfit — no params checkpoint in artifacts")
        return None
    log_path = directory / "postfit.log"
    cmd = [
        sys.executable, "-m", "dev.forward_epsf_wcs.diagnostics.run_postfit",
        str(artifacts),
        "--bundle", str(bundle),
        "--params", params_name,
        "--frame-indices", postfit_frame_indices(state.get("expect_frames")),
    ]
    if state.get("expect_frames") == 1:
        cmd = [
            sys.executable, "-m", "dev.forward_epsf_wcs.diagnostics.run_postfit_static",
            "--bundle", str(bundle), "--params", str(artifacts / params_name),
            "--out-dir", str(artifacts / "plots"),
        ]
        metadata = load_json(bundle.parent / "fit_bundle_meta.json", {})
        workspace = metadata.get("workspace")
        if workspace:
            reference = Path(workspace).parent / "wcs" / "temporal_cheb5_bspline_v1"
            if reference.is_dir():
                cmd.extend(["--temporal-wcs-root", str(reference)])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    unit = "syndiff-colab-postfit-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)[:180]
    postfit_pid: int | None = None
    launcher = "popen"
    systemd = shutil.which("systemd-run")
    if systemd:
        # A child of an AI/sandbox shell can be reaped as soon as its launcher
        # exits.  Put expensive post-fit work under the user manager instead.
        # The cap prevents this optional analysis from provoking a global OOM.
        systemd_cmd = [
            systemd, "--user", "--unit", unit, "--collect",
            "--working-directory", str(ROOT),
            "--setenv=PYTHONUNBUFFERED=1",
            "--setenv=OMP_NUM_THREADS=1",
            "--setenv=OPENBLAS_NUM_THREADS=1",
            "--setenv=MKL_NUM_THREADS=1",
            "--property=MemoryHigh=60G",
            "--property=MemoryMax=80G",
            f"--property=StandardOutput=append:{log_path}",
            f"--property=StandardError=append:{log_path}",
            *cmd,
        ]
        started = subprocess.run(
            systemd_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False,
        )
        if started.returncode == 0:
            launcher = "systemd"
            try:
                shown = subprocess.run(
                    ["systemctl", "--user", "show", unit, "--property=MainPID", "--value"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
                )
                if shown.returncode == 0 and shown.stdout.strip().isdigit():
                    postfit_pid = int(shown.stdout.strip()) or None
            except OSError:
                pass
        else:
            _supervisor_print(
                f"supervise: systemd postfit launch failed; using non-durable fallback: "
                f"{started.stderr.strip()}", error=True,
            )
    if launcher == "popen":
        log = log_path.open("ab", buffering=0)
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        postfit_pid = proc.pid
    updated = load_json(directory / "controller.json", state)
    updated.update(
        postfit_launched=True,
        postfit_pid=postfit_pid,
        postfit_unit=unit if launcher == "systemd" else None,
        postfit_launcher=launcher,
        postfit_started_at=time.time(),
        postfit_params=params_name,
    )
    updated.pop("postfit_error", None)
    atomic_json(directory / "controller.json", updated)
    _supervisor_print(
        f"supervise: launched postfit launcher={launcher} pid={postfit_pid} log={log_path}"
    )
    return postfit_pid or 0


def submit(args) -> int:
    run_id = args.run_id or time.strftime("mag711-%Y%m%d-%H%M%S")
    args.run_id = run_id
    directory = JOBS / run_id
    manifest = load_json(MANIFEST, {"bundles": {}})
    prior_state = load_json(directory / "controller.json", {}) if args.resume else {}
    session = prior_state.get("session", session_name(run_id))
    directory.mkdir(parents=True, exist_ok=args.resume)
    entry = manifest.get("bundles", {}).get(args.archive.name)
    if not entry:
        raise SystemExit("archive is not published; run publish first")
    use_cpu = bool(getattr(args, "cpu", False))
    gpu = args.gpu.upper()
    if args.steps_per_stage:
        steps = args.steps_per_stage
    elif args.mode == "probe":
        steps = "5,5,1" if args.isolate_stages else "0,0,5"
    else:
        steps = "8,600,600"
    first_stage = int(args.first_stage)
    epsf_lr_scale = float(args.epsf_lr_scale)
    lr_per_stage = args.lr_per_stage
    stage2_freeze = int(args.stage2_freeze_wcs_steps)
    chunk = args.stamp_chunk if args.stamp_chunk is not None else 5
    discord_webhook_url = resolve_submit_discord_webhook(args)
    bootstrap_sha256 = None
    if args.bootstrap_init_params:
        bootstrap_path = Path(args.bootstrap_init_params).resolve()
        if bootstrap_path.is_file():
            bootstrap_sha256 = sha256(bootstrap_path)
    state = {"run_id": run_id, "session": session, "mode": args.mode, "state": "provisioning",
             "phase": "validate_local",
             "archive_name": args.archive.name,
             "bundle_file_id": entry["file_id"], "bundle_sha256": entry["sha256"],
             "created_at": time.time(), "sync_state": "pending", "stamp_chunk": chunk,
             "gpu": "CPU" if use_cpu else gpu, "cpu": use_cpu,
             "expect_frames": args.expect_frames, "steps_per_stage": steps,
             "first_stage": first_stage, "epsf_lr_scale": epsf_lr_scale,
             "w_lr_scale": getattr(args, "w_lr_scale", None),
             "lr_per_stage": lr_per_stage,
             "post_terminal_grace_s": float(args.post_terminal_grace_s),
             "emergency_upload_attempts": 3,
             "huber_delta": float(args.huber_delta),
             "lambda_pixel_lap": float(args.lambda_pixel_lap),
             "lambda_lap": float(args.lambda_lap),
             "lambda_smooth_wcs": float(args.lambda_smooth_wcs),
             "lambda_smooth_w": float(args.lambda_smooth_w),
             "stage2_freeze_wcs_steps": stage2_freeze,
             "isolate_stages": bool(args.isolate_stages), "reject_mode": args.reject_mode,
             "reject_every": args.reject_every, "early_stop_patience": args.early_stop_patience,
             "early_stop_tol": args.early_stop_tol}
    if discord_webhook_url:
        state["discord_webhook_configured"] = True
    # Persist the default explicitly so every controller records its post-fit
    # policy.  Full runs default to post-fit after any terminal verified sync.
    state["no_postfit"] = bool(args.no_postfit)
    if args.bootstrap_init_params:
        state["bootstrap_init_params"] = str(args.bootstrap_init_params)
        state["bootstrap_sha256"] = bootstrap_sha256
    state["config_fingerprint"] = config_fingerprint(state)
    validated = validate_submit_local(
        args,
        jobs_root=JOBS,
        manifest=manifest,
        colab_binary=colab_binary,
        resolve_submit_bundle_path=resolve_submit_bundle_path,
        bundle_path_usable=_bundle_path_usable,
        root=ROOT,
        min_disk_gb=2.0,
        prior_state=prior_state,
        new_fingerprint=state["config_fingerprint"],
    )
    entry = validated["archive_entry"]
    bundle_path = validated.get("bundle_path") or resolve_submit_bundle_path(args, entry)
    if bundle_path:
        state["bundle_path"] = str(bundle_path)
    atomic_json(directory / "controller.json", state)

    endpoint = prior_state.get("endpoint")
    launch_token = hashlib.sha256(
        f"{run_id}:{state['config_fingerprint']}:{time.time()}".encode(),
    ).hexdigest()[:16]
    local_paths = local_run_paths(directory, run_id)
    allocated = bool(endpoint)
    crash_watchdog_pid: int | None = None
    try:
        inventory = get_session_inventory()
        if inventory.health != "ok":
            raise SystemExit(
                f"cannot verify Colab sessions ({inventory.health}): "
                f"{inventory.error or 'refusing to allocate'}"
            )
        anonymous_assignments = [record.endpoint for record in inventory.assignments if record.orphan]
        if anonymous_assignments:
            raise SystemExit(
                "refusing to allocate while anonymous Colab assignment(s) remain active: "
                + ", ".join(anonymous_assignments)
            )
        foreign_orphans = [
            item for item in controller_owned_orphans(inventory)
            if item["run_id"] != run_id
        ]
        if foreign_orphans:
            details = ", ".join(f"{item['run_id']} ({item['endpoint']})" for item in foreign_orphans)
            raise SystemExit(
                "refusing to allocate while controller-owned Colab orphan(s) remain assigned: "
                f"{details}; recover or explicitly release the recorded endpoint first"
            )
        existing = inventory.find_by_name(session)
        state = _save_controller(run_id, state, "allocate")
        if args.resume:
            if existing is not None:
                endpoint = existing.endpoint
                allocated = True
            else:
                probe = subprocess.run(
                    [colab_binary(), "status", "-s", session],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if probe.returncode != 0:
                    raise SystemExit(
                        f"--resume: session {session!r} not active and status lookup failed"
                    )
                endpoint = lookup_local_session_endpoint(session) or endpoint
                if endpoint:
                    allocated = True
        elif existing is not None:
            raise SystemExit(
                f"session {session!r} already assigned (endpoint={existing.endpoint}); "
                "use --resume or stop the existing session first"
            )
        else:
            state = _save_controller(
                run_id,
                load_json(directory / "controller.json", state),
                submit_pid=os.getpid(),
            )
            crash_watchdog_pid = _start_submit_crash_watchdog(run_id)
            state = _save_controller(
                run_id,
                load_json(directory / "controller.json", state),
                submit_crash_watchdog_pid=crash_watchdog_pid,
            )
            run_cmd(colab_new_command(session, args))
            allocated = True
            endpoint = lookup_local_session_endpoint(session)
            if not endpoint:
                endpoint = resolve_session_endpoint(session, inventory=get_session_inventory())
            if not endpoint:
                raise RuntimeError(
                    f"allocated session {session!r} missing endpoint in local colab-cli state"
                )
        if args.resume and allocated:
            state = _save_controller(
                run_id,
                load_json(directory / "controller.json", state),
                submit_pid=os.getpid(),
            )
            crash_watchdog_pid = _start_submit_crash_watchdog(run_id)
            state = _save_controller(
                run_id,
                load_json(directory / "controller.json", state),
                submit_crash_watchdog_pid=crash_watchdog_pid,
            )
        state = _save_controller(
            run_id,
            load_json(directory / "controller.json", state),
            "allocate",
            endpoint=endpoint,
            allocated_at=time.time(),
        )
        watchdog_pid = _start_provisioning_watchdog(run_id)
        state = _save_controller(
            run_id,
            load_json(directory / "controller.json", state),
            provision_watchdog_pid=watchdog_pid,
            provision_launch_token=launch_token,
        )

        ready, ready_err = readiness_handshake(
            session,
            remote_exec=remote_exec,
            inventory=get_session_inventory,
            shell_probe=READINESS_SHELL_PROBE,
        )
        if not ready:
            raise RuntimeError(f"session readiness failed: {ready_err}")
        state = _save_controller(run_id, load_json(directory / "controller.json", state), "readiness")

        paths = remote_paths(run_id)
        if args.resume:
            remote_exec(session, build_resume_cleanup_script(run_id), timeout=120)

        remote_exec(session, f"mkdir -p {shlex.quote(paths['job_dir'])}", timeout=60)
        token_shipped = False
        try:
            token_shipped = ship_gdrive_oauth_to_session(session, run_id)
        except Exception as exc:
            print(f"warning: OAuth shipment failed (continuing): {exc}", file=sys.stderr)
        state = _save_controller(
            run_id,
            load_json(directory / "controller.json", state),
            gdrive_token_shipped=token_shipped,
            gdrive_token_warning=not token_shipped,
        )

        bootstrap_remote = None
        if args.bootstrap_init_params:
            bootstrap = Path(args.bootstrap_init_params).resolve()
            bootstrap_remote = f"{paths['job_dir']}/params_bootstrap.npz"
            run_cmd([
                colab_binary(), "upload", "-s", session,
                str(bootstrap), bootstrap_remote,
            ])

        post_grace = float(args.post_terminal_grace_s)
        pip_packages = [
            f"jax=={RUNTIME['jax']}" if use_cpu else f"jax[cuda12]=={RUNTIME['jax']}",
            f"optax=={RUNTIME['optax']}",
            f"numpy=={RUNTIME['numpy']}",
            "gdown",
            "google-auth",
            "google-api-python-client",
        ]
        if use_cpu:
            jax_backend_assert = (
                "assert jax.default_backend() in ('cpu', 'gpu'), "
                "f'JAX backend={jax.default_backend()}'"
            )
            gpu_device_assert = ""
            train_allow_cpu = " --allow-cpu"
        else:
            jax_backend_assert = (
                "assert jax.default_backend() == 'gpu', "
                "f'JAX backend={jax.default_backend()}'"
            )
            gpu_device_assert = _gpu_assert(gpu)
            train_allow_cpu = ""
        preflight_py = f"""import jax, pathlib, sys
import numpy, optax
name = str(jax.devices()[0]).lower() if jax.devices() else ''
mem = int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:')))
avail = int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
{jax_backend_assert}
{gpu_device_assert}
assert mem >= {args.min_ram_gb} * 1024 * 1024, f'host RAM too low: {{mem/1024/1024:.1f}} GiB'
assert avail >= {args.min_available_ram_gb} * 1024 * 1024, f'available RAM too low: {{avail/1024/1024:.1f}} GiB'
assert jax.__version__ == '{RUNTIME['jax']}', jax.__version__
assert numpy.__version__ == '{RUNTIME['numpy']}', numpy.__version__
assert optax.__version__ == '{RUNTIME['optax']}', optax.__version__
print(f"SYNDIFF_BACKEND={{jax.default_backend()}}")
if jax.devices():
    print(f"SYNDIFF_DEVICE_KIND={{getattr(jax.devices()[0], 'device_kind', '')}}")
print(jax.__version__, jax.devices(), mem)
"""
        isolate_prefix = ""
        if args.isolate_stages:
            isolate_prefix = (
                f"--first-stage {first_stage} --last-stage 3 "
                f"--steps-per-stage {steps} "
            )
            if args.bootstrap_init_params:
                isolate_prefix += f"--bootstrap-init-params {paths['job_dir']}/params_bootstrap.npz "
        runner_cmd = [
            "python", "-m", "syndiff_pipeline.forward_model.gpu_job_runner",
            "--run-id", run_id,
            "--job-dir", paths["job_dir"],
            "--cwd", "/content/syndiff",
            "--post-terminal-grace-s", str(post_grace),
            "--emergency-upload-attempts", "3",
            "--",
            "python", "-m",
            f"syndiff_pipeline.forward_model.{'isolated_stage_runner' if args.isolate_stages else 'train_from_bundle'}",
            "--from-bundle", "/content/syndiff/bundle/fit_bundle.npz",
            "--out-dir", paths["job_dir"],
            "--stop-file", f"{paths['job_dir']}/STOP",
        ]
        if args.isolate_stages:
            # ``isolated_stage_runner`` owns ADVANCE_STAGE and treats the
            # first argument after its training-argument remainder as the
            # boundary.  Passing --advance-file here made it consume the
            # subsequently managed --steps-per-stage as a train argument.
            # That fails before the trainer can start.  Keep only its own
            # managed flags before ``--``.
            runner_cmd.extend(isolate_prefix.split())
            runner_cmd.append("--")
        else:
            runner_cmd.extend([
                "--advance-file", f"{paths['job_dir']}/ADVANCE_STAGE",
                "--stage", "3",
                "--start-stage", str(first_stage),
                "--steps-per-stage", steps,
            ])
        runner_cmd.extend([
            "--lr-per-stage", lr_per_stage,
            "--epsf-lr-scale", str(epsf_lr_scale),
            "--stage2-freeze-wcs-steps", str(stage2_freeze),
            "--stamp-chunk", str(chunk),
            "--flux-objective", "l2",
            "--huber-delta", str(args.huber_delta),
            "--lambda-pixel-lap", str(args.lambda_pixel_lap),
            "--lambda-lap", str(args.lambda_lap),
            "--lambda-smooth-wcs", str(args.lambda_smooth_wcs),
            "--lambda-smooth-w", str(args.lambda_smooth_w),
            "--reject-mode", args.reject_mode,
            "--reject-every", str(args.reject_every),
            "--support-size-weight-power", "0",
            "--log-every", "1",
            "--checkpoint-every", "5",
            "--early-stop-patience", str(args.early_stop_patience),
            "--early-stop-tol", str(args.early_stop_tol),
        ])
        if getattr(args, "w_lr_scale", None) is not None:
            runner_cmd.extend(["--w-lr-scale", str(args.w_lr_scale)])
        if getattr(args, "chroma", False):
            # Chromatic PSF term. Requires a bundle carrying per-star bp_rp; the
            # trainer exits with a clear message if it is missing. Unfrozen at
            # stage 3 only.
            runner_cmd.extend([
                "--chroma",
                "--chroma-lr-scale", str(getattr(args, "chroma_lr_scale", 1.0)),
            ])
            if getattr(args, "chroma_affine", False):
                # C1 colour-affine extension (chroma_aniso/chroma_shear). Only
                # forwarded when --chroma is also set, matching train_from_bundle's
                # own "Requires --chroma" contract.
                runner_cmd.append("--chroma-affine")
            if getattr(args, "chroma_kurt", False):
                # C1 optional flux-neutral kurtosis leaf; requires --chroma-affine.
                runner_cmd.append("--chroma-kurt")
        if getattr(args, "freeze_epsf_modes", False):
            runner_cmd.append("--freeze-epsf-modes")
        if getattr(args, "stamp_pedestal", False):
            # Task M4: additive per-group-per-frame pedestal solved jointly
            # with flux. Forwarded only when set; default off is unaffected.
            runner_cmd.append("--stamp-pedestal")
        if getattr(args, "profile_w", False):
            # Task PW: profile the temporal mode amplitudes out in closed form
            # per frame, jointly with the per-stamp fluxes, instead of training
            # the w_coeff spline. Forwarded only when set; default off is
            # bit-identical. Costs (n_modes+1)x the render per step.
            runner_cmd.append("--profile-w")
            if getattr(args, "profile_w_iters", None) is not None:
                runner_cmd.extend(["--profile-w-iters", str(int(args.profile_w_iters))])
        if getattr(args, "w_spatial", False):
            # T3: per-node temporal-mode amplitude field w_k(t, x, y).
            # Forwarded only when set; default off is bit-identical to the
            # pre-T3 global-w model.
            runner_cmd.append("--w-spatial")
        if train_allow_cpu.strip():
            runner_cmd.append(train_allow_cpu.strip())
        discord_secret_path = None
        if discord_webhook_url:
            secret_local = directory / f".discord_webhook.{os.getpid()}.tmp"
            secret_local.write_text(discord_webhook_url, encoding="utf-8")
            os.chmod(secret_local, 0o600)
            discord_secret_path = f"{paths['job_dir']}/.discord_webhook"
            try:
                run_cmd([
                    colab_binary(), "upload", "-s", session,
                    str(secret_local), discord_secret_path,
                ])
                remote_exec(
                    session,
                    f"chmod 600 {shlex.quote(discord_secret_path)}",
                    timeout=30,
                )
            finally:
                secret_local.unlink(missing_ok=True)
        worker_spec = {
            "run_id": run_id,
            "job_dir": paths["job_dir"],
            "bundle_file_id": entry["file_id"],
            "bundle_sha256": entry["sha256"],
            "expect_frames": args.expect_frames,
            "runtime": RUNTIME,
            "pip_packages": pip_packages,
            "preflight_py": preflight_py,
            "bootstrap_remote": bootstrap_remote,
            "runner_cmd": runner_cmd,
            "launch_token": launch_token,
            "config_fingerprint": state["config_fingerprint"],
            "expected_gpu": state["gpu"],
            "use_cpu": use_cpu,
            "discord_secret_path": discord_secret_path,
        }
        worker_source = build_provision_worker_source(worker_spec)
        worker_local = directory / "provision_worker.py"
        worker_local.write_text(worker_source, encoding="utf-8")
        worker_remote = f"{paths['job_dir']}/provision_worker.py"
        run_cmd([colab_binary(), "upload", "-s", session, str(worker_local), worker_remote])
        state = _save_controller(run_id, load_json(directory / "controller.json", state), "provision_launch")
        launch_started_at = time.time()
        _launch_provision_worker(
            session,
            run_id,
            worker_remote=worker_remote,
            paths=paths,
            launch_token=launch_token,
            launch_started_at=launch_started_at,
        )

        state = _save_controller(run_id, load_json(directory / "controller.json", state), "provision_monitor")
        prov_status, prov_err = monitor_provision(
            session,
            run_id,
            launch_token=launch_token,
            colab_download=colab_download,
            paths=paths,
            local_paths=local_paths,
        )
        if prov_err:
            relaunch_started_at = time.time()

            def _clear_stale_status() -> None:
                remote_exec(
                    session,
                    clear_remote_provision_status_script(paths["job_dir"]),
                    timeout=30,
                )
                local_paths["provision_status"].unlink(missing_ok=True)

            if _maybe_gdown_fallback(
                session,
                args.archive,
                prov_status,
                colab_binary=colab_binary,
                run_cmd=run_cmd,
                launch_token=launch_token,
                clear_remote_status=_clear_stale_status,
                relaunch=lambda: _launch_provision_worker(
                    session,
                    run_id,
                    worker_remote=worker_remote,
                    paths=paths,
                    launch_token=launch_token,
                    timeout=PROVISION_LAUNCH_RETRY_TIMEOUT,
                    launch_started_at=relaunch_started_at,
                ),
            ):
                prov_status, prov_err = monitor_provision(
                    session,
                    run_id,
                    launch_token=launch_token,
                    colab_download=colab_download,
                    paths=paths,
                    local_paths=local_paths,
                    min_updated_at=relaunch_started_at,
                )
        if prov_err:
            capture_remote_logs(session, run_id, directory, colab_download=colab_download)
            raise RuntimeError(prov_err)
        state = _save_controller(
            run_id,
            load_json(directory / "controller.json", state),
            "runner_proof",
            provision_status=prov_status,
        )
        ok, proof = validate_runner_proof(
            session,
            run_id,
            state,
            colab_download=colab_download,
            remote_exec=remote_exec,
            paths=paths,
            local_paths=local_paths,
        )
        if not ok:
            capture_remote_logs(session, run_id, directory, colab_download=colab_download)
            raise RuntimeError(proof.get("failure") or "runner proof failed")
        state = load_json(directory / "controller.json", state)
        state.update(
            state="running",
            submitted_at=time.time(),
            sync_state="pending",
            final_sync=False,
            runner_proof=redact_obj(proof),
            phase="supervisor",
        )
        atomic_json(directory / "controller.json", state)
        if not args.no_supervisor:
            _start_supervisor(run_id)
        _save_controller(run_id, load_json(directory / "controller.json", state), "done")
        print(run_id)
        return 0
    except KeyboardInterrupt:
        state = _load_controller(run_id, state)
        exposed = allocated or endpoint or state.get("endpoint")
        if exposed and not getattr(args, "keep_vm_on_failure", False):
            _submit_cleanup(
                run_id,
                state,
                reason="submit interrupted: KeyboardInterrupt",
                session=session,
                endpoint=endpoint or state.get("endpoint"),
            )
        raise
    except SystemExit as exc:
        if args.resume and prior_state:
            atomic_json(directory / "controller.json", prior_state)
        else:
            state = _load_controller(run_id, state)
            state.update(state="submit_rejected", submit_error=str(exc), rejected_at=time.time())
            atomic_json(directory / "controller.json", state)
        raise
    except Exception as exc:
        state = _load_controller(run_id, state)
        if not getattr(args, "keep_vm_on_failure", False):
            _submit_cleanup(
                run_id,
                state,
                reason=str(exc),
                session=session,
                endpoint=endpoint or state.get("endpoint"),
            )
        else:
            state.update(submit_error=str(exc), failed_at=time.time())
            atomic_json(directory / "controller.json", state)
        raise SystemExit(str(exc)) from exc


def fetch_remote_status(directory: Path, state: dict) -> tuple[dict, bool]:
    """Return ``(status_dict, fresh_from_remote)``. ``fresh`` is false when only cache is available."""
    session = state["session"]
    run_id = state["run_id"]
    remote_path = f"/content/jobs/{run_id}/status.json"
    cached = directory / "remote_status.json"
    tmp = cached.with_name(cached.name + f".tmp.{os.getpid()}")
    if colab_download(session, remote_path, tmp, timeout=120):
        try:
            remote = json.loads(tmp.read_text())
            if isinstance(remote, dict):
                os.replace(tmp, cached)
                return remote, True
        except json.JSONDecodeError:
            tmp.unlink(missing_ok=True)
    try:
        text = remote_exec(
            session,
            f"cat {shlex.quote(remote_path)} 2>/dev/null || true",
            timeout=60,
        )
        remote = last_json_object(text, {})
        if remote:
            atomic_json(cached, remote)
            return remote, True
        return load_json(cached, {}), False
    except (subprocess.CalledProcessError, RuntimeError):
        return load_json(cached, {}), False


def _fetch_remote_manifest(session: str, run_id: str, *, timeout: int = 120) -> dict:
    script = (
        "cd /content/syndiff && python -c "
        f"\"import json; from pathlib import Path; from syndiff_pipeline.forward_model.gpu_job_runner import artifact_manifest; "
        f"print(json.dumps(artifact_manifest(Path('/content/jobs/{run_id}'))))\""
    )
    try:
        listing = remote_exec(session, script, timeout=timeout)
    except (subprocess.CalledProcessError, RuntimeError):
        return {}
    return last_json_object(listing, {}) or {}


def _has_training_outputs(directory: Path) -> bool:
    artifacts = directory / "artifacts"
    return (
        (artifacts / "params_stage3.npz").is_file()
        or (artifacts / "params_latest.npz").is_file()
        or any((artifacts / "checkpoints").glob("params_s*_step*.npz"))
    )


def _newest_parameter_artifact(directory: Path) -> str | None:
    artifacts = directory / "artifacts"
    checkpoints = []
    pattern = re.compile(r"params_s(?P<stage>\d+)_step(?P<step>\d+)\.npz$")
    for path in (artifacts / "checkpoints").glob("params_s*_step*.npz"):
        match = pattern.match(path.name)
        if match:
            checkpoints.append((int(match.group("stage")), int(match.group("step")), path))
    if checkpoints:
        return str(max(checkpoints, key=lambda item: (item[0], item[1], item[2].name))[2].relative_to(artifacts))
    for name in ("params_stage3.npz", "params.npz", "params_latest.npz"):
        if (artifacts / name).is_file():
            return name
    return None


def _download_artifact(
    session: str,
    run_id: str,
    rel: str,
    directory: Path,
    cache: dict,
    *,
    remote_info: dict | None = None,
) -> bool:
    sys.path.insert(0, str(ROOT))
    from syndiff_pipeline.forward_model.colab_emergency_upload import is_private_artifact

    if is_private_artifact(rel):
        return False
    target = directory / "artifacts" / rel
    if remote_info:
        cached_hash = cache.get(rel)
        if rel in STREAMING_ARTIFACTS and target.is_file():
            remote_size = int(remote_info["size"])
            if remote_size > target.stat().st_size:
                pass  # remote grew — always try download
            elif cached_hash == remote_info["sha256"]:
                return False
        elif cached_hash == remote_info["sha256"]:
            return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="colab-sync-", dir=directory))
    try:
        downloaded = tmpdir / Path(rel).name
        if not colab_download(session, f"/content/jobs/{run_id}/{rel}", downloaded):
            return False
        digest = sha256(downloaded)
        got_size = downloaded.stat().st_size
        if remote_info:
            remote_size = int(remote_info["size"])
            hash_ok = digest == remote_info["sha256"]
            size_ok = got_size == remote_size
            if rel in STREAMING_ARTIFACTS:
                # Manifest is a point-in-time snapshot; the file often grows before
                # download finishes. Accept any snapshot at least as large as the
                # manifest reported (reject only obvious truncation).
                if got_size < remote_size:
                    return False
            elif not (hash_ok and size_ok):
                return False
        os.replace(downloaded, target)
        cache[rel] = digest
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _sync_essential_artifacts(session: str, run_id: str, directory: Path, cache: dict) -> list[str]:
    changed_paths: list[str] = []
    for rel in ESSENTIAL_ARTIFACTS:
        if _download_artifact(session, run_id, rel, directory, cache):
            changed_paths.append(rel)
    index_path = directory / "artifacts" / "checkpoints" / "checkpoint_index.jsonl"
    checkpoint_paths: list[str] = []
    if index_path.is_file():
        for line in index_path.read_text(errors="replace").splitlines():
            try:
                rel = str(json.loads(line).get("path", ""))
            except json.JSONDecodeError:
                continue
            if rel.startswith("checkpoints/") and rel.endswith(".npz"):
                checkpoint_paths.append(rel)
    for rel in dict.fromkeys(checkpoint_paths[-64:]):
        if _download_artifact(session, run_id, rel, directory, cache):
            changed_paths.append(rel)
    return changed_paths


def sync_artifacts_local(run_id: str, *, final: bool = False) -> tuple[bool, list[str]]:
    directory, _state = job_state(run_id)
    purge_private_artifacts(directory)
    session = _state["session"]
    remote_manifest = _fetch_remote_manifest(session, run_id)
    cache = load_json(directory / "sync_manifest.json", {})
    changed_paths: list[str] = []
    if not remote_manifest:
        if final:
            changed_paths = _sync_essential_artifacts(session, run_id, directory, cache)
        atomic_json(directory / "sync_manifest.json", cache)
        return _has_training_outputs(directory) or bool(changed_paths), changed_paths
    for rel, remote_info in remote_manifest.items():
        # During training, skip checkpoints already preserved locally, but
        # always fetch newly appearing checkpoint paths before Colab pruning
        # can remove them.  This avoids replaying the full history while
        # retaining every checkpoint observed by the supervisor.
        if not final and rel not in ESSENTIAL_ARTIFACTS:
            if not rel.startswith("checkpoints/") or rel in cache:
                continue
        if _download_artifact(session, run_id, rel, directory, cache, remote_info=remote_info):
            changed_paths.append(rel)
    atomic_json(directory / "sync_manifest.json", cache)
    return True, changed_paths


def upload_artifacts_to_drive(run_id: str, rel_paths: list[str]) -> str | None:
    if not rel_paths:
        return None
    try:
        sys.path.insert(0, str(PROJECT / "scripts"))
        from upload_to_gdrive import upload_file_to_gdrive
        creds, token = drive_auth_paths()
        directory = JOBS / run_id
        sys.path.insert(0, str(ROOT))
        from syndiff_pipeline.forward_model.colab_emergency_upload import is_private_artifact

        for rel in sorted(rel_paths):
            if is_private_artifact(rel):
                continue
            upload_file_to_gdrive(
                directory / "artifacts" / rel,
                credentials_file=creds,
                token_file=token,
                subfolder_name=f"syndiff/runs/{run_id}",
            )
        return None
    except Exception as exc:
        return str(exc)


def finalize_abnormal_job(run_id: str, state: dict, reason: str) -> int:
    """Final sync and colab stop when the VM dies or training hangs."""
    directory = JOBS / run_id
    _supervisor_print(f"supervise: abnormal end ({reason}); final sync + colab stop")
    final_sync_ok = False
    try:
        final_sync_ok = sync_job(run_id, final=True)
    except Exception as exc:
        _supervisor_print(f"supervise: final sync failed: {exc}", error=True)
    state = load_json(directory / "controller.json", state)
    state["state"] = "failed"
    state["abnormal_end_reason"] = reason
    atomic_json(directory / "controller.json", state)
    session = state.get("session")
    if session:
        if stop_colab_session(session, endpoint=state.get("endpoint")):
            state = load_json(directory / "controller.json", state)
            state["session_stopped"] = True
            state.pop("session_stop_error", None)
            atomic_json(directory / "controller.json", state)
    # Failed/stalled runs can still leave a scientifically useful checkpoint.
    # Launch diagnostics after a verified final sync unless the operator opted out.
    if final_sync_ok:
        launch_postfit_diagnostics(run_id, load_json(directory / "controller.json", state))
    return 2


def sync_job(run_id: str, *, final: bool = False) -> bool:
    directory, state = job_state(run_id)
    local_ok, changed_paths = sync_artifacts_local(run_id, final=final)
    if not local_ok:
        state["sync_state"] = "failed"
        state["sync_error"] = "artifact download did not verify"
        state["last_sync_at"] = time.time()
        atomic_json(directory / "controller.json", state)
        return False
    final_param = _newest_parameter_artifact(directory) if final else None
    if final and final_param is None:
        state["final_parameter_error"] = "final sync found no parameter artifact"
        atomic_json(directory / "controller.json", state)
        return False
    cache = load_json(directory / "sync_manifest.json", {})
    upload_paths = sorted(cache.keys()) if final else changed_paths
    state.update(sync_state="verified", last_sync_at=time.time(), final_sync=bool(final))
    if final_param is not None:
        state["final_parameter_artifact"] = final_param
        state.pop("final_parameter_error", None)
    state.pop("sync_error", None)
    state.pop("drive_upload_error", None)
    # Publish the verified artifact-sync timestamp before rendering plots.
    # Plot generation can be much slower than downloading the logs/checkpoints;
    # it must not make the dashboard report an old sync or block supervisor
    # health accounting behind optional diagnostics.
    atomic_json(directory / "controller.json", state)
    history_path = directory / "artifacts" / "history.jsonl"
    train_log_path = directory / "artifacts" / "train.log"
    remote = load_json(directory / "remote_status.json", {})
    try:
        if (
            "history.jsonl" in changed_paths
            or "train.log" in changed_paths
            or (final and history_path.is_file())
        ):
            maybe_refresh_watch_plots(run_id, state, remote)
        # WCS/ePSF plots run in a detached 16 GiB-capped systemd worker.  The
        # supervisor only schedules it and never waits for its bundle load.
        if _params_plots_need_refresh(changed_paths):
            launch_live_params_plot_worker(run_id, state)
        # Keep LC/aperture diagnostics disabled live: that path still needs a
        # separate saved-flux snapshot rather than a full CPU recomputation.
        # if _lc_plots_need_refresh(changed_paths):
        #     maybe_refresh_lc_plots(run_id, state, remote)
    except Exception as exc:
        # Plotting is diagnostic-only; retain the verified sync state and let
        # the next supervisor cycle continue monitoring/training.
        state["plot_refresh_error"] = str(exc)
    atomic_json(directory / "controller.json", state)
    # Training artifacts stay on the login host under /astro/.../artifacts.
    # The Drive archive is only needed for publishing the input bundle; do not
    # upload live or final checkpoints/logs from this supervisor.  The remote
    # runner retains its independent emergency-upload path for VM-only failure
    # recovery.
    return True


def supervise(args) -> int:
    directory, _ = job_state(args.run_id)
    _supervisor_pid_path(directory).write_text(str(os.getpid()))
    exit_code = 0
    remote_fetch_failures = 0
    hang_stop_requested_at: float | None = None
    try:
        while True:
            _, state = job_state(args.run_id)
            try:
                remote, fresh = fetch_remote_status(directory, state)
            except Exception as exc:
                state.update(last_error=str(exc))
                atomic_json(directory / "controller.json", state)
                time.sleep(15)
                continue
            if fresh:
                remote_fetch_failures = 0
                if state.get("last_error"):
                    state.pop("last_error", None)
                    atomic_json(directory / "controller.json", state)
            else:
                remote_fetch_failures += 1
            now = time.time()
            # Lightweight breadcrumbs make host-side monitoring diagnosable
            # even when an external CLI call is slow.
            state["supervisor_last_loop_at"] = now
            state["supervisor_last_remote_state"] = remote.get("state") if remote else None
            atomic_json(directory / "controller.json", state)
            ages = _local_artifact_ages(directory, now)
            train_age = ages.get("train.log")
            remote_train_age = float(remote.get("train_log_mtime_age_s", 0.0) or 0.0)
            if (
                fresh
                and remote.get("state") == "running"
                and remote_train_age > SUPERVISOR_TRAIN_LOG_STALL_S
                and hang_stop_requested_at is None
            ):
                hang_stop_requested_at = now
                _supervisor_print(
                    f"supervise: remote train.log stale {remote_train_age:.0f}s while running; "
                    "touching STOP on VM",
                )
                try:
                    remote_exec(
                        state["session"],
                        f"touch /content/jobs/{shlex.quote(args.run_id)}/STOP",
                    )
                except (subprocess.CalledProcessError, RuntimeError) as exc:
                    _supervisor_print(f"supervise: STOP touch failed: {exc}", error=True)
                state = load_json(directory / "controller.json", state)
                state["hang_detected_at"] = now
                state["hang_train_log_age_s"] = train_age
                atomic_json(directory / "controller.json", state)
            if remote and now - state.get("last_sync_at", 0) >= 60:
                state["supervisor_last_sync_attempt_at"] = now
                atomic_json(directory / "controller.json", state)
                sync_job(args.run_id)
                _, state = job_state(args.run_id)
                remote, fresh = fetch_remote_status(directory, state)
                if fresh:
                    remote_fetch_failures = 0
            if remote.get("state") in TERMINAL:
                if sync_job(args.run_id, final=True):
                    state = load_json(directory / "controller.json", state)
                    state["state"] = remote["state"]
                    session = state["session"]
                    _supervisor_print(f"supervise: colab stop -s {session}")
                    if not stop_colab_session(session, endpoint=state.get("endpoint")):
                        state["session_stop_error"] = (
                            f"colab stop -s {session} failed; still listed in colab sessions"
                        )
                        atomic_json(directory / "controller.json", state)
                        time.sleep(30)
                        continue
                    _supervisor_print(f"supervise: verified {session} absent from colab sessions")
                    state.pop("session_stop_error", None)
                    state["session_stopped"] = True
                    atomic_json(directory / "controller.json", state)
                    # A terminal failed/stopped run can still have a valid final
                    # checkpoint.  Post-fit is based on verified final sync, not
                    # only on a successful terminal state.
                    launch_postfit_diagnostics(args.run_id, state)
                    return 0
                state = load_json(directory / "controller.json", state)
                state["sync_error"] = "final sync incomplete; will retry"
                atomic_json(directory / "controller.json", state)
                exit_code = 2
                time.sleep(30)
                continue
            if remote_fetch_failures >= SUPERVISOR_REMOTE_FAIL_LIMIT:
                # A failed local colab-cli probe is not evidence that a live VM
                # died.  The old path turned this transient local failure into a
                # final-sync-and-stop, killing healthy training.  Record degraded
                # monitoring and keep retrying; only a confirmed terminal state or
                # a post-STOP stall may end the run automatically.
                state["remote_status_degraded_at"] = state.get("remote_status_degraded_at", now)
                state["remote_status_fetch_failures"] = remote_fetch_failures
                atomic_json(directory / "controller.json", state)
                if mark_runtime_unreachable(
                    directory,
                    state,
                    failures=remote_fetch_failures,
                    error=state.get("last_error"),
                ):
                    _supervisor_print(
                        "supervise: runtime unreachable but controller endpoint remains assigned; "
                        "leaving billing untouched and requiring operator recovery",
                        error=True,
                    )
                    return 2
            else:
                state.pop("remote_status_degraded_at", None)
                state.pop("remote_status_fetch_failures", None)
                atomic_json(directory / "controller.json", state)
            abnormal_reason: str | None = None
            if (
                hang_stop_requested_at is not None
                and remote_train_age > SUPERVISOR_TRAIN_LOG_STALL_S + SUPERVISOR_HANG_STOP_GRACE_S
            ):
                abnormal_reason = "train_log_stall_after_stop"
            if abnormal_reason:
                return finalize_abnormal_job(args.run_id, state, abnormal_reason)
            time.sleep(15)
    finally:
        _clear_supervisor_pid(directory)
    return exit_code


def postfit_job(args) -> int:
    directory, state = job_state(args.run_id)
    pid = launch_postfit_diagnostics(args.run_id, state, force=args.force)
    if pid is None:
        raise SystemExit("postfit was not launched (see messages above)")
    print(pid)
    return 0


def verify_emergency(args) -> int:
    import io
    import zipfile
    from googleapiclient.http import MediaIoBaseDownload

    directory, _ = job_state(args.run_id)
    run_id = args.run_id
    zip_name = f"{run_id}_emergency_artifacts.zip"
    sys.path.insert(0, str(PROJECT / "scripts"))
    from upload_to_gdrive import (
        DEFAULT_PARENT_FOLDER_ID,
        ensure_subfolder,
        find_file_in_folder,
        get_gdrive_service,
    )
    creds, token = drive_auth_paths()
    service = get_gdrive_service(creds, token)
    folder_id = ensure_subfolder(service, DEFAULT_PARENT_FOLDER_ID, f"syndiff/runs/{run_id}")
    file_id = find_file_in_folder(service, zip_name, folder_id)
    if not file_id:
        raise SystemExit(f"emergency zip not found on Drive: syndiff/runs/{run_id}/{zip_name}")
    dest = directory / "artifacts" / zip_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with dest.open("wb") as stream:
        downloader = MediaIoBaseDownload(stream, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    with zipfile.ZipFile(dest) as archive:
        members = archive.namelist()
    required = ["status.json", "train.log", "artifact_manifest.json"]
    missing = [name for name in required if name not in members]
    if missing:
        raise SystemExit(f"emergency zip missing required members: {missing}")
    if not any(name.startswith("params_") and name.endswith(".npz") for name in members):
        raise SystemExit("emergency zip missing params_*.npz artifact")
    digest = sha256(dest)
    print(json.dumps({
        "run_id": run_id,
        "file_id": file_id,
        "sha256": digest,
        "local_path": str(dest),
        "member_count": len(members),
        "members": members,
    }, indent=2, sort_keys=True))
    print(f"gdown: gdown {file_id} -O {zip_name}")
    return 0


def show_status(args) -> int:
    ids = [args.run_id] if getattr(args, "run_id", None) else ([p.name for p in JOBS.iterdir()] if JOBS.exists() else [])
    for run_id in ids:
        directory, state = job_state(run_id)
        if state.get("state") == "orphaned_runtime":
            remote = load_json(directory / "remote_status.json", {})
        else:
            try:
                remote, _fresh = fetch_remote_status(directory, state)
            except Exception:
                remote = load_json(directory / "remote_status.json", {})
        now = time.time()
        remote = enrich_remote_from_local(directory, remote, now)
        status_at = remote.get("timestamp") or remote.get("heartbeat_at")
        status_age = now - float(status_at) if status_at else None
        print(json.dumps({
            **state,
            "remote": remote,
            "status_age_s": status_age,
            "heartbeat_age_s": status_age,
            "heartbeat_stale": remote_status_stale(remote, directory, now),
        }, sort_keys=True))
    return 0


def watch(args) -> int:
    use_json = getattr(args, "json", False) or not sys.stdout.isatty()
    flash: str | None = None
    while True:
        directory, state = job_state(args.run_id)
        if state.get("state") == "orphaned_runtime":
            remote = load_json(directory / "remote_status.json", {})
        else:
            try:
                remote, _fresh = fetch_remote_status(directory, state)
            except Exception:
                remote = load_json(directory / "remote_status.json", {})
        now = time.time()
        remote = enrich_remote_from_local(directory, remote, now)
        if use_json:
            status_at = remote.get("timestamp") or remote.get("heartbeat_at")
            status_age = now - float(status_at) if status_at else None
            print(json.dumps({
                **state,
                "remote": remote,
                "status_age_s": status_age,
                "heartbeat_age_s": status_age,
                "heartbeat_stale": remote_status_stale(remote, directory, now),
            }, sort_keys=True))
        else:
            _ensure_repo_on_path()
            from syndiff_pipeline.forward_model.training_history import load_history_jsonl

            history_rows = load_history_jsonl(directory / "artifacts" / "history.jsonl")
            term_width = shutil.get_terminal_size((100, 24)).columns
            print(
                render_watch_dashboard(
                    run_id=args.run_id,
                    controller=state,
                    remote=remote or {},
                    history_rows=history_rows,
                    term_width=term_width,
                    now=now,
                    flash=flash,
                ),
                end="",
                flush=True,
            )
            flash = None
        if remote.get("state") in TERMINAL:
            if not use_json:
                print(f"\nJob finished: {remote['state']}")
            return 0
        if state.get("state") == "orphaned_runtime":
            if not use_json:
                print("\nRuntime unavailable; endpoint ownership is retained in controller.json.")
            return 2
        if use_json:
            time.sleep(args.interval)
            continue
        key = _poll_watch_key(args.interval)
        if key == "p":
            directory, state = job_state(args.run_id)
            remote = load_json(directory / "remote_status.json", {})
            if refresh_local_plots(args.run_id, state, remote):
                atomic_json(directory / "controller.json", state)
                bits = ["plots refreshed"]
                for label, path_key, err_key in (
                    ("wcs", "watch_wcs_plot_path", "watch_wcs_plot_error"),
                    ("epsf", "watch_epsf_plot_path", "watch_epsf_plot_error"),
                    ("s2", "watch_epsf_stage2_plot_path", "watch_epsf_stage2_plot_error"),
                    ("ac/w0", "watch_ac_w0_plot_path", "watch_ac_w0_plot_error"),
                    ("lc", "watch_lc_grid_plot_path", "watch_lc_grid_plot_error"),
                ):
                    if state.get(path_key):
                        bits.append(f"{label} ok")
                    elif state.get(err_key):
                        bits.append(f"{label}: {state[err_key][:48]}")
                    elif int(remote.get("stage", 0) or 0) >= (
                        2 if label in ("epsf", "ac/w0", "lc") else 1
                    ):
                        bits.append(f"{label}: skipped")
                flash = " · ".join(bits)
            else:
                if resolve_job_bundle_path(state) is None:
                    flash = "plot refresh failed — no local bundle (missing bundle_path)"
                else:
                    flash = "plot refresh failed (missing or empty history.jsonl?)"
        elif key == "s":
            if sync_job(args.run_id):
                flash = "synced artifacts from remote"
            else:
                flash = "sync failed"
    return 0


def plot_job(args) -> int:
    directory, state = job_state(args.run_id)
    remote = load_json(directory / "remote_status.json", {})
    if not refresh_local_plots(args.run_id, state, remote):
        raise SystemExit("could not render watch plots (missing or empty history.jsonl)")
    atomic_json(directory / "controller.json", state)
    for key in (
        "watch_plot_path",
        "watch_log_plot_path",
        "watch_wcs_plot_path",
        "watch_epsf_plot_path",
        "watch_epsf_stage2_plot_path",
        "watch_ac_w0_plot_path",
        "watch_lc_grid_plot_path",
    ):
        if key in state:
            print(state[key])
    return 0


def _load_train_log_text(directory: Path, state: dict, run_id: str, *, remote: bool) -> str:
    local_log = directory / "artifacts" / "train.log"
    if not remote and local_log.is_file():
        return local_log.read_text(errors="replace")
    try:
        return remote_exec(
            state["session"],
            f"cat /content/jobs/{shlex.quote(run_id)}/train.log",
            timeout=300,
        )
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        if local_log.is_file():
            return local_log.read_text(errors="replace")
        raise SystemExit(f"could not read train.log: {exc}") from exc


def metrics_job(args) -> int:
    """Print a scrollable table: time, stage, step, loss, med_chi2_red."""
    _ensure_repo_on_path()
    from syndiff_pipeline.forward_model.training_history import (
        format_metrics_table,
        load_history_jsonl,
        metrics_from_history,
        parse_train_log_metrics,
    )

    directory, state = job_state(args.run_id)
    run_id = args.run_id
    text = _load_train_log_text(directory, state, run_id, remote=args.remote)
    rows = parse_train_log_metrics(text)
    time_header = "time"
    if not rows:
        history = directory / "artifacts" / "history.jsonl"
        rows = metrics_from_history(load_history_jsonl(history))
        time_header = "elapsed"
        if not rows:
            raise SystemExit("no training metrics found in train.log or history.jsonl")
    if args.stage is not None:
        rows = [r for r in rows if int(r["stage"]) == args.stage]
    if args.tail is not None:
        rows = rows[-args.tail :]
    print(format_metrics_table(rows, time_header=time_header))
    return 0


def tail(args) -> int:
    directory, state = job_state(args.run_id)
    local_log = directory / "artifacts" / "train.log"
    remote = load_json(directory / "remote_status.json", {})
    remote_state = remote.get("state")
    prefer_remote = remote_state not in TERMINAL and remote_state is not None
    if prefer_remote:
        try:
            print(
                remote_exec(
                    state["session"],
                    f"tail -n {args.lines} /content/jobs/{shlex.quote(args.run_id)}/train.log",
                    timeout=90,
                ),
                end="",
            )
            return 0
        except (subprocess.CalledProcessError, RuntimeError):
            pass
    if local_log.is_file():
        lines = local_log.read_text(errors="replace").splitlines()
        print("\n".join(lines[-args.lines:]))
        return 0
    try:
        print(
            remote_exec(
                state["session"],
                f"tail -n {args.lines} /content/jobs/{shlex.quote(args.run_id)}/train.log",
                timeout=90,
            ),
            end="",
        )
    except subprocess.CalledProcessError:
        raise SystemExit(
            "remote tail failed (colab console may hold the session); "
            f"use `python scripts/colab_job.py status {args.run_id}` or "
            f"close the console and retry"
        ) from None
    return 0


def stop(args) -> int:
    directory, state = job_state(args.run_id)
    session = state["session"]
    endpoint = state.get("endpoint")
    if not args.force:
        print("syncing current artifacts before requesting graceful stop", flush=True)
        pre_stop_ok = False
        try:
            pre_stop_ok = sync_job(args.run_id)
            if not pre_stop_ok:
                print("pre-stop sync did not verify training outputs; continuing to request stop", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"pre-stop sync failed ({exc}); continuing to request stop", file=sys.stderr, flush=True)
        state = load_json(directory / "controller.json", state)
        state["pre_stop_sync_at"] = time.time()
        state["pre_stop_sync_ok"] = pre_stop_ok
        atomic_json(directory / "controller.json", state)
        try:
            remote_exec(session, f"touch /content/jobs/{shlex.quote(args.run_id)}/STOP")
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            # A failed graceful request must not silently become an endpoint
            # release: a runtime proxy can vanish while its L4 is still billed.
            if mark_runtime_unreachable(
                directory,
                load_json(directory / "controller.json", state),
                failures=1,
                error=str(exc),
            ):
                raise SystemExit(
                    "graceful STOP could not reach the runtime; the recorded endpoint remains assigned. "
                    f"Inspect local artifacts, then explicitly authorize `stop {args.run_id} --force` to release it."
                ) from None
            raise
        print("graceful stop requested; supervisor will sync before ending billing", flush=True)
    else:
        sup_pid = _supervisor_running(directory)
        if sup_pid is not None:
            try:
                os.kill(sup_pid, 9)
            except OSError:
                pass
            _clear_supervisor_pid(directory)
        ok, diag = stop_assignment_verified(
            session,
            endpoint=endpoint,
            colab_binary=colab_binary,
            verify_session_stopped=lambda name, endpoint=None: verify_session_stopped(name),
            force_release=True,
        )
        state["state"] = "force_stopped"
        state["session_stop_diagnostics"] = redact_obj(diag)
        if not ok:
            state["session_stop_error"] = "verified stop failed after --force"
        else:
            state["session_stopped"] = True
            state.pop("session_stop_error", None)
        atomic_json(directory / "controller.json", state)
    return 0


def advance_stage(args) -> int:
    """Request a checkpoint-safe advance from the current stage."""
    directory, state = job_state(args.run_id)
    if state.get("state") != "running" or not state.get("session"):
        raise SystemExit(f"cannot advance non-running job {args.run_id}: {state.get('state')}")
    remote_exec(state["session"], f"touch /content/jobs/{shlex.quote(args.run_id)}/ADVANCE_STAGE")
    state["advance_requested_at"] = time.time()
    atomic_json(directory / "controller.json", state)
    print("stage advance requested; current step will checkpoint before continuing", flush=True)
    return 0


def recover(args) -> int:
    directory, state = job_state(args.run_id)
    paths = remote_paths(args.run_id)
    local_paths = local_run_paths(directory, args.run_id)
    ok, proof = validate_runner_proof(
        state["session"],
        args.run_id,
        state,
        colab_download=colab_download,
        remote_exec=remote_exec,
        paths=paths,
        local_paths=local_paths,
    )
    if not ok:
        raise SystemExit(proof.get("failure") or "recover refused: runner proof failed")
    if state.get("state") == "provisioning":
        state["state"] = "running"
        state["runner_proof"] = redact_obj(proof)
        atomic_json(directory / "controller.json", state)
    pid = _start_supervisor(args.run_id)
    print(f"reattached {args.run_id} to {state['session']} (supervisor pid={pid})")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor"); d.add_argument("--install", action="store_true"); d.add_argument("--min-disk-gb", type=float, default=2); d.set_defaults(func=doctor)
    pub = sub.add_parser("publish"); pub.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE); pub.add_argument("--archive", type=Path, default=DEFAULT_ZIP); pub.set_defaults(func=publish)
    s = sub.add_parser("submit")
    s.add_argument("--mode", choices=("probe", "full"), required=True)
    s.add_argument("--run-id")
    s.add_argument("--archive", type=Path, default=DEFAULT_ZIP)
    s.add_argument("--gpu", choices=("T4", "L4"), default="L4")
    s.add_argument("--cpu", action="store_true",
                   help="use Colab CPU runtime (omit --gpu on colab new)")
    s.add_argument("--post-terminal-grace-s", type=float, default=300.0,
                   help="seconds before emergency Drive upload after terminal state")
    s.add_argument("--expect-frames", type=int, default=590)
    s.add_argument("--steps-per-stage", help="comma-separated stage 1,2,3 step counts")
    s.add_argument("--first-stage", type=int, choices=(1, 2, 3), default=1)
    s.add_argument("--bootstrap-init-params", type=Path,
                   help="local params npz uploaded as init for --first-stage when > 1")
    s.add_argument("--epsf-lr-scale", type=float, default=1.0)
    s.add_argument("--chroma", action="store_true",
                   help="enable the chromatic PSF term (12 params on a 2x2 node "
                        "grid, driven by per-star Gaia BP-RP; needs a bundle with "
                        "bp_rp; unfrozen at stage 3 only)")
    s.add_argument("--chroma-lr-scale", type=float, default=1.0,
                   help="learning-rate multiplier for the chromatic leaves")
    s.add_argument("--chroma-affine", action="store_true",
                   help="C1: extend --chroma with per-node anisotropic-stretch and "
                        "45-degree-shear leaves (chroma_aniso/chroma_shear, +8 params "
                        "on a 2x2 grid, same train_chroma bucket/lr as --chroma). "
                        "Requires --chroma. Only needed to start these leaves from "
                        "scratch (zero) with no checkpoint; resuming from a checkpoint "
                        "that already carries them does not require this flag.")
    s.add_argument("--chroma-kurt", action="store_true",
                   help="C1 optional: also add the flux-neutral kurtosis leaf "
                        "chroma_kurt (+4 params on a 2x2 grid). Requires --chroma-affine.")
    s.add_argument("--freeze-epsf-modes", action="store_true",
                   help="stage 3: freeze epsf_modes' shape (optax.set_to_zero()); "
                        "w_coeff (its per-frame amplitude) still trains")
    s.add_argument("--stamp-pedestal", action="store_true",
                   help="Task M4: solve one additive per-group-per-frame pedestal "
                        "jointly with flux (K+1-unknown weighted LS). Default off, "
                        "bit-identical; forwarded to the trainer only when set.")
    s.add_argument("--profile-w", action="store_true",
                   help="Task PW: solve the temporal ePSF mode amplitudes w_k(t) "
                        "in closed form per frame, jointly with every stamp flux "
                        "(and pedestal), instead of training the w_coeff spline "
                        "(which is then frozen and leaves the model). Default off, "
                        "bit-identical; forwarded to the trainer only when set.")
    s.add_argument("--profile-w-iters", type=int, default=None,
                   help="Gauss-Newton iterations of task PW's joint flux/amplitude "
                        "solve. Default: the trainer's 2.")
    s.add_argument("--w-lr-scale", type=float, default=None,
                   help="learning-rate multiplier for w_coeff (the temporal ePSF "
                        "mode amplitude). Its natural scale is ~1e-5 against "
                        "epsf_base_raw's ~14, so sharing the ePSF rate moved it by "
                        "5x its own value per step and it never converged; see "
                        "dev/forward_epsf_wcs/docs/TEMPORAL_RESIDUAL_ROOT_CAUSE_20260906.md. "
                        "Default: the trainer's fit.W_LR_SCALE_DEFAULT.")
    s.add_argument("--w-spatial", action="store_true",
                   help="T3: give the temporal ePSF mode's amplitude w_k(t) a "
                        "per-node field w_k(t, x, y) instead of one value shared "
                        "by the whole field (leaf grows from (K, n_basis) to "
                        "(K, n_rows, n_cols, n_basis)). Default off, "
                        "bit-identical to the pre-T3 global-w model.")
    s.add_argument("--lr-per-stage", default="1e-2,3e-4,1e-4",
                   help="comma-separated stage 1,2,3 learning rates")
    s.add_argument("--stage2-freeze-wcs-steps", type=int, default=20)
    s.add_argument("--huber-delta", type=float, default=3.0)
    s.add_argument("--lambda-pixel-lap", type=float, default=1e-2)
    s.add_argument("--lambda-lap", type=float, default=1e-3,
                   help="Adjacent-node P_base smoothness (LossWeights.lambda_lap)")
    s.add_argument("--lambda-smooth-wcs", type=float, default=1e-4)
    s.add_argument("--lambda-smooth-w", type=float, default=1e-3)
    s.add_argument("--stamp-chunk", type=int)
    s.add_argument("--min-ram-gb", type=float, default=12)
    s.add_argument("--min-available-ram-gb", type=float, default=10)
    s.add_argument("--high-mem", action="store_true", help="request T4 high-memory (ignored on L4)")
    s.add_argument("--isolate-stages", action="store_true", help="run each stage in a fresh JAX process")
    s.add_argument("--reject-mode", choices=("audit", "two-level", "standardized", "hysteresis", "legacy", "static"), default="hysteresis")
    s.add_argument("--reject-every", type=int, default=20)
    s.add_argument("--early-stop-patience", type=int, default=20,
                   help="stop a stage when relative loss change stays below --early-stop-tol "
                        "over this many consecutive --log-every checks (0 disables)")
    s.add_argument("--early-stop-tol", type=float, default=1e-4)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--no-supervisor", action="store_true")
    s.add_argument("--deployment", type=Path, default=DEFAULT_DEPLOYMENT_DIR,
                   help="site config dir containing deployment.yaml (Discord webhook)")
    s.add_argument("--discord-webhook-url", default=None,
                   help="override discord_webhook_url from deployment.yaml")
    s.add_argument("--no-discord", action="store_true",
                   help="disable Colab-side Discord notification for this submit")
    s.add_argument("--bundle", type=Path,
                   help="local fit_bundle directory (default: from publish manifest)")
    s.add_argument("--no-postfit", action="store_true",
                   help="do not auto-launch run_postfit diagnostics after terminal final sync")
    s.add_argument("--keep-vm-on-failure", action="store_true",
                   help="dangerous: leave VM allocated when submit fails (debug only)")
    s.set_defaults(func=submit)
    dg = sub.add_parser("diagnose", help="safe remote snapshot for a run (human or --json)")
    dg.add_argument("run_id", nargs="?")
    dg.add_argument("--json", action="store_true")
    dg.set_defaults(func=diagnose_job)
    pw = sub.add_parser("_provision_watchdog")
    pw.add_argument("run_id")
    pw.add_argument("deadline_s")
    pw.set_defaults(func=provision_watchdog)
    scw = sub.add_parser("_submit_crash_watchdog")
    scw.add_argument("run_id")
    scw.set_defaults(func=submit_crash_watchdog)
    pf = sub.add_parser("postfit", help="launch run_postfit on synced artifacts")
    pf.add_argument("run_id", nargs="?")
    pf.add_argument("--force", action="store_true", help="relaunch even if already started")
    pf.set_defaults(func=postfit_job)
    lp = sub.add_parser("_live_params_plot")
    lp.add_argument("run_id")
    lp.set_defaults(func=live_params_plot_worker)
    ve = sub.add_parser("verify-emergency", help="download emergency artifact zip from Drive and validate")
    ve.add_argument("run_id", nargs="?")
    ve.set_defaults(func=verify_emergency)
    for name, func in (("watch", watch), ("tail", tail), ("sync", lambda a: 0 if sync_job(a.run_id) else 2), ("stop", stop), ("recover", recover), ("_supervise", supervise)):
        q = sub.add_parser(name)
        if name != "_supervise":
            q.add_argument("run_id", nargs="?")
        else:
            q.add_argument("run_id")
        q.set_defaults(func=func)
        if name == "watch":
            q.add_argument("--interval", type=float, default=5)
            q.add_argument("--json", action="store_true", help="emit JSON status each tick (legacy)")
        if name == "tail": q.add_argument("--lines", type=int, default=50)
        if name == "stop": q.add_argument("--force", action="store_true")
    pl = sub.add_parser("plot", help="render loss/chi2 live plot PNG from synced history.jsonl")
    pl.add_argument("run_id", nargs="?")
    pl.set_defaults(func=plot_job)
    mt = sub.add_parser(
        "metrics",
        help="scrollable table of time, stage, step, loss, chi2_red from train.log",
    )
    mt.add_argument("run_id", nargs="?")
    mt.add_argument("--tail", type=int, default=None, help="only last N steps")
    mt.add_argument("--stage", type=int, default=None, help="filter to one stage")
    mt.add_argument(
        "--remote",
        action="store_true",
        help="read train.log from Colab VM (default: local synced copy)",
    )
    mt.set_defaults(func=metrics_job)
    adv = sub.add_parser("advance", help="checkpoint current stage and continue with the next stage")
    adv.add_argument("run_id")
    adv.set_defaults(func=advance_stage)
    st = sub.add_parser("status"); st.add_argument("run_id", nargs="?"); st.set_defaults(func=show_status)
    return p


_RUN_ID_OPTIONAL_COMMANDS = frozenset({
    "watch", "tail", "sync", "stop", "recover", "plot", "metrics", "postfit",
    "verify-emergency", "diagnose",
})


if __name__ == "__main__":
    _args = parser().parse_args()
    if _args.command in _RUN_ID_OPTIONAL_COMMANDS and getattr(_args, "run_id", None) is None:
        _args.run_id = resolve_run_id(None)
    raise SystemExit(_args.func(_args))
