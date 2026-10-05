# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Parse ``history.jsonl`` and render CLI sparklines (stdlib only)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_TRAIN_TS = re.compile(r"^\[(?P<time>[^\]]+)\]")
_TRAIN_STEP = re.compile(
    r"stage (?P<stage>\d+) step\s+(?P<step>\d+)/\d+\s+loss=(?P<loss>[-+\d.eE]+)"
)
_TRAIN_REJECT = re.compile(
    r"stage (?P<stage>\d+) reject refresh @ step (?P<step>\d+).*"
    r"med_chi2_red=(?P<chi2>[-+\d.eE]+)"
)


def load_history_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def training_rows(rows: list[dict]) -> list[dict]:
    """One row per (stage, step) with loss; prefer rows that include med_chi2_red."""
    by_key: dict[tuple[int, int], dict] = {}
    for row in rows:
        if "loss" not in row:
            continue
        key = (int(row.get("stage", 0)), int(row["step"]))
        prev = by_key.get(key)
        if prev is None or ("med_chi2_red" in row and "med_chi2_red" not in prev):
            by_key[key] = row
    return [by_key[k] for k in sorted(by_key)]


def chi2_series(rows: list[dict], stage: int | None = None) -> list[tuple[int, int, float]]:
    """(stage, step, med_chi2_red) at reject refreshes and logged training rows."""
    out: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        chi2 = row.get("med_chi2_red")
        if chi2 is None:
            continue
        try:
            value = float(chi2)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        st = int(row.get("stage", 0))
        if stage is not None and st != stage:
            continue
        step = int(row["step"])
        key = (st, step)
        if key in seen:
            continue
        seen.add(key)
        out.append((st, step, value))
    return sorted(out)


def parse_steps_per_stage(controller: dict) -> list[int]:
    raw = controller.get("steps_per_stage")
    if not raw:
        return []
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def stage_budgets(controller: dict) -> list[tuple[int, int]]:
    """(stage_number, n_steps) for each slot in ``steps_per_stage``.

    The comma-separated triple is always indexed by absolute stage number
    (stage 1 → first value, stage 2 → second, …), matching ``train_loop`` and
    ``isolated_stage_runner``.  ``first_stage`` only selects which stage runs;
    it does not shift this mapping.
    """
    steps = parse_steps_per_stage(controller)
    if not steps:
        return []
    return [(index + 1, n) for index, n in enumerate(steps)]


def global_step_index(stage: int, step: int, budgets: list[tuple[int, int]]) -> int:
    offset = 0
    for st, n in budgets:
        if st < stage:
            offset += n
        elif st == stage:
            return offset + step
    return step


def overall_progress(
    stage: int,
    step: int,
    budgets: list[tuple[int, int]],
) -> tuple[float, int, int]:
    """Return (fraction 0..1, completed_steps, total_steps)."""
    if not budgets:
        return 0.0, 0, 0
    total = sum(n for _, n in budgets)
    done = 0
    for st, n in budgets:
        if st < stage:
            done += n
        elif st == stage:
            done += min(step + 1, n)
            break
    return (done / total if total else 0.0), done, total


def rows_for_stage(rows: list[dict], stage: int) -> list[dict]:
    return [r for r in rows if int(r.get("stage", 0)) == stage]


def stage_step_span(rows: list[dict], stage: int) -> int:
    sub = rows_for_stage(rows, stage)
    if not sub:
        return 1
    return max(int(r["step"]) for r in sub) + 1


def global_step_array(rows: list[dict], budgets: list[tuple[int, int]] | None = None) -> list[int]:
    if budgets:
        return [
            global_step_index(int(r["stage"]), int(r["step"]), budgets)
            for r in rows
        ]
    offsets: dict[int, int] = {}
    cursor = 0
    for st in sorted({int(r["stage"]) for r in rows}):
        offsets[st] = cursor
        cursor += stage_step_span(rows, st)
    return [offsets[int(r["stage"])] + int(r["step"]) for r in rows]


def sparkline(
    values: list[float],
    width: int = 50,
    *,
    log_scale: bool = False,
) -> str:
    if not values or width <= 0:
        return ""
    series = list(values[-width:])
    if log_scale:
        mapped = []
        for v in series:
            if v <= 0 or not math.isfinite(v):
                mapped.append(float("nan"))
            else:
                mapped.append(math.log10(v))
        series = [x for x in mapped if math.isfinite(x)]
        if not series:
            return _SPARK_CHARS[0] * min(len(values), width)
    lo = min(series)
    hi = max(series)
    if hi == lo:
        return _SPARK_CHARS[len(_SPARK_CHARS) // 2] * len(series)
    n = len(_SPARK_CHARS) - 1
    chars = []
    for v in series:
        if not math.isfinite(v):
            chars.append(" ")
            continue
        idx = int(round((v - lo) / (hi - lo) * n))
        chars.append(_SPARK_CHARS[max(0, min(n, idx))])
    return "".join(chars)


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m ago"


def pct_change(old: float, new: float) -> float | None:
    if not math.isfinite(old) or not math.isfinite(new) or abs(old) < 1e-12:
        return None
    return 100.0 * (new - old) / abs(old)


def parse_train_log_metrics(text: str) -> list[dict]:
    """Extract (time, stage, step, loss, med_chi2_red) rows from ``train.log`` text."""
    chi2_at: dict[tuple[int, int], float] = {}
    for line in text.splitlines():
        match = _TRAIN_REJECT.search(line)
        if match:
            key = (int(match.group("stage")), int(match.group("step")))
            chi2_at[key] = float(match.group("chi2"))

    rows: list[dict] = []
    last_chi2: dict[int, float] = {}
    for line in text.splitlines():
        step_match = _TRAIN_STEP.search(line)
        if not step_match:
            continue
        stage = int(step_match.group("stage"))
        step = int(step_match.group("step"))
        key = (stage, step)
        if key in chi2_at:
            last_chi2[stage] = chi2_at[key]
        ts_match = _TRAIN_TS.match(line)
        rows.append({
            "time": ts_match.group("time") if ts_match else "",
            "stage": stage,
            "step": step,
            "loss": float(step_match.group("loss")),
            "med_chi2_red": last_chi2.get(stage),
        })
    return rows


def metrics_from_history(rows: list[dict]) -> list[dict]:
    """Fallback metrics table from ``history.jsonl`` (``elapsed_s`` in the time column)."""
    out: list[dict] = []
    for row in training_rows(rows):
        elapsed = row.get("elapsed_s")
        time_col = f"+{float(elapsed):.1f}s" if elapsed is not None else ""
        chi2 = row.get("med_chi2_red")
        out.append({
            "time": time_col,
            "stage": int(row["stage"]),
            "step": int(row["step"]),
            "loss": float(row["loss"]),
            "med_chi2_red": float(chi2) if chi2 is not None else None,
        })
    return out


def format_metrics_table(rows: list[dict], *, time_header: str = "time") -> str:
    if not rows:
        return f"{time_header:>19}  st   step         loss    chi2_red\n(no training steps found)"
    header = f"{time_header:>19}  st   step         loss    chi2_red"
    lines = [header, "-" * len(header)]
    for row in rows:
        chi2 = row.get("med_chi2_red")
        if chi2 is not None and math.isfinite(float(chi2)):
            chi2_s = f"{float(chi2):10.3f}"
        else:
            chi2_s = f"{'':>10}"
        lines.append(
            f"{str(row.get('time', '')):>19}  {int(row['stage']):2d}  "
            f"{int(row['step']):5d}  {float(row['loss']):10.4f}  {chi2_s}"
        )
    return "\n".join(lines)

