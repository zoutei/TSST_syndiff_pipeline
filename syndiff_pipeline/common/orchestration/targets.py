"""Load template pipeline targets from normalized CSV or SN event catalog."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import math

_TESS_COVERAGE_RE = re.compile(r"S(\d+)C(\d+)D(\d+)", re.IGNORECASE)

NORMALIZED_HEADER = frozenset(
    {"sector", "camera", "ccd", "target_ra", "target_dec", "target_name", "enabled"}
)
NORMALIZED_HEADER_MIN = frozenset(
    {"sector", "camera", "ccd", "target_name", "enabled"}
)
EVENT_HEADER = frozenset({"id", "ra", "dec", "tess_coverage"})
SCC_HEADER = frozenset({"sector", "camera", "ccd"})
_SCC_LABEL_RE = re.compile(r"^s(\d+)_c(\d+)_k(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Target:
    """Target."""
    sector: int
    camera: int
    ccd: int
    target_ra: float
    target_dec: float
    target_name: str
    enabled: bool = True

    def scc_key(self) -> str:
        """Scc key.
        
        Returns
        -------
        str"""
        return f"{self.sector}/{self.camera}/{self.ccd}"

    def event_name(self) -> str:
        """Sanitized event name (same rules as the name suffix in ``label()``)."""
        return re.sub(r"[^\w.-]+", "_", self.target_name.strip())

    def scc_label(self) -> str:
        """SCC-only label without event name suffix."""
        return f"s{self.sector:04d}_c{self.camera}_k{self.ccd}"

    def label(self) -> str:
        """Label.

        Returns
        -------
        str"""
        scc = self.scc_label()
        event = self.event_name()
        if event == scc:
            return scc
        return f"{scc}_{event}"

    def diff_scheduler_key(self) -> str:
        """Scheduler key for event/SCC split: ``{event_name}/{scc_label}``."""
        return f"{self.event_name()}/{self.scc_label()}"

    def coords_missing(self) -> bool:
        """True when seed RA/Dec were not provided in the targets CSV."""
        return not (
            math.isfinite(self.target_ra)
            and math.isfinite(self.target_dec)
        )


def parse_tess_coverage(value: str) -> List[tuple[int, int, int]]:
    """Parse ``S20C3D3`` or ``S44C2D1; S45C1D4`` into SCC triples."""
    text = str(value or "").strip()
    if not text:
        return []
    out: List[tuple[int, int, int]] = []
    for part in re.split(r"[;,]", text):
        part = part.strip()
        if not part:
            continue
        m = _TESS_COVERAGE_RE.search(part)
        if not m:
            raise ValueError(f"Invalid tess_coverage token: {part!r}")
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def _parse_bool(value: str | None, default: bool = True) -> bool:
    """Parse bool.
    
    Parameters
    ----------
    value : str | None
    default : bool, optional, default ``True``
    
    Returns
    -------
    bool"""
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _target_name_from_event_id(event_id: str) -> str:
    """Target name from event id.
    
    Parameters
    ----------
    event_id : str
    
    Returns
    -------
    str"""
    name = str(event_id or "").strip()
    if name.upper().startswith("SN "):
        name = name[3:].strip()
    return name or "unknown"


def _parse_optional_coord(value: str | None) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def _load_normalized_rows(rows: Sequence[dict]) -> List[Target]:
    """Load normalized rows.
    
    Parameters
    ----------
    rows : Sequence[dict]
    
    Returns
    -------
    List[Target]"""
    out: List[Target] = []
    for row in rows:
        if not _parse_bool(row.get("enabled"), default=True):
            continue
        out.append(
            Target(
                sector=int(row["sector"]),
                camera=int(row["camera"]),
                ccd=int(row["ccd"]),
                target_ra=_parse_optional_coord(row.get("target_ra")),
                target_dec=_parse_optional_coord(row.get("target_dec")),
                target_name=str(row["target_name"]).strip(),
                enabled=True,
            )
        )
    return out


def _load_event_rows(rows: Sequence[dict]) -> List[Target]:
    """Load event rows.
    
    Parameters
    ----------
    rows : Sequence[dict]
    
    Returns
    -------
    List[Target]"""
    out: List[Target] = []
    for row in rows:
        name = _target_name_from_event_id(row.get("id", row.get("ID", "")))
        ra = float(row.get("ra", row.get("RA")))
        dec = float(row.get("dec", row.get("DEC")))
        coverages = parse_tess_coverage(row.get("tess_coverage", row.get("TESS_COVERAGE", "")))
        if not coverages:
            raise ValueError(f"Event {name!r} has no tess_coverage")
        for sector, camera, ccd in coverages:
            out.append(
                Target(
                    sector=sector,
                    camera=camera,
                    ccd=ccd,
                    target_ra=ra,
                    target_dec=dec,
                    target_name=name,
                    enabled=True,
                )
            )
    return out


def _read_csv_rows(path: Path) -> tuple[List[dict], frozenset[str]]:
    """Read csv rows.
    
    Parameters
    ----------
    path : Path
    
    Returns
    -------
    tuple[List[dict], frozenset[str]]"""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        fields = frozenset(f.strip().lower() for f in reader.fieldnames)
        rows = [{k.strip().lower(): v for k, v in row.items()} for row in reader]
    return rows, fields


def load_sccs(path: str | Path) -> List[Target]:
    """Load SCC references from ``sector,camera,ccd[,enabled]`` CSV."""
    p = Path(path).expanduser().resolve()
    rows, fields = _read_csv_rows(p)
    if EVENT_HEADER.issubset(fields) and "sector" not in fields:
        raise ValueError(
            f"SCC CSV cannot use event catalog headers in {p}; "
            "use sector,camera,ccd[,enabled]."
        )
    if not SCC_HEADER.issubset(fields):
        missing = sorted(SCC_HEADER - fields)
        raise ValueError(f"SCC CSV missing columns {missing} in {p}")
    seen: set[tuple[int, int, int]] = set()
    out: List[Target] = []
    for row in rows:
        sector = int(row["sector"])
        camera = int(row["camera"])
        ccd = int(row["ccd"])
        key = (sector, camera, ccd)
        if key in seen:
            continue
        seen.add(key)
        if not _parse_bool(row.get("enabled"), default=True):
            continue
        label = f"s{sector:04d}_c{camera}_k{ccd}"
        out.append(
            Target(
                sector=sector,
                camera=camera,
                ccd=ccd,
                target_ra=float("nan"),
                target_dec=float("nan"),
                target_name=label,
                enabled=True,
            )
        )
    return out


def scc_from_cli(sector: int | str, camera: int | str, ccd: int | str) -> Target:
    """Build one SCC reference from CLI ``sector camera ccd`` arguments."""
    s, c, k = int(sector), int(camera), int(ccd)
    label = f"s{s:04d}_c{c}_k{k}"
    return Target(
        sector=s,
        camera=c,
        ccd=k,
        target_ra=float("nan"),
        target_dec=float("nan"),
        target_name=label,
        enabled=True,
    )


def write_sccs(path: str | Path, targets: Sequence[Target]) -> Path:
    """Write ``sector,camera,ccd[,enabled]`` SCC CSV for a run directory."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sector", "camera", "ccd", "enabled"]
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for t in targets:
            writer.writerow(
                {
                    "sector": t.sector,
                    "camera": t.camera,
                    "ccd": t.ccd,
                    "enabled": "true" if t.enabled else "false",
                }
            )
    return p


def load_targets(path: str | Path) -> List[Target]:
    """Load targets from normalized CSV or SN event catalog CSV."""
    p = Path(path).expanduser().resolve()
    rows, fields = _read_csv_rows(p)
    if EVENT_HEADER.issubset(fields) and "sector" not in fields:
        return _load_event_rows(rows)
    if NORMALIZED_HEADER.issubset(fields) or NORMALIZED_HEADER_MIN.issubset(fields):
        return _load_normalized_rows(rows)
    missing_min = sorted(NORMALIZED_HEADER_MIN - fields)
    missing_evt = sorted(EVENT_HEADER - fields)
    raise ValueError(
        f"Unrecognized CSV header in {p}. "
        f"Need normalized columns (missing {missing_min}) or event catalog (missing {missing_evt})."
    )


def write_normalized_targets(path: str | Path, targets: Sequence[Target]) -> Path:
    """Write orchestrator-compatible ``targets.csv`` for a run directory."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sector",
        "camera",
        "ccd",
        "target_ra",
        "target_dec",
        "target_name",
        "enabled",
    ]
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for t in targets:
            writer.writerow(
                {
                    "sector": t.sector,
                    "camera": t.camera,
                    "ccd": t.ccd,
                    "target_ra": t.target_ra,
                    "target_dec": t.target_dec,
                    "target_name": t.target_name,
                    "enabled": "true" if t.enabled else "false",
                }
            )
    return p


def parse_scc(scc: str) -> tuple[int, int, int]:
    """Parse ``sector,camera,ccd`` or ``sector/camera/ccd``."""
    parts = re.split(r"[,/]", scc.strip())
    if len(parts) != 3:
        raise ValueError(f"Expected SCC as S,C,K got {scc!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _find_target_matches(
    targets: Iterable[Target],
    *,
    event_name: str | None = None,
    scc_label: str | None = None,
    sector: int | None = None,
    camera: int | None = None,
    ccd: int | None = None,
) -> List[Target]:
    matches: List[Target] = []
    for t in targets:
        if event_name is not None and t.event_name() != event_name:
            continue
        if scc_label is not None and t.scc_label() != scc_label:
            continue
        if sector is not None and t.sector != sector:
            continue
        if camera is not None and t.camera != camera:
            continue
        if ccd is not None and t.ccd != ccd:
            continue
        matches.append(t)
    return matches


def _raise_ambiguous_target(query: str, matches: Sequence[Target]) -> None:
    names = ", ".join(sorted(t.label() for t in matches))
    raise KeyError(
        f"Query {query!r} is ambiguous ({len(matches)} targets: {names}); "
        "use the full target label or event/scc form"
    )


def find_target(targets: Iterable[Target], scc: str) -> Target:
    """Find target by label, ``event/scc``, SCC label, or numeric SCC key."""
    key = scc.strip()
    target_list = list(targets)

    for t in target_list:
        if t.label() == key:
            return t

    if "/" in key:
        event_part, scc_part = key.split("/", 1)
        matches = _find_target_matches(
            target_list, event_name=event_part, scc_label=scc_part
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            _raise_ambiguous_target(key, matches)
        raise KeyError(f"No target for event/SCC query {key!r}")

    if _SCC_LABEL_RE.fullmatch(key):
        matches = _find_target_matches(target_list, scc_label=key)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            _raise_ambiguous_target(key, matches)
        raise KeyError(f"No target for SCC label {key!r}")

    sector, camera, ccd = parse_scc(key)
    matches = _find_target_matches(
        target_list, sector=sector, camera=camera, ccd=ccd
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        _raise_ambiguous_target(key, matches)
    raise KeyError(f"No target for SCC {sector}/{camera}/{ccd}")


def find_target_for_run(ctx, state, scc: str) -> Target:
    """Resolve SCC from frozen targets CSV, falling back to run DB rows."""
    try:
        return find_target(ctx.targets, scc)
    except KeyError:
        sector, camera, ccd = parse_scc(scc)
        row = state.get_run_target(ctx.run_id, sector, camera, ccd)
        if row is None:
            raise KeyError(f"No target for SCC {sector}/{camera}/{ccd}") from None
        return Target(
            sector=int(row["sector"]),
            camera=int(row["camera"]),
            ccd=int(row["ccd"]),
            target_ra=0.0,
            target_dec=0.0,
            target_name=str(row["target_name"]),
            enabled=bool(row.get("enabled", True)),
        )
