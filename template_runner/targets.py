"""Load template pipeline targets from normalized CSV or SN event catalog."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

_TESS_COVERAGE_RE = re.compile(r"S(\d+)C(\d+)D(\d+)", re.IGNORECASE)

NORMALIZED_HEADER = frozenset(
    {"sector", "camera", "ccd", "target_ra", "target_dec", "target_name", "enabled"}
)
EVENT_HEADER = frozenset({"id", "ra", "dec", "tess_coverage"})


@dataclass(frozen=True)
class Target:
    sector: int
    camera: int
    ccd: int
    target_ra: float
    target_dec: float
    target_name: str
    enabled: bool = True

    def scc_key(self) -> str:
        return f"{self.sector}/{self.camera}/{self.ccd}"

    def label(self) -> str:
        safe = re.sub(r"[^\w.-]+", "_", self.target_name.strip())
        return f"s{self.sector:04d}_c{self.camera}_k{self.ccd}_{safe}"


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
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _target_name_from_event_id(event_id: str) -> str:
    name = str(event_id or "").strip()
    if name.upper().startswith("SN "):
        name = name[3:].strip()
    return name or "unknown"


def _load_normalized_rows(rows: Sequence[dict]) -> List[Target]:
    out: List[Target] = []
    for row in rows:
        if not _parse_bool(row.get("enabled"), default=True):
            continue
        out.append(
            Target(
                sector=int(row["sector"]),
                camera=int(row["camera"]),
                ccd=int(row["ccd"]),
                target_ra=float(row["target_ra"]),
                target_dec=float(row["target_dec"]),
                target_name=str(row["target_name"]).strip(),
                enabled=True,
            )
        )
    return out


def _load_event_rows(rows: Sequence[dict]) -> List[Target]:
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
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        fields = frozenset(f.strip().lower() for f in reader.fieldnames)
        rows = [{k.strip().lower(): v for k, v in row.items()} for row in reader]
    return rows, fields


def load_targets(path: str | Path) -> List[Target]:
    """Load targets from normalized CSV or SN event catalog CSV."""
    p = Path(path).expanduser().resolve()
    rows, fields = _read_csv_rows(p)
    if EVENT_HEADER.issubset(fields) and "sector" not in fields:
        return _load_event_rows(rows)
    if NORMALIZED_HEADER.issubset(fields):
        return _load_normalized_rows(rows)
    missing_norm = sorted(NORMALIZED_HEADER - fields)
    missing_evt = sorted(EVENT_HEADER - fields)
    raise ValueError(
        f"Unrecognized CSV header in {p}. "
        f"Need normalized columns (missing {missing_norm}) or event catalog (missing {missing_evt})."
    )


def find_target(targets: Iterable[Target], scc: str) -> Target:
    """Find target by ``sector,camera,ccd`` or ``sector/camera/ccd`` key."""
    parts = re.split(r"[,/]", scc.strip())
    if len(parts) != 3:
        raise ValueError(f"Expected SCC as S,C,K got {scc!r}")
    sector, camera, ccd = (int(p) for p in parts)
    for t in targets:
        if t.sector == sector and t.camera == camera and t.ccd == ccd:
            return t
    raise KeyError(f"No target for SCC {sector}/{camera}/{ccd}")
