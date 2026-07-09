"""Load host-star requests from a separate stars CSV (independent of targets.csv)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_STAR_HOSTS_HEADER = frozenset({"tic_id", "gaia_source_id", "label"})


@dataclass(frozen=True)
class StarHostRequest:
    tic_id: Optional[int]
    gaia_source_id: Optional[int]
    label: Optional[str]

    def __post_init__(self) -> None:
        has_tic = self.tic_id is not None
        has_gaia = self.gaia_source_id is not None
        if has_tic == has_gaia:
            which = "both" if has_tic else "neither"
            raise ValueError(
                f"Exactly one of tic_id or gaia_source_id must be set (got {which})"
            )


def _parse_optional_int(cell: str | None, field: str, row_num: int) -> int | None:
    if cell is None or str(cell).strip() == "":
        return None
    text = str(cell).strip()
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_num}: {field} must be an integer, got {text!r}"
        ) from exc


def _parse_optional_label(cell: str | None) -> str | None:
    if cell is None:
        return None
    text = str(cell).strip()
    return text or None


def load_star_hosts_file(path: str | Path) -> list[StarHostRequest]:
    """Parse ``tic_id,gaia_source_id,label`` rows from *path*."""
    p = Path(path).expanduser().resolve()
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {p}")
        fields = frozenset(f.strip().lower() for f in reader.fieldnames)
        if not _STAR_HOSTS_HEADER.issubset(fields):
            missing = sorted(_STAR_HOSTS_HEADER - fields)
            raise ValueError(
                f"Stars CSV {p} missing required columns: {missing}"
            )
        out: list[StarHostRequest] = []
        for row_num, raw in enumerate(reader, start=2):
            row = {k.strip().lower(): v for k, v in raw.items()}
            tic_id = _parse_optional_int(row.get("tic_id"), "tic_id", row_num)
            gaia_source_id = _parse_optional_int(
                row.get("gaia_source_id"), "gaia_source_id", row_num
            )
            label = _parse_optional_label(row.get("label"))
            if tic_id is not None and gaia_source_id is not None:
                raise ValueError(
                    f"Row {row_num}: exactly one of tic_id or gaia_source_id "
                    "must be set (got both)"
                )
            if tic_id is None and gaia_source_id is None:
                raise ValueError(
                    f"Row {row_num}: exactly one of tic_id or gaia_source_id "
                    "must be set (got neither)"
                )
            out.append(
                StarHostRequest(
                    tic_id=tic_id,
                    gaia_source_id=gaia_source_id,
                    label=label,
                )
            )
    return out
