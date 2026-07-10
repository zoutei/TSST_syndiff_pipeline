"""Load star site policy, star_targets CSV, and merged per-run configuration."""

from __future__ import annotations

import copy
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from syndiff_pipeline.common.orchestration.targets import Target, _parse_bool, find_target

log = logging.getLogger(__name__)

STAR_TARGETS_HEADER = frozenset(
    {
        "sector",
        "camera",
        "ccd",
        "target_name",
        "stars_file",
        "baseline_workspace_run_id",
        "baseline_diffs",
        "baseline_convolved",
        "phot_bkg",
        "enabled",
    }
)

VALID_PS1_SOURCES = frozenset({"zarr_local_only", "zarr_download", "stream"})
_LEGACY_PS1_SOURCE = {
    "zarr": "zarr_download",
    "download": "stream",
}


@dataclass
class StarBaselineConfig:
    """Resolved baseline workspace labels for one star run."""

    workspace_run_id: str = "none"
    diffs: str = "hp_d"
    convolved: str | None = "hp_c"
    phot_bkg: str | None = "ks_b_s"


@dataclass
class StarEpsfConfig:
    """Gridded ePSF fitting on baseline diff images before gepsf photometry."""

    enabled: bool = False
    diffs: str | None = None
    output: str = "epsf_r1"
    tile_nx: int = 2
    tile_ny: int = 2
    epsf_oversample: int = 4
    psf_size: int = 11
    extract_size: int | None = 11
    min_stars_per_tile: int = 5
    mag_max_rp: float | None = 12.95
    epsf_maxiters: int = 15
    epsf_recentering_maxiters: int = 20
    epsf_n_jobs: int | None = 8


@dataclass
class StarCondorConfig:
    """HTCondor resource request for the star stage."""

    request_cpus: int = 8
    request_memory: int = 100_000
    requirements: str | None = 'Memory >= 100000 && LoadAvg < 10 && Machine != "plscience10.stsci.edu"'
    rank: str | None = "-LoadAvg"


@dataclass
class StarSitePolicy:
    """Site-level star policy from ``star_config.yaml``."""

    deployment_file: str = "deployment.yaml"
    defaults: dict = field(default_factory=dict)
    baseline: StarBaselineConfig = field(default_factory=StarBaselineConfig)
    photometry: dict = field(default_factory=dict)
    epsf: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    condor: StarCondorConfig = field(default_factory=StarCondorConfig)
    config_path: str = ""
    ps1_zarr_path: str | None = None


@dataclass(frozen=True)
class StarTargetRow:
    """One enabled row from ``star_targets.csv``."""

    target: Target
    stars_file: str
    baseline_workspace_run_id: str | None = None
    baseline_diffs: str | None = None
    baseline_convolved: str | None = None
    phot_bkg: str | None = None

    def scc_key(self) -> str:
        return self.target.scc_key()


@dataclass
class StarRunConfig:
    """Merged knobs for one SCC star run (row > overrides > defaults)."""

    cutout_size: int = 96
    stamp_size: int = 24
    kernel_margin_px: int = 470
    ps1_source: str = "zarr_download"
    debug_plots: bool = True
    workspace_run_id: str | None = None
    max_ffis: int | None = None
    overwrite: bool = False
    baseline: StarBaselineConfig = field(default_factory=StarBaselineConfig)
    photometry_methods: list[dict] = field(default_factory=list)
    epsf: StarEpsfConfig | None = None
    stars_file: str = ""
    ps1_zarr_path: str | None = None


def normalize_ps1_source(value: str | None, *, warn_legacy: bool = True) -> str:
    """Normalize ``ps1_source`` including legacy ``zarr`` / ``download`` aliases."""
    raw = str(value or "zarr_download").strip()
    if raw in _LEGACY_PS1_SOURCE:
        normalized = _LEGACY_PS1_SOURCE[raw]
        if warn_legacy:
            log.warning(
                "ps1_source=%r is deprecated; use %r instead",
                raw,
                normalized,
            )
        return normalized
    if raw not in VALID_PS1_SOURCES:
        raise ValueError(
            f"Invalid ps1_source {raw!r}; expected one of: "
            f"{', '.join(sorted(VALID_PS1_SOURCES))}"
        )
    return raw


def _parse_baseline_block(raw: dict | None, *, fallback: StarBaselineConfig) -> StarBaselineConfig:
    raw = raw or {}
    return StarBaselineConfig(
        workspace_run_id=str(
            raw.get("workspace_run_id", fallback.workspace_run_id) or fallback.workspace_run_id
        ).strip()
        or "none",
        diffs=str(raw.get("diffs", fallback.diffs) or fallback.diffs).strip() or "hp_d",
        convolved=_optional_str(raw.get("convolved", fallback.convolved)),
        phot_bkg=_optional_str(raw.get("phot_bkg", fallback.phot_bkg)),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _method_inputs(method: dict) -> dict:
    raw = method.get("inputs") or {}
    return raw if isinstance(raw, dict) else {}


def epsf_workspace_from_method(method: dict) -> str | None:
    """Return the ePSF workspace label from a photometry method ``inputs.epsf``."""
    label = _optional_str(_method_inputs(method).get("epsf"))
    return label


def required_epsf_workspaces(methods: list[dict]) -> list[str]:
    """Unique ePSF workspace labels referenced by ``psf_type: epsf`` methods."""
    seen: set[str] = set()
    out: list[str] = []
    for method in methods:
        mtype = str(method.get("type", "")).strip().lower()
        if mtype not in ("psf", "prf"):
            continue
        if str(method.get("psf_type", "prf")).strip().lower() != "epsf":
            continue
        label = epsf_workspace_from_method(method)
        if not label:
            name = str(method.get("name", "?"))
            raise ValueError(
                f"photometry method {name!r} has psf_type epsf but no inputs.epsf"
            )
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def normalize_photometry_methods(methods: list[dict]) -> list[dict]:
    """Validate photometry methods and normalize ``inputs`` blocks."""
    normalized: list[dict] = []
    for method in methods:
        entry = copy.deepcopy(method)
        mtype = str(entry.get("type", "")).strip().lower()
        if mtype in ("psf", "prf") and str(entry.get("psf_type", "prf")).strip().lower() == "epsf":
            label = epsf_workspace_from_method(entry)
            if not label:
                name = str(entry.get("name", "?"))
                raise ValueError(
                    f"photometry method {name!r} has psf_type epsf but no inputs.epsf"
                )
            entry["epsf_workspace"] = label
        normalized.append(entry)
    return normalized


def _parse_star_epsf(raw: dict | None) -> StarEpsfConfig | None:
    raw = raw or {}
    if not raw:
        return None
    enabled = bool(raw.get("enabled", True))
    if not enabled:
        return None
    inputs = raw.get("inputs") or {}
    diffs = _optional_str(inputs.get("diffs")) if isinstance(inputs, dict) else None
    extract_size = raw.get("extract_size")
    return StarEpsfConfig(
        enabled=True,
        diffs=diffs,
        output=str(raw.get("output", "epsf_r1")).strip() or "epsf_r1",
        tile_nx=int(raw.get("tile_nx", 2)),
        tile_ny=int(raw.get("tile_ny", 2)),
        epsf_oversample=int(raw.get("epsf_oversample", 4)),
        psf_size=int(raw.get("psf_size", 11)),
        extract_size=int(extract_size) if extract_size not in (None, "") else None,
        min_stars_per_tile=int(raw.get("min_stars_per_tile", 5)),
        mag_max_rp=float(raw["mag_max_rp"]) if raw.get("mag_max_rp") not in (None, "") else 12.95,
        epsf_maxiters=int(raw.get("epsf_maxiters", 15)),
        epsf_recentering_maxiters=int(raw.get("epsf_recentering_maxiters", 20)),
        epsf_n_jobs=int(raw["epsf_n_jobs"]) if raw.get("epsf_n_jobs") not in (None, "") else None,
    )


def _parse_star_condor(raw: dict | None) -> StarCondorConfig:
    raw = raw or {}
    mem = int(raw.get("request_memory", 100_000))
    req = raw.get("requirements")
    if req is None:
        req = f"Memory >= {mem} && LoadAvg < 10"
    return StarCondorConfig(
        request_cpus=int(raw.get("request_cpus", 8)),
        request_memory=mem,
        requirements=str(req) if req else None,
        rank=raw.get("rank", "-LoadAvg"),
    )


def load_star_site_policy(path: str | Path) -> StarSitePolicy:
    """Load ``star_config.yaml`` site policy."""
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    defaults = dict(raw.get("defaults") or {})
    baseline_raw = raw.get("baseline") or {}
    baseline = _parse_baseline_block(baseline_raw, fallback=StarBaselineConfig())
    photometry = dict(raw.get("photometry") or {})
    epsf = dict(raw.get("epsf") or {})
    overrides = dict(raw.get("overrides") or {})
    ps1_zarr = _optional_str(raw.get("ps1_zarr_path"))
    return StarSitePolicy(
        deployment_file=str(raw.get("deployment_file", "deployment.yaml")).strip()
        or "deployment.yaml",
        defaults=defaults,
        baseline=baseline,
        photometry=photometry,
        epsf=epsf,
        overrides=overrides,
        condor=_parse_star_condor(raw.get("condor")),
        config_path=str(config_path),
        ps1_zarr_path=ps1_zarr,
    )


def _resolve_stars_file(path: str, *, site_dir: Path) -> str:
    text = str(path or "").strip()
    if not text:
        raise ValueError("stars_file is required in star_targets row")
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (site_dir / p).resolve()
    return str(p)


def _read_star_targets_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        fields = frozenset(f.strip().lower() for f in reader.fieldnames)
        if not STAR_TARGETS_HEADER.issubset(fields):
            missing = sorted(STAR_TARGETS_HEADER - fields)
            raise ValueError(
                f"Unrecognized star_targets CSV header in {path}; missing columns: {missing}"
            )
        return [{k.strip().lower(): v for k, v in row.items()} for row in reader]


def load_star_targets(path: str | Path, *, site_dir: str | Path | None = None) -> list[StarTargetRow]:
    """Load enabled rows from ``star_targets.csv``."""
    csv_path = Path(path).expanduser().resolve()
    site = Path(site_dir).expanduser().resolve() if site_dir else csv_path.parent
    rows = _read_star_targets_csv(csv_path)
    out: list[StarTargetRow] = []
    for row in rows:
        if not _parse_bool(row.get("enabled"), default=True):
            continue
        target = Target(
            sector=int(row["sector"]),
            camera=int(row["camera"]),
            ccd=int(row["ccd"]),
            target_ra=0.0,
            target_dec=0.0,
            target_name=str(row["target_name"]).strip(),
            enabled=True,
        )
        out.append(
            StarTargetRow(
                target=target,
                stars_file=_resolve_stars_file(row.get("stars_file", ""), site_dir=site),
                baseline_workspace_run_id=_optional_str(row.get("baseline_workspace_run_id")),
                baseline_diffs=_optional_str(row.get("baseline_diffs")),
                baseline_convolved=_optional_str(row.get("baseline_convolved")),
                phot_bkg=_optional_str(row.get("phot_bkg")),
            )
        )
    return out


def find_star_target_row(rows: list[StarTargetRow], scc: str) -> StarTargetRow:
    """Find a star target row by SCC key or full event label."""
    targets = [row.target for row in rows]
    target = find_target(targets, scc)
    for row in rows:
        if row.target.label() == target.label():
            return row
    raise KeyError(f"No star_targets row for {scc!r}")


def resolve_star_run_config(
    policy: StarSitePolicy,
    star_target_row: StarTargetRow,
    *,
    site_dir: str | Path,
) -> StarRunConfig:
    """Merge policy defaults, SCC overrides, and star_targets row."""
    site = Path(site_dir).expanduser().resolve()
    merged_defaults = copy.deepcopy(policy.defaults)
    override_block = policy.overrides.get(star_target_row.scc_key()) or {}
    if override_block:
        merged_defaults = _deep_merge_dict(merged_defaults, override_block)

    baseline = _parse_baseline_block(
        merged_defaults.pop("baseline", None) or {},
        fallback=policy.baseline,
    )
    if star_target_row.baseline_workspace_run_id:
        baseline.workspace_run_id = star_target_row.baseline_workspace_run_id
    if star_target_row.baseline_diffs:
        baseline.diffs = star_target_row.baseline_diffs
    if star_target_row.baseline_convolved:
        baseline.convolved = star_target_row.baseline_convolved
    if star_target_row.phot_bkg:
        baseline.phot_bkg = star_target_row.phot_bkg

    photometry_block = merged_defaults.pop("photometry", None) or policy.photometry or {}
    epsf_block = merged_defaults.pop("epsf", None) or policy.epsf or {}
    methods = list(photometry_block.get("methods") or [])
    if not methods:
        methods = [
            {"name": "ap3", "type": "aperture", "tar_ap": 3, "sky_in": 5, "sky_out": 9},
        ]
    methods = normalize_photometry_methods(methods)
    epsf_cfg = _parse_star_epsf(epsf_block)
    if epsf_cfg is not None:
        for label in required_epsf_workspaces(methods):
            if label != epsf_cfg.output:
                raise ValueError(
                    f"photometry references inputs.epsf={label!r} but epsf.output "
                    f"is {epsf_cfg.output!r}; add a matching epsf block or align labels"
                )

    ps1_source = normalize_ps1_source(merged_defaults.get("ps1_source", "zarr_download"))
    workspace_run_id = _optional_str(merged_defaults.get("workspace_run_id"))
    max_ffis_raw = merged_defaults.get("max_ffis")
    max_ffis = int(max_ffis_raw) if max_ffis_raw not in (None, "") else None

    return StarRunConfig(
        cutout_size=int(merged_defaults.get("cutout_size", 96)),
        stamp_size=int(merged_defaults.get("stamp_size", 24)),
        kernel_margin_px=int(merged_defaults.get("kernel_margin_px", 470)),
        ps1_source=ps1_source,
        debug_plots=bool(merged_defaults.get("debug_plots", True)),
        workspace_run_id=workspace_run_id,
        max_ffis=max_ffis,
        overwrite=bool(merged_defaults.get("overwrite", False)),
        baseline=baseline,
        photometry_methods=methods,
        epsf=epsf_cfg,
        stars_file=star_target_row.stars_file,
        ps1_zarr_path=policy.ps1_zarr_path,
    )


def star_targets_to_orchestrator_targets(rows: list[StarTargetRow]) -> list[Target]:
    """Convert star target rows to orchestrator :class:`Target` objects."""
    return [row.target for row in rows]


def write_frozen_star_config(policy: StarSitePolicy, dest: str | Path) -> Path:
    """Write merged site policy YAML to a frozen run path."""
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "deployment_file": policy.deployment_file,
        "defaults": policy.defaults,
        "baseline": {
            "workspace_run_id": policy.baseline.workspace_run_id,
            "diffs": policy.baseline.diffs,
            "convolved": policy.baseline.convolved,
            "phot_bkg": policy.baseline.phot_bkg,
        },
        "photometry": policy.photometry,
        "epsf": policy.epsf,
        "overrides": policy.overrides,
    }
    if policy.ps1_zarr_path:
        payload["ps1_zarr_path"] = policy.ps1_zarr_path
    payload["condor"] = {
        "request_cpus": policy.condor.request_cpus,
        "request_memory": policy.condor.request_memory,
        "requirements": policy.condor.requirements,
        "rank": policy.condor.rank,
    }
    dest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dest_path


def resolve_star_config_path(*, meta: dict | None, runner_cfg) -> Path:
    """Resolve frozen or site star config path from run metadata."""
    for key in ("source_star_config_path", "star_config_path"):
        raw = (meta or {}).get(key) or getattr(runner_cfg, "star_config_path", "")
        if raw:
            return Path(str(raw)).expanduser().resolve()
    raise ValueError(
        "Star stage requires source_star_config_path in run_meta or "
        "star_config_path on RunnerConfig"
    )
