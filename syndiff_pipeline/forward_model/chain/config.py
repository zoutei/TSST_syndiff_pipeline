"""Chain configuration: one YAML file per field -> frozen dataclasses (schema in CONTRACT.md).

``load_config(path) -> ChainConfig`` validates the file (unknown keys are rejected so typos fail loudly) and
resolves derived paths. Paths that are not known yet may be ``null``; a stage that needs one calls
``cfg.need("inputs.source_scene")`` which raises a clear error naming the missing key.

Stage bookkeeping shared by every stage: ``cfg.stage_dir(name)`` (= ``out_root/<name>``), ``write_provenance``,
``mark_done`` / ``is_done`` (``<stage>/DONE`` marker + ``<stage>/provenance.json``).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

# Stage directories under out_root (CONTRACT.md).
STAGES: tuple[str, ...] = (
    "bootstrap", "scene_boot", "fit", "wcs", "mapping", "perband", "kernels", "hotpants",
    "final", "score", "scene_final", "refit", "compare",
)
KERNEL_SOURCES = ("k_sigma", "phasea")
# TESS FFI geometry: science pixels start at FFI column 44, row 0; 2048 x 2048.
SCIENCE_ORIGIN_FFI = (44, 0)
SCIENCE_SHAPE = (2048, 2048)
# Files up to this size get a sha256 in provenance; larger ones record mtime + size.
SHA_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_CPUS = {"fit": 16, "mapping": 48, "f03": 8}
DEFAULT_MEM_MB = {"fit": 16000, "mapping": 64000, "f03": 80000}


class ConfigError(ValueError):
    """Invalid or incomplete chain configuration."""


@dataclass(frozen=True)
class SccCfg:
    sector: int
    camera: int
    ccd: int


@dataclass(frozen=True)
class CodeCfg:
    sha: str | None          # pinned code sha (None -> git HEAD of forward_model_root at provenance time)
    forward_model_root: Path  # checkout whose ``syndiff_pipeline`` the jobs import (PYTHONPATH)


@dataclass(frozen=True)
class InputsCfg:
    colour_file: Path
    source_scene: Path | None = None
    exclusion_csv: Path | None = None
    exclusion_strict: bool = False       # True: every listed source_id must be in the scene (old e2e assert)
    strap_mask: bool = False
    bootstrap_mapping: Path | None = None
    band_cells: Path | None = None
    adopted_weights: Path | None = None
    init_params: Path | None = None
    bootstrap_hp_d: Path | None = None   # optional: hp_d used to swap the boot scene (default bootstrap/ search)
    # per-band / kernel / final / score stages (Agent B)
    colour_map: Mapping[str, Any] | None = None   # {a, b} floats, or {summary_json: path, key: str}
    xp_synth: Path | None = None                  # Gaia-XP synthetic photometry csv (kernel colour checks)
    scorer_dir: Path | None = None                # frozen scorer star lists (score stage)
    pass2_hp: Mapping[str, Path] | None = None    # optional extra baselines: stem -> hp_d path
    combined_store_weights: str = "production"    # band weights the combined store was built with: production|adopted
    skylist: Path | None = None                   # optional skycell list override (default: from the mapping)
    lane_dir: Path | None = None                  # F=1 lane (ks_b/, shared_mask, substamp stars); default out_root/lane_f1


@dataclass(frozen=True)
class BackgroundCfg:
    fill: str = "harmonic"          # production default since 70cba3c
    star_mask_pad_px: int = 0       # tessreduce_star_mask_pad_px (bkg-star-pad 2ecf558); 0 = unchanged


@dataclass(frozen=True)
class FitCfg:
    recipe: str = "paper1_dataset"
    extra_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelsCfg:
    source: str = "k_sigma"
    kernel_bright_q: float = 0.0


@dataclass(frozen=True)
class MaskCfg:
    bit2_radius: float | None = None


@dataclass(frozen=True)
class CondorCfg:
    request_cpus: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_CPUS))
    request_memory_mb: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_MEM_MB))


@dataclass(frozen=True)
class ReferenceCfg:
    """Optional read-only comparison products for the WCS / mapping gates (``{scc}`` expands to the SCC data dir)."""
    old_store: Path | None = None      # reference WCS store (gate B, gates C)
    tvwcs_store: Path | None = None    # per-epoch temporal WCS store (gate B)
    old_mapping: Path | None = None    # reference OS4 mapping dir (gates C, D)
    c5_fit: Path | None = None         # fit whose params.npz wcs_coeff is the reference (gate C2)
    w3_wcs_offset_npz: Path | None = None  # figure only


@dataclass(frozen=True)
class ChainConfig:
    field: str
    scc: SccCfg
    stem: str
    unseen_stem: str | None
    data_root: Path
    out_root: Path
    code: CodeCfg
    inputs: InputsCfg
    fit: FitCfg
    kernels: KernelsCfg
    mask: MaskCfg
    condor: CondorCfg
    reference: ReferenceCfg
    wcs_version: str
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    config_path: Path | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------------ paths
    @property
    def scc_root(self) -> Path:
        """``{data_root}/s{SSSS}/c{C}/k{K}``."""
        s = self.scc
        return self.data_root / f"s{s.sector:04d}" / f"c{s.camera}" / f"k{s.ccd}"

    def stage_dir(self, name: str) -> Path:
        """``out_root/<name>`` (name must be a known stage). Not created."""
        if name not in STAGES:
            raise ConfigError(f"unknown stage {name!r}; known: {', '.join(STAGES)}")
        return self.out_root / name

    def ffi_path(self, stem: str | None = None) -> Path:
        """On-disk science FFI of ``stem`` (default: the fit frame); ``.fits.fz`` / ``.fits.gz`` / ``.fits``."""
        stem = stem or self.stem
        d = self.scc_root / "ffi"
        for ext in (".fits.fz", ".fits.gz", ".fits"):
            hits = sorted(d.glob(f"{stem}-*_ffic{ext}"))
            if hits:
                return hits[0]
        raise FileNotFoundError(f"no FFI for {stem} in {d}")

    def wcs_store_dir(self) -> Path:
        return self.stage_dir("wcs") / self.wcs_version

    def need(self, dotted: str) -> Any:
        """Value of a possibly-null config entry (``"inputs.source_scene"``); raises ConfigError if it is null."""
        obj: Any = self
        for part in dotted.split("."):
            obj = getattr(obj, part)
        if obj is None:
            raise ConfigError(f"config key {dotted!r} is null/unset; fill it in {self.config_path or 'the config'}")
        return obj

    # ------------------------------------------------------------------ identity
    def to_dict(self) -> dict:
        """JSON-able normalised form (the thing that is hashed)."""
        d = dataclasses.asdict(self)
        d.pop("raw", None)
        d.pop("config_path", None)
        return _jsonable(d)

    def config_hash(self) -> str:
        """sha256 of the canonical normalised config (independent of YAML formatting/comments/key order)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def code_sha(self) -> str:
        """Pinned sha, else git HEAD of ``code.forward_model_root`` (``"unknown"`` if not a git checkout)."""
        return self.code.sha or _git_head(self.code.forward_model_root)

    def check_code_sha(self) -> None:
        """With ``code.sha`` pinned, refuse to run unless ``forward_model_root`` is at that commit with a clean
        package tree (otherwise ``code_sha()`` would record a sha the running code does not have)."""
        if not self.code.sha:
            return
        root = self.code.forward_model_root
        head = _git_head(root)
        if head != self.code.sha:
            raise ConfigError(f"code.sha {self.code.sha} pinned but {root} is at {head}")
        try:
            dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "syndiff_pipeline"],
                                            text=True, stderr=subprocess.DEVNULL).strip()
        except Exception as e:  # noqa: BLE001
            raise ConfigError(f"cannot check {root} for local changes: {e}")
        if dirty:
            raise ConfigError(f"code.sha pinned but {root}/syndiff_pipeline has local changes:\n{dirty}")


# ---------------------------------------------------------------------- loading
def _jsonable(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _git_dirty(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _section(raw: Mapping, key: str, allowed: set[str], *, required: bool = False) -> dict:
    v = raw.get(key)
    if v is None:
        if required:
            raise ConfigError(f"missing required section {key!r}")
        return {}
    if not isinstance(v, dict):
        raise ConfigError(f"{key!r} must be a mapping, got {type(v).__name__}")
    extra = set(v) - allowed
    if extra:
        raise ConfigError(f"unknown key(s) in {key!r}: {sorted(extra)}; allowed: {sorted(allowed)}")
    return v


def _path(v: Any, name: str, *, required: bool = False, subst: Mapping[str, str] | None = None) -> Path | None:
    if v is None:
        if required:
            raise ConfigError(f"{name} is required")
        return None
    if not isinstance(v, (str, Path)) or str(v) == "":
        raise ConfigError(f"{name} must be a path string, got {v!r}")
    s = str(v)
    if subst:
        s = s.format(**subst)
    p = Path(s).expanduser()
    if not p.is_absolute():
        raise ConfigError(f"{name} must be an absolute path, got {s!r}")
    return p


def _int(v: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ConfigError(f"{name} must be an integer in [{lo}, {hi}], got {v!r}")
    return v


def _bool(v: Any, name: str) -> bool:
    if not isinstance(v, bool):
        raise ConfigError(f"{name} must be true/false, got {v!r}")
    return v


def _resources(sec: Mapping, key: str, defaults: Mapping[str, int]) -> dict[str, int]:
    v = sec.get(key)
    out = dict(defaults)
    if v is None:
        return out
    if not isinstance(v, dict):
        raise ConfigError(f"condor.{key} must be a mapping stage -> int")
    for k, x in v.items():
        if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
            raise ConfigError(f"condor.{key}.{k} must be a positive integer, got {x!r}")
        out[str(k)] = x
    return out


_TOP = {"field", "scc", "stem", "unseen_stem", "data_root", "out_root", "code", "inputs", "fit", "kernels",
        "mask", "condor", "reference", "wcs_version", "background"}


def config_from_dict(raw: Mapping[str, Any], config_path: Path | None = None) -> ChainConfig:
    """Validate a parsed YAML mapping and build the frozen config."""
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    extra = set(raw) - _TOP
    if extra:
        raise ConfigError(f"unknown top-level key(s): {sorted(extra)}")

    label = raw.get("field")
    if not isinstance(label, str) or not label or "/" in label:
        raise ConfigError(f"field must be a non-empty label without '/', got {label!r}")

    sc = _section(raw, "scc", {"sector", "camera", "ccd"}, required=True)
    for k in ("sector", "camera", "ccd"):
        if k not in sc:
            raise ConfigError(f"scc.{k} is required")
    scc = SccCfg(_int(sc["sector"], "scc.sector", 1, 999), _int(sc["camera"], "scc.camera", 1, 4),
                 _int(sc["ccd"], "scc.ccd", 1, 4))

    def stem_check(v: Any, name: str) -> str:
        want = f"-s{scc.sector:04d}-{scc.camera}-{scc.ccd}"
        if not isinstance(v, str) or not v.startswith("tess") or not v.endswith(want):
            raise ConfigError(f"{name} must look like tess<timestamp>{want}, got {v!r}")
        return v

    stem = stem_check(raw.get("stem"), "stem")
    unseen = raw.get("unseen_stem")
    unseen = None if unseen is None else stem_check(unseen, "unseen_stem")
    data_root = _path(raw.get("data_root"), "data_root", required=True)
    out_root = _path(raw.get("out_root"), "out_root", required=True)

    cd = _section(raw, "code", {"sha", "forward_model_root"})
    sha = cd.get("sha")
    if sha is not None and not isinstance(sha, str):
        raise ConfigError("code.sha must be a string or null")
    fm_root = _path(cd.get("forward_model_root"), "code.forward_model_root")
    if fm_root is None:  # default: the checkout this module was imported from
        fm_root = Path(__file__).resolve().parents[3]
    code = CodeCfg(sha, fm_root)

    ip = _section(raw, "inputs", {"colour_file", "source_scene", "exclusion_csv", "exclusion_strict", "strap_mask", "bootstrap_mapping",
                                  "band_cells", "adopted_weights", "init_params", "bootstrap_hp_d", "colour_map", "xp_synth",
                                  "scorer_dir", "pass2_hp", "combined_store_weights", "skylist", "lane_dir"}, required=True)
    cmap = ip.get("colour_map")
    if cmap is not None:
        if not isinstance(cmap, dict) or not (set(cmap) == {"a", "b"} or set(cmap) == {"summary_json", "key"}):
            raise ConfigError("inputs.colour_map must be {a, b} or {summary_json, key}")
        if "a" in cmap and not all(isinstance(cmap[k], (int, float)) and not isinstance(cmap[k], bool) for k in "ab"):
            raise ConfigError("inputs.colour_map a/b must be numbers")
        if "summary_json" in cmap:
            cmap = {"summary_json": _path(cmap["summary_json"], "inputs.colour_map.summary_json", required=True),
                    "key": str(cmap["key"])}
    p2 = ip.get("pass2_hp")
    if p2 is not None:
        if not isinstance(p2, dict):
            raise ConfigError("inputs.pass2_hp must be a mapping stem -> hp_d path")
        p2 = {str(k): _path(v, f"inputs.pass2_hp.{k}", required=True) for k, v in p2.items()}
    csw = ip.get("combined_store_weights", "production")
    if csw not in ("production", "adopted"):
        raise ConfigError(f"inputs.combined_store_weights must be production|adopted, got {csw!r}")
    inputs = InputsCfg(
        colour_file=_path(ip.get("colour_file"), "inputs.colour_file", required=True),
        source_scene=_path(ip.get("source_scene"), "inputs.source_scene"),
        exclusion_csv=_path(ip.get("exclusion_csv"), "inputs.exclusion_csv"),
        exclusion_strict=_bool(ip.get("exclusion_strict", False), "inputs.exclusion_strict"),
        strap_mask=_bool(ip.get("strap_mask", False), "inputs.strap_mask"),
        bootstrap_mapping=_path(ip.get("bootstrap_mapping"), "inputs.bootstrap_mapping"),
        band_cells=_path(ip.get("band_cells"), "inputs.band_cells"),
        adopted_weights=_path(ip.get("adopted_weights"), "inputs.adopted_weights"),
        init_params=_path(ip.get("init_params"), "inputs.init_params"),
        bootstrap_hp_d=_path(ip.get("bootstrap_hp_d"), "inputs.bootstrap_hp_d"),
        colour_map=cmap,
        xp_synth=_path(ip.get("xp_synth"), "inputs.xp_synth"),
        scorer_dir=_path(ip.get("scorer_dir"), "inputs.scorer_dir"),
        pass2_hp=p2,
        combined_store_weights=csw,
        skylist=_path(ip.get("skylist"), "inputs.skylist"),
        lane_dir=_path(ip.get("lane_dir"), "inputs.lane_dir"),
    )
    bp = _section(raw, "background", {"fill", "star_mask_pad_px"})
    fill = bp.get("fill", "harmonic")
    if fill not in ("harmonic", "biharmonic"):
        raise ConfigError(f"background.fill must be harmonic|biharmonic, got {fill!r}")
    background = BackgroundCfg(fill=fill, star_mask_pad_px=_int(bp.get("star_mask_pad_px", 0), "background.star_mask_pad_px", 0, 64))

    fp = _section(raw, "fit", {"recipe", "extra_flags"})
    recipe = fp.get("recipe", "paper1_dataset")
    if not isinstance(recipe, str) or not recipe:
        raise ConfigError("fit.recipe must be a recipe name")
    flags = fp.get("extra_flags") or []
    if not isinstance(flags, list) or not all(isinstance(f, (str, int, float)) and not isinstance(f, bool)
                                              for f in flags):
        raise ConfigError("fit.extra_flags must be a list of strings (scene_fit argv tokens)")
    fit = FitCfg(recipe, tuple(str(f) for f in flags))

    kp = _section(raw, "kernels", {"source", "kernel_bright_q"})
    src = kp.get("source", "k_sigma")
    if src not in KERNEL_SOURCES:
        raise ConfigError(f"kernels.source must be one of {KERNEL_SOURCES}, got {src!r}")
    q = kp.get("kernel_bright_q", 0.0)
    if isinstance(q, bool) or not isinstance(q, (int, float)) or q < 0:
        raise ConfigError(f"kernels.kernel_bright_q must be a number >= 0, got {q!r}")
    kernels = KernelsCfg(src, float(q))

    mp = _section(raw, "mask", {"bit2_radius"})
    r = mp.get("bit2_radius")
    if r is not None and (isinstance(r, bool) or not isinstance(r, (int, float)) or r <= 0):
        raise ConfigError(f"mask.bit2_radius must be null or a positive number, got {r!r}")
    mask = MaskCfg(None if r is None else float(r))

    cp = _section(raw, "condor", {"request_cpus", "request_memory_mb"})
    condor = CondorCfg(_resources(cp, "request_cpus", DEFAULT_CPUS),
                       _resources(cp, "request_memory_mb", DEFAULT_MEM_MB))

    scc_dir = str(data_root / f"s{scc.sector:04d}" / f"c{scc.camera}" / f"k{scc.ccd}")
    rp = _section(raw, "reference", {"old_store", "tvwcs_store", "old_mapping", "c5_fit", "w3_wcs_offset_npz"})
    sub = {"scc": scc_dir}
    reference = ReferenceCfg(**{k: _path(rp.get(k), f"reference.{k}", subst=sub) for k in (
        "old_store", "tvwcs_store", "old_mapping", "c5_fit", "w3_wcs_offset_npz")})

    ver = raw.get("wcs_version", f"{label}_v1")
    if not isinstance(ver, str) or not ver or "/" in ver:
        raise ConfigError(f"wcs_version must be a plain directory name, got {ver!r}")

    return ChainConfig(field=label, scc=scc, stem=stem, unseen_stem=unseen, data_root=data_root, out_root=out_root,
                       code=code, inputs=inputs, fit=fit, kernels=kernels, mask=mask, condor=condor,
                       reference=reference, wcs_version=ver, background=background, config_path=config_path,
                       raw=dict(raw))


def load_config(path: str | Path) -> ChainConfig:
    """Parse + validate a chain YAML file."""
    p = Path(path)
    with open(p) as f:
        raw = yaml.safe_load(f)
    return config_from_dict(raw, config_path=p.resolve())


# ---------------------------------------------------------------------- stage bookkeeping
def _describe(path: Path) -> dict:
    """Provenance record of one input: sha256 for small files, mtime+size for large files / directories."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    rec: dict = {"path": str(path), "exists": True, "mtime": st.st_mtime}
    if path.is_dir():
        rec["kind"] = "dir"
        rec["n_entries"] = sum(1 for _ in path.iterdir())
    else:
        rec["kind"] = "file"
        rec["size"] = st.st_size
        if st.st_size <= SHA_MAX_BYTES:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            rec["sha256"] = h.hexdigest()
    return rec


def write_provenance(stage_dir: str | Path, cfg: ChainConfig, inputs: Mapping[str, Any]) -> Path:
    """Write ``<stage_dir>/provenance.json``: config hash + config, code sha, and the inputs.

    ``inputs`` maps a label to a path (str/Path; described by sha256 or mtime+size) or to a plain JSON value
    (numbers, strings that are not paths should be wrapped by the caller as ``{"value": ...}``)."""
    d = Path(stage_dir)
    d.mkdir(parents=True, exist_ok=True)
    recs = {}
    for k, v in inputs.items():
        recs[k] = _describe(Path(v)) if isinstance(v, (str, Path)) else _jsonable(v)
    root = cfg.code.forward_model_root
    prov = {
        "stage": d.name,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "config_hash": cfg.config_hash(),
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "config": cfg.to_dict(),
        "code": {"sha": cfg.code_sha(), "pinned": cfg.code.sha is not None,
                 "forward_model_root": str(root), "git_status": _git_dirty(root)},
        "inputs": recs,
    }
    out = d / "provenance.json"
    out.write_text(json.dumps(prov, indent=1, default=str) + "\n")
    return out


def mark_done(stage_dir: str | Path) -> Path:
    """Create ``<stage_dir>/DONE`` (timestamp inside)."""
    d = Path(stage_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "DONE"
    p.write_text(_dt.datetime.now().isoformat(timespec="seconds") + "\n")
    return p


def is_done(stage_dir: str | Path) -> bool:
    return (Path(stage_dir) / "DONE").is_file()
