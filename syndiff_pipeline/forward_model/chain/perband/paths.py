"""Paths and config accessors shared by the per-band chain, kernels, Hotpants reference, final image and score stages.

Everything is derived from the chain config (``config.load_config``) and the stage directories
``cfg.stage_dir(name)``; no stage reads anything else. Replaces e2e ``common_f.py`` / ``kc.py`` path blocks.

Config fields read (CONTRACT.md schema plus the optional ones marked *):
  cfg.field, cfg.stem, cfg.unseen_stem*, cfg.scc.{sector,camera,ccd}, cfg.data_root, cfg.out_root,
  cfg.code.forward_model_root*, cfg.inputs.{colour_file, adopted_weights, band_cells*, colour_map, xp_synth,
  scorer_dir, pass2_hp*, combined_store_weights*, skylist*}.
Dotted lookups go through :func:`opt`, which accepts dataclasses, mappings and namespaces alike.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL = "a3"                      # historic file-name label of "the fit named in the config" (kept for the e2e products)
BANDS = ("r", "i", "z", "y")
LAM = (617.0, 752.0, 866.0, 962.0)   # PS1 effective wavelengths [nm]
PSF_SIGMA = 40.0                  # template pre-blur sigma [PS1 px] (production ps1_process default)
RADIUS = 470                      # pre-blur kernel radius [PS1 px]
RECIPE_CFG = {"remove_saturated_stars": True, "enable_saturation_correction": False}
OS = 4
SCIENCE_ORIGIN_FFI = (44, 0)      # science-local (0, 0) = FFI (col 44, row 0), same for every TESS SPOC FFI
SCIENCE_SHAPE = (2048, 2048)

_MISSING = object()


def opt(obj: Any, dotted: str, default: Any = _MISSING) -> Any:
    """``obj.a.b`` via attributes or mapping keys; ``default`` (or KeyError) when absent or None."""
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            break
        if isinstance(cur, dict):
            cur = cur.get(part, None)
        else:
            cur = getattr(cur, part, None)
    if cur is None:
        if default is _MISSING:
            raise KeyError(f"config field {dotted!r} is required by this stage")
        return default
    return cur


@dataclass(frozen=True)
class ChainPaths:
    """All directories/files the stages of this module family read and write, from one config."""

    cfg: Any
    field: str
    sector: int
    camera: int
    ccd: int
    stem: str
    unseen_stem: Any
    data: Path
    scc: Path
    out_root: Path

    # ---- per-band chain (stage dir perband/)
    @property
    def perband(self) -> Path:
        return Path(self.cfg.stage_dir("perband"))

    @property
    def cells_json(self) -> Path:
        return self.perband / "cells.json"

    @property
    def publisher_lists(self) -> Path:
        return self.perband / "publisher_lists.json"

    @property
    def band_cells(self) -> Path:
        bc = opt(self.cfg, "inputs.band_cells", None)
        return Path(bc) if bc else self.perband / "band_cells"

    @property
    def combined_cells(self) -> Path:
        """Rebuilt (S22-style) combined cells for cells absent from the shared store; optional."""
        return self.perband / "combined_cells"

    @property
    def contrib(self) -> Path:
        return self.perband / "contrib"

    @property
    def band_templates(self) -> Path:
        return self.perband / "band_templates"

    @property
    def store_weights_json(self) -> Path:
        return self.band_templates / "store_weights.json"

    @property
    def figs(self) -> Path:
        return self.perband / "figs"

    # ---- other stages
    @property
    def mapping_dir(self) -> Path:
        return Path(self.cfg.stage_dir("mapping")) / "oversampling_4"

    @property
    def master_name(self) -> str:
        return f"tess_s{self.sector:04d}_{self.camera}_{self.ccd}_master_pixels2skycells_os4.fits.fz"

    @property
    def skylist(self) -> Path:
        s = opt(self.cfg, "inputs.skylist", None)
        return Path(s) if s else self.mapping_dir / f"tess_s{self.sector:04d}_{self.camera}_{self.ccd}_master_skycells_list_os4.csv"

    @property
    def raw_zarr(self) -> Path:
        return self.data / "ps1_skycells_zarr/ps1_skycells.zarr"

    @property
    def fit_dir(self) -> Path:
        return Path(self.cfg.stage_dir("fit"))

    @property
    def scene_dir(self) -> Path:
        """Scene the calibration fit was trained on: ``nbr_boot`` when neighbours are configured, else ``scene_boot``."""
        if opt(self.cfg, "neighbours", None) is not None:
            return Path(self.cfg.stage_dir("nbr_boot"))
        return Path(self.cfg.stage_dir("scene_boot"))

    @property
    def kernels(self) -> Path:
        return Path(self.cfg.stage_dir("kernels"))

    @property
    def hotpants(self) -> Path:
        return Path(self.cfg.stage_dir("hotpants"))

    def hotpants_ref(self, stem: str) -> Path:
        return self.hotpants / stem / "hp_d" / f"{stem}_hp_d.fits.fz"

    @property
    def final(self) -> Path:
        return Path(self.cfg.stage_dir("final"))

    @property
    def score(self) -> Path:
        return Path(self.cfg.stage_dir("score"))

    @property
    def frames(self) -> list[str]:
        return [self.stem] + ([self.unseen_stem] if self.unseen_stem else [])

    def ffi(self, stem: str) -> Path:
        hits = sorted((self.scc / "ffi").glob(f"{stem}-*-s_ffic.fits*"))
        if not hits:
            raise FileNotFoundError(f"no FFI for {stem} under {self.scc / 'ffi'}")
        return hits[0]

    def background(self, stem: str) -> Path:
        hits = sorted((self.scc / "diff_linear/ks_b").glob(f"{stem}_ks_b.fits*"))
        if not hits:
            raise FileNotFoundError(f"no ks_b background for {stem}")
        return hits[0]

    # ---- config-derived scalars
    @property
    def adopted_weights(self) -> dict:
        return json.loads(Path(self.cfg.inputs.adopted_weights).read_text())

    @property
    def colour_file(self) -> Path:
        return Path(self.cfg.inputs.colour_file)


def chain_paths(cfg: Any) -> ChainPaths:
    return ChainPaths(
        cfg=cfg, field=str(cfg.field),
        sector=int(cfg.scc.sector), camera=int(cfg.scc.camera), ccd=int(cfg.scc.ccd),
        stem=str(cfg.stem), unseen_stem=opt(cfg, "unseen_stem", None),
        data=Path(cfg.data_root),
        scc=Path(cfg.data_root) / f"s{int(cfg.scc.sector):04d}/c{int(cfg.scc.camera)}/k{int(cfg.scc.ccd)}",
        out_root=Path(cfg.out_root),
    )


def combined_store_weights_mode(cfg: Any) -> str:
    """Which band-weight set the shared combined store holds for this field: ``production`` (default) or ``adopted``."""
    mode = str(opt(cfg, "inputs.combined_store_weights", "production"))
    if mode not in ("production", "adopted"):
        raise ValueError(f"inputs.combined_store_weights must be production|adopted, got {mode!r}")
    return mode


def chain_band_weights(cfg: Any) -> dict[str, float]:
    """The band weights the per-band chain ASSUMES its combined-store cells carry (see
    ``combined_store_weights_mode``): the production defaults, or the adopted D13 ``weights_rizy``."""
    if combined_store_weights_mode(cfg) == "production":
        from syndiff_pipeline.template_creation.processing.combined_store import DEFAULT_BAND_WEIGHTS
        return {b: float(DEFAULT_BAND_WEIGHTS[b]) for b in BANDS}
    w = json.loads(Path(opt(cfg, "inputs.adopted_weights")).read_text())["weights_rizy"]
    return {b: float(v) for b, v in zip(BANDS, w)}
