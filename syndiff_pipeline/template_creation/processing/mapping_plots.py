"""Debug figures for the SCC mapping stage."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries

from syndiff_pipeline.common.scc_paths import scc_debug_plots_dir, scc_mapping_master_skycells_csv
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.verify import (
    mapping_master_pixels2skycells_path,
)

log = logging.getLogger(__name__)

MAPPING_PROJECTION_OVERLAY_BASENAME = "mapping_projection_overlay.png"


def mapping_projection_overlay_path(resolved: ResolvedTargetConfig) -> Path:
    t = resolved.target
    mp = resolved.stages.mapping
    category = (
        f"mapping_tvwcs_os{int(mp.oversampling_factor)}"
        if mp.store_name == "tvwcs" else None
    )
    return scc_debug_plots_dir(resolved.data_root, t.sector, t.camera, t.ccd, category) / (
        MAPPING_PROJECTION_OVERLAY_BASENAME
    )


def _projection_from_skycell_name(name: str) -> str:
    parts = str(name).strip().split(".")
    if len(parts) >= 2 and parts[0] == "skycell":
        return parts[1]
    return "unknown"


def _skycell_projection_lookup(skycells_df: pd.DataFrame) -> dict[str, str]:
    if "NAME" not in skycells_df.columns:
        raise ValueError("skycells CSV missing NAME column")
    out: dict[str, str] = {}
    if "projection" in skycells_df.columns:
        for _, row in skycells_df.iterrows():
            name = str(row["NAME"]).strip()
            out[name] = str(row["projection"]).strip()
        return out
    for _, row in skycells_df.iterrows():
        name = str(row["NAME"]).strip()
        out[name] = _projection_from_skycell_name(name)
    return out


def _load_master_skycell_maps(
    master_path: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    with fits.open(master_path) as hdul:
        master = np.asarray(hdul[1].data)
        name_to_id: dict[str, int] = {}
        if len(hdul) > 2 and hdul[2].data is not None:
            tab = hdul[2].data
            name_to_id = {
                str(n).strip(): int(i) for n, i in zip(tab["SKYCELL"], tab["SKYCIND"])
            }
    return master, name_to_id


def _build_projection_raster(
    master: np.ndarray,
    name_to_id: dict[str, int],
    skycell_to_projection: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    id_to_proj: dict[int, str] = {}
    for name, sid in name_to_id.items():
        id_to_proj[int(sid)] = skycell_to_projection.get(name, "unknown")

    projections = sorted(set(id_to_proj.values()))
    proj_to_code = {proj: idx + 1 for idx, proj in enumerate(projections)}

    max_id = int(master.max()) + 1 if master.size else 1
    lut = np.zeros(max(max_id, 1), dtype=np.int32)
    for sid, proj in id_to_proj.items():
        if 0 <= sid < lut.shape[0]:
            lut[sid] = proj_to_code[proj]

    raster = np.zeros(master.shape, dtype=np.int32)
    valid = master >= 0
    if np.any(valid):
        raster[valid] = lut[master[valid].astype(np.int64)]
    return raster, projections


def write_mapping_projection_overlay(
    master_path: str | Path,
    skycells_csv_path: str | Path,
    out_path: str | Path,
    *,
    sector: int,
    camera: int,
    ccd: int,
) -> Path | None:
    """
    Overlay PS1 projection regions on the TESS FFI pixel grid.

    Colors distinguish projection IDs; boundaries mark projection seams only
    (not individual skycell edges).
    """
    master_path = Path(master_path)
    skycells_csv_path = Path(skycells_csv_path)
    out_path = Path(out_path)
    if not master_path.is_file():
        raise FileNotFoundError(f"master pixels2skycells missing: {master_path}")
    if not skycells_csv_path.is_file():
        raise FileNotFoundError(f"master skycells CSV missing: {skycells_csv_path}")

    master, name_to_id = _load_master_skycell_maps(master_path)
    skycells_df = pd.read_csv(skycells_csv_path)
    skycell_to_projection = _skycell_projection_lookup(skycells_df)
    raster, projections = _build_projection_raster(master, name_to_id, skycell_to_projection)
    if not projections:
        log.warning("No projections found for mapping overlay; skipping.")
        return None

    n_proj = len(projections)
    base_cmap = plt.colormaps.get_cmap("tab20")
    colors = [base_cmap(i / max(n_proj, 1)) for i in range(n_proj)]
    colors.insert(0, (0, 0, 0, 0))
    cmap = ListedColormap(colors)

    display = np.ma.masked_where(raster <= 0, raster)
    boundaries = find_boundaries(raster, mode="outer", background=0)

    fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
    ax.imshow(display, origin="lower", cmap=cmap, interpolation="nearest")
    yy, xx = np.where(boundaries)
    ax.scatter(
        xx,
        yy,
        s=0.15,
        c="black",
        alpha=0.65,
        linewidths=0,
        marker="s",
        rasterized=True,
    )
    ax.set_title(
        f"S{int(sector):04d} C{int(camera)} K{int(ccd)} — PS1 projection regions"
    )
    ax.set_xlabel("TESS x (FFI pixels)")
    ax.set_ylabel("TESS y (FFI pixels)")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=8,
            markerfacecolor=colors[i + 1],
            markeredgecolor="black",
            label=str(proj),
        )
        for i, proj in enumerate(projections)
    ]
    ax.legend(handles=handles, title="projection", loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote mapping projection overlay: %s", out_path)
    return out_path


def write_mapping_projection_overlay_for_scc(
    resolved: ResolvedTargetConfig,
    *,
    force_rerun: bool = False,
) -> Path | None:
    """Best-effort mapping overlay under SCC ``debug_plots/``."""
    out_path = mapping_projection_overlay_path(resolved)
    if out_path.is_file() and not force_rerun:
        return out_path

    t = resolved.target
    os_factor = int(resolved.stages.mapping.oversampling_factor or 1)
    master_path = mapping_master_pixels2skycells_path(resolved)
    skycells_csv_path = scc_mapping_master_skycells_csv(
        resolved.data_root,
        t.sector,
        t.camera,
        t.ccd,
        oversampling_factor=os_factor,
    )
    return write_mapping_projection_overlay(
        master_path,
        skycells_csv_path,
        out_path,
        sector=t.sector,
        camera=t.camera,
        ccd=t.ccd,
    )
