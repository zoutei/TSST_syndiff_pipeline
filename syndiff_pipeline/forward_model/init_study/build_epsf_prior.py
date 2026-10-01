# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Build pooled 2×2 ePSF prior for forward_epsf_wcs init studies."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.common.download import list_local_ffis
from syndiff_pipeline.common.scc_paths import scc_ffi_dir, scc_ffi_list_parquet
from syndiff_pipeline.common.wcs_header_cache import load_ffi_list
from syndiff_pipeline.difference_imaging.masking.ffi_mask import (
    load_catalog_for_scc_lane,
    load_ffi_times_table_for_lane,
)
from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
    ffi_path_by_stem_from_wcs_table,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    tess_product_id_from_ffi_path,
)

from syndiff_pipeline.forward_model.data import RegionSpec, list_orbit_frames
from syndiff_pipeline.forward_model.init_study.photutils_to_forward_epsf import (
    gridded_stack_to_epsf_base,
    save_epsf_base_init,
)
from syndiff_pipeline.forward_model._vendor.ref_epsf_photometry.orbit_windows import _central_slice
from syndiff_pipeline.forward_model._vendor.ref_epsf_photometry.pool_epsf import (
    EpsfBuildParams,
    _PreparedFrame,
    _TileResult,
    _fit_pooled_tile,
    save_window_npz,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StudyConfig:
    workspace: Path
    sector: int
    camera: int
    ccd: int
    orbit_index: int
    region: RegionSpec
    region_margin_px: float
    n_mid_ffis: int
    tile_nx: int
    tile_ny: int
    mag_lo: float
    mag_hi: float
    study_dir: Path


def _load_config(path: Path) -> StudyConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    region = raw["region"]
    mag = raw["epsf_pool"]["tess_mag"]
    study_dir = path.parent
    return StudyConfig(
        workspace=(_REPO / raw["workspace"]).resolve(),
        sector=int(raw["sector"]),
        camera=int(raw["camera"]),
        ccd=int(raw["ccd"]),
        orbit_index=int(raw["orbit_index"]),
        region=RegionSpec(*[int(v) for v in region]),
        region_margin_px=float(raw.get("region_margin_px", 15.0)),
        n_mid_ffis=int(raw["epsf_pool"]["n_mid_ffis"]),
        tile_nx=int(raw["epsf_pool"]["tile_nx"]),
        tile_ny=int(raw["epsf_pool"]["tile_ny"]),
        mag_lo=float(mag[0]),
        mag_hi=float(mag[1]),
        study_dir=study_dir,
    )


def _hp_d_stem_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".fits.fz", ".fits.gz", ".fits"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.endswith("_hp_d"):
        name = name[: -len("_hp_d")]
    return tess_product_id_from_ffi_path(name) or name


def _stem_to_product_id(stem: str) -> str:
    pid = tess_product_id_from_ffi_path(stem)
    if pid:
        return pid
    return stem.split("-")[0]


def _collect_hp_d_index(hp_d_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(hp_d_dir.glob("tess*")):
        if not path.is_file():
            continue
        stem = _hp_d_stem_from_path(path)
        out[stem] = path
        out[_stem_to_product_id(stem)] = path
    return out


def _ordered_stems_btjd(
    lane_root: Path,
    data_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    hp_index: dict[str, Path],
) -> tuple[list[str], dict[str, float]]:
    timing = load_ffi_times_table_for_lane(
        lane_root,
        data_root=data_root,
        sector=sector,
        camera=camera,
        ccd=ccd,
    )
    rows: list[tuple[str, float]] = []
    for _, row in timing.iterrows():
        fn = str(row.get("filename", ""))
        stem = tess_product_id_from_ffi_path(fn) or Path(fn).stem
        if stem not in hp_index:
            continue
        btjd = float(pd.to_numeric(row.get("btjd"), errors="coerce"))
        if not np.isfinite(btjd):
            continue
        rows.append((stem, btjd))
    rows.sort(key=lambda t: t[1])
    stems = [r[0] for r in rows]
    btjd_by_stem = {r[0]: r[1] for r in rows}
    return stems, btjd_by_stem


def _mid_orbit_stems(
    workspace: Path,
    *,
    sector: int,
    orbit_index: int,
    n_mid: int,
) -> list[str]:
    frames, _ = list_orbit_frames(workspace, sector=sector, orbit_index=orbit_index)
    stems = [f.stem for f in frames]
    if len(stems) < n_mid:
        raise ValueError(f"orbit {orbit_index} has {len(stems)} frames (< {n_mid})")
    sl = _central_slice(len(stems), n_mid)
    return stems[sl]


def _crop_prepared_frames(
    prepared: list[_PreparedFrame],
    *,
    x0: int,
    y0: int,
) -> list[_PreparedFrame]:
    cropped: list[_PreparedFrame] = []
    for frame in prepared:
        img = np.asarray(frame.diff_img)[y0:, x0:]
        mask = None
        if frame.full_mask is not None:
            mask = np.asarray(frame.full_mask)[y0:, x0:]
        gaia = frame.gaia_frame.copy()
        gaia["x"] = pd.to_numeric(gaia["x"], errors="coerce") - float(x0)
        gaia["y"] = pd.to_numeric(gaia["y"], errors="coerce") - float(y0)
        cropped.append(
            _PreparedFrame(
                stem=frame.stem,
                diff_img=img.astype(np.float64),
                gaia_frame=gaia,
                full_mask=mask,
            )
        )
    return cropped


def _tile_ckpt_path(ckpt_dir: Path, i: int, j: int) -> Path:
    return ckpt_dir / f"tile_{i}_{j}.npz"


def _tile_progress_log_path(ckpt_dir: Path, i: int, j: int) -> Path:
    return ckpt_dir / f"tile_{i}_{j}.progress.log"


def _fit_pooled_tile_logged(
    i: int,
    j: int,
    *,
    prepared_frames: list[_PreparedFrame],
    ny: int,
    nx: int,
    params: EpsfBuildParams,
    progress_log: str | None,
) -> _TileResult:
    """Worker entry: fit one tile; photutils tqdm goes to ``progress_log`` if set.

    Must stay at module scope so loky can pickle it. Redirects stdout/stderr so
    parallel tiles do not garbled-interleave on the parent TTY.
    """
    import contextlib

    if not progress_log:
        return _fit_pooled_tile(
            i, j, prepared_frames=prepared_frames, ny=ny, nx=nx, params=params
        )
    path = Path(progress_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", buffering=1, encoding="utf-8") as fh:
        fh.write(f"# tile ({i},{j}) photutils EPSFBuilder progress\n")
        fh.flush()
        with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            return _fit_pooled_tile(
                i, j, prepared_frames=prepared_frames, ny=ny, nx=nx, params=params
            )


def _save_tile_ckpt(path: Path, result: _TileResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "i": int(result.i),
        "j": int(result.j),
        "x_center": float(result.x_center),
        "y_center": float(result.y_center),
        "status": str(result.status),
        "n_cutouts": int(result.n_cutouts),
    }
    if result.stamp is not None:
        payload["stamp"] = np.asarray(result.stamp, dtype=np.float64)
    np.savez_compressed(path, **payload)


def _load_tile_ckpt(path: Path) -> _TileResult:
    z = np.load(path, allow_pickle=False)
    stamp = z["stamp"] if "stamp" in z.files else None
    return _TileResult(
        i=int(z["i"]),
        j=int(z["j"]),
        x_center=float(z["x_center"]),
        y_center=float(z["y_center"]),
        stamp=None if stamp is None else np.asarray(stamp),
        status=str(z["status"]),
        n_cutouts=int(z["n_cutouts"]),
    )


def _build_pooled_from_prepared(
    prepared_frames: list[_PreparedFrame],
    params: EpsfBuildParams,
    *,
    tile_ckpt_dir: Path | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]], dict]:
    """Fit tiles in parallel with loky *processes* (not threads — GIL-bound).

    If ``tile_ckpt_dir`` is set, finished tiles are written as soon as each
    completes and are skipped on resume.
    """
    from joblib import delayed

    from syndiff_pipeline.common.joblib_progress import parallel_map_with_optional_tqdm
    from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs

    if not prepared_frames:
        raise RuntimeError("no prepared frames")
    ny, nx = prepared_frames[0].diff_img.shape
    tile_ny, tile_nx = int(params.tile_ny), int(params.tile_nx)
    n_workers = resolve_effective_n_jobs(params.n_jobs)
    # Cap workers to tile count; EPSFBuilder is Python-heavy so processes win.
    n_workers = max(1, min(n_workers, tile_ny * tile_nx))

    cached: dict[tuple[int, int], _TileResult] = {}
    n_cached_before = 0
    if tile_ckpt_dir is not None:
        tile_ckpt_dir.mkdir(parents=True, exist_ok=True)
        for i in range(tile_ny):
            for j in range(tile_nx):
                path = _tile_ckpt_path(tile_ckpt_dir, i, j)
                if path.is_file():
                    cached[(i, j)] = _load_tile_ckpt(path)
                    n_cached_before += 1
                    log.info("resume: loaded tile (%d,%d) from %s", i, j, path)

    pending = [
        (i, j)
        for i in range(tile_ny)
        for j in range(tile_nx)
        if (i, j) not in cached
    ]
    log.info(
        "ePSF tiles: %d cached, %d to fit (n_jobs=%d, backend=loky/processes)",
        n_cached_before,
        len(pending),
        n_workers,
    )
    if tile_ckpt_dir is not None and pending:
        log.info(
            "per-tile photutils progress logs: %s/tile_i_j.progress.log "
            "(tail -f … while tiles run)",
            tile_ckpt_dir,
        )

    def _on_result(result: _TileResult) -> None:
        if tile_ckpt_dir is None:
            return
        _save_tile_ckpt(_tile_ckpt_path(tile_ckpt_dir, result.i, result.j), result)
        log.info(
            "checkpointed tile (%d,%d) status=%s n_cutouts=%d",
            result.i,
            result.j,
            result.status,
            result.n_cutouts,
        )

    if pending:
        delayed_calls = [
            delayed(_fit_pooled_tile_logged)(
                i,
                j,
                prepared_frames=prepared_frames,
                ny=ny,
                nx=nx,
                params=params,
                progress_log=(
                    str(_tile_progress_log_path(tile_ckpt_dir, i, j))
                    if tile_ckpt_dir is not None
                    else None
                ),
            )
            for i, j in pending
        ]
        # prefer=None keeps joblib_progress's loky process backend (true parallel).
        # prefer="threads" was GIL-serializing EPSFBuilder to ~1 core.
        new_results: list[_TileResult] = parallel_map_with_optional_tqdm(
            delayed_calls,
            n_tasks=len(delayed_calls),
            desc="pooled ePSF tiles",
            n_jobs_eff=n_workers,
            prefer=None,
            on_result=_on_result,
        )
        for result in new_results:
            cached[(result.i, result.j)] = result

    tile_results = [cached[(i, j)] for i in range(tile_ny) for j in range(tile_nx)]

    epsf_grid: dict[tuple[int, int], np.ndarray | str] = {}
    tile_cutout_counts: dict[tuple[int, int], int] = {}
    grid_xypos: list[tuple[float, float]] = []
    for result in sorted(tile_results, key=lambda r: (r.i, r.j)):
        grid_xypos.append((result.x_center, result.y_center))
        epsf_grid[(result.i, result.j)] = (
            result.stamp if result.stamp is not None else result.status
        )
        tile_cutout_counts[(result.i, result.j)] = result.n_cutouts

    valid = [v for v in epsf_grid.values() if isinstance(v, np.ndarray)]
    if not valid:
        raise RuntimeError("all grid tiles failed")

    fallback = np.mean(valid, axis=0)
    psf_list: list[np.ndarray] = []
    n_ok = 0
    for i in range(tile_ny):
        for j in range(tile_nx):
            result = epsf_grid.get((i, j), "too_few")
            if isinstance(result, np.ndarray):
                psf_list.append(result)
                n_ok += 1
            else:
                psf_list.append(fallback)

    stack = np.array(psf_list, dtype=np.float64)
    stats = {
        "n_tiles_ok": n_ok,
        "n_tiles_total": tile_nx * tile_ny,
        "tile_cutout_counts": {
            f"{i}_{j}": tile_cutout_counts.get((i, j), 0)
            for i in range(tile_ny)
            for j in range(tile_nx)
        },
        "n_prepared_frames": len(prepared_frames),
        "n_jobs": n_workers,
        "n_tiles_cached": n_cached_before,
        "n_tiles_fit": len(pending),
    }
    return stack, grid_xypos, stats


def _prepare_cropped_reference_frames(
    reference_stems: list[str],
    *,
    hp_d_by_stem: dict[str, Path],
    ffi_path_by_stem: dict[str, str],
    gaia_filtered: pd.DataFrame,
    ffi_list_df: pd.DataFrame,
    science_bounds: dict,
    mask_catalog,
    btjd_by_stem: dict[str, float],
    crop_x0: int,
    crop_y0: int,
) -> list[_PreparedFrame]:
    from astropy.io import fits

    from syndiff_pipeline.common.wcs_grouping import gaia_science_xy_for_frame
    from syndiff_pipeline.difference_imaging.masking.bits import epsf_reject_mask

    prepared: list[_PreparedFrame] = []
    for stem in reference_stems:
        hp_path = hp_d_by_stem.get(stem)
        ffi_path = ffi_path_by_stem.get(stem)
        if hp_path is None or not hp_path.is_file() or ffi_path is None:
            continue
        try:
            diff_img = fits.getdata(hp_path).astype(np.float64)
        except Exception as exc:
            log.warning("skip %s: %s", stem, exc)
            continue
        try:
            gaia_frame = gaia_science_xy_for_frame(
                gaia_filtered, ffi_path, ffi_list_df, science_bounds
            )
        except Exception as exc:
            log.warning("gaia xy failed %s: %s", stem, exc)
            continue
        full_mask = None
        if mask_catalog is not None:
            btjd = btjd_by_stem.get(stem)
            full_mask = np.asarray(
                epsf_reject_mask(mask_catalog.mask_at(btjd, which="full")),
                dtype=bool,
            )
        prepared.append(
            _PreparedFrame(
                stem=stem,
                diff_img=diff_img,
                gaia_frame=gaia_frame,
                full_mask=full_mask,
            )
        )
    return _crop_prepared_frames(prepared, x0=crop_x0, y0=crop_y0)


def build_epsf_prior(cfg: StudyConfig, *, force: bool = False) -> dict:
    study_epsf_dir = cfg.study_dir / "epsf_prior"
    init_path = study_epsf_dir / "epsf_base_init.npz"
    meta_path = study_epsf_dir / "build_meta.json"
    if not force and init_path.is_file() and meta_path.is_file():
        log.info("skip build: %s already exists (pass force=True to rebuild)", init_path)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    lane_root = cfg.workspace
    hp_d_dir = lane_root / "hp_d"
    hp_index = _collect_hp_d_index(hp_d_dir)
    if not hp_index:
        raise FileNotFoundError(f"no hp_d frames under {hp_d_dir}")

    from syndiff_pipeline.common.orchestration.deployment import load_deployment_file
    from syndiff_pipeline.common.scc_paths import scc_diff_dir

    dep = load_deployment_file(_REPO / "config" / "deployment.yaml")
    data_root = Path(dep["data_root"])
    lane = scc_diff_dir(
        data_root,
        cfg.sector,
        cfg.camera,
        cfg.ccd,
        store_name="linear",
    )
    if not lane.is_dir():
        lane = lane_root

    reference_stems = _mid_orbit_stems(
        cfg.workspace,
        sector=cfg.sector,
        orbit_index=cfg.orbit_index,
        n_mid=cfg.n_mid_ffis,
    )
    reference_stems = [_stem_to_product_id(s) for s in reference_stems]
    log.info(
        "orbit_index=%d (1-based first orbit): pooling %d mid FFIs",
        cfg.orbit_index,
        len(reference_stems),
    )

    ffi_dir = scc_ffi_dir(data_root, cfg.sector, cfg.camera, cfg.ccd)
    ffi_paths = list_local_ffis(str(ffi_dir), cfg.sector, cfg.camera, cfg.ccd)
    ffi_path_by_stem = {
        _stem_to_product_id(stem): p
        for p in ffi_paths
        if (stem := tess_product_id_from_ffi_path(p))
    }
    ffi_list_df = load_ffi_list(
        scc_ffi_list_parquet(data_root, cfg.sector, cfg.camera, cfg.ccd)
    )
    if "path" not in ffi_list_df.columns and "filename" in ffi_list_df.columns:
        ffi_list_df = ffi_list_df.rename(columns={"filename": "path"})
    wcs_table = ffi_path_by_stem_from_wcs_table(ffi_list_df)
    for stem, path in wcs_table.items():
        ffi_path_by_stem.setdefault(_stem_to_product_id(stem), path)

    _, btjd_by_stem = _ordered_stems_btjd(
        lane, data_root, cfg.sector, cfg.camera, cfg.ccd, hp_index
    )
    mask_catalog = load_catalog_for_scc_lane(
        lane,
        data_root=data_root,
        sector=cfg.sector,
        camera=cfg.camera,
        ccd=cfg.ccd,
    )
    science_bounds = mask_catalog.crop_bounds or {}

    gaia_csv = lane / "gaia_catalog_pipeline.csv"
    gaia_base = pd.read_csv(gaia_csv)
    params = EpsfBuildParams(
        tile_nx=cfg.tile_nx,
        tile_ny=cfg.tile_ny,
        mag_min_tess=cfg.mag_lo,
        mag_max_tess=cfg.mag_hi,
        n_jobs=cfg.tile_nx * cfg.tile_ny,
        progress_bar=True,  # per-tile logs under tile_ckpts/*.progress.log
    )
    # Inclusive mag bounds for pooling (pool_epsf uses strict inequalities).
    mag = pd.to_numeric(gaia_base["tess_mag"], errors="coerce")
    gaia_filtered = gaia_base.loc[
        np.isfinite(mag) & (mag >= cfg.mag_lo) & (mag <= cfg.mag_hi)
    ].copy().reset_index(drop=True)
    log.info("Gaia after inclusive mag %.1f-%.1f: %d stars", cfg.mag_lo, cfg.mag_hi, len(gaia_filtered))

    margin = int(np.ceil(cfg.region_margin_px))
    crop_x0 = max(0, cfg.region.x_min - margin)
    crop_y0 = max(0, cfg.region.y_min - margin)

    prepared = _prepare_cropped_reference_frames(
        reference_stems,
        hp_d_by_stem=hp_index,
        ffi_path_by_stem=ffi_path_by_stem,
        gaia_filtered=gaia_filtered,
        ffi_list_df=ffi_list_df,
        science_bounds=science_bounds,
        mask_catalog=mask_catalog,
        btjd_by_stem=btjd_by_stem,
        crop_x0=crop_x0,
        crop_y0=crop_y0,
    )
    if not prepared:
        raise RuntimeError("no reference frames prepared after crop")

    ny, nx = prepared[0].diff_img.shape
    log.info("cropped image shape (ny,nx)=(%d,%d) origin=(%d,%d)", ny, nx, crop_x0, crop_y0)

    tile_ckpt_dir = study_epsf_dir / "tile_ckpts"
    stack, grid_xypos, stats = _build_pooled_from_prepared(
        prepared, params, tile_ckpt_dir=tile_ckpt_dir,
    )

    lane_out = lane / "forward_epsf_init"
    try:
        lane_out.mkdir(parents=True, exist_ok=True)
        lane_npz = lane_out / f"orbit{cfg.orbit_index}_mid20_epsf_2x2_mag813.npz"
        save_window_npz(
            lane_npz,
            stack,
            grid_xypos,
            params,
            orbit_idx=cfg.orbit_index - 1,
            window="mid20",
            reference_stems=reference_stems,
            extra_meta={
                "region": [
                    cfg.region.x_min, cfg.region.y_min, cfg.region.x_max, cfg.region.y_max,
                ],
                "crop_origin": [crop_x0, crop_y0],
                "mag_range": [cfg.mag_lo, cfg.mag_hi],
                "orbit_index_1based": cfg.orbit_index,
            },
        )
    except OSError as exc:
        log.warning("cannot write lane forward_epsf_init (%s); saving under study dir", exc)
        study_epsf_dir.mkdir(parents=True, exist_ok=True)
        lane_npz = study_epsf_dir / "orbit1_mid20_epsf_2x2_mag813.npz"
        save_window_npz(
            lane_npz,
            stack,
            grid_xypos,
            params,
            orbit_idx=cfg.orbit_index - 1,
            window="mid20",
            reference_stems=reference_stems,
            extra_meta={
                "region": [
                    cfg.region.x_min, cfg.region.y_min, cfg.region.x_max, cfg.region.y_max,
                ],
                "crop_origin": [crop_x0, crop_y0],
                "mag_range": [cfg.mag_lo, cfg.mag_hi],
                "orbit_index_1based": cfg.orbit_index,
            },
        )

    base = gridded_stack_to_epsf_base(
        stack,
        tile_ny=cfg.tile_ny,
        tile_nx=cfg.tile_nx,
        oversample=params.epsf_oversample,
    )
    study_epsf_dir.mkdir(parents=True, exist_ok=True)
    save_epsf_base_init(init_path, base)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "orbit_index_1based": cfg.orbit_index,
        "reference_stems": reference_stems,
        "region": [cfg.region.x_min, cfg.region.y_min, cfg.region.x_max, cfg.region.y_max],
        "crop_origin": [crop_x0, crop_y0],
        "grid_xypos": grid_xypos,
        "stats": stats,
        "lane_npz": str(lane_npz),
        "epsf_base_init": str(init_path),
        "tile_ckpt_dir": str(tile_ckpt_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s and %s", lane_npz, init_path)
    return meta


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO / "dev/forward_epsf_wcs/init_study/orbit1_midhalf_mag813_center1k/config.yaml",
    )
    p.add_argument("--force", action="store_true", help="rebuild even if epsf_base_init.npz exists")
    args = p.parse_args(argv)
    cfg = _load_config(args.config.resolve())
    meta = build_epsf_prior(cfg, force=args.force)
    print(json.dumps({"ok": True, "epsf_base_init": meta["epsf_base_init"]}, indent=2))


if __name__ == "__main__":
    main()
