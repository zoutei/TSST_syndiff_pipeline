"""Publish the SCC-level temporal Chebyshev WCS artifact.

The stage is deliberately independent of ``diff_linear`` output paths: its
inputs are read from that lane, while the published model is written below the
SCC ``wcs/`` directory for mapping and template stages to consume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from syndiff_pipeline.common.scc_paths import (
    scc_ffi_list_parquet,
    scc_ffi_dir,
    scc_ffi_list_parquet,
    scc_per_ffi_wcs_dir,
    scc_temporal_wcs_dir,
    scc_wcs_debug_dir,
)
from syndiff_pipeline.difference_imaging.stages.per_ffi_wcs import list_frames_for_lane
from syndiff_pipeline.difference_imaging.wcs.reference import reference_wcs_from_tesswcs
from syndiff_pipeline.difference_imaging.wcs.sci2idl import (
    StarSelectionConfig,
    crop_bounds_from_header,
    join_stars,
    select_good_stars,
)
from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import (
    temporal_frame_contract,
    validate_temporal_frame_contract,
    TemporalChebWcs,
    canonical_temporal_wcs_stem,
    fit_per_ffi_chebyshev,
    fit_temporal_coefficients,
)

log = logging.getLogger(__name__)

MODEL_VERSION = "temporal_cheb5_bspline_v1"
ORBIT_BSPLINE_PREDICTION = "orbit_bspline_prediction"


def republish_temporal_runtime_index(
    model_dir: str | Path, *, data_root: str | Path, sector: int, camera: int, ccd: int
) -> tuple[int, int]:
    """Extend an existing temporal artifact to every local FFI without refitting.

    The orbit NPZ models and per-FFI fit products are left untouched.  Only the
    runtime frame index and manifest counts/ranges are republished, adding
    FFIs that have no usable header/centroid fit as B-spline predictions.
    """
    model_dir = Path(model_dir)
    frames_path = model_dir / "frames.parquet"
    manifest_path = model_dir / "manifest.json"
    if not frames_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete temporal WCS artifact: {model_dir}")
    ffi_path = scc_ffi_list_parquet(data_root, sector, camera, ccd)
    ffi_df = pd.read_parquet(ffi_path)
    if "filename" not in ffi_df.columns:
        raise RuntimeError(f"FFI list lacks filename column: {ffi_path}")
    runtime = _extend_runtime_frame_index(
        pd.read_parquet(frames_path),
        ffi_list_df=ffi_df,
        ffi_filenames=ffi_df["filename"].astype(str).tolist(),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = []
    for spec in manifest.get("models", []):
        orbit = int(spec["orbit_index"])
        rows = runtime.index[runtime["orbit_index"].astype(int) == orbit]
        if len(rows):
            spec = dict(spec)
            spec["start"] = int(rows.min())
            spec["end"] = int(rows.max()) + 1
        specs.append(spec)
    manifest["models"] = specs
    manifest["n_frames"] = int(len(runtime))
    manifest["n_runtime_frames"] = int(len(runtime))
    manifest["n_fit"] = int((runtime["fit_provenance"] == "fit").sum())
    manifest["n_orbit_bspline_predictions"] = int(
        (runtime["fit_provenance"] == ORBIT_BSPLINE_PREDICTION).sum()
    )
    tmp_frames = frames_path.with_suffix(".parquet.tmp")
    runtime.to_parquet(tmp_frames, index=False)
    os.replace(tmp_frames, frames_path)
    _atomic_json(manifest_path, manifest)
    return int(manifest["n_fit"]), int(manifest["n_frames"])


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _orbit_bounds(frames, sector: int) -> np.ndarray:
    """Use the same MIT orbit partitioner as field remapping."""
    from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
        ensure_tess_orbit_times_csv,
    )
    from syndiff_pipeline.template_creation.processing.shift_schedule import (
        _split_orbit_segments_from_csv,
    )

    dates = [fits.getheader(frame.hp_d_path, ext=1).get("DATE-OBS") for frame in frames]
    if any(not date for date in dates):
        raise RuntimeError("temporal_wcs requires DATE-OBS for every hp_d frame")
    return _split_orbit_segments_from_csv(
        int(sector), dates, ensure_tess_orbit_times_csv()
    )


def _btjd_from_date_obs(value: object, *, stem: str) -> float:
    """Convert one authoritative FFI-list DATE-OBS to BTJD, fail closed."""
    from astropy.time import Time

    date_obs = str(value or "").strip()
    if not date_obs:
        raise RuntimeError(f"temporal_wcs: FFI {stem!r} lacks DATE-OBS in ffi_list")
    try:
        btjd = float(Time(date_obs, format="isot", scale="utc").jd - 2457000.0)
    except Exception as exc:
        raise RuntimeError(
            f"temporal_wcs: invalid DATE-OBS for FFI {stem!r}: {date_obs!r}"
        ) from exc
    if not np.isfinite(btjd):
        raise RuntimeError(f"temporal_wcs: non-finite BTJD for FFI {stem!r}")
    return btjd


def _nearest_orbit_index(btjd: float, fitted_frames: pd.DataFrame) -> int:
    """Assign a runtime-only cadence to the closest fitted temporal orbit.

    This deliberately permits the first/last valid FFI cadence to sit just
    outside the centroid lane's sampled endpoints.  TemporalChebWcs clamps
    its B-spline coordinate at those endpoints, so this is an evaluation of
    the adjacent orbit model rather than a fabricated per-FFI fit.
    """
    ranges = (
        fitted_frames.groupby("orbit_index", sort=True)["btjd"]
        .agg(["min", "max"])
        .reset_index()
    )
    if ranges.empty or not np.isfinite(ranges[["min", "max"]].to_numpy()).all():
        raise RuntimeError("temporal_wcs: cannot assign runtime FFI to a temporal orbit")
    t = float(btjd)
    distances = np.maximum(
        ranges["min"].to_numpy(dtype=float) - t,
        np.maximum(0.0, t - ranges["max"].to_numpy(dtype=float)),
    )
    return int(ranges.iloc[int(np.argmin(distances))]["orbit_index"])


def _extend_runtime_frame_index(
    fitted_frames: pd.DataFrame,
    *,
    ffi_list_df: pd.DataFrame,
    ffi_filenames: list[str],
) -> pd.DataFrame:
    """Add local FFIs missing from the centroid lane as orbit predictions.

    Per-FFI Chebyshev fit products remain exactly the rows in
    ``fitted_frames``.  This function constructs the *temporal runtime*
    index, which must cover every local FFI so mapping/remap can evaluate the
    published B-spline even for a cadence with no valid SPOC header WCS.
    """
    required: dict[str, str] = {}
    for logical in ffi_filenames:
        key = canonical_temporal_wcs_stem(logical)
        if key in required and required[key] != str(logical):
            raise RuntimeError(
                f"temporal_wcs: ambiguous local FFI keys {required[key]!r} and {logical!r}"
            )
        required[key] = str(logical)

    runtime = fitted_frames.copy()
    runtime["stem"] = runtime["stem"].astype(str).map(canonical_temporal_wcs_stem)
    if runtime["stem"].duplicated().any():
        raise RuntimeError("temporal_wcs: duplicate fitted temporal-WCS stem")
    fitted_keys = set(runtime["stem"])
    absent_fitted = fitted_keys.difference(required)
    if absent_fitted:
        sample = ", ".join(sorted(absent_fitted)[:3])
        raise RuntimeError(
            "temporal_wcs: centroid lane contains frames absent from local FFI set: " + sample
        )

    runtime["fit_frame_index"] = runtime.get("frame_index", pd.Series(-1, index=runtime.index))
    runtime["runtime_source"] = "centroid_lane"
    additions: list[dict] = []
    for key, logical in required.items():
        if key in fitted_keys:
            continue
        if logical not in ffi_list_df.index:
            raise RuntimeError(f"temporal_wcs: FFI {logical!r} missing from ffi_list")
        row = ffi_list_df.loc[logical]
        additions.append(
            {
                "stem": key,
                "btjd": _btjd_from_date_obs(row.get("date_obs"), stem=logical),
                "fit_frame_index": -1,
                "n_stars_qc": 0,
                "fit_provenance": ORBIT_BSPLINE_PREDICTION,
                "median_residual": np.nan,
                "ffi_wcs_ok": bool(row.get("wcs_ok", False)),
                "runtime_source": "ffi_list_no_header_wcs",
            }
        )
    if additions:
        extra = pd.DataFrame(additions)
        extra["orbit_index"] = [
            _nearest_orbit_index(float(t), runtime) for t in extra["btjd"]
        ]
        runtime = pd.concat([runtime, extra], ignore_index=True, sort=False)
    if runtime["orbit_index"].isna().any() or (runtime["orbit_index"] < 0).any():
        raise RuntimeError("temporal_wcs: runtime frame index has no temporal orbit")
    if not np.isfinite(pd.to_numeric(runtime["btjd"], errors="coerce")).all():
        raise RuntimeError("temporal_wcs: runtime frame index has non-finite BTJD")
    runtime = runtime.sort_values(["btjd", "stem"], kind="mergesort").reset_index(drop=True)
    runtime["frame_index"] = np.arange(len(runtime), dtype=np.int32)
    if len(runtime) != len(required) or set(runtime["stem"]) != set(required):
        raise RuntimeError("temporal_wcs: runtime frame index does not cover local FFIs exactly")
    return runtime


def _write_debug_plots(
    debug_dir: Path,
    coeffs: pd.DataFrame,
    frames: pd.DataFrame,
    temporal_models: list[tuple[int, TemporalChebWcs]] | None = None,
) -> None:
    """Best-effort plots; absence of matplotlib must not invalidate science data."""
    try:
        import matplotlib.pyplot as plt

        debug_dir.mkdir(parents=True, exist_ok=True)
        coeff_cols = [c for c in coeffs.columns if c.startswith(("cx_", "cy_"))]
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in coeff_cols:
            ax.plot(coeffs["btjd"], coeffs[col], ".", ms=2, label=col)
        ax.set(xlabel="BTJD", ylabel="per-FFI Chebyshev coefficient")
        if len(coeff_cols) <= 20:
            ax.legend(ncol=3, fontsize=6)
        fig.tight_layout()
        fig.savefig(debug_dir / "per_ffi_coefficients.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frames["btjd"], frames["median_residual"], ".")
        ax.set(xlabel="BTJD", ylabel="median residual (pixel)")
        fig.tight_layout()
        fig.savefig(debug_dir / "residuals_vs_time.png", dpi=150)
        plt.close(fig)

        # Keep the temporal B-spline diagnostic separate from the per-FFI
        # scatter plot.  This makes it possible to see whether a smooth model
        # is actually interpolating the fitted rows (and not merely that the
        # rows exist).
        if temporal_models:
            from scipy.interpolate import BSpline

            fig, ax = plt.subplots(figsize=(12, 5))
            for orbit, model in temporal_models:
                t = np.linspace(0.0, 1.0, 200)
                basis = BSpline.design_matrix(
                    t, model.knot_vector, model.spline_degree, extrapolate=False
                ).toarray()
                values = basis @ model.coeff_matrix.T
                for col in range(values.shape[1]):
                    ax.plot(
                        model.btjd_ref + model.btjd_scale * t,
                        values[:, col],
                        lw=0.6,
                        alpha=0.35,
                    )
            ax.set(xlabel="BTJD", ylabel="temporal Chebyshev coefficient")
            ax.set_title("Temporal B-spline WCS coefficients")
            fig.tight_layout()
            fig.savefig(debug_dir / "temporal_bspline_coefficients.png", dpi=150)
            plt.close(fig)

        summary = {
            "n_frames": int(len(frames)),
            "n_fit": int((frames["fit_provenance"] == "fit").sum()),
            "n_predicted": int((frames["fit_provenance"] == "predicted").sum()),
            "n_orbits": int(frames["orbit_index"].nunique())
            if "orbit_index" in frames
            else None,
            "median_residual": float(np.nanmedian(frames["median_residual"])),
            "p95_residual": float(np.nanpercentile(frames["median_residual"], 95))
            if np.isfinite(frames["median_residual"]).any()
            else None,
        }
        _atomic_json(debug_dir / "fit_qc_summary.json", summary)
    except Exception as exc:  # diagnostics must not cause an otherwise valid publish to fail
        log.warning("temporal_wcs debug plots skipped: %s", exc)


def _validate_published_artifacts(
    per_dir: Path,
    model_dir: Path,
    frame_df: pd.DataFrame,
    expected_stems,
    models: list[dict],
) -> None:
    """Fail closed when the published store cannot answer every FFI lookup.

    A rejected centroid frame may legitimately be ``predicted`` and therefore
    have no per-FFI NPZ.  It must nevertheless belong to exactly one temporal
    model.  This check catches the more dangerous cases: missing model files,
    stale frame manifests, and orbit ranges that no longer cover the input.
    """
    if len(frame_df) != len(expected_stems):
        raise RuntimeError("temporal_wcs: frame manifest length mismatch")
    stems = [str(getattr(v, "stem", v)) for v in expected_stems]
    if frame_df["stem"].astype(str).tolist() != stems:
        raise RuntimeError("temporal_wcs: frame manifest is not in input-frame order")
    if frame_df["orbit_index"].isna().any() or (frame_df["orbit_index"] < 0).any():
        raise RuntimeError("temporal_wcs: one or more frames has no temporal model")
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    contract = validate_temporal_frame_contract(manifest.get("frame_contract"))
    domain = manifest.get("domain", {})
    if [int(domain.get("x_min", -1)), int(domain.get("y_min", -1))] != [0, 0]:
        raise RuntimeError("temporal_wcs: model domain must be science-local")
    if [int(domain.get("y_max", -1)), int(domain.get("x_max", -1))] != contract["science_shape"]:
        raise RuntimeError("temporal_wcs: frame contract shape disagrees with model domain")
    if int(manifest.get("n_frames", -1)) != len(expected_stems):
        raise RuntimeError("temporal_wcs: manifest n_frames does not match inputs")
    listed = {int(m["orbit_index"]): m for m in manifest.get("models", [])}
    expected = set(int(v) for v in frame_df["orbit_index"])
    if set(listed) != expected:
        raise RuntimeError("temporal_wcs: manifest orbit list does not cover frames")
    for orbit, spec in listed.items():
        path = model_dir / str(spec["path"])
        if not path.is_file():
            raise RuntimeError(f"temporal_wcs: missing temporal model for orbit {orbit}")
        actual = _fingerprint(path)
        if spec.get("fingerprint") != actual:
            raise RuntimeError(f"temporal_wcs: fingerprint mismatch for orbit {orbit}")
        model = TemporalChebWcs.load(path)
        if model.coeff_matrix.ndim != 2 or model.coeff_matrix.shape[0] != 42:
            raise RuntimeError(f"temporal_wcs: invalid coefficient shape for orbit {orbit}")
    per_manifest = json.loads((per_dir / "manifest.json").read_text(encoding="utf-8"))
    fit_stems = set(str(v) for v in per_manifest.get("models", {}))
    actual_fit_stems = set(frame_df.loc[frame_df["fit_provenance"] == "fit", "stem"].astype(str))
    if fit_stems != actual_fit_stems:
        raise RuntimeError("temporal_wcs: per-FFI manifest disagrees with frame QC")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically replace one parquet artifact without touching its siblings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def republish_temporal_runtime_index(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> dict[str, object]:
    """Republish only the temporal-WCS runtime FFI index.

    This is intentionally a metadata/index repair operation.  It reads the
    existing orbit B-spline NPZs and their fitted-frame index, adds local FFIs
    absent from that index as ``orbit_bspline_prediction`` rows, and atomically
    replaces only ``temporal_cheb5_bspline_v1/frames.parquet`` and its
    ``manifest.json``.  It never refits or rewrites orbit models or any
    ``per_ffi_cheb5`` artifact.
    """
    from syndiff_pipeline.common.download import list_local_ffis, manifest_basename_from_local
    from syndiff_pipeline.common.wcs_header_cache import (
        ffi_list_is_complete,
        load_ffi_list,
    )

    model_dir = scc_temporal_wcs_dir(data_root, sector, camera, ccd)
    per_dir = scc_per_ffi_wcs_dir(data_root, sector, camera, ccd)
    manifest_path = model_dir / "manifest.json"
    frames_path = model_dir / "frames.parquet"
    if not manifest_path.is_file() or not frames_path.is_file():
        raise FileNotFoundError(
            f"temporal_wcs republish requires existing {manifest_path} and {frames_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_kind") != "temporal_wcs":
        raise RuntimeError(f"{model_dir}: not a temporal_wcs artifact")
    if str(manifest.get("version")) != MODEL_VERSION:
        raise RuntimeError(
            f"{model_dir}: expected temporal WCS version {MODEL_VERSION!r}, "
            f"found {manifest.get('version')!r}"
        )
    fitted_frames = pd.read_parquet(frames_path)
    required_columns = {"stem", "btjd", "fit_provenance", "orbit_index"}
    missing_columns = sorted(required_columns.difference(fitted_frames.columns))
    if missing_columns:
        raise RuntimeError(
            f"temporal_wcs: existing runtime index missing columns {missing_columns}"
        )
    # Validate the existing publication before deriving an expanded runtime
    # index.  The subsequent checks validate every prospective orbit/model
    # relation without staging or copying any orbit NPZs.
    _validate_published_artifacts(
        per_dir,
        model_dir,
        fitted_frames,
        fitted_frames["stem"].astype(str).tolist(),
        list(manifest.get("models", [])),
    )

    ffi_dir = scc_ffi_dir(data_root, sector, camera, ccd)
    local_ffis = sorted(list_local_ffis(str(ffi_dir), sector, camera, ccd))
    if not local_ffis:
        raise RuntimeError(f"temporal_wcs: no local FFIs under {ffi_dir}")
    ffi_list = load_ffi_list(scc_ffi_list_parquet(data_root, sector, camera, ccd))
    if not ffi_list_is_complete(local_ffis, ffi_list):
        raise RuntimeError(
            "temporal_wcs: ffi_list is incomplete; refusing to publish a partial runtime index"
        )
    runtime_frames = _extend_runtime_frame_index(
        fitted_frames,
        ffi_list_df=ffi_list,
        ffi_filenames=[manifest_basename_from_local(p) for p in local_ffis],
    )

    models = [dict(spec) for spec in manifest.get("models", [])]
    listed_orbits = {int(spec["orbit_index"]) for spec in models}
    runtime_orbits = set(runtime_frames["orbit_index"].astype(int))
    if listed_orbits != runtime_orbits:
        raise RuntimeError(
            "temporal_wcs: existing models do not cover the republished runtime index"
        )
    for spec in models:
        orbit = int(spec["orbit_index"])
        model_path = model_dir / str(spec["path"])
        if not model_path.is_file():
            raise RuntimeError(f"temporal_wcs: missing temporal model for orbit {orbit}")
        if str(spec.get("fingerprint")) != _fingerprint(model_path):
            raise RuntimeError(f"temporal_wcs: fingerprint mismatch for orbit {orbit}")
        rows = runtime_frames.index[runtime_frames["orbit_index"] == orbit]
        spec.setdefault("fit_start", int(spec.get("start", 0)))
        spec.setdefault("fit_end", int(spec.get("end", 0)))
        spec["start"] = int(rows.min())
        spec["end"] = int(rows.max()) + 1

    updated_manifest = dict(manifest)
    updated_manifest["models"] = models
    updated_manifest["n_frames"] = int(len(runtime_frames))
    updated_manifest["n_fit_input_frames"] = int(
        (runtime_frames["fit_provenance"] != ORBIT_BSPLINE_PREDICTION).sum()
    )
    updated_manifest["n_fit"] = int(
        (runtime_frames["fit_provenance"] == "fit").sum()
    )
    updated_manifest["n_orbit_bspline_predictions"] = int(
        (runtime_frames["fit_provenance"] == ORBIT_BSPLINE_PREDICTION).sum()
    )

    _atomic_parquet(frames_path, runtime_frames)
    _atomic_json(manifest_path, updated_manifest)
    return {
        "model_dir": str(model_dir),
        "n_frames": int(len(runtime_frames)),
        "n_fit": int(updated_manifest["n_fit"]),
        "n_orbit_bspline_predictions": int(
            updated_manifest["n_orbit_bspline_predictions"]
        ),
    }


def _write_debug_fits(debug_dir: Path, frames, *, version: str) -> None:
    out = debug_dir / "fits"
    out.mkdir(parents=True, exist_ok=True)
    for frame in (frames[0], frames[len(frames) // 2], frames[-1]):
        header = fits.getheader(frame.hp_d_path, ext=1).copy()
        header["WCSMODEL"] = ("temporal_wcs", "debug only; NPZ is authoritative")
        header["WCSVERS"] = version
        fits.PrimaryHDU(header=header).writeto(out / f"{frame.stem}_temporal_cheb5.fits", overwrite=True)


def run_temporal_wcs_all_frames(
    lane_root: str,
    gaia_df: pd.DataFrame,
    params,
    *,
    centroids_label: str,
    hp_d_label: str,
    data_root: str,
    sector: int,
    camera: int,
    ccd: int,
) -> tuple[int, int]:
    """Fit per-FFI Chebyshev rows and publish orbit-wise temporal models."""
    frames = list_frames_for_lane(Path(lane_root), centroids_label=centroids_label, hp_d_label=hp_d_label)
    if not frames:
        raise RuntimeError(f"No centroid frames found under {lane_root!r}")
    crop = crop_bounds_from_header(fits.getheader(frames[0].hp_d_path, ext=1))
    ny, nx = crop["shape"]
    reference = reference_wcs_from_tesswcs(int(sector), int(camera), int(ccd), crop)
    center = np.array([nx / 2.0, ny / 2.0])
    half = np.array([max(nx / 2.0, 1.0), max(ny / 2.0, 1.0)])
    selector = StarSelectionConfig(
        clip_n_sigma=float(params.clip_n_sigma), clip_max_iter=int(params.clip_max_iter),
        min_stars=int(params.min_stars),
    )
    n_terms = (int(params.cheb_degree) + 1) * (int(params.cheb_degree) + 2) // 2
    rows: list[dict] = []
    coeff_rows: list[dict] = []
    coeff_matrix = np.full((len(frames), 2 * n_terms), np.nan)
    for idx, frame in enumerate(frames):
        phot = Table.read(frame.phot_path, format="ascii.ecsv")
        good = select_good_stars(join_stars(phot, gaia_df), selector)
        row = {"stem": frame.stem, "btjd": float(frame.btjd), "frame_index": idx,
               "n_stars_qc": int(len(good)), "fit_provenance": "predicted",
               "median_residual": np.nan}
        if len(good) >= int(params.min_stars):
            fit = fit_per_ffi_chebyshev(
                reference, good["ra"].to_numpy(), good["dec"].to_numpy(),
                good["x_fit"].to_numpy(), good["y_fit"].to_numpy(),
                center=center, half_extents=half, poly_degree=int(params.cheb_degree),
                n_sigma=float(params.clip_n_sigma), max_iter=int(params.clip_max_iter),
            )
            if int(fit["n_stars"]) >= int(params.min_stars):
                coeff_matrix[idx] = np.r_[fit["coeff_x"], fit["coeff_y"]]
                row["fit_provenance"] = "fit"
                row["median_residual"] = float(np.nanmedian(fit["residual"][fit["keep_mask"]]))
                coeff_row = {"stem": frame.stem, "btjd": float(frame.btjd), "fit_ok": True}
                coeff_row.update({f"cx_{j}": float(v) for j, v in enumerate(fit["coeff_x"])})
                coeff_row.update({f"cy_{j}": float(v) for j, v in enumerate(fit["coeff_y"])})
                coeff_rows.append(coeff_row)
        rows.append(row)
    frame_df = pd.DataFrame(rows)
    if not coeff_rows:
        raise RuntimeError("temporal_wcs: no per-FFI Chebyshev fit reached min_stars")

    per_dir = scc_per_ffi_wcs_dir(data_root, sector, camera, ccd)
    model_dir = scc_temporal_wcs_dir(data_root, sector, camera, ccd)
    debug_dir = scc_wcs_debug_dir(data_root, sector, camera, ccd)
    per_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coeff_rows).to_parquet(per_dir / "coeffs.parquet", index=False)
    frame_df.to_parquet(per_dir / "frames.parquet", index=False)
    per_models = per_dir / "models"
    per_models.mkdir(exist_ok=True)
    per_model_paths: dict[str, str] = {}
    for idx, frame in enumerate(frames):
        if np.isfinite(coeff_matrix[idx]).all():
            rel = Path("models") / f"{frame.stem}.npz"
            np.savez_compressed(per_dir / rel, coeff_x=coeff_matrix[idx, :n_terms],
                                coeff_y=coeff_matrix[idx, n_terms:], btjd=float(frame.btjd))
            per_model_paths[frame.stem] = str(rel)

    bounds = _orbit_bounds(frames, sector)
    models: list[dict] = []
    temporal_models: list[tuple[int, TemporalChebWcs]] = []
    model_indices = np.full(len(frames), -1, dtype=int)
    for orbit_idx, (start, end) in enumerate(bounds):
        seg_times = frame_df.loc[start:end - 1, "btjd"].to_numpy()
        seg_coeffs = coeff_matrix[start:end]
        if np.isfinite(seg_coeffs).all(axis=1).sum() >= 2:
            smooth = fit_temporal_coefficients(
                seg_times, seg_coeffs, n_interior=int(params.n_interior_knots),
                spline_degree=int(params.spline_degree),
            )
        else:
            valid = np.flatnonzero(np.isfinite(seg_coeffs).all(axis=1))
            if not len(valid):
                raise RuntimeError(f"temporal_wcs: orbit {orbit_idx} has no valid per-FFI fits")
            # A one-fit orbit remains evaluable and explicitly predict-only;
            # repeat the constant coefficient over the clamped cubic basis.
            smooth = {"btjd_ref": float(seg_times[valid[0]]), "btjd_scale": 1.0,
                      "knot_vector": np.r_[np.zeros(4), np.ones(4)],
                      "coeff_matrix": np.repeat(seg_coeffs[valid[0]][:, None], 4, axis=1)}
        model = TemporalChebWcs.from_reference_wcs(
            reference, center=center, half_extents=half, poly_degree=int(params.cheb_degree),
            btjd_ref=smooth["btjd_ref"], btjd_scale=smooth["btjd_scale"],
            knot_vector=smooth["knot_vector"], spline_degree=int(params.spline_degree),
            coeff_matrix=smooth["coeff_matrix"],
        )
        rel = Path("models") / f"orbit_{orbit_idx:02d}.npz"
        model.save(model_dir / rel)
        model_indices[start:end] = orbit_idx
        models.append({"orbit_index": orbit_idx, "start": int(start), "end": int(end),
                       "path": str(rel), "fingerprint": _fingerprint(model_dir / rel)})
        temporal_models.append((orbit_idx, model))
    frame_df["orbit_index"] = model_indices
    # The fit lane omits FFIs with no valid SPOC header WCS.  Those cadences
    # still belong to the science FFI sequence and must be evaluable by
    # mapping/remap using their orbit B-spline, so publish a separate runtime
    # index covering every local FFI.  Do not add these prediction-only rows
    # to per_ffi_cheb5/: that directory represents actual per-FFI fits.
    from syndiff_pipeline.common.download import list_local_ffis, manifest_basename_from_local
    from syndiff_pipeline.common.wcs_header_cache import (
        ffi_list_is_complete,
        load_ffi_list,
    )

    ffi_dir = scc_ffi_dir(data_root, sector, camera, ccd)
    local_ffis = sorted(list_local_ffis(str(ffi_dir), sector, camera, ccd))
    if not local_ffis:
        raise RuntimeError(f"temporal_wcs: no local FFIs under {ffi_dir}")
    ffi_list = load_ffi_list(scc_ffi_list_parquet(data_root, sector, camera, ccd))
    if not ffi_list_is_complete(local_ffis, ffi_list):
        raise RuntimeError(
            "temporal_wcs: ffi_list is incomplete; refusing to publish a partial runtime index"
        )
    runtime_frame_df = _extend_runtime_frame_index(
        frame_df,
        ffi_list_df=ffi_list,
        ffi_filenames=[manifest_basename_from_local(p) for p in local_ffis],
    )
    for spec in models:
        orbit = int(spec["orbit_index"])
        rows = runtime_frame_df.index[runtime_frame_df["orbit_index"] == orbit]
        if len(rows) == 0:
            raise RuntimeError(f"temporal_wcs: temporal orbit {orbit} has no runtime frames")
        spec["fit_start"] = int(spec["start"])
        spec["fit_end"] = int(spec["end"])
        spec["start"] = int(rows.min())
        spec["end"] = int(rows.max()) + 1
    runtime_frame_df.to_parquet(model_dir / "frames.parquet", index=False)
    manifest = {
        "model_kind": "temporal_wcs", "version": MODEL_VERSION,
        "spatial_basis": "chebyshev", "spatial_degree": int(params.cheb_degree),
        "temporal_basis": "bspline", "temporal_spline_degree": int(params.spline_degree),
        "coordinate_direction": "gaia_tan_to_detector_pixel", "pixel_origin": 0,
        "domain": {"x_min": 0, "x_max": int(nx), "y_min": 0, "y_max": int(ny)},
        "sector": int(sector), "camera": int(camera), "ccd": int(ccd),
        "n_frames": len(runtime_frame_df),
        "n_fit_input_frames": len(frames),
        "n_fit": int(np.isfinite(coeff_matrix).all(axis=1).sum()),
        "n_orbit_bspline_predictions": int(
            (runtime_frame_df["fit_provenance"] == ORBIT_BSPLINE_PREDICTION).sum()
        ),
        "models": models,
    }
    # The temporal model is fitted in the cropped reference-WCS frame.  Keep
    # the crop origin explicit so runtime callers cannot accidentally evaluate
    # it with full-FFI coordinates.
    manifest["frame_contract"] = temporal_frame_contract(
        origin_ffi=(crop["x_min"], crop["y_min"]), shape=(ny, nx), pixel_origin=0
    )
    _atomic_json(
        per_dir / "manifest.json",
        {
            **manifest,
            "kind": "per_ffi_cheb5",
            "n_frames": len(frames),
            "n_runtime_frames": len(runtime_frame_df),
            "models": per_model_paths,
        },
    )
    _atomic_json(model_dir / "manifest.json", manifest)
    _validate_published_artifacts(
        per_dir,
        model_dir,
        runtime_frame_df,
        runtime_frame_df["stem"].astype(str).tolist(),
        models,
    )
    if bool(params.debug_plots):
        _write_debug_plots(debug_dir, pd.DataFrame(coeff_rows), runtime_frame_df, temporal_models)
        _write_debug_fits(debug_dir, frames, version=MODEL_VERSION)
    return int(np.isfinite(coeff_matrix).all(axis=1).sum()), len(runtime_frame_df)
