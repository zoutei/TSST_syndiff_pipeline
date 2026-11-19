"""Build and measure unweighted target components for a frozen star lane."""

from functools import lru_cache
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.ps1_wcs import table_ps1_crpix_offset
from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
    build_ps1_wcs,
    compute_ps1_shift_for_skycell,
    load_tess_wcs,
)
from syndiff_pipeline.template_creation.processing.field_remap import (
    _find_regmap,
    _master_skycell_id_map,
)
from syndiff_pipeline.template_creation.processing.perband import BANDS

from .component_window import ComponentOperator, convolve_target_window, transport_target
from .context import resolve_star_manifest_path
from .diff_runner import (
    _centered_window_bounds,
    _resolve_frame_fits_path,
    baseline_window,
    hotpants_fit_geometry,
    load_frame_kernel_for_diff,
    star_product_ids,
)
from .identifiers import resolve_host, write_host_gaia_row_csv, write_identifier_json
from .kernel_window import FrameKernel
from .perband_photometry import measure_bands
from .template_recipe import TemplateRecipe, read_donor


def _write_npz(path, **arrays):
    """Replace an archive file only after the full payload has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        staged = Path(tmp.name)
    try:
        np.savez_compressed(staged, **arrays)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def group_schedule(manifest):
    required = ["group_id", "group_dx", "group_dy"]
    if any(col not in manifest for col in required):
        raise ValueError("frames.csv must supply group_id, group_dx and group_dy")
    groups = manifest[required].drop_duplicates()
    if groups.isna().any().any() or groups.group_id.duplicated().any():
        raise ValueError("Each lane group must define one finite offset pair")
    if not np.isfinite(groups[required].to_numpy(float)).all():
        raise ValueError("Lane group offsets must be finite")
    return groups.set_index("group_id").sort_index()


class LaneComponents:
    """Immutable mapping and offset schedule reused for all targets in one lane."""

    def __init__(self, ctx, recipe, groups):
        if ctx.oversampling_factor != 1:
            raise ValueError("Per-band component binning currently requires an OS1 lane")
        self.recipe = recipe
        self.groups = groups
        mapping = recipe.mapping_table or Path(ctx.mapping_csv)
        read_mapping = pd.read_parquet if mapping.suffix == ".parquet" else pd.read_csv
        self.rows = read_mapping(mapping).set_index("NAME", drop=False)
        if table_ps1_crpix_offset(self.rows) != recipe.ps1_crpix:
            raise ValueError("Frozen PS1 CRPIX disagrees with mapping table")
        self.master, ids = _master_skycell_id_map(Path(ctx.master_mapping_fits))
        self.cell_by_index = {value: key for key, value in ids.items()}
        if not ids:
            self.cell_by_index = dict(enumerate(self.rows.NAME))
        with fits.open(ctx.master_mapping_fits) as h:
            header = h[1].header
            self.master_origin = (
                int(header.get("XMIN", recipe.map_origin[0])),
                int(header.get("YMIN", recipe.map_origin[1])),
            )
        self.reference_wcs, _ = load_tess_wcs(recipe.reference_ffi or Path(ctx.reference_ffi_path))
        self.crop_origin = (int(ctx.crop_bounds["x_min"]), int(ctx.crop_bounds["y_min"]))
        self.scc = (ctx.sector, ctx.camera, ctx.ccd)
        self.registration_dir = recipe.registration_dir or Path(ctx.mapping_dir)
        self.data_root = ctx.data_root

    def reference_xy(self, host):
        full = self.reference_wcs.world_to_pixel_values(host.ra, host.dec)
        return tuple(float(v - origin) for v, origin in zip(full, self.crop_origin))

    @lru_cache(maxsize=8)
    def receiver(self, name):
        row = self.rows.loc[name]
        wcs, _ = build_ps1_wcs(row)
        if self.recipe.registration_pattern:
            sector, camera, ccd = self.scc
            path = self.registration_dir / self.recipe.registration_pattern.format(
                sector=sector, camera=camera, ccd=ccd, cell=name
            )
        else:
            path = _find_regmap(self.registration_dir, *self.scc, name, oversampling_factor=1)
        with fits.open(path) as hdul:
            hdu = hdul["TESS_PIXEL_MAP"]
            registration = np.asarray(hdu.data)
            if float(hdu.header.get("PS1CRPIX", self.recipe.ps1_crpix)) != self.recipe.ps1_crpix:
                raise ValueError(f"{name}: regmap PS1 CRPIX disagrees with the recipe")
        shifts = [
            tuple(round(v) for v in compute_ps1_shift_for_skycell(
                self.reference_wcs, dx, dy, row.RA, row.DEC, wcs
            ))
            for dx, dy in self.groups[["group_dx", "group_dy"]].itertuples(index=False, name=None)
        ]
        return name, wcs, registration, shifts

    def target(self, host, reference_xy, size):
        """Resolve all local recipient cells and render once for each distinct group."""
        cx, cy = map(round, reference_xy)
        half = size // 2
        origin = (cx - half, cy - half)
        full_xy = tuple(v + o for v, o in zip(origin, self.crop_origin))
        index_xy = tuple(v - o for v, o in zip(full_xy, self.master_origin))
        x, y = index_xy
        master = self.master[max(y, 0):max(y + size, 0), max(x, 0):max(x + size, 0)]
        cells = [self.cell_by_index[index] for index in np.unique(master)
                 if index in self.cell_by_index]
        if not cells:
            raise ValueError("Target does not intersect a mapped recipient cell")
        reg_origin = tuple(v - o for v, o in zip(full_xy, self.recipe.map_origin))

        @lru_cache(maxsize=18)
        def donor(name):
            return read_donor(self.recipe, name, self.rows.loc[name], host, self.data_root)

        operator = ComponentOperator(self.recipe, self.rows, donor)
        planes = transport_target(
            operator, (self.receiver(name) for name in cells), (host.ra, host.dec),
            self.groups.index, reg_origin, (size, size),
        )
        if self.recipe.reflect_below is not None:
            # The June FITS writer reflected flux rows below the original support.
            boundary = self.recipe.reflect_below - full_xy[1]
            if 0 < boundary < size:
                reflected = 2 * boundary - np.arange(boundary)
                indices = np.flatnonzero(reflected < size)
                planes[..., indices, :] = planes[..., reflected[indices], :]
        flags = operator.component_flags
        identity = "retained"
        if not np.any(planes):
            identity = "no_retained_component"
        if "raw_fallback_provisional" in flags:
            identity = "raw_fallback_provisional"
        if flags.intersection({"truncated", "shared_PS1_footprint"}):
            identity = "truncated" if "truncated" in flags else "shared_PS1_footprint"
        return planes, origin, identity


class CadencePositions:
    """Catalogue-propagated crop-local centroids with an optional frozen sidecar."""

    def __init__(self, ctx, path):
        self.ctx = ctx
        self.tables = {}
        if path:
            path = Path(path)
            table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(
                path, dtype={"source_id": "int64", "product_id": str}
            )
            if table.duplicated(["product_id", "source_id"]).any():
                raise ValueError("Duplicate source and cadence in star positions")
            self.tables = {
                str(pid): rows.set_index("source_id")
                for pid, rows in table.groupby("product_id")
            }
            self.catalog = None
        else:
            table = pd.read_csv(
                ctx.gaia_catalog_path, usecols=lambda col: col in
                ("source_id", "ra", "dec", "pmra", "pmdec")
            )
            self.catalog = table.drop_duplicates("source_id").set_index("source_id")

    @lru_cache(maxsize=3)
    def for_frame(self, product_id, ffi_path, btjd):
        if self.catalog is None:
            table = self.tables[product_id]
        else:
            table = self.catalog
            with wcs_grouping.open_fits_memmap(ffi_path) as hdus:
                wcs = WCS(hdus[1].header).celestial
            years = (btjd + 2457000.0 - 2457388.5) / 365.25 if np.isfinite(btjd) else 0.0
            pmra = table.get("pmra", pd.Series(0.0, index=table.index)).fillna(0.0)
            pmdec = table.get("pmdec", pd.Series(0.0, index=table.index)).fillna(0.0)
            ra = table.ra + pmra * years / (3.6e6 * np.cos(np.deg2rad(table.dec)))
            dec = table.dec + pmdec * years / 3.6e6
            x, y = wcs.world_to_pixel_values(ra, dec)
            x = x - self.ctx.crop_bounds["x_min"]
            y = y - self.ctx.crop_bounds["y_min"]
            table = pd.DataFrame({"x": x, "y": y}, index=table.index)
        table = table.loc[np.isfinite(table[["x", "y"]]).all(axis=1), ["x", "y"]]
        xy = table.to_numpy(float)
        return table, xy, cKDTree(xy)


def _window(hdus, name, bounds, default=None):
    if name not in hdus:
        if default is None:
            raise ValueError(f"Difference FITS lacks {name}")
        return default
    x, y, x1, y1 = bounds
    hdu = hdus[name]
    data = hdu.section if isinstance(hdu, fits.CompImageHDU) else hdu.data
    return data[y:y1, x:x1]


def cadence_support(path, bounds, physical_mask=None):
    """Use one detector/solver mask and one lane-noise plane for every band."""
    with wcs_grouping.open_fits_memmap(path) as hdus:
        noise = np.asarray(_window(hdus, "NOISE", bounds), dtype="f4")
        mask = _window(hdus, "MASK", bounds, np.zeros(noise.shape, dtype="u1"))
    good = np.isfinite(noise) & (noise > 0)
    if physical_mask is not None:
        x, y, x1, y1 = bounds
        good &= physical_mask[y:y1, x:x1] == 0
    else:
        good &= (mask == 0) | (mask == 64)
    return noise, good, mask


def _profile_provider(methods, stem, product_id, origin, xy, neighbours, shape):
    y, x = np.indices(shape)
    ox, oy = origin

    def profiles(method):
        catalog = methods[method["name"]].get("gridded_catalog")
        if catalog is None:
            return None
        model = catalog.load_model(stem, window_origin=origin, normalize=True)
        if model is None:
            model = catalog.load_model(product_id, window_origin=origin, normalize=True)
        if model is None:
            return None

        def render(center):
            return model.evaluate(x, y, 1.0, center[0] - ox, center[1] - oy)

        return np.asarray([render(xy), *(render(center) for center in neighbours)])

    return profiles


def run_perband(ctx, config, requests, root, epsf_catalogs):
    """Run the target branch from lane manifests, with all outputs under root."""
    from .runner import _manifest_row, _resolve_photometry_methods

    recipe_path = config.template_recipe or str(Path(ctx.templates_dir) / "star_template_recipe.yaml")
    assembly_path = Path(ctx.templates_dir) / "linear_mode_assembly.json"
    assembly = json.loads(assembly_path.read_text()) if assembly_path.is_file() else None
    recipe = TemplateRecipe.load(recipe_path, assembly)
    manifest = pd.read_csv(resolve_star_manifest_path(ctx))
    manifest["product_id"] = star_product_ids(manifest)
    manifest = manifest.loc[manifest.product_id.ne("")].dropna(subset=["group_id"])
    groups = group_schedule(manifest)
    lane = LaneComponents(ctx, recipe, groups)
    positions = CadencePositions(ctx, config.cadence_positions)
    shape = tuple(ctx.crop_bounds["shape"])
    height, width = shape
    fit_shape, fit_offset = hotpants_fit_geometry(ctx)
    frame_rows = list(manifest.itertuples(index=False))
    if config.max_ffis:
        frame_rows = frame_rows[:config.max_ffis]
    scc = (ctx.sector, ctx.camera, ctx.ccd)
    root.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=24)
    def kernel_for(pid):
        solution, hp = load_frame_kernel_for_diff(ctx.baseline_kernels_dir, pid)
        return FrameKernel(solution, hp, fit_shape)

    records = []
    for request in requests:
        host = None
        try:
            host = resolve_host(request, gaia_catalog_path=ctx.gaia_catalog_path)
            sid = host.gaia_source_id
            host_root = root / str(sid)
            host_root.mkdir(exist_ok=True)
            write_identifier_json(host, str(host_root / "identifier.json"))
            write_host_gaia_row_csv(host, str(host_root / "host_gaia_row.csv"))
            reference = positions.tables.get("reference")
            ref = (
                tuple(reference.loc[sid, ["x", "y"]].to_numpy())
                if reference is not None else lane.reference_xy(host)
            )
            planes, origin, identity = lane.target(host, ref, config.cutout_size)
            for index, (group_id, group) in enumerate(groups.iterrows()):
                _write_npz(
                    host_root / "pre_kernel" / f"group_{int(group_id)}.npz",
                    flux=planes[index], bands=BANDS, origin_crop_xy=origin,
                    shift=group.to_numpy(), group_id=group_id,
                    registration_origin_full_ffi=recipe.map_origin, fit_shape_yx=fit_shape,
                    fit_offset_xy=fit_offset, removal_recipe_id=recipe.removal_id,
                    sigma_ps1=recipe.sigma, radius_ps1=recipe.radius,
                )
            resolved = _resolve_photometry_methods(
                ctx, methods=[m for m in config.photometry_methods if m["type"] != "aperture"],
                x_ref=ref[0] + lane.crop_origin[0], y_ref=ref[1] + lane.crop_origin[1],
                epsf_catalogs=epsf_catalogs,
            )
            methods = {m["name"]: m for m in resolved}
            table_rows = []
            failures = []
            frames_ok = 0
            for frame in frame_rows:
                pid = frame.product_id
                try:
                    diff = _resolve_frame_fits_path(ctx.baseline_diffs_dir, pid, scc)
                    stem = f"{pid}-s{ctx.sector:04d}-{ctx.camera}-{ctx.ccd}"
                    btjd = float(getattr(frame, "btjd", float("nan")))
                    ffi = getattr(frame, "path", None) or getattr(frame, "filename", None)
                    catalogue, coords, tree = positions.for_frame(pid, ffi, btjd)
                    xy = tuple(catalogue.loc[sid].to_numpy())
                    if not (0 <= xy[0] < width and 0 <= xy[1] < height):
                        raise ValueError("Target centroid is outside the science crop")
                    bounds = _centered_window_bounds(*xy, config.stamp_size, max_x=width, max_y=height)
                    frame_bands = convolve_target_window(
                        kernel_for(pid), planes[groups.index.get_loc(frame.group_id)],
                        origin, bounds, fit_offset,
                    )
                    data = baseline_window(ctx, pid, bounds, config.stamp_source, ffi)
                    mask = None
                    if config.physical_mask_dir:
                        mask = np.load(Path(config.physical_mask_dir) / f"{stem}.npy", mmap_mode="r")
                        if mask.shape != shape:
                            raise ValueError("Physical mask does not match the science crop")
                    noise, good, solver_mask = cadence_support(diff, bounds, mask)
                    images = np.concatenate([data[None], frame_bands])
                    good &= np.isfinite(images).all(axis=0)
                    x0, y0 = bounds[:2]
                    near = np.asarray(tree.query_ball_point(xy, 14), int)
                    near = near[catalogue.index.to_numpy()[near] != sid]
                    profile = _profile_provider(
                        methods, stem, pid, (x0, y0), xy, coords[near], noise.shape
                    )
                    measured = measure_bands(
                        images, noise, good, (xy[0] - x0, xy[1] - y0),
                        config.photometry_methods, profile,
                    )
                    _write_npz(
                        host_root / "post_kernel" / f"{pid}.npz",
                        images=images.astype("f4"), bands=("D", *BANDS), good=good,
                        solver_mask=solver_mask, noise=noise, origin_crop_xy=(x0, y0),
                        center_crop_xy=xy, group_id=frame.group_id, stem=stem,
                    )
                    usable = False
                    for row in measured:
                        valid = identity == "retained" and np.isfinite(row["D"])
                        usable |= valid
                        table_rows.append(dict(
                            source_id=sid, stem=stem, btjd=btjd, group_id=frame.group_id,
                            x=xy[0], y=xy[1], x0=x0, y0=y0, identity=identity,
                            valid=valid, **row, **{f"lane_w_{b}": recipe.weights[b] for b in BANDS},
                        ))
                    frames_ok += usable
                    if not usable:
                        failures.append(f"{pid}: {identity} or insufficient measurement support")
                except (FileNotFoundError, ValueError, KeyError, OSError, IndexError) as exc:
                    failures.append(f"{pid}: {exc}")
            table_path = host_root / "perband.parquet"
            if table_rows:
                pd.DataFrame(table_rows).to_parquet(table_path.with_suffix(".partial"), index=False)
                table_path.with_suffix(".partial").replace(table_path)
            records.append(_manifest_row(
                host=host, status="ok" if frames_ok else "error",
                blend_flag=identity == "shared_PS1_footprint", frames_processed=frames_ok,
                frames_failed=len(failures), lightcurve_paths=[str(table_path)] if table_rows else [],
                error="; ".join(failures),
            ))
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            records.append(_manifest_row(
                host=host, status="error", blend_flag=False, frames_processed=0,
                frames_failed=len(frame_rows), lightcurve_paths=[], request_label=request.label,
                error=str(exc),
            ))
    path = root / "batch_manifest.csv"
    pd.DataFrame(records).to_csv(path.with_suffix(".partial"), index=False)
    path.with_suffix(".partial").replace(path)
    return path
