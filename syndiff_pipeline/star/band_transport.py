"""Replay retained PS1 components through a frozen per-band template snapshot.

The supported row operator is explicitly versioned. This command is process
isolated because the historical padding operator uses a module-level loader.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter, label

from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
from syndiff_pipeline.template_creation.processing import convolution_utils
from syndiff_pipeline.template_creation.processing import padding_correction as PC
from syndiff_pipeline.template_creation.processing import perband as PB
from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
    build_ps1_wcs,
)
from syndiff_pipeline.template_creation.processing.ps1_process import (
    extract_projection_metadata,
)

from .band_addback import convolve_target_patch, native_patch


def sparse_gaussian(image, sigma=60.0, radius=470, *, cval=np.nan):
    a = np.asarray(image)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Target blur requires finite 2D input")
    if not (cval == 0 or np.isnan(cval)):
        raise ValueError("Unsupported exterior value")
    out = np.zeros_like(a)
    ys, xs = np.nonzero(a)
    if len(ys):
        y0 = max(0, int(ys.min()) - radius)
        y1 = min(a.shape[0], int(ys.max()) + radius + 1)
        x0 = max(0, int(xs.min()) - radius)
        x1 = min(a.shape[1], int(xs.max()) + radius + 1)
        out[y0:y1, x0:x1] = gaussian_filter(
            a[y0:y1, x0:x1], sigma=sigma, radius=radius, mode="constant", cval=0
        )
    if np.isnan(cval) and radius:
        out[:radius] = np.nan
        out[-radius:] = np.nan
        out[:, :radius] = np.nan
        out[:, -radius:] = np.nan
    return out


class Snapshot:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.get("operator_version") != "publisher_row_v1":
            raise ValueError(
                "Unsupported template operator; do not reuse an older row operator for new products"
            )
        self.mapping = Path(cfg["mapping_dir"])
        self.bands = Path(cfg["band_cells_dir"])
        self.grid = load_mapping_grid_from_master(self.mapping / cfg["master_filename"])
        self.sky = pd.read_csv(cfg["skycell_list"]).set_index("NAME", drop=False)
        self.publishers = json.loads(Path(cfg["publisher_lists"]).read_text())
        self.metadata = self.sky.copy()
        for path in set(self.publishers["lists"].values()):
            rows = pd.read_csv(path).set_index("NAME", drop=False)
            self.metadata = pd.concat(
                [self.metadata, rows.loc[~rows.index.isin(self.metadata.index)]]
            )
        self.sigma = float(cfg["psf_sigma"])
        self.radius = int(cfg["blur_radius"])
        self.mask_signature = hashlib.sha256(
            json.dumps(
                {
                    key: cfg.get(key)
                    for key in (
                        "operator_version",
                        "psf_sigma",
                        "blur_radius",
                        "store_weights",
                        "mask_resolution",
                        "historical_store_root",
                    )
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.masks = Path(cfg["mask_cache_dir"])
        self.masks.mkdir(parents=True, exist_ok=True)
        with np.load(cfg["kernel_npz"], allow_pickle=False) as z:
            self.k0 = z["K0"]
            self.k1 = z["K1"]
            self.nx = z["node_x"]
            self.ny = z["node_y"]
            meta = json.loads(str(z["meta"]))
        adopted = json.loads(Path(cfg["adopted_weights_json"]).read_text())
        alpha, beta = adopted["colour_map_label_from_new"]
        self.delta = (
            alpha + beta * np.array([617.0, 752.0, 866.0, 962.0]) - meta["u_ref_nm"]
        ) / meta["du_dc_nm_per_mag"]
        self.kernels = np.array([self.k0 + v * self.k1 for v in self.delta])
        self.store_weights = np.array([cfg["store_weights"][b] for b in "rizy"])
        self.scales = np.asarray(adopted["weights_rizy"]) / self.store_weights

    def mask(self, name):
        band_path = self.bands / f"{name}.npz"
        digest = hashlib.sha256()
        with band_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        band_sha = digest.hexdigest()
        cache_key = f"{name}_{band_sha[:16]}_{self.mask_signature[:16]}"
        dest = self.masks / f"{cache_key}.npz"
        record = self.masks / f"{cache_key}.json"
        if dest.exists() and record.exists():
            cached = json.loads(record.read_text())
            if (
                cached.get("band_cells_sha256") == band_sha
                and cached.get("mask_recipe_signature") == self.mask_signature
            ):
                with np.load(dest, allow_pickle=False) as z:
                    return z["mask"]
        if self.cfg.get("mask_resolution") != "legacy_archive_image_match":
            raise ValueError("Missing frozen mask " + name)
        _, proj, cell = name.split(".")
        key = Path("skycell." + proj) / cell
        store = Path(self.cfg["historical_store_root"])
        with np.load(self.bands / f"{name}.npz", allow_pickle=False) as z:
            total = np.zeros_like(z["r"])
            for b in "rizy":
                total += z[b]
        matches = []
        for p in (store / "ps1_convolved.zarr" / key).glob("*/_provenance.json"):
            meta = json.loads(p.read_text())
            r = meta.get("recipe_params", {})
            inputs = meta.get("inputs", [])
            if (
                r.get("psf_sigma") != self.sigma
                or r.get("radius") != self.radius
                or r.get("padding") != "same_projection_only"
                or len(inputs) != 1
            ):
                continue
            cp = store / "ps1_combined.zarr" / key / inputs[0]
            if not (cp / "arrays.npz").exists():
                continue
            cm = json.loads((cp / "_provenance.json").read_text())
            if cm["recipe_params"].get("band_weights") != self.cfg["store_weights"]:
                continue
            with np.load(cp / "arrays.npz", allow_pickle=False) as z:
                original = z["combined_image"]
            if original.shape != total.shape or np.any(
                np.isfinite(original) != np.isfinite(total)
            ):
                continue
            good = np.isfinite(original)
            error = float(
                np.max(abs(original[good] - total[good]))
                / max(float(np.max(abs(original[good]))), 1e-30)
            )
            if error <= 1e-6:
                matches.append((p.parent, cp, error, meta))
        if len(matches) != 1:
            raise ValueError(
                f"{name}: expected one image-matched archive, found {len(matches)}"
            )
        p, cp, error, meta = matches[0]
        with np.load(p / "arrays.npz", allow_pickle=False) as z:
            mask = z["convolved_mask"]
        tmp = dest.with_name(dest.stem + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, mask=mask)
        tmp.replace(dest)
        record_tmp = record.with_name(record.name + f".{os.getpid()}.tmp")
        record_tmp.write_text(
            json.dumps(
                dict(
                    convolved_artifact=str(p),
                    combined_artifact=str(cp),
                    convolved_fingerprint=meta["fingerprint"],
                    band_sum_vs_combined_max_rel=error,
                    band_cells_sha256=band_sha,
                    mask_recipe_signature=self.mask_signature,
                ),
                indent=2,
            )
        )
        record_tmp.replace(record)
        return mask

    def seam(self, name, band, get, shape):
        spec = PC.cross_projection_padding_spec(self.sky.loc[name])
        if not spec:
            return None
        own = np.asarray(get.band(name, band), float)
        original = PC._load_combined_image

        def loader(data_root, projection, cell, *, combined_recipe=None):
            prefix = str(projection)
            key = (
                f"{prefix}.{cell}"
                if prefix.startswith("skycell.")
                else f"skycell.{prefix}.{cell}"
            )
            a = get.band(key, band)
            return None if a is None else np.asarray(a, float)

        PC._load_combined_image = loader
        try:
            out = np.zeros(shape)
            wcs = PC._cell_wcs(self.sky.loc[name])
            for location, neighbours in PC._grouped_padding_spec(spec).items():
                out += PC._location_correction(
                    location=location,
                    neighbors=neighbours,
                    skycell=name,
                    recipient_wcs=wcs,
                    cell_shape=shape,
                    own_combined=own,
                    data_root=Path(self.cfg.get("data_root", ".")),
                    skycell_df=self.sky,
                    psf_sigma=self.sigma,
                    kernel_radius=self.radius,
                )
            return out
        finally:
            PC._load_combined_image = original


class TargetCells:
    def __init__(self, snapshot, row):
        self.snapshot = snapshot
        self.row = row
        self.cache = {}
        self.notes = []

    def source(self, name):
        if name in self.cache:
            return self.cache[name]
        ctx = self.snapshot
        path = ctx.bands / f"{name}.npz"
        if not path.exists():
            self.cache[name] = None
            return None
        if name not in ctx.metadata.index:
            raise ValueError("Missing donor WCS " + name)
        wcs, (w, h) = build_ps1_wcs(ctx.metadata.loc[name])
        x, y = wcs.all_world2pix(self.row.raMean, self.row.decMean, 0)
        x, y = int(round(float(x))), int(round(float(y)))
        nan = np.zeros((h, w), bool)
        patches = {}
        bounds = None
        inside = 0 <= x < w and 0 <= y < h
        r = int(ctx.cfg.get("component_search_radius", 160))
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        raw = []
        with np.load(path, allow_pickle=False) as z:
            for b in "rizy":
                a = z[b]
                nan |= np.isnan(a)
                if inside:
                    raw.append(a[y0:y1, x0:x1].copy())
        if inside:
            raw = np.array(raw)
            labels, _ = label(
                np.any(np.isfinite(raw) & (raw != 0), axis=0), structure=np.ones((3, 3))
            )
            cy, cx = y - y0, x - x0
            near = labels[max(0, cy - 3) : cy + 4, max(0, cx - 3) : cx + 4]
            ids = np.unique(near[near > 0])
            if len(ids) > 1:
                raise ValueError("Ambiguous retained component " + name)
            if len(ids) == 1:
                seg = labels == ids[0]
                if (
                    (y0 > 0 and seg[0].any())
                    or (y1 < h and seg[-1].any())
                    or (x0 > 0 and seg[:, 0].any())
                    or (x1 < w and seg[:, -1].any())
                ):
                    raise ValueError("Retained component exceeds search " + name)
                yy, xx = np.nonzero(seg)
                sy = slice(yy.min(), yy.max() + 1)
                sx = slice(xx.min(), xx.max() + 1)
                patches = {
                    b: np.where(seg[sy, sx], raw[j, sy, sx], 0)
                    for j, b in enumerate("rizy")
                }
                bounds = (
                    y0 + yy.min(),
                    y0 + yy.max() + 1,
                    x0 + xx.min(),
                    x0 + xx.max() + 1,
                )
        self.cache[name] = (nan, patches, bounds)
        self.notes.append(
            dict(
                cell=name,
                has_component=bool(patches),
                bounds=None if bounds is None else list(map(int, bounds)),
            )
        )
        return self.cache[name]

    def band(self, name, b):
        item = self.source(name)
        if item is None:
            return None
        nan, patches, bounds = item
        out = np.zeros(nan.shape, np.float32)
        out[nan] = np.nan
        if b in patches:
            y0, y1, x0, x1 = bounds
            out[y0:y1, x0:x1] = patches[b]
        return out


def transport(snapshot, row, output_dir, stamp_half_size=10):
    ctx = snapshot
    grid = ctx.grid
    get = TargetCells(ctx, row)
    recipients = []
    reach = ctx.radius + int(ctx.cfg.get("component_search_radius", 160)) + 20
    for name, record in ctx.sky.iterrows():
        wcs, shape = build_ps1_wcs(record)
        x, y = wcs.all_world2pix(row.raMean, row.decMean, 0)
        if -reach < x < shape[0] + reach and -reach < y < shape[1] + reach:
            recipients.append(name)
    x = float(row.x_ffi) - grid.science_xmin_ffi
    y = float(row.y_ffi) - grid.science_ymin_ffi
    ix, iy = int(round(x)), int(round(y))
    factor = int(grid.oversampling)
    # Every source subcell that can reach the output stamp is included.
    half = stamp_half_size + int(np.ceil((ctx.kernels.shape[-1] // 2) / factor)) + 1
    gx0 = (ix - half + grid.science_xmin_ffi - grid.ffi_xmin) * factor
    gy0 = (iy - half + grid.science_ymin_ffi - grid.ffi_ymin) * factor
    size = (2 * half + 1) * factor
    bands = np.zeros((4, size, size))
    used = []
    for name in recipients:
        with np.load(
            Path(ctx.cfg["contribution_dir"]) / f"{name}.npz", allow_pickle=False
        ) as z:
            check = json.loads(str(z["check"]))
        publisher = check["list_chosen"]
        if publisher not in ctx.publishers["lists"].values():
            raise ValueError("Unrecorded publisher list")
        df = pd.read_csv(publisher).set_index("NAME", drop=False)
        md = extract_projection_metadata(
            df.reset_index(drop=True), str(ctx.sky.loc[name, "projection"])
        )
        mask = ctx.mask(name)
        with fits.open(
            ctx.mapping / ctx.cfg["regmap_pattern"].format(skycell=name)
        ) as hdus:
            assignment = np.asarray(hdus["TESS_PIXEL_MAP"].data)
        if mask.shape != assignment.shape:
            raise ValueError("Frozen mask and registration geometry differ")
        valid = (assignment >= 0) & ((mask.astype(np.int64) & 4096) == 0)
        yy, xx = np.divmod(assignment.astype(np.int64), grid.width_os)
        selected = (
            valid & (xx >= gx0) & (xx < gx0 + size) & (yy >= gy0) & (yy < gy0 + size)
        )
        dest = (yy[selected] - gy0) * size + xx[selected] - gx0
        del assignment, mask, xx, yy, valid
        for ib, b in enumerate("rizy"):
            image = PB.blur_cell_row_path(
                name, md, lambda n: get.band(n, b), ctx.sigma, ctx.radius
            )
            if image is None:
                raise ValueError("Missing row-path recipient " + name)
            corr = ctx.seam(name, b, get, image.shape)
            if corr is not None:
                finite = np.isfinite(image)
                tmp = image.astype(float)
                tmp[finite] += corr[finite]
                image = tmp.astype(np.float32)
            bands[ib] += (
                np.bincount(
                    dest,
                    weights=np.nan_to_num(image[selected], nan=0),
                    minlength=size * size,
                )
                .reshape(size, size)
                .astype(np.float32)
                .astype(float)
            )
        used.append(dict(cell=name, publisher=publisher))
    image, origin = convolve_target_patch(
        bands * ctx.scales[:, None, None],
        ctx.kernels,
        ctx.nx,
        ctx.ny,
        origin_os=(gy0, gx0),
        grid=grid,
    )
    h = stamp_half_size
    bounds = (ix - h, ix + h + 1, iy - h, iy + h + 1)
    stamp = native_patch(image, origin, grid=grid, bounds_sci=bounds)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = int(row.target_index)
    dest = out / f"{idx:04d}.npz"
    tmp = out / f"{idx:04d}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        tmp,
        addback_unscaled=stamp,
        bounds=bounds,
        band_templates=bands,
        origin_os=(gy0, gx0),
        kernel_delta=ctx.delta,
        objID=np.array(str(row.objID)),
        centroid_sci=[x, y],
    )
    tmp.replace(dest)
    record = dict(
        target_index=idx,
        objID=str(row.objID),
        operator_version=ctx.cfg["operator_version"],
        recipients=used,
        sources=get.notes,
        zero_retained_component=not any(n["has_component"] for n in get.notes),
        component_policy="Retained connected component, potentially blended; fit all relevant restored neighbours",
        kernel_delta=ctx.delta.tolist(),
    )
    record_path = out / f"{idx:04d}.json"
    record_tmp = out / f"{idx:04d}.{os.getpid()}.json.tmp"
    record_tmp.write_text(json.dumps(record, indent=2))
    record_tmp.replace(record_path)
    return record


def prepare(manifest_path, target_indices=None):
    manifest_path = Path(manifest_path)
    cfg = json.loads(manifest_path.read_text())
    if cfg.get("schema_version") != 1 or "template_snapshot" not in cfg:
        raise ValueError("A version-1 template snapshot manifest is required")
    snapshot = Snapshot(cfg["template_snapshot"])
    targets = pd.read_csv(
        cfg["targets_csv"], dtype={"objID": str, "gaia_source_id": str}
    )
    if not targets.target_index.is_unique or not targets.objID.is_unique:
        raise ValueError("Duplicate preparation targets")
    if target_indices is not None:
        missing = set(target_indices) - set(targets.target_index)
        if missing:
            raise ValueError(f"Unknown target indices: {sorted(missing)}")
        targets = targets[targets.target_index.isin(target_indices)]
    original = convolution_utils.apply_gaussian_convolution
    convolution_utils.apply_gaussian_convolution = sparse_gaussian
    records = []
    try:
        for row in targets.itertuples():
            record = transport(
                snapshot,
                row,
                cfg["prepared_components_dir"],
                int(cfg.get("stamp_half_size", 10)),
            )
            record["manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            record["producer_source_sha256"] = hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
            record["artifact_state"] = cfg.get("artifact_state", "provisional")
            destination = (
                Path(cfg["prepared_components_dir"])
                / f"{int(row.target_index):04d}.json"
            )
            temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(destination)
            records.append(record)
    finally:
        convolution_utils.apply_gaussian_convolution = original
    return records
