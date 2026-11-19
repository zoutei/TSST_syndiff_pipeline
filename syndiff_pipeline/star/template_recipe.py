"""Read the producing template's photometric and geometry contracts."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import yaml

from syndiff_pipeline.template_creation.processing.band_convolve import (
    load_band_factors_json,
)
from syndiff_pipeline.template_creation.processing.perband import (
    BANDS,
    validate_weights,
)


@dataclass(frozen=True)
class TemplateRecipe:
    sigma: float
    radius: int
    weights: dict
    factors: dict | None
    veto_bits: int
    geometry: str
    removal_id: str | tuple[str, ...]
    cells: dict
    map_width: int
    map_origin: tuple
    raw_store: Path | None
    patches: dict
    ps1_crpix: float = 0.0
    registration_dir: Path | None = None
    reference_ffi: Path | None = None
    reflect_below: int | None = None
    left_writer_cells: frozenset[str] = frozenset()
    raw_fallback_cells: frozenset[str] = frozenset()
    raw_fallback_sigma: float | None = None
    registration_pattern: str | None = None
    mapping_table: Path | None = None

    @classmethod
    def load(cls, path, assembly=None):
        """All paths in the descriptor are relative to the descriptor's directory."""
        path = Path(path)
        spec = yaml.safe_load(path.read_text())

        def resolve(value):
            return path.parent / value

        factors_base = path.parent
        if "frozen_config" in spec:
            frozen = resolve(spec["frozen_config"])
            factors_base = frozen.parent
            cfg = yaml.safe_load(frozen.read_text())["stages"]
            ps1, down = cfg["ps1_process"], cfg["downsample"]
            sigma, weights = ps1["psf_sigma"], down["band_weights"]
            bits, factors = down["ignore_mask_bits"], down["zpcell_factors_path"]
        else:
            sigma, weights = spec["psf_sigma"], spec["band_weights"]
            bits, factors = spec["ignore_mask_bits"], spec["zpcell_factors_path"]
        weights = validate_weights(weights, "template band weights")
        if assembly and "band_weights" in assembly and assembly["band_weights"] != weights:
            raise ValueError("Frozen template weights disagree with assembly")
        sigma = float(sigma)
        radius = int(spec["kernel_radius"])
        if not np.isfinite(sigma) or sigma <= 0 or radius < 1:
            raise ValueError("Invalid frozen Gaussian parameters")
        geometry = spec["geometry"]
        if geometry not in ("canonical_v3", "june_row"):
            raise ValueError(f"Unknown template geometry {geometry}")
        index = json.loads(resolve(spec["cell_index"]).read_text())
        cells = {k: resolve(v) if v else None for k, v in index.items()}
        raw_store = resolve(spec["raw_store"]) if geometry == "june_row" else None
        left_writers = frozenset(
            json.loads(resolve(spec["left_writer_cells"]).read_text())
            if "left_writer_cells" in spec else ()
        )
        fallback = spec.get("raw_fallback", {})
        fallback_cells = frozenset(
            json.loads(resolve(fallback["cells"]).read_text()) if fallback else ()
        )
        fallback_sigma = float(fallback["psf_sigma"]) if fallback else None
        if (left_writers or fallback) and geometry != "june_row":
            raise ValueError("Left-writer overrides and raw fallback use June row geometry")
        if fallback and (not np.isfinite(fallback_sigma) or fallback_sigma <= 0):
            raise ValueError("Invalid frozen raw-fallback Gaussian width")
        if any(cells.get(cell) is None for cell in fallback_cells | left_writers):
            raise ValueError("June overrides must reference indexed cells")
        if "removal_recipe_ids" in spec:
            removal = tuple(map(str, spec["removal_recipe_ids"]))
            if not removal:
                raise ValueError("At least one removal archive recipe ID is required")
        else:
            removal = str(spec["removal_recipe_id"])
        patches = (
            json.loads(resolve(spec["padding_patches"]).read_text())
            if raw_store
            else {}
        )
        grid = spec["registration_grid"]
        if int(grid["width"]) < 1 or len(grid["origin_full_ffi"]) != 2:
            raise ValueError("Invalid registration grid")
        crpix = float(spec["ps1_crpix"]) if "ps1_crpix" in spec else 0.0
        if crpix not in (0.0, 0.5):
            raise ValueError("ps1_crpix must be 0 or 0.5")
        return cls(
            float(sigma),
            radius,
            weights,
            load_band_factors_json(factors_base / factors) if factors else None,
            sum(1 << int(b) for b in set(bits)),
            geometry,
            removal,
            cells,
            int(grid["width"]),
            tuple(grid["origin_full_ffi"]),
            raw_store,
            patches,
            crpix,
            resolve(spec["registration_dir"]) if "registration_dir" in spec else None,
            resolve(spec["reference_ffi"]) if "reference_ffi" in spec else None,
            int(spec["reflect_below"]) if "reflect_below" in spec else None,
            left_writers,
            fallback_cells,
            fallback_sigma,
            spec.get("registration_pattern"),
            resolve(spec["mapping_table"]) if "mapping_table" in spec else None,
        )


def retained_component(stack, xy, origin, shape, neighbours=()):
    """Select the connected retained footprint nearest the PS1 catalogue position."""
    from scipy.ndimage import label

    live = np.any(np.isfinite(stack) & (stack != 0), axis=0)
    labels, _ = label(live, np.ones((3, 3), bool))
    y, x = np.nonzero(labels)
    if not len(x):
        return None, "no_retained_component"
    ox, oy = origin
    distance = (x + ox - xy[0]) ** 2 + (y + oy - xy[1]) ** 2
    nearest = distance.argmin()
    if distance[nearest] > 25:
        return None, "no_retained_component"
    seg = labels == labels[y[nearest], x[nearest]]
    h, w = live.shape
    if (
        (ox > 0 and seg[:, 0].any())
        or (oy > 0 and seg[0].any())
        or (ox + w < shape[1] and seg[:, -1].any())
        or (oy + h < shape[0] and seg[-1].any())
    ):
        return None, "truncated"
    shared = any(
        0 <= (ix := round(nx) - ox) < w
        and 0 <= (iy := round(ny) - oy) < h
        and seg[iy, ix]
        for nx, ny in neighbours
    )
    return np.nan_to_num(np.where(seg, stack, 0)), (
        "shared_PS1_footprint" if shared else "retained"
    )


def read_donor(recipe, name, row, host, data_root):
    """Read frozen post-removal support. Keep donor NaNs separately for seam replacement."""
    import pandas as pd
    from astropy.io import fits
    from astropy.time import Time
    from astropy.wcs import WCS
    from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
        build_ps1_wcs,
    )
    from syndiff_pipeline.template_creation.processing.band_utils import (
        apply_flux_conversion,
        extract_header_values,
    )

    root = recipe.cells.get(name)
    if root is None:
        raise FileNotFoundError(f"{name}: producing band cell missing from recipe")
    prov = json.loads((root / "_provenance.json").read_text())
    allowed = (recipe.removal_id,) if isinstance(recipe.removal_id, str) else recipe.removal_id
    if prov["recipe_id"] not in allowed:
        raise ValueError(f"{name}: removal recipe mismatch")
    hdr = fits.Header.fromstring(json.loads((root / "headers.json").read_text())["i"])
    wcs, _ = build_ps1_wcs(row)
    if recipe.raw_store:
        wcs = WCS(hdr).celestial
        wcs.wcs.crpix += recipe.ps1_crpix
    dt = Time(hdr["MJD-OBS"], format="mjd").jyear - 2016
    ra = host.ra + (host.pmra or 0.0) * dt / (3.6e6 * np.cos(np.deg2rad(host.dec)))
    dec = host.dec + (host.pmdec or 0.0) * dt / 3.6e6
    x, y = map(float, wcs.all_world2pix(ra, dec, 0))
    pointer = (
        Path(data_root)
        / "ps1_removal_ledger/v1"
        / root.parents[1].name
        / root.parent.name
        / root.name
        / "ledger.json"
    )
    neighbours = []
    if pointer.is_file():
        path = Path(json.loads(pointer.read_text())["ledger"]) / "sources.parquet"
        cat = pd.read_parquet(
            path, columns=["source_id", "catalogue", "pixel_x", "pixel_y"]
        )
        cat = cat[cat.catalogue.eq("gaia_dr3")].dropna().drop_duplicates("source_id")
        target = cat.source_id.astype(str).eq(str(host.gaia_source_id))
        if target.any():
            x, y = cat.loc[target, ["pixel_x", "pixel_y"]].iloc[0]
        near = cat.loc[
            ~target
            & cat.pixel_x.between(x - 160, x + 160)
            & cat.pixel_y.between(y - 160, y + 160)
        ]
        neighbours = near[["pixel_x", "pixel_y"]].to_numpy()
    with np.load(root / "arrays.npz") as archive:
        if not recipe.raw_store and (
            prov["meta"]["band_weighted"] or prov["meta"]["missing_bands"]
        ):
            raise ValueError(f"{name}: expected four unweighted post-removal bands")
        base = archive["combined_image" if recipe.raw_store else "band_r"]
        height, width = base.shape
        invalid = np.packbits(~np.isfinite(base), axis=1)
        veto = np.packbits(
            (archive["combined_mask"].astype("uint32") & recipe.veto_bits) != 0, axis=1
        )
        x0, x1 = max(0, round(x) - 155), min(width, round(x) + 156)
        y0, y1 = max(0, round(y) - 155), min(height, round(y) + 156)
        data = None
        flag = "no_retained_component"
        if x0 < x1 and y0 < y1:
            box = np.s_[y0:y1, x0:x1]
            if recipe.raw_store:
                import zarr

                store = zarr.open_group(
                    str(recipe.raw_store / name.split(".")[1] / name), mode="r"
                )
                support = np.isfinite(base[box]) & (base[box] != 0)
                stack = np.array(
                    [
                        np.where(
                            support,
                            apply_flux_conversion(
                                store[b][box].astype("f4"),
                                *extract_header_values(store[b].attrs["header"]),
                            ),
                            0,
                        )
                        for b in BANDS
                    ]
                )
            else:
                stack = np.stack(
                    [base[box]] + [archive["band_" + b][box] for b in BANDS[1:]]
                )
            data, flag = retained_component(
                stack, (x, y), (x0, y0), base.shape, neighbours
            )
            if data is not None and recipe.factors is not None:
                data = (
                    data
                    / np.array([recipe.factors[name][b] for b in BANDS])[:, None, None]
                )
    return Donor(base.shape, invalid, veto, (x0, y0), data, flag)


@dataclass
class Donor:
    shape: tuple
    invalid: np.ndarray
    veto: np.ndarray
    origin: tuple
    data: np.ndarray | None
    flag: str

    def read(self, box, light=True):
        from .component_window import empty, intersect, sections

        valid = empty(box, bool)
        overlap = intersect(box, (0, 0, self.shape[1], self.shape[0]))
        if overlap:
            x, y, X, Y = overlap
            bits = np.unpackbits(self.invalid[y:Y, x // 8 : (X + 7) // 8], axis=1)
            valid[sections(overlap, box)] = bits[:, x % 8 : x % 8 + X - x] == 0
        if not light:
            return valid
        dtype = self.data.dtype if self.data is not None else "f4"
        data = empty(box, dtype, (4,))
        if self.data is not None:
            x, y = self.origin
            rect = (x, y, x + self.data.shape[2], y + self.data.shape[1])
            overlap = intersect(rect, box)
            if overlap:
                data[(...,) + sections(overlap, box)] = self.data[
                    (...,) + sections(overlap, rect)
                ]
        return valid, data
