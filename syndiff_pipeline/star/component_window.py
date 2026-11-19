"""Render four unweighted retained PS1 components on the producing registration grid."""

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs.utils import pixel_to_pixel
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from syndiff_pipeline.template_creation.processing import (
    canonical_cell as cc,
    padding_correction as pc,
)
from syndiff_pipeline.template_creation.processing.convolution_utils import (
    gaussian_kernel_1d,
)


def intersect(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def sections(a, b):
    return (slice(a[1] - b[1], a[3] - b[1]), slice(a[0] - b[0], a[2] - b[0]))


def empty(box, dtype, channels=()):
    return np.zeros((*channels, box[3] - box[1], box[2] - box[0]), dtype=dtype)


class ComponentOperator:
    def __init__(self, recipe, rows, donor):
        self.recipe = recipe
        self.rows = rows
        self.donor = donor
        self.component_flags = set()
        self.gaussian = gaussian_kernel_1d(recipe.sigma, recipe.radius)
        self.raw_cells = recipe.raw_fallback_cells
        self.left_writers = recipe.left_writer_cells
        self.raw_gaussian = (
            gaussian_kernel_1d(recipe.raw_fallback_sigma, recipe.radius) if self.raw_cells else None
        )

    def canonical(self, name, box, light=False):
        """Reproduce column single-writer and the two cross-row copies over this box."""
        md = cc.metadata_for_cell(self.rows, name)
        r, c = cc._row_and_x(md, name)
        w, h = int(md["cell_width"]), int(md["cell_height"])
        present = {name, *cc.canonical_neighbour_names(md, name)}
        if self.recipe.geometry == "june_row":
            present = {
                nm
                for group in md["rows"].values()
                for nm, col in group
                if self.recipe.cells.get(nm)
            }
            anchor = cc.projection_anchor_x(md)
        xx = np.arange(box[0], box[2])
        yy = np.arange(box[1], box[3])[:, None]
        writer = {}
        coverage = {}
        for k in (r - 1, r, r + 1):
            names = [(nm, col) for nm, col in md["rows"].get(k, []) if nm in present]
            xs = {int(col) for _, col in names}
            idx = np.zeros(len(xx), bool)
            vv = empty(box, bool)
            dd = empty(box, float, (4,)) if light else None
            for nm, col in sorted(names, key=lambda a: a[1]):
                dx = (int(col) - c) * (w - 480)
                dy = (k - r) * (h - 480)
                idx |= (xx >= dx) & (xx < dx + w)
                skip = (
                    col != anchor
                    if self.recipe.geometry == "june_row"
                    else col - 1 in xs
                )
                src = (480 if nm in self.left_writers else 470) if skip else 0
                ov = intersect(box, (dx + src, dy, dx + w, dy + h))
                if not ov:
                    continue
                loc = (ov[0] - dx, ov[1] - dy, ov[2] - dx, ov[3] - dy)
                donor = self.donor(nm)
                result = donor.read(loc, light)
                if light and (donor.flag == "truncated" or np.any(result[1])):
                    self.component_flags.add(donor.flag)
                sl = sections(ov, box)
                if not light:
                    vv[sl] = result
                else:
                    vv[sl], dd[(...,) + sl] = result
            writer[k] = (vv, dd)
            coverage[k] = idx
        valid, data = writer[r]
        for k in [r - 1, r + 1]:
            if self.recipe.geometry == "june_row":
                if k < r and r == min(md["rows"]):
                    continue
            elif not md["rows"].get(k):
                continue
            take = (
                (yy < 10) | ((yy < 470) & coverage[k])
                if k < r
                else (yy >= h - 10) | ((yy >= h - 470) & coverage[k] & ~coverage[r])
            )
            if self.recipe.geometry == "june_row":
                take = yy < 10 if k < r else yy >= h - 10
            take = np.broadcast_to(take, valid.shape)
            valid[take] = writer[k][0][take]
            if light:
                data[:, take] = writer[k][1][:, take]
        return valid if not light else (valid, data)

    def corrected_input(self, name, box):
        valid, light = self.canonical(name, box, True)
        light = np.where(valid, light, 0).astype(float)
        own = self.donor(name)
        h, w = own.shape
        patches = []
        if self.recipe.geometry == "june_row":
            for entry in self.recipe.patches.get(name, ()):
                patches.append(
                    (
                        tuple(entry["bounds"]),
                        WCS(fits.Header(entry["wcs"])),
                        [entry["source"]],
                    )
                )
        else:
            for location, neighbors in pc._grouped_padding_spec(
                pc.cross_projection_padding_spec(self.rows.loc[name])
            ).items():
                geo = pc.padding_work_image_geometry(
                    pc._cell_wcs(self.rows.loc[name]),
                    cell_width=w,
                    cell_height=h,
                    location=location,
                    inward_radius=self.recipe.radius,
                )
                y, Y, x, X = geo.patch.bounds
                patches.append(((x, y, X, Y), geo.patch.wcs, neighbors))
        for (x, y, X, Y), patch_wcs, neighbors in patches:
            ov = intersect(box, (x, y, X, Y))
            if not ov:
                continue
            oy, ox = np.mgrid[ov[1] : ov[3], ov[0] : ov[2]]
            replaced = empty(ov, bool)
            full = empty(ov, float, (4,))
            for nm in neighbors:
                d = self.donor(nm)
                px, py = pixel_to_pixel(
                    patch_wcs, pc._cell_wcs(self.rows.loc[nm]), ox - x, oy - y
                )
                rect = (
                    int(np.floor(px.min())) - 2,
                    int(np.floor(py.min())) - 2,
                    int(np.ceil(px.max())) + 3,
                    int(np.ceil(py.max())) + 3,
                )
                v, a = d.read(rect)
                if d.flag == "truncated" or np.any(a):
                    self.component_flags.add(d.flag)
                iy, ix = np.ogrid[rect[1] : rect[3], rect[0] : rect[2]]
                v &= (
                    (ix >= 10)
                    & (ix < d.shape[1] - 10)
                    & (iy >= 10)
                    & (iy < d.shape[0] - 10)
                )
                coords = [py - rect[1], px - rect[0]]
                # Bilinear interpolation on original finite support (including NaN propagation).
                warped = np.array(
                    [
                        map_coordinates(
                            np.where(v, band, np.nan), coords, order=1, cval=np.nan
                        )
                        for band in a
                    ]
                )
                ok = np.isfinite(warped[0])
                full[:, ok] = warped[:, ok]
                replaced |= ok
            v, a = own.read(ov)
            sl = sections(ov, box)
            if self.recipe.geometry == "june_row":
                light[(...,) + sl][:, replaced] = full[:, replaced]
                valid[sl][replaced] = True
            else:
                light[(...,) + sl] += full - np.where(replaced & v, a, 0)
        return valid, light

    def rendered(self, name, cx, cy):
        x, y = int(round(cx)), int(round(cy))
        extents = (185, 370, 740, 1480) if self.recipe.geometry == "june_row" else (185,)
        for extent in extents:
            box = (x - extent, y - extent, x + extent + 1, y + extent + 1)
            if name in self.raw_cells:
                donor = self.donor(name)
                _, light = donor.read(box)
                if donor.flag == "truncated" or np.any(light):
                    self.component_flags.add(donor.flag)
            else:
                _, light = self.corrected_input(name, box)
            yy, xx = np.nonzero(np.any(light != 0, axis=0))
            if not len(xx):
                return None
            if (
                0 < xx.min() and xx.max() < light.shape[2] - 1
                and 0 < yy.min() and yy.max() < light.shape[1] - 1
            ):
                break
        else:
            raise ValueError("PS1 component exceeds rendering window")
        light = light[:, yy.min():yy.max()+1, xx.min():xx.max()+1].astype(float, copy=False)
        gaussian = self.raw_gaussian if name in self.raw_cells else self.gaussian
        blurred = fftconvolve(
            fftconvolve(light, gaussian[None, None, :], axes=2),
            gaussian[None, :, None],
            axes=1,
        ).astype("f4")
        bx = box[0] + xx.min() - self.recipe.radius
        by = box[1] + yy.min() - self.recipe.radius
        rect = (bx, by, bx + blurred.shape[2], by + blurred.shape[1])
        return rect, blurred

    def bin_groups(self, name, center, registration, shifts, origin, shape):
        """Bin once per distinct group roll. Origin is expressed in registration indices."""
        out = np.zeros((len(shifts), 4, *shape), dtype="f4")
        rendered = self.rendered(name, *center)
        if rendered is None:
            return out
        donor = self.donor(name)
        h, w = donor.shape
        if registration.shape != (h, w):
            raise ValueError("PS1 registration shape mismatch")
        rect, blur = rendered
        ov = intersect(rect, (0, 0, w, h))
        if not ov:
            return out
        x, y, X, Y = ov
        if name in self.raw_cells:
            valid = donor.read(ov, light=False)
        else:
            valid = self.canonical(name, ov)
            if self.recipe.geometry == "june_row" and any(
                intersect(ov, p["bounds"]) for p in self.recipe.patches.get(name, ())
            ):
                valid = self.corrected_input(name, ov)[0]
        veto = np.unpackbits(donor.veto[y:Y, x // 8 : (X + 7) // 8], axis=1)[
            :, x % 8 : x % 8 + X - x
        ]
        bright = blur[(...,) + sections(ov, rect)] * (valid & (veto == 0))
        cache = {}
        for i, (sx, sy) in enumerate(shifts):
            if (sx, sy) not in cache:
                idx = registration[
                    np.ix_(np.arange(y + sy, Y + sy) % h, np.arange(x + sx, X + sx) % w)
                ]
                dy = idx // self.recipe.map_width - origin[1]
                dx = idx % self.recipe.map_width - origin[0]
                good = (
                    (idx >= 0)
                    & (dy >= 0)
                    & (dx >= 0)
                    & (dy < shape[0])
                    & (dx < shape[1])
                )
                lost = (idx >= 0) & ~good
                if lost.any() and np.max(abs(bright[:, lost])) > 1.0e-12 * max(
                    1, np.max(abs(bright))
                ):
                    raise ValueError("Native target window clips retained light")
                ids = (dy[good] * shape[1] + dx[good]).astype("i8")
                cache[sx, sy] = np.array(
                    [
                        np.bincount(
                            ids, weights=b[good], minlength=np.prod(shape)
                        ).reshape(shape)
                        for b in bright
                    ]
                )
            out[i] = cache[sx, sy]
        return out


def transport_target(operator, receivers, ra_dec, groups, origin, shape):
    """Accumulate each receiver's source component on one native window per group.

    Receivers supply a WCS, a PS1-to-native registration and an ordered roll list.
    The origin is in native registration indices, independent of the Hotpants grid.
    """
    planes = np.zeros((len(groups), 4, *shape), dtype="f4")
    raw_norm = 0.0
    for name, wcs, registration, shifts in receivers:
        center = tuple(map(float, wcs.all_world2pix(*ra_dec, 0)))
        contribution = operator.bin_groups(name, center, registration, shifts, origin, shape)
        planes += contribution
        if name in operator.raw_cells:
            raw_norm += float(np.abs(contribution).sum(dtype=float))
    if raw_norm and raw_norm > 1e-7 * np.abs(planes).sum(dtype=float):
        operator.component_flags.add("raw_fallback_provisional")
    return planes


def extract_native_window(planes, origin, bounds):
    """Take a zero-padded window in native coordinates, preserving leading axes."""
    x, y = origin
    extent = (x, y, x + planes.shape[-1], y + planes.shape[-2])
    result = empty(bounds, planes.dtype, planes.shape[:-2])
    overlap = intersect(extent, bounds)
    if overlap:
        result[(...,) + sections(overlap, bounds)] = planes[(...,) + sections(overlap, extent)]
    return result


def convolve_target_window(kernel, planes, origin, bounds, fit_offset):
    """Apply a frame kernel to a precomputed component and return the output stamp."""
    radius = kernel.radius
    x, y, x1, y1 = bounds
    guard = (x - radius, y - radius, x1 + radius, y1 + radius)
    if kernel.factor != 1:
        raise ValueError("Native pre-kernel band windows require oversampling 1")
    return kernel.apply(
        extract_native_window(planes, origin, guard),
        guard[0] + fit_offset[0], guard[1] + fit_offset[1],
    )
