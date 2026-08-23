"""Precomputed per-basis-function convolution for field-mode ``convolved_templates``
(Lane A -- H.2, plan Part H).

Exploits two facts, both established by direct reads of ``pyhotpants``
(``hotpants/pure/{kernel,convolution,os_precompute}.py``) and this repo's
own ``convolved_templates.py``:

1. Convolution is linear over the DIA kernel's basis-function decomposition:
   ``conv(image, sum_i c_i * basis_i) == sum_i c_i * conv(image, basis_i)``.
   ``hotpants.pure.convolution.jit_spatial_convolve`` already exploits this
   in the *other* direction (combine coefficients into one small kernel,
   then convolve once per block) -- this module exploits it the other way:
   convolve each basis function against the template *once*, then combine
   with per-block coefficients (cheap array arithmetic, no more real
   convolution).
2. For Lane A specifically (``kernel_fit`` fits exactly one global
   ``kernel_solution`` per SCC run, reused unchanged by every
   ``convolved_templates`` group), there is no per-frame coefficient
   variation to handle -- one recombination per group is exact.

``recombine_basis_maps_full`` is written to reproduce
``hotpants.pure.convolution.jit_spatial_convolve``'s exact block-based
convolution *pixel-for-pixel* (mod. float32-vs-float64 rounding -- today's
production path runs that function in float32; this module runs in
float64 throughout, so it is more precise, not less, but the two will not
match to float32's own machine epsilon, only somewhat below it -- see the
"precision" note in the plan doc). It deliberately does **not** reproduce
mask/variance/background handling: ``convolve_template_with_kernel_solution``
(this module's target replacement) discards all three (`_bkg, _var, _mask`
are ignored by its caller), and ``mask``/``variance`` are always zero-filled
at that call site, so the mask logic in ``jit_spatial_convolve`` never
actually gates anything in this specific call path -- reproducing it would
be dead code here.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2


def kernel_coeffs_at_block_center(
    kernel_sol: np.ndarray,
    *,
    r_pix_x: float,
    r_pix_y: float,
    n_comp_ker: int,
    ker_order: int,
    block_center_x: float,
    block_center_y: float,
) -> np.ndarray:
    """Exact port of the coefficient loop inside
    ``hotpants.pure.convolution.jit_make_kernel`` (lines ~68-84) -- same
    formula, same C ``kernelSol`` layout (leading slot), same variable
    names, minus the final "sum coefficient * basis image" step (the part
    this module replaces with array reuse instead of a fresh combine).
    """
    kernel_coeffs = np.zeros(int(n_comp_ker), dtype=np.float64)
    xf = (block_center_x - 0.5 * r_pix_x) / (0.5 * r_pix_x)
    yf = (block_center_y - 0.5 * r_pix_y) / (0.5 * r_pix_y)

    kernel_coeffs[0] = kernel_sol[1]
    k = 2
    for i1 in range(1, int(n_comp_ker)):
        coeff = 0.0
        ax = 1.0
        for ix in range(ker_order + 1):
            ay = 1.0
            for iy in range(ker_order - ix + 1):
                coeff += kernel_sol[k] * ax * ay
                k += 1
                ay *= yf
            ax *= xf
        kernel_coeffs[i1] = coeff
    return kernel_coeffs


def _valid_slice(fshape: tuple[int, int], hr_shape: tuple[int, int], k_shape: tuple[int, int]):
    """Same convention as ``hotpants.pure.os_precompute._valid_slice``."""
    hr_ny, hr_nx = hr_shape
    kh, kw = k_shape
    oy, ox = hr_ny - kh + 1, hr_nx - kw + 1
    return (slice(kh - 1, kh - 1 + oy), slice(kw - 1, kw - 1 + ox))


def precompute_basis_valid_maps(
    patch: np.ndarray,
    basis_funcs,
) -> np.ndarray:
    """``(n_basis, ph - kh + 1, pw - kw + 1)`` array of
    ``fftconvolve(patch, basis_k, mode="valid")`` for every basis function,
    sharing one forward FFT of *patch* across all of them -- same core
    algorithm as ``hotpants.pure.os_precompute.precompute_basis_lr_maps``'s
    ``_one``, but returning the raw "valid" crop (this module's own
    ``recombine_basis_maps_full`` consumes that directly; the LR
    pad-then-downsample wrapper in ``os_precompute`` is for a different
    consumer -- stamp/region vector gathering during the DIA fit -- and
    does not apply here).

    Matches ``scipy.signal.fftconvolve(patch, basis_k, mode="valid")`` to
    float64 FFT precision (~1e-13), per ``os_precompute``'s own documented
    accuracy for this identical FFT construction.
    """
    tpl = np.ascontiguousarray(patch, dtype=np.float64)
    kstack = np.ascontiguousarray(np.asarray(basis_funcs, dtype=np.float64))
    if kstack.ndim != 3:
        raise ValueError(f"basis_funcs must be (n,kh,kw), got {kstack.shape}")
    n_ker, kh, kw = kstack.shape
    hr_ny, hr_nx = tpl.shape
    if hr_ny < kh or hr_nx < kw:
        raise ValueError(
            f"patch shape {tpl.shape} smaller than basis shape {(kh, kw)}; "
            "caller must pad the patch by at least the kernel half-width"
        )

    fshape = (next_fast_len(hr_ny + kh - 1), next_fast_len(hr_nx + kw - 1))
    valid_idx = _valid_slice(fshape, (hr_ny, hr_nx), (kh, kw))
    F_tpl = rfft2(tpl, s=fshape)

    out = np.empty((n_ker, hr_ny - kh + 1, hr_nx - kw + 1), dtype=np.float64)
    for k in range(n_ker):
        F_k = rfft2(kstack[k], s=fshape)
        full = irfft2(F_tpl * F_k, s=fshape)
        out[k] = np.ascontiguousarray(full[valid_idx], dtype=np.float64)
    return out


def recombine_basis_maps_full(
    basis_maps_valid: np.ndarray,
    kernel_sol: np.ndarray,
    *,
    ny_hr: int,
    nx_hr: int,
    hw_kernel: int,
    kc_step: int,
    n_comp_ker: int,
    ker_order: int,
    oversample: int,
) -> np.ndarray:
    """Reconstruct the full ``(ny_hr, nx_hr)`` convolved image (zero border,
    interior = valid region) from precomputed per-basis valid-mode maps,
    reproducing ``hotpants.pure.convolution.jit_spatial_convolve``'s exact
    block grid and block-center formula.

    ``basis_maps_valid`` : ``(n_basis, ny_hr - 2*hw_kernel, nx_hr - 2*hw_kernel)``
    -- output of :func:`precompute_basis_valid_maps` on the *full* (already
    group-assembled) template, or a sum of per-patch calls scatter-added
    into that same full-size valid-region shape (patch decomposition is
    the caller's responsibility -- see
    :func:`scatter_add_patch_valid_maps`).
    """
    n_basis = basis_maps_valid.shape[0]
    expected_valid_shape = (ny_hr - 2 * hw_kernel, nx_hr - 2 * hw_kernel)
    if basis_maps_valid.shape[1:] != expected_valid_shape:
        raise ValueError(
            f"basis_maps_valid shape {basis_maps_valid.shape[1:]} != expected "
            f"{expected_valid_shape} for ny_hr={ny_hr}, nx_hr={nx_hr}, "
            f"hw_kernel={hw_kernel}"
        )

    conv_image = np.zeros((ny_hr, nx_hr), dtype=np.float64)
    r_pix_x = float(nx_hr) / oversample
    r_pix_y = float(ny_hr) / oversample

    n_blocks_y = (ny_hr + kc_step - 1) // kc_step
    n_blocks_x = (nx_hr + kc_step - 1) // kc_step

    for by in range(n_blocks_y):
        j0 = by * kc_step + hw_kernel
        if j0 >= ny_hr - hw_kernel:
            continue
        j1 = min(j0 + kc_step, ny_hr - hw_kernel)
        for bx in range(n_blocks_x):
            i0 = bx * kc_step + hw_kernel
            if i0 >= nx_hr - hw_kernel:
                continue
            i1 = min(i0 + kc_step, nx_hr - hw_kernel)

            cx = (i0 + hw_kernel) / float(oversample)
            cy = (j0 + hw_kernel) / float(oversample)
            coeffs = kernel_coeffs_at_block_center(
                kernel_sol,
                r_pix_x=r_pix_x,
                r_pix_y=r_pix_y,
                n_comp_ker=n_comp_ker,
                ker_order=ker_order,
                block_center_x=cx,
                block_center_y=cy,
            )

            # basis_maps_valid[k, j - hw_kernel, i - hw_kernel] == the k-th
            # basis's convolution result at full-image pixel (j, i).
            vj0, vj1 = j0 - hw_kernel, j1 - hw_kernel
            vi0, vi1 = i0 - hw_kernel, i1 - hw_kernel
            block = np.tensordot(
                coeffs, basis_maps_valid[:, vj0:vj1, vi0:vi1], axes=(0, 0)
            )
            conv_image[j0:j1, i0:i1] = block

    return conv_image


def dilate_footprint_for_patch_convolution(
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    *,
    hw_kernel: int,
    array_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Bounds to extract (and zero-pad implicitly at array edges) around a
    patch's true nonzero footprint ``[y0:y1, x0:x1)`` before calling
    :func:`precompute_basis_valid_maps`, so that the resulting valid-mode
    map's coverage exactly equals the footprint expanded by ``hw_kernel``
    on every side (the "expand the edges, add the flux from these" margin
    the final convolved image legitimately needs).

    This needs **2x** ``hw_kernel`` of padding on each side, not 1x: one
    factor is consumed by "valid" mode's own unavoidable edge-cropping
    (the output of a valid-mode convolution is always ``hw_kernel``
    narrower than its input on each side), the other factor is the actual
    output margin being requested. Getting this factor wrong silently
    truncates a patch's real contribution near its own edges without
    raising -- see the regression test
    ``test_patch_scatter_add_matches_whole_image_convolution`` in
    ``tests/test_convolved_templates_patch_cache.py``, which caught
    exactly this bug (using only 1x padding) during development.

    Returns ``(py0, py1, px0, px1)`` clipped to ``array_shape``. Pass the
    *returned* (already-clipped) ``py0``/``px0`` directly as
    :func:`scatter_add_patch_valid_maps`'s ``(row_offset, col_offset)`` --
    the algebra (valid-mode row 0 == full-image row ``py0 + hw_kernel`` ==
    global-valid-map row ``py0``) holds whether or not clipping occurred,
    because clipping only happens where the array truly has no data past
    its own edge, and that's exactly where the ideal (unclipped) target
    range would have had nothing to contribute anyway.
    """
    ny, nx = array_shape
    py0 = max(0, int(y0) - 2 * hw_kernel)
    py1 = min(ny, int(y1) + 2 * hw_kernel)
    px0 = max(0, int(x0) - 2 * hw_kernel)
    px1 = min(nx, int(x1) + 2 * hw_kernel)
    return py0, py1, px0, px1


def scatter_add_patch_valid_maps(
    patch_valid_maps: list[tuple[np.ndarray, tuple[int, int]]],
    *,
    n_basis: int,
    ny_valid: int,
    nx_valid: int,
) -> np.ndarray:
    """Sum multiple patches' valid-mode basis maps into one full-size
    ``(n_basis, ny_valid, nx_valid)`` accumulator, at each patch's own
    offset.

    ``patch_valid_maps`` : list of ``(maps, (row_offset, col_offset))``
    where ``maps`` has shape ``(n_basis, mh, mw)`` and
    ``maps[k]`` belongs at ``accum[k, row_offset:row_offset+mh,
    col_offset:col_offset+mw]``. Offsets/shapes may extend past the
    accumulator bounds (patches near the array edge); the overlap is
    clipped, matching how ``assemble_group_from_contribs``'s scatter-add
    already tolerates out-of-bounds indices via ``MappingGrid`` filtering
    upstream -- here the caller is expected to have already restricted each
    patch to its true in-bounds support before calling.

    This is the exact numerical realization of ``sum_i conv(patch_i,
    basis_k) == conv(sum_i patch_i, basis_k)``: each patch's *own*
    convolution (already computed over patch-plus-halo, see
    :func:`precompute_basis_valid_maps`) is added into the group total
    without being cropped back to the patch's un-dilated footprint first --
    cropping before summing would silently discard the flux the kernel
    legitimately spread past the patch's own boundary, which is exactly
    the "expand the edges, add the flux from these" requirement this
    module exists to satisfy.
    """
    accum = np.zeros((int(n_basis), int(ny_valid), int(nx_valid)), dtype=np.float64)
    for maps, (row_off, col_off) in patch_valid_maps:
        mh, mw = maps.shape[1], maps.shape[2]
        r0 = max(0, row_off)
        c0 = max(0, col_off)
        r1 = min(ny_valid, row_off + mh)
        c1 = min(nx_valid, col_off + mw)
        if r0 >= r1 or c0 >= c1:
            continue
        sr0, sc0 = r0 - row_off, c0 - col_off
        sr1, sc1 = sr0 + (r1 - r0), sc0 + (c1 - c0)
        accum[:, r0:r1, c0:c1] += maps[:, sr0:sr1, sc0:sc1]
    return accum


# ---------------------------------------------------------------------------
# Disk-cached per-skycell basis convolution + group assembly (H.2, the
# actual integration point for ``convolved_templates.py``).


def _basis_conv_dir(store_root: str | Path, kind: str) -> Path:
    return Path(store_root) / "basis_conv" / kind


def _skycell_patch_basename(skycell: str, sx_int: int, sy_int: int) -> str:
    name = str(skycell).strip()
    if not name.startswith("skycell."):
        name = f"skycell.{name}"
    return f"{name}_sx{int(sx_int):+d}_sy{int(sy_int):+d}.npz"


def interior_basis_conv_path(store_root: str | Path, skycell: str, sx_int: int, sy_int: int) -> Path:
    return _basis_conv_dir(store_root, "interior") / _skycell_patch_basename(skycell, sx_int, sy_int)


def seam_delta_basis_conv_path(
    store_root: str | Path, skycell: str, sx_int: int, sy_int: int, group_id: int
) -> Path:
    base = _skycell_patch_basename(skycell, sx_int, sy_int).replace(".npz", f"_gid{int(group_id)}.npz")
    return _basis_conv_dir(store_root, "seam_delta") / base


def _footprint_from_indices(indices: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1) half-open bounding box of nonzero flat *indices*, or
    None if empty."""
    if indices.size == 0:
        return None
    nx = shape[1]
    rows = indices // nx
    cols = indices % nx
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _materialize_dense_patch(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Dense ``(y1-y0, x1-x0)`` array from sparse (flat-index, value) pairs,
    restricted to *bounds* (a superset of the indices' own bbox)."""
    y0, y1, x0, x1 = bounds
    nx = shape[1]
    rows = indices // nx
    cols = indices % nx
    out = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
    out[rows - y0, cols - x0] = values
    return out


def _write_basis_conv_npz(path: Path, maps: np.ndarray, offset: tuple[int, int]) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(
            tmp_path,
            maps=maps.astype(np.float64),
            row_offset=np.int64(offset[0]),
            col_offset=np.int64(offset[1]),
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_basis_conv_npz(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with np.load(path) as z:
        maps = np.asarray(z["maps"], dtype=np.float64)
        offset = (int(z["row_offset"]), int(z["col_offset"]))
    return maps, offset


def get_or_build_skycell_basis_conv(
    store_root: str | Path,
    skycell: str,
    sx_int: int,
    sy_int: int,
    *,
    kind: str,
    group_id: int | None,
    base_tess_shape: tuple[int, int],
    basis_funcs: np.ndarray,
    hw_kernel: int,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Load (or compute-and-cache) one skycell's basis-convolved patch.

    ``kind`` is ``"interior"`` (group-independent, cached once per
    ``(skycell, sx, sy)`` and reused across every group that shares it --
    the actual H.1/H.2 payoff) or ``"seam_delta"`` (small, group-specific,
    keyed additionally by ``group_id``). Returns ``(valid_maps,
    (row_offset, col_offset))`` ready for
    :func:`scatter_add_patch_valid_maps`, or ``None`` if the underlying
    contrib is empty (e.g. a seam delta that turned out to need no
    correction for this skycell/group).
    """
    from syndiff_pipeline.template_creation.processing.field_templates import (
        interior_contrib_path,
        load_contrib,
        seam_delta_contrib_path,
    )

    if kind == "interior":
        contrib_path = interior_contrib_path(store_root, skycell, sx_int, sy_int)
        cache_path = interior_basis_conv_path(store_root, skycell, sx_int, sy_int)
    elif kind == "seam_delta":
        if group_id is None:
            raise ValueError("seam_delta requires group_id")
        contrib_path = seam_delta_contrib_path(store_root, skycell, sx_int, sy_int, int(group_id))
        cache_path = seam_delta_basis_conv_path(store_root, skycell, sx_int, sy_int, int(group_id))
    else:
        raise ValueError(f"kind must be interior|seam_delta, got {kind!r}")

    if cache_path.is_file():
        return _load_basis_conv_npz(cache_path)

    if not contrib_path.is_file():
        if kind == "seam_delta":
            return None
        raise FileNotFoundError(f"missing interior contrib: {contrib_path}")

    data = load_contrib(contrib_path, keys=["indices", "flux_sum"])
    indices = np.asarray(data["indices"], dtype=np.int64)
    if indices.size == 0:
        return None
    flux = np.asarray(data["flux_sum"], dtype=np.float64)

    y0, y1, x0, x1 = _footprint_from_indices(indices, base_tess_shape)
    py0, py1, px0, px1 = dilate_footprint_for_patch_convolution(
        y0, y1, x0, x1, hw_kernel=hw_kernel, array_shape=base_tess_shape
    )
    dense = _materialize_dense_patch(
        indices, flux, shape=base_tess_shape, bounds=(py0, py1, px0, px1)
    )
    kh = basis_funcs.shape[1]
    if dense.shape[0] < kh or dense.shape[1] < basis_funcs.shape[2]:
        # Patch (plus dilation) smaller than the basis kernel itself --
        # pad with zeros rather than fail; happens only for a vanishingly
        # small skycell footprint right at hw_kernel scale.
        pad_y = max(0, kh - dense.shape[0])
        pad_x = max(0, basis_funcs.shape[2] - dense.shape[1])
        dense = np.pad(dense, ((0, pad_y), (0, pad_x)))
    maps = precompute_basis_valid_maps(dense, basis_funcs)
    offset = (py0, px0)
    _write_basis_conv_npz(cache_path, maps, offset)
    return maps, offset


def build_group_convolved_template(
    store_root: str | Path,
    shifts,
    *,
    group_id: int,
    base_tess_shape: tuple[int, int],
    crop_hr: tuple[int, int, int, int],
    basis_funcs: np.ndarray,
    kernel_solution: np.ndarray,
    hw_kernel: int,
    kc_step: int,
    n_comp_ker: int,
    ker_order: int,
    oversample: int,
) -> np.ndarray:
    """HR convolved template, cropped to ``crop_hr``, for one group, built
    entirely from cached per-skycell basis convolutions -- the H.2
    replacement for "assemble dense group template (cropped the same way),
    then convolve it fresh". *shifts* is the group's ``(skycell, sx_int,
    sy_int)`` list (same as consumed by ``assemble_group_from_contribs``/
    ``assemble_group_from_split_contribs``).

    ``crop_hr`` is ``(x0, x1, y0, y1)`` in the *full* SCC-grid's own HR
    pixel coordinates (``base_tess_shape``'s own indexing) -- the same
    convention as ``assemble_group_from_contribs``'s ``crop`` parameter /
    ``FieldModeTemplateContext.template_roi_bounds``. Each skycell's own
    basis-convolved patch cache (keyed only by ``(skycell, sx, sy)``, from
    :func:`get_or_build_skycell_basis_conv`) is built and stored in
    full-grid coordinates -- crop-independent, so it is reusable across
    every group *and* every run regardless of that run's own crop_bounds;
    only the final scatter-add into this call's crop-sized accumulator
    needs the crop offset.
    """
    n_basis = basis_funcs.shape[0]
    x0, x1, y0, y1 = (int(v) for v in crop_hr)
    ny_hr, nx_hr = y1 - y0, x1 - x0
    ny_valid, nx_valid = ny_hr - 2 * hw_kernel, nx_hr - 2 * hw_kernel

    patch_maps: list[tuple[np.ndarray, tuple[int, int]]] = []
    for skycell, sx_i, sy_i in shifts:
        interior = get_or_build_skycell_basis_conv(
            store_root, skycell, sx_i, sy_i,
            kind="interior", group_id=None,
            base_tess_shape=base_tess_shape, basis_funcs=basis_funcs, hw_kernel=hw_kernel,
        )
        if interior is not None:
            maps, (row_off, col_off) = interior
            patch_maps.append((maps, (row_off - y0, col_off - x0)))
        delta = get_or_build_skycell_basis_conv(
            store_root, skycell, sx_i, sy_i,
            kind="seam_delta", group_id=int(group_id),
            base_tess_shape=base_tess_shape, basis_funcs=basis_funcs, hw_kernel=hw_kernel,
        )
        if delta is not None:
            maps, (row_off, col_off) = delta
            patch_maps.append((maps, (row_off - y0, col_off - x0)))

    summed_valid = scatter_add_patch_valid_maps(
        patch_maps, n_basis=n_basis, ny_valid=ny_valid, nx_valid=nx_valid
    )
    return recombine_basis_maps_full(
        summed_valid, kernel_solution,
        ny_hr=ny_hr, nx_hr=nx_hr, hw_kernel=hw_kernel, kc_step=kc_step,
        n_comp_ker=n_comp_ker, ker_order=ker_order, oversample=oversample,
    )
