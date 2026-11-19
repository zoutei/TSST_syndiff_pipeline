"""Tiny frozen-lane test through the same runner used by syndiff star."""
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from hotpants import HotpantsConfig
from hotpants.pure.convolution import apply_kernel
from hotpants.pure.kernel import calculate_kernel_basis
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
import yaml

from syndiff_pipeline.difference_imaging.stages.hotpants import (
    write_diff_noise_mask_fits,
    write_frame_kernel_npz,
)
from syndiff_pipeline.star.context import StarEventContext
from syndiff_pipeline.star.perband_photometry import score
from syndiff_pipeline.star.perband_runner import LaneComponents
from syndiff_pipeline.star.runner import run_star_pipeline
from syndiff_pipeline.star.site_config import StarRunConfig
from syndiff_pipeline.template_creation.processing.convolution_utils import gaussian_kernel_1d


def test_star_perband_pipeline_closure(tmp_path, monkeypatch):
    source_id = 1234567890123456789
    shape = (47, 49)
    x, y = 23, 24
    bands = "rizy"
    lane_weights = dict(zip(bands, (.12, .48, .26, .14)))
    ps1 = WCS(naxis=2)
    ps1.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ps1.wcs.crval = [65, -33]
    ps1.wcs.crpix = [x+1, y+1]
    ps1.wcs.cdelt = [-.001, .001]
    header = ps1.to_header()
    header["MJD-OBS"] = 57388.0
    reference = tmp_path / "reference.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros(shape), header=header)]).writeto(reference)
    name = "skycell.2345.010"
    cell = tmp_path / "cells/skycell.2345/010/fingerprint"
    cell.mkdir(parents=True)
    positions = {
        "NAME": name, "projection": 2345, "x": 0, "y": 1, "RA": 65, "DEC": -33,
        "NAXIS1": shape[1], "NAXIS2": shape[0], "PS1CRPIX": 0, **dict(header),
        "CD1_1": -.001, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": .001,
    }
    pd.DataFrame([positions]).to_csv(tmp_path / "cells.csv", index=False)
    master = tmp_path / "master.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros(shape, np.int32))]).writeto(master)
    regmap = tmp_path / f"tess_s26_2_2_{name}.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.arange(np.prod(shape), dtype="i4").reshape(shape), name="TESS_PIXEL_MAP"),
    ]).writeto(regmap)
    arrays = np.zeros((4, *shape), dtype="f4")
    arrays[:, y, x] = [1.25, 2.5, 4, 8]
    arrays[:, y, x+1] = [0.5, 2, 0.25, 2.0]
    mask = np.zeros(shape, "u2")
    mask[y, x-1] = 1 << 12
    np.savez(cell / "arrays.npz", combined_mask=mask, **dict(zip(("band_" + b for b in bands), arrays)))
    (cell / "headers.json").write_text(json.dumps({"i": header.tostring()}))
    (cell / "_provenance.json").write_text(json.dumps(dict(
        recipe_id="synthetic", meta=dict(band_weighted=False, missing_bands=[])
    )))
    (tmp_path / "index.json").write_text(json.dumps({name: str(cell)}))
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(dict(
        psf_sigma=0.9, kernel_radius=2, band_weights=lane_weights,
        ignore_mask_bits=[12], zpcell_factors_path=None, geometry="canonical_v3",
        removal_recipe_id="synthetic", cell_index="index.json",
        registration_grid=dict(width=shape[1], origin_full_ffi=[0, 0]),
    )))

    diffs = tmp_path / "hp_d"
    kernels = tmp_path / "hp_d_kernels"
    diffs.mkdir()
    frame_rows = []
    psf = HotpantsConfig(rkernel=2, ko=1, bgo=-1, ngauss=1, deg_fixe=[1], sigma_gauss=[1.0])
    basis = np.asarray(calculate_kernel_basis((5, 5), psf.sigma_gauss, psf.deg_fixe))
    gaussian = gaussian_kernel_1d(0.9, 2)
    pre = convolve1d(convolve1d(arrays, gaussian, axis=-1), gaussian, axis=-2)
    pre[:, mask != 0] = 0
    expected = {}
    for j, (gid, shift) in enumerate([(0, 0), (0, 0), (1, 1)]):
        pid = f"tess202600000000{j}"
        frame_rows.append(dict(path=str(reference), ffi_basename=pid, btjd=2450.0 + j,
                               group_id=gid, group_dx=shift, group_dy=0.0))
        group = np.roll(pre, shift, axis=-1)
        sol = np.full(psf.n_comp_total+1, .01, dtype=float)
        sol[1] = 1.0 + 0.45*j

        def full(img):
            return apply_kernel(img, sol, np.zeros(shape), np.zeros(shape, "i4"), psf, basis)[0]

        weighted = np.einsum("b,byx->yx", list(lane_weights.values()), group)
        template = full(weighted)
        expected[pid] = np.stack([full(plane) for plane in group])
        np.testing.assert_allclose(
            np.einsum("b,byx->yx", list(lane_weights.values()), expected[pid]),
            template, atol=5e-7,
        )
        diff = np.full(shape, 4.0) - template
        diff[y, x] += 123
        write_diff_noise_mask_fits(
            str(diffs / f"{pid}-s0026-2-2_hp_d.fits.fz"),
            diff, np.ones(shape), np.zeros(shape, "i4"),
        )
        write_frame_kernel_npz(str(kernels), pid, sol, psf)
    manifest = tmp_path / "frames.csv"
    pd.DataFrame(frame_rows).to_csv(manifest, index=False)
    pd.DataFrame([dict(source_id=source_id, ra=65, dec=-33)]).to_csv(tmp_path/"gaia.csv", index=False)
    pd.DataFrame([dict(gaia_source_id=source_id, tic_id="", label="")]).to_csv(tmp_path/"targets.csv", index=False)
    pd.DataFrame([
        dict(source_id=source_id, product_id=fr["ffi_basename"], x=float(x), y=float(y))
        for fr in frame_rows
    ]).to_parquet(tmp_path/"positions.parquet", index=False)
    ctx = StarEventContext(
        target=None, event_dir=str(tmp_path), workspace_root=str(tmp_path), data_root=str(tmp_path),
        cluster_job_path=str(tmp_path/"diff_job.json"), cluster_job={},
        crop_bounds=dict(x_min=0, x_max=shape[1], y_min=0, y_max=shape[0], shape=shape),
        mapping_dir=str(tmp_path), mapping_csv=str(tmp_path/"cells.csv"),
        master_mapping_fits=str(master), gaia_catalog_path=str(tmp_path/"gaia.csv"),
        templates_dir=str(tmp_path), reference_ffi_path=str(reference), sector=26, camera=2, ccd=2,
        baseline_workspace_dir=str(tmp_path), baseline_diffs_label="hp_d", baseline_diffs_dir=str(diffs),
        baseline_convolved_dir="", baseline_phot_bkg_dir="", baseline_phot_bkg_label="ks_b",
        baseline_kernels_dir=str(kernels), manifest_path=str(manifest),
    )
    config = StarRunConfig(
        template_recipe=str(recipe), output_directory=str(tmp_path/"output"),
        cadence_positions=str(tmp_path/"positions.parquet"), stars_file=str(tmp_path/"targets.csv"),
        stamp_size=35, cutout_size=25, debug_plots=False,
        photometry_methods=[dict(name="r8", type="aperture", radius=8, annulus=(12, 16))],
    )

    calls = []
    build = LaneComponents.target

    def count(*args):
        calls.append(1)
        return build(*args)

    monkeypatch.setattr(LaneComponents, "target", count)
    result = pd.read_csv(run_star_pipeline(ctx, run_config=config, validate=False))
    assert result.status.tolist() == ["ok"], result.error.tolist()
    assert result.frames_processed.tolist() == [3]
    assert len(calls) == 1
    host_dir = Path(config.output_directory) / str(source_id)
    assert len(list((host_dir/"pre_kernel").glob("*.npz"))) == 2
    for fr in frame_rows:
        pid = fr["ffi_basename"]
        with np.load(host_dir/"post_kernel"/f"{pid}.npz") as stamp:
            sx, sy = stamp["origin_crop_xy"]
            np.testing.assert_allclose(
                stamp["images"][1:], expected[pid][:, sy:sy+35, sx:sx+35],
                atol=5e-7,
            )
    table = pd.read_parquet(host_dir/"perband.parquet")
    restored = score(table, lane_weights)
    np.testing.assert_allclose(restored.restored_flux, 123.0, atol=3e-6)
    rescored = score(table, dict.fromkeys(bands, .25))
    np.testing.assert_array_equal(restored.restored_flux, rescored.restored_flux)
    assert not np.array_equal(restored.template_flux, rescored.template_flux)
