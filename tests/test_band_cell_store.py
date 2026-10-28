"""Tests for template_creation.processing.band_cell_store (shared per-band cell store)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import band_cell_store as bcs
from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import perband
from syndiff_pipeline.template_creation.processing.band_utils import process_skycell_bands
from syndiff_pipeline.template_creation.processing.ps1_process import (
    _materialize_shm_result,
    process_single_cell,
)

def _make_header():
    from astropy.io import fits

    h = fits.Header()
    h["BOFFSET"] = 10.0
    h["BSOFTEN"] = 20.0
    h["EXPTIME"] = 30.0
    return h.tostring()


HEADER = _make_header()
SIZE = 48
PROJ, CELL = "2484", "033"
SKYCELL = f"skycell.{PROJ}.{CELL}"


def _raw_bands(seed=0, nan_in_r=True):
    rng = np.random.default_rng(seed)
    bands = {}
    for b in perband.BANDS:
        img = rng.normal(5.0, 1.0, (SIZE, SIZE)).astype(np.float32)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        img += 12.0 * np.exp(-((xx - 20) ** 2 + (yy - 24) ** 2) / (2 * 2.5**2)).astype(np.float32)
        bands[b] = img
    if nan_in_r:
        bands["r"][3:5, 40:44] = np.nan  # NaN in ONE band only
    return bands


def _headers():
    return {b: HEADER for b in perband.BANDS}


def _bundle(bands, weights, dtype):
    combined, mask, uncert = process_skycell_bands(
        bands, headers_data=_headers(), band_weights=weights
    )
    wb = perband.weighted_band_images(bands, _headers(), weights=weights)
    return {
        "skycell_id": SKYCELL,
        "projection": f"skycell.{PROJ}",
        "row_id": 0,
        "x_coord": 0,
        "combined_image": combined,
        "combined_mask": mask,
        "combined_uncert": np.full_like(combined, 0.5),
        "headers_data": _headers(),
        "remove_saturated_stars": False,
        "band_images": {b: np.asarray(a, dtype=dtype) for b, a in wb.items()},
    }


def _run(bands, weights=None, dtype="float64"):
    result = _materialize_shm_result(process_single_cell(_bundle(bands, weights, dtype)))
    assert "band_images" in result
    return result


def _same_bits(a, b):
    """Bitwise equal on every non-NaN pixel (+0/-0 treated equal), identical NaN pattern."""
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = ~np.isnan(a)
    return np.array_equal(a[ok] + np.float32(0), b[ok] + np.float32(0)) and np.array_equal(
        (a[ok] + np.float32(0)).view(np.uint32), (b[ok] + np.float32(0)).view(np.uint32)
    )


def test_sum_of_float64_band_cells_is_bitwise_the_combined_image():
    result = _run(_raw_bands(), dtype="float64")
    assert (result["combined_image"] == 0).sum() > 0  # Z really zeroed something
    cells = result["band_images"]
    # the NaN pattern of the combined image is in every band
    for b in perband.BANDS:
        assert np.array_equal(np.isnan(cells[b]), np.isnan(result["combined_image"]))
    total = perband.sum_bands(cells)
    assert total.dtype == np.float32
    assert _same_bits(total, result["combined_image"])


def test_nan_pattern_goes_into_every_band_and_sum_is_bitwise():
    """Direct check with NaN surviving in the combined image (NaN in one band only)."""
    rng = np.random.default_rng(5)
    bands = _raw_bands(5, nan_in_r=True)
    wb = perband.weighted_band_images(bands, _headers())
    before = perband.sum_bands(wb)
    assert np.isnan(before).any()
    after = before.copy()
    after[10:20, 10:20] = 0.0  # Z zeroes a block (finite pixels only)
    after[np.isnan(before)] = np.nan
    cells = bcs.build_band_cells(wb, before, after, "float64")
    for b in perband.BANDS:
        assert np.array_equal(np.isnan(cells[b]), np.isnan(after))
        assert np.all(cells[b][10:20, 10:20][~np.isnan(cells[b][10:20, 10:20])] == 0)
    assert _same_bits(perband.sum_bands(cells), after)


def test_float32_band_cells_sum_within_a_few_ulp():
    result = _run(_raw_bands(1), dtype="float32")
    total = perband.sum_bands(result["band_images"])
    c = result["combined_image"]
    ok = np.isfinite(c)
    assert np.array_equal(np.isnan(total), np.isnan(c))
    big = np.max([np.abs(result["band_images"][b][ok]) for b in perband.BANDS] + [np.abs(c[ok])], axis=0)
    ulp = np.spacing(big.astype(np.float32))
    d = np.abs(total[ok].astype(np.float64) - c[ok])
    assert np.all(d <= 4 * ulp)
    assert d.max() / np.abs(c[ok]).max() < 1e-6


def test_non_default_weights_are_the_recipes():
    w = {"r": 0.3, "i": 0.3, "z": 0.2, "y": 0.2}
    result = _run(_raw_bands(2), weights=w, dtype="float64")
    assert _same_bits(perband.sum_bands(result["band_images"]), result["combined_image"])


def test_build_band_cells_rejects_non_zeroing_change():
    before = np.ones((4, 4), np.float32)
    after = before.copy()
    after[0, 0] = 2.0
    wb = {b: before.astype(np.float64) for b in perband.BANDS}
    with pytest.raises(ValueError):
        bcs.build_band_cells(wb, before, after, "float64")


@pytest.fixture
def recipe():
    return cs.combined_recipe(gaia_version="none")


def _publish(tmp_path, result, recipe, dtype="float64", **kw):
    return bcs.publish_band_cells(
        tmp_path, PROJ, CELL,
        band_images=result["band_images"],
        combined_mask=result["combined_mask"],
        headers_data=result["headers_data"],
        removed_stars=result["removed_stars"],
        recipe=recipe, input_fingerprints=["rawfp"], band_dtype=dtype, **kw,
    )


def test_round_trip_and_same_fingerprint_as_combined(tmp_path, recipe):
    result = _run(_raw_bands(), dtype="float64")
    info = _publish(tmp_path, result, recipe)
    assert info and not info["already_published"]
    # same fingerprint as the combined cell of the same recipe
    cinfo = cs.publish_combined_cell(
        tmp_path, PROJ, CELL, combined_image=result["combined_image"],
        combined_mask=result["combined_mask"], headers_data=result["headers_data"],
        removed_stars=result["removed_stars"], recipe=recipe, input_fingerprints=["rawfp"],
    )
    assert cinfo["fingerprint"] == info["fingerprint"]
    d = bcs.band_cell_dir(tmp_path, PROJ, CELL, info["fingerprint"])
    assert d.parent.parent.parent.name == "ps1_band_cells.zarr"
    assert bcs.band_cells_complete(tmp_path, PROJ, CELL, info["fingerprint"])

    loaded = bcs.try_load_band_cells(tmp_path, PROJ, CELL, info["fingerprint"])
    assert set(loaded["band_images"]) == set(perband.BANDS)
    assert loaded["band_weights"] == cs.DEFAULT_BAND_WEIGHTS
    assert loaded["headers_data"] == result["headers_data"]
    assert _same_bits(bcs.combined_from_band_cells(loaded), result["combined_image"])
    comb = bcs.try_load_combined_cell_from_band_cells(tmp_path, PROJ, CELL, info["fingerprint"])
    assert _same_bits(comb["combined_image"], cs.try_load_combined_cell(
        tmp_path, PROJ, CELL, info["fingerprint"])["combined_image"])
    side = json.loads((d / "_provenance.json").read_text())
    assert side["kind"] == bcs.KIND and side["meta"]["band_dtype"] == "float64"


def test_resolve_for_recipe(tmp_path, recipe):
    result = _run(_raw_bands(), dtype="float32")
    raw_fp = "rawfp"
    assert bcs.resolve_band_cells_fingerprint_for_recipe(tmp_path, PROJ, CELL, recipe, raw_fp=raw_fp) is None
    # the producer fingerprint uses sorted input fps == [raw_fp] when the recipe has no catalogue
    info = _publish(tmp_path, result, recipe, dtype="float32")
    assert bcs.resolve_band_cells_fingerprint_for_recipe(
        tmp_path, PROJ, CELL, recipe, raw_fp=raw_fp) == info["fingerprint"]
    other = dict(recipe, bright_star_mag_threshold=11.0)
    assert bcs.resolve_band_cells_fingerprint_for_recipe(tmp_path, PROJ, CELL, other, raw_fp=raw_fp) is None


def test_incomplete_dir_not_loaded(tmp_path, recipe):
    result = _run(_raw_bands(), dtype="float32")
    info = _publish(tmp_path, result, recipe, dtype="float32")
    d = bcs.band_cell_dir(tmp_path, PROJ, CELL, info["fingerprint"])
    (d / "removed_stars.json").unlink()
    assert bcs.try_load_band_cells(tmp_path, PROJ, CELL, info["fingerprint"]) is None
    assert not bcs.band_cells_complete(tmp_path, PROJ, CELL, info["fingerprint"])


def test_idempotent_concurrent_publish(tmp_path, recipe):
    result = _run(_raw_bands(), dtype="float32")
    infos, errors = [], []

    def work():
        try:
            infos.append(_publish(tmp_path, result, recipe, dtype="float32"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    ts = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and all(i is not None for i in infos)
    assert len({i["fingerprint"] for i in infos}) == 1
    root = bcs.band_cell_dir(tmp_path, PROJ, CELL, infos[0]["fingerprint"]).parent
    assert [p.name for p in root.iterdir() if not p.name.startswith("_tmp_")] == [infos[0]["fingerprint"]]
    again = _publish(tmp_path, result, recipe, dtype="float32")
    assert again["already_published"]


def test_publish_never_raises(tmp_path, recipe):
    blocker = tmp_path / "ps1_skycells_zarr"
    blocker.write_text("not a dir")
    result = _run(_raw_bands(), dtype="float32")
    assert _publish(tmp_path, result, recipe, dtype="float32") is None
    assert bcs.publish_band_cells(
        tmp_path / "x", PROJ, CELL, band_images={"r": np.zeros(2)}, combined_mask=np.zeros(2),
        headers_data={}, removed_stars=[], recipe=recipe) is None


def test_scc_paths_helper(tmp_path):
    from syndiff_pipeline.common import scc_paths

    assert scc_paths.PS1_BAND_CELLS_ZARR_BASENAME == "ps1_band_cells.zarr"
    p = scc_paths.ps1_band_cells_zarr_path(tmp_path)
    assert p == bcs.band_cells_root(tmp_path)
    assert p.name == "ps1_band_cells.zarr" and p.parent.name == "ps1_skycells_zarr"
