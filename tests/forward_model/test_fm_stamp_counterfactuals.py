# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.forward_model.diagnostics import stamp_counterfactuals as SC


def test_stream_selected_compressed_npy_rows(tmp_path: Path):
    array = np.arange(9 * 7 * 5, dtype=np.float32).reshape(9, 7, 5)
    path = tmp_path / "bundle.npz"
    np.savez_compressed(path, pt0_data=array)
    selected = SC.read_selected_npy_rows(path, "pt0_data.npy", [7, 1, 5])
    np.testing.assert_array_equal(selected, array[[7, 1, 5]])


def test_background_offset_solve_separates_flux_and_constant():
    rng = np.random.default_rng(3)
    template = rng.uniform(0.01, 0.2, (2, 1, 12, 20))
    true_flux = rng.uniform(500, 900, (2, 12, 1))
    true_bg = rng.normal(4, 0.5, (2, 12))
    model = np.einsum("gktp,gtk->gtp", template, true_flux)
    data = model + true_bg[..., None]
    ivar = np.ones_like(data)
    coverage = np.ones_like(data)
    flux, bg, solved, chi2 = SC.solve_profiles(
        template, data, ivar, coverage, background=True,
    )
    # The production-compatible 1e-6 ridge gives a tiny, intentional bias.
    np.testing.assert_allclose(flux, true_flux, rtol=5e-5, atol=3e-4)
    np.testing.assert_allclose(bg, true_bg, rtol=1e-3, atol=3e-3)
    np.testing.assert_allclose(solved, data, rtol=3e-4, atol=3e-3)
    assert np.nanmax(chi2) < 2e-6


def test_recovery_recovers_sine_and_transit():
    btjd = np.linspace(0, 8, 800, endpoint=False)
    for kind in ("sinusoid", "transit"):
        signal = SC._signal(btjd, kind, 0.005, 2.0)
        recovered = 1000 * (1 + signal)
        result = SC._recovery(signal, recovered, btjd, kind, 0.005, 2.0)
        assert abs(result["amplitude_bias_fraction"]) < 1e-3
        assert result["passes_amplitude"]
        assert result["passes_timing"]


def test_selection_covers_unique_groups_and_strata():
    n = 24
    frame = pd.DataFrame({
        "star_row": np.arange(n), "group_index": np.arange(n),
        "slot_index": np.zeros(n, dtype=int), "tess_mag": np.linspace(8, 13, n),
        "mag_bin": np.repeat(["mag79", "mag910", "mag1011", "mag1112", "mag1213", "mag1314"], 4),
        "x": np.tile([100, 1800, 100, 1800], 6),
        "y": np.tile([100, 100, 1800, 1800], 6),
        "group_size": np.tile([1, 2], 12),
        "pc1_loading": np.linspace(-2, 2, n),
        "active_fraction": np.full(n, 0.9),
    })
    selected = SC.select_representative_stars(frame, max_stars=10)
    assert len(selected) == 10
    assert selected.group_index.is_unique
    assert selected.quadrant.nunique() == 4
    assert set(selected.crowding_class) == {"isolated", "blended"}


def test_preview_basis_writes_explicit_noncausal_scope(tmp_path: Path):
    from astropy.io import fits

    artifact = tmp_path / "artifacts"
    root = artifact / "plots" / "fits_export"
    root.mkdir(parents=True)
    yy, xx = np.mgrid[:21, :21]
    model = np.exp(-((xx - 10.2) ** 2 + (yy - 9.8) ** 2) / 5.0)
    residual = 0.03 * model + 0.01
    header = fits.Header({"FRAMEIDX": 7, "BTJD": 2723.5})
    fits.PrimaryHDU(model.astype("f4"), header=header).writeto(root / "model_demo.fits")
    fits.PrimaryHDU(residual.astype("f4"), header=header).writeto(root / "residual_demo.fits")
    out = tmp_path / "out"
    summary = SC.run_preview_basis_decomposition(artifact, out)
    table = pd.read_csv(out / "pixel_basis_preview.csv")
    assert summary["status"] == "descriptive_only"
    assert table.loc[0, "scope"] == "three_preview_mosaics_not_representative_stamp_causal"
    assert (out / "figures" / "pixel_basis_preview.png").is_file()


def test_full_stamp_runner_fails_closed_before_cpu_render(monkeypatch, tmp_path: Path):
    import jax

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="SIGSEGVed inside XLA"):
        SC.run_stamp_counterfactuals(
            bundle_path=tmp_path / "missing_bundle.npz",
            artifact_dir=tmp_path / "missing_artifacts",
            output_dir=tmp_path / "out",
            candidates=pd.DataFrame(),
            strict_cadence_mask=np.ones(2, dtype=bool),
        )
    assert not (tmp_path / "out").exists()
