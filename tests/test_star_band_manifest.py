import json

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.star.band_photometry import run
from syndiff_pipeline.star.cli import build_parser


def test_manifest_two_epochs_preserves_signed_flux_and_identity(tmp_path):
    y, x = np.mgrid[-10:11, -10:11]
    p = np.exp(-(x * x + y * y) / 2)
    p /= p.sum()
    np.savez(
        tmp_path / "p.npz",
        profile=(2 * p)[None],
        x=[15.0],
        y=[15.0],
        objID=np.array(["999"]),
    )
    pd.DataFrame(
        [dict(target_index=0, objID="999", gaia_source_id="111", tmag_ps1=18.0)]
    ).to_csv(tmp_path / "targets.csv", index=False)
    tr = tmp_path / "transport"
    tr.mkdir()
    np.savez(tr / "0000.npz", addback_unscaled=7 * p, bounds=[5, 26, 5, 26])
    mask = np.zeros((31, 31), np.int32)
    mask[15, 15] = 1
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(mask)]).writeto(
        tmp_path / "mask.fits"
    )
    frames = []
    for k, flux in enumerate([5.0, -3.0]):
        a = np.full((31, 31), 2.0)
        a[5:26, 5:26] += (flux - 14) * p
        f = tmp_path / f"f{k}.fits"
        fits.HDUList(
            [fits.PrimaryHDU(), fits.ImageHDU(a), fits.ImageHDU(np.ones_like(a))]
        ).writeto(f)
        frames.append(
            dict(
                id=f"frame{k}",
                time_btjd=float(k),
                difference_fits=str(f),
                noise_fits=str(f),
                raw_science_fits=str(f),
                physical_mask_fits=str(tmp_path / "mask.fits"),
                scale=2.0,
            )
        )
    cfg = dict(
        schema_version=1,
        artifact_state="provisional",
        profile_normalization="unit_sum_full_support",
        output_dir=str(tmp_path / "out"),
        targets_csv=str(tmp_path / "targets.csv"),
        profiles_npz=str(tmp_path / "p.npz"),
        flux_zero_point=20.0,
        transport_dirs=[str(tr)],
        frames=frames,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(cfg))
    record = run(path)
    d = pd.read_csv(tmp_path / "out/measurements.csv")
    np.testing.assert_allclose(d.flux, [5.0, -3.0], atol=1e-12)
    np.testing.assert_allclose(d.raw_flux, [-9.0, -17.0], atol=1e-12)
    assert np.isnan(d.measured_mag.iloc[1])
    assert (d.status == "ok").all()
    assert (d.artifact_state == "provisional").all()
    assert record["n_targets"] == 1 and record["n_frames"] == 2
    assert (tmp_path / "out/lightcurves/999.csv").exists()
    assert (
        build_parser().parse_args(["extract-band", "--manifest", str(path)]).command
        == "extract-band"
    )
