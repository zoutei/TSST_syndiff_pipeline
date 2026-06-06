#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

RELEVANT_WCS_KEYS = [
    "NAXIS1",
    "NAXIS2",
    "CRVAL1",
    "CRVAL2",
    "CRPIX1",
    "CRPIX2",
    "PC1_1",
    "PC1_2",
    "PC2_1",
    "PC2_2",
    "CDELT1",
    "CDELT2",
    "RADESYS",
    "CTYPE1",
    "CTYPE2",
]


def load_tess_wcs(tess_fits_path: Path) -> tuple[WCS, tuple[int, int]]:
    with fits.open(tess_fits_path) as hdul:
        chosen_hdu_idx = None
        for idx, hdu in enumerate(hdul):
            if getattr(hdu, "data", None) is None:
                continue
            try:
                w = WCS(hdu.header)
                if w.has_celestial:  # type: ignore[attr-defined]
                    chosen_hdu_idx = idx
                    break
            except Exception:
                continue

        if chosen_hdu_idx is None:
            # Fallback to primary
            chosen_hdu_idx = 0

        data_shape = None
        if getattr(hdul[chosen_hdu_idx], "data", None) is not None:
            data_shape = hdul[chosen_hdu_idx].data.shape
        else:
            # Try first HDU with data
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    data_shape = hdu.data.shape
                    break

        tess_wcs = WCS(hdul[chosen_hdu_idx].header)

        if data_shape is None:
            raise RuntimeError("Could not determine data shape for TESS FITS")

        # Ensure array shape set for any downstream use
        try:
            tess_wcs.array_shape = data_shape
        except Exception:
            pass

    # Normalize to (ny, nx)
    if len(data_shape) >= 2:
        ny, nx = data_shape[-2], data_shape[-1]
    else:
        raise RuntimeError(f"Unexpected TESS data shape: {data_shape}")

    return tess_wcs, (nx, ny)


def build_ps1_wcs(row: pd.Series) -> tuple[WCS, tuple[int, int]]:
    header_dict = {k: row[k] for k in RELEVANT_WCS_KEYS if k in row}
    # astropy expects integers for NAXIS
    header_dict["NAXIS1"] = int(header_dict["NAXIS1"])  # type: ignore[index]
    header_dict["NAXIS2"] = int(header_dict["NAXIS2"])  # type: ignore[index]
    ps1_header = fits.Header(header_dict)
    ps1_wcs = WCS(ps1_header)
    ps1_shape = (int(row["NAXIS1"]), int(row["NAXIS2"]))
    return ps1_wcs, ps1_shape


def compute_ps1_shift_for_skycell(
    tess_wcs: WCS,
    dx_tess_pix: float,
    dy_tess_pix: float,
    sky_ra_deg: float,
    sky_dec_deg: float,
    ps1_wcs: WCS,
) -> tuple[float, float]:
    # Use the skycell center as evaluation point: map to TESS pixel, then perturb in TESS pixel space
    sc = SkyCoord(sky_ra_deg * u.deg, sky_dec_deg * u.deg, frame="icrs")
    x_tess, y_tess = tess_wcs.world_to_pixel(sc)

    # World at original and shifted TESS pixels
    world1 = tess_wcs.pixel_to_world(x_tess, y_tess)
    world2 = tess_wcs.pixel_to_world(x_tess + dx_tess_pix, y_tess + dy_tess_pix)

    # Map those world points into PS1 pixel coordinates
    u1, v1 = ps1_wcs.world_to_pixel(world1)
    u2, v2 = ps1_wcs.world_to_pixel(world2)

    return float(u2 - u1), float(v2 - v1)


def main():
    parser = argparse.ArgumentParser(description="Compute per-PS1 skycell pixel shifts from a small TESS pixel offset using WCS transforms.")
    parser.add_argument("--tess-fits", required=True, type=Path, help="Path to the TESS FITS file with WCS")
    parser.add_argument("--skycell-csv", required=True, type=Path, help="CSV with PS1 skycell WCS info")
    parser.add_argument("--dx", required=True, type=float, help="Delta x in TESS pixels")
    parser.add_argument("--dy", required=True, type=float, help="Delta y in TESS pixels")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")

    args = parser.parse_args()

    tess_wcs, _ = load_tess_wcs(args.tess_fits)

    usecols = ["NAME", "RA", "DEC"] + RELEVANT_WCS_KEYS
    df = pd.read_csv(args.skycell_csv, usecols=usecols)

    # Compute shifts per row
    shift_x_list = []
    shift_y_list = []

    for _, row in df.iterrows():
        ps1_wcs, _ps1_shape = build_ps1_wcs(row)
        sx, sy = compute_ps1_shift_for_skycell(
            tess_wcs,
            args.dx,
            args.dy,
            float(row["RA"]),
            float(row["DEC"]),
            ps1_wcs,
        )
        shift_x_list.append(sx)
        shift_y_list.append(sy)

    out_df = pd.DataFrame(
        {
            "NAME": df["NAME"],
            "shift_x": shift_x_list,
            "shift_y": shift_y_list,
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
