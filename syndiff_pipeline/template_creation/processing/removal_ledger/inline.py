"""Removal ledger captured inside ``ps1_process``, at the moment SEP removal runs.

The pixel operations are recorded through ``remove_background(recorder=...)``;
the image algorithm is unchanged and ``CellLedger.finish`` proves it per cell.
Catalogue association uses the uncut per-projection Gaia catalogue only (no
bulk PS1 stack catalogue); PS1 magnitudes are attached per projection later,
for associated Gaia stars only (``gaia_projection_catalog.ensure_gaia_ps1_mags``).

Ledger capture never decides whether a template cell is produced: on any
failure the caller reruns the cell without a recorder and reports the error.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .cell import CellLedger, attach_unmatched_components

LEDGER_STORE = Path("ps1_removal_ledger") / "v1"
ACCOUNTING_RADIUS_PX = 5.0
SOURCE_WINDOW_PX = 600.0
POINTER_NAME = "ledger.json"


def ledger_fingerprint_dir(data_root, projection, skycell, combined_fingerprint) -> Path:
    """``{data_root}/ps1_removal_ledger/v1/{projection}/{skycell}/{combined_fp}``."""
    return Path(data_root) / LEDGER_STORE / str(projection) / str(skycell) / str(combined_fingerprint)


def published_ledger(data_root, projection, skycell, combined_fingerprint) -> Path | None:
    """Path of the published ledger for one combined cell, or ``None``."""
    pointer = ledger_fingerprint_dir(data_root, projection, skycell, combined_fingerprint) / POINTER_NAME
    try:
        path = Path(json.loads(pointer.read_text())["ledger"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return path if (path / "manifest.json").is_file() else None


def capture_removal(image, uncert, mask, gaia_catalog_pixels, identity, *,
                    remove_saturated_stars, bright_star_mag_threshold):
    """Run footprint_v1 removal with a recorder; return ``(result, removed, ledger, raw)``.

    ``image`` is not modified. Raises on any recorder or validation failure.
    """
    from syndiff_pipeline.template_creation.processing import band_utils as bu

    raw = np.array(image, copy=True)
    ledger = CellLedger(raw, identity)
    result, removed = bu.remove_background(
        np.array(image, copy=True), uncert, mask=mask,
        remove_saturated_stars=remove_saturated_stars,
        gaia_catalog_pixels=gaia_catalog_pixels,
        bright_star_mag_threshold=bright_star_mag_threshold,
        convention=bu.REMOVAL_CONVENTION_FOOTPRINT,
        recorder=ledger,
    )
    ledger.finish(raw, result)
    seg = getattr(ledger, "segmentation", None)
    if seg is not None:
        # The SEP result itself: enough to replay the removal without SEP.
        union = (np.asarray(seg.segmap) > 0) | np.asarray(seg.mask_bright_stars, dtype=bool)
        ledger.geometry["segmentation_union"] = np.packbits(union.ravel())
        ledger.geometry["sep_bright_mask"] = np.packbits(np.asarray(seg.mask_bright_stars, dtype=bool).ravel())
        objects = getattr(seg, "objects", None)
        if objects is not None and len(objects):
            ledger.geometry["sep_objects"] = np.asarray(objects)
        del ledger.segmentation
    return result, removed, ledger, raw


def associate_gaia(ledger, raw, mask, header_str, gaia_all):
    """Gaia-only association, following the C4 pilot recipe without PS1 stacks.

    Returns ``(sources, associations, image_calibration, calibrators)``.
    """
    from astropy.io import fits
    from astropy.time import Time
    from astropy.wcs import WCS

    from syndiff_pipeline.template_creation.processing import band_utils as bu
    from .astrometry import calibrate_image_positions
    from .matching import gaia_sources

    header = fits.Header.fromstring(header_str)
    wcs = WCS(header)
    epoch = float(Time(header["MJD-OBS"], format="mjd").jyear) if "MJD-OBS" in header else None
    if gaia_all is None or len(gaia_all) == 0:
        raise ValueError("No Gaia catalogue rows for ledger association")
    sources = gaia_sources(gaia_all.reset_index(drop=True), wcs, epoch_year=epoch)
    h, w = raw.shape
    localized = np.isfinite(sources.pixel_x) & np.isfinite(sources.pixel_y)
    m = SOURCE_WINDOW_PX
    keep = (~localized) | (
        (sources.pixel_x >= -m) & (sources.pixel_x < w + m)
        & (sources.pixel_y >= -m) & (sources.pixel_y < h + m)
    )
    sources = sources[keep].reset_index(drop=True)
    sources["canonical_entity_key"] = sources["entity_key"]
    sources["identity_status"] = "gaia"
    sources, image_calibration, calibrators = calibrate_image_positions(raw, mask, sources)
    sources["source_type"] = "catalogue_source"
    sources["accounting_radius_px"] = ACCOUNTING_RADIUS_PX
    sources["support_definition"] = "5-pixel candidate aperture; unmeasured exterior PSF remains unknown"
    t = bu.compute_tess_mag(
        sources["phot_g_mean_mag"].to_numpy(float),
        sources["phot_bp_mean_mag"].to_numpy(float),
        sources["phot_rp_mean_mag"].to_numpy(float),
    )
    bright = np.flatnonzero(t < 13)
    if len(bright):
        sources.loc[bright, "accounting_radius_px"] = bu.star_footprint_radius(t[bright])
        sources.loc[bright, "support_definition"] = (
            "historical bright-star halo search radius; association candidate only"
        )
    sources, links = ledger.associate_centres(sources, support_radius_column="accounting_radius_px")
    sources, links = attach_unmatched_components(ledger, sources, links, wcs)
    return sources, links, image_calibration, calibrators


def publish_cell_ledger(data_root, projection, skycell, combined_fingerprint, ledger, *,
                        sources=None, associations=None, legacy_records=None,
                        image_calibration=None, calibrators=None, catalogue=None, metadata=None) -> Path:
    """Publish a validated ledger and point the combined fingerprint at it."""
    fp_dir = ledger_fingerprint_dir(data_root, projection, skycell, combined_fingerprint)
    extra = {}
    if legacy_records is not None:
        extra["legacy_records"] = pd.DataFrame(legacy_records)
    if calibrators is not None and len(calibrators):
        extra["image_calibrators"] = calibrators
    meta = dict(metadata or {})
    meta.update(
        capture="inline_ps1_process",
        image_calibration=image_calibration,
        geometry_units="native PS1 pixels; source positions zero based",
        candidate_support_radius_px=ACCOUNTING_RADIUS_PX,
        support_interpretation="candidate only; not total stellar PSF",
    )
    manifests = [] if catalogue is None else [dict(catalogue="gaia", **catalogue)]
    dest = ledger.publish(fp_dir, sources=sources, associations=associations,
                          catalogue_manifests=manifests, extra_tables=extra, metadata=meta)
    pointer = fp_dir / POINTER_NAME
    tmp = fp_dir / f".{POINTER_NAME}.{os.getpid()}"
    tmp.write_text(json.dumps(dict(ledger=str(dest), fingerprint=dest.name), indent=2))
    os.replace(tmp, pointer)
    return dest


def associated_gaia_ids(ledger_path) -> set[int]:
    """Gaia source IDs linked to any non-background operation of a published ledger."""
    path = Path(ledger_path)
    if not (path / "associations.parquet").is_file():
        return set()
    links = pd.read_parquet(path / "associations.parquet", columns=["source_key", "reason"])
    keys = links.loc[(links.reason != "background") & links.source_key.astype(str).str.startswith("gaia:"), "source_key"]
    return {int(k.split(":", 1)[1]) for k in keys.astype(str).unique()}
