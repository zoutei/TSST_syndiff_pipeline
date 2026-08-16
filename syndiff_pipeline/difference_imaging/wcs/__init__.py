"""Per-FFI WCS fitting utilities."""

from syndiff_pipeline.difference_imaging.wcs.audit_matrix import (
    AUDIT_NPZ,
    load_stars_fit_audit,
    to_long_dataframe,
)
from syndiff_pipeline.difference_imaging.wcs.reference import reference_wcs_from_tesswcs
from syndiff_pipeline.difference_imaging.wcs.temporal_adapter import TemporalWcsAdapter
from syndiff_pipeline.difference_imaging.wcs.sci2idl import (
    FitConfig,
    Sci2IdlFitResult,
    StarSelectionConfig,
    build_frame_stars,
    crop_bounds_from_header,
    fit_sci2idl_distortion,
    join_stars,
    sci2idl_du_dv_px,
    select_good_stars,
    uv_from_linear_wcs,
    warmstart_frame,
    warmstart_table_row,
)

__all__ = [
    "AUDIT_NPZ",
    "FitConfig",
    "Sci2IdlFitResult",
    "StarSelectionConfig",
    "build_frame_stars",
    "crop_bounds_from_header",
    "fit_sci2idl_distortion",
    "join_stars",
    "load_stars_fit_audit",
    "reference_wcs_from_tesswcs",
    "TemporalWcsAdapter",
    "sci2idl_du_dv_px",
    "select_good_stars",
    "to_long_dataframe",
    "uv_from_linear_wcs",
    "warmstart_frame",
    "warmstart_table_row",
]
