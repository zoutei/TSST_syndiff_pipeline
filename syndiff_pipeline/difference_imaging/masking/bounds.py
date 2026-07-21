"""Science-grid FFI bounds for masking (1-based row/col conventions)."""

from __future__ import annotations

from typing import Any

# Legacy full-chip defaults (pre-v2); prefer :func:`science_bounds_1based`.
SCI_COL_LO, SCI_COL_HI = 45, 2092
SCI_ROW_LO, SCI_ROW_HI = 1, 2048


def science_bounds_1based(crop_bounds: dict[str, Any] | None = None) -> dict[str, int]:
    """
    Return 1-based inclusive FFI limits for masking catalogs.

    When *crop_bounds* is a v2 science dict (from ``MappingGrid.science_ffi_bounds``),
    derive limits from ``x_min``/``x_max``/``y_max``. Otherwise return legacy defaults.
    """
    if crop_bounds and all(k in crop_bounds for k in ("x_min", "x_max", "y_max")):
        return {
            "col_lo": int(crop_bounds["x_min"]) + 1,
            "col_hi": int(crop_bounds["x_max"]),
            "row_lo": 1,
            "row_hi": int(crop_bounds["y_max"]),
        }
    return {
        "col_lo": SCI_COL_LO,
        "col_hi": SCI_COL_HI,
        "row_lo": SCI_ROW_LO,
        "row_hi": SCI_ROW_HI,
    }
