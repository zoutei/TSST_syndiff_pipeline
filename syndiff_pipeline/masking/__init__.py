"""Shared-mask package: empirical / TESSreduce bits, TNS, asteroids, MaskCatalog."""

from syndiff_pipeline.masking.bits import (
    ASTEROID,
    BRIGHT_CAT,
    EDGE,
    EPSF_IGNORE_BITS,
    FAINT_CAT,
    PS1,
    SAT_CROSS,
    STRAP,
    STRAP_SOURCE_BITS,
    TNS,
    epsf_reject_mask,
    full_mask_bool,
    strap_column_mask,
    strap_source_mask,
)
from syndiff_pipeline.masking.catalog import MaskCatalog
from syndiff_pipeline.masking.settings import (
    DEFAULT_TNS_PUBLIC_ZIP_URL,
    DEFAULT_TESS_ORBIT_TIMES_URL,
    MaskSettings,
    load_mask_settings,
    resolve_mask_settings,
)
from syndiff_pipeline.masking.shared import Cat_mask, build_static_mask, make_shared_mask

__all__ = [
    "ASTEROID",
    "BRIGHT_CAT",
    "EDGE",
    "EPSF_IGNORE_BITS",
    "FAINT_CAT",
    "PS1",
    "SAT_CROSS",
    "STRAP",
    "STRAP_SOURCE_BITS",
    "TNS",
    "DEFAULT_TNS_PUBLIC_ZIP_URL",
    "DEFAULT_TESS_ORBIT_TIMES_URL",
    "Cat_mask",
    "MaskCatalog",
    "MaskSettings",
    "build_static_mask",
    "epsf_reject_mask",
    "full_mask_bool",
    "load_mask_settings",
    "make_shared_mask",
    "resolve_mask_settings",
    "strap_column_mask",
    "strap_source_mask",
]
