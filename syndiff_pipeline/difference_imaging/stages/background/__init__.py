"""Unified background stage: spatial → temporal → strap."""

from syndiff_pipeline.difference_imaging.stages.background.io import (
    STACK_BASENAME,
    FrameRecord,
    build_frame_records,
    build_frame_records_from_stack_ws,
    load_flux_cubes,
    load_stack,
    load_stack_or_fits,
    save_stack,
    write_per_frame_fits,
)
from syndiff_pipeline.difference_imaging.stages.background.pipeline import (
    BackgroundParams,
    BackgroundStepSpatialParams,
    BackgroundStepStrapParams,
    BackgroundStepTemporalParams,
    btjd_for_records,
    run_background_pipeline,
)

__all__ = [
    "STACK_BASENAME",
    "BackgroundParams",
    "BackgroundStepSpatialParams",
    "BackgroundStepStrapParams",
    "BackgroundStepTemporalParams",
    "FrameRecord",
    "build_frame_records",
    "build_frame_records_from_stack_ws",
    "btjd_for_records",
    "load_flux_cubes",
    "load_stack",
    "load_stack_or_fits",
    "run_background_pipeline",
    "save_stack",
    "write_per_frame_fits",
]
