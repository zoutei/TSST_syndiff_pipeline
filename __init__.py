"""
SynDiff Production Pipeline
===========================
TESS FFI difference imaging pipeline.

Modules:
    download        -- TESS FFI downloader (tesscurl manifest + urllib; optional astroquery)
    config          -- Configuration dataclass and YAML I/O
    ffi_naming      -- FFI product-id parsing and per-workspace FITS basenames
    wcs_grouping    -- WCS extraction, crop bounds, template groups, Gaia catalog
    masking         -- Shared mask creation and hotpants reference star selection
    hotpants_runner -- Hotpants differencing loop (rounds 1 & 2)
    epsf_fitting    -- TGLC tiled ePSF fitting and ePSF stack I/O
    sat_template    -- Saturated star template construction
    background      -- TESSreduce-like spatial background + AdaptiveBackground temporal smoothing
    photometry      -- Forced PSF photometry (PRF or ePSF)
    run_pipeline    -- CLI entry point (requires YAML ``pipeline:``)

Example YAML and smoke-test layout live under ``example/``.
"""
