"""
SynDiff Production Pipeline
===========================
TESS FFI difference imaging pipeline.

Modules:
    download        -- TESS FFI downloader (tesscurl manifest + urllib; optional astroquery)
    config          -- Configuration dataclass and YAML I/O
    wcs_grouping    -- WCS extraction, crop bounds, template groups, Gaia catalog
    masking         -- Shared mask creation and hotpants reference star selection
    hotpants_runner -- Hotpants differencing loop (rounds 1 & 2)
    epsf_fitting    -- TGLC tiled ePSF fitting
    temporal_smooth -- Temporal smoothing of ePSF and background stacks
    sat_template    -- Saturated star template construction
    background      -- Background estimation and combination
    final_diff      -- Final difference image production
    photometry      -- Forced PSF photometry (PRF or ePSF)
    run_pipeline    -- CLI entry point (requires YAML ``pipeline:``)

Example YAML and smoke-test layout live under ``example/``.
"""
