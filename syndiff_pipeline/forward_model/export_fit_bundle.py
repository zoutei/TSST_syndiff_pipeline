# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Export a prepared fit_bundle.npz (workspace prep only; no Adam).

Example matching orbit1_half_mag710_K1:

    python -m syndiff_pipeline.forward_model.export_fit_bundle \\
        --workspace data/data/s0020/c3/k3/diff_linear --sector 20 --orbit-index 1 \\
        --region 1536,1536,2048,2048 --epsf-grid 2x2 --cheb-degree 3 \\
        --n-frames 295 --frame-offset start --tess-mag 7,10 \\
        --wcs-n-interior-knots 20 --w-n-interior-knots 10 \\
        --edge-densify-knots --knots-anchor full-orbit \\
        --mode-init iso_defocus --no-mask-reject \\
        --out-dir dev/forward_epsf_wcs/output/bundles/orbit1_half_mag710
"""

from __future__ import annotations

import sys

from .run_fit import main as run_fit_main


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Force export-only; --out-dir doubles as bundle dir if --export-bundle omitted.
    if "--export-only" not in argv:
        argv.append("--export-only")
    if "--export-bundle" not in argv:
        # Prefer explicit --out-dir as bundle destination.
        if "--out-dir" in argv:
            i = argv.index("--out-dir")
            if i + 1 < len(argv):
                argv.extend(["--export-bundle", argv[i + 1]])
        else:
            raise SystemExit("export_fit_bundle requires --out-dir or --export-bundle")
    # Avoid accidentally starting Adam from leftover --stage flags: force start_stage
    # irrelevant via export-only; still pass through for meta.
    run_fit_main(argv)


if __name__ == "__main__":
    main()
