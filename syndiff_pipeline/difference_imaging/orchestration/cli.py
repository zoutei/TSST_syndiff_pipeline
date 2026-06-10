"""
run_pipeline.py
===============
Orchestrator / CLI entry point for the SynDiff TESS FFI difference imaging pipeline.

Runs the config-driven pipeline defined by the ``pipeline:`` list in YAML.

Usage:
    python -m syndiff_pipeline.difference_imaging.orchestration.cli --config config.yaml
    python -m syndiff_pipeline.difference_imaging.orchestration.cli --config recipe.yaml --validate-only
    python -m syndiff_pipeline.difference_imaging.orchestration.cli --config config.yaml --download
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

from astropy.wcs import FITSFixedWarning


def _ensure_pyhotpants_path():
    """Optionally add a nearby ``pyhotpants`` source tree (clone/install as documented)."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        php = p / "pyhotpants"
        if php.is_dir() and str(php) not in sys.path:
            sys.path.insert(0, str(php))


_ensure_pyhotpants_path()

warnings.filterwarnings("ignore", category=FITSFixedWarning)

from syndiff_pipeline.common.download import download_ffis, nested_ffi_dir
from syndiff_pipeline.difference_imaging.orchestration.config import add_config_args, config_from_args

log = logging.getLogger(__name__)


def run_pipeline(cfg, *, validate_only: bool = False):
    """
    Run SynDiff according to *cfg*.

    Requires a non-empty ``cfg.pipeline`` (YAML ``pipeline:`` list). Use
    ``--validate-only`` to validate stages without executing.
    """
    if not cfg.pipeline:
        raise ValueError(
            "Config must define a non-empty ``pipeline:`` list of stages. "
            "See syndiff_pipeline/example/recipe_*.yaml and README.md."
        )
    from syndiff_pipeline.difference_imaging.orchestration.execute import run_config_pipeline

    run_config_pipeline(cfg, validate_only=validate_only)


def main():
    parser = argparse.ArgumentParser(
        description="SynDiff TESS FFI difference imaging pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(parser)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download FFIs from MAST before running the pipeline.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate ``pipeline:`` stages and exit (no execution).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in ("astropy", "matplotlib", "PIL"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    cfg = config_from_args(args)

    if args.download:
        log.info("Downloading FFIs ...")
        download_ffis(
            cfg.sector,
            cfg.camera,
            cfg.ccd,
            nested_ffi_dir(
                cfg.sector, cfg.camera, cfg.ccd, root=cfg.ffi_dir
            ),
        )

    run_pipeline(cfg, validate_only=args.validate_only)


if __name__ == "__main__":
    main()
