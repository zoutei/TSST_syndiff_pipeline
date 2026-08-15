"""``syndiff mask export`` — per-FFI temporal mask FITS from an SCC diff lane."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

from syndiff_pipeline.common.orchestration import cli as orch_cli
from syndiff_pipeline.common.orchestration.deployment import load_deployment_file
from syndiff_pipeline.common.orchestration.targets import parse_scc
from syndiff_pipeline.common.scc_paths import (
    normalize_store_name,
    scc_diff_dir,
    scc_diff_pipeline_plots_dir,
    scc_root,
)
from syndiff_pipeline.difference_imaging.masking.ffi_mask import (
    load_catalog_for_scc_lane,
    load_ffi_times_table_for_lane,
    mask_bit_summary,
    normalize_ffi_product_id,
    write_mask_fits_for_ffi,
)
from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    SitePaths,
    load_diff_site_policy,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    resolve_pipeline_artifact_path,
)
from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME

logger = logging.getLogger(__name__)

_MASK_WHICH = ("full", "static", "temporal")
_SCC_PART = re.compile(r"^[sScCkK]?(\d+)$")


def _parse_scc_arg(scc: str) -> tuple[int, int, int]:
    """Parse ``22/3/3``, ``s0022/c3/k3``, or ``s0022_c3_k3``."""
    parts = re.split(r"[,/]", scc.strip())
    if len(parts) == 3:

        def _int_part(part: str) -> int:
            m = _SCC_PART.match(part.strip())
            if not m:
                raise ValueError(f"invalid SCC component {part!r}")
            return int(m.group(1))

        return _int_part(parts[0]), _int_part(parts[1]), _int_part(parts[2])
    return parse_scc(scc)


def _resolve_scc(args: argparse.Namespace) -> tuple[int, int, int]:
    if getattr(args, "scc", None):
        return _parse_scc_arg(args.scc)
    sector = getattr(args, "sector", None)
    camera = getattr(args, "camera", None)
    ccd = getattr(args, "ccd", None)
    if sector is None or camera is None or ccd is None:
        raise SystemExit("Specify --scc or --sector, --camera, and --ccd")
    return int(sector), int(camera), int(ccd)


def _resolve_data_root(args: argparse.Namespace) -> Path:
    deploy_path = orch_cli._resolve_deployment_from_args(args)
    deployment = load_deployment_file(deploy_path)
    data_root = deployment.get("data_root")
    if not data_root:
        raise SystemExit(f"deployment.yaml missing data_root: {deploy_path}")
    return Path(str(data_root)).expanduser().resolve()


def _lanes_with_shared_mask(scc_dir: Path) -> list[tuple[str | None, Path]]:
    hits: list[tuple[str | None, Path]] = []
    for path in sorted(scc_dir.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if name != "diff" and not name.startswith("diff_"):
            continue
        if not resolve_pipeline_artifact_path(str(path), SHARED_MASK_FITS_BASENAME):
            continue
        store_name = None if name == "diff" else name[len("diff_") :]
        hits.append((store_name, path))
    return hits


def _store_name_from_site(site: str | None) -> str | None:
    if not site:
        return None
    paths = SitePaths.from_site_dir(site)
    if not paths.diff_config.is_file():
        return None
    policy = load_diff_site_policy(paths.diff_config)
    raw = (policy.paths or {}).get("output_store_name")
    if raw is None:
        return None
    return normalize_store_name(str(raw) if str(raw).strip() else None)


def _resolve_lane_root(
    data_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    lane: str | None,
    site: str | None,
) -> tuple[Path, str | None]:
    scc_dir = scc_root(data_root, sector, camera, ccd)
    if lane is not None:
        store_name = normalize_store_name(lane)
        lane_root = scc_diff_dir(data_root, sector, camera, ccd, store_name=store_name)
        if not resolve_pipeline_artifact_path(str(lane_root), SHARED_MASK_FITS_BASENAME):
            raise SystemExit(
                f"No shared_mask under diff lane {lane_root!s} "
                f"(lane={lane!r}, store_name={store_name!r})"
            )
        return lane_root, store_name

    preferred = _store_name_from_site(site)
    if preferred is not None:
        lane_root = scc_diff_dir(
            data_root, sector, camera, ccd, store_name=preferred
        )
        if resolve_pipeline_artifact_path(str(lane_root), SHARED_MASK_FITS_BASENAME):
            return lane_root, preferred

    hits = _lanes_with_shared_mask(scc_dir)
    if len(hits) == 1:
        store_name, lane_root = hits[0]
        return lane_root, store_name
    if not hits:
        raise SystemExit(
            f"No diff lane with {SHARED_MASK_FITS_BASENAME!r} under {scc_dir}"
        )
    lines = "\n".join(f"  {p} (lane={store!r})" for store, p in hits)
    raise SystemExit(
        f"Multiple diff lanes with shared_mask under {scc_dir}; pass --lane:\n{lines}"
    )


def _default_out_path(
    data_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    store_name: str | None,
    product_id: str,
) -> Path:
    out_dir = scc_diff_pipeline_plots_dir(
        data_root, sector, camera, ccd, "masks", store_name=store_name
    )
    return out_dir / f"mask_full_{product_id}.fits"


def cmd_export(args: argparse.Namespace) -> int:
    sector, camera, ccd = _resolve_scc(args)
    data_root = _resolve_data_root(args)
    site = getattr(args, "site", None)

    lane_root, store_name = _resolve_lane_root(
        data_root,
        sector,
        camera,
        ccd,
        lane=getattr(args, "lane", None),
        site=site,
    )

    product_id = normalize_ffi_product_id(args.ffi)
    which = str(args.which)

    catalog = load_catalog_for_scc_lane(
        lane_root,
        data_root=data_root,
        sector=sector,
        camera=camera,
        ccd=ccd,
    )
    manifest = load_ffi_times_table_for_lane(
        lane_root,
        data_root=data_root,
        sector=sector,
        camera=camera,
        ccd=ccd,
    )

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else _default_out_path(
            data_root, sector, camera, ccd, store_name, product_id
        )
    )

    write_mask_fits_for_ffi(
        catalog,
        product_id,
        out_path,
        wcs_table=manifest,
        which=which,  # type: ignore[arg-type]
        overwrite=bool(args.overwrite),
    )

    from astropy.io import fits

    data = fits.getdata(out_path)
    bits_present, b128 = mask_bit_summary(np.asarray(data, dtype=np.int16))
    print(f"Wrote {out_path}")
    print(f"  shape={data.shape} bits={bits_present} b128_pixels={b128}")
    if which in ("full", "temporal") and catalog.has_temporal() and b128 == 0:
        logger.warning(
            "No bit-128 pixels for %s at this epoch (cadence may be inactive)",
            product_id,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syndiff mask",
        description="Mask utilities for SCC diff lanes",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser(
        "export",
        help="Write per-FFI full mask FITS (static + asteroids) to debug_plots/masks/",
    )
    export.add_argument(
        "--site",
        default=None,
        help="Site config dir (optional when a single supervisor daemon is running)",
    )
    export.add_argument(
        "--deployment",
        default=None,
        help="deployment.yaml path (optional; default from --site or live daemon)",
    )
    export.add_argument("--scc", default=None, help="SCC as sector/camera/ccd")
    export.add_argument("--sector", type=int, default=None)
    export.add_argument("--camera", type=int, default=None)
    export.add_argument("--ccd", type=int, default=None)
    export.add_argument(
        "--ffi",
        required=True,
        help="FFI product id, stem, or basename (e.g. tess2020050192921)",
    )
    export.add_argument(
        "--lane",
        default=None,
        help="Diff store name (e.g. linear → diff_linear); auto-detect if omitted",
    )
    export.add_argument("--out", default=None, help="Output FITS path (override)")
    export.add_argument(
        "--which",
        choices=_MASK_WHICH,
        default="full",
        help="Mask layer: full (static+asteroids), static, or temporal",
    )
    export.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing output FITS (default: true)",
    )
    export.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
