"""
cli.py
======
``syndiff bookkeeping {reindex,stats,query,verify}`` -- read/rebuild access to
the provenance graph. Wired additively into
``syndiff_pipeline/common/orchestration/cli.py``'s ``build_parser`` (see
``register_bookkeeping_subparser`` below), following the existing ``notify``
nested-subcommand pattern. No existing command's behavior changes.

- ``reindex``: rebuild (or top up) ``provenance.db`` from disk. Offline only.
- ``stats``: row counts by kind/state.
- ``query``: look up one artifact by fingerprint, or list artifacts for a
  ``(kind, spatial_key)``.
- ``verify``: freshly compute the expected fingerprint for a template stage
  (via the *existing* ``config_fingerprint``/``resolve_config`` machinery,
  read-only) and report whether the store already has it -- a manual,
  read-only echo of what PR3's scheduler cutover will do automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

__all__ = ["register_bookkeeping_subparser", "main"]


def _store_for_data_root(data_root: str, *, read_only: bool = False):
    from syndiff_pipeline.common.provenance.store import ProvenanceStore
    from syndiff_pipeline.common.scc_paths import provenance_db_path

    return ProvenanceStore(provenance_db_path(data_root), read_only=read_only)


def _resolve_data_root(args: argparse.Namespace) -> str:
    if getattr(args, "data_root", None):
        return args.data_root
    config_path = getattr(args, "config", None)
    if not config_path:
        site = getattr(args, "site", None)
        if site:
            from syndiff_pipeline.difference_imaging.orchestration.site_config import SitePaths

            config_path = str(SitePaths.from_site_dir(site).template_config)
    if not config_path:
        raise SystemExit(
            "syndiff bookkeeping: specify --data-root, or --config/--site to derive it"
        )
    from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config

    cfg = load_runner_config(config_path)
    if not cfg.data_root:
        raise SystemExit(f"{config_path}: no data_root configured")
    return cfg.data_root


def cmd_bookkeeping_reindex(args: argparse.Namespace) -> int:
    from syndiff_pipeline.common.provenance.reindex import reindex_data_root

    data_root = _resolve_data_root(args)
    store = _store_for_data_root(data_root)
    result = reindex_data_root(data_root, store, clear_first=not args.incremental)
    print(
        json.dumps(
            {
                "data_root": data_root,
                "shared_store_artifacts": result.shared_store_artifacts,
                "shared_store_legacy": result.shared_store_legacy,
                "scc_legacy_artifacts": result.scc_legacy_artifacts,
                "sccs_scanned": result.sccs_scanned,
                "total_ingested": result.total_ingested,
                "errors": result.errors,
            },
            indent=2,
        )
    )
    return 1 if result.errors else 0


def cmd_bookkeeping_stats(args: argparse.Namespace) -> int:
    data_root = _resolve_data_root(args)
    store = _store_for_data_root(data_root, read_only=True)
    print(json.dumps(store.stats(), indent=2))
    return 0


def cmd_bookkeeping_query(args: argparse.Namespace) -> int:
    data_root = _resolve_data_root(args)
    store = _store_for_data_root(data_root, read_only=True)

    if args.fingerprint:
        row = store.artifact(args.fingerprint)
        if row is None:
            print(f"not found: {args.fingerprint}")
            return 1
        print(
            json.dumps(
                {
                    "fingerprint": row.fingerprint,
                    "kind": row.kind,
                    "spatial_key": row.spatial_key,
                    "recipe_id": row.recipe_id,
                    "location": row.location,
                    "state": row.state,
                    "bytes": row.bytes,
                    "wall_time_s": row.wall_time_s,
                    "produced_by": row.produced_by,
                    "created_at": row.created_at,
                    "inputs": store.inputs_of(row.fingerprint),
                },
                indent=2,
            )
        )
        return 0

    if args.kind and args.spatial_key:
        try:
            spatial_key = json.loads(args.spatial_key)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--spatial-key must be JSON: {exc}") from exc
        rows = store.artifacts_by_kind_spatial(args.kind, spatial_key)
        print(
            json.dumps(
                [
                    {
                        "fingerprint": r.fingerprint,
                        "recipe_id": r.recipe_id,
                        "state": r.state,
                        "location": r.location,
                        "created_at": r.created_at,
                    }
                    for r in rows
                ],
                indent=2,
            )
        )
        return 0

    raise SystemExit(
        "syndiff bookkeeping query: specify --fingerprint, or --kind + --spatial-key"
    )


def cmd_bookkeeping_verify(args: argparse.Namespace) -> int:
    """Read-only echo of what the checkpoint-first verify path will do (PR3):
    recompute today's expected fingerprint for one template stage and report
    whether the store already has it. Does not touch the run-state DB or the
    scheduler; safe to run any time."""
    from syndiff_pipeline.common.orchestration.targets import find_target, load_targets
    from syndiff_pipeline.common.provenance import model
    from syndiff_pipeline.common.provenance.fingerprint import fingerprint as make_fp
    from syndiff_pipeline.common.provenance.fingerprint import recipe_id as make_rid
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        load_runner_config,
        resolve_config,
    )

    data_root = _resolve_data_root(args)
    store = _store_for_data_root(data_root, read_only=True)

    if not args.config:
        raise SystemExit("syndiff bookkeeping verify requires --config")
    if not args.targets or not args.scc:
        raise SystemExit("syndiff bookkeeping verify requires --targets and --scc")

    cfg = load_runner_config(args.config)
    targets = load_targets(args.targets)
    target = find_target(targets, args.scc)
    resolved = resolve_config(target, cfg)

    builders = {
        "mapping": model.mapping_recipe_params,
        "remap_store": model.remap_store_recipe_params,
        "downsample": model.downsample_recipe_params,
    }
    stage = args.stage
    if stage not in builders:
        raise SystemExit(f"stage must be one of {sorted(builders)}, got {stage!r}")

    params = builders[stage](resolved)
    rid = make_rid(stage, params, 1)
    spatial_key = {"s": target.sector, "c": target.camera, "k": target.ccd}
    fp = make_fp(stage, spatial_key, rid, [])
    row = store.artifact(fp)
    print(
        json.dumps(
            {
                "stage": stage,
                "target": target.label(),
                "params": params,
                "recipe_id": rid,
                "fingerprint": fp,
                "in_store": row is not None,
                "state": row.state if row is not None else None,
            },
            indent=2,
        )
    )
    return 0 if row is not None and row.state == "complete" else 1


def _add_data_root_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--data-root", default=None, help="Pipeline data root (direct)")
    sp.add_argument("--config", default=None, help="RunnerConfig YAML to derive data_root from")
    sp.add_argument("--site", default=None, help="Site dir to derive --config from")


def register_bookkeeping_subparser(sub: "argparse._SubParsersAction") -> None:
    """Register ``bookkeeping`` under an existing ``syndiff`` subparsers action.

    Called from ``common/orchestration/cli.py::build_parser`` -- additive
    only, no existing subcommand is touched.
    """
    sp = sub.add_parser("bookkeeping", help="Provenance graph: reindex, stats, query, verify")
    bk_sub = sp.add_subparsers(dest="bookkeeping_action", required=True)

    sp_reindex = bk_sub.add_parser("reindex", help="Rebuild provenance.db from on-disk content")
    _add_data_root_args(sp_reindex)
    sp_reindex.add_argument(
        "--incremental",
        action="store_true",
        help="Top up the existing DB instead of clearing it first",
    )
    sp_reindex.set_defaults(func=cmd_bookkeeping_reindex)

    sp_stats = bk_sub.add_parser("stats", help="Row counts by kind/state")
    _add_data_root_args(sp_stats)
    sp_stats.set_defaults(func=cmd_bookkeeping_stats)

    sp_query = bk_sub.add_parser("query", help="Look up one artifact or list by kind/spatial_key")
    _add_data_root_args(sp_query)
    sp_query.add_argument("--fingerprint", default=None)
    sp_query.add_argument("--kind", default=None)
    sp_query.add_argument("--spatial-key", default=None, help="JSON dict, e.g. '{\"s\":20,\"c\":1,\"k\":1}'")
    sp_query.set_defaults(func=cmd_bookkeeping_query)

    sp_verify = bk_sub.add_parser(
        "verify", help="Read-only: recompute a stage fingerprint and check the store"
    )
    _add_data_root_args(sp_verify)
    sp_verify.add_argument("--targets", default=None, help="Targets CSV")
    sp_verify.add_argument("--scc", default=None, help="Target label or SCC key (e.g. 20/1/1)")
    sp_verify.add_argument(
        "--stage", default="mapping", help="mapping | remap_store | downsample"
    )
    sp_verify.set_defaults(func=cmd_bookkeeping_verify)


def main(argv: Optional[list] = None) -> int:
    """Standalone entry point (``python -m syndiff_pipeline.common.provenance.cli``),
    useful for testing this module without the full ``syndiff`` parser tree."""
    parser = argparse.ArgumentParser(prog="syndiff bookkeeping")
    sub = parser.add_subparsers(dest="command", required=True)
    register_bookkeeping_subparser(sub)
    # register_bookkeeping_subparser expects to add one "bookkeeping" level;
    # for standalone use we want its subactions at the top level instead.
    args = parser.parse_args(["bookkeeping", *(argv or [])])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
