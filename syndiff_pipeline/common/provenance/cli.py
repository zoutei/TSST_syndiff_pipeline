"""``bookkeeping`` CLI: reindex, query, stats, and DB-vs-disk verification.

Thin argparse front-end over :class:`ProvenanceStore` and :mod:`reindex`. All
subcommands take ``--data-root`` and operate on ``provenance.db`` under it. This
is the offline/operator surface; the hot pipeline path never shells out here.

Usage::

    python -m syndiff_pipeline.common.provenance.cli reindex --data-root /path
    python -m syndiff_pipeline.common.provenance.cli stats   --data-root /path
    python -m syndiff_pipeline.common.provenance.cli query   --data-root /path --kind combined_skycell
    python -m syndiff_pipeline.common.provenance.cli verify  --data-root /path
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from syndiff_pipeline.common.provenance.reindex import reindex_data_root
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_db_path


def _open_store(data_root: str) -> ProvenanceStore:
    return ProvenanceStore(provenance_db_path(data_root))


def _cmd_reindex(args: argparse.Namespace) -> int:
    store = _open_store(args.data_root)
    result = reindex_data_root(store, args.data_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    store = _open_store(args.data_root)
    by_kind: Counter = Counter()
    by_kind_state: Counter = Counter()
    for row in store.iter_artifacts():
        by_kind[row["kind"]] += 1
        by_kind_state[(row["kind"], row["state"])] += 1
    out = {
        "total": sum(by_kind.values()),
        "by_kind": dict(sorted(by_kind.items())),
        "by_kind_state": {f"{k}/{s}": n for (k, s), n in sorted(by_kind_state.items())},
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    store = _open_store(args.data_root)
    if args.fingerprint:
        art = store.get_artifact(args.fingerprint)
        if art is None:
            print(f"no artifact: {args.fingerprint}", file=sys.stderr)
            return 1
        art["inputs"] = store.artifact_inputs(args.fingerprint)
        art["recipe"] = store.get_recipe(art["recipe_id"])
        print(json.dumps(art, indent=2, sort_keys=True))
        return 0
    if args.recipe:
        recipe = store.get_recipe(args.recipe)
        arts = store.artifacts_by_recipe(args.recipe)
        print(json.dumps({"recipe": recipe, "artifacts": arts}, indent=2, sort_keys=True))
        return 0
    rows = list(store.iter_artifacts(kind=args.kind))
    if args.limit:
        rows = rows[: args.limit]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Report DB artifacts whose ``location`` no longer exists on disk (drift)."""
    store = _open_store(args.data_root)
    missing = []
    checked = 0
    for row in store.iter_artifacts(kind=args.kind):
        loc = row.get("location")
        checked += 1
        if loc and not Path(loc).exists():
            missing.append({"fingerprint": row["fingerprint"], "kind": row["kind"], "location": loc})
    out = {"checked": checked, "missing_on_disk": len(missing), "missing": missing[: args.limit or 50]}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 1 if missing else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bookkeeping", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_re = sub.add_parser("reindex", help="rebuild provenance.db from on-disk stores")
    p_re.add_argument("--data-root", required=True)
    p_re.set_defaults(func=_cmd_reindex)

    p_st = sub.add_parser("stats", help="artifact counts by kind/state")
    p_st.add_argument("--data-root", required=True)
    p_st.set_defaults(func=_cmd_stats)

    p_q = sub.add_parser("query", help="query artifacts / recipes")
    p_q.add_argument("--data-root", required=True)
    p_q.add_argument("--kind")
    p_q.add_argument("--recipe", help="recipe_id: show recipe + its artifacts")
    p_q.add_argument("--fingerprint", help="show one artifact with inputs + recipe")
    p_q.add_argument("--limit", type=int, default=0)
    p_q.set_defaults(func=_cmd_query)

    p_v = sub.add_parser("verify", help="report DB artifacts missing on disk")
    p_v.add_argument("--data-root", required=True)
    p_v.add_argument("--kind")
    p_v.add_argument("--limit", type=int, default=50)
    p_v.set_defaults(func=_cmd_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
