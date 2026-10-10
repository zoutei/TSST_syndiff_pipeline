"""Run scene_fit with a versioned recipe (``recipes/<name>.yaml``) plus per-run arguments.

    python -m syndiff_pipeline.forward_model.recipe paper1_provisional -- \\
        --scene-dir S --out-dir O --init-params-file P --colour-file C

Arguments after ``--`` are appended after the recipe's flags, so they win for any flag given twice
(argparse keeps the last value). The recipe name and resolved flags are written to
``<out-dir>/recipe.json`` for provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

RECIPE_DIR = Path(__file__).resolve().parent / "recipes"


def recipe_argv(name: str) -> list[str]:
    """Flags for scene_fit from ``recipes/<name>.yaml`` (``key: value`` -> ``--key value``)."""
    spec = yaml.safe_load((RECIPE_DIR / f"{name}.yaml").read_text())["scene_fit"]
    argv: list[str] = []
    for key, value in spec.items():
        if value is True:
            argv.append(f"--{key}")
        elif value is False or value is None:
            continue
        else:
            argv += [f"--{key}", str(value)]
    return argv


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", help="recipe name (file stem in recipes/)")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then scene_fit arguments")
    a = ap.parse_args(argv)
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    full = recipe_argv(a.recipe) + rest

    from . import scene_fit as SF  # jax-heavy; import after argument parsing

    args = SF.build_parser().parse_args(full)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "recipe.json").write_text(
        json.dumps({"recipe": a.recipe, "argv": full,
                                                    "role": yaml.safe_load((RECIPE_DIR / f"{a.recipe}.yaml").read_text()).get("role")}, indent=1))
    SF.run(args)


if __name__ == "__main__":
    main()
