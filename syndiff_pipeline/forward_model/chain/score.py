"""Stage ``score``: run the frozen SPEC v1.1 scorer on the final images of a frame (e2e ``f06_score.sh``).

The scorer is a frozen, separate code tree (``inputs.scorer_dir``, its own ``score.py``); this module only assembles its
command line from the chain outputs and runs it with that directory as cwd:

  --diff hp=<hotpants/<stem>/hp_d>            the Hotpants baseline (also ``--ref``: noise + mask planes)
  --diff <variant>=<final/<stem>/<variant>/hp_d>   every final variant of the chosen set
  --diff pass2=<inputs.pass2_hp[stem]>        optional reference product (the bootstrap diff), if configured
  --baseline hp --candidate band_w|band       G4 verdict of the per-band candidate against the baseline

``weighted=True`` (default) scores the adopted-weight set (band_w, achrom_w) with candidate ``band_w``; ``False`` the
production-weight set (band, achrom) with candidate ``band``.  Output: ``score/<stem>/{weighted|unweighted}/`` (the
scorer's metrics.json, figures, report).  The scorer's ``--field`` must be one of the fields it was frozen for
(``cfg.field``).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import is_done, mark_done, write_provenance
from .perband.paths import chain_paths, opt


def score_command(cfg, stem: str, weighted: bool = True) -> tuple[list[str], Path, Path]:
    """(argv, cwd, out_dir) of the scorer run for ``stem``."""
    P = chain_paths(cfg)
    scorer = Path(cfg.inputs.scorer_dir) if opt(cfg, "inputs.scorer_dir", None) is not None else None
    if scorer is None:
        raise KeyError("config field 'inputs.scorer_dir' (frozen scorer directory containing score.py) is required by the score stage")
    ref = P.hotpants_ref(stem)
    variants = ("band_w", "achrom_w") if weighted else ("band", "achrom")
    cand = variants[0]
    out = P.score / stem / ("weighted" if weighted else "unweighted")
    args = ["--diff", f"hp={ref}"]
    for v in variants:
        args += ["--diff", f"{v}={P.final / stem / v / 'hp_d' / f'{stem}_hp_d.fits.fz'}"]
    pass2 = opt(cfg, "inputs.pass2_hp", None)
    if pass2 and stem in pass2:
        args += ["--diff", f"pass2={pass2[stem]}"]
    argv = [sys.executable, "score.py", "--field", P.field, "--frame", stem, "--out", str(out), "--ref", str(ref),
            "--baseline", "hp", "--candidate", cand, "--label", f"chain_{P.field}", *args]
    return argv, scorer, out


def run(cfg, stems: Optional[Sequence[str]] = None, weighted: bool = True) -> Path:
    """Stage ``score``: needs ``final`` done.  Raises ``CalledProcessError`` if the scorer fails."""
    P = chain_paths(cfg)
    stage = cfg.stage_dir("score")
    if not is_done(cfg.stage_dir("final")):
        raise FileNotFoundError(f"stage final not done: {cfg.stage_dir('final')}")
    for stem in (list(stems) if stems else P.frames):
        argv, cwd, out = score_command(cfg, stem, weighted)
        out.mkdir(parents=True, exist_ok=True)
        print(" ".join(argv), flush=True)
        subprocess.run(argv, cwd=str(cwd), check=True)
    if all((P.score / s / ("weighted" if weighted else "unweighted") / "metrics.json").is_file() for s in P.frames):
        write_provenance(stage, cfg, {"scorer_dir": cfg.inputs.scorer_dir, "final": P.final / "summary.json"})
        mark_done(stage)
    return stage


def main(argv: Optional[list] = None) -> int:
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--stem", action="append")
    ap.add_argument("--unweighted", action="store_true", help="score the production-weight set (band, achrom)")
    a = ap.parse_args(argv)
    run(load_config(a.config), a.stem, weighted=not a.unweighted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
