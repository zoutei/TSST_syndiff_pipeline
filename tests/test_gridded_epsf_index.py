"""Tests for incremental gridded ePSF index updates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.stages import gridded_epsf


def test_register_gridded_epsf_index_entry(tmp_path):
    out = str(tmp_path / "epsf_r1")
    gridded_epsf.register_gridded_epsf_index_entry(out, "tess111")
    gridded_epsf.register_gridded_epsf_index_entry(out, "tess222")
    index = gridded_epsf.load_gridded_epsf_index(out)
    assert set(index.keys()) == {"tess111", "tess222"}
    assert index["tess111"].endswith("tess111_gridded_epsf.npz")

    raw = json.loads((Path(out) / gridded_epsf.GRIDDED_EPSF_INDEX_BASENAME).read_text())
    assert len(raw) == 2
