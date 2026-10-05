# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Optional PRF fork on sys.path (not pip-installed; only ``psf_type: prf`` init paths need it).

In dev this module also wired sibling dev directories onto sys.path; those modules are now vendored
under ``_vendor/`` and imported by absolute name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

_EXTRA_PATHS = [
    _REPO / "tess_prf_oversample" / "src",
]

for _p in _EXTRA_PATHS:
    sp = str(_p)
    if _p.is_dir() and sp not in sys.path:
        sys.path.insert(0, sp)
