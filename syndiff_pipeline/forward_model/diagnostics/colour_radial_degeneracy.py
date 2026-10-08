"""Print the weighted cosine matrix between the round A3 colour terms / shift and the radial family on one ePSF node.

    python -m syndiff_pipeline.forward_model.diagnostics.colour_radial_degeneracy <params.npz> [row col]
"""
from __future__ import annotations

import sys

import jax  # noqa: F401  (before numpy-heavy imports)
import jax.numpy as jnp
import numpy as np

from .. import epsf_model as EM


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    z = np.load(a[0])
    base = np.asarray(EM.decode_epsf_base(jnp.asarray(z["epsf_base_raw"])))
    i, j = (int(a[1]), int(a[2])) if len(a) >= 3 else (base.shape[0] // 2, base.shape[1] // 2)
    rows, cols, M = EM.colour_radial_degeneracy(base[i, j])
    np.set_printoptions(linewidth=250, precision=2, suppress=True)
    print(f"node ({i},{j}) of grid {base.shape[:2]}; knots {EM.get_radial_knots()}")
    print("cols:", " ".join(cols))
    for r, row in zip(rows, M):
        print(f"{r:8s}", " ".join(f"{v:+.2f}" for v in row))
    return rows, cols, M


if __name__ == "__main__":
    main()
