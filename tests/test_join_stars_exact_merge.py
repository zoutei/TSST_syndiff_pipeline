"""join_stars: the exact float merge must not short-circuit the KD-tree fallback
when only a few rows coincide (template reference FFI of each CCD)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.table import Table

from syndiff_pipeline.common.proper_motion import GAIA_DR3_EPOCH_BTJD
from syndiff_pipeline.difference_imaging.wcs.sci2idl import _propagate_joined, join_stars

EPOCH = GAIA_DR3_EPOCH_BTJD + 4.0 * 365.25
N = 40


def _old_join_stars(phot, gaia, *, max_sep_px=0.25, epoch_btjd=None):
    """Verbatim pre-fix join_stars (exact merge accepted if non-empty)."""
    df = phot.to_pandas()
    pm_cols = ("pmra", "pmdec") if epoch_btjd is not None else ()
    gcols = [c for c in ("source_id", "ra", "dec", "x", "y", *pm_cols) if c in gaia.columns]
    g = gaia[gcols].copy()
    df_bare = df.drop(columns=[c for c in gcols if c in df.columns])
    exact = df_bare.merge(g, left_on=["x_init", "y_init"], right_on=["x", "y"], how="inner")
    if len(exact) > 0:
        return _propagate_joined(exact, epoch_btjd)
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([g["x"].to_numpy(float), g["y"].to_numpy(float)]))
    xy = np.column_stack([df["x_init"].to_numpy(float), df["y_init"].to_numpy(float)])
    dist, idx = tree.query(xy, k=1)
    keep = np.isfinite(dist) & (dist <= max_sep_px)
    if not np.any(keep):
        return exact
    matched = df.loc[keep].copy().reset_index(drop=True)
    g_hit = g.iloc[idx[keep]].reset_index(drop=True)
    for col in gcols:
        matched[col] = g_hit[col].to_numpy()
    matched["gaia_match_sep_px"] = dist[keep]
    return _propagate_joined(matched, epoch_btjd)


def _gaia():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "source_id": np.arange(N) + 100,
        "ra": 10.0 + rng.uniform(0, 0.1, N), "dec": 5.0 + rng.uniform(0, 0.1, N),
        "x": rng.uniform(5, 2000, N), "y": rng.uniform(5, 2000, N),
        "pmra": rng.normal(0, 50, N), "pmdec": rng.normal(0, 50, N),
    })


def _phot(gaia, n_exact, offset=0.05):
    x = gaia["x"].to_numpy().copy()
    y = gaia["y"].to_numpy().copy()
    x[n_exact:] += offset
    return Table({"x_init": x, "y_init": y, "x_fit": x + 0.01, "y_fit": y, "flags": np.zeros(N, int),
                  "source_id": np.full(N, -1), "ra": np.zeros(N), "dec": np.zeros(N)})


def test_one_coincidence_no_longer_hides_the_rest():
    g = _gaia()
    phot = _phot(g, 1)
    assert len(_old_join_stars(phot, g)) == 1  # the bug
    new = join_stars(phot, g)
    assert len(new) == N
    assert "gaia_match_sep_px" in new.columns
    assert (new["source_id"].to_numpy() == g["source_id"].to_numpy()).all()


def test_full_exact_uses_exact_path_identically():
    g = _gaia()
    phot = _phot(g, N)
    new, old = join_stars(phot, g), _old_join_stars(phot, g)
    pd.testing.assert_frame_equal(new, old)
    assert "gaia_match_sep_px" not in new.columns


def test_majority_exact_uses_exact_path():
    g = _gaia()
    phot = _phot(g, N // 2)
    pd.testing.assert_frame_equal(join_stars(phot, g), _old_join_stars(phot, g))


def test_none_exact_unchanged():
    g = _gaia()
    phot = _phot(g, 0)
    pd.testing.assert_frame_equal(join_stars(phot, g), _old_join_stars(phot, g))
    assert len(join_stars(phot, g)) == N


def test_no_other_neighbours_same_rows_as_before():
    g = _gaia()
    phot = _phot(g, 1, offset=5.0)  # offset beyond max_sep_px; one exact hit
    new = join_stars(phot, g)
    # the lone coincident star is still found (now via the KD path, which only
    # adds the gaia_match_sep_px column)
    assert (new["gaia_match_sep_px"] == 0).all()
    pd.testing.assert_frame_equal(new.drop(columns="gaia_match_sep_px"), _old_join_stars(phot, g))


def test_proper_motion_both_paths():
    g = _gaia()
    for n_exact in (N, 1):
        phot = _phot(g, n_exact)
        out = join_stars(phot, g, epoch_btjd=EPOCH)
        assert len(out) == N
        assert "ra_j2016" in out.columns
        moved = np.abs(out["ra"].to_numpy() - out["ra_j2016"].to_numpy()) > 0
        assert moved.any()
        np.testing.assert_allclose(np.sort(out["ra_j2016"].to_numpy()), np.sort(g["ra"].to_numpy()))
    pd.testing.assert_frame_equal(
        join_stars(_phot(g, N), g, epoch_btjd=EPOCH), _old_join_stars(_phot(g, N), g, epoch_btjd=EPOCH))
