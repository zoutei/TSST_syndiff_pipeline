"""Optional star-isolation cut in the centroids stage (Paper III E5 rule)."""
import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    CentroidsParams,
    parse_centroids,
)
from syndiff_pipeline.difference_imaging.orchestration.provenance_glue import _asdict_any
from syndiff_pipeline.difference_imaging.stages import centroids

E5 = [[2.5, 3e-2], [4.0, 0.1], [7.0, 0.5]]


def _df(rows):
    """rows: (x, y, tess_mag). Gaia photometry is faked so tess_mag round-trips."""
    df = pd.DataFrame(rows, columns=["x", "y", "tess_mag"])
    df["source_id"] = np.arange(len(df))
    return df


def _patch_mag(monkeypatch):
    from syndiff_pipeline.difference_imaging.stages import gridded_epsf

    def ensure(df):
        return df.copy()

    monkeypatch.setattr(gridded_epsf, "_ensure_tess_mag_column", ensure)


def _flux_mag(m0, ratio):
    return m0 - 2.5 * np.log10(ratio)


def _pair(sep, ratio, m0=10.0):
    return [(100.0, 100.0, m0), (100.0 + sep, 100.0, _flux_mag(m0, ratio))]


@pytest.mark.parametrize(
    "sep,ratio,kept",
    [(2.0, 0.05, False), (2.0, 0.01, True), (6.0, 0.6, False), (6.0, 0.4, True)],
)
def test_e5_rule_pair(monkeypatch, sep, ratio, kept):
    _patch_mag(monkeypatch)
    # neighbour is made the faint-end (>12.95) star for the ratio<1 cases -> still counts
    df = _df(_pair(sep, ratio))
    p = CentroidsParams(isolation_criteria=E5, tess_mag_max=12.95)
    out = centroids._filter_gaia_for_centroids(df, p)
    assert (0 in set(out.source_id)) is kept


def test_faint_neighbour_beyond_mag_max_counts(monkeypatch):
    _patch_mag(monkeypatch)
    df = _df(_pair(2.0, 0.05, m0=12.5))  # neighbour mag ~15.8 > tess_mag_max
    assert df.tess_mag[1] > 12.95
    out = centroids._filter_gaia_for_centroids(df, CentroidsParams(isolation_criteria=E5))
    assert 0 not in set(out.source_id) and 1 not in set(out.source_id)
    # neighbour fainter than neighbor_mag_max is ignored
    out = centroids._filter_gaia_for_centroids(
        df, CentroidsParams(isolation_criteria=E5, isolation_neighbor_mag_max=14.0))
    assert 0 in set(out.source_id)


def test_neighbours_never_dropped_only_candidates(monkeypatch):
    _patch_mag(monkeypatch)
    # bright star 3 px from a faint candidate: faint one dropped (ratio of bright/faint huge); bright kept
    df = _df([(50, 50, 9.0), (53, 50, 12.0)])
    out = centroids._filter_gaia_for_centroids(df, CentroidsParams(isolation_criteria=E5))
    assert list(out.source_id) == [0]


def test_none_identical_to_before(monkeypatch):
    _patch_mag(monkeypatch)
    rng = np.random.default_rng(0)
    df = _df(np.c_[rng.uniform(0, 100, 500), rng.uniform(0, 100, 500), rng.uniform(6, 16, 500)])
    out = centroids._filter_gaia_for_centroids(df, CentroidsParams())
    m = df.tess_mag
    ref = df[(m < 12.95) & (m > 7.5)].reset_index(drop=True)
    pd.testing.assert_frame_equal(out, ref)


def test_vectorised_matches_bruteforce():
    rng = np.random.default_rng(1)
    n = 3000
    x, y = rng.uniform(0, 150, n), rng.uniform(0, 150, n)
    mag = rng.uniform(7, 17, n)
    cand = (mag > 7.5) & (mag < 12.95)
    got = centroids.isolated_candidate_mask(x, y, mag, cand, E5, 17.0)
    f = 10 ** (-0.4 * mag)
    exp = cand.copy()
    for i in np.flatnonzero(cand):
        d = np.hypot(x - x[i], y - y[i]); fr = f / f[i]; d[i] = 1e9
        for s, r in E5:
            if np.any((d < s) & (fr > r)):
                exp[i] = False
    assert np.array_equal(got, exp)


def test_fingerprint_unchanged_when_unset():
    d = _asdict_any(CentroidsParams())
    assert "isolation_criteria" not in d and "isolation_neighbor_mag_max" not in d
    assert set(d) == {"tess_mag_max", "tess_mag_min", "fit_shape", "aperture_radius",
                      "psf_grouper_min_separation", "centroids_max_group_size", "centroids_n_jobs"}
    d2 = _asdict_any(CentroidsParams(isolation_criteria=E5))
    assert d2["isolation_criteria"] == E5


def test_parse_and_validate():
    p = parse_centroids({"kind": "centroids", "isolation_criteria": [[2.5, 0.03]]}, 0)
    assert p.isolation_criteria == [[2.5, 0.03]]
    assert parse_centroids({"kind": "centroids"}, 0).isolation_criteria is None
    with pytest.raises(ValueError):
        parse_centroids({"kind": "centroids", "isolation_criteria": [[2.5]]}, 0)
