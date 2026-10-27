"""Per-node template-offset kernel shift (chain/template_shift.py): config keys, no-op when unset/zero, exact translation."""
import json

import numpy as np
import pytest
import yaml

from syndiff_pipeline.forward_model.chain import config as C
from syndiff_pipeline.forward_model.chain import template_shift as TS

from chain_fixtures import raw_config

NODES = np.linspace(0.0, 2048.0, 6)


def _cfg(tmp_path, **inputs):
    raw = raw_config(tmp_path)
    raw["inputs"].update(inputs)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return C.load_config(p)


def _spec(path, tx, ty, nx=NODES, ny=NODES):
    path.write_text(json.dumps(dict(node_x=list(nx), node_y=list(ny), tx_mpx=np.asarray(tx).tolist(), ty_mpx=np.asarray(ty).tolist())))
    return path


def _gauss_kernels(n=63, sig=4.0):
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
    g /= g.sum()
    return np.broadcast_to(g, (6, 6, n, n)).copy()


def _centroid(K):
    n = K.shape[-1]
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    s = K.sum((-2, -1))
    return (K * xx).sum((-2, -1)) / s, (K * yy).sum((-2, -1)) / s


def test_config_unset_auto_and_path(tmp_path):
    assert _cfg(tmp_path).inputs.template_shift is None
    assert TS.spec_path(_cfg(tmp_path)) is None
    c = _cfg(tmp_path, template_shift="auto", template_shift_ps1=str(tmp_path / "ps1.json"), template_shift_gaia=str(tmp_path / "g"))
    assert c.inputs.template_shift == "auto"
    assert c.inputs.template_shift_ps1.name == "ps1.json"
    with pytest.raises(FileNotFoundError):          # auto needs the template_shift stage DONE
        TS.spec_path(c)
    p = _spec(tmp_path / "s.json", np.zeros((6, 6)), np.zeros((6, 6)))
    assert TS.spec_path(_cfg(tmp_path, template_shift=str(p))) == p
    assert "template_shift" in C.STAGES


def test_unknown_input_key_still_rejected(tmp_path):
    with pytest.raises(C.ConfigError):
        _cfg(tmp_path, template_shiftt="auto")


def test_zero_spec_is_bit_identical(tmp_path):
    K = _gauss_kernels()
    ts = TS.load_spec(_spec(tmp_path / "z.json", np.zeros((6, 6)), np.zeros((6, 6))), NODES, NODES)
    out = TS.apply(K, ts)
    assert out is K
    Kb = np.stack([K, 2 * K])
    assert TS.apply(Kb, ts) is Kb


def test_nonzero_spec_translates_each_node(tmp_path):
    K = _gauss_kernels()
    rng = np.random.default_rng(0)
    tx, ty = rng.uniform(-12, 12, (6, 6)), rng.uniform(-12, 12, (6, 6))   # mpx
    ts = TS.load_spec(_spec(tmp_path / "s.json", tx, ty), NODES, NODES)
    out = TS.apply(K, ts)
    cx0, cy0 = _centroid(K)
    cx1, cy1 = _centroid(out)
    os_ = 4                                         # kernels are on the OS4 subcell grid
    np.testing.assert_allclose((cx1 - cx0) / os_, -tx / 1e3, atol=2e-6)
    np.testing.assert_allclose((cy1 - cy0) / os_, -ty / 1e3, atol=2e-6)
    np.testing.assert_allclose(out.sum((-2, -1)), K.sum((-2, -1)), atol=1e-10)
    # per-band stack broadcasts over the leading band axis
    Kb = TS.apply(np.stack([K, K]), ts)
    np.testing.assert_array_equal(Kb[0], out)


def test_spec_node_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError):
        TS.load_spec(_spec(tmp_path / "a.json", np.zeros((5, 6)), np.zeros((5, 6))), NODES, NODES)
    with pytest.raises(ValueError):
        TS.load_spec(_spec(tmp_path / "b.json", np.zeros((6, 6)), np.zeros((6, 6)), nx=NODES + 1), NODES, NODES)
    bad = np.zeros((6, 6)); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        TS.load_spec(_spec(tmp_path / "c.json", bad, np.zeros((6, 6))), NODES, NODES)


# ---------------------------------------------------------------- PS1 CRPIX convention (inputs.template_shift_ps1_crpix)
import types

import pandas as pd

from syndiff_pipeline.forward_model.chain import kernels as KN

N_CELLS = 12


def _cells(east=-125.0, north=125.0, n=N_CELLS):
    return [dict(proj="2386", cell=f"{i:03d}", name=f"skycell.2386.{i:03d}", n=50, ra_c=10.0 + 0.01 * i, dec_c=20.0,
                 mjd=56000.0, east_mas=east, north_mas=north) for i in range(n)]


def _master(path, o=None, n=N_CELLS):
    df = pd.DataFrame(dict(NAME=[f"skycell.2386.{i:03d}" for i in range(n)], CDELT1=6.9444446e-05, CDELT2=6.9444446e-05,
                           PC1_1=-1.0, PC1_2=0.0, PC2_1=0.0, PC2_2=1.0))
    if o is not None:
        df["PS1CRPIX"] = o
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _master_in_mapping(cfg, o=None):
    return _master(cfg.stage_dir("mapping") / "oversampling_4" / "tess_s0024_2_2_master_skycells_list_os4.csv", o)


def _setup(tmp_path, monkeypatch, cells, **inputs):
    """cfg + monkeypatched fitted-WCS stand-ins (J = 10 px/arcsec identity) and an empty Gaia dir (no PM term)."""
    ps1 = tmp_path / "ps1.json"
    ps1.write_text(json.dumps(cells))
    gdir = tmp_path / "gaia"
    gdir.mkdir(exist_ok=True)
    cols = ["source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
    pd.DataFrame({k: np.array([], float) for k in cols}).to_parquet(gdir / "proj_2386.parquet")
    cfg = _cfg(tmp_path, template_shift_ps1=str(ps1), template_shift_gaia=str(gdir), **inputs)
    nodes = np.linspace(0.0, 2048.0, 4)
    monkeypatch.setattr(KN, "load_model", lambda c: dict(params=dict(wcs_coeff=np.zeros(3))))
    monkeypatch.setattr(KN, "load_bundle", lambda c: types.SimpleNamespace(
        cheb_static=None, epsf_grid=types.SimpleNamespace(node_x=nodes, node_y=nodes)))
    monkeypatch.setattr(KN, "make_pix", lambda coeff, st: (lambda ra, dec: (np.asarray(ra) * 100.0, np.asarray(ra) * 100.0)))
    monkeypatch.setattr(KN, "jac", lambda pix, ra, dec: np.tile(10.0 * np.eye(2), (len(np.atleast_1d(ra)), 1, 1)))
    return cfg


def _real_cells(n=N_CELLS, **kw):          # x = 100 ra in [1000, 1110]: all on the CCD
    return _cells(n=n, **kw)


def test_crpix_knob_config_and_identity_hash(tmp_path):
    c0 = _cfg(tmp_path)
    assert c0.inputs.template_shift_ps1_crpix == "auto"
    assert "template_shift_ps1_crpix" not in c0.to_dict()["inputs"]          # default knob stays out of the identity
    assert _cfg(tmp_path, template_shift_ps1_crpix="auto").config_hash() == c0.config_hash()
    c5 = _cfg(tmp_path, template_shift_ps1_crpix=0.5)
    assert c5.inputs.template_shift_ps1_crpix == 0.5
    assert c5.config_hash() != c0.config_hash()
    assert _cfg(tmp_path, template_shift_ps1_crpix=0).inputs.template_shift_ps1_crpix == 0.0
    for bad in ("yes", True, 99, float("nan"), None, [0.5]):
        with pytest.raises(C.ConfigError):
            _cfg(tmp_path, template_shift_ps1_crpix=bad)


def test_convention_offset_sign_matches_dx_dy():
    # measured record: dx=0.2588898, dy=0.6397260 PS1 px -> east -64.722, north +159.932 mas (CDELT 6.9444446e-5, PC=[[-1,0],[0,1]])
    row = pd.DataFrame(dict(NAME=["skycell.2386.060"], CDELT1=6.9444446e-05, CDELT2=6.9444446e-05,
                            PC1_1=-1.0, PC1_2=0.0, PC2_1=0.0, PC2_2=1.0))
    cell = [dict(proj="2386", cell="060", name="skycell.2386.060")]
    np.testing.assert_allclose(TS.convention_offset_mas(cell, 0.2588898398953461, row)[0][0], -64.72246154105093, atol=1e-3)
    np.testing.assert_allclose(TS.convention_offset_mas(cell, 0.6397260010675154, row)[0][1], 159.93150413952142, atol=1e-3)
    np.testing.assert_allclose(TS.convention_offset_mas(cell, 0.5, row)[0], [-125.0, 125.0], atol=1e-3)
    # master CSV missing the skycell -> bundled table row; neither -> raise
    np.testing.assert_allclose(TS.convention_offset_mas(cell, 0.5, None)[0], [-125.0, 125.0], atol=1e-3) if _bundled_has("skycell.2386.060") else None
    with pytest.raises(KeyError):
        TS.convention_offset_mas([dict(proj="9", cell="999", name="skycell.9999.999")], 0.5, row)


def _bundled_has(name):
    from importlib import resources
    t = pd.read_csv(resources.files("syndiff_pipeline") / "resources" / "skycell_wcs.csv")
    return name in set(t.NAME)


def test_o0_legacy_spec_unchanged(tmp_path, monkeypatch):
    cells = _real_cells(east=-64.7, north=159.9)
    cfg = _setup(tmp_path, monkeypatch, cells)
    _master_in_mapping(cfg)                                                  # legacy CSV: no PS1CRPIX column
    spec = TS.compute(cfg)
    assert "ps1_crpix_offset" not in spec and "convention_mean_mas" not in spec    # legacy spec keys unchanged
    # reference: the pre-change arithmetic, J (10 px/arcsec) @ (east, north) [arcsec] * 1e3 -> mpx
    np.testing.assert_allclose(spec["ps1_x_mpx"], 10.0 * -64.7, atol=1e-3)
    np.testing.assert_allclose(spec["ps1_y_mpx"], 10.0 * 159.9, atol=1e-3)
    cfg0 = _setup(tmp_path, monkeypatch, cells, template_shift_ps1_crpix=0)    # explicit 0 == auto-on-legacy
    spec0 = TS.compute(cfg0)
    for k in ("tx_mpx", "ty_mpx", "ps1_x_mpx", "ps1_y_mpx", "field_mean_mpx"):
        assert spec0[k] == spec[k]


def test_o05_pure_convention_gives_zero_ps1_term(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch, _real_cells(east=-125.0, north=125.0), template_shift_ps1_crpix=0.5)
    _master_in_mapping(cfg, o=0.5)
    spec = TS.compute(cfg)
    np.testing.assert_allclose(spec["ps1_x_mpx"], 0.0, atol=1e-6)
    np.testing.assert_allclose(spec["ps1_y_mpx"], 0.0, atol=1e-6)
    assert spec["ps1_crpix_offset"] == 0.5
    np.testing.assert_allclose(spec["convention_mean_mas"], [-125.0, 125.0], atol=1e-6)
    # a remainder on top of the convention survives
    cfg2 = _setup(tmp_path, monkeypatch, _real_cells(east=-125.0 + 10.0, north=125.0 - 20.0), template_shift_ps1_crpix=0.5)
    spec2 = TS.compute(cfg2)
    np.testing.assert_allclose(spec2["ps1_x_mpx"], 10.0 * 10.0, atol=1e-6)
    np.testing.assert_allclose(spec2["ps1_y_mpx"], 10.0 * -20.0, atol=1e-6)


def test_auto_reads_ps1crpix_from_master_csv(tmp_path, monkeypatch):
    cells = _real_cells(east=-125.0, north=125.0)
    cfg = _setup(tmp_path, monkeypatch, cells)                               # default auto
    _master_in_mapping(cfg, o=0.5)
    assert TS.resolve_crpix_offset(cfg)[0] == 0.5
    spec = TS.compute(cfg)
    assert spec["ps1_crpix_offset"] == 0.5
    np.testing.assert_allclose(spec["ps1_x_mpx"], 0.0, atol=1e-6)
    _master_in_mapping(cfg, o=None)                                          # legacy CSV -> 0
    assert TS.resolve_crpix_offset(cfg)[0] == 0.0
    assert "ps1_crpix_offset" not in TS.compute(cfg)


def test_auto_without_mapping_raises(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch, _real_cells())
    with pytest.raises(FileNotFoundError, match="master skycell CSV"):
        TS.compute(cfg)
    # a numeric override does not need the mapping (legacy/test use)
    cfg0 = _setup(tmp_path, monkeypatch, _real_cells(), template_shift_ps1_crpix=0.0)
    assert "tx_mpx" in TS.compute(cfg0)
