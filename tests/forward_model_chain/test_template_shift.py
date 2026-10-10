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
