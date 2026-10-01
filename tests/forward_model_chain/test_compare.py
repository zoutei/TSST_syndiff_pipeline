import json

import numpy as np
import pytest

from syndiff_pipeline.forward_model.chain import compare as K
from syndiff_pipeline.forward_model.chain import config as C

from chain_fixtures import make_scene, raw_config

NT, G = 21, 33        # 21 Chebyshev terms (degree 5), 33x33 ePSF grid at OS4


def _setup(tmp_path):
    scene = tmp_path / "scene"
    z = make_scene(scene)
    n = len(z["cx"])
    rng = np.random.default_rng(1)
    ny, nx = 3, 4
    np.savez(tmp_path / "bundle.npz", cheb_basis=rng.normal(size=(n, NT)), epsf_node_col_ccd=np.linspace(0, 2200, nx),
             epsf_node_row_ccd=np.linspace(0, 2048, ny))
    meta = json.loads((scene / "scene_meta.json").read_text())
    meta["source_bundle"] = str(tmp_path / "bundle.npz")
    (scene / "scene_meta.json").write_text(json.dumps(meta))
    u = (np.arange(G) - G // 2) / 4.0
    X, Y = np.meshgrid(u, u)
    epsf = np.broadcast_to(np.exp(-(X ** 2 + Y ** 2) / (2 * 1.2 ** 2)), (ny, nx, G, G)).copy()
    return scene, n, epsf


def _fit(d, n, epsf, wcs, flux, loss=1.0, bw=None):
    d.mkdir(parents=True)
    p = dict(wcs_coeff=wcs[:, None], epsf_base=epsf, chroma_g8=np.zeros(8))
    if bw is not None:
        p["bright_width"] = np.array([bw])
    np.savez(d / "params.npz", **p)
    np.savez(d / "flux_solved.npz", flux=flux, bg_coef=np.array([0.5]))
    (d / "history.jsonl").write_text(json.dumps({"loss": loss, "data_term": loss - 0.1}) + "\n")
    return d


def test_compare_identical_is_zero_and_known_shift(tmp_path):
    scene, n, epsf = _setup(tmp_path)
    flux = np.linspace(1e3, 1e5, n)
    w = np.zeros(2 * NT)
    a = _fit(tmp_path / "a", n, epsf, w, flux, loss=1.0)
    w2 = w.copy()
    w2[0] = 0.01                                  # shifts x of every star by 0.01 * cheb_basis[:,0]
    b = _fit(tmp_path / "b", n, epsf, w2, flux * 1.001, loss=1.25, bw=0.4)
    c = _fit(tmp_path / "c", n, epsf, w, flux, loss=1.0, bw=0.4)
    out = tmp_path / "out"
    res = K.compare_fits({"a": a, "b": b, "c": c, "missing": tmp_path / "nope"},
                         [("c", "a", "bw only"), ("b", "a", "shift"), ("b", "missing", "skipped")], scene, out)
    assert set(res) == {"c - a", "b - a"}
    z = res["c - a"]
    assert z["wcs_rms_mpx"] == 0 and z["dT_1e3px2_median"] == 0 and z["dloss"] == 0
    assert z["b_bright"] == [pytest.approx(0.4e-3), None]
    # q=0 effective size: b(0 - q_ref=0) = 0 -> unchanged; at q=1e5 isotropic 2*b*1e5/1e4
    assert z["dTeff_q1e5_1e3px2"] == pytest.approx(1e3 * 2 * 0.4e-3 * 1e5 / 1e4)
    s = res["b - a"]
    B = np.load(tmp_path / "bundle.npz")["cheb_basis"]
    assert s["wcs_rms_mpx"] == pytest.approx(1e3 * 0.01 * np.sqrt(np.mean(B[:, 0] ** 2)))
    assert s["dloss"] == pytest.approx(0.25)
    assert s["flux_ratio_by_T"]["7-9"]["median_ppt"] == pytest.approx(1.0, rel=1e-6)
    for f in ("compare_fits.json", "compare_fits.md", "fig_compare_fits.png", "fig_compare_epsf.png"):
        assert (out / f).stat().st_size > 0
    assert json.loads((out / "compare_fits.json").read_text())["b - a"]["label"] == "shift"
    assert "| pair |" in (out / "compare_fits.md").read_text()


def test_run_compare_stage(tmp_path):
    scene, n, epsf = _setup(tmp_path)
    cfg = C.config_from_dict(raw_config(tmp_path))
    flux = np.linspace(1e3, 1e5, n)
    w = np.zeros(2 * NT)
    _fit(cfg.stage_dir("fit"), n, epsf, w, flux)
    _fit(cfg.stage_dir("refit"), n, epsf, w, flux)
    import shutil
    shutil.copytree(scene, cfg.stage_dir("scene_boot"))
    for s in ("fit", "refit"):
        C.mark_done(cfg.stage_dir(s))
    out = K.run_compare(cfg)
    assert C.is_done(out) and (out / "provenance.json").exists()
    assert list(json.loads((out / "compare_fits.json").read_text())) == ["refit - boot"]


def test_run_compare_requires_fits(tmp_path):
    cfg = C.config_from_dict(raw_config(tmp_path))
    with pytest.raises(FileNotFoundError):
        K.run_compare(cfg)
