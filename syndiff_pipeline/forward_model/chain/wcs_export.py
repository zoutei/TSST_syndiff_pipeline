"""Export the fit's static WCS as a production ``TemporalChebWcsStore`` + gates A and B (e2e ``m01_wcs_export.py``).

The store (``wcs/<wcs_version>/``) is what ``mapping`` feeds to pancakes as ``tess_wcs_override``.

  gate A  round trip: the fitter's sky->pix (float64, ``star_basis`` + ``eval_all_positions``) vs the exported store's
          ``raw_for_stem().world_to_pixel_values`` on all scene stars (< 1e-5 px); the full-FFI adapter offset and the
          pixel->world->pixel inverse over the padded area.
  gate B  extrapolation: the new store vs the reference stores (``reference.old_store``, ``reference.tvwcs_store``) on
          a 32-px grid over the mapping's padded area: no blow-up, smooth. Fields whose reference is not configured are
          skipped; if ``old_store`` is missing the gate B pass flag is ``None`` (not evaluated).

Numerics are identical to the e2e script. jax is imported first (pyarrow-before-jax segfaults XLA) and x64 enabled.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import (SCIENCE_ORIGIN_FFI, SCIENCE_SHAPE, ChainConfig, is_done, mark_done, write_provenance)


def _jax():
    """(jax, jnp, cheb_wcs, fit_bundle) with x64 on; jax before anything that could pull pyarrow."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    (jnp.zeros(2) + 1).block_until_ready()   # initialise the XLA backend before pandas/pyarrow can be imported
    from syndiff_pipeline.forward_model import cheb_wcs as CW, fit_bundle as FB
    return jax, jnp, CW, FB


@dataclass
class WcsInputs:
    """Fit + scene the WCS comes from, and the fitter's static basis (the scene's source FitBundle)."""
    fit_dir: Path
    scene_dir: Path
    source_bundle: Path
    _bundle: object = None

    @classmethod
    def from_dirs(cls, fit_dir, scene_dir) -> "WcsInputs":
        meta = json.loads((Path(scene_dir) / "scene_meta.json").read_text())
        return cls(Path(fit_dir), Path(scene_dir), Path(meta["source_bundle"]))

    def bundle(self):
        """The scene's source FitBundle (``cheb_static`` = the fitter's TAN + Chebyshev basis definition)."""
        if self._bundle is None:
            *_, FB = _jax()
            self._bundle = FB.load_fit_bundle(self.source_bundle)
        return self._bundle


def wcs_coeff(fit_dir) -> "np.ndarray":
    """(42, 1) static Chebyshev coefficients, fitter (Sci2Idl) row order, float64."""
    import numpy as np
    c = np.asarray(np.load(Path(fit_dir) / "params.npz")["wcs_coeff"], np.float64)
    assert c.shape == (42, 1), c.shape
    return c


def fitter_sky_to_pix(inp: WcsInputs, ra, dec, coeff):
    """Fitter sky -> science-local pixel, float64 (``star_basis`` + ``eval_all_positions``, static frame)."""
    import numpy as np
    _, jnp, CW, _ = _jax()
    b = inp.bundle()
    st = b.cheb_static
    fb = np.asarray(b.wcs_frame_basis, np.float64)
    assert fb.shape == (1, 1) and fb[0, 0] == 1.0, fb
    xl, yl, basis = CW.star_basis(jnp.asarray(ra, jnp.float64), jnp.asarray(dec, jnp.float64), st)
    x, y = CW.eval_all_positions(xl, yl, basis, jnp.asarray(coeff), jnp.asarray(fb), st.n_terms)
    return np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)


def fitter_to_production_rows(inp: WcsInputs, coeff_col):
    """Permute fitter (Sci2Idl) term order to production ``_exponents`` order, x block then y block."""
    import numpy as np
    from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import _exponents
    st = inp.bundle().cheb_static
    fit_exp = [tuple(e) for e in st.exponents]
    prod_exp = list(_exponents(int(st.poly_degree)))
    n = len(fit_exp)
    assert sorted(fit_exp) == sorted(prod_exp) and 2 * n == coeff_col.size
    perm = [fit_exp.index(e) for e in prod_exp]
    return np.r_[coeff_col[:n][perm], coeff_col[n:][perm]]


def export_wcs(cfg: ChainConfig, fit_dir=None, scene_dir=None, out_dir=None, version: str | None = None) -> dict:
    """Write the store + gates under ``out_dir`` (default ``stage_dir('wcs')``). Returns the gates dict."""
    jax, jnp, CW, FB = _jax()
    import numpy as np
    import pandas as pd
    from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import (
        TemporalChebWcs, TemporalChebWcsStore, temporal_frame_contract)

    fit_dir = Path(fit_dir or cfg.stage_dir("fit"))
    scene_dir = Path(scene_dir or cfg.stage_dir("scene_boot"))
    OUT = Path(out_dir or cfg.stage_dir("wcs"))
    version = version or cfg.wcs_version
    R = OUT / version
    inp = WcsInputs.from_dirs(fit_dir, scene_dir)
    ref = cfg.reference

    b = inp.bundle()
    st = b.cheb_static
    coeff = wcs_coeff(fit_dir)
    tv = TemporalChebWcsStore(ref.tvwcs_store) if ref.tvwcs_store else None
    if tv is not None:
        tv_model, btjd = tv.raw_for_stem(cfg.stem)
    else:  # no temporal store: the frame time recorded in the scene
        tv_model, btjd = None, float(json.loads((scene_dir / "scene_meta.json").read_text())["frame_btjd"])
    deg = 3
    knots = np.r_[np.zeros(deg + 1), np.ones(deg + 1)]
    nb = len(knots) - deg - 1
    col = fitter_to_production_rows(inp, coeff[:, 0])
    model = TemporalChebWcs(float(st.ra0_deg), float(st.dec0_deg), np.asarray(st.cd_inv, float),
                            np.asarray(st.crpix, float), np.asarray(st.center, float),
                            np.asarray(st.half_extents, float), int(st.poly_degree), knots, deg,
                            float(btjd) - 0.5, 1.0, np.repeat(col[:, None], nb, axis=1))
    (R / "models").mkdir(parents=True, exist_ok=True)
    model.save(R / "models/orbit_00.npz")
    fr = pd.DataFrame({"stem": [cfg.stem], "btjd": [float(btjd)], "frame_index": np.array([0], np.int32),
                       "n_stars_qc": [0], "fit_provenance": ["scene_fit_static"], "median_residual": [np.nan],
                       "orbit_index": [0], "fit_frame_index": [0], "runtime_source": ["scene_fit"],
                       "ffi_wcs_ok": [None]})
    fr.to_parquet(R / "frames.parquet", index=False)
    contract = temporal_frame_contract(origin_ffi=SCIENCE_ORIGIN_FFI, shape=SCIENCE_SHAPE)
    mfp = hashlib.sha256((R / "models/orbit_00.npz").read_bytes()).hexdigest()
    manifest = {
        "camera": cfg.scc.camera, "ccd": cfg.scc.ccd, "sector": cfg.scc.sector,
        "coordinate_direction": "gaia_tan_to_detector_pixel",
        "domain": {"x_min": 0, "x_max": 2048, "y_min": 0, "y_max": 2048},
        "frame_contract": contract, "model_kind": "temporal_wcs",
        "models": [{"orbit_index": 0, "path": "models/orbit_00.npz", "start": 0, "end": 1,
                    "fit_start": 0, "fit_end": 1, "fingerprint": mfp}],
        "n_fit": 1, "n_frames": 1, "pixel_origin": 0, "spatial_basis": "chebyshev", "spatial_degree": 5,
        "temporal_basis": "bspline", "temporal_spline_degree": deg, "version": version,
        "source": {"scene_fit": str(fit_dir), "source_bundle": str(inp.source_bundle),
                   "code_sha": cfg.code_sha(),
                   "note": "static single-FFI WCS of the configured scene fit; time basis is constant"},
    }
    (R / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # ---------------- gate A
    store = TemporalChebWcsStore(R)
    adapter, bt = store.for_stem(cfg.stem)
    raw, _ = store.raw_for_stem(cfg.stem)
    z = np.load(scene_dir / "scene_bundle.npz")
    idx = z["star_bundle_index"]
    ra, dec = np.asarray(b.ra, float)[idx], np.asarray(b.dec, float)[idx]
    xf, yf = fitter_sky_to_pix(inp, ra, dec, coeff)
    xp, yp = raw.world_to_pixel_values(ra, dec, bt)
    xa, ya = adapter.world_to_pixel_values(ra, dec)
    # the fitter's scene hot path uses the bundle's BAKED x_lin / cheb_basis (cast to float32 in
    # build_static_context); report how far that is from the float64 equations (information only)
    xh = np.asarray(b.x_lin, np.float32)[idx].astype(float) + np.asarray(b.cheb_basis, np.float32)[idx].astype(float) @ coeff[:21, 0]
    yh = np.asarray(b.y_lin, np.float32)[idx].astype(float) + np.asarray(b.cheb_basis, np.float32)[idx].astype(float) @ coeff[21:, 0]
    gateA = {"n": int(ra.size), "max_dx": float(np.abs(xp - xf).max()), "max_dy": float(np.abs(yp - yf).max()),
             "adapter_minus_raw_x": [float((xa - xp).min()), float((xa - xp).max())],
             "adapter_minus_raw_y": [float((ya - yp).min()), float((ya - yp).max())],
             "info_baked_float32_hotpath_minus_float64_max_px": float(max(np.abs(xh - xf).max(), np.abs(yh - yf).max())),
             "info_baked_float32_hotpath_minus_float64_median_px": float(np.median(np.hypot(xh - xf, yh - yf)))}
    gx, gy = np.meshgrid(np.linspace(-512, 2559, 41), np.linspace(-512, 2559, 41))
    r2, d2 = raw.pixel_to_world(gx.ravel(), gy.ravel(), bt)
    x3, y3 = raw.world_to_pixel_values(r2, d2, bt)
    gateA["inverse_max_px_pad512"] = float(max(np.abs(x3 - gx.ravel()).max(), np.abs(y3 - gy.ravel()).max()))
    # production coefficients vs the fit (permuted) and vs the reference store: what actually changed
    new_c = np.load(R / "models/orbit_00.npz")["coeff_matrix"]
    gateA["store_coeff_minus_A3_absmax"] = float(np.abs(new_c - col[:, None]).max())
    if ref.old_store:
        old_c = np.load(ref.old_store / "models/orbit_00.npz")["coeff_matrix"]
        gateA["store_coeff_minus_C5store_absmax"] = float(np.abs(new_c - old_c).max())
    gateA["pass"] = bool(max(gateA["max_dx"], gateA["max_dy"]) < 1e-5 and gateA["inverse_max_px_pad512"] < 1e-5)

    # ---------------- gate B
    old_raw = TemporalChebWcsStore(ref.old_store).raw_for_stem(cfg.stem)[0] if ref.old_store else None

    def delta(src, dst, xs, ys):
        rr, dd = src.pixel_to_world(xs, ys, bt)
        xn, yn = dst.world_to_pixel_values(rr, dd, bt)
        return xn - xs, yn - ys

    g = np.unique(np.r_[np.arange(-520, 2048 + 521, 32, dtype=float), -8.0, 2055.0])
    GX, GY = np.meshgrid(g, g)
    regions = {
        "science": (GX >= 0) & (GX <= 2047) & (GY >= 0) & (GY <= 2047),
        "template": (GX >= -8) & (GX <= 2055) & (GY >= -8) & (GY <= 2055),       # mapped 8256^2 OS4 array
        "margin64": (GX >= -72) & (GX <= 2119) & (GY >= -72) & (GY <= 2119),
        "wide512": np.ones_like(GX, bool),                                         # information only
    }
    fields = {}
    pairs = {}
    if old_raw is not None:
        pairs["A3_minus_C5"] = (old_raw, raw)
    if tv_model is not None and tv is not None:
        pairs["A3_minus_tvwcs"] = (tv_model, raw)
        if old_raw is not None:
            pairs["C5_minus_tvwcs"] = (tv_model, old_raw)
    for name, (s, d) in pairs.items():
        DX, DY = delta(s, d, GX.ravel(), GY.ravel())
        fields[name] = (DX.reshape(GX.shape), DY.reshape(GX.shape))
    gateB = {"grid": "32 px (+ the template edges -8, 2055), science-local -520..2568"}
    gu = (np.abs(g - np.round((g + 520) / 32) * 32 + 520) < 1e-9)                # the regular 32-px nodes
    for name, (DX, DY) in fields.items():
        mag = np.hypot(DX, DY)
        e = {}
        for rn, m in regions.items():
            e[f"{rn}_median_mpx"] = float(np.median(mag[m]) * 1e3)
            e[f"{rn}_max_mpx"] = float(mag[m].max() * 1e3)
            # smoothness: max |second difference| of the offset on the regular 32-px nodes within the region
            mu = m[np.ix_(gu, gu)]
            dxu, dyu = DX[np.ix_(gu, gu)], DY[np.ix_(gu, gu)]
            d2_ = max(np.abs(np.diff(dxu, 2, axis=1)[mu[:, 1:-1]]).max(), np.abs(np.diff(dyu, 2, axis=0)[mu[1:-1, :]]).max(),
                      np.abs(np.diff(dxu, 2, axis=0)[mu[1:-1, :]]).max(), np.abs(np.diff(dyu, 2, axis=1)[mu[:, 1:-1]]).max())
            e[f"{rn}_max_2nd_diff_mpx"] = float(d2_ * 1e3)
        gateB[name] = e
    if "A3_minus_C5" in gateB:
        ab = gateB["A3_minus_C5"]
        gateB["pass"] = bool(ab["template_max_mpx"] < 2.0 * ab["science_max_mpx"]
                             and ab["margin64_max_mpx"] < 3.0 * ab["science_max_mpx"]
                             and ab["template_max_2nd_diff_mpx"] < 2.0 * max(ab["science_max_2nd_diff_mpx"], 0.1))
    else:
        gateB["pass"] = None
        gateB["note"] = "reference.old_store not configured: A3-C5 comparison skipped, pass not evaluated"
    gateB["pass_rule"] = ("no blow-up: |A3-C5| max over the mapped template area (-8..2055) < 2x its science-area max, "
                          "over a 64-px margin < 3x, and the 32-px second difference over the template area < 2x the "
                          "science-area value. Beyond ~200 px both degree-5 fits diverge polynomially (wide512, info only; "
                          "outside the mapped array).")
    np.savez(OUT / "gateB_fields.npz", grid=g, **{f"{k}_dx": v[0] for k, v in fields.items()},
             **{f"{k}_dy": v[1] for k, v in fields.items()})
    if fields:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5.6), squeeze=False)
        for a, name in zip(ax[0], fields):
            DX, DY = fields[name]
            im = a.pcolormesh(g, g, np.hypot(DX, DY) * 1e3, cmap="viridis", shading="nearest",
                              vmax=np.percentile(np.hypot(DX, DY)[regions["margin64"]] * 1e3, 100))
            a.set_aspect("equal")
            a.plot([0, 2048, 2048, 0, 0], [0, 0, 2048, 2048, 0], "w-", lw=1)
            a.plot([-8, 2056, 2056, -8, -8], [-8, -8, 2056, 2056, -8], "r:", lw=1)
            plt.colorbar(im, ax=a, label="|offset| [mpx]")
            a.set_title(name.replace("_", " "))
            a.set_xlabel("science-local x [px]")
            a.set_ylabel("y [px]")
        fig.tight_layout()
        fig.savefig(OUT / "gateB_extrapolation.png", dpi=100)
        plt.close(fig)
    res = {"store": str(R), "btjd": bt, "gateA": gateA, "gateB": gateB}
    (OUT / "wcs_export_gates.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def run_wcs(cfg: ChainConfig, force: bool = False) -> Path:
    """Stage ``wcs``: needs ``fit`` and ``scene_boot`` done. Raises if a gate fails."""
    out = cfg.stage_dir("wcs")
    if is_done(out) and not force:
        print(f"[wcs] already done: {out}")
        return out
    for dep in ("fit", "scene_boot"):
        if not is_done(cfg.stage_dir(dep)):
            raise FileNotFoundError(f"stage {dep} not done: {cfg.stage_dir(dep)}")
    (out / "DONE").unlink(missing_ok=True)
    res = export_wcs(cfg)
    write_provenance(out, cfg, {"fit_params": cfg.stage_dir("fit") / "params.npz",
                                "scene_meta": cfg.stage_dir("scene_boot") / "scene_meta.json",
                                **{k: v for k, v in (("old_store_manifest", cfg.reference.old_store and cfg.reference.old_store / "manifest.json"),
                                                     ("tvwcs_manifest", cfg.reference.tvwcs_store and cfg.reference.tvwcs_store / "manifest.json")) if v}})
    print(json.dumps({"gateA_pass": res["gateA"]["pass"], "gateB_pass": res["gateB"]["pass"]}))
    if not res["gateA"]["pass"] or res["gateB"]["pass"] is False:
        raise RuntimeError(f"WCS gates failed; see {out / 'wcs_export_gates.json'}")
    mark_done(out)
    return out
