"""Hotpants reference stage: parameter set, Condor submission, frames; slow parity of the F=4 baseline is the e2e product
itself (a 25-minute Hotpants run) and is checked by re-reading its validation numbers only."""
import json
from pathlib import Path

import pytest

import b_fixtures as BF
from syndiff_pipeline.forward_model.chain import hotpants_ref as HR


def test_hotpants_params_are_valid_and_match_the_baseline_recipe():
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
    hp = HotpantsParams(**HR.HP_KWARGS)
    assert hp.hp_ko == 4 and hp.stamp_mode == "connected_regions" and hp.hp_bgo == 0 and list(hp.hp_sigma_gauss) == [0.752, 1.88, 3.76]
    assert hp.write_kernel_solutions and hp.write_convolved and hp.hp_force_convolve == "t"


def test_submit_writes_one_job_per_frame(tmp_path):
    cfg = BF.make_cfg(tmp_path)
    subs = HR.submit(cfg)
    assert [p.name for p in subs] == [f"hp_{BF.STEM}.sub", f"hp_{BF.UNSEEN}.sub"]
    t = subs[0].read_text()
    assert "request_cpus = 8" in t and "request_memory = 64000" in t and "chain.hotpants_ref" in t
    assert f"--stem {BF.STEM}" in t and "--config" in t and "queue 1" in t
    assert [p.name for p in HR.submit(cfg, [BF.STEM])] == [f"hp_{BF.STEM}.sub"]
    assert HR.frames_for(cfg) == [BF.STEM, BF.UNSEEN]


def test_run_requires_the_band_sum_template(tmp_path):
    with pytest.raises(FileNotFoundError, match="T_sum"):
        HR.run(BF.make_cfg(tmp_path))


@pytest.mark.slow
def test_e2e_baseline_product_is_consistent_with_its_template():
    """The e2e hp_d baseline validation: robust chi scale and the T_sum it was built from."""
    v = BF.E2E_RUN / "hotpants" / BF.STEM / "validation.json"
    BF.need(v, BF.E2E_RUN / "perband/out/band_templates/F1/a3/T_sum.npy")
    d = json.loads(v.read_text())
    assert d["noise_pos_frac_on_mask0"] == 1.0 and d["good_pixels"] > 100000
    assert Path(d["template"]).name == "T_sum.npy" and abs(d["robust_std_chi"] - 1.1568) < 1e-3
