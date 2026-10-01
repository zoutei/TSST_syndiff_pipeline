"""Score wrapper: command-line assembly for the frozen scorer."""
import pytest

import b_fixtures as BF
from syndiff_pipeline.forward_model.chain import score as SC


def _args(argv):
    d = {}
    for i, a in enumerate(argv):
        if a == "--diff":
            k, v = argv[i + 1].split("=", 1)
            d[k] = v
    return d


def test_score_command_weighted(tmp_path):
    cfg = BF.make_cfg(tmp_path, extra_inputs={"scorer_dir": str(tmp_path / "scorer"),
                                               "pass2_hp": {BF.STEM: "/x/pass2.fits.fz"}})
    argv, cwd, out = SC.score_command(cfg, BF.STEM, weighted=True)
    assert cwd == tmp_path / "scorer" and argv[1] == "score.py"
    assert out == tmp_path / f"out/score/{BF.STEM}/weighted"
    diffs = _args(argv)
    assert set(diffs) == {"hp", "band_w", "achrom_w", "pass2"}
    assert diffs["hp"] == str(tmp_path / f"out/hotpants/{BF.STEM}/hp_d/{BF.STEM}_hp_d.fits.fz")
    assert diffs["band_w"] == str(tmp_path / f"out/final/{BF.STEM}/band_w/hp_d/{BF.STEM}_hp_d.fits.fz")
    assert argv[argv.index("--candidate") + 1] == "band_w" and argv[argv.index("--baseline") + 1] == "hp"
    assert argv[argv.index("--field") + 1] == "T1" and argv[argv.index("--frame") + 1] == BF.STEM
    assert argv[argv.index("--ref") + 1] == diffs["hp"]


def test_score_command_unweighted_and_no_pass2_for_other_frame(tmp_path):
    cfg = BF.make_cfg(tmp_path, extra_inputs={"scorer_dir": str(tmp_path / "scorer"), "pass2_hp": {BF.STEM: "/x/p.fits.fz"}})
    argv, _, out = SC.score_command(cfg, BF.UNSEEN, weighted=False)
    assert set(_args(argv)) == {"hp", "band", "achrom"} and argv[argv.index("--candidate") + 1] == "band"
    assert out.name == "unweighted"


def test_score_requires_scorer_dir_and_final_done(tmp_path):
    with pytest.raises(KeyError, match="scorer_dir"):
        SC.score_command(BF.make_cfg(tmp_path), BF.STEM)
    with pytest.raises(FileNotFoundError, match="final"):
        SC.run(BF.make_cfg(tmp_path, extra_inputs={"scorer_dir": str(tmp_path)}))
