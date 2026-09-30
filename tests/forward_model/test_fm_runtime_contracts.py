# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
from __future__ import annotations

import pytest

from syndiff_pipeline.forward_model import run_fit


@pytest.mark.parametrize("flag", ["--group-frac", "--frame-frac"])
def test_minibatch_flags_fail_instead_of_silently_running_full_batch(flag):
    with pytest.raises(SystemExit, match="minibatching is not implemented"):
        run_fit.main(["--from-bundle", "/does/not/exist", flag, "0.5"])


def test_new_objective_cli_options_parse():
    args = run_fit.parse_args([
        "--from-bundle", "/does/not/exist",
        "--flux-objective", "huber-irls",
        "--huber-irls-iters", "3",
        "--support-size-weight-power", "0.5",
    ])
    assert args.flux_objective == "huber-irls"
    assert args.huber_irls_iters == 3
    assert args.support_size_weight_power == pytest.approx(0.5)

