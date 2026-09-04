"""Tests for cluster host-stats Condor integration."""

from __future__ import annotations

import subprocess
import unittest.mock

from syndiff_pipeline.common.orchestration import condor
from syndiff_pipeline.common.orchestration.host_stats import (
    HostSample,
    _parse_condor_status_output,
    apply_host_stats_policy,
    build_load15_rank,
    evaluate_host,
    expected_hosts,
    plan_host_selection,
    query_condor_host_samples,
)


def _sample(hostname: str, *, mem_available_mb: int, load15: float, mem_total_mb: int = 515_450) -> HostSample:
    return HostSample(
        hostname=hostname,
        mem_available_mb=mem_available_mb,
        mem_total_mb=mem_total_mb,
        load15=load15,
    )


class TestEvaluateHost:
    def test_excludes_low_mem_and_high_load15(self):
        sample = _sample("plscience1.stsci.edu", mem_available_mb=100_000, load15=10.5)
        reasons = evaluate_host(sample, min_mem_mb=128_000, max_load15=10.0)
        assert "low mem" in reasons[0]
        assert any("load15" in r for r in reasons)

    def test_missing_host(self):
        assert evaluate_host(None, min_mem_mb=128_000, max_load15=10.0) == ("missing",)


class TestPlanHostSelection:
    def test_ranks_eligible_by_load15(self):
        host_samples = {
            "plscience1.stsci.edu": _sample("plscience1.stsci.edu", mem_available_mb=400_000, load15=8.0),
            "plscience2.stsci.edu": _sample("plscience2.stsci.edu", mem_available_mb=400_000, load15=1.2),
            "plscience3.stsci.edu": _sample("plscience3.stsci.edu", mem_available_mb=50_000, load15=0.5),
        }

        selection = plan_host_selection(host_samples=host_samples, min_mem_mb=300_000, max_load15=10.0)
        assert selection.usable
        assert [s.hostname for s in selection.eligible] == [
            "plscience3.stsci.edu",
            "plscience2.stsci.edu",
            "plscience1.stsci.edu",
        ]
        # Low mem no longer hard-excludes -- it's filtered live via
        # MemAvailableMB in apply_host_stats_policy's requirements, not by
        # name here. Only hosts with no live data at all land in `excluded`.
        assert "plscience3.stsci.edu" not in selection.excluded
        assert set(selection.excluded) == set(expected_hosts()) - set(host_samples)

    def test_no_samples_not_usable(self):
        selection = plan_host_selection(host_samples={}, min_mem_mb=128_000, max_load15=10.0)
        assert not selection.usable
        assert selection.eligible == ()


class TestBuildLoad15Rank:
    def test_weighted_rank_expression(self):
        samples = [
            _sample("plscience2.stsci.edu", mem_available_mb=400_000, load15=1.0),
            _sample("plscience1.stsci.edu", mem_available_mb=400_000, load15=3.0),
        ]
        rank = build_load15_rank(samples)
        assert rank == (
            '(Machine == "plscience2.stsci.edu") * 2 + '
            '(Machine == "plscience1.stsci.edu") * 1'
        )


class TestApplyHostStatsPolicy:
    def test_fallback_when_no_samples(self):
        base = condor.CondorResourceRequest(
            request_memory_mb=300_000,
            host_stats_min_mem_mb=300_000,
            host_stats_max_load15=10.0,
        )
        out = apply_host_stats_policy(base, host_samples={})
        assert out.requirements == "Memory >= 300000"
        assert out.rank == "-LoadAvg"

    def test_low_mem_and_load_exclusions_are_live_not_frozen(self):
        """A host over the load threshold or under the mem floor (but present
        in condor_status) must not be hard-excluded by name -- that would
        bake a stale, un-reevaluated snapshot into the job's Requirements for
        its entire time in the queue, so it could never match that host
        again even after its memory/load actually recovers. Only the live
        MemAvailableMB/LoadAvg clauses should gate it, which Condor's
        negotiator re-checks against the machine's current state every
        cycle. Only a host with NO live data at all ("missing") still gets a
        hard, name-based exclusion -- there's no live substitute for that.
        """
        host_samples = {
            "plscience1.stsci.edu": _sample("plscience1.stsci.edu", mem_available_mb=50_000, load15=54.3),
            "plscience2.stsci.edu": _sample("plscience2.stsci.edu", mem_available_mb=400_000, load15=1.0),
        }
        base = condor.CondorResourceRequest(
            request_memory_mb=300_000,
            host_stats_min_mem_mb=300_000,
            host_stats_max_load15=10.0,
        )
        out = apply_host_stats_policy(base, host_samples=host_samples)
        assert out.requirements is not None
        assert 'Machine != "plscience1.stsci.edu"' not in out.requirements
        assert "MemAvailableMB >= 300000" in out.requirements
        assert "LoadAvg <= 10.0" in out.requirements
        # still present and still last in rank preference, just not blocked outright
        assert "plscience1.stsci.edu" in (out.rank or "")

    def test_applies_exclusions_and_rank(self):
        host_samples = {
            "plscience1.stsci.edu": _sample("plscience1.stsci.edu", mem_available_mb=400_000, load15=2.0),
            "plscience2.stsci.edu": _sample("plscience2.stsci.edu", mem_available_mb=400_000, load15=1.0),
        }

        base = condor.CondorResourceRequest(
            request_memory_mb=300_000,
            host_stats_min_mem_mb=300_000,
            host_stats_max_load15=10.0,
        )
        out = apply_host_stats_policy(base, host_samples=host_samples)
        assert out.requirements is not None
        assert "Memory >= 300000" in out.requirements
        # plscience3 (and every other expected host) has no live sample at
        # all -- that's still a hard, name-based exclusion.
        assert 'Machine != "plscience3.stsci.edu"' in out.requirements
        assert "MemAvailableMB >= 300000" in out.requirements
        assert "LoadAvg <= 10.0" in out.requirements
        assert out.rank is not None
        assert "plscience2.stsci.edu" in out.rank
        assert "* 2" in out.rank

    def test_bad_machines_merge_after_host_stats(self, tmp_path):
        import json

        host_samples = {
            "plscience1.stsci.edu": _sample("plscience1.stsci.edu", mem_available_mb=400_000, load15=1.0),
        }
        base = condor.CondorResourceRequest(
            request_memory_mb=128_000,
            host_stats_min_mem_mb=128_000,
        )
        out = apply_host_stats_policy(base, host_samples=host_samples)
        artifacts = {
            "bad_machines": tmp_path / "bad.json",
        }
        artifacts["bad_machines"].write_text(
            json.dumps({"hosts": ["plscience5.stsci.edu"]}),
            encoding="utf-8",
        )
        merged = condor.apply_bad_machine_exclusions(out, artifacts)
        assert 'Machine != "plscience5.stsci.edu"' in (merged.requirements or "")


class TestParseCondorStatusOutput:
    def test_parses_normal_output(self):
        stdout = (
            "plscience1.stsci.edu 515452 462637 1.64\n"
            "plscience2.stsci.edu 128380 123785 0.0\n"
        )
        samples = _parse_condor_status_output(stdout)
        assert set(samples) == {"plscience1.stsci.edu", "plscience2.stsci.edu"}
        assert samples["plscience1.stsci.edu"] == HostSample(
            hostname="plscience1.stsci.edu",
            mem_available_mb=462637,
            mem_total_mb=515452,
            load15=1.64,
        )

    def test_dedups_multiple_slots_per_machine(self):
        stdout = (
            "plscience5.stsci.edu 515452 462637 35.97\n"
            "plscience5.stsci.edu 515452 462637 35.97\n"
            "plscience5.stsci.edu 515452 462637 35.97\n"
        )
        samples = _parse_condor_status_output(stdout)
        assert list(samples) == ["plscience5.stsci.edu"]

    def test_skips_undefined_fields(self):
        stdout = "plscience1.stsci.edu 515452 undefined 1.0\n"
        samples = _parse_condor_status_output(stdout)
        assert samples == {}

    def test_skips_malformed_lines(self):
        stdout = "plscience1.stsci.edu 515452\nnot even close to a row\n"
        samples = _parse_condor_status_output(stdout)
        assert samples == {}


class TestQueryCondorHostSamples:
    def test_returns_empty_on_nonzero_exit(self):
        with unittest.mock.patch(
            "subprocess.run",
            return_value=unittest.mock.Mock(returncode=1, stdout="", stderr="not found"),
        ):
            assert query_condor_host_samples() == {}

    def test_returns_empty_on_timeout(self):
        with unittest.mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="condor_status", timeout=15.0),
        ):
            assert query_condor_host_samples() == {}

    def test_parses_successful_output(self):
        with unittest.mock.patch(
            "subprocess.run",
            return_value=unittest.mock.Mock(
                returncode=0,
                stdout="plscience1.stsci.edu 515452 462637 1.64\n",
                stderr="",
            ),
        ):
            samples = query_condor_host_samples()
        assert samples["plscience1.stsci.edu"].mem_available_mb == 462637
