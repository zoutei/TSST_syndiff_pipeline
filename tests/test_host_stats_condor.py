"""Tests for cluster host-stats Condor integration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from syndiff_pipeline.common.orchestration import condor
from syndiff_pipeline.common.orchestration.host_stats import (
    apply_host_stats_policy,
    build_load15_rank,
    evaluate_host,
    plan_host_selection,
)


def _write_sample(
    stats_dir: Path,
    hostname: str,
    *,
    mem_available_mb: int,
    load15: float,
    age_s: int = 0,
) -> None:
    payload = {
        "hostname": hostname,
        "login_hostname": hostname.replace("plscience", "science"),
        "timestamp": int(time.time()) - age_s,
        "mem_available_mb": mem_available_mb,
        "mem_total_mb": 515_450,
        "load1": load15,
        "load5": load15,
        "load15": load15,
    }
    path = stats_dir / f"{hostname}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestEvaluateHost:
    def test_excludes_low_mem_and_high_load15(self):
        from syndiff_pipeline.common.orchestration.host_stats import HostSample

        sample = HostSample(
            hostname="plscience1.stsci.edu",
            login_hostname="science1.stsci.edu",
            timestamp=int(time.time()),
            mem_available_mb=100_000,
            mem_total_mb=515_450,
            load1=12.0,
            load5=11.0,
            load15=10.5,
            path=Path("x.json"),
        )
        reasons = evaluate_host(
            sample, min_mem_mb=128_000, max_load15=10.0, max_age_s=300
        )
        assert "low mem" in reasons[0]
        assert any("load15" in r for r in reasons)

    def test_missing_host(self):
        assert evaluate_host(None, min_mem_mb=128_000, max_load15=10.0, max_age_s=300) == (
            "missing",
        )


class TestPlanHostSelection:
    def test_ranks_eligible_by_load15(self, tmp_path: Path):
        _write_sample(
            tmp_path,
            "plscience1.stsci.edu",
            mem_available_mb=400_000,
            load15=8.0,
        )
        _write_sample(
            tmp_path,
            "plscience2.stsci.edu",
            mem_available_mb=400_000,
            load15=1.2,
        )
        _write_sample(
            tmp_path,
            "plscience3.stsci.edu",
            mem_available_mb=50_000,
            load15=0.5,
        )

        selection = plan_host_selection(
            stats_dir=tmp_path,
            min_mem_mb=300_000,
            max_load15=10.0,
        )
        assert selection.usable
        assert [s.hostname for s in selection.eligible] == [
            "plscience2.stsci.edu",
            "plscience1.stsci.edu",
        ]
        assert "plscience3.stsci.edu" in selection.excluded

    def test_empty_dir_not_usable(self, tmp_path: Path):
        selection = plan_host_selection(
            stats_dir=tmp_path,
            min_mem_mb=128_000,
            max_load15=10.0,
        )
        assert not selection.usable
        assert selection.eligible == ()


class TestBuildLoad15Rank:
    def test_weighted_rank_expression(self):
        from syndiff_pipeline.common.orchestration.host_stats import HostSample

        samples = [
            HostSample(
                hostname="plscience2.stsci.edu",
                login_hostname=None,
                timestamp=0,
                mem_available_mb=400_000,
                mem_total_mb=515_450,
                load1=1.0,
                load5=1.0,
                load15=1.0,
                path=Path("a.json"),
            ),
            HostSample(
                hostname="plscience1.stsci.edu",
                login_hostname=None,
                timestamp=0,
                mem_available_mb=400_000,
                mem_total_mb=515_450,
                load1=3.0,
                load5=3.0,
                load15=3.0,
                path=Path("b.json"),
            ),
        ]
        rank = build_load15_rank(samples)
        assert rank == (
            '(Machine == "plscience2.stsci.edu") * 2 + '
            '(Machine == "plscience1.stsci.edu") * 1'
        )


class TestApplyHostStatsPolicy:
    def test_fallback_when_no_samples(self, tmp_path: Path):
        base = condor.CondorResourceRequest(
            request_memory_mb=300_000,
            host_stats_min_mem_mb=300_000,
            host_stats_max_load15=10.0,
        )
        out = apply_host_stats_policy(base, stats_dir=tmp_path)
        assert out.requirements == "Memory >= 300000"
        assert out.rank == "-LoadAvg"

    def test_applies_exclusions_and_rank(self, tmp_path: Path):
        _write_sample(
            tmp_path,
            "plscience1.stsci.edu",
            mem_available_mb=400_000,
            load15=2.0,
        )
        _write_sample(
            tmp_path,
            "plscience2.stsci.edu",
            mem_available_mb=400_000,
            load15=1.0,
        )

        base = condor.CondorResourceRequest(
            request_memory_mb=300_000,
            host_stats_min_mem_mb=300_000,
            host_stats_max_load15=10.0,
        )
        out = apply_host_stats_policy(base, stats_dir=tmp_path)
        assert out.requirements is not None
        assert "Memory >= 300000" in out.requirements
        assert 'Machine != "plscience3.stsci.edu"' in out.requirements
        assert out.rank is not None
        assert "plscience2.stsci.edu" in out.rank
        assert "* 2" in out.rank

    def test_bad_machines_merge_after_host_stats(self, tmp_path: Path):
        _write_sample(
            tmp_path,
            "plscience1.stsci.edu",
            mem_available_mb=400_000,
            load15=1.0,
        )
        base = condor.CondorResourceRequest(
            request_memory_mb=128_000,
            host_stats_min_mem_mb=128_000,
        )
        out = apply_host_stats_policy(base, stats_dir=tmp_path)
        artifacts = {
            "bad_machines": tmp_path / "bad.json",
        }
        artifacts["bad_machines"].write_text(
            json.dumps({"hosts": ["plscience5.stsci.edu"]}),
            encoding="utf-8",
        )
        merged = condor.apply_bad_machine_exclusions(out, artifacts)
        assert 'Machine != "plscience5.stsci.edu"' in (merged.requirements or "")
