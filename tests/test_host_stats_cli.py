"""Tests for syndiff cluster / host_stats_cli."""

from __future__ import annotations

import json
import unittest.mock

import pytest

from syndiff_pipeline.common.orchestration.host_stats import HostSample
from syndiff_pipeline.common.orchestration.host_stats_cli import (
    main,
    render_cluster_table_text,
)
from syndiff_pipeline.template_creation.orchestration.discord_bot import (
    cluster_status_trigger,
    run_cluster_status_command,
)


def _sample(hostname: str, *, mem_available_mb: int, mem_total_mb: int, load15: float) -> HostSample:
    return HostSample(
        hostname=hostname,
        mem_available_mb=mem_available_mb,
        mem_total_mb=mem_total_mb,
        load15=load15,
    )


def _patch_query(monkeypatch, samples: dict[str, HostSample]):
    monkeypatch.setattr(
        "syndiff_pipeline.common.orchestration.host_stats_cli.query_condor_host_samples",
        lambda: samples,
    )


class TestClusterTable:
    def test_compact_table_has_no_verdict_column(self):
        host_samples = {
            "plscience1.stsci.edu": _sample(
                "plscience1.stsci.edu", mem_available_mb=400_000, mem_total_mb=515_000, load15=1.0
            ),
        }
        text = render_cluster_table_text(host_samples=host_samples, include_verdict=False)
        assert "VERDICT" not in text
        assert "AGE" not in text
        assert "515GB" in text
        assert "plscience1.stsci.edu" in text

    def test_columns_align_with_wide_avail_values(self):
        host_samples = {
            "plscience4.stsci.edu": _sample(
                "plscience4.stsci.edu", mem_available_mb=361_700, mem_total_mb=515_000, load15=37.31
            ),
            "plscience5.stsci.edu": _sample(
                "plscience5.stsci.edu", mem_available_mb=21_900, mem_total_mb=128_000, load15=4.18
            ),
        }
        lines = render_cluster_table_text(host_samples=host_samples, include_verdict=False).splitlines()
        header = lines[0]
        load15_start = header.index("LOAD15")
        load15_width = len("LOAD15")
        seen = set()
        for line in lines[2:]:
            if "plscience4.stsci.edu" not in line and "plscience5.stsci.edu" not in line:
                continue
            seen.add(line[load15_start : load15_start + load15_width].strip())
        assert seen == {"37.31", "4.18"}

    def test_check_table_includes_verdict(self):
        host_samples = {
            "plscience1.stsci.edu": _sample(
                "plscience1.stsci.edu", mem_available_mb=50_000, mem_total_mb=128_000, load15=1.0
            ),
        }
        text = render_cluster_table_text(
            host_samples=host_samples,
            include_verdict=True,
            min_mem_mb=128_000,
            max_load15=10.0,
        )
        assert "VERDICT" in text
        assert "EXCLUDE" in text


class TestSyndiffClusterMain:
    def test_main_default_no_verdict(self, monkeypatch, capsys):
        _patch_query(
            monkeypatch,
            {
                "plscience2.stsci.edu": _sample(
                    "plscience2.stsci.edu", mem_available_mb=200_000, mem_total_mb=515_000, load15=2.0
                ),
            },
        )
        assert main([], default_check=False) == 0
        out = capsys.readouterr().out
        assert "VERDICT" not in out
        assert "Excluded:" not in out

    def test_main_check_shows_verdict(self, monkeypatch, capsys):
        _patch_query(
            monkeypatch,
            {
                "plscience2.stsci.edu": _sample(
                    "plscience2.stsci.edu", mem_available_mb=50_000, mem_total_mb=128_000, load15=2.0
                ),
            },
        )
        assert main(["--check", "--preset", "128gb"]) == 0
        out = capsys.readouterr().out
        assert "VERDICT" in out
        assert "Excluded:" in out
        assert "live clauses" in out


class TestIncludeOkFormats:
    """--include-ok must only affect --format hosts (a genuine "list everyone"
    view); requirements/bad-machines are exclusion structures by definition,
    so silently turning them into "every host" would be a real footgun given
    both formats are documented as scripting-consumable and bad-machines'
    schema matches condor.py's real exclusion-file format.

    Only genuinely-missing hosts (no live condor_status data at all) show up
    in these exclusion formats now -- a low-mem/high-load host that's still
    present in condor_status is filtered live via MemAvailableMB/LoadAvg
    Requirements clauses instead, not by name. So `_setup` supplies a sample
    for plscience1 only; every other expected host (plscience2..15) is
    "missing" and lands in the exclusion formats.
    """

    def _setup(self, monkeypatch):
        _patch_query(
            monkeypatch,
            {
                "plscience1.stsci.edu": _sample(
                    "plscience1.stsci.edu", mem_available_mb=200_000, mem_total_mb=515_000, load15=1.0
                ),
            },
        )

    def test_include_ok_lists_every_host_for_hosts_format(self, monkeypatch, capsys):
        self._setup(monkeypatch)
        assert main(["--format", "hosts", "--include-ok", "--preset", "128gb"]) == 0
        out = capsys.readouterr().out
        for n in range(1, 16):
            assert f"plscience{n}.stsci.edu" in out

    def test_include_ok_is_ignored_with_warning_for_bad_machines(self, monkeypatch, capsys):
        self._setup(monkeypatch)
        assert main(["--format", "bad-machines", "--include-ok", "--preset", "128gb"]) == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "--include-ok has no effect" in captured.err
        hosts = json.loads(captured.out)["hosts"]
        # plscience1 has live data -> not excluded; must NOT appear.
        assert "plscience1.stsci.edu" not in hosts
        # plscience2 has no live data at all -> hard, name-based exclusion.
        assert "plscience2.stsci.edu" in hosts

    def test_include_ok_is_ignored_with_warning_for_requirements(self, monkeypatch, capsys):
        self._setup(monkeypatch)
        assert main(["--format", "requirements", "--include-ok", "--preset", "128gb"]) == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "plscience1.stsci.edu" not in captured.out
        assert "plscience2.stsci.edu" in captured.out


class TestDiscordClusterTrigger:
    def test_cluster_status_trigger_substring(self):
        assert cluster_status_trigger("how is the cluster?")
        assert cluster_status_trigger("syndiff cluster")
        assert not cluster_status_trigger("condor_q")

    def test_run_cluster_status_command(self, monkeypatch):
        _patch_query(
            monkeypatch,
            {
                "plscience3.stsci.edu": _sample(
                    "plscience3.stsci.edu", mem_available_mb=300_000, mem_total_mb=515_000, load15=1.5
                ),
            },
        )
        messages = run_cluster_status_command()
        assert len(messages) >= 1
        assert "**syndiff cluster**" in messages[0]
        assert "plscience3.stsci.edu" in messages[0]
