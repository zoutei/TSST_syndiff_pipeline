"""Tests for syndiff cluster / host_stats_cli."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from syndiff_pipeline.common.orchestration.host_stats_cli import (
    main,
    render_cluster_table_text,
)
from syndiff_pipeline.template_creation.orchestration.discord_bot import (
    cluster_status_trigger,
    run_cluster_status_command,
)


def _write_sample(
    stats_dir: Path,
    hostname: str,
    *,
    mem_available_mb: int,
    mem_total_mb: int,
    load15: float,
) -> None:
    payload = {
        "hostname": hostname,
        "login_hostname": hostname.replace("plscience", "science"),
        "timestamp": int(time.time()),
        "mem_available_mb": mem_available_mb,
        "mem_total_mb": mem_total_mb,
        "load1": load15,
        "load5": load15,
        "load15": load15,
    }
    (stats_dir / f"{hostname}.json").write_text(json.dumps(payload), encoding="utf-8")


class TestClusterTable:
    def test_compact_table_has_no_verdict_column(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOST_STATS_DIR", str(tmp_path))
        _write_sample(
            tmp_path,
            "plscience1.stsci.edu",
            mem_available_mb=400_000,
            mem_total_mb=515_000,
            load15=1.0,
        )
        text = render_cluster_table_text(include_verdict=False)
        assert "VERDICT" not in text
        assert "515GB" in text
        assert "plscience1.stsci.edu" in text

    def test_columns_align_with_wide_avail_values(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOST_STATS_DIR", str(tmp_path))
        _write_sample(
            tmp_path,
            "plscience4.stsci.edu",
            mem_available_mb=361_700,
            mem_total_mb=515_000,
            load15=37.31,
        )
        _write_sample(
            tmp_path,
            "plscience5.stsci.edu",
            mem_available_mb=21_900,
            mem_total_mb=128_000,
            load15=4.18,
        )
        lines = render_cluster_table_text(include_verdict=False).splitlines()
        header = lines[0]
        load15_start = header.index("LOAD15")
        load15_width = len("LOAD15")
        seen = set()
        for line in lines[2:]:
            if "plscience4.stsci.edu" not in line and "plscience5.stsci.edu" not in line:
                continue
            seen.add(line[load15_start : load15_start + load15_width].strip())
        assert seen == {"37.31", "4.18"}

    def test_check_table_includes_verdict(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOST_STATS_DIR", str(tmp_path))
        _write_sample(
            tmp_path,
            "plscience1.stsci.edu",
            mem_available_mb=50_000,
            mem_total_mb=128_000,
            load15=1.0,
        )
        text = render_cluster_table_text(
            include_verdict=True,
            min_mem_mb=128_000,
            max_load15=10.0,
        )
        assert "VERDICT" in text
        assert "EXCLUDE" in text


class TestSyndiffClusterMain:
    def test_main_default_no_verdict(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.setenv("HOST_STATS_DIR", str(tmp_path))
        _write_sample(
            tmp_path,
            "plscience2.stsci.edu",
            mem_available_mb=200_000,
            mem_total_mb=515_000,
            load15=2.0,
        )
        assert main(["--no-stats-dir-line"], default_check=False) == 0
        out = capsys.readouterr().out
        assert "VERDICT" not in out
        assert "Excluded:" not in out

    def test_main_check_shows_verdict(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.setenv("HOST_STATS_DIR", str(tmp_path))
        _write_sample(
            tmp_path,
            "plscience2.stsci.edu",
            mem_available_mb=50_000,
            mem_total_mb=128_000,
            load15=2.0,
        )
        assert main(["--check", "--no-stats-dir-line", "--preset", "128gb"]) == 0
        out = capsys.readouterr().out
        assert "VERDICT" in out
        assert "Excluded:" in out


class TestIncludeOkFormats:
    """--include-ok must only affect --format hosts (a genuine "list everyone"
    view); requirements/bad-machines are exclusion structures by definition,
    so silently turning them into "every host" would be a real footgun given
    both formats are documented as scripting-consumable and bad-machines'
    schema matches condor.py's real exclusion-file format."""

    def _setup(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOST_STATS_DIR", str(tmp_path))
        _write_sample(
            tmp_path,
            "plscience1.stsci.edu",
            mem_available_mb=200_000,
            mem_total_mb=515_000,
            load15=1.0,
        )
        _write_sample(
            tmp_path,
            "plscience2.stsci.edu",
            mem_available_mb=10_000,
            mem_total_mb=128_000,
            load15=1.0,
        )

    def test_include_ok_lists_every_host_for_hosts_format(self, tmp_path, monkeypatch, capsys):
        self._setup(tmp_path, monkeypatch)
        assert (
            main(
                [
                    "--format",
                    "hosts",
                    "--include-ok",
                    "--preset",
                    "128gb",
                    "--no-stats-dir-line",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        for n in range(1, 16):
            assert f"plscience{n}.stsci.edu" in out

    def test_include_ok_is_ignored_with_warning_for_bad_machines(
        self, tmp_path, monkeypatch, capsys
    ):
        self._setup(tmp_path, monkeypatch)
        assert (
            main(
                [
                    "--format",
                    "bad-machines",
                    "--include-ok",
                    "--preset",
                    "128gb",
                    "--no-stats-dir-line",
                ]
            )
            == 0
        )
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "--include-ok has no effect" in captured.err
        hosts = json.loads(captured.out)["hosts"]
        # plscience1 has plenty of free mem -> not excluded; must NOT appear.
        assert "plscience1.stsci.edu" not in hosts
        assert "plscience2.stsci.edu" in hosts

    def test_include_ok_is_ignored_with_warning_for_requirements(
        self, tmp_path, monkeypatch, capsys
    ):
        self._setup(tmp_path, monkeypatch)
        assert (
            main(
                [
                    "--format",
                    "requirements",
                    "--include-ok",
                    "--preset",
                    "128gb",
                    "--no-stats-dir-line",
                ]
            )
            == 0
        )
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "plscience1.stsci.edu" not in captured.out
        assert "plscience2.stsci.edu" in captured.out


class TestDiscordClusterTrigger:
    def test_cluster_status_trigger_substring(self):
        assert cluster_status_trigger("how is the cluster?")
        assert cluster_status_trigger("syndiff cluster")
        assert not cluster_status_trigger("condor_q")

    def test_run_cluster_status_command(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOST_STATS_DIR", str(tmp_path))
        _write_sample(
            tmp_path,
            "plscience3.stsci.edu",
            mem_available_mb=300_000,
            mem_total_mb=515_000,
            load15=1.5,
        )
        messages = run_cluster_status_command()
        assert len(messages) >= 1
        assert "**syndiff cluster**" in messages[0]
        assert "plscience3.stsci.edu" in messages[0]
