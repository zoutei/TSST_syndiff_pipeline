#!/usr/bin/env python3
"""Read cluster host sampler JSON files (placement check with VERDICT by default)."""

from __future__ import annotations

from syndiff_pipeline.common.orchestration.host_stats_cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_check=True))
