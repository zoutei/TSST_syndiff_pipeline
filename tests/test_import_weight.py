"""Import-time budget tests for lightweight CLI/daemon startup."""

from __future__ import annotations

import subprocess
import sys
import unittest


class TestSchedulerImportWeight(unittest.TestCase):
    def test_scheduler_cold_import_under_budget(self) -> None:
        """Daemon boot must not pull the full template stack at import time."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import time\n"
                    "t = time.perf_counter()\n"
                    "import syndiff_pipeline.template_creation.orchestration.scheduler\n"
                    "print(time.perf_counter() - t)\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        elapsed = float(result.stdout.strip())
        self.assertLess(
            elapsed,
            3.0,
            f"scheduler cold import took {elapsed:.2f}s (budget 3.0s)",
        )


if __name__ == "__main__":
    unittest.main()
