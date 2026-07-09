"""Tests for syndiff_pipeline.star.hosts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.star.hosts import StarHostRequest, load_star_hosts_file


class TestStarHosts(unittest.TestCase):
    def test_valid_csv_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.csv"
            path.write_text(
                "tic_id,gaia_source_id,label\n"
                "142748283,,\n"
                ",1234567890123456789,my_host_b\n",
                encoding="utf-8",
            )
            hosts = load_star_hosts_file(path)
            self.assertEqual(len(hosts), 2)
            self.assertEqual(hosts[0].tic_id, 142748283)
            self.assertIsNone(hosts[0].gaia_source_id)
            self.assertIsNone(hosts[0].label)
            self.assertIsNone(hosts[1].tic_id)
            self.assertEqual(hosts[1].gaia_source_id, 1234567890123456789)
            self.assertEqual(hosts[1].label, "my_host_b")

    def test_both_ids_raises_with_row_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.csv"
            path.write_text(
                "tic_id,gaia_source_id,label\n"
                "1,2,both\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Row 2.*both"):
                load_star_hosts_file(path)

    def test_neither_id_raises_with_row_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.csv"
            path.write_text(
                "tic_id,gaia_source_id,label\n"
                ",,empty\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Row 2.*neither"):
                load_star_hosts_file(path)

    def test_malformed_integer_raises_with_row_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stars.csv"
            path.write_text(
                "tic_id,gaia_source_id,label\n"
                "not_an_int,,\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Row 2.*tic_id"):
                load_star_hosts_file(path)

    def test_star_host_request_post_init(self):
        with self.assertRaises(ValueError):
            StarHostRequest(tic_id=1, gaia_source_id=2, label=None)
        with self.assertRaises(ValueError):
            StarHostRequest(tic_id=None, gaia_source_id=None, label=None)


if __name__ == "__main__":
    unittest.main()
