"""Tests for syndiff_pipeline.star.identifiers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from syndiff_pipeline.star.hosts import StarHostRequest
from syndiff_pipeline.star.identifiers import (
    ResolvedHost,
    resolve_host,
    write_host_gaia_row_csv,
    write_identifier_json,
)


def _write_gaia_catalog(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


class TestStarIdentifiers(unittest.TestCase):
    def test_rejects_short_gaia_source_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "gaia.csv"
            _write_gaia_catalog(
                catalog,
                [
                    {
                        "source_id": 1234567890123456789,
                        "ra": 100.0,
                        "dec": 20.0,
                        "ra_error": 0.01,
                        "dec_error": 0.01,
                        "parallax": 1.0,
                        "parallax_error": 0.1,
                        "phot_g_mean_mag": 12.0,
                        "phot_bp_mean_mag": 13.0,
                        "phot_rp_mean_mag": 11.0,
                    }
                ],
            )
            request = StarHostRequest(
                tic_id=None, gaia_source_id=142748283, label=None
            )
            with self.assertRaisesRegex(
                ValueError, "at least 15 digits.*TIC id"
            ):
                resolve_host(
                    request,
                    gaia_catalog_path=str(catalog),
                    allow_remote=False,
                )

    def test_local_gaia_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "gaia.csv"
            _write_gaia_catalog(
                catalog,
                [
                    {
                        "source_id": 1060421588522505216,
                        "ra": 210.5,
                        "dec": 81.8,
                        "ra_error": 0.01,
                        "dec_error": 0.01,
                        "parallax": 1.0,
                        "parallax_error": 0.1,
                        "phot_g_mean_mag": 10.5,
                        "phot_bp_mean_mag": 11.0,
                        "phot_rp_mean_mag": 10.0,
                    }
                ],
            )
            request = StarHostRequest(
                tic_id=None,
                gaia_source_id=1060421588522505216,
                label="host_a",
            )
            host = resolve_host(
                request,
                gaia_catalog_path=str(catalog),
                allow_remote=False,
            )
            self.assertEqual(host.input_kind, "gaia")
            self.assertEqual(host.gaia_source_id, 1060421588522505216)
            self.assertEqual(host.resolution_method, "local_catalog")
            self.assertAlmostEqual(host.ra, 210.5)
            self.assertAlmostEqual(host.dec, 81.8)
            self.assertEqual(host.label, "host_a")

    def test_tic_without_gaia_column_uses_nearest_local_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "gaia.csv"
            _write_gaia_catalog(
                catalog,
                [
                    {
                        "source_id": 1060421588522505216,
                        "ra": 100.0,
                        "dec": 20.0,
                        "ra_error": 0.01,
                        "dec_error": 0.01,
                        "parallax": 1.0,
                        "parallax_error": 0.1,
                        "phot_g_mean_mag": 12.0,
                        "phot_bp_mean_mag": 13.0,
                        "phot_rp_mean_mag": 11.0,
                    },
                    {
                        "source_id": 9999999999999999999,
                        "ra": 200.0,
                        "dec": 30.0,
                        "ra_error": 0.01,
                        "dec_error": 0.01,
                        "parallax": 1.0,
                        "parallax_error": 0.1,
                        "phot_g_mean_mag": 14.0,
                        "phot_bp_mean_mag": 15.0,
                        "phot_rp_mean_mag": 13.0,
                    },
                ],
            )

            tic_table = mock.Mock()
            tic_table.__len__ = mock.Mock(return_value=1)
            tic_row = mock.Mock()
            tic_row.colnames = ["ra", "dec"]
            tic_row.__getitem__ = mock.Mock(
                side_effect=lambda key: {"ra": 100.00005, "dec": 20.00005}[key]
            )
            tic_table.__getitem__ = mock.Mock(return_value=tic_row)

            with mock.patch(
                "syndiff_pipeline.star.identifiers._query_tic",
                return_value=tic_table,
            ):
                host = resolve_host(
                    StarHostRequest(tic_id=142748283, gaia_source_id=None, label=None),
                    gaia_catalog_path=str(catalog),
                    allow_remote=False,
                )

            self.assertEqual(host.input_kind, "tic")
            self.assertEqual(host.tic_id, 142748283)
            self.assertEqual(host.gaia_source_id, 1060421588522505216)
            self.assertEqual(host.resolution_method, "tic_local_match")

    def test_tic_with_gaia_column_resolves_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "gaia.csv"
            _write_gaia_catalog(
                catalog,
                [
                    {
                        "source_id": 1060421588522505216,
                        "ra": 100.0,
                        "dec": 20.0,
                        "ra_error": 0.01,
                        "dec_error": 0.01,
                        "parallax": 1.0,
                        "parallax_error": 0.1,
                        "phot_g_mean_mag": 12.0,
                        "phot_bp_mean_mag": 13.0,
                        "phot_rp_mean_mag": 11.0,
                    }
                ],
            )

            tic_table = mock.Mock()
            tic_table.__len__ = mock.Mock(return_value=1)
            tic_row = mock.Mock()
            tic_row.colnames = ["ra", "dec", "GAIA"]
            tic_row.__getitem__ = mock.Mock(
                side_effect=lambda key: {
                    "ra": 100.0,
                    "dec": 20.0,
                    "GAIA": 1060421588522505216,
                }[key]
            )
            tic_table.__getitem__ = mock.Mock(return_value=tic_row)

            with mock.patch(
                "syndiff_pipeline.star.identifiers._query_tic",
                return_value=tic_table,
            ):
                host = resolve_host(
                    StarHostRequest(tic_id=142748283, gaia_source_id=None, label=None),
                    gaia_catalog_path=str(catalog),
                    allow_remote=False,
                )

            self.assertEqual(host.resolution_method, "tic_local_match")
            self.assertEqual(host.gaia_source_id, 1060421588522505216)

    def test_persistence_helpers(self):
        host = ResolvedHost(
            input_kind="gaia",
            input_value=1060421588522505216,
            tic_id=None,
            gaia_source_id=1060421588522505216,
            ra=100.0,
            dec=20.0,
            phot_g_mean_mag=12.0,
            phot_bp_mean_mag=13.0,
            phot_rp_mean_mag=11.0,
            resolution_method="local_catalog",
            label="x",
        )
        with tempfile.TemporaryDirectory() as tmp:
            ident = Path(tmp) / "identifier.json"
            row = Path(tmp) / "host_gaia_row.csv"
            write_identifier_json(host, str(ident))
            write_host_gaia_row_csv(host, str(row))
            self.assertTrue(ident.is_file())
            self.assertTrue(row.is_file())
            self.assertIn("1060421588522505216", ident.read_text(encoding="utf-8"))
            csv_text = row.read_text(encoding="utf-8")
            self.assertIn("source_id", csv_text)
            self.assertIn("1060421588522505216", csv_text)


if __name__ == "__main__":
    unittest.main()
