"""Asteroid intervals → bit 128; load without ephem; radius uses geometry."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.masking import bits
from syndiff_pipeline.difference_imaging.masking.asteroid_discover import (
    ensure_tess_orbit_times,
    horizons_id_from_object_name,
)
from syndiff_pipeline.difference_imaging.masking.asteroids import (
    load_asteroid_products,
    radius_from_vmag,
)
from syndiff_pipeline.difference_imaging.masking.catalog import MaskCatalog
from syndiff_pipeline.difference_imaging.masking.geometry import radius_from_mag
from syndiff_pipeline.difference_imaging.masking.settings import DEFAULT_TESS_ORBIT_TIMES_URL


def test_radius_from_vmag_uses_geometry_not_drifted_bins():
    # V=12 → T ≈ 11.3 → circle rad 7 (geometry), not drifted tracks.py bins
    r = radius_from_vmag(12.0)
    assert r == radius_from_mag(12.0 - 0.671)


def test_load_without_ephem(tmp_path):
    iv = pd.DataFrame(
        {
            "target_id": ["1"],
            "row": [50],
            "col": [50],
            "cadence_lo": [0],
            "cadence_hi": [2],
        }
    )
    tm = pd.DataFrame({"cadence": [0, 1, 2], "btjd": [1.0, 2.0, 3.0]})
    iv.to_parquet(tmp_path / "pixel_intervals.parquet", index=False)
    tm.to_parquet(tmp_path / "asteroid_ffi_times.parquet", index=False)
    loaded_iv, loaded_tm = load_asteroid_products(tmp_path)
    assert loaded_iv is not None and len(loaded_iv) == 1
    assert loaded_tm is not None and len(loaded_tm) == 3

    static = np.zeros((20, 20), dtype=np.int16)
    crop = {"x_min": 40, "y_min": 40, "x_max": 60, "y_max": 60}
    cat = MaskCatalog.from_arrays(
        static,
        asteroid_intervals_ffi=loaded_iv,
        asteroid_times=loaded_tm,
        crop_bounds=crop,
    )
    m = cat.mask_at(1, which="full")
    assert m[9, 9] & bits.ASTEROID  # row50-1-40=9, col50-1-40=9


def test_horizons_id_from_object_name():
    assert horizons_id_from_object_name("433 Eros") == "433"
    assert horizons_id_from_object_name("2019 AB (2019 AB1)") == "2019 AB1"


def test_ensure_orbit_times_downloads_when_missing(tmp_path):
    dest = tmp_path / "catalogs" / "TESS_orbit_times.csv"
    csv = "Sector,Start of Orbit,End of Orbit\n20,2019-12-24,2020-01-21\n"

    class _Resp:
        def read(self):
            return csv.encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch(
        "syndiff_pipeline.difference_imaging.masking.asteroid_discover.urlopen",
        return_value=_Resp(),
    ) as mock_open:
        path = ensure_tess_orbit_times(20, dest)
        assert mock_open.called
        assert path == dest
        assert dest.is_file()
        assert "20" in dest.read_text()


def test_ensure_orbit_times_redownloads_when_sector_missing(tmp_path):
    dest = tmp_path / "TESS_orbit_times.csv"
    dest.write_text("Sector,Start of Orbit,End of Orbit\n1,2018-07-25,2018-08-22\n")
    updated = (
        "Sector,Start of Orbit,End of Orbit\n"
        "1,2018-07-25,2018-08-22\n"
        "20,2019-12-24,2020-01-21\n"
    )

    class _Resp:
        def read(self):
            return updated.encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch(
        "syndiff_pipeline.difference_imaging.masking.asteroid_discover.urlopen",
        return_value=_Resp(),
    ) as mock_open:
        ensure_tess_orbit_times(20, dest)
        assert mock_open.called
        assert "20" in dest.read_text()


def test_orbit_times_url_default_constant():
    assert DEFAULT_TESS_ORBIT_TIMES_URL.endswith("TESS_orbit_times.csv")


def test_ffi_fov_uses_science_bounds_from_crop(monkeypatch):
    """FOV corners must follow science_bounds_1based(crop_bounds), not hardcoded SCI_*."""
    from types import SimpleNamespace
    import sys
    import types

    import numpy as np

    from syndiff_pipeline.difference_imaging.masking import asteroid_discover as ad

    calls: list[tuple[np.ndarray, np.ndarray]] = []

    class _FakeSky:
        def separation(self, other):
            return SimpleNamespace(deg=np.array([1.0, 1.0, 1.0, 1.0]))

    class _FakeWCS:
        @staticmethod
        def from_sector(sector, camera, ccd):
            return _FakeWCS()

        def pixel_to_world(self, cols, rows):
            calls.append(
                (np.asarray(cols, dtype=float), np.asarray(rows, dtype=float))
            )
            return _FakeSky()

    fake_tesswcs = types.ModuleType("tesswcs")
    fake_tesswcs.WCS = _FakeWCS
    monkeypatch.setitem(sys.modules, "tesswcs", fake_tesswcs)

    crop = {"x_min": 44, "x_max": 2004, "y_min": 0, "y_max": 2018}
    ad.ffi_fov_center_hwidth(20, 1, 1, crop_bounds=crop)
    assert len(calls) >= 1
    cols, rows = calls[0]
    # 1-based inclusive → 0-based: col 45..2004 → 44..2003; row 1..2018 → 0..2017
    np.testing.assert_allclose(cols, [44.0, 2003.0, 2003.0, 44.0])
    np.testing.assert_allclose(rows, [0.0, 0.0, 2017.0, 2017.0])
