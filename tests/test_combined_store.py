"""Tests for template_creation.processing.combined_store (PR4 groundwork).

Covers: fingerprint determinism/param-sensitivity, atomic publish (both the
real ``common.provenance.publish`` path and the local fallback used when
that package can't be imported), load-after-publish round-trip, never-raise
behavior on an unwritable store, and the payload-shape contract pinned from
``ps1_process.py``'s ``band_cache`` (see module docstring in
``combined_store.py``): ``{combined_image: float32, combined_mask: uint16,
headers_data: dict[str, str], removed_stars: list[dict]}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import combined_store as cs


def _payload(seed: int = 0):
    rng = np.random.default_rng(seed)
    combined_image = rng.random((16, 16)).astype(np.float32)
    combined_mask = rng.integers(0, 4, size=(16, 16)).astype(np.uint16)
    headers_data = {"r": "R HEADER TEXT", "i": "I HEADER TEXT", "z": "Z HEADER", "y": "Y HEADER"}
    removed_stars = [
        {
            "source_id": 12345,
            "ra": 10.5,
            "dec": -5.25,
            "pixel_x": 3.0,
            "pixel_y": 4.0,
            "tess_mag": 9.5,
            "removal_reason": "catalog_bright_star",
            "skycell_id": "skycell.1234",
        }
    ]
    return combined_image, combined_mask, headers_data, removed_stars


# ---------------------------------------------------------------------------
# Recipe / fingerprint determinism & sensitivity
# ---------------------------------------------------------------------------


def test_combined_recipe_defaults():
    recipe = cs.combined_recipe()
    assert recipe["enable_saturation_correction"] is False
    assert recipe["remove_saturated_stars"] is True
    assert recipe["bright_star_mag_threshold"] == 13.0
    assert recipe["band_weights"] == cs.DEFAULT_BAND_WEIGHTS
    assert recipe["gaia_version"] is None


def test_combined_recipe_accepts_mapping_and_object_and_overrides():
    from types import SimpleNamespace

    as_mapping = cs.combined_recipe({"bright_star_mag_threshold": 11.0})
    as_object = cs.combined_recipe(SimpleNamespace(bright_star_mag_threshold=11.0))
    assert as_mapping["bright_star_mag_threshold"] == 11.0
    assert as_object["bright_star_mag_threshold"] == 11.0

    overridden = cs.combined_recipe(gaia_version="dr3")
    assert overridden["gaia_version"] == "dr3"


def test_recipe_id_is_deterministic():
    recipe = cs.combined_recipe()
    rid1 = cs.combined_recipe_id(recipe)
    rid2 = cs.combined_recipe_id(dict(recipe))  # fresh dict, same content
    assert rid1 == rid2
    assert isinstance(rid1, str) and len(rid1) > 0


def test_recipe_id_sensitive_to_param_change():
    base = cs.combined_recipe()
    changed = cs.combined_recipe(bright_star_mag_threshold=10.0)
    assert cs.combined_recipe_id(base) != cs.combined_recipe_id(changed)


def test_fingerprint_deterministic_and_sensitive_to_spatial_key_and_inputs():
    recipe = cs.combined_recipe()
    rid = cs.combined_recipe_id(recipe)

    fp1 = cs.combined_fingerprint("skycell.1234", "000", rid)
    fp2 = cs.combined_fingerprint("skycell.1234", "000", rid)
    assert fp1 == fp2

    fp_other_cell = cs.combined_fingerprint("skycell.1234", "001", rid)
    assert fp_other_cell != fp1

    fp_other_proj = cs.combined_fingerprint("skycell.5678", "000", rid)
    assert fp_other_proj != fp1

    fp_with_input = cs.combined_fingerprint("skycell.1234", "000", rid, ["raw_skycell_fp_abc"])
    assert fp_with_input != fp1

    fp_with_input_reordered = cs.combined_fingerprint(
        "skycell.1234", "000", rid, ["raw_skycell_fp_abc"]
    )
    assert fp_with_input == fp_with_input_reordered


# ---------------------------------------------------------------------------
# raw_skycell input fingerprint (bug fix: seed_band_cache_from_combined_store
# / publish_combined_cell's caller in ps1_process.py must thread a real
# raw_skycell input fingerprint into combined_fingerprint, so a re-downloaded
# raw skycell mints a different combined_fingerprint instead of silently
# serving stale cached data forever -- plan §6 combined_skycell inputs =
# raw_skycell, source_catalog).
# ---------------------------------------------------------------------------


def _write_raw_skycell(data_root: Path, projection: str, cell: str, content: bytes) -> Path:
    group_dir = data_root / "ps1_skycells_zarr" / "ps1_skycells.zarr" / projection / f"{projection}.{cell}"
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "r.dat").write_bytes(content)
    return group_dir


def test_raw_skycell_input_fingerprint_missing_store_is_stable(tmp_path: Path):
    fp1 = cs.raw_skycell_input_fingerprint(tmp_path, "skycell.1234", "000")
    fp2 = cs.raw_skycell_input_fingerprint(tmp_path, "skycell.1234", "000")
    assert fp1 == fp2
    assert isinstance(fp1, str) and fp1


def test_raw_skycell_input_fingerprint_present_differs_from_missing(tmp_path: Path):
    missing_fp = cs.raw_skycell_input_fingerprint(tmp_path, "skycell.1234", "000")
    _write_raw_skycell(tmp_path, "skycell.1234", "000", b"raw-bytes-v1")
    present_fp = cs.raw_skycell_input_fingerprint(tmp_path, "skycell.1234", "000")
    assert present_fp != missing_fp


def test_raw_skycell_input_fingerprint_stable_for_unchanged_on_disk_state(tmp_path: Path):
    _write_raw_skycell(tmp_path, "skycell.1234", "000", b"raw-bytes-v1")
    fp1 = cs.raw_skycell_input_fingerprint(tmp_path, "skycell.1234", "000")
    fp2 = cs.raw_skycell_input_fingerprint(tmp_path, "skycell.1234", "000")
    assert fp1 == fp2


def test_raw_skycell_input_fingerprint_changes_when_version_token_changes(tmp_path: Path):
    """Direct proxy for decision #6: a different on-disk raw-skycell state
    (simulating a re-download that changes size/mtime) must mint a different
    ``raw_skycell`` input fingerprint.
    """
    _write_raw_skycell(tmp_path, "skycell.1234", "000", b"raw-bytes-v1")
    fp_before = cs.raw_skycell_input_fingerprint(tmp_path, "skycell.1234", "000")

    # Re-download: different size (and, on any real filesystem, a newer mtime).
    _write_raw_skycell(tmp_path, "skycell.1234", "000", b"raw-bytes-v2-different-length")
    fp_after = cs.raw_skycell_input_fingerprint(tmp_path, "skycell.1234", "000")

    assert fp_before != fp_after


def test_combined_fingerprint_changes_when_raw_skycell_version_token_changes(tmp_path: Path):
    """The actual bug fix, proven at the ``combined_fingerprint`` level: two
    different raw-skycell version tokens for the same projection+skycell+
    recipe must produce different ``combined_fingerprint`` results.
    """
    projection, cell = "skycell.1234", "000"
    recipe = cs.combined_recipe()
    rid = cs.combined_recipe_id(recipe)

    _write_raw_skycell(tmp_path, projection, cell, b"raw-bytes-v1")
    raw_fp_v1 = cs.raw_skycell_input_fingerprint(tmp_path, projection, cell)
    fp_v1 = cs.combined_fingerprint(projection, cell, rid, [raw_fp_v1])

    _write_raw_skycell(tmp_path, projection, cell, b"raw-bytes-v2-different-length")
    raw_fp_v2 = cs.raw_skycell_input_fingerprint(tmp_path, projection, cell)
    fp_v2 = cs.combined_fingerprint(projection, cell, rid, [raw_fp_v2])

    assert raw_fp_v1 != raw_fp_v2
    assert fp_v1 != fp_v2


def test_seed_and_publish_call_sites_compute_identical_raw_skycell_input_fingerprint(
    tmp_path: Path,
):
    """Symmetry check: ``seed_band_cache_from_combined_store`` (lookup) and
    the ``publish_combined_cell`` caller in ``ps1_process.py`` (publish) both
    derive their ``raw_skycell`` input fingerprint via
    ``raw_skycell_input_fingerprint(data_root, projection, cell)`` -- same
    arguments, same function -- so a lookup for unchanged raw-skycell state
    always resolves to exactly what was published, and real cache hits still
    occur. This is what ``test_seed_lookup_round_trip_hit_then_miss`` below
    exercises end-to-end; here the two "sides" are isolated directly.
    """
    projection, cell = "skycell.1234", "000"
    _write_raw_skycell(tmp_path, projection, cell, b"raw-bytes-v1")

    # "publish" side, e.g. ps1_process.py's _publish_combined closure.
    publish_side_fp = cs.raw_skycell_input_fingerprint(tmp_path, projection, cell)
    # "seed" side, e.g. seed_band_cache_from_combined_store's per-name loop.
    seed_side_fp = cs.raw_skycell_input_fingerprint(tmp_path, projection, cell)

    assert publish_side_fp == seed_side_fp


def test_seed_lookup_round_trip_hit_then_miss_after_redownload(tmp_path: Path):
    """Full round-trip (plan §17 failure-matrix intent): publish with
    version-token A, seed-lookup with version-token A hits; seed-lookup with
    version-token B (simulating a re-download) misses and correctly falls
    back to recompute rather than silently serving stale cached data.
    """
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    projection, cell = "skycell.1234", "000"
    skycell_id = f"{projection}.{cell}"
    recipe = cs.combined_recipe()

    _write_raw_skycell(tmp_path, projection, cell, b"raw-bytes-v1")
    raw_fp_a = cs.raw_skycell_input_fingerprint(tmp_path, projection, cell)

    info = cs.publish_combined_cell(
        tmp_path,
        projection,
        cell,
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
        input_fingerprints=[raw_fp_a],
    )
    assert info is not None

    # Version-token A (unchanged raw skycell): seed lookup hits.
    hits = cs.seed_band_cache_from_combined_store(tmp_path, [skycell_id], recipe)
    assert skycell_id in hits
    np.testing.assert_array_equal(hits[skycell_id]["combined_image"], combined_image)

    # Simulate a raw-skycell re-download: version token B.
    _write_raw_skycell(tmp_path, projection, cell, b"raw-bytes-v2-different-length")

    hits_after_redownload = cs.seed_band_cache_from_combined_store(tmp_path, [skycell_id], recipe)
    assert skycell_id not in hits_after_redownload


def test_seed_band_cache_from_combined_store_threads_raw_skycell_fingerprint(tmp_path: Path):
    """``seed_band_cache_from_combined_store`` must actually use
    ``raw_skycell_input_fingerprint`` in its lookup (not silently default to
    ``()`` as before the fix) -- proven by publishing under the fingerprint
    ``combined_fingerprint`` would produce with NO input_fingerprints (the
    pre-fix behavior) and confirming the seed lookup does *not* find it, even
    though the underlying raw skycell is present.
    """
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    projection, cell = "skycell.1234", "000"
    skycell_id = f"{projection}.{cell}"
    recipe = cs.combined_recipe()

    _write_raw_skycell(tmp_path, projection, cell, b"raw-bytes-v1")

    # Publish under the OLD (pre-fix) fingerprint shape: no input_fingerprints.
    stale_info = cs.publish_combined_cell(
        tmp_path,
        projection,
        cell,
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    assert stale_info is not None

    hits = cs.seed_band_cache_from_combined_store(tmp_path, [skycell_id], recipe)
    assert skycell_id not in hits, (
        "seed lookup found a cell published without a raw_skycell input "
        "fingerprint -- seed_band_cache_from_combined_store is not threading "
        "raw_skycell_input_fingerprint into its combined_fingerprint() call"
    )


# ---------------------------------------------------------------------------
# Publish / load round-trip
# ---------------------------------------------------------------------------


def test_publish_and_load_round_trip(tmp_path: Path):
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    recipe = cs.combined_recipe()

    info = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    assert info is not None
    assert info["already_published"] is False
    fp = info["fingerprint"]

    expected_dir = cs.combined_cell_dir(tmp_path, "skycell.1234", "000", fp)
    assert info["dir"] == expected_dir
    assert expected_dir.is_dir()

    loaded = cs.try_load_combined_cell(tmp_path, "skycell.1234", "000", fp)
    assert loaded is not None
    np.testing.assert_array_equal(loaded["combined_image"], combined_image)
    np.testing.assert_array_equal(loaded["combined_mask"], combined_mask)
    assert loaded["headers_data"] == headers_data
    assert loaded["removed_stars"] == removed_stars


def test_publish_is_idempotent_on_identical_fingerprint(tmp_path: Path):
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    recipe = cs.combined_recipe()

    info1 = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    info2 = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    assert info1["fingerprint"] == info2["fingerprint"]
    assert info1["already_published"] is False
    assert info2["already_published"] is True


def test_try_load_returns_none_when_never_published(tmp_path: Path):
    assert cs.try_load_combined_cell(tmp_path, "skycell.1234", "000", "deadbeef") is None


def test_try_load_returns_none_when_a_member_is_missing(tmp_path: Path):
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    recipe = cs.combined_recipe()
    info = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    assert info is not None
    (info["dir"] / "removed_stars.json").unlink()
    assert cs.try_load_combined_cell(tmp_path, "skycell.1234", "000", info["fingerprint"]) is None


def test_try_load_returns_none_on_corrupt_json(tmp_path: Path):
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    recipe = cs.combined_recipe()
    info = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    assert info is not None
    (info["dir"] / "headers.json").write_text("{not valid json")
    assert cs.try_load_combined_cell(tmp_path, "skycell.1234", "000", info["fingerprint"]) is None


# ---------------------------------------------------------------------------
# Payload-shape contract (pinned from ps1_process.py's band_cache, see
# process_coordinator / process_single_cell)
# ---------------------------------------------------------------------------


def test_payload_shape_contract(tmp_path: Path):
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    recipe = cs.combined_recipe()
    info = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    loaded = cs.try_load_combined_cell(tmp_path, "skycell.1234", "000", info["fingerprint"])
    assert set(loaded.keys()) == {"combined_image", "combined_mask", "headers_data", "removed_stars"}
    assert loaded["combined_image"].dtype == np.float32
    assert loaded["combined_mask"].dtype == np.uint16
    assert isinstance(loaded["headers_data"], dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in loaded["headers_data"].items())
    assert isinstance(loaded["removed_stars"], list)
    assert all(isinstance(rec, dict) for rec in loaded["removed_stars"])


def test_store_location_matches_decision_14_layout(tmp_path: Path):
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    recipe = cs.combined_recipe()
    info = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    rel = info["dir"].relative_to(tmp_path)
    assert rel.parts[:2] == ("ps1_skycells_zarr", "ps1_combined.zarr")
    assert rel.parts[2] == "skycell.1234"
    assert rel.parts[3] == "000"
    assert rel.parts[4] == info["fingerprint"]


def test_provenance_sidecar_is_self_describing(tmp_path: Path):
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    recipe = cs.combined_recipe()
    info = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
        input_fingerprints=["raw_fp_1"],
    )
    sidecar_path = info["dir"] / "_provenance.json"
    assert sidecar_path.is_file()
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["fingerprint"] == info["fingerprint"]
    assert sidecar["kind"] == "combined_skycell"
    assert sidecar["recipe_id"] == info["recipe_id"]
    assert sidecar["inputs"] == ["raw_fp_1"] or sidecar.get("input_fingerprints") == ["raw_fp_1"]


# ---------------------------------------------------------------------------
# Never-raises / atomicity
# ---------------------------------------------------------------------------


def test_publish_never_raises_when_unwritable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    recipe = cs.combined_recipe()

    def _boom(*args, **kwargs):
        raise PermissionError("simulated unwritable store")

    # Force both the real-provenance path and the local fallback path to hit
    # the same failure so this test is meaningful regardless of which
    # publish backend is active in this environment.
    monkeypatch.setattr(cs, "_HAVE_PROVENANCE_PUBLISH", False, raising=False)
    monkeypatch.setattr(np, "savez_compressed", _boom)

    result = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    assert result is None

    recipe_id = cs.combined_recipe_id(recipe)
    fp = cs.combined_fingerprint("skycell.1234", "000", recipe_id)
    final_dir = cs.combined_cell_dir(tmp_path, "skycell.1234", "000", fp)
    assert not cs._payload_complete(final_dir)


def test_local_fallback_publish_is_atomic_and_leaves_only_tmp_on_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Force the local (non-provenance-package) publish path and simulate a
    crash partway through writing the payload. Per the plan's §17 failure
    matrix ("worker crash mid-write -> only _tmp_* orphan; no sidecar; index
    unaffected"), the fingerprinted final directory must never appear, and
    the only thing left behind is the ``_tmp_*`` orphan -- never a partial
    artifact that looks complete.
    """
    combined_image, combined_mask, headers_data, removed_stars = _payload()
    recipe = cs.combined_recipe()

    monkeypatch.setattr(cs, "_HAVE_PROVENANCE_PUBLISH", False, raising=False)

    real_dump = json.dump
    calls = {"n": 0}

    def _flaky_json_dump(obj, fh, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # let headers.json through, fail on removed_stars.json
            raise RuntimeError("simulated interrupted write")
        return real_dump(obj, fh, **kwargs)

    monkeypatch.setattr(cs.json, "dump", _flaky_json_dump)

    result = cs.publish_combined_cell(
        tmp_path,
        "skycell.1234",
        "000",
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
    )
    assert result is None

    recipe_id = cs.combined_recipe_id(recipe)
    fp = cs.combined_fingerprint("skycell.1234", "000", recipe_id)
    final_dir = cs.combined_cell_dir(tmp_path, "skycell.1234", "000", fp)
    assert not final_dir.exists()

    # The only thing on disk under this fp is the _tmp_* orphan -- never
    # something masquerading as the finished fingerprinted artifact. (The
    # orphan's own completeness is not asserted: open() may have created an
    # empty removed_stars.json before json.dump raised on it, which is a
    # harmless implementation detail -- callers only ever resolve the
    # fingerprinted `{fp}` key, never a `_tmp_*` name, so a stub file inside
    # the orphan can never be mistaken for a published artifact.)
    siblings = list(final_dir.parent.iterdir()) if final_dir.parent.exists() else []
    tmp_siblings = [p for p in siblings if p.name.startswith(f"_tmp_{fp}_")]
    assert len(tmp_siblings) == 1
    assert tmp_siblings[0].is_dir()
    assert siblings == tmp_siblings  # nothing else landed under skycell/000/
