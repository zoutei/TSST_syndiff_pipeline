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
    assert recipe["enable_saturation_correction"] is True
    assert recipe["remove_saturated_stars"] is False
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
