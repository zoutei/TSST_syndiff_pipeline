"""Tests for template_creation.processing.convolved_store (PR5 groundwork).

Structural mirror of ``test_combined_store.py`` plus the convolved-specific
contract: the recipe always carries ``padding="same_projection_only"``, and
the fingerprint Merkle-inputs the upstream ``combined_skycell`` fingerprint
(required, not optional) per plan §13.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import convolved_store as vs


def _payload(seed: int = 0):
    rng = np.random.default_rng(seed)
    convolved_image = rng.random((16, 16)).astype(np.float32)
    convolved_mask = rng.integers(0, 4, size=(16, 16)).astype(np.uint16)
    headers_data = {"r": "R HEADER TEXT", "i": "I HEADER TEXT", "z": "Z HEADER", "y": "Y HEADER"}
    removed_stars = [
        {
            "source_id": 999,
            "ra": 11.0,
            "dec": -3.0,
            "removal_reason": "catalog_neighbor",
            "skycell_id": "skycell.1234",
        }
    ]
    return convolved_image, convolved_mask, headers_data, removed_stars


_UPSTREAM_COMBINED_FP = "combined_fp_abcdef0123456789"


# ---------------------------------------------------------------------------
# Recipe / fingerprint determinism & sensitivity
# ---------------------------------------------------------------------------


def test_convolved_recipe_defaults():
    recipe = vs.convolved_recipe()
    assert recipe["psf_sigma"] == vs.DEFAULT_PSF_SIGMA == 40.0
    assert recipe["radius"] == vs.DEFAULT_RADIUS == 470
    assert recipe["mode"] == "constant"
    assert recipe["padding"] == "same_projection_only"


def test_convolved_recipe_padding_is_fixed_unless_explicitly_overridden():
    # Not settable via a resolved-config attribute (padding decouple is a
    # scc_assembly-time decision, not a per-cell knob) ...
    from types import SimpleNamespace

    recipe = vs.convolved_recipe(SimpleNamespace(padding="cross_projection"))
    assert recipe["padding"] == "same_projection_only"
    # ... but explicit keyword overrides win, for forward-compat testing.
    recipe2 = vs.convolved_recipe(padding="future_mode")
    assert recipe2["padding"] == "future_mode"


def test_recipe_id_is_deterministic():
    recipe = vs.convolved_recipe()
    rid1 = vs.convolved_recipe_id(recipe)
    rid2 = vs.convolved_recipe_id(dict(recipe))
    assert rid1 == rid2


def test_recipe_id_sensitive_to_psf_sigma_change():
    base = vs.convolved_recipe()
    changed = vs.convolved_recipe(psf_sigma=45.0)
    assert vs.convolved_recipe_id(base) != vs.convolved_recipe_id(changed)


def test_fingerprint_deterministic_and_sensitive_to_upstream_combined_fp():
    recipe = vs.convolved_recipe()
    rid = vs.convolved_recipe_id(recipe)

    fp1 = vs.convolved_fingerprint("skycell.1234", "000", rid, [_UPSTREAM_COMBINED_FP])
    fp2 = vs.convolved_fingerprint("skycell.1234", "000", rid, [_UPSTREAM_COMBINED_FP])
    assert fp1 == fp2

    # Changing only the upstream combined_skycell fingerprint (e.g. because
    # star-removal params changed) must re-fingerprint this cell too --
    # that's the whole point of Merkle-chaining the input (plan §13).
    fp_other_upstream = vs.convolved_fingerprint("skycell.1234", "000", rid, ["a_different_combined_fp"])
    assert fp_other_upstream != fp1


# ---------------------------------------------------------------------------
# Publish / load round-trip
# ---------------------------------------------------------------------------


def test_publish_and_load_round_trip(tmp_path: Path):
    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    recipe = vs.convolved_recipe()

    info = vs.publish_convolved_cell(
        tmp_path,
        "skycell.1234",
        "000",
        convolved_image=convolved_image,
        convolved_mask=convolved_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
        combined_fingerprint=_UPSTREAM_COMBINED_FP,
    )
    assert info is not None
    assert info["already_published"] is False
    fp = info["fingerprint"]

    expected_dir = vs.convolved_cell_dir(tmp_path, "skycell.1234", "000", fp)
    assert info["dir"] == expected_dir
    assert expected_dir.is_dir()

    loaded = vs.try_load_convolved_cell(tmp_path, "skycell.1234", "000", fp)
    assert loaded is not None
    np.testing.assert_array_equal(loaded["convolved_image"], convolved_image)
    np.testing.assert_array_equal(loaded["convolved_mask"], convolved_mask)
    assert loaded["headers_data"] == headers_data
    assert loaded["removed_stars"] == removed_stars


def test_publish_is_idempotent_on_identical_fingerprint(tmp_path: Path):
    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    recipe = vs.convolved_recipe()
    kwargs = dict(
        convolved_image=convolved_image,
        convolved_mask=convolved_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
        combined_fingerprint=_UPSTREAM_COMBINED_FP,
    )
    info1 = vs.publish_convolved_cell(tmp_path, "skycell.1234", "000", **kwargs)
    info2 = vs.publish_convolved_cell(tmp_path, "skycell.1234", "000", **kwargs)
    assert info1["fingerprint"] == info2["fingerprint"]
    assert info1["already_published"] is False
    assert info2["already_published"] is True


def test_republish_rekeys_when_upstream_combined_fp_changes(tmp_path: Path):
    """Mirrors the real gaia_version_stamp migration: an old convolved cell
    chained on a since-superseded combined_fingerprint should be
    republished under the new one, once the two combined cells' pixel
    content is confirmed identical."""
    from syndiff_pipeline.template_creation.processing import combined_store as cs

    combined_image, combined_mask, chd, crs = (
        np.random.default_rng(1).random((16, 16)).astype(np.float32),
        np.random.default_rng(2).integers(0, 4, size=(16, 16)).astype(np.uint16),
        {"r": "R"}, [],
    )
    old_recipe = cs.combined_recipe(gaia_version="/some/path/catalog.csv:1:2")
    new_recipe = cs.combined_recipe(gaia_version="loaded")
    old_info = cs.publish_combined_cell(
        tmp_path, "skycell.1234", "000",
        combined_image=combined_image, combined_mask=combined_mask,
        headers_data=chd, removed_stars=crs, recipe=old_recipe,
    )
    new_info = cs.publish_combined_cell(
        tmp_path, "skycell.1234", "000",
        combined_image=combined_image, combined_mask=combined_mask,
        headers_data=chd, removed_stars=crs, recipe=new_recipe,
    )
    assert old_info["fingerprint"] != new_info["fingerprint"]

    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    conv_recipe = vs.convolved_recipe()
    old_conv_info = vs.publish_convolved_cell(
        tmp_path, "skycell.1234", "000",
        convolved_image=convolved_image, convolved_mask=convolved_mask,
        headers_data=headers_data, removed_stars=removed_stars,
        recipe=conv_recipe, combined_fingerprint=old_info["fingerprint"],
    )
    assert old_conv_info is not None

    assert vs.resolve_convolved_fingerprint_for_recipe(
        tmp_path, "skycell.1234", "000", conv_recipe, new_info["fingerprint"]
    ) is None

    result = vs.republish_convolved_cell_for_recipe(
        tmp_path, "skycell.1234", "000", conv_recipe, new_info["fingerprint"]
    )
    assert result is not None
    assert result["fingerprint"] != old_conv_info["fingerprint"]

    resolved_fp = vs.resolve_convolved_fingerprint_for_recipe(
        tmp_path, "skycell.1234", "000", conv_recipe, new_info["fingerprint"]
    )
    assert resolved_fp == result["fingerprint"]
    loaded = vs.try_load_convolved_cell(tmp_path, "skycell.1234", "000", resolved_fp)
    assert loaded is not None
    np.testing.assert_array_equal(loaded["convolved_image"], convolved_image)


def test_republish_rekeys_when_upstream_combined_content_has_matching_nans(tmp_path: Path):
    """Regression test: a real s0020/c3/k3 skycell had a single masked NaN
    pixel in its combined image, and a naive ``np.array_equal`` (without
    ``equal_nan=True``) reported the identical-content old/new combined
    cells as "different" purely because NaN != NaN, silently skipping an
    otherwise-safe rekey."""
    from syndiff_pipeline.template_creation.processing import combined_store as cs

    combined_image = np.random.default_rng(1).random((16, 16)).astype(np.float32)
    combined_image[3, 5] = np.nan
    combined_mask = np.zeros((16, 16), dtype=np.uint16)
    chd, crs = {"r": "R"}, []
    old_info = cs.publish_combined_cell(
        tmp_path, "skycell.1234", "000",
        combined_image=combined_image, combined_mask=combined_mask,
        headers_data=chd, removed_stars=crs, recipe=cs.combined_recipe(gaia_version="a"),
    )
    new_info = cs.publish_combined_cell(
        tmp_path, "skycell.1234", "000",
        combined_image=combined_image, combined_mask=combined_mask,
        headers_data=chd, removed_stars=crs, recipe=cs.combined_recipe(gaia_version="loaded"),
    )

    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    conv_recipe = vs.convolved_recipe()
    vs.publish_convolved_cell(
        tmp_path, "skycell.1234", "000",
        convolved_image=convolved_image, convolved_mask=convolved_mask,
        headers_data=headers_data, removed_stars=removed_stars,
        recipe=conv_recipe, combined_fingerprint=old_info["fingerprint"],
    )

    result = vs.republish_convolved_cell_for_recipe(
        tmp_path, "skycell.1234", "000", conv_recipe, new_info["fingerprint"]
    )
    assert result is not None


def test_republish_skips_when_upstream_combined_content_actually_differs(tmp_path: Path):
    from syndiff_pipeline.template_creation.processing import combined_store as cs

    old_image = np.random.default_rng(1).random((16, 16)).astype(np.float32)
    new_image = np.random.default_rng(3).random((16, 16)).astype(np.float32)
    chd, crs = {"r": "R"}, []
    old_combined = cs.publish_combined_cell(
        tmp_path, "skycell.1234", "000",
        combined_image=old_image, combined_mask=np.zeros((16, 16), dtype=np.uint16),
        headers_data=chd, removed_stars=crs, recipe=cs.combined_recipe(gaia_version="a"),
    )
    new_combined = cs.publish_combined_cell(
        tmp_path, "skycell.1234", "000",
        combined_image=new_image, combined_mask=np.zeros((16, 16), dtype=np.uint16),
        headers_data=chd, removed_stars=crs, recipe=cs.combined_recipe(gaia_version="b"),
    )

    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    conv_recipe = vs.convolved_recipe()
    vs.publish_convolved_cell(
        tmp_path, "skycell.1234", "000",
        convolved_image=convolved_image, convolved_mask=convolved_mask,
        headers_data=headers_data, removed_stars=removed_stars,
        recipe=conv_recipe, combined_fingerprint=old_combined["fingerprint"],
    )

    result = vs.republish_convolved_cell_for_recipe(
        tmp_path, "skycell.1234", "000", conv_recipe, new_combined["fingerprint"]
    )
    assert result is None


def test_republish_convolved_returns_none_when_no_payload_exists(tmp_path: Path):
    result = vs.republish_convolved_cell_for_recipe(
        tmp_path, "skycell.1234", "000", vs.convolved_recipe(), "some_new_combined_fp"
    )
    assert result is None


def test_try_load_returns_none_when_never_published(tmp_path: Path):
    assert vs.try_load_convolved_cell(tmp_path, "skycell.1234", "000", "deadbeef") is None


def test_try_load_returns_none_when_a_member_is_missing(tmp_path: Path):
    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    recipe = vs.convolved_recipe()
    info = vs.publish_convolved_cell(
        tmp_path,
        "skycell.1234",
        "000",
        convolved_image=convolved_image,
        convolved_mask=convolved_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
        combined_fingerprint=_UPSTREAM_COMBINED_FP,
    )
    assert info is not None
    (info["dir"] / "arrays.npz").unlink()
    assert vs.try_load_convolved_cell(tmp_path, "skycell.1234", "000", info["fingerprint"]) is None


# ---------------------------------------------------------------------------
# Payload-shape contract
# ---------------------------------------------------------------------------


def test_payload_shape_contract(tmp_path: Path):
    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    recipe = vs.convolved_recipe()
    info = vs.publish_convolved_cell(
        tmp_path,
        "skycell.1234",
        "000",
        convolved_image=convolved_image,
        convolved_mask=convolved_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
        combined_fingerprint=_UPSTREAM_COMBINED_FP,
    )
    loaded = vs.try_load_convolved_cell(tmp_path, "skycell.1234", "000", info["fingerprint"])
    assert set(loaded.keys()) == {"convolved_image", "convolved_mask", "headers_data", "removed_stars"}
    assert loaded["convolved_image"].dtype == np.float32
    assert loaded["convolved_mask"].dtype == np.uint16
    assert isinstance(loaded["headers_data"], dict)
    assert isinstance(loaded["removed_stars"], list)


def test_store_location_matches_decision_14_layout(tmp_path: Path):
    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    recipe = vs.convolved_recipe()
    info = vs.publish_convolved_cell(
        tmp_path,
        "skycell.1234",
        "000",
        convolved_image=convolved_image,
        convolved_mask=convolved_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
        combined_fingerprint=_UPSTREAM_COMBINED_FP,
    )
    rel = info["dir"].relative_to(tmp_path)
    assert rel.parts[:2] == ("ps1_skycells_zarr", "ps1_convolved.zarr")
    assert rel.parts[2] == "skycell.1234"
    assert rel.parts[3] == "000"
    assert rel.parts[4] == info["fingerprint"]


# ---------------------------------------------------------------------------
# Never-raises / atomicity
# ---------------------------------------------------------------------------


def test_publish_never_raises_when_unwritable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    recipe = vs.convolved_recipe()

    def _boom(*args, **kwargs):
        raise PermissionError("simulated unwritable store")

    monkeypatch.setattr(vs, "_HAVE_PROVENANCE_PUBLISH", False, raising=False)
    monkeypatch.setattr(np, "savez_compressed", _boom)

    result = vs.publish_convolved_cell(
        tmp_path,
        "skycell.1234",
        "000",
        convolved_image=convolved_image,
        convolved_mask=convolved_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
        combined_fingerprint=_UPSTREAM_COMBINED_FP,
    )
    assert result is None

    recipe_id = vs.convolved_recipe_id(recipe)
    fp = vs.convolved_fingerprint("skycell.1234", "000", recipe_id, [_UPSTREAM_COMBINED_FP])
    final_dir = vs.convolved_cell_dir(tmp_path, "skycell.1234", "000", fp)
    assert not vs._payload_complete(final_dir)


def test_local_fallback_publish_is_atomic_and_leaves_only_tmp_on_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    convolved_image, convolved_mask, headers_data, removed_stars = _payload()
    recipe = vs.convolved_recipe()

    monkeypatch.setattr(vs, "_HAVE_PROVENANCE_PUBLISH", False, raising=False)

    real_dump = json.dump
    calls = {"n": 0}

    def _flaky_json_dump(obj, fh, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated interrupted write")
        return real_dump(obj, fh, **kwargs)

    monkeypatch.setattr(vs.json, "dump", _flaky_json_dump)

    result = vs.publish_convolved_cell(
        tmp_path,
        "skycell.1234",
        "000",
        convolved_image=convolved_image,
        convolved_mask=convolved_mask,
        headers_data=headers_data,
        removed_stars=removed_stars,
        recipe=recipe,
        combined_fingerprint=_UPSTREAM_COMBINED_FP,
    )
    assert result is None

    recipe_id = vs.convolved_recipe_id(recipe)
    fp = vs.convolved_fingerprint("skycell.1234", "000", recipe_id, [_UPSTREAM_COMBINED_FP])
    final_dir = vs.convolved_cell_dir(tmp_path, "skycell.1234", "000", fp)
    assert not final_dir.exists()

    # The only thing on disk under this fp is the _tmp_* orphan -- see
    # test_combined_store.py's identical assertion for the full rationale.
    siblings = list(final_dir.parent.iterdir()) if final_dir.parent.exists() else []
    tmp_siblings = [p for p in siblings if p.name.startswith(f"_tmp_{fp}_")]
    assert len(tmp_siblings) == 1
    assert siblings == tmp_siblings
