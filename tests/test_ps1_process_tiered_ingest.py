"""Tests for the unified tiered-ingest architecture (Part A + Part B of
``doc/ps1_process_tiered_ingest_architecture_plan.md``):

- Part A: lazy, single-cell combined-store lookup inside ``ingest_worker``
  (replacing the old eager bulk preload).
- Part B: per-projection classification (``classify_projection_missing_cells``),
  the per-skycell convolution path (``convolve_single_skycell`` /
  ``process_sparse_projection``), and neighbor-finding
  (``_find_projection_neighbors``).
"""

from __future__ import annotations

import queue as _thread_queue
from pathlib import Path

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing import canonical_cell
from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import convolved_store as vs
from syndiff_pipeline.template_creation.processing import ps1_process as pp

_PROJECTION = "skycell.1234"


def _make_grid_df(projection: str, rows: int, cols: int, cell_size: int = 16) -> pd.DataFrame:
    """A ``rows x cols`` grid of skycells, named ``skycell.<proj>.<r><c>``,
    laid out with sequential integer x/y grid-column/row indices (matching
    ``extract_projection_metadata``'s expected columns)."""
    records = []
    for r in range(rows):
        for c in range(cols):
            records.append(
                {
                    "projection": projection,
                    "y": r,
                    "x": c,
                    "NAME": f"skycell.{projection.split('.')[-1]}.{r}{c}",
                    "NAXIS1": cell_size,
                    "NAXIS2": cell_size,
                }
            )
    return pd.DataFrame.from_records(records)


def _row_metadata(cells: list[str], cell_size: int = 16) -> dict:
    """Projection metadata for one row of ``cells`` (full names, x = 0, 1, ...).
    Schema v2 convolved fingerprints Merkle in the canonical neighbour set, so
    classification/publish need the caller's mapping metadata."""
    df = pd.DataFrame.from_records(
        [
            {"projection": _PROJECTION, "y": 0, "x": i, "NAME": name, "NAXIS1": cell_size, "NAXIS2": cell_size}
            for i, name in enumerate(cells)
        ]
    )
    return pp.extract_projection_metadata(df, _PROJECTION)


def _combined_bundle(seed: int, size: int = 16) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "combined_image": rng.random((size, size)).astype(np.float32),
        "combined_mask": rng.integers(0, 4, size=(size, size)).astype(np.uint16),
        "headers_data": {"r": "R"},
        "removed_stars": [],
    }


def _publish_combined_and_convolved(
    tmp_path: Path, projection: str, cell: str, combined_recipe: dict, convolved_recipe: dict, metadata: dict
) -> None:
    rng = np.random.default_rng(hash(cell) % 1000)
    combined_image = rng.random((16, 16)).astype(np.float32)
    combined_mask = rng.integers(0, 4, size=(16, 16)).astype(np.uint16)
    raw_fp = cs.raw_skycell_input_fingerprint(tmp_path, projection, cell)
    info = cs.publish_combined_cell(
        tmp_path, projection, cell,
        combined_image=combined_image, combined_mask=combined_mask,
        headers_data={"r": "R"}, removed_stars=[],
        recipe=combined_recipe, input_fingerprints=[raw_fp],
    )
    assert info is not None
    vs.publish_convolved_cell(
        tmp_path, projection, cell,
        convolved_image=combined_image, convolved_mask=combined_mask,
        headers_data={"r": "R"}, removed_stars=[],
        recipe=convolved_recipe, combined_fingerprint=info["fingerprint"],
        extra_input_fingerprints=canonical_cell.neighbour_input_fingerprints(
            tmp_path, f"{projection}.{cell}", metadata, combined_recipe
        ),
    )


def _publish_combined_only(
    tmp_path: Path, projection: str, cell: str, combined_recipe: dict, size: int = 16
) -> None:
    rng = np.random.default_rng(hash(cell) % 1000)
    combined_image = rng.random((size, size)).astype(np.float32)
    combined_mask = rng.integers(0, 4, size=(size, size)).astype(np.uint16)
    raw_fp = cs.raw_skycell_input_fingerprint(tmp_path, projection, cell)
    info = cs.publish_combined_cell(
        tmp_path, projection, cell,
        combined_image=combined_image, combined_mask=combined_mask,
        headers_data={"r": "R"}, removed_stars=[],
        recipe=combined_recipe, input_fingerprints=[raw_fp],
    )
    assert info is not None


# ---------------------------------------------------------------------------
# Part A: lazy tier-2 lookup inside ingest_worker
# ---------------------------------------------------------------------------


def test_ingest_worker_tier2_hit_skips_raw_fetch(tmp_path: Path, monkeypatch):
    combined_recipe = cs.combined_recipe()
    _publish_combined_only(tmp_path, _PROJECTION, "000", combined_recipe)

    def _boom(*a, **k):
        raise AssertionError("raw fetch should not be attempted on a combined-store hit")

    monkeypatch.setattr(pp, "fetch_skycell_bands_masks_and_headers", _boom)

    task_queue: _thread_queue.Queue = _thread_queue.Queue()
    raw_cell_queue: _thread_queue.Queue = _thread_queue.Queue()
    task_queue.put((f"skycell.{_PROJECTION.split('.')[-1]}.000", _PROJECTION, 0, 0, "regular"))
    task_queue.put(None)

    band_cache: dict = {}
    pp.ingest_worker(
        task_queue,
        raw_cell_queue,
        ps1_source="stream",
        band_cache=band_cache,
        combined_store_data_root=str(tmp_path),
        combined_store_recipe=combined_recipe,
    )

    bundle = raw_cell_queue.get_nowait()
    assert bundle["task_type"] == "regular_cache_hit"
    assert bundle["skycell_id"] in band_cache


def test_ingest_worker_tier2_miss_falls_through_to_raw_fetch(tmp_path: Path, monkeypatch):
    combined_recipe = cs.combined_recipe()

    called = {"n": 0}

    def _fake_fetch(*a, **k):
        called["n"] += 1
        return {}, {}, {}, {}, {}

    monkeypatch.setattr(pp, "fetch_skycell_bands_masks_and_headers", _fake_fetch)

    task_queue: _thread_queue.Queue = _thread_queue.Queue()
    raw_cell_queue: _thread_queue.Queue = _thread_queue.Queue()
    task_queue.put((f"skycell.{_PROJECTION.split('.')[-1]}.999", _PROJECTION, 0, 0, "regular"))
    task_queue.put(None)

    pp.ingest_worker(
        task_queue,
        raw_cell_queue,
        ps1_source="stream",
        band_cache={},
        combined_store_data_root=str(tmp_path),
        combined_store_recipe=combined_recipe,
    )
    assert called["n"] == 1
    assert raw_cell_queue.empty()  # no bands -> "No band data" branch, skipped


# ---------------------------------------------------------------------------
# Part B step 1: classify_projection_missing_cells
# ---------------------------------------------------------------------------


def test_classify_projection_missing_cells_partitions_correctly(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe()

    cell_names = [
        f"skycell.{_PROJECTION.split('.')[-1]}.000",
        f"skycell.{_PROJECTION.split('.')[-1]}.001",
        f"skycell.{_PROJECTION.split('.')[-1]}.002",
    ]
    metadata = _row_metadata(cell_names)
    _publish_combined_and_convolved(tmp_path, _PROJECTION, "000", combined_recipe, convolved_recipe, metadata)
    _publish_combined_only(tmp_path, _PROJECTION, "001", combined_recipe)
    # "002" never published at all.

    missing = vs.classify_projection_missing_cells(
        tmp_path, _PROJECTION, cell_names, combined_recipe, convolved_recipe, metadata=metadata,
    )
    assert missing == {cell_names[1], cell_names[2]}

    # New v2 behaviour: without the caller's mapping metadata the neighbour set
    # is unknown, so nothing can be classified canonical.
    assert vs.classify_projection_missing_cells(
        tmp_path, _PROJECTION, cell_names, combined_recipe, convolved_recipe,
    ) == set(cell_names)


def test_classify_projection_missing_cells_empty_store_marks_everything_missing(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe()
    cell_names = [f"skycell.{_PROJECTION.split('.')[-1]}.{i:03d}" for i in range(5)]
    missing = vs.classify_projection_missing_cells(
        tmp_path, _PROJECTION, cell_names, combined_recipe, convolved_recipe,
        metadata=_row_metadata(cell_names),
    )
    assert missing == set(cell_names)


# ---------------------------------------------------------------------------
# Part B step 3a: neighbor-finding + single-skycell convolution
# ---------------------------------------------------------------------------


def test_find_projection_neighbors_interior_cell_has_8_neighbors():
    df = _make_grid_df(_PROJECTION, rows=3, cols=3)
    metadata = pp.extract_projection_metadata(df, _PROJECTION)
    proj_id = _PROJECTION.split(".")[-1]
    center = f"skycell.{proj_id}.11"
    neighbors = pp._find_projection_neighbors(metadata, center, row_id=1, x_coord=1)
    names = {n for n, _, _ in neighbors}
    expected = {f"skycell.{proj_id}.{r}{c}" for r in range(3) for c in range(3)} - {center}
    assert names == expected
    assert len(neighbors) == 8


def test_find_projection_neighbors_edge_cell_has_fewer_neighbors():
    df = _make_grid_df(_PROJECTION, rows=3, cols=3)
    metadata = pp.extract_projection_metadata(df, _PROJECTION)
    proj_id = _PROJECTION.split(".")[-1]
    # Top-left corner cell: only 3 neighbors exist (right, below, below-right).
    corner = f"skycell.{proj_id}.00"
    neighbors = pp._find_projection_neighbors(metadata, corner, row_id=0, x_coord=0)
    names = {n for n, _, _ in neighbors}
    assert names == {f"skycell.{proj_id}.01", f"skycell.{proj_id}.10", f"skycell.{proj_id}.11"}


def test_convolve_single_skycell_matches_whole_mosaic_convolution():
    """convolve_single_skycell must reproduce the convolution of the whole
    continuous sky (v2 rewrite: the old test stitched a no-overlap 3x3 mosaic,
    which is not the PS1 geometry; neighbouring cells overlap by CELL_OVERLAP=480
    px and the canonical cell uses the fixed 480 px pad, so cells are 1000 px).

    The synthetic sky is one continuous random image; cell (r, c) is the slice
    that PS1's geometry gives it (stride W - 480 in both axes), so overlapping
    pixels agree exactly. Convolving the whole sky and cropping the centre cell
    is the ground truth the canonical path has to match."""
    from syndiff_pipeline.template_creation.processing import convolution_utils

    cell_size = 1000
    stride = cell_size - canonical_cell.CELL_OVERLAP
    radius = 15
    sigma = 5.0
    df = _make_grid_df(_PROJECTION, rows=3, cols=3, cell_size=cell_size)
    metadata = pp.extract_projection_metadata(df, _PROJECTION)
    proj_id = _PROJECTION.split(".")[-1]

    rng = np.random.default_rng(7)
    total = 2 * stride + cell_size
    sky = rng.random((total, total)).astype(np.float32)
    bundles = {}
    for r in range(3):
        for c in range(3):
            img = sky[r * stride:r * stride + cell_size, c * stride:c * stride + cell_size].copy()
            bundles[f"skycell.{proj_id}.{r}{c}"] = {
                "combined_image": img,
                "combined_mask": np.zeros((cell_size, cell_size), dtype=np.uint16),
                "headers_data": {},
                "removed_stars": [],
            }

    center_name = f"skycell.{proj_id}.11"
    result = pp.convolve_single_skycell(
        center_name, row_id=1, x_coord=1, projection=_PROJECTION, metadata=metadata,
        fetch_cell=bundles.get, psf_sigma=sigma, radius=radius,
    )
    assert result is not None
    assert result["combined_image"].shape == (cell_size, cell_size)

    convolved_sky = convolution_utils.apply_gaussian_convolution(sky, sigma=sigma, radius=radius, cval=0.0)
    expected_center = convolved_sky[stride:stride + cell_size, stride:stride + cell_size]

    np.testing.assert_allclose(result["combined_image"], expected_center, atol=1e-4)


def test_convolve_single_skycell_returns_none_on_center_miss():
    df = _make_grid_df(_PROJECTION, rows=1, cols=1, cell_size=16)
    metadata = pp.extract_projection_metadata(df, _PROJECTION)
    result = pp.convolve_single_skycell(
        "skycell.1234.00", row_id=0, x_coord=0, projection=_PROJECTION, metadata=metadata,
        fetch_cell=lambda name: None, psf_sigma=3.0, radius=8,
    )
    assert result is None


def test_convolve_single_skycell_missing_neighbor_returns_none():
    """New v2 behaviour (was: "missing neighbour stays NaN, centre still
    convolves"): the canonical cell is all-or-nothing. A neighbour that is not
    available would silently change the pixels under an unchanged fingerprint,
    so the cell is not computed and the caller falls back to the dense path."""
    df = _make_grid_df(_PROJECTION, rows=1, cols=2, cell_size=1000)
    metadata = pp.extract_projection_metadata(df, _PROJECTION)
    center_bundle = _combined_bundle(seed=0, size=1000)

    def fetch_cell(name):
        if name.endswith(".00"):
            return center_bundle
        return None  # right neighbor "01" is a cold miss

    result = pp.convolve_single_skycell(
        "skycell.1234.00", row_id=0, x_coord=0, projection=_PROJECTION, metadata=metadata,
        fetch_cell=fetch_cell, psf_sigma=5.0, radius=15,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Part B step 3a/3b: process_sparse_projection driver
# ---------------------------------------------------------------------------


def test_process_sparse_projection_publishes_when_all_cells_combined(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe(radius=15, psf_sigma=5.0)
    # 1000 px cells: the canonical cell uses the fixed 480 px pad / overlap geometry.
    df = _make_grid_df(_PROJECTION, rows=3, cols=3, cell_size=1000)
    metadata = pp.extract_projection_metadata(df, _PROJECTION)
    proj_id = _PROJECTION.split(".")[-1]

    # Publish combined (but not convolved) data for every cell in the grid.
    for r in range(3):
        for c in range(3):
            _publish_combined_only(tmp_path, _PROJECTION, f"{r}{c}", combined_recipe, size=1000)

    missing_cells = {f"skycell.{proj_id}.11"}
    published = pp.process_sparse_projection(
        _PROJECTION, metadata, missing_cells, {}, str(tmp_path),
        combined_recipe, convolved_recipe, psf_sigma=5.0,
    )
    assert published == missing_cells

    # Now canonical under this exact recipe chain (own combined fp + neighbour set).
    assert vs.skycell_already_canonical(
        tmp_path, _PROJECTION, "11", combined_recipe, convolved_recipe, metadata=metadata,
    ) is True


def test_process_sparse_projection_falls_back_on_cold_cell(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe(radius=8, psf_sigma=3.0)
    df = _make_grid_df(_PROJECTION, rows=3, cols=3, cell_size=20)
    metadata = pp.extract_projection_metadata(df, _PROJECTION)
    # Nothing published anywhere -- the center cell itself is a cold miss.
    missing_cells = {f"skycell.{_PROJECTION.split('.')[-1]}.11"}
    published = pp.process_sparse_projection(
        _PROJECTION, metadata, missing_cells, {}, str(tmp_path),
        combined_recipe, convolved_recipe, psf_sigma=3.0,
    )
    assert published is None


# ---------------------------------------------------------------------------
# Cold-start guarantee: classification never fabricates canonical cells
# ---------------------------------------------------------------------------


def test_cold_start_collapses_to_all_missing_no_sparse_path():
    """With nothing published anywhere, every cell in a projection is
    classified as missing, so the count-based cut always routes to the
    dense/full-loop path (i.e. behaves exactly like the pre-existing
    pipeline, per the plan's cold-start guarantee)."""
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe()
    proj_id = "9999"
    cell_names = [f"skycell.{proj_id}.{i:03d}" for i in range(20)]
    missing = vs.classify_projection_missing_cells(
        f"/tmp/does-not-exist-{proj_id}", f"skycell.{proj_id}", cell_names,
        combined_recipe, convolved_recipe, metadata=_row_metadata(cell_names),
    )
    assert missing == set(cell_names)
    assert len(missing) > 5  # exceeds MISSING_CELL_THRESHOLD -> dense path
