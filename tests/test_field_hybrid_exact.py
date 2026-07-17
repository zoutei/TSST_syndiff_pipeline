"""Unit tests for L4a Exact helpers and L4b abutting id sets."""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
    abutting_border_tess_ids,
    candidate_tess_ids_for_l4a,
    shared_abutting_border_tess_ids,
)
from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
    build_l4a_hybrid_assignment,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    verify_field_store,
    write_contrib,
    write_template_manifest,
    FieldManifest,
)


def test_candidate_tess_ids_cover_l4a_mask():
    rng = np.random.default_rng(0)
    frozen = rng.integers(-1, 50, size=(32, 32), dtype=np.int32)
    frozen[0, :] = -1
    tids, mask = candidate_tess_ids_for_l4a(frozen, 2, -1, hybrid_R=1)
    assert mask.any()
    assert tids.size > 0
    linear, _ = build_l4a_hybrid_assignment(frozen, 2, -1, exact_tid=None, hybrid_R=1)
    assert set(np.unique(linear[mask])).issubset(set(tids.tolist()) | {-1})


def test_abutting_border_ids_nonzero():
    master = np.zeros((8, 8), dtype=np.int32)
    master[:, :4] = 1
    master[:, 4:] = 2
    ids = abutting_border_tess_ids(master, 1)
    assert ids.size > 0
    # All returned ids belong to skycell 1
    t_x = master.shape[1]
    for tid in ids:
        y, x = divmod(int(tid), t_x)
        assert master[y, x] == 1


def test_shared_abutting_border_both_sides():
    master = np.zeros((6, 6), dtype=np.int32)
    master[:, :3] = 10
    master[:, 3:] = 20
    a_ids, b_ids = shared_abutting_border_tess_ids(master, 10, 20)
    assert a_ids.size > 0 and b_ids.size > 0
    t_x = master.shape[1]
    for tid in a_ids:
        y, x = divmod(int(tid), t_x)
        assert master[y, x] == 10
    for tid in b_ids:
        y, x = divmod(int(tid), t_x)
        assert master[y, x] == 20


def test_verify_field_store_require_nonempty(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "contribs").mkdir()
    write_template_manifest(
        store,
        FieldManifest(
            geometry_mode="field",
            scope="scc",
            assembly="sparse_sum",
            materialize_fits=False,
            sector=20,
            camera=3,
            ccd=3,
            contribs_dir="contribs",
            groups=[],
        ),
    )
    write_contrib(
        store,
        "skycell.0000.000",
        1,
        0,
        indices=np.array([], dtype=np.int64),
        flux_sum=np.array([], dtype=np.float64),
        count=np.array([], dtype=np.float64),
        mask_count=np.array([], dtype=np.float64),
    )
    ok = verify_field_store(
        store,
        required_keys=[("skycell.0000.000", 1, 0)],
        require_nonempty=False,
    )
    assert ok["ok"]
    bad = verify_field_store(
        store,
        required_keys=[("skycell.0000.000", 1, 0)],
        require_nonempty=True,
    )
    assert not bad["ok"]
    assert bad["empty_contribs"]
