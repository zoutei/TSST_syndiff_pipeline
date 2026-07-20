"""Correctness gate for the PR5 shared-convolved-store padding decouple fix
(``doc/template_bookkeeping_plan.md`` §13, decision #3).

Background / the bug this guards against
-----------------------------------------
The originally-landed ``process_coordinator._publish_convolved`` (PR5,
commit 4e05b89) convolved a single *isolated* skycell with zero row/neighbor
context and published that as the "same_projection_only" canonical cell.
That was architecturally wrong: the real canonical cell must be convolved on
the same-projection-padded ROW master array, which only exists inside
``process_row_step_from_queue`` (a different function, on a different
thread) -- specifically at the point right after ``apply_cross_row_padding``
runs and before ``apply_cross_projection_padding`` runs. That buggy function
has been removed entirely (see ``test_process_coordinator_convolved_store_kwargs_removed``
below); the corrected logic now lives in
``ps1_process._publish_canonical_convolved_snapshot``, called from
``process_row_step_from_queue`` between steps 3 and 4.

What this module proves
------------------------
1. ``test_canonical_snapshot_matches_main_path_when_no_cross_projection_padding``:
   the actual mathematical property the fix depends on. With
   ``csv_path=None`` (zero cross-projection padding sources -- the simplest
   way to guarantee the precondition), the row master array's content at the
   snapshot point (after cross-row padding, before cross-projection padding)
   is *identical* to what it will be when the unchanged main path (step 5)
   convolves it moments later, because step 4 is skipped entirely. So the
   shared-store-published canonical cell must numerically match
   ``extract_cell_results`` on the main path's own output for that same
   cell. Verified against real production code (``process_row_step_from_queue``
   itself, not a reimplementation), with real (non-mocked)
   ``combined_store``/``convolved_store`` publish/load round-trips.

2. ``test_shared_convolved_store_disabled_by_default_never_invoked``: with
   ``convolved_store_recipe=None`` (what ``run_modern_sliding_window_pipeline``
   passes whenever ``use_shared_convolved_store=False``, its default), the
   new snapshot/publish function is never entered -- proving the default
   pipeline path's behavior, output, and call graph are unaffected by this
   change.

3. ``test_process_coordinator_convolved_store_kwargs_removed``: proves the
   old, wrong, isolated-cell convolution path is not merely dead code but
   has been fully removed from ``process_coordinator`` -- calling it with
   the old ``convolved_store_data_root``/``convolved_store_recipe``/
   ``psf_sigma`` kwargs now raises ``TypeError``, so it cannot silently
   resurrect and publish scientifically wrong cells under any config.

4. ``test_snapshot_skips_cell_with_no_published_combined_record``: the
   snapshot function must never fabricate a ``combined_fingerprint`` edge to
   a combined-store record that was never actually published -- a cell with
   no matching on-disk combined-store entry is silently skipped (best
   effort), not published with a dangling/incorrect provenance pointer.

Note on scope: this is a synthetic, reproducible, no-real-data test. It
proves the snapshot hook computes the right thing under the precondition it
requires. It does **not** replace the plan's real-SCC blocking
numeric-equivalence gate (§13, §17) -- see ``convolved_gate.py`` and the
report for that.
"""

from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import convolved_store as cvs
from syndiff_pipeline.template_creation.processing import ps1_process as pp


def _gaussian_image(size: int, cx: float, cy: float, amp: float, sigma: float) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    data = np.full((size, size), 5.0, dtype=np.float32)
    data += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return data.astype(np.float32)


def _two_cell_row_df() -> pd.DataFrame:
    """A single projection, single row, two side-by-side skycells.

    Columns match what ``extract_projection_metadata`` (ps1_process.py)
    expects from a loaded CSV DataFrame.
    """
    return pd.DataFrame(
        [
            {"projection": "skycell.9999", "y": 0, "NAME": "skycell.9999.001", "x": 0, "NAXIS1": 520, "NAXIS2": 520},
            {"projection": "skycell.9999", "y": 0, "NAME": "skycell.9999.002", "x": 1, "NAXIS1": 520, "NAXIS2": 520},
        ]
    )


def _bundle(skycell_id: str, x_coord: int, image: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "skycell_id": skycell_id,
        "projection": "skycell.9999",
        "row_id": 0,
        "x_coord": x_coord,
        "combined_image": image,
        "combined_mask": mask,
        "headers_data": {"r": f"HEADER-{skycell_id}"},
        "removed_stars": [],
    }


class CanonicalSnapshotMathTests(unittest.TestCase):
    """Proves the actual mathematical property (item 1 in the module docstring)."""

    def _run(self, *, publish_combined: bool, data_root: Path):
        df = _two_cell_row_df()
        metadata = pp.extract_projection_metadata(df, "skycell.9999")
        config = pp.create_master_array_config(metadata)
        state = pp.initialize_processing_state(config)

        img1 = _gaussian_image(520, 260, 260, amp=500.0, sigma=30.0)
        img2 = _gaussian_image(520, 260, 260, amp=300.0, sigma=30.0)
        mask1 = np.zeros((520, 520), dtype=np.uint16)
        mask2 = np.zeros((520, 520), dtype=np.uint16)

        combined_recipe = cs.combined_recipe(gaia_version="none")
        if publish_combined:
            for cell_id, img, mask in (
                ("skycell.9999.001", img1, mask1),
                ("skycell.9999.002", img2, mask2),
            ):
                projection, cell = cs._projection_and_cell(cell_id)
                raw_fp = cs.raw_skycell_input_fingerprint(data_root, projection, cell)
                info = cs.publish_combined_cell(
                    data_root,
                    projection,
                    cell,
                    combined_image=img,
                    combined_mask=mask,
                    headers_data={"r": f"HEADER-{cell_id}"},
                    removed_stars=[],
                    recipe=combined_recipe,
                    input_fingerprints=[raw_fp],
                    producer="test",
                )
                self.assertIsNotNone(info, f"setup: failed to publish combined cell {cell_id}")

        cell_queue: queue.Queue = queue.Queue()
        cell_queue.put(_bundle("skycell.9999.001", 0, img1, mask1))
        cell_queue.put(_bundle("skycell.9999.002", 1, img2, mask2))

        convolved_recipe = cvs.convolved_recipe(psf_sigma=20.0)

        results_data, results_masks, _removed = pp.process_row_step_from_queue(
            state,
            config,
            metadata,
            current_row_id=0,
            next_row_id=None,
            combined_cell_queue=cell_queue,
            cell_buffer={},
            psf_sigma=20.0,
            ingest_config={},
            projection="skycell.9999",
            catalog=None,
            enable_saturation_correction=False,
            remove_saturated_stars=False,
            # Precondition: zero cross-projection padding sources. With
            # csv_path=None, step 4 (apply_cross_projection_padding) is
            # skipped entirely -- the array snapshotted at step 3b is
            # therefore *exactly* the array the unchanged main path (step 5)
            # goes on to convolve, with nothing in between mutating it.
            csv_path=None,
            pipeline_paused_event=None,
            band_cache=None,
            band_cache_uses=None,
            row_padding_map=None,
            bright_star_mag_threshold=13.0,
            convolved_store_data_root=str(data_root),
            combined_store_recipe=combined_recipe,
            convolved_store_recipe=convolved_recipe,
        )
        return results_data

    def test_canonical_snapshot_matches_main_path_when_no_cross_projection_padding(self) -> None:
        """The real correctness proof.

        Tolerance: exact equality (``np.testing.assert_array_equal``).
        Reasoning -- both the shared-store publish and the main path's
        returned ``results_data`` are produced by the exact same function
        (``convolution_utils.apply_gaussian_convolution``) called on
        independent-but-value-identical float32 copies of the exact same
        row master array content (proven identical because step 4 is a
        true no-op here, not merely "not visibly different"). Empirically
        confirmed bit-identical (max abs diff == 0.0) during development;
        an exact-equality assertion is the tightest, most honest check of
        that empirical fact. If this ever becomes flaky due to
        thread/chunking non-associativity in dask's Gaussian filter, that
        would itself be a real finding worth investigating, not something
        to paper over with a loose tolerance.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            results_data = self._run(publish_combined=True, data_root=data_root)

            shared_root = data_root / "ps1_skycells_zarr" / "ps1_convolved.zarr"
            compared = 0
            for cell_id in ("skycell.9999.001", "skycell.9999.002"):
                projection, cell = cs._projection_and_cell(cell_id)
                cell_dir = shared_root / projection / cell
                self.assertTrue(
                    cell_dir.is_dir(),
                    f"expected a published shared-store cell dir at {cell_dir}",
                )
                fp_dirs = [d for d in cell_dir.iterdir() if d.is_dir()]
                self.assertEqual(
                    len(fp_dirs), 1,
                    f"expected exactly one fingerprint dir for {cell_id}, got {fp_dirs}",
                )
                loaded = cvs.try_load_convolved_cell(data_root, projection, cell, fp_dirs[0].name)
                self.assertIsNotNone(loaded, f"failed to load published cell {cell_id}")

                main_path_result = results_data[cell_id]
                np.testing.assert_array_equal(
                    loaded["convolved_image"],
                    main_path_result,
                    err_msg=(
                        f"shared-store canonical cell for {cell_id} does not exactly "
                        f"match the unchanged main path's own convolution output for "
                        f"the same cell under the zero-cross-projection-padding "
                        f"precondition"
                    ),
                )
                compared += 1
            self.assertEqual(compared, 2, "expected to compare both cells")

    def test_snapshot_skips_cell_with_no_published_combined_record(self) -> None:
        """Best-effort skip, never a dangling/fabricated combined_fingerprint edge.

        Same setup as above but the combined-store cells are never
        published (``publish_combined=False``). The snapshot function must
        not publish anything for these cells (no on-disk record exists to
        attribute the convolution to), and must not raise.
        """
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            self._run(publish_combined=False, data_root=data_root)

            shared_root = data_root / "ps1_skycells_zarr" / "ps1_convolved.zarr"
            self.assertFalse(
                shared_root.is_dir() and any(shared_root.rglob("arrays.npz")),
                "expected no shared convolved-store cell to be published when the "
                "upstream combined_skycell record was never actually published",
            )


class DefaultPathUnchangedTests(unittest.TestCase):
    """Proves item 2: use_shared_convolved_store=False touches nothing new."""

    def test_shared_convolved_store_disabled_by_default_never_invoked(self) -> None:
        df = _two_cell_row_df()
        metadata = pp.extract_projection_metadata(df, "skycell.9999")
        config = pp.create_master_array_config(metadata)
        state = pp.initialize_processing_state(config)

        img1 = _gaussian_image(520, 260, 260, amp=500.0, sigma=30.0)
        img2 = _gaussian_image(520, 260, 260, amp=300.0, sigma=30.0)
        mask1 = np.zeros((520, 520), dtype=np.uint16)
        mask2 = np.zeros((520, 520), dtype=np.uint16)

        cell_queue: queue.Queue = queue.Queue()
        cell_queue.put(_bundle("skycell.9999.001", 0, img1, mask1))
        cell_queue.put(_bundle("skycell.9999.002", 1, img2, mask2))

        # combined_store_recipe is set (mirrors production: it is built
        # unconditionally in run_modern_sliding_window_pipeline regardless
        # of use_shared_convolved_store) but convolved_store_recipe is None
        # -- exactly what the pipeline passes when
        # use_shared_convolved_store=False (its default).
        combined_recipe = cs.combined_recipe(gaia_version="none")

        with mock.patch.object(
            pp, "_publish_canonical_convolved_snapshot"
        ) as mocked_snapshot:
            results_data, results_masks, _removed = pp.process_row_step_from_queue(
                state,
                config,
                metadata,
                current_row_id=0,
                next_row_id=None,
                combined_cell_queue=cell_queue,
                cell_buffer={},
                psf_sigma=20.0,
                ingest_config={},
                projection="skycell.9999",
                catalog=None,
                enable_saturation_correction=False,
                remove_saturated_stars=False,
                csv_path=None,
                pipeline_paused_event=None,
                band_cache=None,
                band_cache_uses=None,
                row_padding_map=None,
                bright_star_mag_threshold=13.0,
                convolved_store_data_root="/does/not/matter",
                combined_store_recipe=combined_recipe,
                convolved_store_recipe=None,  # the use_shared_convolved_store=False case
            )

        mocked_snapshot.assert_not_called()
        self.assertEqual(set(results_data.keys()), {"skycell.9999.001", "skycell.9999.002"})

    def test_shared_convolved_store_default_kwargs_never_invoked(self) -> None:
        """Same proof again, but relying purely on the function's own
        defaults (no explicit convolved_store_* kwargs at all), which is
        exactly how every pre-existing caller of process_row_step_from_queue
        invokes it.
        """
        df = _two_cell_row_df()
        metadata = pp.extract_projection_metadata(df, "skycell.9999")
        config = pp.create_master_array_config(metadata)
        state = pp.initialize_processing_state(config)

        img1 = _gaussian_image(520, 260, 260, amp=500.0, sigma=30.0)
        mask1 = np.zeros((520, 520), dtype=np.uint16)
        img2 = _gaussian_image(520, 260, 260, amp=300.0, sigma=30.0)
        mask2 = np.zeros((520, 520), dtype=np.uint16)

        cell_queue: queue.Queue = queue.Queue()
        cell_queue.put(_bundle("skycell.9999.001", 0, img1, mask1))
        cell_queue.put(_bundle("skycell.9999.002", 1, img2, mask2))

        with mock.patch.object(
            pp, "_publish_canonical_convolved_snapshot"
        ) as mocked_snapshot:
            pp.process_row_step_from_queue(
                state,
                config,
                metadata,
                current_row_id=0,
                next_row_id=None,
                combined_cell_queue=cell_queue,
                cell_buffer={},
                psf_sigma=20.0,
                ingest_config={},
                projection="skycell.9999",
                catalog=None,
                enable_saturation_correction=False,
                remove_saturated_stars=False,
                csv_path=None,
            )

        mocked_snapshot.assert_not_called()


class OldIsolatedCellConvolutionRemovedTests(unittest.TestCase):
    """Proves item 3: the buggy path is fully removed, not just neutralized."""

    def test_process_coordinator_convolved_store_kwargs_removed(self) -> None:
        import inspect

        sig = inspect.signature(pp.process_coordinator)
        for stale_param in ("convolved_store_data_root", "convolved_store_recipe", "psf_sigma"):
            self.assertNotIn(
                stale_param,
                sig.parameters,
                f"process_coordinator should no longer accept {stale_param!r} -- the "
                f"isolated-cell convolved-store publish path must be fully removed, "
                f"not merely no-oped",
            )

    def test_process_coordinator_source_has_no_publish_convolved(self) -> None:
        import inspect

        source = inspect.getsource(pp.process_coordinator)
        self.assertNotIn("_publish_convolved", source)
        self.assertNotIn("publish_convolved_cell", source)


if __name__ == "__main__":
    unittest.main()
