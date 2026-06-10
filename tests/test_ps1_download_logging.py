"""Tests for ps1_download progress logging and skip/corruption checks."""
from __future__ import annotations

import logging
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import zarr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.processing import ps1_download


def _write_array(group, name: str, shape: tuple[int, int] = (4, 4)) -> None:
    if "mask" in name:
        group.create_array(name, shape=shape, dtype="u2")
    else:
        group.create_array(name, shape=shape, dtype="f4")


def _write_complete_skycell(root, skycell_name: str) -> None:
    projection_id = skycell_name.split(".")[1]
    if projection_id not in root:
        root.create_group(projection_id)
    group = root[projection_id].create_group(skycell_name)
    for array_name in ps1_download.expected_array_names():
        _write_array(group, array_name)


def _make_writer(tmpdir: str) -> tuple[zarr.Group, Path, ps1_download.ZarrWriter]:
    zarr_path = Path(tmpdir) / "ps1_skycells.zarr"
    lock_file = zarr_path.parent / "ps1_skycells.zarr.lock"
    root = zarr.open(str(zarr_path), mode="w")
    writer = ps1_download.ZarrWriter(root, lock_file)
    return root, lock_file, writer


class TestSkycellProgress(unittest.TestCase):
    def test_concurrent_increments(self):
        progress = ps1_download.SkycellProgress(total=100)
        seen_finished: list[int] = []

        def worker():
            for _ in range(10):
                progress.mark_started()
                finished, total = progress.mark_finished()
                seen_finished.append(finished)
                self.assertEqual(total, 100)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(seen_finished), 50)
        self.assertEqual(max(seen_finished), 50)


class TestIsArrayComplete(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        zarr_path = Path(self.tmpdir.name) / "ps1_skycells.zarr"
        self.lock_file = zarr_path.parent / "ps1_skycells.zarr.lock"
        self.root = zarr.open(str(zarr_path), mode="w")
        self.skycell_name = "skycell.1520.080"
        self.projection_id = "1520"

    def test_missing_array_is_incomplete(self):
        self.assertFalse(
            ps1_download.is_array_complete(
                self.root, self.projection_id, self.skycell_name, "r", self.lock_file
            )
        )

    def test_complete_array_passes(self):
        group = self.root.create_group(self.projection_id).create_group(self.skycell_name)
        _write_array(group, "r")
        self.assertTrue(
            ps1_download.is_array_complete(
                self.root, self.projection_id, self.skycell_name, "r", self.lock_file
            )
        )

    def test_empty_shape_is_incomplete(self):
        group = self.root.create_group(self.projection_id).create_group(self.skycell_name)
        group.create_array("r", shape=(0,), dtype="f4")
        self.assertFalse(
            ps1_download.is_array_complete(
                self.root, self.projection_id, self.skycell_name, "r", self.lock_file
            )
        )

    def test_overwrite_forces_incomplete(self):
        group = self.root.create_group(self.projection_id).create_group(self.skycell_name)
        _write_array(group, "r")
        self.assertFalse(
            ps1_download.is_array_complete(
                self.root,
                self.projection_id,
                self.skycell_name,
                "r",
                self.lock_file,
                overwrite=True,
            )
        )

    def test_count_complete_arrays(self):
        group = self.root.create_group(self.projection_id).create_group(self.skycell_name)
        _write_array(group, "r")
        _write_array(group, "i")
        count = ps1_download.count_complete_arrays(
            self.root, self.projection_id, self.skycell_name, self.lock_file
        )
        self.assertEqual(count, 2)


class TestStoreSkycellBatch(unittest.TestCase):
    def test_store_skycell_batch_single_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root, lock_file, _writer = _make_writer(tmpdir)
            skycell_name = "skycell.1520.080"
            projection_id = "1520"
            shape = (8, 8)
            items = [
                ps1_download.ArrayWriteItem(
                    projection_id=projection_id,
                    skycell_name=skycell_name,
                    array_name=name,
                    band=name.split("_")[0] if "_" in name else name,
                    data=np.ones(shape, dtype=np.float32),
                    header="TEST",
                )
                for name in ("r", "i", "z")
            ]

            lock_calls = 0

            class CountingLock:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    nonlocal lock_calls
                    lock_calls += 1
                    return self

                def __exit__(self, *args):
                    return False

            with mock.patch.object(ps1_download, "FileLock", CountingLock):
                ps1_download.store_skycell_batch(
                    root, projection_id, skycell_name, items, lock_file
                )

            self.assertEqual(lock_calls, 1)
            group = root[projection_id][skycell_name]
            for name in ("r", "i", "z"):
                self.assertIn(name, group)
                self.assertEqual(group[name].shape, shape)


class TestZarrWriter(unittest.TestCase):
    def test_drains_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root, lock_file, writer = _make_writer(tmpdir)
            skycell_a = "skycell.1520.080"
            skycell_b = "skycell.1520.081"
            shape = (4, 4)

            def make_items(skycell_name: str, band: str) -> list[ps1_download.ArrayWriteItem]:
                projection_id = skycell_name.split(".")[1]
                return [
                    ps1_download.ArrayWriteItem(
                        projection_id=projection_id,
                        skycell_name=skycell_name,
                        array_name=band,
                        band=band,
                        data=np.full(shape, 1.0, dtype=np.float32),
                        header="HDR",
                    )
                ]

            self.assertTrue(writer.submit_batch(make_items(skycell_a, "r")))
            self.assertTrue(writer.submit_batch(make_items(skycell_b, "i")))
            writer.close()

            self.assertIn("r", root["1520"][skycell_a])
            self.assertIn("i", root["1520"][skycell_b])
            self.assertFalse(writer._thread.is_alive())

    def test_bounded_backpressure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root, lock_file, writer = _make_writer(tmpdir)
            self.assertEqual(writer._queue.maxsize, ps1_download._WRITE_QUEUE_MAXSIZE)

            block_entered = threading.Event()
            release_write = threading.Event()

            def slow_store(*args, **kwargs):
                block_entered.set()
                release_write.wait(timeout=5)

            try:
                with mock.patch.object(ps1_download, "store_skycell_batch", side_effect=slow_store):
                    item = ps1_download.ArrayWriteItem(
                        projection_id="1520",
                        skycell_name="skycell.1520.080",
                        array_name="r",
                        band="r",
                        data=np.ones((4, 4), dtype=np.float32),
                        header="HDR",
                    )
                    submitter = threading.Thread(
                        target=writer.submit_batch,
                        args=([item],),
                    )
                    submitter.start()
                    self.assertTrue(block_entered.wait(timeout=5))

                    filled = 1
                    for idx in range(ps1_download._WRITE_QUEUE_MAXSIZE):
                        batch = ps1_download._WriteBatch(
                            items=[
                                ps1_download.ArrayWriteItem(
                                    projection_id="1520",
                                    skycell_name=f"skycell.1520.{idx:03d}",
                                    array_name="r",
                                    band="r",
                                    data=np.ones((4, 4), dtype=np.float32),
                                    header="HDR",
                                )
                            ]
                        )
                        writer._queue.put(batch, timeout=1)
                        filled += 1

                    with self.assertRaises(queue.Full):
                        writer._queue.put(ps1_download._WriteBatch(items=[item]), timeout=0.01)

                    release_write.set()
                    submitter.join(timeout=5)
            finally:
                release_write.set()
                writer.close()


class TestDownloadAndStoreSkycellLogging(unittest.TestCase):
    def test_skip_logs_complete_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root, lock_file, writer = _make_writer(tmpdir)
            skycell_name = "skycell.1520.080"
            _write_complete_skycell(root, skycell_name)
            progress = ps1_download.SkycellProgress(total=1)

            try:
                with self.assertLogs(level=logging.INFO) as logs:
                    ps1_download.download_and_store_skycell(
                        root, skycell_name, lock_file, writer, progress=progress
                    )

                messages = "\n".join(logs.output)
                self.assertIn("Skipping skycell skycell.1520.080 (12/12 arrays complete in zarr)", messages)
                self.assertIn("Finished skycell skycell.1520.080 (1/1)", messages)
            finally:
                writer.close()

    def test_process_logs_start_and_finish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root, lock_file, writer = _make_writer(tmpdir)
            skycell_name = "skycell.1520.080"
            progress = ps1_download.SkycellProgress(total=1)

            fake_result = {"data": np.ones((4, 4), dtype=np.float32), "header": "HDR"}

            try:
                with mock.patch.object(
                    ps1_download, "download_and_process_band", return_value=fake_result
                ):
                    with self.assertLogs(level=logging.INFO) as logs:
                        ps1_download.download_and_store_skycell(
                            root, skycell_name, lock_file, writer, progress=progress
                        )

                messages = "\n".join(logs.output)
                self.assertIn(
                    "Processing skycell skycell.1520.080 (1/1; 0/12 arrays already in zarr)",
                    messages,
                )
                self.assertIn("Finished skycell skycell.1520.080 (1/1)", messages)
                self.assertIn("Stored 12 arrays for skycell.1520.080 in zarr", messages)
            finally:
                writer.close()


class TestProcessSkycellsWithDask(unittest.TestCase):
    def test_uses_threaded_scheduler(self):
        root = mock.Mock()
        lock_file = Path("/tmp/ps1_skycells.zarr.lock")
        fake_bag = mock.Mock()
        fake_computation = mock.Mock()
        fake_bag.map.return_value = fake_computation
        fake_computation.compute.return_value = [1, 1]

        with mock.patch.object(ps1_download, "ZarrWriter") as mock_writer_cls:
            mock_writer = mock_writer_cls.return_value
            with mock.patch.object(ps1_download.db, "from_sequence", return_value=fake_bag):
                with mock.patch.object(ps1_download.sys.stdout, "isatty", return_value=False):
                    ps1_download.process_skycells_with_dask(
                        root, ["skycell.1520.080", "skycell.1520.081"], lock_file, num_workers=4
                    )

        fake_computation.compute.assert_called_once_with(scheduler="threads", num_workers=4)
        mock_writer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
