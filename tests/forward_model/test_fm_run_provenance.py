# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
import hashlib
import json

from syndiff_pipeline.forward_model import train_from_bundle as TFB


def test_run_provenance_records_bundle_code_and_host(tmp_path):
    bundle = tmp_path / "fit_bundle.npz"
    bundle.write_bytes(b"not really a bundle")
    TFB._append_run_provenance(tmp_path, bundle, ["--from-bundle", str(bundle)])
    TFB._append_run_provenance(tmp_path, bundle, ["--stage", "3"])
    lines = (tmp_path / "run_provenance.jsonl").read_text().splitlines()
    assert len(lines) == 2  # appended, not overwritten
    rec = json.loads(lines[0])
    assert rec["bundle_path"] == str(bundle.resolve())
    assert rec["bundle_sha256"] == hashlib.sha256(b"not really a bundle").hexdigest()
    assert len(rec["git_commit"]) == 40
    assert isinstance(rec["git_dirty_files"], list)
    assert rec["host"] and rec["argv"] == ["--from-bundle", str(bundle)]
    assert rec["epsf_repr"] == "pixel_integrated_v1"


def test_run_provenance_never_raises_on_missing_bundle(tmp_path):
    TFB._append_run_provenance(tmp_path, tmp_path / "missing.npz", [])
    rec = json.loads((tmp_path / "run_provenance.jsonl").read_text())
    assert rec["bundle_sha256"].startswith("unreadable")
