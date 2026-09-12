"""Raw-data immutability (research invariant #8)."""

import json
from pathlib import Path

import pytest

from wsr.utils import integrity
from wsr.utils.paths import DATA_RAW, MANIFESTS


def test_hash_tree_ignores_bookkeeping_files_and_is_stable(fake_raw_tree: Path):
    hashes = integrity.hash_tree(fake_raw_tree)
    assert set(hashes) == {"S2/signal.bin", "S3/signal.bin"}
    assert hashes == integrity.hash_tree(fake_raw_tree)


def test_snapshot_then_verify_detects_modification(fake_raw_tree: Path, tmp_path: Path):
    manifests = tmp_path / "manifests"
    out = integrity.snapshot(fake_raw_tree, manifests)
    assert out.name == "raw_checksums_fakeset.json"
    assert json.loads(out.read_text())["n_files"] == 2
    assert integrity.verify(fake_raw_tree, manifests) == {"added": [], "removed": [], "modified": []}

    (fake_raw_tree / "S2" / "signal.bin").write_bytes(b"tampered")
    (fake_raw_tree / "S3" / "signal.bin").unlink()
    (fake_raw_tree / "S4").mkdir()
    (fake_raw_tree / "S4" / "signal.bin").write_bytes(b"new")
    diff = integrity.verify(fake_raw_tree, manifests)
    assert diff == {"added": ["S4/signal.bin"], "removed": ["S3/signal.bin"], "modified": ["S2/signal.bin"]}


@pytest.mark.parametrize("dataset", ["wesad", "nurse", "stress_predict"])
def test_real_raw_tree_matches_manifest(dataset: str):
    """Runs only once a dataset has been downloaded and snapshotted."""
    dataset_dir = DATA_RAW / dataset
    manifest = integrity.manifest_path(dataset_dir, MANIFESTS)
    if not manifest.is_file():
        pytest.skip(f"no checksum manifest for {dataset} yet (run: python -m wsr.utils.integrity snapshot {dataset_dir})")
    diff = integrity.verify(dataset_dir, MANIFESTS)
    assert diff == {"added": [], "removed": [], "modified": []}, f"raw {dataset} tree changed: {diff}"
