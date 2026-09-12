"""Raw-data immutability (research invariant #8).

Covers the checksum-baseline workflow in wsr.utils.integrity:
first snapshot, refusal to overwrite, detection of modified / deleted / added
files, protection of dataset-supplied documentation, and the explicit
replace-baseline opt-in.
"""

import json
from pathlib import Path

import pytest

from wsr.utils import integrity
from wsr.utils.integrity import BaselineExistsError
from wsr.utils.paths import DATA_RAW, MANIFESTS


@pytest.fixture
def raw(fake_raw_tree: Path, tmp_path: Path):
    """(dataset_dir, manifests_dir) for a fake dataset with a fresh manifests folder."""
    return fake_raw_tree, tmp_path / "manifests"


def _unchanged():
    return {"added": [], "removed": [], "modified": []}


# A. first snapshot succeeds ------------------------------------------------

def test_first_snapshot_writes_deterministic_manifest(raw):
    dataset_dir, manifests = raw
    out = integrity.snapshot(dataset_dir, manifests)
    assert out == manifests / "raw_checksums_fakeset.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["root"] == "fakeset"
    assert payload["n_files"] == len(payload["files"])
    # deterministic ordering: keys sorted, and byte-identical on re-hash
    assert list(payload["files"]) == sorted(payload["files"])
    assert payload["files"] == integrity.hash_tree(dataset_dir)
    assert integrity.verify(dataset_dir, manifests) == _unchanged()


def test_snapshot_refuses_empty_tree(tmp_path: Path):
    empty = tmp_path / "raw" / "empty"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        integrity.snapshot(empty, tmp_path / "manifests")


# B. ordinary second snapshot refuses to overwrite --------------------------

def test_second_snapshot_refuses_and_leaves_baseline_untouched(raw):
    dataset_dir, manifests = raw
    out = integrity.snapshot(dataset_dir, manifests)
    original = out.read_bytes()

    (dataset_dir / "S2" / "signal.bin").write_bytes(b"tampered")
    with pytest.raises(BaselineExistsError, match="already exists"):
        integrity.snapshot(dataset_dir, manifests)

    assert out.read_bytes() == original, "refused snapshot must not touch the baseline"
    # the tampering is still visible against the original baseline
    assert integrity.verify(dataset_dir, manifests)["modified"] == ["S2/signal.bin"]


def test_cli_snapshot_refuses_with_nonzero_exit(raw, capsys):
    dataset_dir, manifests = raw
    assert integrity._main(["snapshot", str(dataset_dir), "--manifests-dir", str(manifests)]) == 0
    assert integrity._main(["snapshot", str(dataset_dir), "--manifests-dir", str(manifests)]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert integrity._main(["verify", str(dataset_dir), "--manifests-dir", str(manifests)]) == 0


# C / D / E. verification detects modified / deleted / added --------------

def test_verify_detects_modified_file(raw):
    dataset_dir, manifests = raw
    integrity.snapshot(dataset_dir, manifests)
    (dataset_dir / "S2" / "signal.bin").write_bytes(b"tampered")
    diff = integrity.verify(dataset_dir, manifests)
    assert diff == {"added": [], "removed": [], "modified": ["S2/signal.bin"]}
    assert not integrity.is_unchanged(diff)


def test_verify_detects_deleted_file(raw):
    dataset_dir, manifests = raw
    integrity.snapshot(dataset_dir, manifests)
    (dataset_dir / "S3" / "signal.bin").unlink()
    diff = integrity.verify(dataset_dir, manifests)
    assert diff == {"added": [], "removed": ["S3/signal.bin"], "modified": []}


def test_verify_detects_added_file(raw):
    dataset_dir, manifests = raw
    integrity.snapshot(dataset_dir, manifests)
    (dataset_dir / "S4").mkdir()
    (dataset_dir / "S4" / "signal.bin").write_bytes(b"new")
    diff = integrity.verify(dataset_dir, manifests)
    assert diff == {"added": ["S4/signal.bin"], "removed": [], "modified": []}


def test_verify_without_baseline_raises(raw):
    dataset_dir, manifests = raw
    with pytest.raises(FileNotFoundError, match="No checksum baseline"):
        integrity.verify(dataset_dir, manifests)


# F. dataset-supplied documentation is protected ---------------------------

def test_dataset_documentation_is_in_baseline_and_protected(raw):
    dataset_dir, manifests = raw
    out = integrity.snapshot(dataset_dir, manifests)
    files = json.loads(out.read_text(encoding="utf-8"))["files"]
    assert "README.md" in files, "dataset-shipped README must be hashed"
    assert "docs/description.txt" in files

    (dataset_dir / "README.md").write_text("edited dataset readme", encoding="utf-8")
    assert integrity.verify(dataset_dir, manifests)["modified"] == ["README.md"]


def test_only_top_level_gitkeep_is_excluded(raw):
    dataset_dir, manifests = raw
    (dataset_dir / "S2" / ".gitkeep").write_text("not a placeholder here", encoding="utf-8")
    files = integrity.hash_tree(dataset_dir)
    assert ".gitkeep" not in files, "repo placeholder at dataset root is excluded"
    assert "S2/.gitkeep" in files, "same name deeper in the tree is dataset content"


# G. explicit replacement -------------------------------------------------

def test_replace_requires_explicit_opt_in_and_then_replaces(raw):
    dataset_dir, manifests = raw
    out = integrity.snapshot(dataset_dir, manifests)
    (dataset_dir / "S2" / "signal.bin").write_bytes(b"new release")

    with pytest.raises(BaselineExistsError):
        integrity.snapshot(dataset_dir, manifests)               # default: refuse
    with pytest.raises(BaselineExistsError):
        integrity.snapshot(dataset_dir, manifests, replace=False)  # explicit False: refuse

    replaced = integrity.snapshot(dataset_dir, manifests, replace=True)
    assert replaced == out
    assert integrity.verify(dataset_dir, manifests) == _unchanged()
    assert json.loads(out.read_text(encoding="utf-8"))["files"]["S2/signal.bin"] == integrity.sha256_file(
        dataset_dir / "S2" / "signal.bin"
    )


def test_cli_rejects_replace_flag_on_verify(raw):
    dataset_dir, _ = raw
    with pytest.raises(SystemExit):
        integrity._main(["verify", str(dataset_dir), "--replace-baseline"])


# Real datasets (run once a baseline has been committed) -------------------

@pytest.mark.parametrize("dataset", ["wesad", "nurse", "stress_predict"])
def test_real_raw_tree_matches_baseline(dataset: str):
    dataset_dir = DATA_RAW / dataset
    manifest = integrity.manifest_path(dataset_dir, MANIFESTS)
    if not manifest.is_file():
        pytest.skip(f"no checksum baseline for {dataset} yet (run: python -m wsr.utils.integrity snapshot {dataset_dir})")
    diff = integrity.verify(dataset_dir, MANIFESTS)
    assert integrity.is_unchanged(diff), f"raw {dataset} tree changed relative to committed baseline: {diff}"
