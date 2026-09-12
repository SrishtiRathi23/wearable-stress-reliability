"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fake_raw_tree(tmp_path: Path) -> Path:
    """A small stand-in for a raw dataset directory (never the real data).

    Mimics what a downloaded release looks like: signal files per participant,
    dataset-shipped documentation (README.md, docs/description.txt) that MUST be
    protected by the checksum baseline, and a repository-owned `.gitkeep` at the
    dataset root that is the one allowed exclusion.
    """
    root = tmp_path / "raw" / "fakeset"
    (root / "S2").mkdir(parents=True)
    (root / "S2" / "signal.bin").write_bytes(b"\x00\x01\x02" * 100)
    (root / "S3").mkdir()
    (root / "S3" / "signal.bin").write_bytes(b"\x03\x04" * 100)
    (root / "README.md").write_text("dataset-supplied readme", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "description.txt").write_text("dataset descriptor", encoding="utf-8")
    (root / ".gitkeep").write_text("", encoding="utf-8")
    return root
