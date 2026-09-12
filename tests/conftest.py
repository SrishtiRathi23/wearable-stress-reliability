"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fake_raw_tree(tmp_path: Path) -> Path:
    """A small stand-in for a raw dataset directory (never the real data)."""
    root = tmp_path / "raw" / "fakeset"
    (root / "S2").mkdir(parents=True)
    (root / "S2" / "signal.bin").write_bytes(b"\x00\x01\x02" * 100)
    (root / "S3").mkdir()
    (root / "S3" / "signal.bin").write_bytes(b"\x03\x04" * 100)
    (root / "README.md").write_text("ignored by hashing", encoding="utf-8")
    (root / ".gitkeep").write_text("", encoding="utf-8")
    return root
