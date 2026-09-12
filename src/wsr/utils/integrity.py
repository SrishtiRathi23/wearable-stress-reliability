"""Raw-data immutability: checksum snapshots and verification.

Research invariant #8: raw files are never changed. This module makes that
checkable. After downloading a dataset, run

    python -m wsr.utils.integrity snapshot data/raw/wesad

which writes data/manifests/raw_checksums_wesad.json (commit it). Then

    python -m wsr.utils.integrity verify data/raw/wesad

(and tests/test_raw_immutable.py) re-hashes the tree and reports any file that
was added, removed or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from wsr.utils.paths import MANIFESTS

_CHUNK = 1 << 20
_IGNORED_NAMES = (".gitkeep", "README.md")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path, ignore_names: tuple[str, ...] = _IGNORED_NAMES) -> dict[str, str]:
    """Map of POSIX-style relative path -> sha256 for every file under `root`."""
    root = Path(root)
    out: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in ignore_names:
            continue
        out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def manifest_path(dataset_dir: Path, manifests_dir: Path = MANIFESTS) -> Path:
    return Path(manifests_dir) / f"raw_checksums_{Path(dataset_dir).name}.json"


def snapshot(dataset_dir: Path, manifests_dir: Path = MANIFESTS) -> Path:
    hashes = hash_tree(dataset_dir)
    if not hashes:
        raise FileNotFoundError(f"No files found under {dataset_dir}; nothing to snapshot")
    out = manifest_path(dataset_dir, manifests_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"root": Path(dataset_dir).name, "n_files": len(hashes), "files": hashes}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out


def verify(dataset_dir: Path, manifests_dir: Path = MANIFESTS) -> dict[str, list[str]]:
    """Compare the current tree with its manifest.

    Returns {"added": [...], "removed": [...], "modified": [...]}; all empty
    means the raw tree is unchanged.
    """
    mpath = manifest_path(dataset_dir, manifests_dir)
    with open(mpath, "r", encoding="utf-8") as fh:
        expected: dict[str, str] = json.load(fh)["files"]
    actual = hash_tree(dataset_dir)
    common = expected.keys() & actual.keys()
    return {
        "added": sorted(set(actual) - set(expected)),
        "removed": sorted(set(expected) - set(actual)),
        "modified": sorted(k for k in common if expected[k] != actual[k]),
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["snapshot", "verify"])
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        print(f"wrote {snapshot(args.dataset_dir)}")
        return 0
    diff = verify(args.dataset_dir)
    changed = any(diff.values())
    print(json.dumps(diff, indent=2))
    print("RAW TREE MODIFIED" if changed else "raw tree unchanged")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(_main())
