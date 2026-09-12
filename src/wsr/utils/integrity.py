"""Raw-data immutability: a committed checksum baseline per dataset.

Research invariant #8: raw files are never changed. This module makes that
checkable and makes the baseline itself hard to overwrite by accident.

Workflow
--------
1. After downloading a dataset once, establish its baseline:

       python -m wsr.utils.integrity snapshot data/raw/wesad

   This writes data/manifests/raw_checksums_wesad.json. Commit it. It records
   the SHA-256 of EVERY file in the dataset tree, including documentation the
   dataset ships with (readme, license, descriptor files).

2. At any later time (and in tests/test_raw_immutable.py):

       python -m wsr.utils.integrity verify data/raw/wesad

   re-hashes the tree and reports added, removed and modified files.

3. `snapshot` REFUSES to run if a baseline already exists. A baseline is only
   replaced when we deliberately obtain a different dataset release, which is
   a dataset-version change that must be reviewed and logged in
   docs/decisions.md. For that case, and only that case:

       python -m wsr.utils.integrity snapshot data/raw/wesad --replace-baseline

   Nothing in the codebase calls `snapshot(..., replace=True)` automatically.

Exclusions
----------
The only file excluded from hashing is a `.gitkeep` at the top level of the
dataset directory. It is repository-owned (it keeps the empty directory in
Git before the dataset is downloaded) and is not part of the dataset. Files
with that name deeper in the tree, and every other file, are hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from wsr.utils.paths import MANIFESTS

_CHUNK = 1 << 20

# Relative POSIX paths (from the dataset root) that are repository placeholders,
# not dataset content. Deliberately narrow; see module docstring.
REPO_PLACEHOLDERS: frozenset[str] = frozenset({".gitkeep"})


class BaselineExistsError(FileExistsError):
    """Raised when `snapshot` would overwrite an established checksum baseline."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    """Map of POSIX-style relative path -> sha256 for every file under `root`.

    Keys are sorted, so the mapping (and any JSON dumped from it with
    sort_keys=True) is deterministic for a given tree.
    """
    root = Path(root)
    out: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in REPO_PLACEHOLDERS:
            continue
        out[rel] = sha256_file(path)
    return out


def manifest_path(dataset_dir: Path, manifests_dir: Path = MANIFESTS) -> Path:
    return Path(manifests_dir) / f"raw_checksums_{Path(dataset_dir).name}.json"


def snapshot(dataset_dir: Path, manifests_dir: Path = MANIFESTS, *, replace: bool = False) -> Path:
    """Write the checksum baseline for `dataset_dir`.

    Refuses with BaselineExistsError if a baseline already exists, unless
    `replace=True`. `replace=True` discards the old baseline: use it only for a
    deliberate, reviewed dataset-version change, never to make `verify` pass.
    """
    out = manifest_path(dataset_dir, manifests_dir)
    if out.exists() and not replace:
        raise BaselineExistsError(
            f"A raw-data checksum baseline already exists at {out}.\n"
            f"Run `verify` against it instead of re-snapshotting. Re-snapshotting would\n"
            f"silently bless any modification of the raw tree. If this is a deliberate new\n"
            f"dataset release, log the version change in docs/decisions.md and re-run with\n"
            f"--replace-baseline (CLI) or replace=True (API)."
        )
    hashes = hash_tree(dataset_dir)
    if not hashes:
        raise FileNotFoundError(f"No files found under {dataset_dir}; nothing to snapshot")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"root": Path(dataset_dir).name, "n_files": len(hashes), "files": hashes}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out


def verify(dataset_dir: Path, manifests_dir: Path = MANIFESTS) -> dict[str, list[str]]:
    """Compare the current tree with its baseline.

    Returns {"added": [...], "removed": [...], "modified": [...]}; all empty
    means the raw tree is unchanged. Raises FileNotFoundError if no baseline
    exists yet.
    """
    mpath = manifest_path(dataset_dir, manifests_dir)
    if not mpath.is_file():
        raise FileNotFoundError(f"No checksum baseline for {dataset_dir} at {mpath}; run `snapshot` first")
    with open(mpath, "r", encoding="utf-8") as fh:
        expected: dict[str, str] = json.load(fh)["files"]
    actual = hash_tree(dataset_dir)
    common = expected.keys() & actual.keys()
    return {
        "added": sorted(set(actual) - set(expected)),
        "removed": sorted(set(expected) - set(actual)),
        "modified": sorted(k for k in common if expected[k] != actual[k]),
    }


def is_unchanged(diff: dict[str, list[str]]) -> bool:
    return not any(diff.values())


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["snapshot", "verify"])
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=MANIFESTS,
        help="where baselines live (default: data/manifests; override only for tests/tools)",
    )
    parser.add_argument(
        "--replace-baseline",
        action="store_true",
        help=(
            "DANGEROUS: discard the existing checksum baseline and write a new one. "
            "Only for a deliberate, reviewed dataset-version change logged in docs/decisions.md. "
            "Never use this to make `verify` pass."
        ),
    )
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        try:
            out = snapshot(args.dataset_dir, args.manifests_dir, replace=args.replace_baseline)
        except BaselineExistsError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(f"{'REPLACED' if args.replace_baseline else 'wrote'} baseline {out}")
        return 0

    if args.replace_baseline:
        parser.error("--replace-baseline only applies to `snapshot`")
    diff = verify(args.dataset_dir, args.manifests_dir)
    print(json.dumps(diff, indent=2))
    print("raw tree unchanged" if is_unchanged(diff) else "RAW TREE MODIFIED")
    return 0 if is_unchanged(diff) else 1


if __name__ == "__main__":
    sys.exit(_main())
