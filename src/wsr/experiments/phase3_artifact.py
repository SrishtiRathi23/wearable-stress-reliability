"""Access to the APPROVED, FROZEN Phase-3 prediction artifact (D-032).

Any later stage (Study A/B/C) must obtain the predictions through
`load_approved_predictions`, which verifies the file's SHA-256 against the
frozen manifest before returning anything. The artifact is never
regenerated or replaced on the basis of downstream results; a genuine
defect leads to a NEW versioned artifact and a decision-log entry, never a
silent overwrite (invariant #21).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wsr.utils import integrity
from wsr.utils.paths import MANIFESTS, ROOT

PREDICTIONS_MANIFEST = MANIFESTS / "phase3_predictions_manifest.json"


class FrozenArtifactError(RuntimeError):
    pass


def read_frozen_manifest(manifest_path: Path = PREDICTIONS_MANIFEST) -> dict:
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if m.get("frozen") is not True:
        raise FrozenArtifactError(f"{manifest_path} is not marked frozen; Phase-3 predictions are not approved for downstream use")
    return m


def verify_approved_predictions(manifest_path: Path = PREDICTIONS_MANIFEST, root: Path = ROOT) -> Path:
    m = read_frozen_manifest(manifest_path)
    path = Path(root) / m["artifact"]["path"]
    if not path.is_file():
        raise FrozenArtifactError(f"frozen artifact missing: {path}")
    actual = integrity.sha256_file(path)
    if actual != m["artifact"]["sha256"]:
        raise FrozenArtifactError(f"{path} sha256 {actual[:12]}... != frozen {m['artifact']['sha256'][:12]}...; refusing to use")
    return path


def load_approved_predictions(manifest_path: Path = PREDICTIONS_MANIFEST, root: Path = ROOT) -> pd.DataFrame:
    path = verify_approved_predictions(manifest_path, root)
    m = read_frozen_manifest(manifest_path)
    df = pd.read_parquet(path)
    if len(df) != m["artifact"]["n_rows"]:
        raise FrozenArtifactError("row count differs from the frozen manifest")
    return df
