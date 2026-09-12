"""Verified WESAD loader for Phase 2 (wrist Empatica E4 + reference labels).

Safety contract (docs/decisions.md D-022):

1. The committed raw checksum baseline is verified BEFORE any pickle is
   deserialised (`VerifiedRelease.open` hashes the whole raw tree; every
   subsequent `load_participant` call re-hashes the specific pickle it is
   about to open and compares it with the baseline). No load path exists
   that skips verification.
2. Fails closed: no baseline, a changed tree, or a changed file -> the
   loader raises and nothing is unpickled.
3. The requested file must lie inside the verified raw WESAD release.
4. Only then is the official pickle loaded.
5. The structure Phase 2 needs is validated: participant id, wrist device,
   ACC (n,3) / BVP / EDA / TEMP present, numeric, finite, expected rates,
   stream lengths consistent with one synchronised recording, label vector
   present, integer-valued and aligned to the same duration.

Nothing under data/raw is ever written.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wsr.data.audit_wesad import EXPECTED_CHANNELS, LABEL_CODES, LABEL_RATE_HZ, NOMINAL_RATES_HZ, subject_number
from wsr.utils import integrity
from wsr.utils.paths import DATA_RAW, MANIFESTS

WRIST_MODALITIES = ("ACC", "BVP", "EDA", "TEMP")
WRIST_RATES_HZ: dict[str, float] = dict(NOMINAL_RATES_HZ["wrist"])  # readme II.2
WRIST_CHANNELS: dict[str, int] = dict(EXPECTED_CHANNELS["wrist"])

#: Streams of one synchronised recording must agree in duration to within this
#: many seconds (the audit found <= 0.25 s on the real release).
MAX_STREAM_DURATION_DISAGREEMENT_S = 1.0


class RawIntegrityError(RuntimeError):
    """The raw tree (or a file in it) does not match the committed baseline."""


class LoaderValidationError(ValueError):
    """The pickle loaded but does not have the structure Phase 2 requires."""


@dataclass(frozen=True)
class VerifiedRelease:
    """Handle proving that `raw_root` matched its committed baseline when opened.

    Obtain via `VerifiedRelease.open(...)`; the constructor is not a bypass
    because `load_participant` re-verifies each file against the manifest.
    """

    raw_root: Path
    manifests_dir: Path
    release_dir: Path
    expected_hashes: dict[str, str] = field(repr=False)

    @classmethod
    def open(cls, raw_root: Path = DATA_RAW / "wesad", manifests_dir: Path = MANIFESTS, release_subdir: str = "WESAD") -> "VerifiedRelease":
        raw_root = Path(raw_root).resolve()
        mpath = integrity.manifest_path(raw_root, manifests_dir)
        if not mpath.is_file():
            raise RawIntegrityError(f"no committed checksum baseline for {raw_root} at {mpath}; refusing to load")
        diff = integrity.verify(raw_root, manifests_dir)
        if not integrity.is_unchanged(diff):
            raise RawIntegrityError(f"raw tree {raw_root} differs from the committed baseline: {diff}; refusing to load")
        with open(mpath, "r", encoding="utf-8") as fh:
            expected = json.load(fh)["files"]
        release_dir = raw_root / release_subdir
        if not release_dir.is_dir():
            raise RawIntegrityError(f"release directory {release_dir} not found inside verified raw root")
        return cls(raw_root=raw_root, manifests_dir=Path(manifests_dir), release_dir=release_dir, expected_hashes=expected)

    def verified_path(self, relative: str) -> Path:
        """Resolve a release-relative path, re-hash it, and confirm it matches the baseline."""
        path = (self.raw_root / relative).resolve()
        if self.raw_root not in path.parents:
            raise RawIntegrityError(f"{path} is outside the verified raw root {self.raw_root}")
        if relative not in self.expected_hashes:
            raise RawIntegrityError(f"{relative} is not part of the committed baseline; refusing to load")
        if not path.is_file():
            raise RawIntegrityError(f"{path} listed in the baseline but missing on disk")
        actual = integrity.sha256_file(path)
        if actual != self.expected_hashes[relative]:
            raise RawIntegrityError(f"{relative} hash {actual[:12]}... != baseline {self.expected_hashes[relative][:12]}...; refusing to load")
        return path


@dataclass
class WesadParticipant:
    """One participant's synchronised wrist streams and reference labels (read-only views)."""

    participant_id: str
    acc: np.ndarray  # (n_acc, 3) float64, raw 1/64 g counts as shipped
    bvp: np.ndarray  # (n_bvp,) float64
    eda: np.ndarray  # (n_eda,) float64, microsiemens
    temp: np.ndarray  # (n_temp,) float64, degrees C
    label: np.ndarray  # (n_label,) int, raw codes 0-7 at 700 Hz
    rates_hz: dict[str, float]
    label_rate_hz: float
    source_file: str
    source_sha256: str
    validation: dict[str, Any]

    @property
    def stream_durations_s(self) -> dict[str, float]:
        d = {m: self.n_samples(m) / self.rates_hz[m] for m in WRIST_MODALITIES}
        d["label"] = self.label.size / self.label_rate_hz
        return d

    def n_samples(self, modality: str) -> int:
        return int(getattr(self, modality.lower()).shape[0])

    @property
    def duration_s(self) -> float:
        """Synchronised recording duration usable by every stream = the shortest stream."""
        return min(self.stream_durations_s.values())


def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise LoaderValidationError(f"wrist.{name}: expected a 1-D or 2-D array, got ndim={arr.ndim}")
    return arr


def validate_participant_dict(d: Any, expected_id: str) -> dict[str, Any]:
    """Validate the pickle structure Phase 2 depends on. Returns a report; raises on failure."""
    report: dict[str, Any] = {"checks": []}

    def check(ok: bool, msg: str) -> None:
        report["checks"].append({"ok": bool(ok), "check": msg})
        if not ok:
            raise LoaderValidationError(msg)

    check(isinstance(d, dict), "pickle top level is a dict")
    check({"signal", "label", "subject"} <= set(d.keys()), "pickle has keys signal/label/subject")
    check(str(d["subject"]) == expected_id, f"pickle subject field {d['subject']!r} equals folder id {expected_id!r}")
    check(isinstance(d["signal"], dict) and "wrist" in d["signal"], "signal.wrist exists")
    wrist = d["signal"]["wrist"]
    for m in WRIST_MODALITIES:
        check(m in wrist, f"wrist.{m} present")
        arr = _as_2d(wrist[m], m)
        check(np.issubdtype(arr.dtype, np.number), f"wrist.{m} numeric dtype (got {arr.dtype})")
        check(arr.shape[1] == WRIST_CHANNELS[m], f"wrist.{m} has {WRIST_CHANNELS[m]} channel(s) (got {arr.shape[1]})")
        check(arr.shape[0] > 0, f"wrist.{m} non-empty")
        n_nonfinite = int((~np.isfinite(arr)).sum())
        report[f"wrist.{m}.n_nonfinite"] = n_nonfinite
        check(n_nonfinite == 0, f"wrist.{m} all finite (found {n_nonfinite} non-finite)")

    lab = np.asarray(d["label"]).ravel()
    check(lab.size > 0, "label vector non-empty")
    check(np.issubdtype(lab.dtype, np.number), f"label numeric dtype (got {lab.dtype})")
    check(bool(np.all(np.isfinite(lab))), "label finite")
    check(bool(np.array_equal(lab, np.round(lab))), "label integer-valued")
    codes = {int(c) for c in np.unique(lab)}
    check(codes <= set(LABEL_CODES), f"label codes documented (found {sorted(codes)})")

    # durations: every wrist stream and the label vector describe one synchronised recording
    durations = {m: _as_2d(wrist[m], m).shape[0] / WRIST_RATES_HZ[m] for m in WRIST_MODALITIES}
    durations["label"] = lab.size / LABEL_RATE_HZ
    spread = max(durations.values()) - min(durations.values())
    report["stream_durations_s"] = durations
    report["duration_spread_s"] = spread
    check(spread <= MAX_STREAM_DURATION_DISAGREEMENT_S, f"stream durations agree within {MAX_STREAM_DURATION_DISAGREEMENT_S} s (spread {spread:.3f} s)")
    report["label_codes_present"] = sorted(codes)
    return report


def load_participant(participant_id: str, release: VerifiedRelease | None = None) -> WesadParticipant:
    """Load one participant's wrist streams + labels through the verified path.

    If `release` is None the whole raw tree is verified first (slow but safe);
    pass a `VerifiedRelease` to amortise the tree check across participants -
    the pickle itself is still re-hashed against the baseline on every call.
    """
    subject_number(participant_id)  # raises on malformed ids
    rel = release or VerifiedRelease.open()
    relative = f"{rel.release_dir.name}/{participant_id}/{participant_id}.pkl"
    path = rel.verified_path(relative)
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")  # trusted: official release, hash-verified above
    report = validate_participant_dict(d, participant_id)
    wrist = d["signal"]["wrist"]
    part = WesadParticipant(
        participant_id=participant_id,
        acc=np.ascontiguousarray(_as_2d(wrist["ACC"], "ACC"), dtype=np.float64),
        bvp=np.ascontiguousarray(_as_2d(wrist["BVP"], "BVP")[:, 0], dtype=np.float64),
        eda=np.ascontiguousarray(_as_2d(wrist["EDA"], "EDA")[:, 0], dtype=np.float64),
        temp=np.ascontiguousarray(_as_2d(wrist["TEMP"], "TEMP")[:, 0], dtype=np.float64),
        label=np.ascontiguousarray(np.round(np.asarray(d["label"]).ravel()).astype(np.int16)),
        rates_hz=dict(WRIST_RATES_HZ),
        label_rate_hz=LABEL_RATE_HZ,
        source_file=relative,
        source_sha256=rel.expected_hashes[relative],
        validation=report,
    )
    for arr in (part.acc, part.bvp, part.eda, part.temp, part.label):
        arr.setflags(write=False)
    del d
    return part
