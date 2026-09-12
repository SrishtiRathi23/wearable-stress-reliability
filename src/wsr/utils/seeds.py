"""Explicit, reproducible random seeding.

Design: one global seed from the config; every stochastic component obtains a
child generator through `child_seed`/`rng` with a *named* stream, so adding a
new random consumer does not silently shift the random state of existing ones.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python `random`, NumPy's legacy global RNG and PYTHONHASHSEED.

    Library-specific RNGs (scikit-learn `random_state`, xgboost `seed`) must be
    passed explicitly from `child_seed(seed, "<component>")`; this function does
    not (and cannot) reach them.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def child_seed(seed: int, stream: str) -> int:
    """Derive a deterministic 32-bit seed for a named random stream.

    The derivation is a hash of (seed, stream) so it is stable across Python
    versions and platforms, unlike `hash()`.
    """
    digest = hashlib.sha256(f"{seed}:{stream}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def rng(seed: int, stream: str) -> np.random.Generator:
    """A NumPy Generator for a named stream derived from the global seed."""
    return np.random.default_rng(child_seed(seed, stream))
