"""Deterministic seeding invariants."""

import numpy as np

from wsr.utils.seeds import child_seed, rng, set_global_seed


def test_child_seed_is_deterministic_and_stream_specific():
    assert child_seed(42, "split") == child_seed(42, "split")
    assert child_seed(42, "split") != child_seed(42, "model")
    assert child_seed(42, "split") != child_seed(43, "split")
    assert 0 <= child_seed(42, "split") < 2**32


def test_named_rng_reproduces_same_sequence():
    a = rng(42, "mask").random(5)
    b = rng(42, "mask").random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, rng(42, "other").random(5))


def test_set_global_seed_makes_legacy_numpy_reproducible():
    set_global_seed(7)
    first = np.random.rand(3)
    set_global_seed(7)
    second = np.random.rand(3)
    assert np.array_equal(first, second)
