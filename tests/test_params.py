import dataclasses

import numpy as np
import pytest

from lora import LoRaParams, rng, seed


def test_derived_properties():
    p = LoRaParams(SF=7, B=125e3)
    assert p.M == 128
    assert p.T == 1 / 125e3
    assert p.Ts == 128 / 125e3
    assert p.preamble_len == 8 * 128 + 9 * 128 // 4


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        LoRaParams(7, 125e3).SF = 8  # type: ignore[misc]


def test_seed_reproducible():
    seed(3)
    a = rng.integers(0, 2, 100)
    seed(3)
    b = rng.integers(0, 2, 100)
    assert np.array_equal(a, b)
