import numpy as np

from lora import channel, metrics, rng, seed, tx


def test_frame_error(p, clean_frame):
    r, symbols, gap, _ = clean_frame
    assert not metrics.frame_error(r, gap * p.M, symbols, p)
    assert metrics.frame_error(r, gap * p.M + 1, symbols, p)  # Una muestra de diferencia
    assert metrics.frame_error(r, (gap - 1) * p.M, symbols, p)
    assert metrics.frame_error(r, None, symbols, p)


def test_noise_power_matches_snr(p):
    c = tx.modulator(rng.integers(0, p.M, 2000), p)
    z = channel.noise_from_SNR(c, -10)
    assert np.isclose(metrics.power(z), metrics.power(c) * 10, rtol=0.05)


def test_seed_resets_noise(p):
    c = tx.preamble(p)
    seed(5)
    a = channel.noise_from_SNR(c, 0)
    seed(5)
    b = channel.noise_from_SNR(c, 0)
    assert np.array_equal(a, b)
