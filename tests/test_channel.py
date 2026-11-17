import numpy as np

from lora import channel, metrics, rng, seed, tx


def test_noise_from_snr_power(p):
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


def test_noise_power_matches_frame_noise(p):
    # noise_power es la potencia por muestra del ruido que transmit_frame suma a la trama
    symbols = rng.integers(0, p.M, 200)
    r = channel.transmit_frame(symbols, 0, p, -10, channel.FLAT)
    z = r - np.concatenate([tx.preamble(p), tx.modulator(symbols, p)])
    assert np.isclose(metrics.power(z), channel.noise_power(p, -10), rtol=0.05)


def test_exponential():
    h = channel.exponential(0.5, 8)
    assert h.size == 8 and np.isclose(np.sum(h**2), 1)
    assert np.allclose(h[1:] / h[:-1], np.exp(-0.5))


def test_system_is_causal(p):
    c = tx.modulator(np.array([3, 7]), p)
    r = channel.system(c, channel.EXP_SLOW, 0)
    assert r.size == c.size and np.isclose(r[0], c[0] * channel.EXP_SLOW[0])
    # Los canales de uno y dos coeficientes no cambian respecto de mode="same"
    for h in (channel.FLAT, channel.FREQ_SELECTIVE):
        assert np.allclose(channel.system(c, h, 0), np.convolve(c, h, mode="same"))


def test_transmit_frame_offset(p):
    r = channel.transmit_frame(np.array([1]), 2, p, 100, channel.FLAT, offset=37)
    assert np.allclose(r[: 2 * p.M + 37], 0, atol=1e-3)
    assert abs(r[2 * p.M + 37]) > 1e-3
