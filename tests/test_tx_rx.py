import numpy as np
import pytest

from lora import LoRaParams, metrics, rx, tx

p = LoRaParams(7, 125e3)


def test_encoder():
    assert np.array_equal(tx.encoder(np.array([0, 0, 0, 1, 0, 0, 0]), p), [8])
    assert np.array_equal(
        tx.encoder(np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]), p), [8, 127]
    )


def test_decoder():
    assert np.array_equal(rx.decoder(np.array([8]), p), [0, 0, 0, 1, 0, 0, 0])
    assert np.array_equal(
        rx.decoder(np.array([8, 127]), p), [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    )


@pytest.mark.parametrize("SF", [7, 8, 9, 10])
def test_roundtrip(SF):
    q = LoRaParams(SF, 125e3)
    symbols = np.array([0, 32, 80, q.M - 1])
    assert np.array_equal(rx.demodulator(tx.modulator(symbols, q), q), symbols)


def test_dechirp_fft_matches_demodulator():
    symbols = np.array([0, 32, 80])
    D = rx.dechirp_fft(tx.modulator(symbols, p), p)
    assert D.shape == (3, p.M)
    assert np.array_equal(np.argmax(np.abs(D) ** 2, axis=1), symbols)


def test_instantaneous_frequency():
    f = tx.instantaneous_frequency(np.array([0, p.M - 1]), p)
    assert f.shape == (2 * p.M,)
    assert f[0] == 0 and f[p.M] == (p.M - 1) * p.B / p.M
    assert f[p.M + 1] == 0  # Se repliega al llegar a B


@pytest.mark.parametrize("n_downchirps", [0, 1, 2.25, 2.3])
@pytest.mark.parametrize("SF", [7, 9])
def test_preamble_len(SF, n_downchirps):
    q = LoRaParams(SF, 125e3, n_downchirps=n_downchirps)
    assert tx.preamble(q).size == q.preamble_len


def test_power_and_ber():
    assert np.isclose(metrics.power(tx.modulator(np.array([0]), p)), 1 / p.M)
    bits = np.array([0, 1, 1, 0])
    assert metrics.BER(bits, bits) == 0
    assert metrics.BER(bits, bits ^ np.array([1, 0, 0, 0])) == 0.25
