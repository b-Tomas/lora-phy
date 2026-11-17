import matplotlib
import numpy as np
import pytest

from lora import LoRaParams, channel, simulation, tx
from lora.anexos import plots, vangelista

matplotlib.use("Agg")

p = LoRaParams(7, 125e3)
symbols = np.array([0, 32, 80, p.M - 1])


def test_modulator_is_eq2():
    k = np.tile(np.arange(p.M), symbols.size)
    s = np.repeat(symbols, p.M)
    expected = np.exp(2j * np.pi * (s + k) * k / p.M) / np.sqrt(p.M)
    assert np.allclose(vangelista.modulator(symbols, p), expected)
    assert np.allclose(vangelista.modulator(symbols, p, 8)[::8], expected)
    # Es el chirp de LoRa por e^{jπ k^2 / 2^SF}
    assert np.allclose(expected, tx.modulator(symbols, p) * np.exp(1j * np.pi * k**2 / p.M))


def test_roundtrip():
    assert np.array_equal(vangelista.demodulator(vangelista.modulator(symbols, p), p), symbols)


def test_mismatch_fails():
    # Chirp de LoRa con el demodulador de Ec. 16: no decide ningún símbolo
    s = np.arange(p.M)
    assert not np.any(vangelista.demodulator(tx.modulator(s, p), p) == s)


@pytest.mark.parametrize("h", [channel.FLAT, channel.FREQ_SELECTIVE])
def test_no_bit_errors_without_noise(h):
    # Sin ruido el sistema de Ec. 2 no comete errores, como el de LoRa
    errors, _ = simulation.bit_errors(p, 100.0, h, vangelista.modulator, vangelista.demodulator)
    assert errors == 0


def test_chirp_comparison():
    assert isinstance(plots.chirp_comparison(symbols[:3], p), matplotlib.figure.Figure)
