import numpy as np
import pytest

from lora import LoRaParams, channel, rng, sdr

p = LoRaParams(SF=8, B=1e6)
NUM_SYMBOLS, GAP = 10, (5, 10)


@pytest.mark.parametrize(("SNR_dB", "all_errors"), [(np.inf, False), (-30.0, True)])
def test_measure_and_evaluate(SNR_dB, all_errors):
    symbols, gaps, r, _ = sdr.measure(sdr.SimulatedPluto(), p, 4, NUM_SYMBOLS, GAP, SNR_dB)
    errors, timing, _, _ = sdr.evaluate(r, symbols, gaps, p, 1e-3)
    assert errors.size == 4
    assert sdr.frames(symbols, gaps, p).size % 2 == 0  # El Pluto repite mal los buffers impares
    assert errors.all() if all_errors else not errors.any() and not timing.any()


def test_noise_added_in_tx():
    # Con ruido en el TX y un canal sin ruido, el SNR medido es el agregado
    symbols, gaps, r, _ = sdr.measure(sdr.SimulatedPluto(), p, 3, NUM_SYMBOLS, GAP, -6.0)
    assert sdr.evaluate(r, symbols, gaps, p, 1e-3).snr == pytest.approx(-6.0, abs=0.3)


def test_receive_finds_every_frame():
    # Silencios de largo al azar: cada inicio detectado es el de una trama del buffer
    symbols = rng.integers(0, p.M, size=(3, NUM_SYMBOLS))
    gaps = rng.integers(GAP[0] * p.M, GAP[1] * p.M, size=3)
    x = sdr.frames(symbols, gaps, p)
    r = np.roll(np.tile(x, 2), 1234) * sdr.TX_SCALE
    starts, payloads = sdr.receive(r, p, sdr.ADC_NOISE_FLOOR, 1e-3, NUM_SYMBOLS)
    in_buffer = np.cumsum(gaps) + np.arange(3) * (p.preamble_len + NUM_SYMBOLS * p.M)
    assert set((starts - 1234) % x.size) <= set(in_buffer)
    assert starts.size >= 4 and {tuple(s) for s in payloads} == {tuple(s) for s in symbols}


@pytest.mark.parametrize("SNR_dB", [-10.0, 0.0])
def test_snr_matches_simulation(SNR_dB):
    symbols = rng.integers(0, p.M, NUM_SYMBOLS)
    r = channel.transmit_frame(symbols, 5, p, SNR_dB, channel.FLAT)
    x = sdr.frames(symbols[None], np.array([5 * p.M]), p)
    assert sdr.snr_dB(r * 1e3 * np.exp(1j), x) == pytest.approx(SNR_dB, abs=0.3)


def test_save_load(tmp_path):
    captures = np.array([[1 + 2j, -3 - 4j]])
    sdr.save(tmp_path / "c.npz", captures, gain=0.5)
    loaded, meta = sdr.load(tmp_path / "c.npz")
    assert np.array_equal(loaded, captures)
    assert meta["gain"] == 0.5
