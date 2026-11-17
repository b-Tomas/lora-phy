import numpy as np
import pytest

from lora import LoRaParams, channel, rng, sdr

p = LoRaParams(SF=8, B=1e6)
NUM_SYMBOLS, GAP = 10, (5, 10)


@pytest.mark.parametrize(("scale_dB", "all_errors"), [(0.0, False), (-80.0, True)])
@pytest.mark.filterwarnings("ignore:divide by zero")  # Captura en cero: SNR de -inf
def test_measure_and_evaluate(scale_dB, all_errors):
    # Con ganancia fija y -80 dB de escala, la señal no llega a un nivel del ADC
    symbols, gaps, r, _ = sdr.measure(
        sdr.SimulatedPluto(), p, 4, NUM_SYMBOLS, GAP, calibrate=False, scale_dB=scale_dB
    )
    errors, timing, _ = sdr.evaluate(r, symbols, gaps, p, 1e-3)
    assert errors.size == 4
    assert sdr.frames(symbols, gaps, p).size % 2 == 0  # El Pluto repite mal los buffers impares
    assert errors.all() if all_errors else not errors.any() and not timing.any()


def test_capture_snr_matches_evaluate():
    symbols, gaps, r, _ = sdr.measure(sdr.SimulatedPluto(), p, 4, NUM_SYMBOLS, GAP, calibrate=False)
    assert sdr.capture_snr(r, symbols, gaps, p) == sdr.evaluate(r, symbols, gaps, p, 1e-3).snr


def _fake_air(monkeypatch, snr_offsets):
    """Reemplaza `measure` y `capture_snr` por un canal donde el SNR medido es la escala más el
    siguiente valor de `snr_offsets` (la ganancia del canal y el ruido del RX en esa captura)."""

    offsets = iter(snr_offsets)
    scales = []

    def measure(_sdr, _p, _n_frames, _num_symbols, _gap_range, scale_dB=0.0):
        scales.append(scale_dB)
        return None, None, scale_dB + next(offsets), 0.0  # La "captura" es su SNR

    monkeypatch.setattr(sdr, "measure", measure)
    monkeypatch.setattr(sdr, "capture_snr", lambda r, *_: r)
    return scales


def test_measure_at_snr_corrects_the_scale(monkeypatch):
    # El canal da 1.2 dB menos que lo estimado: con una corrección llega al SNR buscado
    scales = _fake_air(monkeypatch, [12.8, 12.9])
    _, scale_dB, snr = sdr.measure_at_snr(None, p, -20.0, -34.0, 1, NUM_SYMBOLS, GAP)
    assert scales == [-34.0, pytest.approx(-32.8)]
    assert scale_dB == scales[-1] and abs(snr + 20) <= 0.5


def test_measure_at_snr_limits_the_step(monkeypatch):
    # Una captura con interferencia (SNR 10 dB por debajo) no lleva la escala 10 dB arriba
    scales = _fake_air(monkeypatch, [4.0, 14.0, 14.0])
    _, scale_dB, snr = sdr.measure_at_snr(None, p, -20.0, -34.0, 1, NUM_SYMBOLS, GAP)
    assert scales == [-34.0, -31.0, -34.0]
    assert scale_dB == -34.0 and snr == -20.0


def test_measure_at_snr_keeps_the_closest(monkeypatch):
    # Ningún intento llega a 0.5 dB: SNR de -22, -17, -22 y -17.5 dB. Se queda el primero, a 2 dB
    scales = _fake_air(monkeypatch, [12.0, 15.0, 13.0, 15.5])
    monkeypatch.setattr(sdr, "SNR_MAX_TRIES", 4)
    _, scale_dB, snr = sdr.measure_at_snr(None, p, -20.0, -34.0, 1, NUM_SYMBOLS, GAP)
    assert scales == [-34.0, -32.0, -35.0, -33.0]
    assert (scale_dB, snr) == (-34.0, -22.0)


def test_measure_snr_sweep_one_point_per_dB(monkeypatch):
    # SNR medido menos escala en cada captura de una medición real por el aire, con pasos de 1 dB
    # de escala: salta entre 13.0 y 14.2 dB y se mantiene unas capturas en cada valor
    measured = [14.1, 13.0, 13.0, 13.0, 14.3, 14.2, 14.2, 14.2, 14.2, 13.0, 13.0, 14.2, 14.2, 13.0]
    _fake_air(monkeypatch, measured * 4)
    targets = np.arange(-22.0, -13.0)
    runs, scales = sdr.measure_snr_sweep(None, p, targets, 1, NUM_SYMBOLS, GAP)
    snrs = np.array([run[2] for run in runs])
    assert len(runs) == len(scales) == targets.size
    assert np.all(np.abs(snrs - targets) <= 0.5)


def test_receive_finds_every_frame():
    # Silencios de largo al azar: cada inicio detectado es el de una trama del buffer
    symbols = rng.integers(0, p.M, size=(3, NUM_SYMBOLS))
    gaps = rng.integers(GAP[0] * p.M, GAP[1] * p.M, size=3)
    x = sdr.frames(symbols, gaps, p)
    r = np.roll(np.tile(x, 2), 1234) * sdr.TX_SCALE
    starts = sdr.receive(r, p, sdr.ADC_NOISE_FLOOR, 1e-3, NUM_SYMBOLS)
    in_buffer = np.cumsum(gaps) + np.arange(3) * (p.preamble_len + NUM_SYMBOLS * p.M)
    assert starts.size >= 4 and set((starts - 1234) % x.size) == set(in_buffer)


@pytest.mark.parametrize("SNR_dB", [-10.0, 0.0])
def test_snr_matches_simulation(SNR_dB):
    # Trama larga: con 10 símbolos de payload, a -10 dB el estimador tiene un desvío de ~0.26 dB
    symbols = rng.integers(0, p.M, 20 * NUM_SYMBOLS)
    r = channel.transmit_frame(symbols, 5, p, SNR_dB, channel.FLAT)
    x = sdr.frames(symbols[None], np.array([5 * p.M]), p)
    assert sdr.snr_dB(r * 1e3 * np.exp(1j), x) == pytest.approx(SNR_dB, abs=0.3)


def test_save_load(tmp_path):
    captures = np.array([[1 + 2j, -3 - 4j]])
    sdr.save(tmp_path / "c.npz", captures, gain=0.5)
    loaded, meta = sdr.load(tmp_path / "c.npz")
    assert np.array_equal(loaded, captures)
    assert meta["gain"] == 0.5
