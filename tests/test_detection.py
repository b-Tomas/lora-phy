import dataclasses
from math import factorial

import numpy as np
import pytest

from lora import LoRaParams, channel, detection, metrics, rng


def test_full(p, clean_frame):
    r, _, gap, npw = clean_frame
    T = detection.preamble_correlation(r, p)
    assert detection.detect_start(T, detection.time_threshold(npw, 1e-3, p), 8 * p.M) == gap * p.M
    assert detection.detect_full(r, p, npw, 1e-3) == gap * p.M


def test_full_matches_direct_correlation(p):
    # La correlación por FFT da lo mismo que la directa
    r = rng.normal(size=3000) + 1j * rng.normal(size=3000)
    template = detection.modulator(np.zeros(8, dtype=int), p)
    expected = np.abs(np.correlate(r, template, mode="valid")) ** 2
    assert np.allclose(detection.preamble_correlation(r, p), expected)


def test_peaks_threshold(p):
    # Con un solo pico, S[k] es exponencial y el umbral coincide con el de T[k]
    q1 = dataclasses.replace(p, n_upchirps=1)
    assert np.isclose(
        detection.peaks_threshold(2.0, 1e-3, q1), detection.time_threshold(2.0, 1e-3, q1)
    )
    # El umbral cumple la P_FA pedida (cola de la Erlang de orden 8)
    x = detection.peaks_threshold(1.0, 1e-3, p)
    assert np.isclose(np.exp(-x) * sum(x**i / factorial(i) for i in range(8)), 1e-3)


def test_peaks(p, clean_frame):
    r, _, gap, npw = clean_frame
    S = detection.peaks_statistic(r, p)
    assert S.size == detection.preamble_correlation(r, p).size
    assert detection.detect_start(S, detection.peaks_threshold(npw, 1e-3, p), 8 * p.M) == gap * p.M
    assert detection.detect_peaks(r, p, npw, 1e-3) == gap * p.M


def test_freq(p, clean_frame):
    r, _, gap, npw = clean_frame
    assert detection.detect_start(detection.preamble_zero_count(r, p), 6, 8) == gap
    assert detection.detect_freq(r, p, npw, 1e-3) == gap * p.M
    # Umbral de Neyman-Pearson. Para SF = 7: P(V >= 2 | H0) ~ 1.7e-3 y P(V >= 3 | H0) ~ 2.6e-5
    assert detection.freq_threshold(1e-3, p) == 3
    assert detection.freq_threshold(1.6e-5, p) == 4
    assert detection.freq_threshold(2e-3, p) == 2
    assert detection.freq_threshold(0.5, p) == 1
    assert np.isclose(detection.zero_count_false_alarm(1, p), 1 - (1 - 2**-7) ** 8)


def test_threshold_per_symbol(p):
    # P_FA por símbolo: en el tiempo se prueban M muestras por símbolo, en frecuencia una ventana
    npw = 2.0
    assert detection.threshold("full", p, npw, 1e-3) == detection.time_threshold(npw, 1e-3 / p.M, p)
    assert detection.threshold("peaks", p, npw, 1e-3) == detection.peaks_threshold(
        npw, 1e-3 / p.M, p
    )
    assert detection.threshold("freq", p, npw, 1e-3) == detection.freq_threshold(1e-3, p)


@pytest.mark.parametrize(
    "statistic, n_samples",
    [
        (detection.upchirp_correlation, 127),
        (detection.preamble_correlation, 8 * 128 - 1),
        (detection.peaks_statistic, 8 * 128 - 1),
        (detection.preamble_zero_count, 8 * 128 - 1),
    ],
)
def test_short_signal_rejected(p, statistic, n_samples):
    # Más corta que la plantilla, np.correlate intercambiaría los argumentos
    with pytest.raises(AssertionError):
        statistic(np.zeros(n_samples, dtype=complex), p)


@pytest.mark.parametrize(
    "q",
    [
        LoRaParams(9, 125e3),
        LoRaParams(7, 125e3, n_downchirps=1),
        LoRaParams(7, 125e3, n_downchirps=2.3),
        LoRaParams(7, 125e3, n_upchirps=4),
    ],
)
@pytest.mark.parametrize(
    "detect", [detection.detect_full, detection.detect_peaks, detection.detect_freq]
)
def test_detectors_other_params(q, detect):
    symbols = rng.integers(0, q.M, 4)
    gap = 2
    r = channel.transmit_frame(symbols, gap, q, 100, channel.FLAT)
    start = detect(r, q, channel.noise_power(q, 100), 1e-3)
    assert start == gap * q.M


def test_roc():
    # Estadísticos separados dan un detector perfecto (pasa por el punto (0, 1))
    P_FA, P_D = metrics.roc(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    assert P_FA[0] == 0 and P_D[0] == 0  # Umbral infinito
    assert np.any((P_FA == 0) & (P_D == 1))
    assert P_FA[-1] == 1 and P_D[-1] == 1  # Umbral mínimo: siempre H1


def test_detect_start_follows_the_peak():
    # El primer cruce llega más de `span` posiciones antes del máximo, que sube de a escalones
    statistic = np.array([0, 2, 0, 3, 0, 4, 0, 9, 1, 0, 1, 0, 0])
    assert detection.detect_start(statistic, 1, 3) == 7
