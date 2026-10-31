import dataclasses
from math import factorial

import numpy as np

from lora import detection, metrics


def test_full(p, clean_frame):
    r, _, gap, npw = clean_frame
    T = detection.preamble_correlation(r, p)
    assert detection.detect_start(T, detection.time_threshold(npw, 1e-3, p), 8 * p.M) == gap * p.M
    assert detection.detect_full(r, p, npw, 1e-3) == gap * p.M


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
    assert (
        detection.detect_start(S, detection.peaks_threshold(npw, 1e-3, p), 8 * p.M) == gap * p.M
    )
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


def test_ideal_detector_gap(p, clean_frame):
    r, _, gap, npw = clean_frame
    assert detection.ideal_detector(gap)(r, p, npw, 1e-3) == gap * p.M


def test_detectors_keys():
    assert set(detection.DETECTORS) == {"full", "peaks", "freq"}


def test_roc():
    # Estadísticos separados dan un detector perfecto (pasa por el punto (0, 1))
    P_FA, P_D = metrics.roc(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    assert P_FA[0] == 0 and P_D[0] == 0  # Umbral infinito
    assert np.any((P_FA == 0) & (P_D == 1))
    assert P_FA[-1] == 1 and P_D[-1] == 1  # Umbral mínimo: siempre H1
