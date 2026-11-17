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
    # Con un solo impulso, S[k] es exponencial y el umbral coincide con el de T[k]
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
    # En frecuencia, la ventana de ruido anterior al preámbulo se decide como símbolo 0 con
    # probabilidad 2^-SF y V[m] empata con el inicio: se elige la primera ventana del empate
    tie = detect is detection.detect_freq and start == (gap - 1) * q.M
    assert start == gap * q.M or tie


def test_preamble_correlation_distributions(p):
    # Las colas teóricas coinciden con T[k] simulado bajo cada hipótesis
    npw, n = 0.4, 20_000
    m = p.n_upchirps * npw
    t = np.linspace(0, 40 * m, 4001)
    _, _, P_FA, P_D = detection.preamble_correlation_distributions(t, npw, p)
    noise = np.sqrt(m / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    for tail, T in ((P_FA, np.abs(noise) ** 2), (P_D, np.abs(p.n_upchirps + noise) ** 2)):
        for eta in (10 * m, 20 * m):
            assert np.isclose(np.interp(eta, t, tail), np.mean(T > eta), atol=0.01)


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


def test_preamble_xcorr_matches_correlation(p):
    r = rng.normal(size=3000) + 1j * rng.normal(size=3000)
    assert np.allclose(
        np.abs(detection.preamble_xcorr(r, p)) ** 2, detection.preamble_correlation(r, p)
    )


def test_channel_ls_recovers_h(p):
    r = channel.transmit_frame(np.array([5]), 3, p, 100, channel.EXP_FAST, offset=11)
    h_ls = detection.channel_ls(r, p, 3 * p.M + 11, channel.EXP_FAST.size)
    assert np.allclose(h_ls, channel.EXP_FAST, atol=0.02)  # Lóbulos laterales de la plantilla


def test_lmmse_weights(p):
    a = detection.lmmse_weights(np.array([1.0, 0.25]), 8.0, p)  # sigma^2 / 8 = 1
    assert np.allclose(a, [0.5, 0.2])


def test_estimator_mse_theory(p):
    ls, lmmse = detection.estimator_mse_theory(np.array([1.0, 0.25]), 8.0, p)
    assert np.isclose(ls, 2) and np.isclose(lmmse, 0.5 + 0.2)


def test_erlang_threshold_matches_peaks(p):
    assert np.isclose(
        detection.erlang_threshold(p.n_upchirps, 2.0, 1e-3), detection.peaks_threshold(2.0, 1e-3, p)
    )


def test_ls_flat_equals_full(p):
    r = channel.transmit_frame(rng.integers(0, p.M, 6), 4, p, -12, channel.FLAT)
    npw = channel.noise_power(p, -12)
    start_ls = detection.detect_ls(r, p, npw, 1e-3, P=np.array([1.0]))
    assert start_ls == detection.detect_full(r, p, npw, 1e-3)


def test_hypoexp_threshold_tail():
    means = channel.EXP_SLOW**2 + 0.1  # 16 medias distintas
    eta = detection.hypoexp_threshold(means, 1e-2)
    G = (rng.exponential(size=(200_000, means.size)) * means).sum(axis=1)
    assert np.isclose(np.mean(G > eta), 1e-2, rtol=0.1)


def test_hypoexp_threshold_single_mean():
    # Con una sola media es una exponencial: eta = -mu ln P_FA
    assert np.isclose(detection.hypoexp_threshold(np.array([2.0]), 1e-3), -2.0 * np.log(1e-3))


def test_energy_statistic_false_alarms(p):
    npw = 1.0
    r = np.sqrt(npw / 2) * (rng.normal(size=400 * p.M) + 1j * rng.normal(size=400 * p.M))
    a = detection.lmmse_weights(channel.EXP_FAST**2, npw, p)
    G = detection.energy_statistic(r, p, a)
    assert G.size == detection.preamble_xcorr(r, p).size - a.size + 1
    eta = detection.hypoexp_threshold(a * npw / p.n_upchirps, 1e-2)
    assert np.isclose(np.mean(G > eta), 1e-2, rtol=0.15)


@pytest.mark.parametrize("h", [channel.FREQ_SELECTIVE, channel.EXP_FAST, channel.EXP_SLOW])
def test_detect_ls_lmmse_offset(p, h):
    r = channel.transmit_frame(rng.integers(0, p.M, 4), 3, p, 100, h, offset=21)
    npw = channel.noise_power(p, 100)
    for detect in (detection.detect_ls, detection.detect_lmmse):
        assert detect(r, p, npw, 1e-3, P=h**2) == 3 * p.M + 21


def test_hypoexp_threshold_close_means():
    # Pesos LMMSE casi iguales (EXP_SLOW a -4 dB con SF 8): la fórmula cerrada cancela mal
    q = LoRaParams(8, 125e3)
    npw = channel.noise_power(q, -4)
    means = detection.lmmse_weights(channel.EXP_SLOW**2, npw, q) * npw / q.n_upchirps
    eta = detection.hypoexp_threshold(means, 1e-2)
    G = (rng.exponential(size=(200_000, means.size)) * means).sum(axis=1)
    assert np.isclose(np.mean(G > eta), 1e-2, rtol=0.1)


def test_hypoexp_threshold_rejects_zero_mean():
    # Un camino sin potencia da una media 0: tasa infinita en la uniformización
    with pytest.raises(AssertionError):
        detection.hypoexp_threshold(np.array([1.0, 0.0]), 1e-2)
