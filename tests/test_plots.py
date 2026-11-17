import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from lora import LoRaParams, channel, detection, plots, rng, rx, simulation, tx


def test_every_plot_returns_a_figure():
    p = LoRaParams(7, 125e3)
    gap, SNR_dB, P_FA = 2, -10, 1e-5
    symbols = np.array([0, 32, 80])
    c = tx.modulator(symbols, p)
    payload = rng.integers(0, p.M, size=4)
    r = channel.transmit_frame(payload, gap, p, SNR_dB, channel.FLAT)
    r_clean = channel.transmit_frame(payload, gap, p, 100, channel.FLAT)
    npw = channel.noise_power(p, SNR_dB)
    y, y_clean = detection.upchirp_correlation(r, p), detection.upchirp_correlation(r_clean, p)
    T = detection.preamble_correlation(r, p)
    S = detection.peaks_statistic(r, p)
    V = detection.preamble_zero_count(r, p)
    snrs = np.array([-12.0, -10.0])
    axis = np.logspace(-7, 0, 10)
    by_method = {m: np.array([0.5, 0.9]) for m in ("full", "peaks", "freq", "ideal")}

    figures = [
        plots.modulated_signal(symbols, p),
        plots.modulated_signal(symbols, p, tx.paper_modulator, "c"),
        plots.fft_decision({"Canal plano": rx.dechirp_fft(c, p)}, symbols, p),
        plots.fft_decision(
            {"A": rx.dechirp_fft(c, p), "B": rx.dechirp_fft(c, p)}, symbols, p, SNR_dB
        ),
        plots.dechirp_steps(c[: p.M], 0, p),
        plots.dechirp_steps(c[: p.M], 0, p, SNR_dB),
        plots.dechirp_steps(
            channel.system(c[: p.M], channel.FREQ_SELECTIVE, 0), 0, p, h=channel.FREQ_SELECTIVE
        ),
        plots.ber_vs_snr(snrs, {"Canal plano": np.array([0.1, 0.01])}),
        plots.frame_structure(payload, gap, p),
        plots.upchirp_correlation(y_clean, y, gap, p, SNR_dB),
        plots.full_preamble_detection(
            T,
            detection.threshold("full", p, npw, P_FA),
            gap * p.M,
            gap,
            p,
            P_FA,
            SNR_dB,
        ),
        plots.peaks_detection(
            y, S, detection.threshold("peaks", p, npw, P_FA), None, gap, p, P_FA, SNR_dB
        ),
        plots.freq_detection(
            rx.demodulator(r[: r.size // p.M * p.M], p), V, 4, gap, gap, p, P_FA, SNR_dB
        ),
        plots.compare_methods(snrs, by_method, by_method, "P(inicio correcto)", "Título"),
        plots.compare_methods(snrs, by_method, by_method, "FER", "Título", logy=True),
        plots.roc({-14: simulation.roc_curves(p, -14, 10, axis, P_FA)}, axis, P_FA),
        plots.false_alarms(simulation.measured_false_alarms(p, 50, np.logspace(-1, -2, 3)), 50),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)
    plt.close("all")


def test_sdr_plots_return_figures():
    p = LoRaParams(7, 125e3)
    gap = 2
    r = 1e3 * channel.transmit_frame(rng.integers(0, p.M, size=4), gap, p, -5, channel.FLAT)
    snrs, fer = np.array([-12.0, -10.0]), np.array([0.5, 0.0])

    figures = [
        plots.snr_vs_atten(np.array([-10.0, -30.0]), snrs, snrs),
        plots.fer_sdr(snrs, fer, snrs, fer, 50),
        plots.sdr_tone(np.array([r, 2 * r]), np.array([30.0, 50.0]), 3, p),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)
    plt.close("all")
