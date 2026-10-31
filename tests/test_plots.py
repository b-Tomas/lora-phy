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
    T, T_clean = detection.preamble_correlation(r, p), detection.preamble_correlation(r_clean, p)
    S = detection.peaks_statistic(r, p)
    V = detection.preamble_zero_count(r, p)
    snrs = np.array([-12.0, -10.0])
    axis = np.logspace(-7, 0, 10)
    by_method = {m: np.array([0.5, 0.9]) for m in ("full", "peaks", "freq", "ideal")}

    figures = [
        plots.modulated_signal(c, symbols, p),
        plots.fft_decision(rx.dechirp_fft(c, p), symbols, rx.demodulator(c, p)),
        plots.fft_decision(rx.dechirp_fft(c, p), symbols, rx.demodulator(c, p), SNR_dB),
        plots.ber_vs_snr(snrs, {"Canal plano": np.array([0.1, 0.01])}),
        plots.frame_structure(r, payload, gap, p, SNR_dB),
        plots.upchirp_correlation(y_clean, y, gap, p, SNR_dB),
        plots.full_preamble_detection(
            T_clean,
            T,
            detection.time_threshold(npw, P_FA / p.M, p),
            gap * p.M,
            gap,
            p,
            P_FA,
            SNR_dB,
        ),
        plots.peaks_detection(
            y, S, detection.peaks_threshold(npw, P_FA / p.M, p), None, gap, p, P_FA, SNR_dB
        ),
        plots.freq_detection(
            rx.demodulator(r[: r.size // p.M * p.M], p), V, 4, gap, gap, p, P_FA, SNR_dB
        ),
        plots.start_vs_snr(snrs, by_method["full"], by_method["peaks"], "Título"),
        plots.compare_methods(snrs, by_method, by_method, "FER", "Título", xlim=(-15, -4)),
        plots.false_alarms(simulation.measured_false_alarms(p, 50, np.logspace(-1, -2, 3)), 50),
        plots.roc({-14: simulation.roc_curves(-14, p, 10, axis, P_FA)}, axis, P_FA),
        plots.cpu_times(np.array([7, 8]), [{"full": 2e-3, "peaks": 5e-4, "freq": 1e-4}] * 2),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)
    plt.close("all")
