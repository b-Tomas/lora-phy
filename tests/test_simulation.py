import numpy as np
import pytest

from lora import channel, detection, seed, simulation


def test_count_errors_runs_min_then_until_errors():
    # Hace el mínimo aunque ya tenga los errores, y después sigue hasta juntarlos
    calls = []

    def trial(e):
        calls.append(1)
        return e, 1

    assert simulation._count_errors(lambda: trial(1), 10, 1000, 5) == (10, 10)
    calls.clear()
    assert simulation._count_errors(lambda: trial(int(len(calls) % 4 == 0)), 3, 1000, 5) == (5, 17)


def test_count_errors_stops_at_max():
    assert simulation._count_errors(lambda: (0, 7), 2, 50, 100) == (0, 350)


def test_fer_noiseless(p):
    # Sin ruido el detector encuentra el inicio exacto de todas las tramas
    curves = {"full": {"detect": detection.detect_full, "h": channel.FLAT}}
    fer = simulation.fer_vs_snr(np.array([100.0]), curves, p, 1e-3, 4, 3, 3, 3, min_errors=1)
    er = fer["full"]
    assert er.rate[0] == 0 and er.trials[0] == 3 and not er.reliable[0]


def test_fer_without_signal(p):
    # Sin señal útil se pierden todas las tramas: el mínimo alcanza para juntar los errores
    curves = {"full": {"detect": detection.detect_full, "h": channel.FLAT}}
    er = simulation.fer_vs_snr(np.array([-60.0]), curves, p, 1e-3, 4, 3, 5, 1000, min_errors=5)
    assert er["full"].trials[0] == 5 and er["full"].rate[0] == 1 and er["full"].reliable[0]


def test_fer_vs_snr_reproducible(p):
    # Con la misma semilla, la corrida en paralelo da el mismo FER
    curves = {
        m: {"detect": d, "h": channel.FLAT}
        for m, d in (("full", detection.detect_full), ("freq", detection.detect_freq))
    }
    args = (np.array([-14.0, -12.0]), curves, p, 1e-3, 4, 3, 5, 20)
    seed(0)
    fer = simulation.fer_vs_snr(*args, min_errors=20)
    seed(0)
    again = simulation.fer_vs_snr(*args, min_errors=20)
    assert all(np.array_equal(fer[m].errors, again[m].errors) for m in curves)


def test_ber_vs_snr_noiseless(p):
    ber = simulation.ber_vs_snr(p, np.array([100.0]), {"plano": {"h": channel.FLAT}}, 2, 4)
    assert np.all(ber["plano"].rate == 0) and ber["plano"].trials[0] == 4 * 7 * 100


def test_roc_curves_shape(p):
    axis = np.logspace(-7, 0, 10)
    curves = simulation.roc_curves(p, -14, 20, axis, 1e-5)
    assert set(curves) == {"full", "peaks", "freq"}
    assert all(c.P_D_theory.shape == axis.shape for c in curves.values())


def test_measured_false_alarms(p):
    # Los umbrales de Neyman-Pearson cumplen la P_FA pedida (por posición) sobre ruido
    targets = np.array([1e-1, 1e-2])
    res = simulation.measured_false_alarms(p, 2000, targets)
    assert set(res) == {"full", "peaks", "freq", "ls", "lmmse"}
    for method in ("full", "peaks", "ls", "lmmse"):
        target, measured = res[method]
        assert np.allclose(measured, target, rtol=0.2)


def test_interval_contains_rate():
    er = simulation.ErrorRate(np.array([0, 5, 100, 300]), np.array([1000, 1000, 1000, 300]))
    lo, hi = er.interval()
    eps = 1e-12  # Redondeo en los extremos
    assert np.all((lo <= er.rate + eps) & (er.rate <= hi + eps) & (lo >= 0) & (hi <= 1))
    # Con 100 errores el intervalo es de aproximadamente ±20 %
    assert hi[2] / er.rate[2] == pytest.approx(1.2, abs=0.03)


def test_estimator_mse_matches_theory(p):
    SNR_dB = -15.0
    mse = simulation.estimator_mse(p, np.array([SNR_dB]), channel.EXP_FAST, 2000)
    ls, lmmse = detection.estimator_mse_theory(
        channel.EXP_FAST**2, channel.noise_power(p, SNR_dB), p
    )
    assert np.isclose(mse["LS"][0], ls, rtol=0.1)
    assert np.isclose(mse["LMMSE"][0], lmmse, rtol=0.1)
    assert lmmse < ls
