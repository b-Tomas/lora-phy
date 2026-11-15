import numpy as np

from lora import channel, detection, seed, simulation


def test_fer_noiseless(p):
    # Sin ruido el detector encuentra el inicio exacto de todas las tramas
    fer = simulation.fer_vs_snr(
        detection.detect_full, np.array([100.0]), channel.FLAT, p, 1e-3, 3, num_symbols=4, gap=3
    )
    assert fer[0] == 0


def test_fer_min_errors(p, monkeypatch):
    # Sin señal útil se pierden todas las tramas: corta a los 5 errores, no a las 1000 tramas
    frames = []
    transmit = simulation.transmit_frame
    monkeypatch.setattr(simulation, "transmit_frame", lambda *a: frames.append(1) or transmit(*a))
    fer = simulation.fer_at_snr(
        detection.detect_full,
        -60.0,
        channel.FLAT,
        p,
        1e-3,
        1000,
        num_symbols=4,
        gap=3,
        min_errors=5,
    )
    assert len(frames) == 5 and fer == 1


def test_fer_vs_snr_reproducible(p):
    # Con la misma semilla, la corrida en paralelo da el mismo FER
    args = (detection.detect_full, np.array([-14.0, -12.0]), channel.FLAT, p, 1e-3, 20)
    seed(0)
    fer = simulation.fer_vs_snr(*args, num_symbols=4, gap=3)
    seed(0)
    assert np.array_equal(fer, simulation.fer_vs_snr(*args, num_symbols=4, gap=3))


def test_ber_vs_snr_noiseless(p):
    assert np.all(simulation.ber_vs_snr(p, np.array([100.0]), channel.FLAT, 2) == 0)


def test_roc_curves_shape(p):
    axis = np.logspace(-7, 0, 10)
    curves = simulation.roc_curves(p, -14, 20, axis, 1e-5)
    assert set(curves) == {"full", "peaks", "freq"}
    assert all(c.P_D_theory.shape == axis.shape for c in curves.values())


def test_measured_false_alarms(p):
    # Los umbrales de Neyman-Pearson cumplen la P_FA pedida (por posición) sobre ruido
    targets = np.array([1e-1, 1e-2])
    res = simulation.measured_false_alarms(p, 2000, targets)
    assert set(res) == {"full", "peaks", "freq"}
    for method in ("full", "peaks"):
        target, measured = res[method]
        assert np.allclose(measured, target, rtol=0.2)
