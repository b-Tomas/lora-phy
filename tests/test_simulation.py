import numpy as np

from lora import LoRaParams, channel, detection, simulation


def test_frame_rates_ideal(p):
    # Con la detección ideal y sin ruido, todas las tramas son correctas
    p_start, fer = simulation.frame_rates(
        detection.ideal_detector(3), np.array([100.0]), channel.FLAT, p, 1e-3, 3, gap=3
    )
    assert p_start[0] == 1 and fer[0] == 0


def test_frame_rates_min_errors(p, monkeypatch):
    # Sin señal útil todas las tramas tienen error: corta a los 5 errores, no a las 1000 tramas
    frames = []
    transmit = simulation.transmit_frame
    monkeypatch.setattr(simulation, "transmit_frame", lambda *a: frames.append(1) or transmit(*a))
    _, fer = simulation.frame_rates(
        detection.ideal_detector(3),
        np.array([-60.0]),
        channel.FLAT,
        p,
        1e-3,
        1000,
        gap=3,
        min_errors=5,
    )
    assert len(frames) == 5 and fer[0] == 1


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


def test_methods_cpu_time_keys(monkeypatch):
    # Se reemplaza la medición, lenta, por una llamada: solo se prueba que cada método corre
    monkeypatch.setattr(simulation, "cpu_time", lambda fn: (fn(), 1e-3)[1])
    times = simulation.methods_cpu_time(LoRaParams(7, 125e3), 1e-5)
    assert set(times) == {"full", "peaks", "freq"}
    assert all(t > 0 for t in times.values())
