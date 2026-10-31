import numpy as np

from lora import LoRaParams, channel, detection, simulation


def test_frame_rates_ideal(p):
    # Con la detección ideal y sin ruido, todas las tramas son correctas
    p_start, fer = simulation.frame_rates(
        detection.ideal_detector(5), [100], channel.FLAT, p, 1e-3, 3
    )
    assert p_start[0] == 1 and fer[0] == 0


def test_ber_vs_snr_noiseless(p):
    assert np.all(simulation.ber_vs_snr(p, np.array([100.0]), channel.FLAT, 2) == 0)


def test_roc_curves_shape(p):
    axis = np.logspace(-7, 0, 10)
    curves = simulation.roc_curves(-14, p, 20, axis, 1e-5)
    assert set(curves) == {"full", "peaks", "freq"}
    assert all(c.P_D_theory.shape == axis.shape for c in curves.values())


def test_measured_false_alarms_keys(p):
    res = simulation.measured_false_alarms(p, 200, np.logspace(-1, -2, 3))
    assert set(res) == {"full", "peaks", "freq"}


def test_methods_cpu_time_keys():
    times = simulation.methods_cpu_time(LoRaParams(7, 125e3), 1e-5)
    assert set(times) == {"full", "peaks", "freq"}
    assert all(t > 0 for t in times.values())
