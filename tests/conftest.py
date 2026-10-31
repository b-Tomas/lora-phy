import pytest

from lora import LoRaParams, channel, rng


@pytest.fixture
def p():
    return LoRaParams(7, 125e3)


@pytest.fixture
def clean_frame(p):
    """Trama de 8 símbolos aleatorios con 3 símbolos de silencio, sin ruido apreciable."""
    symbols = rng.integers(0, p.M, size=8)
    gap = 3
    r = channel.transmit_frame(symbols, gap, p, 100, channel.FLAT)
    return r, symbols, gap, channel.noise_power(p, 100)
