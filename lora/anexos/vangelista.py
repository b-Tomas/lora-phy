"""Sistema con la Ec. 2 tal como está escrita en el paper (Anexo C).

La Ec. 2 escribe la fase como la frecuencia multiplicada por el tiempo, 2π f(k) kT, sin el 1/2 del
término cuadrático. Su fase crece con k^2 en lugar de k^2/2:

    c_V(nT_s + kT) = e^{j2π (s + k) k / 2^SF} / sqrt(2^SF) = c(nT_s + kT) e^{jπ k^2 / 2^SF},

donde c es el chirp de LoRa de `tx.modulator`. Su frecuencia barre de f_0 a f_0 + 2B por símbolo y,
muestreada a B, se repliega: recorre la banda dos veces.
"""

import numpy as np

from .. import simulation
from ..params import LoRaParams


def modulator(symbols: np.ndarray, p: LoRaParams, L: int = 1) -> np.ndarray:
    """Forma de onda de Ec. 2.

    En las muestras (k entero) el mod de Ec. 2 cambia la fase en un número entero de ciclos, así
    que se omite: da las mismas muestras y, entre muestras, la fase continua de un chirp que barre
    2B por símbolo.

    Args:
        symbols (np.ndarray): Conjunto de símbolos $s(nT_s)$. 0 <= s < 2**SF.
        p (LoRaParams): Parámetros del sistema.
        L (int): Puntos por chip. Con L > 1 evalúa Ec. 2 también entre muestras, para graficar.

    Returns:
        np.ndarray: Señal modulada (compleja) en el dominio del tiempo.
    """

    s = symbols.reshape(-1, 1)  # Vector columna de símbolos
    k = np.arange(p.M * L) / L  # Chips, con L puntos por chip
    return (np.exp(1j * 2 * np.pi * (s + k) * k / p.M) / np.sqrt(p.M)).ravel()


def demodulator(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Demodulador de Ec. 16: de-chirpeado con e^{-j2π k^2 / 2^SF}, FFT y máximo.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Símbolos demodulados.
    """

    k = np.arange(p.M)
    d = r.reshape(-1, p.M) * np.exp(-1j * 2 * np.pi * k**2 / p.M)
    return np.argmax(np.abs(np.fft.fft(d)) ** 2, axis=1)


def ber_vs_snr(
    p: LoRaParams, SNRs_dB: np.ndarray, h: np.ndarray, simulations: int, num_symbols: int = 100
) -> np.ndarray:
    """`simulation.ber_vs_snr` con el modulador y el demodulador de Ec. 2."""

    return simulation.ber_vs_snr(
        p, SNRs_dB, h, simulations, num_symbols, modulate=modulator, demodulate=demodulator
    )
