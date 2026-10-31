"""Métricas de la señal y de la transmisión."""

import numpy as np


def power(c: np.ndarray) -> float:
    """Calcula la potencia de una señal.

    Args:
        c (np.ndarray): Señal (compleja) en el dominio del tiempo.

    Returns:
        float: Potencia de la señal.
    """

    # La potencia de una señal compleja es el valor medio de su magnitud al cuadrado
    return float(np.mean(np.abs(c) ** 2))


def BER(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """Calcula el Bit Error Rate (BER) comparando los bits transmitidos y recibidos.

    Args:
        bits_tx (np.ndarray): Conjunto de bits transmitidos.
        bits_rx (np.ndarray): Conjunto de bits recibidos.

    Returns:
        float: Bit Error Rate (BER).
    """

    # Validar que ambos conjuntos de bits tengan la misma longitud
    assert bits_tx.size == bits_rx.size, "Los conjuntos de bits deben tener la misma longitud"

    # Los bits erróneos son los 1 del XOR entre los bits transmitidos y recibidos
    return float(np.mean(bits_tx ^ bits_rx))
