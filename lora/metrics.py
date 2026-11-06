"""Métricas de la señal y de la transmisión."""

import numpy as np

from .params import LoRaParams
from .rx import demodulator


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


def frame_error(r: np.ndarray, start: int | None, symbols_tx: np.ndarray, p: LoRaParams) -> bool:
    """Determina si una trama se recibe con error.

    Demodula el payload a partir de la muestra de inicio detectada y lo compara con el transmitido.

    Args:
        r (np.ndarray): Señal recibida.
        start (int | None): Muestra de inicio de la trama detectada, o None si no se detectó.
        symbols_tx (np.ndarray): Símbolos del payload transmitido.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        bool: True si la trama tiene error.
    """

    # Sin payload no hay nada que comparar: cualquier inicio daría una trama correcta
    assert symbols_tx.size > 0, "El payload no puede estar vacío"

    # Miss: no se detectó el preámbulo, se pierde la trama
    if start is None:
        return True

    # El payload empieza después del preámbulo, cuya estructura conoce el receptor
    begin = start + p.preamble_len
    payload = r[begin : begin + symbols_tx.size * p.M]

    # Detección tardía: el payload queda fuera de la señal recibida
    if payload.size < symbols_tx.size * p.M:
        return True

    # Basta un símbolo incorrecto para descartar la trama
    # Comparar símbolos equivale a comparar bits: el codificador es una biyección
    return not np.array_equal(demodulator(payload, p), symbols_tx)


def roc(statistic_h0: np.ndarray, statistic_h1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Curva ROC empírica a partir de realizaciones de un estadístico bajo cada hipótesis.

    Cada valor observado del estadístico es un umbral posible. Se decide H1 si el estadístico
    alcanza el umbral.

    Args:
        statistic_h0 (np.ndarray): Realizaciones del estadístico bajo H0.
        statistic_h1 (np.ndarray): Realizaciones del estadístico bajo H1.

    Returns:
        tuple[np.ndarray, np.ndarray]: P_FA y P_D para cada umbral, de mayor a menor umbral.
    """

    # Umbrales de mayor a menor. Un umbral infinito agrega el punto (0, 0): nunca se decide H1
    thresholds = np.concatenate(
        [[np.inf], np.unique(np.concatenate([statistic_h0, statistic_h1]))[::-1]]
    )

    # Fracción de realizaciones que alcanzan cada umbral (filas: realizaciones, columnas: umbrales)
    P_FA = np.mean(statistic_h0[:, None] >= thresholds, axis=0)
    P_D = np.mean(statistic_h1[:, None] >= thresholds, axis=0)

    return P_FA, P_D
