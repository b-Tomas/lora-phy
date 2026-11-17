"""Modelo del canal: respuesta al impulso (SLIT) y ruido blanco gaussiano aditivo (AWGN)."""

from math import sqrt

import numpy as np

from . import params
from .metrics import power
from .params import LoRaParams
from .tx import modulator, preamble

# Canal plano: respuesta al impulso neutra (solo AWGN)
FLAT = np.array([1.0])

# Canal selectivo en frecuencia (multicamino): h[n] = sqrt(0.8) δ[n] + sqrt(0.2) δ[n-1]
FREQ_SELECTIVE = np.array([sqrt(0.8), sqrt(0.2)])


def exponential(alpha: float, L: int) -> np.ndarray:
    """Respuesta al impulso exponencial de L caminos: h[n] = e^{-alpha n} para 0 <= n < L.

    Se normaliza a energía 1 (sum h[n]^2 = 1), igual que FLAT y FREQ_SELECTIVE, para que el canal
    no cambie la potencia media de la señal y el SNR siga siendo el de la trama transmitida.

    Args:
        alpha (float): Tasa de caída por muestra. Cuanto menor, más se reparte la energía entre
            los caminos.
        L (int): Cantidad de caminos (largo de la respuesta al impulso).

    Returns:
        np.ndarray: Respuesta al impulso h[n].
    """

    h = np.exp(-alpha * np.arange(L))
    return h / np.sqrt(np.sum(h**2))


# Canales exponenciales: uno que concentra la energía en los primeros caminos y otro que la reparte
EXP_FAST = exponential(0.5, 8)
EXP_SLOW = exponential(0.1, 16)


def noise_from_SNR(c: np.ndarray, SNR_dB: float) -> np.ndarray:
    """Genera ruido blanco gaussiano aditivo (AWGN) a partir de la señal y la relación señal-ruido.

    Args:
        c (np.ndarray): Señal modulada (compleja) en el dominio del tiempo.
        SNR_dB (float): Relación señal-ruido en decibelios.

    Returns:
        np.ndarray: Ruido complejo, de la misma forma que `c`.
    """

    # Convertir SNR de dB a lineal
    SNR_linear = 10 ** (SNR_dB / 10)

    # Potencia del ruido: SNR = signal_power / noise_power => noise_power = signal_power / SNR
    z_power = power(c) / SNR_linear

    # La potencia del ruido gaussiano es igual a la varianza de la distribución normal
    # Se divide la potencia a partes iguales entre la parte real e imaginaria del ruido complejo
    # La desviación estándar es la raíz cuadrada de la varianza
    component_sigma = np.sqrt(z_power / 2)

    # Generar ruido blanco gaussiano.
    # Tiene que ser complejo, por lo que se genera ruido para la parte real e imaginaria.
    return params.rng.normal(0, component_sigma, c.shape) + 1j * params.rng.normal(
        0, component_sigma, c.shape
    )


def noise_power(p: LoRaParams, SNR_dB: float) -> float:
    """Potencia del ruido por muestra ($\\sigma^2$) para un SNR respecto de la potencia de la trama.

    Es la potencia del ruido que generan `transmit_frame` y `noise_from_SNR` sobre el preámbulo.
    Los detectores asumen que el receptor la conoce; en la práctica se estima mientras no hay
    señal.

    Args:
        p (LoRaParams): Parámetros del sistema.
        SNR_dB (float): Relación señal-ruido en decibelios.

    Returns:
        float: Potencia del ruido por muestra.
    """

    return power(preamble(p)) / 10 ** (SNR_dB / 10)


def system(c: np.ndarray, h: np.ndarray, z: np.ndarray | float) -> np.ndarray:
    """Simula la transmisión de la señal modulada a través de un canal ruidoso.

    Args:
        c (np.ndarray): Señal modulada (compleja) en el dominio del tiempo.
        h (np.ndarray): Respuesta al impulso del canal.
        z (np.ndarray | float): Ruido blanco gaussiano aditivo (AWGN), o 0 para un canal sin
            ruido.

    Returns:
        np.ndarray: Señal recibida después de pasar por el canal ruidoso.
    """

    # Convolución causal con la respuesta al impulso: r[n] = sum_l h[l] c[n - l]. Se descartan
    # las últimas L - 1 muestras para que la salida tenga el mismo tamaño que la entrada
    c_channel = np.convolve(c, h)[: c.size]

    # Sumar el ruido a la señal distorsionada por el canal
    return c_channel + z


def transmit_frame(
    symbols: np.ndarray,
    gap: int,
    p: LoRaParams,
    SNR_dB: float,
    h: np.ndarray,
    offset: int = 0,
) -> np.ndarray:
    """Simula la transmisión de una trama a través de un canal ruidoso.

    La trama es el preámbulo seguido de los símbolos del payload. Antes de la trama, el receptor
    escucha `gap` símbolos y `offset` muestras de solo ruido, por lo que no sabe en qué muestra
    empieza la trama: tau = gap * 2^SF + offset.

    Args:
        symbols (np.ndarray): Símbolos del payload.
        gap (int): Cantidad de símbolos de silencio antes de la trama.
        p (LoRaParams): Parámetros del sistema.
        SNR_dB (float): Relación señal-ruido en decibelios, respecto de la potencia de la trama.
        h (np.ndarray): Respuesta al impulso del canal.
        offset (int): Muestras de silencio adicionales, para un inicio que no es múltiplo de un
            símbolo.

    Returns:
        np.ndarray: Señal recibida después del canal ruidoso.
    """

    frame = np.concatenate([preamble(p), modulator(symbols, p)])

    # Silencio antes de la trama: el transmisor no emite nada
    c = np.concatenate([np.zeros(gap * p.M + offset), frame])

    # El SNR se define respecto de la potencia de la trama, no del silencio
    # np.resize repite la trama hasta el largo de c: misma potencia y misma forma que c
    z = noise_from_SNR(np.resize(frame, c.size), SNR_dB)

    return system(c, h, z)
