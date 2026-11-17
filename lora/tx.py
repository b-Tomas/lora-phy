"""Transmisor: codificador y conformador de onda."""

import numpy as np

from .params import LoRaParams


def encoder(bits: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Convierte un conjunto de bits en símbolos.

    Organiza `bits` en grupos $w(nT_s)$ de SF bits y los convierte a símbolos según Ec. 1.

    Args:
        bits (np.ndarray): Conjunto de bits a transmitir.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Conjunto de símbolos a transmitir.
    """

    # Validar la forma del vector de bits a enviar
    assert bits.size % p.SF == 0, "El número de bits debe ser múltiplo de SF"

    # Reorganizar los bits en grupos de SF bits
    # El resultado es un arreglo de forma (N, SF) donde N es el número de símbolos.
    # Cada símbolo es un vector equivalente a $w(nT_s)$
    W = bits.reshape(-1, p.SF)

    # Secuencia h = 0..SF-1
    h = np.arange(p.SF)

    # Convertir cada grupo de SF bits a un símbolo
    # 2**h produce un vector de potencias de 2: [1, 2, 4, ..., 2**(SF-1)]
    # W @ 2**h realiza el producto matricial, sumando los productos de cada bit por su
    # correspondiente potencia de 2.
    return W @ 2**h


def instantaneous_frequency(symbols: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Frecuencia instantánea de la señal modulada.

    Para cada símbolo n y chip k: f(k) = ((s(nT_s) + k) mod 2^SF) B / 2^SF, una sola pasada de 0 a
    B por símbolo: el corchete de Ec. 2 y la frecuencia de `modulator`, cuya fase es su integral.

    Args:
        symbols (np.ndarray): Conjunto de símbolos $s(nT_s)$. 0 <= s < 2**SF.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Frecuencia instantánea de cada chip, en Hz, como vector unidimensional.
    """

    # Convertir s a un vector de forma (N, 1) para poder sumarle k
    s = symbols.reshape(-1, 1)  # Vector columna de símbolos

    # Vector de chips k = 0..M-1
    k = np.arange(p.M)  # Vector fila de chips

    return (((s + k) % p.M) * p.B / p.M).ravel()


def modulator(symbols: np.ndarray, p: LoRaParams, L: int = 1) -> np.ndarray:
    """Genera la forma de onda FSCM: el chirp de LoRa de cada símbolo.

    La frecuencia instantánea es ((s(nT_s) + k) mod 2^SF) B / 2^SF (`instantaneous_frequency`), una
    sola pasada de 0 a B por símbolo, y la fase es su integral. En las muestras vale

        c(nT_s + kT) = e^{j2π (s + k/2) k / 2^SF} / sqrt(2^SF).

    Ec. 2 multiplica la frecuencia por kT y omite el 1/2 del término cuadrático (Anexo A del
    notebook). Muestrea a k = 0..2**SF-1 chips por cada símbolo n.

    Args:
        symbols (np.ndarray): Conjunto de símbolos $s(nT_s)$ a transmitir. 0 <= s < 2**SF.
        p (LoRaParams): Parámetros del sistema.
        L (int): Puntos por chip. Con L > 1 evalúa la fase también entre muestras, para graficar.

    Returns:
        np.ndarray: Señal modulada (compleja) en el dominio del tiempo.
    """

    s = symbols.reshape(-1, 1)  # Vector columna de símbolos
    k = np.arange(p.M * L) / L  # Chips, con L puntos por chip

    # Integral de f de 0 a kT, en ciclos: (s + k/2) k / M, menos una vuelta por chip desde que f
    # llega a B (el mod). En las muestras esa resta es un número entero de ciclos.
    phase = (s + k / 2) * k / p.M - np.maximum(k - (p.M - s), 0)

    # c[n, k] = c(nT_s + kT), de forma (N, M), aplanado a un vector unidimensional de chips
    return (np.exp(1j * 2 * np.pi * phase) / np.sqrt(p.M)).ravel()


def preamble(p: LoRaParams) -> np.ndarray:
    """Genera el preámbulo de una trama LoRa.

    El preámbulo se compone de `p.n_upchirps` up-chirps (símbolos 0) seguidos de
    `p.n_downchirps` down-chirps, que marcan el fin del preámbulo.

    Args:
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Preámbulo modulado (complejo) en el dominio del tiempo.
    """

    # Up-chirps: símbolos 0 modulados
    upchirps = modulator(np.zeros(p.n_upchirps, dtype=int), p)

    # Down-chirps: conjugado de un up-chirp (frecuencia instantánea decreciente)
    # Se generan los down-chirps completos necesarios y se recorta la fracción sobrante
    n_full = int(np.ceil(p.n_downchirps))
    downchirps = np.conj(modulator(np.zeros(n_full, dtype=int), p))[: int(p.n_downchirps * p.M)]

    return np.concatenate([upchirps, downchirps])
