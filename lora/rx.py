"""Receptor: demodulador y decodificador."""

import numpy as np

from .params import LoRaParams


def downchirp(p: LoRaParams, n_symbols: int = 1, L: int = 1) -> np.ndarray:
    """Down-chirp del demodulador: e^{-j2π k^2 / (2 · 2^SF)} para k = 0..2^SF-1, repetido en cada
    símbolo.

    Es la señal que genera el demodulador y que multiplica a la señal recibida, sin r(nT_s + kT).
    Es el conjugado del up-chirp de `tx.modulator` (símbolo 0), así que cancela su término k^2/2 y
    deja d(nT_s + kT) = e^{j2π s k / 2^SF} / sqrt(2^SF), un tono de frecuencia constante s B / 2^SF.
    El down-chirp de Ec. 16, e^{-j2π k^2 / 2^SF}, corresponde a Ec. 2, cuya fase tiene k^2.

    Args:
        p (LoRaParams): Parámetros del sistema.
        n_symbols (int): Cantidad de símbolos.
        L (int): Puntos por chip. Con L > 1 evalúa la fórmula también entre muestras, para graficar.

    Returns:
        np.ndarray: Down-chirp (complejo), de n_symbols * 2^SF * L puntos.
    """

    k = np.arange(p.M * L) / L  # Chips, con L puntos por chip
    return np.tile(np.exp(-1j * 2 * np.pi * k**2 / (2 * p.M)), n_symbols)


def dechirp_fft(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Proyección de la señal recibida sobre los chirps ortogonales, como en Ec. 16.

    Multiplica cada símbolo recibido por el down-chirp de `downchirp` (Ec. 16 con k^2/2 en lugar de
    k^2, porque `tx.modulator` tiene k/2) y calcula su FFT. Es el demodulador sin la decisión:
    permite ver la salida de la FFT antes de decidir.

    Args:
        r (np.ndarray): Señal LoRa modulada recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: D[n, q] = <r, c|_{s=q}> para cada símbolo n, de forma (N, 2^SF).
    """

    # Validar el largo del array
    assert r.size % p.M == 0, "El número de muestras debe ser múltiplo de 2**SF"

    # Convertir r a un arreglo de forma (N, M): R[n, k] = r(nT_s + kT)
    R = r.reshape(-1, p.M)

    # Obtener d(nT_s + kT) = r(nT_s + kT) * downchirp
    d = R * downchirp(p)

    # Transformada de Fourier (discreta) de cada fila (símbolo)
    # norm="ortho" normaliza la FFT para que la salida sea unitaria (conserve la energía de la
    # señal), equivalente a aplicar el factor 1/sqrt(M) de Ec. 16 a la salida de la FFT
    return np.fft.fft(d, norm="ortho")


def demodulator(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Demodula una señal LoRa con el método de Ec. 16 (ver `dechirp_fft`).

    Args:
        r (np.ndarray): Señal LoRa modulada recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Arreglo unidimensional de símbolos demodulados.
    """

    # Decidir el símbolo demodulado como el índice del máximo de la FFT
    return np.argmax(np.abs(dechirp_fft(r, p)) ** 2, axis=1)


def decoder(symbols: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Convierte un conjunto de símbolos en bits.

    Descompone cada símbolo en un arreglo de SF bits de acuerdo a Ec. 1.

    Args:
        symbols (np.ndarray): Conjunto de símbolos a decodificar.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Conjunto de bits decodificados.
    """

    # Convertir los símbolos a un arreglo columna de forma (N, 1) para poder resolver como
    # una operación matricial
    s = symbols.reshape(-1, 1)  # Vector columna de símbolos

    # Secuencia h = 0..SF-1
    h = np.arange(p.SF)

    # Expande cada vector de un símbolo a un arreglo de SF bits
    # Se desplaza el símbolo h veces a la derecha para cada valor de h, para cada símbolo
    # Nos quedamos con el bit menos significativo
    W = (s >> h) & 1

    # Aplanar a un vector unidimensional de bits
    return W.ravel()
