"""Receptor: demodulador y decodificador."""

import numpy as np

from .params import LoRaParams


def dechirp(r: np.ndarray, p: LoRaParams, L: int = 1) -> np.ndarray:
    """Multiplica cada símbolo recibido por el down-chirp (de-chirpeado).

    Args:
        r (np.ndarray): Señal LoRa modulada recibida.
        p (LoRaParams): Parámetros del sistema.
        L (int): Puntos por chip de `r`. Con L > 1 evalúa el down-chirp también entre muestras,
            para graficar.

    Returns:
        np.ndarray: d(nT_s + kT) = r(nT_s + kT) e^{-j2π k²/2^SF} para cada símbolo n, de forma
            (N, 2^SF L).
    """

    # Validar el largo del array
    assert r.size % (p.M * L) == 0, "El número de muestras debe ser múltiplo de L * 2**SF"

    # Convertir r a un arreglo de forma (N, M L): R[n, k] = r(nT_s + kT)
    R = r.reshape(-1, p.M * L)

    # Vector de chips k = 0..M-1, con L puntos por chip
    k = np.arange(p.M * L) / L  # Vector fila de chips

    # Obtener d(nT_s + kT) = r(nT_s + kT) * downchirp
    downchirp = np.exp(-1j * 2 * np.pi * k**2 / p.M)
    return R * downchirp


def dechirp_fft(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Proyección de la señal recibida sobre los chirps ortogonales, según Ec. 16.

    De-chirpea cada símbolo recibido y calcula su FFT. Es el demodulador sin la decisión: permite
    ver la salida de la FFT antes de decidir.

    Args:
        r (np.ndarray): Señal LoRa modulada recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: D[n, q] = <r, c|_{s=q}> para cada símbolo n, de forma (N, 2^SF).
    """

    # Transformada de Fourier (discreta) de cada fila (símbolo)
    # norm="ortho" normaliza la FFT para que la salida sea unitaria (conserve la energía de la
    # señal), equivalente a aplicar el factor 1/sqrt(M) de Ec. 16 a la salida de la FFT
    return np.fft.fft(dechirp(r, p), norm="ortho")


def demodulator(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Demodula una señal LoRa según Ec. 16.

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
