"""Detección de una trama: estadísticos, umbrales de Neyman-Pearson y detectores.

Tres métodos, con la misma probabilidad de falsa alarma por símbolo:

1. Correlación con el preámbulo completo (tiempo, integración coherente): `detect_full`.
2. Búsqueda de 8 picos (tiempo, integración no coherente): `detect_peaks`.
3. Demodulación del preámbulo (frecuencia, decisión dura por ventana): `detect_freq`.

Todos los detectores tienen la firma `detect(r, p, noise_power, P_FA)` (`Detector`) y devuelven
la muestra de inicio de la trama, o None si no la detectan. `threshold` da el umbral de cada método
para una P_FA por símbolo.
"""

from collections.abc import Callable
from math import comb, factorial

import numpy as np

from .params import LoRaParams
from .rx import demodulator
from .tx import modulator

Detector = Callable[[np.ndarray, LoRaParams, float, float], int | None]


def detect_start(statistic: np.ndarray, threshold: float, span: int) -> int | None:
    """Detecta el inicio de la trama a partir de un estadístico de detección.

    Declara la trama presente (H1) en la primera posición donde el estadístico alcanza el umbral.
    Como las posiciones anteriores al inicio ya se superponen con parte del preámbulo, el primer
    cruce suele llegar antes de tiempo: el inicio es el máximo del estadístico en las `span`
    posiciones siguientes al cruce. La búsqueda se repite desde ese máximo hasta que ninguna de
    las `span` posiciones siguientes lo supere, por si el cruce llegó más de `span` posiciones
    antes del inicio (en el SDR, la cola de la trama anterior puede superar el umbral).

    Args:
        statistic (np.ndarray): Estadístico de detección de cada posición (muestra o ventana).
        threshold (float): Umbral de decisión.
        span (int): Posiciones donde se busca el máximo después del cruce: el largo de los
            up-chirps del preámbulo, en ventanas (n_upchirps) o en muestras (n_upchirps * 2^SF).

    Returns:
        int | None: Posición de inicio de la trama, o None si no se detecta.
    """

    # Posiciones que alcanzan el umbral
    above = np.flatnonzero(statistic >= threshold)
    if above.size == 0:
        return None  # No se detecta la trama: H0 en todas las posiciones

    # Máximo del estadístico a partir del primer cruce, hasta que no lo supere ninguna de las
    # `span` posiciones siguientes
    start = int(above[0])
    while (peak := start + int(np.argmax(statistic[start : start + span]))) != start:
        start = peak
    return start


def upchirp_correlation(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Correlación móvil de la señal recibida con un up-chirp (símbolo 0).

    Calcula y[k] = <r_k, c|_{s=0}> para cada muestra k, donde r_k son las 2^SF muestras recibidas
    a partir de k.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Correlación y[k] para cada muestra k donde entra el up-chirp completo.
    """

    # Más corta que la plantilla, np.correlate intercambiaría los argumentos
    assert r.size >= p.M, "La señal es más corta que un símbolo"

    upchirp = modulator(np.zeros(1, dtype=int), p)

    # np.correlate conjuga el segundo argumento: y[k] = sum_i r[k + i] * conj(upchirp[i])
    return np.correlate(r, upchirp, mode="valid")


# --- Tiempo: correlación con el preámbulo completo -------------------------------------------


def preamble_correlation(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Estadístico de la correlación con el preámbulo completo.

    Calcula T[k] = |<r_k, p>|^2 para cada muestra k (filtro adaptado), donde p son los
    `p.n_upchirps` up-chirps del preámbulo y r_k es la señal recibida a partir de la muestra k.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Estadístico T[k] para cada muestra k donde entra el preámbulo completo.
    """

    # Más corta que la plantilla, no hay ninguna posición donde entre el preámbulo completo
    assert r.size >= p.n_upchirps * p.M, "La señal es más corta que los up-chirps del preámbulo"

    template = modulator(np.zeros(p.n_upchirps, dtype=int), p)

    # Correlación por FFT: mismo resultado que np.correlate(r, template, mode="valid"), pero mucho
    # más rápida con una plantilla de n_upchirps * M muestras. Con n >= r.size la correlación
    # circular no se solapa en las posiciones válidas; potencia de 2 para que la FFT sea rápida
    n = 1 << (r.size - 1).bit_length()
    y = np.fft.ifft(np.fft.fft(r, n) * np.conj(np.fft.fft(template, n)))
    y = y[: r.size - template.size + 1]

    # Equivale a sumar los picos y[k + n M] de upchirp_correlation antes del módulo (coherente)
    return np.abs(y) ** 2


def time_threshold(noise_power: float, P_FA: float, p: LoRaParams) -> float:
    """Umbral de Neyman-Pearson para el estadístico de `preamble_correlation`.

    Bajo H0, T[k] tiene distribución exponencial de media n_upchirps * noise_power.

    Args:
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        P_FA (float): Probabilidad de falsa alarma por muestra.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        float: Umbral eta tal que P(T[k] > eta | H0) = P_FA.
    """

    # ||p||^2 = n_upchirps: cada chirp tiene energía 1
    return -p.n_upchirps * noise_power * np.log(P_FA)


def detect_full(r: np.ndarray, p: LoRaParams, noise_power: float, P_FA: float) -> int | None:
    """Detecta la muestra de inicio de la trama con la correlación con el preámbulo completo.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        P_FA (float): Probabilidad de falsa alarma por símbolo.

    Returns:
        int | None: Muestra de inicio de la trama, o None si no se detecta.
    """

    eta = threshold("full", p, noise_power, P_FA)

    # El máximo se busca en el largo de los up-chirps del preámbulo
    return detect_start(preamble_correlation(r, p), eta, p.n_upchirps * p.M)


# --- Tiempo: búsqueda de 8 picos -------------------------------------------------------------


def peaks_statistic(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Estadístico de la búsqueda de picos: energía de `p.n_upchirps` picos separados un símbolo.

    Calcula S[k] = sum_n |y[k + n 2^SF]|^2 a partir de la correlación móvil con un up-chirp.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Estadístico S[k] para cada muestra k donde entran los `p.n_upchirps` picos.
    """

    assert r.size >= p.n_upchirps * p.M, "La señal es más corta que los up-chirps del preámbulo"

    # Energía de cada pico: el módulo se toma antes de sumar (no coherente)
    y2 = np.abs(upchirp_correlation(r, p)) ** 2

    # Posiciones k donde entran los n_upchirps picos: mismo largo que preamble_correlation
    n_valid = y2.size - (p.n_upchirps - 1) * p.M

    # Suma de los picos y[k + n M], desplazando el arreglo un símbolo por cada pico
    # start: la suma arranca de un arreglo de ceros, no del entero 0
    return sum(
        (y2[n * p.M : n * p.M + n_valid] for n in range(p.n_upchirps)), start=np.zeros(n_valid)
    )


def peaks_threshold(noise_power: float, P_FA: float, p: LoRaParams) -> float:
    """Umbral de Neyman-Pearson para el estadístico de `peaks_statistic`.

    Bajo H0, S[k] tiene distribución Gamma de orden n_upchirps y escala noise_power. Su cola no
    se puede invertir en forma cerrada: el umbral se busca por bisección.

    Args:
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        P_FA (float): Probabilidad de falsa alarma por muestra.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        float: Umbral eta_S tal que P(S[k] > eta_S | H0) = P_FA.
    """

    def tail(x: float) -> float:
        # P(S[k] > x sigma^2 | H0): cola de la Gamma de orden entero (Erlang)
        return np.exp(-x) * sum(x**i / factorial(i) for i in range(p.n_upchirps))

    # Intervalo [lo, hi] que contiene la solución: se duplica hi hasta que la cola baje de P_FA
    lo, hi = 0.0, 1.0
    while tail(hi) > P_FA:
        lo, hi = hi, 2 * hi

    # Bisección: la cola es decreciente en x
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) > P_FA else (lo, mid)

    return hi * noise_power


def detect_peaks(r: np.ndarray, p: LoRaParams, noise_power: float, P_FA: float) -> int | None:
    """Detecta la muestra de inicio de la trama con la búsqueda de 8 picos.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        P_FA (float): Probabilidad de falsa alarma por símbolo.

    Returns:
        int | None: Muestra de inicio de la trama, o None si no se detecta.
    """

    eta_S = threshold("peaks", p, noise_power, P_FA)

    # El máximo se busca en el largo de los up-chirps del preámbulo
    return detect_start(peaks_statistic(r, p), eta_S, p.n_upchirps * p.M)


# --- Frecuencia: demodulación del preámbulo --------------------------------------------------


def preamble_zero_count(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Estadístico de detección en frecuencia: ventanas consecutivas decididas como símbolo 0.

    Demodula cada ventana (down-chirp y FFT, Ec. 16) y cuenta cuántas de las `p.n_upchirps`
    ventanas a partir de m se decidieron como símbolo 0.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Estadístico V[m] (ventanas con símbolo 0) para cada ventana m donde entra el
        preámbulo completo.
    """

    # Más corta que el preámbulo, np.convolve intercambiaría los argumentos
    assert r.size >= p.n_upchirps * p.M, "La señal es más corta que los up-chirps del preámbulo"

    # Demodular cada ventana de M muestras (sincronía de símbolo). Se descarta el resto
    symbols_rx = demodulator(r[: r.size // p.M * p.M], p)

    # 1 si el pico de la FFT de la ventana está en el bin 0, como en un up-chirp del preámbulo
    is_zero = (symbols_rx == 0).astype(int)

    # Cantidad de ventanas con símbolo 0 en las n_upchirps ventanas a partir de m
    return np.convolve(is_zero, np.ones(p.n_upchirps, dtype=int), mode="valid")


def zero_count_false_alarm(K: int, p: LoRaParams) -> float:
    """Probabilidad de falsa alarma del estadístico de `preamble_zero_count` con umbral K.

    Bajo H0, cada ventana se decide como símbolo 0 con probabilidad 2^-SF, así que V[m] tiene
    distribución binomial B(n_upchirps, 2^-SF).

    Args:
        K (int): Umbral de decisión.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        float: P(V[m] >= K | H0).
    """

    # Probabilidad de decidir símbolo 0 en una ventana de solo ruido: el pico cae en cualquier bin
    p0 = 2.0**-p.SF
    n = p.n_upchirps

    # Cola de la binomial: suma de P(V[m] = k | H0) para k = K, ..., n_upchirps
    return sum(comb(n, k) * p0**k * (1 - p0) ** (n - k) for k in range(K, n + 1))


def freq_threshold(P_FA: float, p: LoRaParams) -> int:
    """Umbral de Neyman-Pearson para el estadístico de `preamble_zero_count`.

    Como V[m] es discreto, se elige el menor umbral (mayor P_D) cuya probabilidad de falsa alarma
    no supera P_FA.

    Args:
        P_FA (float): Probabilidad de falsa alarma por ventana.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        int: Menor umbral K tal que P(V[m] >= K | H0) <= P_FA.
    """

    # Con K = n_upchirps + 1 la cola es 0 (nunca se decide H1), así que siempre hay solución
    return next(K for K in range(p.n_upchirps + 2) if zero_count_false_alarm(K, p) <= P_FA)


def detect_freq(r: np.ndarray, p: LoRaParams, noise_power: float, P_FA: float) -> int | None:
    """Detecta la muestra de inicio de la trama con la demodulación del preámbulo.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.
        noise_power (float): No se usa: el umbral no depende de sigma^2. Se recibe para tener la
            misma firma que los detectores en el tiempo.
        P_FA (float): Probabilidad de falsa alarma por símbolo (un test por ventana).

    Returns:
        int | None: Muestra de inicio de la trama, o None si no se detecta.
    """

    K = threshold("freq", p, noise_power, P_FA)
    start = detect_start(preamble_zero_count(r, p), K, p.n_upchirps)

    # La ventana m empieza en la muestra m * 2^SF
    return None if start is None else start * p.M


# --- Umbral por símbolo ----------------------------------------------------------------------


def threshold(method: str, p: LoRaParams, noise_power: float, P_FA: float) -> float:
    """Umbral de Neyman-Pearson de un método para una probabilidad de falsa alarma por símbolo.

    En el tiempo se prueban 2^SF muestras por símbolo, así que cada muestra usa P_FA / 2^SF: misma
    tasa de falsas alarmas que un test por símbolo. En frecuencia hay un test por ventana.

    Args:
        method (str): "full" (preámbulo completo), "peaks" (búsqueda de picos) o "freq"
            (demodulación del preámbulo).
        p (LoRaParams): Parámetros del sistema.
        noise_power (float): Potencia del ruido por muestra (sigma^2). No se usa en "freq".
        P_FA (float): Probabilidad de falsa alarma por símbolo.

    Returns:
        float: Umbral del estadístico del método.
    """

    if method == "full":
        return time_threshold(noise_power, P_FA / p.M, p)
    if method == "peaks":
        return peaks_threshold(noise_power, P_FA / p.M, p)
    assert method == "freq", f"Método desconocido: {method}"
    return freq_threshold(P_FA, p)
