"""Detección de una trama: estadísticos, umbrales de Neyman-Pearson y detectores.

Cinco métodos, con la misma probabilidad de falsa alarma por símbolo:

1. Correlación con el preámbulo completo (tiempo, integración coherente): `detect_full`.
2. Búsqueda de 8 impulsos (tiempo, integración no coherente): `detect_peaks`.
3. Demodulación del preámbulo (frecuencia, decisión dura por ventana): `detect_freq`.
4. Energía de la estimación LS del canal (tiempo, suma de los caminos): `detect_ls`.
5. Energía ponderada por los pesos LMMSE (tiempo, suma de los caminos): `detect_lmmse`.

Todos los detectores tienen la firma `detect(r, p, noise_power, P_FA)` (`Detector`); los dos
últimos reciben además el perfil de potencia del canal, `P`, que se fija con `functools.partial`.
Devuelven la muestra de inicio de la trama, o None si no la detectan. `threshold` da el umbral de
los tres primeros métodos para una P_FA por símbolo.
"""

from collections.abc import Callable
from functools import lru_cache
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


def preamble_xcorr(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Correlación compleja con el preámbulo completo: X[k] = <r_k, p> (filtro adaptado).

    p son los `p.n_upchirps` up-chirps del preámbulo y r_k es la señal recibida a partir de la
    muestra k.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: X[k] para cada muestra k donde entra el preámbulo completo.
    """

    # Más corta que la plantilla, no hay ninguna posición donde entre el preámbulo completo
    assert r.size >= p.n_upchirps * p.M, "La señal es más corta que los up-chirps del preámbulo"

    template = modulator(np.zeros(p.n_upchirps, dtype=int), p)

    # Correlación por FFT: mismo resultado que np.correlate(r, template, mode="valid"), pero mucho
    # más rápida con una plantilla de n_upchirps * M muestras. Con n >= r.size la correlación
    # circular no se solapa en las posiciones válidas; potencia de 2 para que la FFT sea rápida
    n = 1 << (r.size - 1).bit_length()
    y = np.fft.ifft(np.fft.fft(r, n) * np.conj(np.fft.fft(template, n)))
    return y[: r.size - template.size + 1]


def preamble_correlation(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Estadístico de la correlación con el preámbulo completo: T[k] = |X[k]|^2.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Estadístico T[k] para cada muestra k donde entra el preámbulo completo.
    """

    # Equivale a sumar los impulsos y[k + n M] de upchirp_correlation antes del módulo (coherente)
    return np.abs(preamble_xcorr(r, p)) ** 2


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


# --- Tiempo: búsqueda de 8 impulsos -------------------------------------------------------------


def peaks_statistic(r: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Estadístico de la búsqueda de impulsos: energía de `p.n_upchirps` impulsos separados un símbolo.

    Calcula S[k] = sum_n |y[k + n 2^SF]|^2 a partir de la correlación móvil con un up-chirp.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Estadístico S[k] para cada muestra k donde entran los `p.n_upchirps` impulsos.
    """

    assert r.size >= p.n_upchirps * p.M, "La señal es más corta que los up-chirps del preámbulo"

    # Energía de cada impulso: el módulo se toma antes de sumar (no coherente)
    y2 = np.abs(upchirp_correlation(r, p)) ** 2

    # Posiciones k donde entran los n_upchirps impulsos: mismo largo que preamble_correlation
    n_valid = y2.size - (p.n_upchirps - 1) * p.M

    # Suma de los impulsos y[k + n M], desplazando el arreglo un símbolo por cada impulso
    # start: la suma arranca de un arreglo de ceros, no del entero 0
    return sum(
        (y2[n * p.M : n * p.M + n_valid] for n in range(p.n_upchirps)), start=np.zeros(n_valid)
    )


def _invert_tail(tail: Callable[[float], float], P_FA: float) -> float:
    """Resuelve tail(x) = P_FA por bisección, para una cola decreciente en x >= 0.

    Args:
        tail (Callable[[float], float]): Cola P(G > x | H0) de un estadístico normalizado.
        P_FA (float): Probabilidad de falsa alarma buscada.

    Returns:
        float: x tal que tail(x) = P_FA.
    """

    # Intervalo [lo, hi] que contiene la solución: se duplica hi hasta que la cola baje de P_FA
    lo, hi = 0.0, 1.0
    while tail(hi) > P_FA:
        lo, hi = hi, 2 * hi

    # Bisección: la cola es decreciente en x
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) > P_FA else (lo, mid)

    return hi


def erlang_threshold(order: int, scale: float, P_FA: float) -> float:
    """Umbral de Neyman-Pearson para una suma de `order` exponenciales independientes de media `scale`.

    La suma tiene distribución Gamma de orden entero (Erlang). Su cola no se puede invertir en
    forma cerrada: el umbral se busca por bisección.

    Args:
        order (int): Cantidad de exponenciales sumadas.
        scale (float): Media de cada exponencial.
        P_FA (float): Probabilidad de falsa alarma.

    Returns:
        float: Umbral eta tal que P(suma > eta) = P_FA.
    """

    def tail(x: float) -> float:
        # P(suma > x scale): cola de la Erlang
        return np.exp(-x) * sum(x**i / factorial(i) for i in range(order))

    return _invert_tail(tail, P_FA) * scale


def peaks_threshold(noise_power: float, P_FA: float, p: LoRaParams) -> float:
    """Umbral de Neyman-Pearson para el estadístico de `peaks_statistic`.

    Bajo H0, S[k] es la suma de n_upchirps exponenciales independientes de media noise_power:
    Gamma de orden n_upchirps y escala noise_power (`erlang_threshold`).

    Args:
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        P_FA (float): Probabilidad de falsa alarma por muestra.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        float: Umbral eta_S tal que P(S[k] > eta_S | H0) = P_FA.
    """

    return erlang_threshold(p.n_upchirps, noise_power, P_FA)


def detect_peaks(r: np.ndarray, p: LoRaParams, noise_power: float, P_FA: float) -> int | None:
    """Detecta la muestra de inicio de la trama con la búsqueda de 8 impulsos.

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

    # 1 si el impulso de la FFT de la ventana está en el bin 0, como en un up-chirp del preámbulo
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

    # Probabilidad de decidir símbolo 0 en una ventana de solo ruido: el máximo cae en cualquier bin
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


# --- Estimación del canal: LS y LMMSE --------------------------------------------------------
#
# Si la trama empieza en tau y el canal tiene L caminos, r[n] = sum_l h[l] c[n - tau - l] + z[n].
# Con la plantilla alineada con el camino l (k = tau + l), sus 8 up-chirps se suman en fase y
# aportan 8 h[l] a X[k]. Los demás caminos quedan corridos: un up-chirp corrido es un up-chirp
# con un corrimiento de frecuencia, aproximadamente ortogonal al original. Entonces
# X[tau + l] / 8 = h[l] + ruido, con ruido gaussiano complejo de varianza sigma^2 / 8.


def channel_ls(r: np.ndarray, p: LoRaParams, k: int, L: int) -> np.ndarray:
    """Estimación LS de los L coeficientes del canal, suponiendo que la trama empieza en k.

    h_LS[l] = X[k + l] / n_upchirps: no usa información previa del canal y deja pasar el ruido
    entero en cada coeficiente (varianza sigma^2 / n_upchirps), aunque el camino no tenga energía.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.
        k (int): Muestra de inicio supuesta.
        L (int): Cantidad de coeficientes a estimar.

    Returns:
        np.ndarray: Estimación h_LS[l], l = 0, ..., L - 1.
    """

    return preamble_xcorr(r, p)[k : k + L] / p.n_upchirps


def lmmse_weights(P: np.ndarray, noise_power: float, p: LoRaParams) -> np.ndarray:
    """Pesos del estimador LMMSE de cada coeficiente: h_LMMSE[l] = a[l] h_LS[l].

    Con h[l] de media nula y potencia P[l] = E|h[l]|^2 (perfil de potencia), independiente entre
    caminos y del ruido, el principio de ortogonalidad da a[l] = P[l] / (P[l] + sigma^2 / 8)
    (Oppenheim y Verghese, §8.3.1). Como los caminos son independientes, el estimador vectorial
    R_hh (R_hh + R_nn)^-1 h_LS (§8.3.2) tiene matrices diagonales y se separa en un peso por
    coeficiente: cerca de 1 donde el camino es fuerte y cerca de 0 donde solo hay ruido.

    Args:
        P (np.ndarray): Perfil de potencia del canal, P[l] = E|h[l]|^2.
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Pesos a[l].
    """

    return P / (P + noise_power / p.n_upchirps)


def estimator_mse_theory(P: np.ndarray, noise_power: float, p: LoRaParams) -> tuple[float, float]:
    """Error cuadrático medio E||h_est - h||^2 teórico de los estimadores LS y LMMSE.

    LS: L sigma^2 / 8, el ruido de los L coeficientes. LMMSE: sum_l P[l] (sigma^2/8) /
    (P[l] + sigma^2/8), menor que el de LS en cada coeficiente.

    Args:
        P (np.ndarray): Perfil de potencia del canal.
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        p (LoRaParams): Parámetros del sistema.

    Returns:
        tuple[float, float]: MSE del LS y del LMMSE.
    """

    var = noise_power / p.n_upchirps
    return float(P.size * var), float(np.sum(P * var / (P + var)))


# --- Detección con la estimación del canal -----------------------------------------------------


def energy_statistic(r: np.ndarray, p: LoRaParams, w: np.ndarray) -> np.ndarray:
    """Energía ponderada de la estimación del canal a partir de cada inicio k.

    G[k] = sum_l w[l] |X[k + l] / n_upchirps|^2. En k = tau junta la energía de todos los caminos,
    en lugar de solo la del principal como T[k]. Los caminos se suman en energía porque cada uno
    llega con su propia fase. Con w = 1 es la energía de h_LS; con w = a (`lmmse_weights`), cada
    coeficiente pesa según su SNR y los caminos débiles casi no suman ruido.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.
        w (np.ndarray): Peso de cada uno de los L coeficientes.

    Returns:
        np.ndarray: G[k] para cada k donde entran el preámbulo y los L caminos.
    """

    h2 = np.abs(preamble_xcorr(r, p) / p.n_upchirps) ** 2
    return np.correlate(h2, w, mode="valid")


def hypoexp_threshold(means: np.ndarray, P_FA: float) -> float:
    """Umbral de Neyman-Pearson para una suma de exponenciales independientes de medias mu_i.

    La suma tiene distribución hipoexponencial. Su cola en forma cerrada,
    sum_i e^{-x/mu_i} prod_{j != i} mu_i / (mu_i - mu_j), suma términos de signo alternado que
    crecen cuando las medias se acercan, y en float64 se cancelan mal (pasa con los pesos LMMSE a
    SNR alto, todos cerca de 1). Por eso se calcula por uniformización, que solo suma términos
    positivos: la suma es el tiempo que tarda en recorrer L etapas un proceso que sale de la etapa
    i con tasa 1/mu_i. Se lo mira en los saltos de un proceso de Poisson de tasa
    lam = max_i 1/mu_i, donde cada salto avanza de la etapa i con probabilidad (1/mu_i) / lam:

        P(G > x) = sum_n Poisson(n; lam x) P(no terminó las L etapas en n saltos)

    Args:
        means (np.ndarray): Media mu_i de cada exponencial.
        P_FA (float): Probabilidad de falsa alarma.

    Returns:
        float: Umbral eta tal que P(G > eta) = P_FA.
    """

    return _hypoexp_threshold(tuple(np.asarray(means, dtype=float)), P_FA)


# El umbral se calcula una vez por SNR: el FER llama al detector en cada trama con las mismas medias
@lru_cache(maxsize=256)
def _hypoexp_threshold(means: tuple[float, ...], P_FA: float) -> float:
    mu = np.array(means) / max(means)  # Normalizado: se busca en unidades de la mayor media
    advance = (1 / mu) / np.max(1 / mu)  # Probabilidad de avanzar de etapa en cada salto
    lam = np.max(1 / mu)

    def tail(x: float) -> float:
        # Saltos hasta que la cola de Poisson(lam x) es despreciable
        n_max = int(lam * x + 10 * np.sqrt(lam * x) + 30)
        v = np.zeros(mu.size)
        v[0] = 1.0  # Distribución de la etapa, entre las que todavía no terminaron
        log_w = -lam * x  # log Poisson(0; lam x)
        total = 0.0
        for n in range(n_max + 1):
            total += np.exp(log_w) * v.sum()
            v = v * (1 - advance) + np.concatenate([[0.0], v[:-1] * advance[:-1]])
            log_w += np.log(lam * x) - np.log(n + 1) if x > 0 else -np.inf
        return total

    return _invert_tail(tail, P_FA) * max(means)


def detect_ls(
    r: np.ndarray, p: LoRaParams, noise_power: float, P_FA: float, *, P: np.ndarray
) -> int | None:
    """Detecta el inicio de la trama con la energía de la estimación LS del canal.

    Bajo H0 cada |X[k + l] / 8|^2 es exponencial de media sigma^2 / 8 y, como la plantilla
    corrida es aproximadamente ortogonal a sí misma, las L son independientes: G[k] es Erlang de
    orden L y escala sigma^2 / 8.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        P_FA (float): Probabilidad de falsa alarma por símbolo.
        P (np.ndarray): Perfil de potencia del canal. LS solo usa su largo L.

    Returns:
        int | None: Muestra de inicio de la trama, o None si no se detecta.
    """

    # P_FA / M por muestra: misma tasa de falsas alarmas que un test por símbolo
    eta = erlang_threshold(P.size, noise_power / p.n_upchirps, P_FA / p.M)
    return detect_start(energy_statistic(r, p, np.ones(P.size)), eta, p.n_upchirps * p.M)


def detect_lmmse(
    r: np.ndarray, p: LoRaParams, noise_power: float, P_FA: float, *, P: np.ndarray
) -> int | None:
    """Detecta el inicio de la trama con la energía ponderada por los pesos LMMSE.

    Bajo H0 cada término a[l] |X[k + l] / 8|^2 es exponencial de media a[l] sigma^2 / 8: G[k] es
    hipoexponencial.

    Args:
        r (np.ndarray): Señal recibida.
        p (LoRaParams): Parámetros del sistema.
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        P_FA (float): Probabilidad de falsa alarma por símbolo.
        P (np.ndarray): Perfil de potencia del canal, P[l] = E|h[l]|^2.

    Returns:
        int | None: Muestra de inicio de la trama, o None si no se detecta.
    """

    a = lmmse_weights(P, noise_power, p)
    eta = hypoexp_threshold(a * noise_power / p.n_upchirps, P_FA / p.M)
    return detect_start(energy_statistic(r, p, a), eta, p.n_upchirps * p.M)


# --- Umbral por símbolo ----------------------------------------------------------------------


def threshold(method: str, p: LoRaParams, noise_power: float, P_FA: float) -> float:
    """Umbral de Neyman-Pearson de un método para una probabilidad de falsa alarma por símbolo.

    En el tiempo se prueban 2^SF muestras por símbolo, así que cada muestra usa P_FA / 2^SF: misma
    tasa de falsas alarmas que un test por símbolo. En frecuencia hay un test por ventana.

    Args:
        method (str): "full" (preámbulo completo), "peaks" (búsqueda de impulsos) o "freq"
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
