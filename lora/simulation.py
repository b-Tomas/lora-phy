"""Simulaciones Monte Carlo del sistema. Devuelven datos; los gráficos están en `plots`."""

from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import numpy as np

from . import params
from .channel import FLAT, noise_from_SNR, noise_power, system, transmit_frame
from .detection import (
    Detector,
    peaks_statistic,
    peaks_threshold,
    preamble_correlation,
    preamble_zero_count,
    threshold,
    time_threshold,
    zero_count_false_alarm,
)
from .metrics import roc
from .params import LoRaParams
from .rx import decoder, demodulator
from .tx import encoder, modulator, preamble

# --- Tasas de error ---------------------------------------------------------------------------

# Errores por punto para que una tasa estimada sea confiable: con k errores el desvío relativo de
# la estimación es de aproximadamente 1 / sqrt(k), un 10 % con 100 errores
MIN_ERRORS = 100


class ErrorRate(NamedTuple):
    """Tasa de error (BER o FER) estimada en cada SNR de una curva.

    Attributes:
        errors (np.ndarray): Errores contados en cada SNR (bits o tramas).
        trials (np.ndarray): Bits o tramas simulados en cada SNR.
        min_errors (int): Errores buscados por punto. Un punto que llegó al máximo de
            simulaciones sin juntarlos es menos confiable.
    """

    errors: np.ndarray
    trials: np.ndarray
    min_errors: int = MIN_ERRORS

    @property
    def rate(self) -> np.ndarray:
        """Tasa de error de cada SNR."""
        return self.errors / self.trials

    def interval(self, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
        """Intervalo de confianza de Wilson de la tasa, del 95 % con z = 1.96.

        Supone simulaciones independientes con la misma probabilidad de error. En el BER los
        errores de bits del mismo símbolo no lo son: el intervalo real es algo más ancho.
        """
        n, rate = self.trials, self.rate
        center = (rate + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z / (1 + z**2 / n) * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
        return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)  # Redondeo en 0 y 1

    @property
    def reliable(self) -> np.ndarray:
        """Si cada SNR juntó `min_errors` errores."""
        return self.errors >= self.min_errors


def _count_errors(trial, min_runs: int, max_runs: int, min_errors: int) -> tuple[int, int]:
    """Repite `trial` al menos `min_runs` veces y después hasta juntar `min_errors` errores, sin
    pasar de `max_runs`.

    Args:
        trial: Sin argumentos, simula una vez y devuelve (errores, bits o tramas simulados).
        min_runs (int): Simulaciones mínimas.
        max_runs (int): Simulaciones máximas: donde la tasa es muy baja no se juntan los errores.
        min_errors (int): Errores buscados.

    Returns:
        tuple[int, int]: Errores y bits o tramas simulados en total.
    """

    runs = errors = trials = 0
    while runs < min_runs or (errors < min_errors and runs < max_runs):
        e, n = trial()
        errors, trials, runs = errors + e, trials + n, runs + 1
    return errors, trials


def _seeded(seed: int, point, kwargs: dict) -> tuple[int, int]:
    """`point(**kwargs)` con el generador reiniciado: corre en otro proceso, con su propia copia."""
    params.seed(seed)
    return point(**kwargs)


def _sweep(point, SNRs_dB: np.ndarray, curves: dict, common: dict) -> dict:
    """Simula cada SNR de cada curva con `point`, todos los puntos en un único pool de procesos.

    Cada punto usa una semilla sacada del generador del paquete, en un orden fijo: la corrida
    sigue siendo reproducible con `seed` y no depende de cuántos procesos haya.

    Args:
        point: Función que simula un punto y devuelve (errores, bits o tramas simulados).
        SNRs_dB (np.ndarray): SNR a simular, en decibelios.
        curves (dict): Argumentos propios de cada curva, por clave de curva.
        common (dict): Argumentos comunes a todas las curvas.

    Returns:
        dict: `ErrorRate` de cada curva, con las mismas claves que `curves`.
    """

    tasks = [{**common, **kw, "SNR_dB": s} for kw in curves.values() for s in SNRs_dB]
    seeds = params.rng.integers(2**63, size=len(tasks))
    # ponytail: los SNR altos son los que llegan al máximo de simulaciones; se mandan primero
    # para que no queden solos al final de la corrida. Estimar el costo si eso no alcanza
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i]["SNR_dB"])
    with ProcessPoolExecutor() as pool:
        futures = {i: pool.submit(_seeded, int(seeds[i]), point, tasks[i]) for i in order}
        counts = np.array([futures[i].result() for i in range(len(tasks))])

    counts = counts.reshape(len(curves), len(SNRs_dB), 2)
    min_errors = common["min_errors"]
    return {key: ErrorRate(c[:, 0], c[:, 1], min_errors) for key, c in zip(curves, counts)}


# --- BER -------------------------------------------------------------------------------------


def bit_errors(
    p: LoRaParams,
    SNR_dB: float,
    h: np.ndarray,
    modulate=modulator,
    demodulate=demodulator,
) -> tuple[int, int]:
    """Simula la transmisión de 100 símbolos aleatorios a través de un canal ruidoso y cuenta
    los bits errados.

    Args:
        p (LoRaParams): Parámetros del sistema.
        SNR_dB (float): Relación señal-ruido en decibelios.
        h (np.ndarray): Respuesta al impulso del canal.
        modulate, demodulate: Modulador y demodulador (los del Anexo A usan Ec. 2).

    Returns:
        tuple[int, int]: Bits errados y bits transmitidos.
    """

    # Conjunto de bits aleatorios a transmitir correspondiente a 100 símbolos
    bits_tx = params.rng.integers(0, 2, size=p.SF * 100)

    # Transmisión completa
    c = modulate(encoder(bits_tx, p), p)  # Señal modulada
    z = noise_from_SNR(c, SNR_dB)  # Ruido generado a partir de la señal y el SNR
    r = system(c, h, z)  # Señal recibida después del canal ruidoso

    # Demodulación y decodificación. Los bits errados son los 1 del XOR
    bits_rx = decoder(demodulate(r, p), p)
    return int(np.sum(bits_tx ^ bits_rx)), bits_tx.size


def transmission_ber(
    p: LoRaParams,
    SNR_dB: float,
    h: np.ndarray,
    modulate=modulator,
    demodulate=demodulator,
) -> float:
    """BER de una transmisión de 100 símbolos aleatorios (`bit_errors`)."""

    errors, bits = bit_errors(p, SNR_dB, h, modulate, demodulate)
    return errors / bits


def _ber_point(
    p, SNR_dB, h, min_transmissions, max_transmissions, min_errors, modulate, demodulate
) -> tuple[int, int]:
    """Bits errados y transmitidos en un SNR, con el criterio de `_count_errors`."""
    return _count_errors(
        lambda: bit_errors(p, SNR_dB, h, modulate, demodulate),
        min_transmissions,
        max_transmissions,
        min_errors,
    )


def ber_vs_snr(
    p: LoRaParams,
    SNRs_dB: np.ndarray,
    curves: dict,
    min_transmissions: int,
    max_transmissions: int,
    min_errors: int = MIN_ERRORS,
) -> dict:
    """BER de varias curvas en función del SNR, todos los puntos en paralelo.

    Cada SNR simula transmisiones de 100 símbolos (`bit_errors`): al menos `min_transmissions`,
    y después hasta juntar `min_errors` bits errados o llegar a `max_transmissions`.

    Args:
        p (LoRaParams): Parámetros del sistema.
        SNRs_dB (np.ndarray): SNR a simular, en decibelios.
        curves (dict): Por etiqueta de curva, un dict con el canal "h" y, opcionalmente,
            "modulate" y "demodulate" (por defecto, los de LoRa).
        min_transmissions, max_transmissions (int): Transmisiones mínimas y máximas por SNR.
        min_errors (int): Bits errados buscados por SNR.

    Returns:
        dict: `ErrorRate` de cada curva, por etiqueta.
    """

    curves = {k: {"modulate": modulator, "demodulate": demodulator, **v} for k, v in curves.items()}
    common = {
        "p": p,
        "min_transmissions": min_transmissions,
        "max_transmissions": max_transmissions,
        "min_errors": min_errors,
    }
    return _sweep(_ber_point, SNRs_dB, curves, common)


# --- Tramas ----------------------------------------------------------------------------------


def frame_lost(
    detect: Detector,
    SNR_dB: float,
    h: np.ndarray,
    p: LoRaParams,
    P_FA: float,
    num_symbols: int,
    gap: int,
) -> bool:
    """Simula una trama y decide si el detector la pierde.

    Una trama se pierde si el detector no estima su inicio en la muestra exacta: no la detecta,
    o la detecta antes (falsa alarma) o después. Lo que pase con el payload lo mide el BER.

    Args:
        detect (Detector): Detector. detect(r, p, noise_power, P_FA) devuelve la muestra de
            inicio de la trama, o None si no la detecta.
        SNR_dB (float): SNR a simular, en decibelios.
        h (np.ndarray): Respuesta al impulso del canal.
        p (LoRaParams): Parámetros del sistema.
        P_FA (float): Probabilidad de falsa alarma por símbolo.
        num_symbols (int): Símbolos del payload por trama.
        gap (int): Símbolos de silencio antes de la trama.

    Returns:
        bool: Si la trama se perdió.
    """

    # Potencia del ruido por muestra (sigma^2), que el receptor conoce
    npw = noise_power(p, SNR_dB)
    r = transmit_frame(params.rng.integers(0, p.M, size=num_symbols), gap, p, SNR_dB, h)
    return detect(r, p, npw, P_FA) != gap * p.M


def _fer_point(
    detect, SNR_dB, h, p, P_FA, num_symbols, gap, min_frames, max_frames, min_errors
) -> tuple[int, int]:
    """Tramas perdidas y simuladas en un SNR, con el criterio de `_count_errors`."""
    return _count_errors(
        lambda: (int(frame_lost(detect, SNR_dB, h, p, P_FA, num_symbols, gap)), 1),
        min_frames,
        max_frames,
        min_errors,
    )


def fer_vs_snr(
    SNRs_dB: np.ndarray,
    curves: dict,
    p: LoRaParams,
    P_FA: float,
    num_symbols: int,
    gap: int,
    min_frames: int,
    max_frames: int,
    min_errors: int = MIN_ERRORS,
) -> dict:
    """FER de varias curvas en función del SNR, todos los puntos en paralelo.

    Cada SNR simula tramas (`frame_lost`): al menos `min_frames`, y después hasta juntar
    `min_errors` tramas perdidas o llegar a `max_frames`.

    Args:
        SNRs_dB (np.ndarray): SNR a simular, en decibelios.
        curves (dict): Por clave de curva, un dict con el detector "detect" y el canal "h".
        p, P_FA, num_symbols, gap: Como en `frame_lost`.
        min_frames, max_frames (int): Tramas mínimas y máximas por SNR.
        min_errors (int): Tramas perdidas buscadas por SNR.

    Returns:
        dict: `ErrorRate` de cada curva, por clave.
    """

    common = {
        "p": p,
        "P_FA": P_FA,
        "num_symbols": num_symbols,
        "gap": gap,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "min_errors": min_errors,
    }
    return _sweep(_fer_point, SNRs_dB, curves, common)


# --- ROC y falsas alarmas --------------------------------------------------------------------


def detection_statistics(
    SNR_dB: float, p: LoRaParams, simulations: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Simula realizaciones de los estadísticos de detección bajo cada hipótesis.

    Bajo H1 la posición 0 está alineada con el inicio del preámbulo; bajo H0 solo hay ruido.

    Args:
        SNR_dB (float): Relación señal-ruido en decibelios.
        p (LoRaParams): Parámetros del sistema.
        simulations (int): Realizaciones por hipótesis.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: Para cada método, realizaciones de su
        estadístico en la posición 0 bajo H0 y bajo H1.
    """

    statistics = {
        "full": preamble_correlation,
        "peaks": peaks_statistic,
        "freq": preamble_zero_count,
    }
    h0: dict[str, list] = {method: [] for method in statistics}
    h1: dict[str, list] = {method: [] for method in statistics}
    for _ in range(simulations):
        # H1: trama sin silencio previo, la posición 0 está alineada con el inicio del preámbulo
        r1 = transmit_frame(params.rng.integers(0, p.M, size=1), 0, p, SNR_dB, FLAT)
        # H0: solo ruido, con la misma potencia que el ruido de la trama
        r0 = noise_from_SNR(preamble(p), SNR_dB)
        # Estadístico de cada método en la posición 0
        for method, statistic in statistics.items():
            h0[method].append(statistic(r0, p)[0])
            h1[method].append(statistic(r1, p)[0])

    return {method: (np.array(h0[method]), np.array(h1[method])) for method in statistics}


class RocCurves(NamedTuple):
    """Curvas ROC de un método, con la P_FA por símbolo.

    Attributes:
        P_D_theory (np.ndarray): P_D para cada P_FA de `P_FA_axis`, con el umbral teórico.
        P_FA_empirical (np.ndarray): P_FA empírica, por símbolo, de cada umbral observado.
        P_D_empirical (np.ndarray): P_D empírica de cada umbral observado.
        P_D_operating (float): P_D en el punto de operación de Neyman-Pearson.
    """

    P_D_theory: np.ndarray
    P_FA_empirical: np.ndarray
    P_D_empirical: np.ndarray
    P_D_operating: float


def roc_curves(
    p: LoRaParams, SNR_dB: float, simulations: int, P_FA_axis: np.ndarray, P_FA: float
) -> dict[str, RocCurves]:
    """Curvas ROC teóricas y empíricas de los tres métodos.

    Bajo H0 la distribución de cada estadístico es conocida: para cada P_FA se calcula el umbral
    con las funciones de Neyman-Pearson y solo se estima la P_D con las realizaciones bajo H1.
    Las curvas empíricas validan las teóricas donde se pueden medir.

    Args:
        p (LoRaParams): Parámetros del sistema.
        SNR_dB (float): Relación señal-ruido en decibelios.
        simulations (int): Realizaciones por hipótesis.
        P_FA_axis (np.ndarray): P_FA por símbolo donde se evalúa la curva teórica.
        P_FA (float): P_FA por símbolo del punto de operación.

    Returns:
        dict[str, RocCurves]: Curvas de cada método.
    """

    npw = noise_power(p, SNR_dB)
    curves = {}
    for method, (h0, h1) in detection_statistics(SNR_dB, p, simulations).items():
        P_FA_emp, P_D_emp = roc(h0, h1)
        # En el tiempo se prueban M posiciones por símbolo: P_FA por símbolo = M P_FA por muestra
        positions_per_symbol = 1 if method == "freq" else p.M
        curves[method] = RocCurves(
            P_D_theory=np.array([np.mean(h1 >= threshold(method, p, npw, q)) for q in P_FA_axis]),
            P_FA_empirical=positions_per_symbol * P_FA_emp,
            P_D_empirical=P_D_emp,
            P_D_operating=float(np.mean(h1 >= threshold(method, p, npw, P_FA))),
        )

    return curves


def measured_false_alarms(
    p: LoRaParams, num_noise_symbols: int, P_FA_targets: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mide la P_FA de cada método sobre una señal larga de solo ruido (-10 dB).

    Valida el cálculo de los umbrales: la fracción de posiciones que superan el umbral debe
    coincidir con la P_FA pedida, por posición.

    Args:
        p (LoRaParams): Parámetros del sistema.
        num_noise_symbols (int): Largo de la señal de ruido, en símbolos.
        P_FA_targets (np.ndarray): P_FA pedidas, por posición, para los métodos en el tiempo. En
            frecuencia se miden los umbrales K = 1 y K = 2, los únicos con P_FA medible.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: Para cada método, P_FA pedida y medida.
    """

    Ks = (1, 2)
    npw = noise_power(p, -10)
    r_noise = noise_from_SNR(np.resize(preamble(p), num_noise_symbols * p.M), -10)
    T = preamble_correlation(r_noise, p)
    S = peaks_statistic(r_noise, p)
    V = preamble_zero_count(r_noise, p)

    return {
        "full": (
            P_FA_targets,
            np.array([np.mean(T >= time_threshold(npw, q, p)) for q in P_FA_targets]),
        ),
        "peaks": (
            P_FA_targets,
            np.array([np.mean(S >= peaks_threshold(npw, q, p)) for q in P_FA_targets]),
        ),
        "freq": (
            np.array([zero_count_false_alarm(K, p) for K in Ks]),
            np.array([np.mean(V >= K) for K in Ks]),
        ),
    }
