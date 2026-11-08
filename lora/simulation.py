"""Simulaciones Monte Carlo del sistema. Devuelven datos; los gráficos están en `plots`."""

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
from .metrics import BER, frame_error, roc
from .params import LoRaParams
from .rx import decoder, demodulator
from .tx import encoder, modulator, preamble

# --- BER -------------------------------------------------------------------------------------


def transmission_ber(p: LoRaParams, SNR_dB: float, h: np.ndarray, num_symbols: int = 100) -> float:
    """Simula la transmisión de bits aleatorios a través de un canal ruidoso y calcula el BER.

    Args:
        p (LoRaParams): Parámetros del sistema.
        SNR_dB (float): Relación señal-ruido en decibelios.
        h (np.ndarray): Respuesta al impulso del canal.
        num_symbols (int): Símbolos transmitidos.

    Returns:
        float: Bit Error Rate (BER) de la transmisión simulada.
    """

    # Conjunto de bits aleatorios a transmitir correspondiente a num_symbols símbolos
    bits_tx = params.rng.integers(0, 2, size=p.SF * num_symbols)

    # Transmisión completa
    c = modulator(encoder(bits_tx, p), p)  # Señal modulada
    z = noise_from_SNR(c, SNR_dB)  # Ruido generado a partir de la señal y el SNR
    r = system(c, h, z)  # Señal recibida después del canal ruidoso

    # Demodulación y decodificación
    bits_rx = decoder(demodulator(r, p), p)

    return BER(bits_tx, bits_rx)


def ber_vs_snr(
    p: LoRaParams, SNRs_dB: np.ndarray, h: np.ndarray, simulations: int, num_symbols: int = 100
) -> np.ndarray:
    """BER promedio de `simulations` transmisiones para cada SNR.

    Args:
        p (LoRaParams): Parámetros del sistema.
        SNRs_dB (np.ndarray): SNR a simular, en decibelios.
        h (np.ndarray): Respuesta al impulso del canal.
        simulations (int): Transmisiones simuladas por SNR.
        num_symbols (int): Símbolos por transmisión.

    Returns:
        np.ndarray: BER para cada SNR.
    """

    return np.array(
        [
            np.mean([transmission_ber(p, SNR_dB, h, num_symbols) for _ in range(simulations)])
            for SNR_dB in SNRs_dB
        ]
    )


# --- Tramas ----------------------------------------------------------------------------------


def frame_rates(
    detect: Detector,
    SNRs_dB: np.ndarray,
    h: np.ndarray,
    p: LoRaParams,
    P_FA: float,
    simulations: int,
    num_symbols: int,
    gap: int,
    min_errors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simula tramas con un detector y estima el inicio correcto y el FER para cada SNR.

    Args:
        detect (Detector): Detector. detect(r, p, noise_power, P_FA) devuelve la muestra de
            inicio de la trama, o None si no la detecta.
        SNRs_dB (np.ndarray): SNR a simular, en decibelios.
        h (np.ndarray): Respuesta al impulso del canal.
        p (LoRaParams): Parámetros del sistema.
        P_FA (float): Probabilidad de falsa alarma por símbolo.
        simulations (int): Tramas simuladas por SNR. Con `min_errors`, el máximo.
        num_symbols (int): Símbolos del payload por trama.
        gap (int): Símbolos de silencio antes de la trama.
        min_errors (int | None): Si se indica, cada SNR termina al juntar esa cantidad de tramas
            con error. Donde el FER es alto alcanzan pocas tramas, y las demás se gastan donde
            es bajo.

    Returns:
        tuple[np.ndarray, np.ndarray]: Fracción de tramas con inicio correcto y FER, para cada SNR.
    """

    p_start, fer = np.zeros(len(SNRs_dB)), np.zeros(len(SNRs_dB))
    for i, SNR_dB in enumerate(SNRs_dB):
        # Potencia del ruido por muestra (sigma^2), que el receptor conoce
        npw = noise_power(p, SNR_dB)
        frames = 0
        while frames < simulations and (min_errors is None or fer[i] < min_errors):
            symbols_tx = params.rng.integers(0, p.M, size=num_symbols)
            r = transmit_frame(symbols_tx, gap, p, SNR_dB, h)
            start = detect(r, p, npw, P_FA)
            p_start[i] += start == gap * p.M
            fer[i] += frame_error(r, start, symbols_tx, p)
            frames += 1
        p_start[i] /= frames
        fer[i] /= frames

    return p_start, fer


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
    p: LoRaParams, num_noise_symbols: int, P_FA_targets: np.ndarray, Ks: tuple[int, ...] = (1, 2)
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mide la P_FA de cada método sobre una señal larga de solo ruido (-10 dB).

    Valida el cálculo de los umbrales: la fracción de posiciones que superan el umbral debe
    coincidir con la P_FA pedida, por posición.

    Args:
        p (LoRaParams): Parámetros del sistema.
        num_noise_symbols (int): Largo de la señal de ruido, en símbolos.
        P_FA_targets (np.ndarray): P_FA pedidas, por posición, para los métodos en el tiempo.
        Ks (tuple[int, ...]): Umbrales a medir en frecuencia (solo los de P_FA medible).

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: Para cada método, P_FA pedida y medida.
    """

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
