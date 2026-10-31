"""Gráficos del notebook.

Cada función recibe datos ya calculados (señales, estadísticos o resultados de `simulation`),
crea su figura y la devuelve sin llamar a `plt.show()`. En Jupyter la figura se muestra sola al
final de la celda: llamar con `;` al final para que no se muestre dos veces.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .params import LoRaParams
from .simulation import RocCurves
from .tx import instantaneous_frequency

# Nombre y color de cada método de detección, por clave de método
_METHODS = {
    "full": ("Tiempo (preámbulo completo)", "C0"),
    "peaks": ("Tiempo (8 picos)", "C2"),
    "freq": ("Frecuencia (demodulación)", "C1"),
    "ideal": ("Detección ideal", "k"),
}

# Muestras por ventana de la FFT de los espectrogramas
_NFFT = 16


def _specgram(ax: Axes, x: np.ndarray, p: LoRaParams) -> None:
    """Espectrograma de `x` con el eje x en muestras, como los demás gráficos."""

    # xextent: cada ventana se ubica en su centro
    ax.specgram(
        x,
        NFFT=_NFFT,
        noverlap=_NFFT - 1,
        Fs=p.B / 1e3,
        sides="twosided",
        scale="linear",
        xextent=(_NFFT / 2, x.size - _NFFT / 2),
    )


def _vlines(ax: Axes, xs) -> None:
    """Líneas verticales de separación: blancas sobre un espectrograma (fondo oscuro), negras en
    el resto."""

    for x in xs:
        ax.axvline(x, color="w" if ax.images else "k", ls="--", lw=0.8)


def _annotate_frame(ax: Axes, gap: int, p: LoRaParams, n_samples: int) -> None:
    """Marca las partes de la trama (silencio, up-chirps, down-chirps y payload) sobre `ax`."""

    bounds = [0, gap * p.M, (gap + p.n_upchirps) * p.M, gap * p.M + p.preamble_len, n_samples]
    names = ["Silencio", "Up-chirps", "Down-chirps", "Payload"]

    _vlines(ax, bounds[1:-1])
    # Nombre de cada parte, centrado entre sus límites (x en muestras, y relativo al gráfico)
    for left, right, name in zip(bounds[:-1], bounds[1:], names):
        ax.text(
            (left + right) / 2,
            0.88,
            name,
            ha="center",
            transform=ax.get_xaxis_transform(),
            bbox=dict(facecolor="w", alpha=0.8),
        )
    ax.set_xlim(0, n_samples)  # Mismo eje en todos los gráficos, incluyendo el silencio


# --- Señal modulada y demodulación -----------------------------------------------------------


def modulated_signal(c: np.ndarray, symbols: np.ndarray, p: LoRaParams) -> Figure:
    """Señal modulada en el tiempo: parte real, imaginaria, frecuencia instantánea y espectrograma.

    Args:
        c (np.ndarray): Señal modulada.
        symbols (np.ndarray): Símbolos modulados en `c`.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        Figure: Figura con los cuatro gráficos.
    """

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(np.real(c))
    axs[0].set(title="Señal modulada (parte real)", xlabel="Muestras", ylabel="Amplitud")
    axs[1].plot(np.imag(c), color="orange")
    axs[1].set(title="Señal modulada (parte imaginaria)", xlabel="Muestras", ylabel="Amplitud")
    # Frecuencia instantánea en el tiempo: f(k) = (s + k) mod M * B / M
    axs[2].plot(instantaneous_frequency(symbols, p) / 1e3, color="green")
    axs[2].set(
        title="Señal modulada (frecuencia instantánea)",
        xlabel="Muestras",
        ylabel="Frecuencia (kHz)",
    )
    # Espectrograma: frecuencia en el tiempo estimada a partir de la señal c
    _specgram(axs[3], c, p)
    axs[3].set(title="Señal modulada (espectrograma)", xlabel="Muestras", ylabel="Frecuencia (kHz)")
    # Separación entre símbolos: un símbolo (Ts) son 2**SF muestras
    for ax in axs:
        _vlines(ax, np.arange(len(symbols) + 1) * p.M)
    fig.tight_layout()
    return fig


def fft_decision(
    D: np.ndarray, symbols_tx: np.ndarray, symbols_rx: np.ndarray, SNR_dB: float | None = None
) -> Figure:
    """Salida de la FFT del demodulador, $|D|^2$, para cada símbolo recibido.

    Args:
        D (np.ndarray): Salida de `rx.dechirp_fft`, de forma (N, 2^SF).
        symbols_tx (np.ndarray): Símbolos transmitidos.
        symbols_rx (np.ndarray): Símbolos decididos.
        SNR_dB (float | None): SNR del canal. Si se indica, se agrega al título y se marca el
            símbolo transmitido, para comparar con el pico.

    Returns:
        Figure: Figura con un gráfico por símbolo.
    """

    n = len(symbols_tx)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.5 * n), squeeze=False)
    for i, ax in enumerate(axs[:, 0]):
        ax.plot(np.abs(D[i]) ** 2, color="green")
        title = f"Símbolo n={i}: transmitido s={symbols_tx[i]}, decidido ŝ={symbols_rx[i]}"
        if SNR_dB is not None:
            ax.axvline(symbols_tx[i], color="k", ls="--", lw=0.8)
            title += f" (SNR={SNR_dB} dB)"
        ax.set(title=title, xlabel="q (índice de la FFT)", ylabel="$|D|^2$")
    fig.tight_layout()
    return fig


def ber_vs_snr(SNRs_dB: np.ndarray, bers: dict[str, np.ndarray]) -> Figure:
    """BER vs SNR, una curva por canal.

    Args:
        SNRs_dB (np.ndarray): SNR simulados, en decibelios.
        bers (dict[str, np.ndarray]): BER de cada curva, por etiqueta.

    Returns:
        Figure: Figura con las curvas.
    """

    fig, ax = plt.subplots()
    for (label, ber), marker in zip(bers.items(), ["s", "o", "^", "d"]):
        ax.semilogy(SNRs_dB, ber, marker=marker, label=label)
    ax.set(title="BER vs SNR", xlabel="SNR (dB)", ylabel="BER")
    ax.grid(True, which="both")
    ax.legend()
    return fig


# --- Detección de una trama ------------------------------------------------------------------


def frame_structure(
    r: np.ndarray, symbols_tx: np.ndarray, gap: int, p: LoRaParams, SNR_dB: float
) -> Figure:
    """Estructura de la trama: frecuencia instantánea transmitida y espectrograma recibido.

    Args:
        r (np.ndarray): Señal recibida (`channel.transmit_frame`).
        symbols_tx (np.ndarray): Símbolos del payload.
        gap (int): Símbolos de silencio antes de la trama.
        p (LoRaParams): Parámetros del sistema.
        SNR_dB (float): SNR del canal.

    Returns:
        Figure: Figura con los dos gráficos.
    """

    # Frecuencia instantánea sin ruido, con fines pedagógicos
    k = np.arange(p.M)
    f_down = ((p.M - k) % p.M) * p.B / p.M  # Down-chirp: de B a 0
    n_down = p.preamble_len - p.n_upchirps * p.M  # Muestras de down-chirps
    f = np.concatenate(
        [
            np.full(gap * p.M, np.nan),  # Silencio: no hay señal
            instantaneous_frequency(np.zeros(p.n_upchirps, dtype=int), p),  # Up-chirps: de 0 a B
            np.resize(f_down, n_down),
            instantaneous_frequency(symbols_tx, p),  # Ec. 2 para cada símbolo del payload
        ]
    )

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(f / 1e3, color="green")
    axs[0].set(title="Trama transmitida (frecuencia instantánea)")
    # Con SNR negativo la trama no se distingue a simple vista en el espectrograma
    _specgram(axs[1], r, p)
    axs[1].set(title=f"Trama recibida (espectrograma, SNR={SNR_dB} dB)")
    for ax in axs:
        ax.set(xlabel="Muestras", ylabel="Frecuencia (kHz)")
        _annotate_frame(ax, gap, p, r.size)
    fig.tight_layout()
    return fig


def upchirp_correlation(
    y_clean: np.ndarray, y: np.ndarray, gap: int, p: LoRaParams, SNR_dB: float
) -> Figure:
    """Correlación móvil con un up-chirp, sin ruido y con ruido, y detalle del inicio.

    Args:
        y_clean (np.ndarray): Correlación sobre la trama sin ruido.
        y (np.ndarray): Correlación sobre la trama con ruido.
        gap (int): Símbolos de silencio antes de la trama.
        p (LoRaParams): Parámetros del sistema.
        SNR_dB (float): SNR de la trama con ruido.

    Returns:
        Figure: Figura con los tres gráficos.
    """

    n_samples = y.size + p.M - 1  # Largo de la señal recibida

    fig, axs = plt.subplots(3, 1, figsize=(10, 9))
    axs[0].plot(np.abs(y_clean) ** 2, color="green", lw=0.8)
    axs[0].set(title="Correlación móvil con un up-chirp (sin ruido)")
    axs[1].plot(np.abs(y) ** 2, color="green", lw=0.8)
    axs[1].set(title=f"Correlación móvil con un up-chirp (SNR={SNR_dB} dB)")
    for ax in axs[:2]:
        _annotate_frame(ax, gap, p, n_samples)

    # Detalle sin ruido del inicio del preámbulo: un pico en cada borde de símbolo y otro en el
    # medio
    axs[2].plot(np.abs(y_clean) ** 2, marker=".", color="green", lw=0.8)
    for n in range(4):
        axs[2].axvline((gap + n) * p.M, color="k", ls=":", lw=0.8)
    axs[2].set_xlim((gap - 1) * p.M, (gap + 3) * p.M)
    axs[2].set(title="Detalle del inicio del preámbulo (sin ruido): bordes de símbolo punteados")

    for ax in axs:
        ax.set(xlabel="k (muestra)", ylabel="$|y[k]|^2$")
    fig.tight_layout()
    return fig


def _detection_marks(ax: Axes, real: int, start: int | None) -> None:
    """Inicio real (negro) e inicio detectado (rojo punteado) sobre `ax`."""

    ax.axvline(real, color="k", ls="--", lw=0.8, label="Inicio real")
    if start is not None:
        ax.axvline(start, color="r", ls=":", label="Inicio detectado")


def full_preamble_detection(
    T_clean: np.ndarray,
    T: np.ndarray,
    eta: float,
    start: int | None,
    gap: int,
    p: LoRaParams,
    P_FA: float,
    SNR_dB: float,
) -> Figure:
    """Estadístico T[k] de la correlación con el preámbulo completo, sin y con ruido.

    Args:
        T_clean (np.ndarray): Estadístico sobre la trama sin ruido.
        T (np.ndarray): Estadístico sobre la trama con ruido.
        eta (float): Umbral de decisión.
        start (int | None): Muestra de inicio detectada, o None.
        gap (int): Símbolos de silencio antes de la trama.
        p (LoRaParams): Parámetros del sistema.
        P_FA (float): Probabilidad de falsa alarma por símbolo del umbral.
        SNR_dB (float): SNR de la trama con ruido.

    Returns:
        Figure: Figura con los dos gráficos.
    """

    n_samples = T.size + p.n_upchirps * p.M - 1  # Largo de la señal recibida

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(T_clean, color="green", lw=0.8)
    axs[0].axvline(gap * p.M, color="k", ls="--", lw=0.8, label="Inicio real")
    axs[0].set(title="Correlación con el preámbulo completo (sin ruido)")
    axs[1].plot(T, color="green", lw=0.8)
    axs[1].axhline(
        eta, color="r", ls="--", lw=0.8, label=f"Umbral η ($P_{{FA}}$={P_FA:.1e} por símbolo)"
    )
    _detection_marks(axs[1], gap * p.M, start)
    axs[1].set(title=f"Inicio real k={gap * p.M}, detectado k={start} (SNR={SNR_dB} dB)")
    for ax in axs:
        ax.set(xlabel="k (muestra)", ylabel="$T[k]$", xlim=(0, n_samples))
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def peaks_detection(
    y: np.ndarray,
    S: np.ndarray,
    eta_S: float,
    start: int | None,
    gap: int,
    p: LoRaParams,
    P_FA: float,
    SNR_dB: float,
) -> Figure:
    """Correlación móvil con los picos del inicio detectado, y estadístico S[k] con su umbral.

    Args:
        y (np.ndarray): Correlación móvil con un up-chirp sobre la trama con ruido.
        S (np.ndarray): Estadístico de la búsqueda de picos.
        eta_S (float): Umbral de decisión.
        start (int | None): Muestra de inicio detectada, o None.
        gap (int): Símbolos de silencio antes de la trama.
        p (LoRaParams): Parámetros del sistema.
        P_FA (float): Probabilidad de falsa alarma por símbolo del umbral.
        SNR_dB (float): SNR de la trama.

    Returns:
        Figure: Figura con los dos gráficos.
    """

    n_samples = y.size + p.M - 1  # Largo de la señal recibida

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(np.abs(y) ** 2, color="green", lw=0.8)
    if start is not None:
        # Los picos que suma S[k] en el inicio detectado
        peaks = start + np.arange(p.n_upchirps) * p.M
        axs[0].plot(
            peaks,
            np.abs(y[peaks]) ** 2,
            "o",
            mfc="none",
            color="r",
            label=f"{p.n_upchirps} picos del inicio detectado",
        )
        axs[0].legend(loc="upper right")
    axs[0].set(title=f"Correlación móvil con un up-chirp (SNR={SNR_dB} dB)", ylabel="$|y[k]|^2$")
    axs[1].plot(S, color="green", lw=0.8)
    axs[1].axhline(
        eta_S, color="r", ls="--", lw=0.8, label=f"Umbral $η_S$ ($P_{{FA}}$={P_FA:.1e} por símbolo)"
    )
    _detection_marks(axs[1], gap * p.M, start)
    axs[1].set(
        title=f"Búsqueda de 8 picos: inicio real k={gap * p.M}, detectado k={start} "
        f"(SNR={SNR_dB} dB)",
        ylabel="$S[k]$",
    )
    axs[1].legend(loc="upper right")
    for ax in axs:
        ax.set(xlabel="k (muestra)", xlim=(0, n_samples))
    fig.tight_layout()
    return fig


def freq_detection(
    symbols_rx: np.ndarray,
    V: np.ndarray,
    K: int,
    start: int | None,
    gap: int,
    p: LoRaParams,
    P_FA: float,
    SNR_dB: float,
) -> Figure:
    """Símbolo decidido en cada ventana y estadístico V[m] con su umbral.

    Args:
        symbols_rx (np.ndarray): Símbolo decidido en cada ventana.
        V (np.ndarray): Estadístico de la demodulación del preámbulo.
        K (int): Umbral de decisión.
        start (int | None): Ventana de inicio detectada (no muestra), o None.
        gap (int): Símbolos de silencio antes de la trama.
        p (LoRaParams): Parámetros del sistema.
        P_FA (float): Probabilidad de falsa alarma por símbolo del umbral.
        SNR_dB (float): SNR de la trama.

    Returns:
        Figure: Figura con los dos gráficos.
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(symbols_rx, "o", color="green")
    axs[0].axvspan(
        gap - 0.5, gap + p.n_upchirps - 0.5, color="k", alpha=0.1, label="Up-chirps del preámbulo"
    )
    axs[0].set(
        title="Símbolo decidido en cada ventana", xlabel="n (ventana)", ylabel="$\\hat{s}[n]$"
    )
    axs[1].plot(V, marker="o", color="green")
    axs[1].axhline(K, color="r", ls="--", lw=0.8, label=f"Umbral K={K} ($P_{{FA}}$={P_FA:.1e})")
    axs[1].axvline(gap, color="k", ls="--", lw=0.8, label="Inicio real")
    axs[1].set(
        title=f"Detección en frecuencia: inicio real m={gap}, detectado m={start} "
        f"(SNR={SNR_dB} dB)",
        xlabel="m (ventana)",
        ylabel="$V[m]$",
    )
    for ax in axs:
        ax.legend()
    fig.tight_layout()
    return fig


# --- Performance de los detectores -----------------------------------------------------------


def start_vs_snr(SNRs_dB: np.ndarray, p_flat: np.ndarray, p_sel: np.ndarray, title: str) -> Figure:
    """Probabilidad de detectar el inicio correcto vs SNR de un método, en ambos canales.

    Args:
        SNRs_dB (np.ndarray): SNR simulados, en decibelios.
        p_flat (np.ndarray): P(inicio correcto) en el canal plano.
        p_sel (np.ndarray): P(inicio correcto) en el canal selectivo en frecuencia.
        title (str): Título del gráfico.

    Returns:
        Figure: Figura con las dos curvas.
    """

    fig, ax = plt.subplots()
    ax.plot(SNRs_dB, p_flat, marker="s", label="Canal plano")
    ax.plot(SNRs_dB, p_sel, marker="o", label="Canal selectivo en frecuencia")
    ax.set(title=title, xlabel="SNR (dB)", ylabel="P(inicio correcto)")
    ax.grid(True)
    ax.legend()
    return fig


def compare_methods(
    SNRs_dB: np.ndarray,
    flat: dict[str, np.ndarray],
    sel: dict[str, np.ndarray],
    ylabel: str,
    title: str,
    xlim: tuple[float, float] | None = None,
) -> Figure:
    """Compara una métrica de los métodos: línea llena en el canal plano, punteada en el selectivo.

    Args:
        SNRs_dB (np.ndarray): SNR simulados, en decibelios.
        flat (dict[str, np.ndarray]): Métrica de cada método en el canal plano, por clave de
            método ("full", "peaks", "freq" o "ideal").
        sel (dict[str, np.ndarray]): Métrica de cada método en el canal selectivo.
        ylabel (str): Nombre de la métrica.
        title (str): Título del gráfico.
        xlim (tuple[float, float] | None): Rango de SNR a mostrar. None muestra todo.

    Returns:
        Figure: Figura con las curvas.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    for method in flat:
        name, color = _METHODS[method]
        ax.plot(SNRs_dB, flat[method], color=color, marker="o", label=f"{name}, plano")
        ax.plot(
            SNRs_dB,
            sel[method],
            color=color,
            marker="o",
            mfc="none",
            ls="--",
            label=f"{name}, selectivo",
        )
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set(title=title, xlabel="SNR (dB)", ylabel=ylabel)
    ax.grid(True)
    ax.legend()
    return fig


def false_alarms(
    results: dict[str, tuple[np.ndarray, np.ndarray]], num_noise_symbols: int
) -> Figure:
    """P_FA medida vs pedida de cada método (`simulation.measured_false_alarms`).

    Args:
        results (dict[str, tuple[np.ndarray, np.ndarray]]): P_FA pedida y medida por método.
        num_noise_symbols (int): Largo de la señal de ruido, en símbolos.

    Returns:
        Figure: Figura con los puntos y la diagonal.
    """

    fig, ax = plt.subplots()
    for (method, (target, measured)), marker in zip(results.items(), ["s", "^", "o"]):
        name, color = _METHODS[method]
        ax.loglog(target, measured, marker, color=color, label=name)
    ax.loglog([1e-6, 1], [1e-6, 1], color="k", ls="--", lw=0.8, label="Medida = pedida")
    ax.set(
        title=f"Falsas alarmas en {num_noise_symbols} símbolos de solo ruido",
        xlabel="$P_{FA}$ pedida (por posición)",
        ylabel="$P_{FA}$ medida",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return fig


def roc(
    curves_by_snr: dict[float, dict[str, RocCurves]], P_FA_axis: np.ndarray, P_FA: float
) -> Figure:
    """Curvas ROC de los tres métodos, un gráfico por SNR (`simulation.roc_curves`).

    Curvas teóricas en línea llena, empíricas punteadas, y el punto de operación de
    Neyman-Pearson de cada método con una estrella.

    Args:
        curves_by_snr (dict[float, dict[str, RocCurves]]): Curvas de cada método, por SNR.
        P_FA_axis (np.ndarray): P_FA por símbolo de las curvas teóricas.
        P_FA (float): P_FA por símbolo del punto de operación.

    Returns:
        Figure: Figura con un gráfico por SNR.
    """

    n = len(curves_by_snr)
    fig, axs = plt.subplots(1, n, figsize=(14, 4.5), squeeze=False)
    for ax, (SNR_dB, curves) in zip(axs[0], curves_by_snr.items()):
        for method, curve in curves.items():
            name, color = _METHODS[method]
            # Curva teórica: P_FA exacta, P_D estimada con las realizaciones bajo H1
            ax.semilogx(P_FA_axis, curve.P_D_theory, color=color, label=name)
            # Punto de operación de Neyman-Pearson
            ax.semilogx(P_FA, curve.P_D_operating, "*", color=color, ms=12, mec="k")
            # Curva empírica. El eje logarítmico no puede mostrar P_FA = 0
            nonzero = curve.P_FA_empirical > 0
            ax.semilogx(
                curve.P_FA_empirical[nonzero],
                curve.P_D_empirical[nonzero],
                color=color,
                ls=":",
                lw=1,
            )
        ax.plot([], [], color="gray", ls=":", label="Empíricas")
        ax.axvline(P_FA, color="k", ls="--", lw=0.8, label=f"$P_{{FA}}$ de operación ({P_FA:.1e})")
        ax.set(
            xlim=(P_FA_axis[0], 1),
            ylim=(-0.02, 1.02),
            title=f"Curva ROC (SNR={SNR_dB} dB)",
            xlabel="$P_{FA}$ por símbolo",
            ylabel="$P_D$",
        )
        ax.grid(True, which="both", alpha=0.3)
    # Una sola leyenda debajo de los gráficos, para no tapar las curvas
    fig.legend(*axs[0, -1].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def cpu_times(SFs: np.ndarray, cpu: list[dict[str, float]]) -> Figure:
    """Tiempo de CPU por trama de cada método vs SF (`simulation.methods_cpu_time`).

    Args:
        SFs (np.ndarray): Spreading factors medidos.
        cpu (list[dict[str, float]]): Tiempo de cada método, en segundos, para cada SF.

    Returns:
        Figure: Figura con una curva por método.
    """

    fig, ax = plt.subplots()
    for method in cpu[0]:
        name, color = _METHODS[method]
        ax.semilogy(SFs, [c[method] * 1e3 for c in cpu], color=color, marker="o", label=name)
    ax.set_xticks(SFs)
    ax.set(title="Tiempo de CPU por trama vs SF", xlabel="SF", ylabel="Tiempo de CPU (ms)")
    ax.grid(True, which="both")
    ax.legend()
    return fig
