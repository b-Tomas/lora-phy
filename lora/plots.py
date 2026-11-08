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
from .rx import dechirp_fft
from .sdr import ADC_FULL_SCALE
from .simulation import RocCurves
from .tx import instantaneous_frequency

# Nombre y color de cada método de detección, por clave de método
_METHODS = {
    "full": ("Tiempo (preámbulo completo)", "C0"),
    "peaks": ("Tiempo (8 picos)", "C2"),
    "freq": ("Frecuencia (demodulación)", "C1"),
    "ideal": ("Detección ideal", "k"),
}

# Marcador de cada método en los gráficos de puntos
_MARKERS = {"full": "s", "peaks": "^", "freq": "o"}

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
            bbox={"facecolor": "w", "alpha": 0.8},
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
        title=f"Búsqueda de {p.n_upchirps} picos: inicio real k={gap * p.M}, detectado k={start} "
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
    K: float,
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
        K (float): Umbral de decisión (entero: cantidad de ventanas).
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


def start_vs_snr(
    SNRs_dB: np.ndarray, start_flat: np.ndarray, start_sel: np.ndarray, title: str
) -> Figure:
    """Probabilidad de detectar el inicio correcto vs SNR de un método, en ambos canales.

    Args:
        SNRs_dB (np.ndarray): SNR simulados, en decibelios.
        start_flat (np.ndarray): P(inicio correcto) en el canal plano.
        start_sel (np.ndarray): P(inicio correcto) en el canal selectivo en frecuencia.
        title (str): Título del gráfico.

    Returns:
        Figure: Figura con las dos curvas.
    """

    fig, ax = plt.subplots()
    ax.plot(SNRs_dB, start_flat, marker="s", label="Canal plano")
    ax.plot(SNRs_dB, start_sel, marker="o", label="Canal selectivo en frecuencia")
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
    logy: bool = False,
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
        logy (bool): Eje y logarítmico. Los puntos en 0 no se dibujan.

    Returns:
        Figure: Figura con las curvas.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    for method, metric_flat in flat.items():
        name, color = _METHODS[method]
        ax.plot(SNRs_dB, metric_flat, color=color, marker="o", label=f"{name}, plano")
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
    if logy:
        ax.set_yscale("log", nonpositive="mask")
    ax.set(title=title, xlabel="SNR (dB)", ylabel=ylabel)
    ax.grid(True, which="both" if logy else "major")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))  # Afuera: no tapa las curvas
    return fig


def fer_sweep(
    x: np.ndarray, fer: dict[str, np.ndarray], xlabel: str, title: str, logx: bool = False
) -> Figure:
    """FER en escala logarítmica en función de un parámetro del sistema, una curva por canal.

    Args:
        x (np.ndarray): Valores del parámetro.
        fer (dict[str, np.ndarray]): FER para cada valor de `x`, por nombre del canal.
        xlabel (str): Nombre del parámetro.
        title (str): Título del gráfico.
        logx (bool): Eje x logarítmico (en base 2).

    Returns:
        Figure: Figura con las curvas. Los puntos con FER = 0 no se dibujan.
    """

    fig, ax = plt.subplots()
    for name, f in fer.items():
        ax.plot(x, f, marker="o", label=name)
    ax.set_yscale("log", nonpositive="mask")
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(v) for v in x])
    ax.set(title=title, xlabel=xlabel, ylabel="FER")
    ax.grid(True, which="both")
    ax.legend()
    return fig


def false_alarms(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    num_noise_symbols: int,
    Ks: tuple[int, ...],
) -> Figure:
    """P_FA medida vs pedida de cada método (`simulation.measured_false_alarms`).

    Args:
        results (dict[str, tuple[np.ndarray, np.ndarray]]): P_FA pedida y medida por método.
        num_noise_symbols (int): Largo de la señal de ruido, en símbolos.
        Ks (tuple[int, ...]): Umbrales medidos en frecuencia, los mismos que se pasaron a
            `measured_false_alarms`. Sus puntos son umbrales enteros, no P_FA pedidas.

    Returns:
        Figure: Figura con los puntos y la diagonal.
    """

    fig, ax = plt.subplots()
    for method, (target, measured) in results.items():
        name, color = _METHODS[method]
        if method == "freq":
            name += f", K = {', '.join(map(str, Ks))}"
        ax.loglog(target, measured, _MARKERS[method], color=color, label=name)
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
            title=f"Curva ROC (SNR={SNR_dB:g} dB)",
            xlabel="$P_{FA}$ por símbolo",
            ylabel="$P_D$",
        )
        ax.grid(True, which="both", alpha=0.3)
    # Una sola leyenda debajo de los gráficos, para no tapar las curvas
    fig.legend(*axs[0, -1].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def cpu_times(SFs: list[int], cpu: list[dict[str, float]]) -> Figure:
    """Tiempo de CPU por trama de cada método vs SF (`simulation.methods_cpu_time`).

    Args:
        SFs (list[int]): Spreading factors medidos.
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


# --- SDR -------------------------------------------------------------------------------------


def _full_scale_lines(ax: Axes) -> None:
    for y, label in ((ADC_FULL_SCALE, "Fondo de escala"), (-ADC_FULL_SCALE, None)):
        ax.axhline(y, color="r", ls="--", lw=0.8, label=label)
    ax.set_ylim(-1.1 * ADC_FULL_SCALE, 1.1 * ADC_FULL_SCALE)


def sdr_capture(
    r: np.ndarray, start: int | None, silence: int, frame_len: int, p: LoRaParams
) -> Figure:
    """Captura de tramas del SDR: muestras contra el fondo de escala del ADC, su distribución, el
    espectro y los up-chirps del preámbulo después del dechirp.

    Args:
        r (np.ndarray): Captura.
        start (int | None): Inicio de una trama detectada en `r`, o None.
        silence (int): Muestras de silencio a mostrar antes de la trama.
        frame_len (int): Muestras de la trama (preámbulo y payload).
        p (LoRaParams): Parámetros del sistema.

    Returns:
        Figure: Figura con los cuatro gráficos.
    """

    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 4)
    ax_t, ax_h = fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3])
    ax_s, ax_d = fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2:])

    # El silencio y la trama detectada
    begin = 0 if start is None else max(start - silence, 0)
    segment = r[begin : begin + silence + frame_len]
    ax_t.plot(np.real(segment), lw=0.5, label="I")
    ax_t.plot(np.imag(segment), lw=0.5, label="Q")
    if start is not None:
        ax_t.axvline(start - begin, color="k", ls="--", lw=0.8, label="Inicio detectado")
        ax_t.axvline(start - begin + p.preamble_len, color="k", ls=":", label="Fin del preámbulo")
    _full_scale_lines(ax_t)
    ax_t.set(title="Silencio y trama", xlabel="Muestra", ylabel="Amplitud")
    ax_t.legend(loc="lower right", ncol=5, fontsize="small")

    # Distribución de toda la captura, con la gaussiana de la misma potencia
    ax_h.hist(
        [np.real(r), np.imag(r)],
        bins=80,
        density=True,
        orientation="horizontal",
        histtype="step",
        label=["I", "Q"],
    )
    sigma = np.sqrt(np.mean(np.abs(r) ** 2) / 2)
    y = np.linspace(-ADC_FULL_SCALE, ADC_FULL_SCALE, 200)
    gauss = np.exp(-(y**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    ax_h.plot(gauss, y, "k--", lw=0.8, label="Gaussiana")
    _full_scale_lines(ax_h)
    ax_h.set(title="Distribución", xlabel="Densidad", yticklabels=[])
    ax_h.set_xticks([])
    ax_h.legend(loc="lower right", fontsize="small")

    # Espectro: promedio de las FFT de bloques de un símbolo
    blocks = r[: r.size // p.M * p.M].reshape(-1, p.M)
    spectrum = np.mean(np.abs(np.fft.fft(blocks, axis=1) / p.M) ** 2, axis=0) / ADC_FULL_SCALE**2
    f = np.fft.fftshift(np.fft.fftfreq(p.M, p.T)) / 1e3
    spectrum_dB = 10 * np.log10(np.fft.fftshift(spectrum))
    ax_s.plot(f, spectrum_dB)
    ax_s.set(
        title="Espectro de la captura (potencia por bin)",
        xlabel="Frecuencia [kHz]",
        ylabel="dB respecto del fondo de escala",
        ylim=(spectrum_dB.max() - 40, spectrum_dB.max() + 5),
    )
    ax_s.grid(True)

    # Dechirp de los up-chirps: la señal se concentra en el bin 0 y el ruido sigue repartido
    if start is not None:
        D = np.mean(np.abs(dechirp_fft(r[start : start + p.n_upchirps * p.M], p)) ** 2, axis=0)
        ax_d.plot(10 * np.log10(D / D.max()))
        ax_d.set(
            title=f"Up-chirps del preámbulo después del dechirp (promedio de {p.n_upchirps})",
            xlabel="Bin",
            ylabel="dB respecto del pico",
            ylim=(-50, 3),
        )
        ax_d.grid(True)
    fig.tight_layout()
    return fig


def sdr_tone(captures: np.ndarray, gains_dB: np.ndarray, k: int, p: LoRaParams) -> Figure:
    """Capturas de un tono (exponencial compleja en el bin `k` de la FFT) con distintas ganancias
    del RX, una por fila: muestras contra el fondo de escala, espectro y plano I/Q.

    Args:
        captures (np.ndarray): Capturas (una por fila) de un número entero de períodos del tono.
        gains_dB (np.ndarray): Ganancia del RX de cada captura, en dB.
        k (int): Bin del tono en la FFT de cada captura.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        Figure: Figura con los tres gráficos por captura.
    """

    captures = np.atleast_2d(captures)
    fig, axes = plt.subplots(
        len(captures),
        3,
        figsize=(14, 3.6 * len(captures)),
        gridspec_kw={"width_ratios": [2, 2, 1.3]},
        squeeze=False,
    )
    square = ADC_FULL_SCALE * np.array([-1, 1, 1, -1, -1])
    for r, gain, (ax_t, ax_f, ax_iq) in zip(captures, gains_dB, axes):
        cycles = 8 * r.size // k  # Muestras de 8 ciclos del tono
        ax_t.plot(np.real(r)[:cycles], ".-", lw=0.8, label="I")
        ax_t.plot(np.imag(r)[:cycles], ".-", lw=0.8, label="Q")
        _full_scale_lines(ax_t)
        ax_t.set(title=f"Ganancia {gain:.0f} dB", xlabel="Muestra", ylabel="Amplitud")
        ax_t.legend(loc="lower right")

        # Un número entero de períodos: el tono cae justo en un bin y no se desparrama
        X = np.fft.fftshift(np.fft.fft(r)) / r.size
        f = np.fft.fftshift(np.fft.fftfreq(r.size, p.T)) / 1e3
        X_dB = 20 * np.log10(np.abs(X) / ADC_FULL_SCALE + 1e-12)
        ax_f.plot(f, X_dB, color="gray", lw=0.6)
        for bin_, name, marker in ((k, "Tono", "o"), (-k, "Imagen", "s"), (0, "Fuga del LO", "^")):
            i = r.size // 2 + bin_  # Índice después de fftshift
            ax_f.plot(f[i], X_dB[i], marker, label=f"{name}: {X_dB[i]:.0f} dB")
        ax_f.set(
            title="Espectro",
            xlabel="Frecuencia [kHz]",
            ylabel="dB respecto del fondo de escala",
            ylim=(-140, 10),
        )
        ax_f.grid(True)
        ax_f.legend(loc="upper left", fontsize="small")

        ax_iq.plot(np.real(r), np.imag(r), ",", alpha=0.5)
        ax_iq.plot(square, np.roll(square, 1), "r--", lw=0.8)
        ax_iq.set(title="Plano I/Q", xlabel="I", ylabel="Q", aspect="equal")
    fig.suptitle("Tono con distintas ganancias del RX")
    fig.tight_layout()
    return fig


def snr_vs_atten(attens_dB: np.ndarray, snr_dB: np.ndarray, fer_range_dB: np.ndarray) -> Figure:
    """SNR medido sin ruido agregado vs atenuación del TX, junto con el rango de SNR donde el FER
    simulado pasa de casi 1 a casi 0.

    Args:
        attens_dB (np.ndarray): Atenuaciones del TX.
        snr_dB (np.ndarray): SNR medido para cada atenuación.
        fer_range_dB (np.ndarray): SNR donde se mide el FER (se marca de su mínimo a su máximo).

    Returns:
        Figure: Figura del SNR.
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhspan(min(fer_range_dB), max(fer_range_dB), color="C1", alpha=0.2, label="Rango del FER")
    ax.plot(attens_dB, snr_dB, "o-", label="SNR medido")
    ax.set(title="Sin ruido agregado", xlabel="Atenuación del TX [dB]", ylabel="SNR [dB]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def snr_added(SNRs_tx_dB: np.ndarray, snr_measured_dB: np.ndarray, snr_hw_dB: float) -> Figure:
    """SNR medido vs SNR del ruido agregado en el TX, con el modelo 1/SNR = 1/SNR_tx + 1/SNR_hw.

    Args:
        SNRs_tx_dB (np.ndarray): SNR del ruido agregado en el TX.
        snr_measured_dB (np.ndarray): SNR medido para cada SNR agregado.
        snr_hw_dB (float): SNR del equipo, medido sin ruido agregado.

    Returns:
        Figure: Figura del SNR.
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))
    snr_tx = np.linspace(SNRs_tx_dB.min(), SNRs_tx_dB.max())
    model = -10 * np.log10(10 ** (-snr_tx / 10) + 10 ** (-snr_hw_dB / 10))
    ax.plot(snr_tx, snr_tx, "k:", lw=0.8, label="Medido = agregado")
    ax.plot(snr_tx, model, color="C0", label=f"Modelo, $SNR_{{hw}}$ = {snr_hw_dB:.1f} dB")
    ax.plot(SNRs_tx_dB, snr_measured_dB, "o", color="C0", label="Medido")
    ax.set(
        title="Con ruido agregado en el TX (atenuación de -10 dB)",
        xlabel="SNR agregado [dB]",
        ylabel="SNR medido [dB]",
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def fer_sdr(
    SNRs_sim_dB: np.ndarray,
    fer_sim: np.ndarray,
    measured: dict[str, tuple[np.ndarray, np.ndarray]],
    n_frames: int,
) -> Figure:
    """FER medido con el SDR sobre la curva simulada en canal plano, en el rango de SNR de la
    simulación. FER = 0 no entra en la escala logarítmica: esos puntos quedan debajo de
    1 / n_frames, el menor FER que se puede medir.

    Args:
        SNRs_sim_dB (np.ndarray): SNR de la simulación, en dB.
        fer_sim (np.ndarray): FER simulado para cada SNR.
        measured (dict[str, tuple[np.ndarray, np.ndarray]]): SNR medido y FER de cada serie del
            SDR, por nombre.
        n_frames (int): Tramas medidas por punto.

    Returns:
        Figure: Figura del FER.
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(SNRs_sim_dB, fer_sim, label="Simulación (canal plano)")
    for (label, (snr, fer)), marker in zip(measured.items(), ["s", "o", "^"]):
        fer = np.asarray(fer)
        ax.plot(snr, np.where(fer > 0, fer, np.nan), marker, ls="none", label=label)
    ax.axhline(1 / n_frames, color="gray", ls=":", label=f"1 error en {n_frames} tramas")
    ax.set_yscale("log")
    ax.set_xlim(SNRs_sim_dB.min() - 0.5, SNRs_sim_dB.max() + 0.5)
    ax.set(title="FER: simulación y SDR", xlabel="SNR [dB]", ylabel="FER")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    return fig
