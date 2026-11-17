"""Gráficos del notebook.

Cada función recibe datos ya calculados (señales, estadísticos o resultados de `simulation`),
crea su figura y la devuelve sin llamar a `plt.show()`. En Jupyter la figura se muestra sola al
final de la celda: llamar con `;` al final para que no se muestre dos veces.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from .channel import FLAT
from .params import LoRaParams
from .rx import dechirp, dechirp_fft
from .sdr import ADC_FULL_SCALE
from .simulation import RocCurves
from .tx import instantaneous_frequency, modulator

# Nombre y color de cada método de detección, por clave de método
_METHODS = {
    "full": ("Tiempo (preámbulo completo)", "C0"),
    "peaks": ("Tiempo (8 impulsos)", "C2"),
    "freq": ("Frecuencia (demodulación)", "C1"),
}

# Muestras por ventana de la FFT de los espectrogramas
_NFFT = 16


def _specgram(ax: Axes, x: np.ndarray, p: LoRaParams) -> None:
    """Espectrograma de `x` con el eje x en muestras, como los demás gráficos.

    Se agregan _NFFT/2 ceros a cada lado para que haya una ventana centrada en cada muestra de `x`,
    también en los bordes."""

    pad = np.zeros(_NFFT // 2)
    ax.specgram(
        np.concatenate([pad, x, pad]),
        NFFT=_NFFT,
        noverlap=_NFFT - 1,
        Fs=p.B / 1e3,
        sides="twosided",
        scale="linear",
        xextent=(0, x.size),
    )


def _centered(c: np.ndarray) -> np.ndarray:
    """Multiplica `c` por (-1)^k = e^{-j2π (B/2) kT}: corre la frecuencia en -B/2 y lleva la banda
    de [0, B] a [-B/2, B/2], como propone el paper (Sec. II)."""

    return c * (-1.0) ** np.arange(c.size)


def _vlines(ax: Axes, xs) -> None:
    """Líneas verticales de separación: blancas sobre un espectrograma (fondo oscuro), negras en
    el resto."""

    for x in xs:
        ax.axvline(x, color="w" if ax.images else "k", ls="--", lw=0.8)


def _real_part(ax: Axes, x_fine: np.ndarray, x: np.ndarray, L: int, title: str) -> None:
    """Parte real de una señal: curva evaluada con L puntos por chip, para que se vea la forma de
    la onda sin aliasing, y puntos en las muestras."""

    ax.plot(np.arange(x_fine.size) / L, np.real(x_fine), lw=0.5)
    ax.plot(np.real(x), "k.", ms=3)
    ax.set(title=title, ylabel="Amplitud")


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


def modulated_signal(symbols: np.ndarray, p: LoRaParams, L: int = 8) -> Figure:
    """Señal modulada en el tiempo: parte real y espectrograma.

    La parte real se evalúa también entre muestras, con L puntos por chip, para que se vea la forma
    de la onda sin aliasing. El espectrograma se calcula sobre las muestras, centrado en la banda.

    Args:
        symbols (np.ndarray): Símbolos a modular.
        p (LoRaParams): Parámetros del sistema.
        L (int): Puntos por chip de la parte real.

    Returns:
        Figure: Figura con los dos gráficos.
    """

    c = modulator(symbols, p)
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.5))
    _real_part(axs[0], modulator(symbols, p, L), c, L, "Parte real de $c$ (puntos: muestras)")
    _specgram(axs[1], _centered(c), p)
    axs[1].set(
        title="Espectrograma de $c\\,(-1)^k$ (centrado en la banda)", ylabel="Frecuencia (kHz)"
    )
    for ax in axs:
        ax.set(xlabel="Muestras")
        _vlines(ax, np.arange(len(symbols) + 1) * p.M)
    fig.tight_layout()
    return fig


def _interpolate(x: np.ndarray, p: LoRaParams, L: int) -> np.ndarray:
    """Interpolación de banda limitada de las muestras de cada símbolo, con L puntos por chip: la
    única señal con frecuencias entre 0 y B que pasa por las 2^SF muestras del símbolo."""

    X = np.fft.fft(x.reshape(-1, p.M), axis=1)
    X_fine = np.zeros((X.shape[0], p.M * L), dtype=complex)
    X_fine[:, : p.M] = X  # Mismos bins de 0 a B, sin componentes nuevas
    return (np.fft.ifft(X_fine, axis=1) * L).ravel()


def _periodogram(ax: Axes, P: np.ndarray, s: int, p: LoRaParams) -> int:
    """Periodograma $|\\langle r, c|_{s=q} \\rangle|^2$ de un símbolo, con el bin del símbolo
    transmitido en rojo y el del decidido en azul (violeta si coinciden). Devuelve el símbolo
    decidido."""

    s_hat = int(np.argmax(P))
    ax.stem(P, linefmt="g-", markerfmt="g.", basefmt=" ")
    marks = {s: "tab:purple"} if s == s_hat else {s: "tab:red", s_hat: "tab:blue"}
    for q, color in marks.items():
        stem = ax.stem([q], [P[q]], linefmt=color, markerfmt=".", basefmt=" ")
        stem.markerline.set_color(color)
        stem.stemlines.set_linewidth(2.5)  # Tapa la línea verde del mismo bin
    ax.set(xlabel="q (índice de la FFT)", ylabel="Potencia")
    # Eje superior en frecuencia: el bin q corresponde a q B / 2^SF
    kHz_per_bin = p.B / p.M / 1e3
    top = ax.secondary_xaxis(
        "top",
        functions=(lambda q: np.asarray(q) * kHz_per_bin, lambda f: np.asarray(f) / kHz_per_bin),
    )
    top.set_xlabel("Frecuencia (kHz)")
    return s_hat


def fft_decision(
    Ds: dict[str, np.ndarray], symbols_tx: np.ndarray, p: LoRaParams, SNR_dB: float | None = None
) -> Figure:
    """Periodograma de cada símbolo recibido, como en `dechirp_steps`, con una columna por canal.

    Args:
        Ds (dict[str, np.ndarray]): Salida de `rx.dechirp_fft`, de forma (N, 2^SF), por nombre del
            canal.
        symbols_tx (np.ndarray): Símbolos transmitidos.
        p (LoRaParams): Parámetros del sistema.
        SNR_dB (float | None): SNR del canal. Si se indica, se agrega al título general.

    Returns:
        Figure: Figura con un gráfico por símbolo y canal.
    """

    n, m = len(symbols_tx), len(Ds)
    # Misma escala en cada fila para comparar los canales
    fig, axs = plt.subplots(
        n, m, figsize=(10 if m == 1 else 7 * m, 3 * n), squeeze=False, sharey="row"
    )
    for j, (channel, D) in enumerate(Ds.items()):
        for i, s in enumerate(symbols_tx):
            s_hat = _periodogram(axs[i, j], np.abs(D[i]) ** 2, s, p)
            axs[i, j].set_title(f"{channel}, símbolo n={i}: transmitido s={s}, decidido ŝ={s_hat}")
    if SNR_dB is not None:
        fig.suptitle(f"SNR={SNR_dB} dB")
    fig.tight_layout()
    return fig


def _path_amplitudes(
    ax: Axes, r_fine: np.ndarray, r: np.ndarray, h: np.ndarray, p: LoRaParams, L: int
) -> None:
    """Amplitud de r (curva sin ruido entre muestras y puntos en las muestras) y amplitud
    constante de cada camino del canal, que se suman en fase o en contrafase según la frecuencia
    del chirp."""

    ax.plot(np.arange(r_fine.size) / L, np.abs(r_fine), lw=0.8, label="$|r|$")
    ax.plot(np.abs(r), "k.", ms=3)
    for i, h_i in enumerate(h):
        label = "Camino directo" if i == 0 else "Eco"
        ax.axhline(abs(h_i) / np.sqrt(p.M), color=f"C{i + 2}", ls="--", lw=1.2, label=label)
    # Lugar arriba de la curva para la leyenda
    ax.set(
        title="Amplitud de $r$ y de cada camino del canal",
        ylabel="Amplitud",
        ylim=(0, 1.35 * np.abs(h).sum() / np.sqrt(p.M)),
    )
    ax.legend(loc="upper center", ncol=3, fontsize="small")


def dechirp_steps(
    r: np.ndarray,
    s: int,
    p: LoRaParams,
    SNR_dB: float | None = None,
    h: np.ndarray = FLAT,
    L: int = 32,
) -> Figure:
    """Demodulación de un símbolo paso a paso: r, su espectrograma, d y periodograma de d.

    r y d se grafican como en `modulated_signal`: la curva evalúa la señal sin ruido entre
    muestras, con L puntos por chip, y los puntos son las muestras recibidas. d solo existe en las
    muestras, porque el demodulador multiplica muestras, así que su curva es la interpolación de
    banda limitada de sus muestras sin ruido.

    Args:
        r (np.ndarray): Un símbolo recibido, de 2^SF muestras.
        s (int): Símbolo transmitido.
        p (LoRaParams): Parámetros del sistema.
        SNR_dB (float | None): SNR del canal. Si se indica, se agrega al título.
        h (np.ndarray): Respuesta al impulso del canal, para que la curva sin ruido pase por las
            muestras recibidas.
        L (int): Puntos por chip de las curvas.

    Returns:
        Figure: Figura con los cuatro gráficos.
    """

    # Señal sin ruido entre muestras: un camino por coeficiente de h, el chirp escalado por h[i]
    # y retrasado i chips (i L puntos)
    c0 = modulator(np.array([s]), p, L)
    paths = [h_i * np.concatenate([np.zeros(i * L), c0])[: c0.size] for i, h_i in enumerate(h)]
    c_fine = np.sum(paths, axis=0)

    # Con eco se agrega después de r un gráfico de amplitud
    echo = len(paths) > 1
    fig, axs = plt.subplots(4 + echo, 1, figsize=(10, 10 + 2.5 * echo))
    ax_r = axs[0]
    ax_amp, ax_spec, ax_d, ax_P = axs[1:] if echo else (None, *axs[1:])

    _real_part(ax_r, c_fine, r, L, "Parte real de $r$ (curva: sin ruido, puntos: muestras)")
    if ax_amp is not None:
        _path_amplitudes(ax_amp, c_fine, r, h, p, L)
    _specgram(ax_spec, _centered(r), p)
    ax_spec.set(
        title="Espectrograma de $r\\,(-1)^k$ (centrado en la banda)", ylabel="Frecuencia (kHz)"
    )
    # Con tx.modulator, el producto con el down-chirp entre muestras cambia de frecuencia donde el
    # chirp vuelve a 0, aunque las muestras sean las de un tono
    _real_part(
        ax_d,
        _interpolate(dechirp(c_fine[::L], p)[0], p, L),
        dechirp(r, p)[0],
        L,
        "Parte real de $d$ (curva: sin ruido, puntos: muestras)",
    )
    for ax in axs[:-1]:
        ax.set(xlabel="Muestras", xlim=(0, p.M))
    s_hat = _periodogram(ax_P, np.abs(dechirp_fft(r, p)[0]) ** 2, s, p)
    ax_P.set_title(
        f"Periodograma de $d$, $|\\langle r, c|_{{s=q}} \\rangle|^2$ (decidido ŝ={s_hat})"
    )
    title = f"Símbolo transmitido s={s}"
    if SNR_dB is not None:
        title += f", SNR={SNR_dB} dB"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))  # Lugar para el título
    return fig


def ber_vs_snr(SNRs_dB: np.ndarray, bers: dict[str, np.ndarray]) -> Figure:
    """BER vs SNR, una curva por canal.

    Args:
        SNRs_dB (np.ndarray): SNR simulados, en decibelios.
        bers (dict[str, np.ndarray]): BER de cada curva, por etiqueta.

    Returns:
        Figure: Figura con las curvas.
    """

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for (label, ber), marker in zip(bers.items(), ["s", "o", "^", "d", "v"]):
        ax.semilogy(SNRs_dB, ber, marker=marker, label=label)
    ax.set(title="BER vs SNR", xlabel="SNR (dB)", ylabel="BER")
    ax.grid(True, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))  # Afuera, para no tapar las curvas
    fig.tight_layout()
    return fig


# --- Detección de una trama ------------------------------------------------------------------


def frame_structure(symbols_tx: np.ndarray, gap: int, p: LoRaParams) -> Figure:
    """Estructura de la trama: frecuencia instantánea transmitida.

    Args:
        symbols_tx (np.ndarray): Símbolos del payload.
        gap (int): Símbolos de silencio antes de la trama.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        Figure: Figura con el gráfico.
    """

    # Frecuencia instantánea de la trama sin ruido
    k = np.arange(p.M)
    f_down = ((p.M - k) % p.M) * p.B / p.M  # Down-chirp: de B a 0
    n_down = p.preamble_len - p.n_upchirps * p.M  # Muestras de down-chirps
    f = np.concatenate(
        [
            np.full(gap * p.M, np.nan),  # Silencio: no hay señal
            instantaneous_frequency(np.zeros(p.n_upchirps, dtype=int), p),  # Up-chirps: de 0 a B
            np.resize(f_down, n_down),
            instantaneous_frequency(symbols_tx, p),  # Payload
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(f / 1e3, color="green")
    ax.set(
        title="Trama transmitida (frecuencia instantánea)",
        xlabel="Muestras",
        ylabel="Frecuencia (kHz)",
    )
    _annotate_frame(ax, gap, p, f.size)
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
        Figure: Figura con los cuatro gráficos.
    """

    n_samples = y.size + p.M - 1  # Largo de la señal recibida

    fig, axs = plt.subplots(3, 1, figsize=(10, 9))
    axs[0].plot(np.abs(y_clean) ** 2, color="green", lw=0.8)
    axs[0].set(title="Correlación móvil con un up-chirp (sin ruido)")
    axs[1].plot(np.abs(y) ** 2, color="green", lw=0.8)
    axs[1].set(title=f"Correlación móvil con un up-chirp (SNR={SNR_dB} dB)")
    for ax in axs[:2]:
        _annotate_frame(ax, gap, p, n_samples)

    # Detalle sin ruido del inicio del preámbulo: un impulso en cada borde de símbolo
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
    T: np.ndarray,
    eta: float,
    start: int | None,
    gap: int,
    p: LoRaParams,
    P_FA: float,
    SNR_dB: float,
) -> Figure:
    """Estadístico T[k] de la correlación con el preámbulo completo, con el umbral.

    Args:
        T (np.ndarray): Estadístico sobre la trama.
        eta (float): Umbral de decisión.
        start (int | None): Muestra de inicio detectada, o None.
        gap (int): Símbolos de silencio antes de la trama.
        p (LoRaParams): Parámetros del sistema.
        P_FA (float): Probabilidad de falsa alarma por símbolo del umbral.
        SNR_dB (float): SNR de la trama.

    Returns:
        Figure: Figura del estadístico.
    """

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(T, color="green", lw=0.8)
    ax.axhline(
        eta, color="r", ls="--", lw=0.8, label=f"Umbral η ($P_{{FA}}$={P_FA:.1e} por símbolo)"
    )
    _detection_marks(ax, gap * p.M, start)
    ax.set(
        title=f"Correlación con el preámbulo completo: inicio real k={gap * p.M}, "
        f"detectado k={start} (SNR={SNR_dB} dB)",
        xlabel="k (muestra)",
        ylabel="$T[k]$",
        xlim=(0, T.size + p.n_upchirps * p.M - 1),  # Largo de la señal recibida
    )
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
    """Correlación móvil con los impulsos del inicio detectado, y estadístico S[k] con su umbral.

    Args:
        y (np.ndarray): Correlación móvil con un up-chirp sobre la trama con ruido.
        S (np.ndarray): Estadístico de la búsqueda de impulsos.
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
        # Los impulsos que suma S[k] en el inicio detectado
        peaks = start + np.arange(p.n_upchirps) * p.M
        axs[0].plot(
            peaks,
            np.abs(y[peaks]) ** 2,
            "o",
            mfc="none",
            color="r",
            label=f"{p.n_upchirps} impulsos del inicio detectado",
        )
        axs[0].legend(loc="upper right")
    axs[0].set(title=f"Correlación móvil con un up-chirp (SNR={SNR_dB} dB)", ylabel="$|y[k]|^2$")
    axs[1].plot(S, color="green", lw=0.8)
    axs[1].axhline(
        eta_S, color="r", ls="--", lw=0.8, label=f"Umbral $η_S$ ($P_{{FA}}$={P_FA:.1e} por símbolo)"
    )
    _detection_marks(axs[1], gap * p.M, start)
    axs[1].set(
        title=f"Búsqueda de {p.n_upchirps} impulsos: inicio real k={gap * p.M}, detectado k={start} "
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


# --- Desempeño de los detectores -----------------------------------------------------------


def compare_methods(
    SNRs_dB: np.ndarray, flat: dict[str, np.ndarray], sel: dict[str, np.ndarray]
) -> Figure:
    """FER de los métodos vs SNR, un gráfico por canal, con eje y logarítmico (los puntos en 0 no
    se dibujan).

    Args:
        SNRs_dB (np.ndarray): SNR simulados, en decibelios.
        flat (dict[str, np.ndarray]): FER de cada método en el canal plano, por clave de método
            ("full", "peaks" o "freq").
        sel (dict[str, np.ndarray]): FER de cada método en el canal selectivo.

    Returns:
        Figure: Figura con los dos gráficos.
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, metric, channel in zip(axs, (flat, sel), ("Canal plano", "Canal selectivo")):
        for method, values in metric.items():
            name, color = _METHODS[method]
            ax.plot(SNRs_dB, values, color=color, marker="o", label=name)
        ax.set_yscale("log", nonpositive="mask")
        ax.set(title=channel, xlabel="SNR (dB)")
        ax.grid(True, which="both", alpha=0.5)
    axs[0].set_ylabel("FER")
    axs[0].legend()
    fig.suptitle("FER vs SNR")
    fig.tight_layout()
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


# --- Anexos ----------------------------------------------------------------------------------


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
    for (method, (target, measured)), marker in zip(results.items(), "s^o"):
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


def sdr_tone(captures: np.ndarray, gains_dB: np.ndarray, k: int, p: LoRaParams) -> Figure:
    """Capturas de un tono en el bin `k` de la FFT con distintas ganancias del RX, una por fila:
    espectro y plano I/Q contra el fondo de escala del ADC.

    Args:
        captures (np.ndarray): Capturas (una por fila) de un número entero de períodos del tono.
        gains_dB (np.ndarray): Ganancia del RX de cada captura, en dB.
        k (int): Bin del tono en la FFT de cada captura.
        p (LoRaParams): Parámetros del sistema.

    Returns:
        Figure: Figura con los dos gráficos por captura.
    """

    fig, axes = plt.subplots(
        len(captures), 2, figsize=(11, 3.6 * len(captures)), width_ratios=[2.5, 1], squeeze=False
    )
    square = ADC_FULL_SCALE * np.array([-1, 1, 1, -1, -1])
    for r, gain, (ax_f, ax_iq) in zip(captures, gains_dB, axes):
        X_dB = 20 * np.log10(
            np.abs(np.fft.fftshift(np.fft.fft(r))) / r.size / ADC_FULL_SCALE + 1e-12
        )
        f = np.fft.fftshift(np.fft.fftfreq(r.size, p.T)) / 1e3
        ax_f.plot(f, X_dB, color="gray", lw=0.6)
        for bin_, name, marker in ((k, "Tono", "o"), (-k, "Imagen", "s"), (0, "Fuga del LO", "^")):
            i = r.size // 2 + bin_  # Índice después de fftshift
            ax_f.plot(f[i], X_dB[i], marker, label=f"{name}: {X_dB[i]:.0f} dB")
        ax_f.set(
            title=f"Espectro con ganancia {gain:.0f} dB",
            xlabel="Frecuencia [kHz]",
            ylabel="dB respecto del fondo de escala",
            ylim=(-140, 10),
        )
        ax_f.grid(True)
        ax_f.legend(loc="upper left", fontsize="small")

        ax_iq.plot(np.real(r), np.imag(r), ",", alpha=0.5)
        ax_iq.plot(square, np.roll(square, 1), "r--", lw=0.8, label="Fondo de escala")
        ax_iq.set(title="Plano I/Q", xlabel="I", ylabel="Q", aspect="equal")
        ax_iq.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


# --- SDR -------------------------------------------------------------------------------------


def snr_vs_atten(attens_dB: np.ndarray, snr_dB: np.ndarray, fer_range_dB: np.ndarray) -> Figure:
    """SNR medido vs atenuación del TX, junto con el rango de SNR donde el FER
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
    ax.set(
        title="SNR según la atenuación del TX", xlabel="Atenuación del TX [dB]", ylabel="SNR [dB]"
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def fer_sdr(
    SNRs_sim_dB: np.ndarray,
    fer_sim: np.ndarray,
    snr: np.ndarray,
    fer: np.ndarray,
    n_frames: int,
) -> Figure:
    """FER medido con el SDR sobre la curva simulada en canal plano. Los puntos sin errores no
    entran en la escala logarítmica: se dibujan en 1 / n_frames, el menor FER que se puede medir,
    como cota superior.

    Args:
        SNRs_sim_dB (np.ndarray): SNR de la simulación, en dB.
        fer_sim (np.ndarray): FER simulado para cada SNR.
        snr (np.ndarray): SNR medido de cada punto del SDR, en dB.
        fer (np.ndarray): FER medido de cada punto del SDR.
        n_frames (int): Tramas medidas por punto.

    Returns:
        Figure: Figura del FER.
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Un punto por SNR simulado, para distinguirlos de los tramos interpolados
    ax.plot(SNRs_sim_dB, fer_sim, "o-", ms=3, label="Simulación (canal plano)")
    ax.plot(snr[fer > 0], fer[fer > 0], "o", color="C1", label="SDR")
    if np.any(fer == 0):
        ax.plot(
            snr[fer == 0],
            np.full(np.sum(fer == 0), 1 / n_frames),
            "v",
            color="C1",
            mfc="none",
            label=f"SDR, sin errores en {n_frames} tramas",
        )
    ax.set_yscale("log", nonpositive="mask")
    snrs = np.r_[SNRs_sim_dB, snr]
    ax.set_xlim(snrs.min() - 0.5, snrs.max() + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))  # Una marca por dB, como los SNR simulados
    ax.set(title="FER: simulación y SDR", xlabel="SNR [dB]", ylabel="FER")
    ax.grid(True, which="both", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig
