"""Gráficos del Anexo A."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .. import tx
from ..params import LoRaParams
from ..plots import _centered, _real_part, _specgram, _vlines
from . import vangelista


def chirp_comparison(symbols: np.ndarray, p: LoRaParams, L: int = 8) -> Figure:
    """Ec. 2 y chirp de LoRa lado a lado: parte real (línea con L puntos por chip, puntos en las
    muestras) y espectrograma centrado en la banda. Eje inferior en muestras, superior en tiempo.

    Args:
        symbols (np.ndarray): Símbolos a modular.
        p (LoRaParams): Parámetros del sistema.
        L (int): Puntos por chip de la parte real.

    Returns:
        Figure: Figura con una columna por modulación.
    """

    signals = {
        "Ec. 2": vangelista.modulator,
        "Chirp de LoRa": tx.modulator,
    }
    fig, axs = plt.subplots(2, 2, figsize=(14, 6), sharex=True)
    for col, (name, modulate) in zip(axs.T, signals.items()):
        c = modulate(symbols, p)
        _real_part(col[0], modulate(symbols, p, L), c, L, f"{name}: parte real")
        _specgram(col[1], _centered(c), p)
        col[1].set(title=f"{name}: espectrograma (centrado en la banda)", ylabel="Frecuencia (kHz)")
        col[1].set_xlabel("Muestras")
        # Eje superior en tiempo: una muestra cada T = 1/B
        top = col[0].secondary_xaxis(
            "top", functions=(lambda k: k * p.T * 1e3, lambda t: t / (p.T * 1e3))
        )
        top.set_xlabel("Tiempo (ms)")
        for ax in col:
            _vlines(ax, np.arange(len(symbols) + 1) * p.M)
    fig.tight_layout()
    return fig
