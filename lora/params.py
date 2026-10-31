"""Parámetros del sistema LoRa y generador de números aleatorios del paquete."""

from dataclasses import dataclass

import numpy as np

# Semilla por defecto: todas las corridas son reproducibles
SEED = 0

# Generador único del paquete: todas las funciones aleatorias (bits, símbolos y ruido) lo usan
rng: np.random.Generator = np.random.default_rng(SEED)


def seed(n: int) -> None:
    """Reinicia el generador aleatorio del paquete con la semilla `n`.

    Se reinicia el mismo objeto `rng` en lugar de crear uno nuevo, para que los módulos que lo
    importaron antes también vean el cambio.

    Args:
        n (int): Semilla.
    """

    rng.bit_generator.state = np.random.default_rng(n).bit_generator.state


@dataclass(frozen=True)
class LoRaParams:
    """Parámetros de la capa física de LoRa.

    Args:
        SF (int): Spreading factor. Cada símbolo transporta SF bits.
        B (float): Ancho de banda de la señal, en Hz. También es la frecuencia de muestreo.
        n_upchirps (int): Cantidad de up-chirps (símbolos 0) del preámbulo.
        n_downchirps (float): Cantidad de down-chirps del preámbulo. Puede ser fraccionaria.
    """

    SF: int
    B: float
    n_upchirps: int = 8
    n_downchirps: float = 2.25

    @property
    def M(self) -> int:
        """Chips (muestras) por símbolo: 2^SF."""
        return 2**self.SF

    @property
    def T(self) -> float:
        """Tiempo de chip: 1 / B."""
        return 1 / self.B

    @property
    def Ts(self) -> float:
        """Tiempo de símbolo: 2^SF chips de duración T."""
        return self.M * self.T

    @property
    def preamble_len(self) -> int:
        """Largo del preámbulo, en muestras: up-chirps completos y la fracción de down-chirps."""
        return self.n_upchirps * self.M + int(self.n_downchirps * self.M)
