"""Modelo de la modulación LoRa (FSCM).

Basado en L. Vangelista, "Frequency Shift Chirp Modulation: The LoRa Modulation",
IEEE Signal Processing Letters, vol. 24, no. 12, pp. 1818–1821, dic. 2017.
"""

from .params import LoRaParams, rng, seed

__all__ = ["LoRaParams", "rng", "seed"]
