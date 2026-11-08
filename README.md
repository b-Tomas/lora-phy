# Modulación LoRa (FSCM)

Modelo en Python de la capa física de LoRa basado en:

> L. Vangelista, ["Frequency Shift Chirp Modulation: The LoRa Modulation"](https://doi.org/10.1109/LSP.2017.2762960), *IEEE Signal Processing Letters*, vol. 24, no. 12, pp. 1818–1821, dic. 2017.

El desarrollo y los resultados están en [`LoRa.ipynb`](LoRa.ipynb). La implementación está en el
paquete [`lora`](lora/), que el notebook importa.

## Uso

Requiere Python 3.10 o posterior. Con [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev        # Crea .venv e instala el paquete lora en modo editable, Jupyter, pytest y ruff
uv run jupyter lab LoRa.ipynb
```

Sin uv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab LoRa.ipynb
```

Los cambios en `lora/` se ven en el notebook después de reiniciar el kernel.

### SDR (ADALM-Pluto)

El último capítulo del notebook prueba el receptor con un ADALM-Pluto del laboratorio. Por defecto
(`USE_SDR = False`) usa las capturas guardadas en `captures/` y no necesita el SDR.

Para Utilizar el SDR:

```bash
sudo pacman -S libiio        # libiio 0.26 del sistema (pylibiio queda fijado < 1)
uv sync --extra dev --extra sdr
# Con la VPN del laboratorio conectada, verificar la conexión
iio_info -u ip:192.168.1.31
```

Después, poner `USE_SDR = True` en el notebook y ejecutar.

### Tests y linting

```bash
uv run pytest
uv run ruff check .     # Linting (paquete, tests y notebook)
uv run ruff format .    # Formato
```

## Paquete `lora`

| Módulo                             | Contenido                                                          |
| ---------------------------------- | ------------------------------------------------------------------ |
| [`params`](lora/params.py)         | `LoRaParams` (SF, B, preámbulo) y semilla global (`seed`, `rng`)   |
| [`tx`](lora/tx.py)                 | Codificador, modulador y preámbulo                                 |
| [`channel`](lora/channel.py)       | AWGN, respuesta al impulso del canal y transmisión de una trama    |
| [`rx`](lora/rx.py)                 | Demodulador y decodificador                                        |
| [`metrics`](lora/metrics.py)       | Potencia, BER, error de trama y curva ROC                          |
| [`detection`](lora/detection.py)   | Detectores de trama y umbrales de Neyman-Pearson                   |
| [`simulation`](lora/simulation.py) | Simulaciones Monte Carlo: BER, tramas, ROC, falsas alarmas y costo |
| [`sdr`](lora/sdr.py)               | Transmisión y recepción con ADALM-Pluto                            |
| [`plots`](lora/plots.py)           | Gráficos del notebook                                              |

## Autores

| Autor                | Email                             |
| -------------------- | --------------------------------- |
| Badenes, Tomás       | tomasbadenes@mi.unc.edu.ar        |
| Pineda, Juan Ignacio | juan.ignacio.pineda@mi.unc.edu.ar |
| Doniquian, Pedro     | pedoniquian@mi.unc.edu.ar         |
