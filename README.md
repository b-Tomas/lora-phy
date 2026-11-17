# Modulación LoRa (FSCM)

Modelo en Python de la capa física de LoRa basado en:

> L. Vangelista, ["Frequency Shift Chirp Modulation: The LoRa Modulation"](https://doi.org/10.1109/LSP.2017.2762960), *IEEE Signal Processing Letters*, vol. 24, no. 12, pp. 1818–1821, dic. 2017.

El desarrollo, la implementación y los resultados están en [`LoRa.ipynb`](LoRa.ipynb): todo el
código está en el notebook, que solo usa numpy y matplotlib.

## Uso

Requiere Python 3.10 o posterior. Con [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev        # Crea .venv e instala numpy, matplotlib, Jupyter y ruff
uv run jupyter lab LoRa.ipynb
```

Sin uv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab LoRa.ipynb
```

### SDR (ADALM-Pluto)

El último capítulo del notebook prueba el receptor con un ADALM-Pluto del laboratorio. Por defecto
(`USAR_SDR = False`) usa las capturas guardadas en `captures/` y no necesita el SDR.

Para Utilizar el SDR:

```bash
sudo pacman -S libiio        # libiio 0.26 del sistema (pylibiio queda fijado < 1)
uv sync --extra dev --extra sdr
# Con la VPN del laboratorio conectada, verificar la conexión
iio_info -u ip:192.168.1.32
```

Después, poner `USAR_SDR = True` en el notebook y ejecutar.

### Linting

```bash
uv run ruff check .     # Linting del notebook
uv run ruff format .    # Formato
```

## Autores

| Autor                | Email                             |
| -------------------- | --------------------------------- |
| Badenes, Tomás       | tomasbadenes@mi.unc.edu.ar        |
| Pineda, Juan Ignacio | juan.ignacio.pineda@mi.unc.edu.ar |
| Doniquian, Pedro     | pedoniquian@mi.unc.edu.ar         |
