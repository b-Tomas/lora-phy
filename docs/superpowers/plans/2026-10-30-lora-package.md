# Paquete `lora` — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mudar todo el código de `LoRa.ipynb` a un paquete instalable `lora/`, con parámetros en `LoRaParams`, semilla global, tests en pytest, y reconectar el notebook para que solo llame al paquete.

**Architecture:** Paquete plano `lora/` en la raíz del TP, un módulo por etapa del sistema (params, tx, channel, rx, metrics, detection, simulation, plots). Funciones puras que reciben `p: LoRaParams` donde hoy reciben `SF, B`. Los barridos devuelven arrays; los gráficos reciben datos y devuelven `Figure`. El notebook conserva el markdown y sus celdas de código pasan a ser llamadas.

**Tech Stack:** Python ≥ 3.10, numpy, matplotlib, pytest, setuptools (`pyproject.toml`), Jupyter.

**Spec:** `docs/superpowers/specs/2026-10-30-lora-package-design.md`

**Fuente del código:** cada función ya existe en `LoRa.ipynb`. Para ver las celdas:
`python3 -c "import json; nb=json.load(open('LoRa.ipynb')); [print(f'--- [{i}] {c[\"cell_type\"]} ---\n' + ''.join(c['source'])) for i,c in enumerate(nb['cells'])]"`.
"Mudar de la celda N" significa: copiar la función con su docstring y comentarios, reemplazar `SF, B` por `p: LoRaParams` (y `2**SF` por `p.M`, `8` del preámbulo por `p.n_upchirps`), y adaptar el docstring. La matemática no cambia.

## Global Constraints

- Identificadores en inglés; docstrings, comentarios y textos de gráficos en español.
- Docstrings estilo Google (`Args:` / `Returns:`), con las referencias a Ec. 1, Ec. 2 y Ec. 16 que tienen hoy. Comentarios se conservan o mejoran, nunca se borran sin reemplazo.
- Type hints como los actuales (`np.ndarray`, `int | None`). Validación con `assert`.
- `requires-python = ">=3.10"`. Dependencias: `numpy`, `matplotlib`; extra `dev`: `jupyter`, `pytest`.
- Ninguna función usa `np.random.*` global: todo sale de `lora.params.rng`.
- Sin clases salvo `LoRaParams` y el `NamedTuple` de la Task 5. Sin CLI ni `__main__`.
- Misma matemática que el notebook: mismos estadísticos, umbrales y decisiones.

## Review Focus

1. **`seed()` tiene que afectar a todos los módulos**: un módulo que hizo `from .params import rng` antes de `seed(n)` debe ver el generador reiniciado. Test en Task 3 (`test_seed_resets_noise`).
2. **SF ≠ 7**: el barrido de CPU usa SF 7 a 10. Modulador/demodulador y `preamble_len` tienen que funcionar en todos. Tests parametrizados en Task 2.
3. **`detect_ideal` con gap ≠ 5**: hoy está hardcodeado. Test en Task 4 (`test_ideal_detector_gap`).
4. **Doble gráfico en Jupyter**: una celda que termina en `plots.x(...)` sin `;` muestra la figura dos veces. Chequeo en Task 7 (una imagen por celda).
5. **`preamble_len` con `n_downchirps` fraccionario**: debe coincidir con `preamble(p).size` (lo usa `frame_error` para ubicar el payload). Test en Task 2.

---

### Task 0: Rama

- [ ] **Step 1:** `git checkout -b refactor/lora-package`
- [ ] **Step 2:** Commit del spec y el plan:

```bash
git add docs/superpowers
git commit -m "docs: spec y plan del paquete lora"
```

### Task 1: Paquete instalable y `LoRaParams`

**Files:**
- Create: `pyproject.toml`, `lora/__init__.py`, `lora/params.py`, `tests/test_params.py`
- Modify: `requirements.txt`, `.gitignore`

**Interfaces:**
- Produces:
  - `lora.params.SEED: int = 0`
  - `lora.params.rng: np.random.Generator`
  - `lora.params.seed(n: int) -> None`: reinicia `rng` **en el lugar** (`rng.bit_generator.state = np.random.default_rng(n).bit_generator.state`), nunca reasigna la variable.
  - `lora.params.LoRaParams` (`@dataclass(frozen=True)`): `SF: int`, `B: float`, `n_upchirps: int = 8`, `n_downchirps: float = 2.25`; propiedades `M -> int` (`2**SF`), `T -> float` (`1 / B`), `Ts -> float` (`M * T`), `preamble_len -> int` (`n_upchirps * M + int(n_downchirps * M)`).
  - `lora/__init__.py` exporta `LoRaParams`, `seed`, `rng`.

- [ ] **Step 1: Packaging.** `pyproject.toml` con `[build-system]` setuptools, `[project] name = "lora"`, `version = "0.1.0"`, `requires-python = ">=3.10"`, `dependencies = ["numpy", "matplotlib"]`, `[project.optional-dependencies] dev = ["jupyter", "pytest"]`, `[tool.setuptools] packages = ["lora"]`. `requirements.txt` = `-e .[dev]`. Agregar `*.egg-info/` a `.gitignore`.

- [ ] **Step 2: Tests que fallan** en `tests/test_params.py`:

```python
def test_derived_properties():
    p = LoRaParams(SF=7, B=125e3)
    assert p.M == 128
    assert p.T == 1 / 125e3
    assert p.Ts == 128 / 125e3
    assert p.preamble_len == 8 * 128 + 9 * 128 // 4

def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        LoRaParams(7, 125e3).SF = 8

def test_seed_reproducible():
    seed(3); a = rng.integers(0, 2, 100)
    seed(3); b = rng.integers(0, 2, 100)
    assert np.array_equal(a, b)
```

- [ ] **Step 3:** `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pytest tests/test_params.py -v` → FAIL (`lora.params` no existe).
- [ ] **Step 4:** Implementar `lora/params.py` y `lora/__init__.py`. Docstring de `LoRaParams` explicando cada campo (M = chips por símbolo, Ts = tiempo de símbolo, etc.).
- [ ] **Step 5:** `pytest -v` → PASS.
- [ ] **Step 6:** `git add pyproject.toml requirements.txt .gitignore lora tests && git commit -m "feat: paquete lora instalable con LoRaParams y semilla"`

### Task 2: TX, RX y métricas básicas

**Files:**
- Create: `lora/tx.py`, `lora/rx.py`, `lora/metrics.py`, `tests/test_tx_rx.py`

**Interfaces:**
- Consumes: `LoRaParams`.
- Produces:
  - `tx.encoder(bits: np.ndarray, p: LoRaParams) -> np.ndarray` (celda 5)
  - `tx.modulator(symbols: np.ndarray, p: LoRaParams) -> np.ndarray` (celda 8)
  - `tx.instantaneous_frequency(symbols: np.ndarray, p: LoRaParams) -> np.ndarray`: `((s + k) % M) * B / M` aplanado, en Hz (hoy duplicado en celdas 11 y 42).
  - `tx.preamble(p: LoRaParams) -> np.ndarray` (celda 39), usa `p.n_upchirps` y `p.n_downchirps`.
  - `rx.dechirp_fft(r: np.ndarray, p: LoRaParams) -> np.ndarray`: matriz `D` de forma `(N, M)`, down-chirp y FFT `norm="ortho"` (celda 15, sin el argmax).
  - `rx.demodulator(r: np.ndarray, p: LoRaParams) -> np.ndarray`: `argmax(|dechirp_fft(r, p)|**2, axis=1)`.
  - `rx.decoder(symbols: np.ndarray, p: LoRaParams) -> np.ndarray` (celda 20)
  - `metrics.power(c: np.ndarray) -> float` (celda 26)
  - `metrics.BER(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float` (celda 23), con `np.mean(bits_tx ^ bits_rx)`.

- [ ] **Step 1: Tests que fallan** (migrados de celdas 6, 16, 21, 40 más Review Focus 2 y 5):

```python
p = LoRaParams(7, 125e3)

def test_encoder():
    assert np.array_equal(tx.encoder(np.array([0, 0, 0, 1, 0, 0, 0]), p), [8])
    assert np.array_equal(tx.encoder(np.array([0,0,0,1,0,0,0, 1,1,1,1,1,1,1]), p), [8, 127])

def test_decoder():
    assert np.array_equal(rx.decoder(np.array([8]), p), [0, 0, 0, 1, 0, 0, 0])
    assert np.array_equal(rx.decoder(np.array([8, 127]), p), [0,0,0,1,0,0,0, 1,1,1,1,1,1,1])

@pytest.mark.parametrize("SF", [7, 8, 9, 10])
def test_roundtrip(SF):
    q = LoRaParams(SF, 125e3)
    symbols = np.array([0, 32, 80, q.M - 1])
    assert np.array_equal(rx.demodulator(tx.modulator(symbols, q), q), symbols)

def test_dechirp_fft_matches_demodulator():
    symbols = np.array([0, 32, 80])
    D = rx.dechirp_fft(tx.modulator(symbols, p), p)
    assert D.shape == (3, p.M)
    assert np.array_equal(np.argmax(np.abs(D) ** 2, axis=1), symbols)

@pytest.mark.parametrize("SF", [7, 9])
def test_preamble_len(SF):
    q = LoRaParams(SF, 125e3)
    assert tx.preamble(q).size == q.preamble_len

def test_power_and_ber():
    assert np.isclose(metrics.power(tx.modulator(np.array([0]), p)), 1 / p.M)
    bits = np.array([0, 1, 1, 0])
    assert metrics.BER(bits, bits) == 0
    assert metrics.BER(bits, bits ^ np.array([1, 0, 0, 0])) == 0.25
```

- [ ] **Step 2:** `pytest tests/test_tx_rx.py -v` → FAIL (módulos no existen).
- [ ] **Step 3:** Mudar las funciones. Arreglar el mensaje del `assert` de `demodulator`: "El número de muestras debe ser múltiplo de 2**SF".
- [ ] **Step 4:** `pytest -v` → PASS.
- [ ] **Step 5:** `git add lora tests && git commit -m "feat: tx, rx y métricas básicas"`

### Task 3: Canal y error de trama

**Files:**
- Create: `lora/channel.py`, `tests/conftest.py`, `tests/test_channel.py`
- Modify: `lora/metrics.py`

**Interfaces:**
- Consumes: `tx.modulator`, `tx.preamble`, `rx.demodulator`, `metrics.power`, `params.rng`.
- Produces:
  - `channel.FLAT = np.array([1.0])`, `channel.FREQ_SELECTIVE = np.array([sqrt(0.8), sqrt(0.2)])`
  - `channel.noise_from_SNR(c: np.ndarray, SNR_dB: float) -> np.ndarray` (celda 29), con `rng.normal`.
  - `channel.noise_power(p: LoRaParams, SNR_dB: float) -> float`: `power(preamble(p)) / 10 ** (SNR_dB / 10)`. Reemplaza las cinco repeticiones.
  - `channel.system(c: np.ndarray, h: np.ndarray, z: np.ndarray) -> np.ndarray` (celda 30)
  - `channel.transmit_frame(symbols: np.ndarray, gap: int, p: LoRaParams, SNR_dB: float, h: np.ndarray) -> np.ndarray` (celda 39)
  - `metrics.frame_error(r: np.ndarray, start: int | None, symbols_tx: np.ndarray, p: LoRaParams) -> bool` (celda 43), con `p.preamble_len` para ubicar el payload.
  - `tests/conftest.py`: fixtures `p` (`LoRaParams(7, 125e3)`) y `clean_frame` → `(r, symbols, gap, noise_power)` con 8 símbolos aleatorios, `gap=3`, SNR 100 dB, `FLAT` (escenario de la celda 44, que reutilizan Task 4 y 5).

- [ ] **Step 1: Tests que fallan** (celda 44 más Review Focus 1):

```python
def test_frame_error(p, clean_frame):
    r, symbols, gap, _ = clean_frame
    assert not metrics.frame_error(r, gap * p.M, symbols, p)
    assert metrics.frame_error(r, gap * p.M + 1, symbols, p)
    assert metrics.frame_error(r, (gap - 1) * p.M, symbols, p)
    assert metrics.frame_error(r, None, symbols, p)

def test_noise_power_matches_snr(p):
    c = tx.modulator(rng.integers(0, p.M, 2000), p)
    z = channel.noise_from_SNR(c, -10)
    assert np.isclose(metrics.power(z), metrics.power(c) * 10, rtol=0.05)

def test_seed_resets_noise(p):
    c = tx.preamble(p)
    seed(5); a = channel.noise_from_SNR(c, 0)
    seed(5); b = channel.noise_from_SNR(c, 0)
    assert np.array_equal(a, b)
```

- [ ] **Step 2:** `pytest tests/test_channel.py -v` → FAIL.
- [ ] **Step 3:** Mudar las funciones.
- [ ] **Step 4:** `pytest -v` → PASS.
- [ ] **Step 5:** `git add lora tests && git commit -m "feat: canal ruidoso y error de trama"`

### Task 4: Detección de trama

**Files:**
- Create: `lora/detection.py`, `tests/test_detection.py`
- Modify: `lora/metrics.py` (`roc`)

**Interfaces:**
- Consumes: `tx.modulator`, `rx.demodulator`, `LoRaParams`.
- Produces (todo de las celdas 43, 47, 51, 59, 67 y 81):
  - `Detector = Callable[[np.ndarray, LoRaParams, float, float], int | None]`: firma `detect(r, p, noise_power, P_FA)`.
  - `upchirp_correlation(r, p) -> np.ndarray`
  - `preamble_correlation(r, p) -> np.ndarray`, `time_threshold(noise_power: float, P_FA: float, p) -> float`
  - `peaks_statistic(r, p) -> np.ndarray`, `peaks_threshold(noise_power: float, P_FA: float, p) -> float`
  - `preamble_zero_count(r, p) -> np.ndarray`, `zero_count_false_alarm(K: int, p) -> float`, `freq_threshold(P_FA: float, p) -> int`
  - `detect_start(statistic: np.ndarray, threshold: float, span: int) -> int | None` (`span` obligatorio).
  - `detect_full`, `detect_peaks`, `detect_freq`: `Detector`.
  - `ideal_detector(gap: int) -> Detector`: devuelve un detector que siempre responde `gap * p.M` (reemplaza `detect_ideal`, que tenía `5` hardcodeado).
  - `DETECTORS: dict[str, Detector] = {"full": detect_full, "peaks": detect_peaks, "freq": detect_freq}`. Estas claves son el vocabulario común de los métodos en `simulation` y `plots`.
  - `metrics.roc(statistic_h0, statistic_h1) -> tuple[np.ndarray, np.ndarray]` (celda 81).

- [ ] **Step 1: Tests que fallan** (celdas 53, 61, 69 y 82; `q1 = dataclasses.replace(p, n_upchirps=1)`; `start = gap * p.M`):

```python
def test_full(p, clean_frame):
    r, _, gap, npw = clean_frame
    T = detection.preamble_correlation(r, p)
    assert detection.detect_start(T, detection.time_threshold(npw, 1e-3, p), 8 * p.M) == gap * p.M
    assert detection.detect_full(r, p, npw, 1e-3) == gap * p.M

def test_peaks_threshold(p):
    q1 = dataclasses.replace(p, n_upchirps=1)
    assert np.isclose(detection.peaks_threshold(2.0, 1e-3, q1), detection.time_threshold(2.0, 1e-3, q1))
    x = detection.peaks_threshold(1.0, 1e-3, p)
    assert np.isclose(np.exp(-x) * sum(x**i / factorial(i) for i in range(8)), 1e-3)

def test_peaks(p, clean_frame):
    r, _, gap, npw = clean_frame
    S = detection.peaks_statistic(r, p)
    assert S.size == detection.preamble_correlation(r, p).size
    assert detection.detect_start(S, detection.peaks_threshold(npw, 1e-3, p), 8 * p.M) == gap * p.M
    assert detection.detect_peaks(r, p, npw, 1e-3) == gap * p.M

def test_freq(p, clean_frame):
    r, _, gap, npw = clean_frame
    assert detection.detect_start(detection.preamble_zero_count(r, p), 6, 8) == gap
    assert detection.detect_freq(r, p, npw, 1e-3) == gap * p.M
    assert detection.freq_threshold(1e-3, p) == 3
    assert detection.freq_threshold(1.6e-5, p) == 4
    assert detection.freq_threshold(2e-3, p) == 2
    assert detection.freq_threshold(0.5, p) == 1
    assert np.isclose(detection.zero_count_false_alarm(1, p), 1 - (1 - 2**-7) ** 8)

def test_ideal_detector_gap(p, clean_frame):
    r, _, gap, npw = clean_frame
    assert detection.ideal_detector(gap)(r, p, npw, 1e-3) == gap * p.M

def test_roc():
    P_FA, P_D = metrics.roc(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    assert P_FA[0] == 0 and P_D[0] == 0
    assert np.any((P_FA == 0) & (P_D == 1))
    assert P_FA[-1] == 1 and P_D[-1] == 1
```

- [ ] **Step 2:** `pytest tests/test_detection.py -v` → FAIL.
- [ ] **Step 3:** Mudar las funciones. `peaks_threshold` sigue usando bisección con `math.factorial`; `zero_count_false_alarm` con `math.comb`.
- [ ] **Step 4:** `pytest -v` → PASS.
- [ ] **Step 5:** `git add lora tests && git commit -m "feat: detectores de trama y umbrales de Neyman-Pearson"`

### Task 5: Simulaciones

**Files:**
- Create: `lora/simulation.py`, `tests/test_simulation.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces (todas devuelven datos, nunca grafican):
  - `transmission_ber(p, SNR_dB: float, h: np.ndarray, num_symbols: int = 100) -> float` (celda 31; se renombra de `simulation` para no chocar con el nombre del módulo).
  - `ber_vs_snr(p, SNRs_dB: np.ndarray, h: np.ndarray, simulations: int, num_symbols: int = 100) -> np.ndarray`: promedio de `transmission_ber` por SNR (celda 37).
  - `frame_rates(detect: Detector, SNRs_dB, h, p, P_FA: float, simulations: int, num_symbols: int = 16, gap: int = 5) -> tuple[np.ndarray, np.ndarray]` (celda 43): `(p_start, fer)`.
  - `detection_statistics(SNR_dB: float, p, simulations: int) -> dict[str, tuple[np.ndarray, np.ndarray]]` (celda 81): clave de método → `(h0, h1)`.
  - `class RocCurves(NamedTuple)`: `P_D_theory: np.ndarray`, `P_FA_empirical: np.ndarray`, `P_D_empirical: np.ndarray`, `P_D_operating: float`.
  - `roc_curves(SNR_dB: float, p, simulations: int, P_FA_axis: np.ndarray, P_FA: float) -> dict[str, RocCurves]`: cálculo de la celda 83 sin el gráfico. `P_FA_axis` y `P_FA` son por símbolo; en el tiempo el umbral usa `P_FA / p.M` y la `P_FA` empírica se multiplica por `p.M`; sin `lambda`s en el notebook.
  - `measured_false_alarms(p, num_noise_symbols: int, P_FA_targets: np.ndarray, Ks: tuple[int, ...] = (1, 2)) -> dict[str, tuple[np.ndarray, np.ndarray]]` (celda 78): método → `(P_FA pedida, P_FA medida)`, por posición, ruido de −10 dB.
  - `cpu_time(fn: Callable[[], object]) -> float` (celda 86).
  - `methods_cpu_time(p, P_FA: float) -> dict[str, float]` (celda 86): método → segundos por trama.

- [ ] **Step 1: Tests que fallan** (celda 44 más checks de forma; corridas chicas para que sea rápido):

```python
def test_frame_rates_ideal(p):
    p_start, fer = simulation.frame_rates(detection.ideal_detector(5), [100], channel.FLAT, p, 1e-3, 3)
    assert p_start[0] == 1 and fer[0] == 0

def test_ber_vs_snr_noiseless(p):
    assert np.all(simulation.ber_vs_snr(p, np.array([100.0]), channel.FLAT, 2) == 0)

def test_roc_curves_shape(p):
    axis = np.logspace(-7, 0, 10)
    curves = simulation.roc_curves(-14, p, 20, axis, 1e-5)
    assert set(curves) == {"full", "peaks", "freq"}
    assert all(c.P_D_theory.shape == axis.shape for c in curves.values())

def test_measured_false_alarms_keys(p):
    res = simulation.measured_false_alarms(p, 200, np.logspace(-1, -2, 3))
    assert set(res) == {"full", "peaks", "freq"}
```

- [ ] **Step 2:** `pytest tests/test_simulation.py -v` → FAIL.
- [ ] **Step 3:** Mudar e implementar. Los símbolos y bits aleatorios salen de `rng.integers`.
- [ ] **Step 4:** `pytest -v` → PASS.
- [ ] **Step 5:** `git add lora tests && git commit -m "feat: simulaciones de BER, tramas, ROC y costo"`

### Task 6: Gráficos

**Files:**
- Create: `lora/plots.py`, `tests/test_plots.py`

**Interfaces:**
- Consumes: datos de las Tasks 2–5; `tx.instantaneous_frequency`.
- Produces: una función por gráfico, cada una crea su figura con `plt.subplots`, la devuelve (`matplotlib.figure.Figure`) y no llama a `plt.show()`. Nombres y argumentos exactos quedan a tu criterio, con este mapeo:

| Celda | Contenido | Recibe |
|---|---|---|
| 11 | Señal modulada: real, imaginaria, frecuencia instantánea, espectrograma | `c`, `symbols`, `p` |
| 18, 34 | `|D|²` por símbolo, con transmitido y decidido (SNR opcional en el título) | `D`, `symbols_tx`, `symbols_rx`, `SNR_dB=None` |
| 37 | BER vs SNR (semilogy) | SNRs, `dict[etiqueta, bers]` |
| 42 | Estructura de la trama: frecuencia instantánea teórica + espectrograma recibido | `r`, `symbols_tx`, `gap`, `p`, `SNR_dB` |
| 48 | Correlación móvil sin ruido, con ruido y detalle del inicio | `y_clean`, `y`, `gap`, `p`, `SNR_dB` |
| 54 | `T[k]` sin ruido y con ruido, umbral, inicio real y detectado | `T_clean`, `T`, `eta`, `start`, `gap`, `p`, `P_FA`, `SNR_dB` |
| 62 | `|y|²` con los 8 picos detectados y `S[k]` con umbral | `y`, `S`, `eta_S`, `start`, `gap`, `p`, `P_FA`, `SNR_dB` |
| 70 | Símbolo por ventana y `V[m]` con umbral `K` | `symbols_rx`, `V`, `K`, `start`, `gap`, `P_FA`, `SNR_dB` |
| 56, 64, 72 | P(inicio correcto) vs SNR, plano y selectivo, de un método | SNRs, `p_flat`, `p_sel`, título |
| 75, 90 | Comparación de métodos: llena = plano, punteada = selectivo | SNRs, `flat: dict[método, arr]`, `sel: dict[método, arr]`, `ylabel`, título, `xlim=None` |
| 78 | Falsas alarmas medidas vs pedidas (loglog, diagonal) | salida de `measured_false_alarms`, `num_noise_symbols` |
| 83 | ROC, un panel por SNR, leyenda única abajo | `dict[SNR_dB, dict[método, RocCurves]]`, `P_FA_axis`, `P_FA` |
| 87 | Tiempo de CPU vs SF | `SFs`, `list[dict[método, float]]` |

Helpers privados: `_METHODS = {"full": ("Tiempo (preámbulo completo)", "C0"), "peaks": ("Tiempo (8 picos)", "C2"), "freq": ("Frecuencia (demodulación)", "C1"), "ideal": ("Detección ideal", "k")}`; límites y nombres de las partes de la trama (Silencio, Up-chirps, Down-chirps, Payload) con sus etiquetas; líneas verticales entre símbolos (blancas sobre el espectrograma, negras en el resto). Títulos, ejes y leyendas: los mismos textos que en el notebook.

- [ ] **Step 1: Smoke test que falla** en `tests/test_plots.py`: con `matplotlib.use("Agg")`, construir datos chicos con el paquete (una trama limpia, `ber_vs_snr` con 1 SNR y 1 simulación, `roc_curves` con 10 realizaciones, etc.), llamar a cada función de `plots` y verificar que devuelve una `Figure`. Cerrar las figuras al final (`plt.close("all")`).
- [ ] **Step 2:** `pytest tests/test_plots.py -v` → FAIL.
- [ ] **Step 3:** Mudar el código de gráficos de cada celda.
- [ ] **Step 4:** `pytest -v` → PASS.
- [ ] **Step 5:** `git add lora tests && git commit -m "feat: gráficos del notebook en lora.plots"`

### Task 7: Reconectar el notebook y README

**Files:**
- Modify: `LoRa.ipynb`, `README.md`

- [ ] **Step 1: Celda de setup** (reemplaza la celda 2 de imports): `import numpy as np`, `from lora import LoRaParams, seed, rng`, `from lora import tx, rx, channel, metrics, detection, simulation, plots`, `seed(0)`.
- [ ] **Step 2: Borrar** las celdas que definen funciones (5, 8, 15, 20, 23, 26, 29, 30, 31, 39, 43, 47, 51, 59, 67, 81, 86) y las de tests (6, 16, 21, 40, 44, 53, 61, 69, 82). Cada markdown que precedía a una definición queda en su lugar: explica la teoría de lo que la celda siguiente usa.
- [ ] **Step 3: Reescribir las celdas de cálculo y gráfico** (10, 11, 18, 24, 27, 32, 34, 37, 42, 45, 48, 54, 56, 62, 64, 70, 72, 75, 78, 83, 87, 90) como llamadas: parámetros arriba (`p = LoRaParams(SF=7, B=125e3)`, `P_FA = R_FA * p.Ts`, `SNRs_rates_dB`, `simulations`), luego `simulation.*`/`detection.*` y al final `plots.*(...);` con `;`. Aleatorios con `rng.integers`. La tabla de tiempos relativos de la celda 87 queda como `print` corto. Ninguna celda define funciones ni `lambda`s.
- [ ] **Step 4: Markdown:** en la celda 41 cambiar `detect(r, SF, B, noise_power, P_FA)` por `detect(r, p, noise_power, P_FA)`. Buscar otras menciones de firmas viejas (`grep` de `SF, B` en las celdas markdown) y actualizarlas. No tocar nada más del texto.
- [ ] **Step 5: Ejecutar** `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 LoRa.ipynb` → termina sin errores.
- [ ] **Step 6: Verificar salidas:**
  - Ninguna celda de código contiene `def ` ni `lambda`.
  - Cada celda de gráfico tiene exactamente una salida `image/png` (sin gráficos dobles).
  - Revisar visualmente cada gráfico contra la versión de `main` (`git show main:LoRa.ipynb`): misma forma de las curvas y mismos órdenes de magnitud. Anotar en el mensaje de entrega qué observaciones del markdown quedaron con números distintos (se revisan en la fase 2).
- [ ] **Step 7: README:** sección de uso con `pip install -r requirements.txt` (instala `lora` editable), `pytest` para los tests, nota de reiniciar el kernel tras editar `lora/`, y una tabla corta de los módulos del paquete.
- [ ] **Step 8: Verificación final en venv limpio:** `V=$(mktemp -d)/venv && python -m venv $V && $V/bin/pip install -r requirements.txt && $V/bin/pytest` → PASS.
- [ ] **Step 9:** `git add LoRa.ipynb README.md && git commit -m "refactor: notebook llama al paquete lora"`
