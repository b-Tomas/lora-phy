# Paquete `lora`: refactor del notebook a un módulo de Python

Fecha: 2026-10-30

## Intención

`LoRa.ipynb` es la entrega del TP y lo lee el profesor. Hoy mezcla tres cosas en las celdas:
la implementación de DSP, el código de los gráficos y los barridos Monte Carlo. Eso hace que
la narrativa sea difícil de seguir y que el código sea difícil de mantener.

Se separa en dos partes:

- **Notebook**: teoría en markdown (ecuaciones, hipótesis, observaciones) y celdas de código
  cortas que llaman al paquete y muestran resultados y gráficos. Al profesor le importa la
  teoría y los resultados, no el código.
- **Paquete `lora/`**: la implementación, para los autores. Legible, bien organizado y comentado,
  con el mismo estilo que el código actual. Expone cada paso por separado (codificar, modular,
  canal, demodular, cada detector, cada umbral, cada gráfico) para que el notebook pueda
  armar la narrativa paso a paso.

Este documento cubre **la fase 1: armar el paquete y reconectar el notebook**. La reescritura y
el completado de la narrativa (diagrama de bloques, sección del canal selectivo, etc.) es la
fase 2 y queda fuera de este spec.

## Objetivos

1. Todo el código de las celdas (funciones de DSP, simulaciones, gráficos) se muda a `lora/`.
2. Los parámetros del sistema viajan juntos en un objeto de configuración (`LoRaParams`) en vez
   de pasar `SF, B` sueltos a cada función.
3. Los `assert` del notebook pasan a ser tests de pytest en `tests/`.
4. Una semilla global hace reproducibles las corridas.
5. El notebook queda con el mismo markdown y celdas de código que llaman al paquete. Al
   ejecutarlo de punta a punta, produce los mismos gráficos que hoy (salvo la variación
   estadística por la semilla).

## No-objetivos

- CLI, `__main__` o `python -m lora...`: el paquete solo se usa con import.
- Publicarlo en PyPI o versionarlo: se instala desde el repo.
- Cache de resultados, paralelismo u optimizaciones de performance (por ejemplo, la correlación
  por FFT). El tiempo de ejecución del notebook no es un problema.
- Clases para los bloques del sistema o un objeto "pipeline" TX→canal→RX. Las funciones siguen
  siendo funciones; lo único nuevo es el objeto de parámetros.
- Reescribir la narrativa (fase 2).

## Anti-objetivos (sería un fracaso aunque funcione)

- Cambiar los resultados numéricos de algún algoritmo. Es un refactor: misma matemática, mismos
  estadísticos, mismos umbrales.
- Que el código del paquete quede peor documentado que el actual: se pierden los comentarios
  que explican cada paso con referencia a las ecuaciones del paper.
- Jerarquías de clases, factories o abstracciones sin un segundo uso.

## Decisiones

### Parámetros: `LoRaParams`

```python
@dataclass(frozen=True)
class LoRaParams:
    SF: int
    B: float
    n_upchirps: int = 8
    n_downchirps: float = 2.25
```

Con propiedades derivadas: `M` (`2**SF`, chips por símbolo), `T` (`1 / B`), `Ts` (`M * T`) y
`preamble_len` (muestras del preámbulo). Reemplazan los `M: int = 2**SF` que hoy se repiten en
cada función. Los `n_upchirps=8` que hoy son argumentos por defecto de varias funciones
(`preamble_correlation`, `peaks_statistic`, `preamble_zero_count`, thresholds, el `8 * M` de
`detect_start`) salen de `p.n_upchirps`.

Todas las funciones que hoy reciben `SF` y/o `B` reciben `p: LoRaParams` en su lugar. Por
ejemplo: `modulator(symbols, p)`, `demodulator(r, p)`, `detect_full(r, p, noise_power, P_FA)`.

El canal (`h`) y `SNR_dB` **no** van en `LoRaParams`: son lo que se barre en las simulaciones,
así que quedan como argumentos aparte.

### Semilla global

Un único `numpy.random.Generator` del paquete, que usan todas las funciones aleatorias
(generación de bits/símbolos y ruido). Se expone `lora.seed(n)` para cambiarla desde el
notebook. Semilla por defecto: `0` (decisión mía).

Cuidado al implementarlo: si los módulos hacen `from .params import rng` y `seed()` reasigna la
variable, los módulos se quedan con el generador viejo. `seed()` tiene que reiniciar el mismo
objeto (por ejemplo, reasignando `rng.bit_generator.state`), o los módulos tienen que acceder al
generador a través del módulo (`params.rng`).

Los números de las observaciones del notebook ("50 % cerca de −17.5 dB", etc.) se escribieron
con corridas sin semilla. Después del refactor hay que revisarlos contra la corrida con
semilla. Eso es parte de la fase 2, salvo que alguno quede claramente mal.

### Instalación

`lora` es un paquete instalable con `pyproject.toml` (setuptools) en la raíz del TP:

- Dependencias del paquete: `numpy` y `matplotlib`. Extra `dev`: `jupyter` y `pytest`.
- `[tool.setuptools] packages = ["lora"]` explícito, para que la autodetección no levante
  `tests/` ni `docs/`.
- `requires-python = ">=3.10"` (el código usa `int | None`).
- `requirements.txt` pasa a ser `-e .[dev]`, así el flujo actual del README
  (`pip install -r requirements.txt`) instala el paquete en modo editable con todo lo necesario.
  Editar `lora/` se refleja en el notebook sin reinstalar (reiniciando el kernel, o con
  `%autoreload`).
- El README se actualiza: instalación, cómo correr los tests y la estructura del paquete.

### Estructura del paquete

Un paquete plano (`lora/` en la raíz del TP, sin `src/`) (decisión mía, ajustable si el builder
encuentra una división más natural, siempre que siga siendo plana y con pocos archivos):

| Módulo | Contenido (de las celdas actuales) |
|---|---|
| `lora/__init__.py` | Exporta `LoRaParams` y `seed`. Nada más. |
| `lora/params.py` | `LoRaParams`, el generador aleatorio y `seed()`. |
| `lora/tx.py` | `encoder`, `modulator`, `preamble`, frecuencia instantánea (hoy está duplicada en dos celdas de gráficos). |
| `lora/channel.py` | `noise_from_SNR`, `system`, `transmit_frame`, `noise_power(p, SNR_dB)` (hoy la expresión `power(preamble(...)) / 10 ** (SNR_dB / 10)` se repite cinco veces), constantes `FLAT` y `FREQ_SELECTIVE`. |
| `lora/rx.py` | `demodulator`, `decoder`. El demodulador se separa en `dechirp_fft(r, p)` (devuelve `D`, la salida de la FFT por símbolo) y `demodulator` (argmax de `D`), porque hoy el notebook repite los pasos del demodulador dos veces solo para graficar `D`. |
| `lora/metrics.py` | `power`, `BER`, `frame_error`, `roc`. |
| `lora/detection.py` | `upchirp_correlation`, los tres estadísticos (`preamble_correlation`, `peaks_statistic`, `preamble_zero_count`), sus umbrales (`time_threshold`, `peaks_threshold`, `zero_count_false_alarm`, `freq_threshold`), `detect_start` y los detectores `detect_full`, `detect_peaks`, `detect_freq`, `detect_ideal`. |
| `lora/simulation.py` | `simulation` (BER de una transmisión), `frame_rates`, `detection_statistics`, `cpu_time`, `methods_cpu_time`. Devuelven arrays; no grafican. |
| `lora/plots.py` | Una función por gráfico del notebook (ver abajo). |
| `tests/` | Los `assert` actuales, en pytest. |

El notebook importa los submódulos: `from lora import LoRaParams, tx, rx, channel, ...`.

### Gráficos

- Una función en `plots.py` por cada gráfico actual del notebook: señal modulada (4 paneles),
  salida de la FFT por símbolo (con y sin ruido), BER vs SNR, estructura de la trama, correlación
  móvil, estadístico de cada detector con su umbral, P(inicio correcto) vs SNR, comparación de
  detectores, falsas alarmas, ROC, tiempo de CPU y FER.
- Reciben los datos ya calculados (señales, resultados de los barridos) y no corren
  simulaciones. Los barridos se corren en el notebook con `simulation.*` y el resultado se pasa
  a `plots.*`. Así la celda muestra qué se simula y con qué parámetros.
- Cada función crea su figura, la devuelve y no llama a `plt.show()`.
- Cuidado: con el backend inline de Jupyter, una figura creada en la celda se muestra
  automáticamente. Si además la celda termina con la expresión `plots.algo(...)`, se muestra dos
  veces. En el notebook se llama con `;` al final o se asigna el resultado.
- Lo repetido en los gráficos actuales (límites y nombres de las partes de la trama, líneas
  verticales entre símbolos, etiquetas de los tres métodos con sus colores `C0`/`C2`/`C1`) se
  factoriza en helpers privados de `plots.py`.
- Los textos de los gráficos (títulos, ejes, leyendas) siguen en español, como ahora.

### Estilo

- Identificadores en inglés y docstrings y comentarios en español, como el código actual.
- Docstrings estilo Google (`Args:` / `Returns:`), como ahora.
- Se conservan los comentarios que explican cada paso y referencian las ecuaciones del paper
  (Ec. 1, Ec. 2, Ec. 16). Se pueden mejorar, nunca se borran sin reemplazo.
- Type hints como los actuales (`np.ndarray`, `int | None`).
- Validación de entradas con `assert` como ahora (no es una frontera de confianza, es un TP).

### Tests

- `tests/` con pytest (en el extra `dev`).
- Se migran **todos** los `assert` de las celdas de tests del notebook (celdas 6, 16, 21, 40, 44,
  53, 61, 69 y 82), adaptados a `LoRaParams`. No hace falta agregar tests nuevos más allá de un
  caso para `seed()` (misma semilla → mismos bits) y uno para `dechirp_fft` coherente con
  `demodulator`.
- Se corren con `pytest` desde la raíz del TP, con el paquete instalado.

### Notebook (fase 1)

- El markdown se conserva tal cual, salvo referencias a firmas que cambian (por ejemplo, la
  celda 41 menciona `detect(r, SF, B, noise_power, P_FA)`).
- Se eliminan las celdas que definen funciones y las celdas de tests.
- Las celdas que calculan y grafican pasan a llamar al paquete. Una celda de setup al principio
  importa el paquete y fija la semilla.
- Se ejecuta de punta a punta y se guarda con las salidas.

## Correcciones menores permitidas

Errores que aparecieron al leer el código. El builder los corrige como parte del refactor,
porque no cambian los resultados:

- `detect_ideal` tiene el `gap` de 5 símbolos hardcodeado. Tiene que recibir el gap del
  escenario (cómo resolverlo queda a criterio del builder, siempre que `frame_rates` siga
  aceptando cualquier detector con la misma firma).
- El mensaje del `assert` en `demodulator` dice "múltiplo de SF" y debería decir múltiplo de
  `2**SF`.
- `BER` usa el `sum` de Python sobre un array de numpy. Cambiarlo por `np.sum`/`np.mean` da el
  mismo resultado.

## Detalles resueltos (decisiones mías, tras la revisión)

- Las celdas que hoy usan `np.random` directo (bits/símbolos aleatorios en las celdas 24, 27, 32
  y 42) sacan los números del generador del paquete, no de `np.random`.
- El cálculo de la ROC teórica (celda 83, hoy con `lambda`s por método) se muda a
  `simulation.py`. La celda solo llama y grafica.
- Los tests de umbrales con un solo pico usan `dataclasses.replace(p, n_upchirps=1)`.
- La tabla de tiempos relativos de la celda 87 queda como un `print` corto en la celda.
- Una sola semilla fijada en la celda de setup implica que agregar una celda aleatoria en el
  medio cambia todos los resultados posteriores. Se acepta. En la fase 2 se puede reiniciar la
  semilla al comienzo de cada sección si molesta.

## Queda a criterio del builder

- Nombres exactos de las funciones de `plots.py` y de los helpers privados.
- Si algún gráfico del notebook conviene partirlo en dos funciones, o dos juntarlos en una.
- El orden de los argumentos nuevos, siempre que `p: LoRaParams` vaya en la posición donde
  hoy están `SF, B`.

## Verificación

1. En un venv nuevo, `pip install -r requirements.txt` instala el paquete y `pytest` pasa.
2. `jupyter nbconvert --to notebook --execute --inplace LoRa.ipynb` corre sin errores.
3. Los gráficos del notebook ejecutado muestran lo mismo que la versión actual (misma forma de
   las curvas y mismos órdenes de magnitud; los valores puntuales pueden variar por la semilla).
4. Ninguna celda de código del notebook define funciones.
