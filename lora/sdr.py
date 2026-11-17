"""Transmisión y recepción de tramas con un ADALM-Pluto (pyadi-iio).

El mismo Pluto transmite y recibe: TX y RX comparten el reloj y el oscilador local, así que la
señal recibida no tiene corrimiento de frecuencia y tiene exactamente el período del TX.

El TX repite un buffer de varias tramas distintas (silencio de largo al azar + preámbulo + payload
cada una), y el RX captura el buffer completo y tres tramas más. `receive` busca las tramas en la
captura sin saber dónde están, como un receptor real. `evaluate` sí conoce el buffer transmitido:
lo ubica en la captura para contar los errores y medir el SNR.
"""

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import params
from .detection import detect_full
from .metrics import power
from .params import LoRaParams
from .tx import modulator, preamble

LO_FREQ = 905e6  # Portadora
TX_ATTEN_DB = -10  # Atenuación del TX al conectar

TX_SCALE = 2**14  # Pico de I y Q del TX: pyadi espera muestras en ±2^14
ADC_FULL_SCALE = 2**11  # El ADC del AD9363 es de 12 bits: muestras en ±2048
AGC_TARGET_DB = -11  # Valor eficaz que busca el AGC slow_attack, en dB respecto del fondo de escala
ADC_NOISE_FLOOR = 1 / 6  # Ruido de cuantización complejo, 2 * Δ²/12 con Δ = 1
TX_ATTEN_RANGE_DB = (-89, -3)  # Con menos de 3 dB de atenuación se puede dañar el receptor

FLUSH = 5  # Buffers del RX descartados después de cambiar el TX. Subir si FER = 1 en loopback
AGC_SETTLE = 15  # Buffers para que el AGC converja. Subir si la ganancia elegida varía
SNR_TOL_DB = 0.5  # Diferencia aceptada entre el SNR medido y el buscado en `measure_at_snr`
SNR_MAX_TRIES = 5  # Mediciones como máximo por SNR buscado
SNR_MAX_STEP_DB = 3.0  # Corrección máxima de la escala entre mediciones
SNR_REF_SCALE_DB = -22.0  # Escala de la medición de referencia de `measure_snr_sweep`
SNR_REF_FRAMES = 20  # Tramas de la medición de referencia
SETTLE_SIZE = 2**13  # Muestras de los buffers descartados: cortos, para no traerlos por la VPN


@contextmanager
def session(uri: str, p: LoRaParams, loopback: int):
    """Abre el Pluto y lo configura para transmitir y recibir tramas. Al salir, aunque haya un
    error, llama a `release`.

    Args:
        uri (str): Dirección del SDR, por ejemplo "ip:192.168.1.32".
        p (LoRaParams): Parámetros del sistema. B es la frecuencia de muestreo y el ancho de banda.
        loopback (int): 0 = por el aire, 1 = loopback digital (no pasa por el ADC ni el DAC).

    Yields:
        adi.Pluto: SDR configurado, con el RX en control automático de ganancia.
    """

    import adi  # Extra opcional `sdr`

    sdr = adi.Pluto(uri)  # pyright: ignore[reportAbstractUsage]
    sdr.loopback = loopback
    sdr.sample_rate = int(p.B)
    sdr.tx_rf_bandwidth = int(p.B)
    sdr.rx_rf_bandwidth = int(p.B)
    sdr.tx_lo = int(LO_FREQ)
    sdr.rx_lo = int(LO_FREQ)
    set_tx_atten(sdr, TX_ATTEN_DB)
    sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.tx_cyclic_buffer = True
    try:
        yield sdr
    finally:
        release(sdr)


def set_tx_atten(sdr, tx_atten_dB: float) -> None:
    """Fija la atenuación del TX, en dB (negativa). La potencia máxima del TX es 10 dBm: con
    -10 dB transmite 0 dBm (1 mW). Rechaza valores fuera de TX_ATTEN_RANGE_DB."""

    low, high = TX_ATTEN_RANGE_DB
    if not low <= tx_atten_dB <= high:
        raise ValueError(f"Atenuación del TX fuera de [{low}, {high}] dB: {tx_atten_dB}")
    sdr.tx_hardwaregain_chan0 = float(tx_atten_dB)  # pyadi rechaza los enteros de numpy


def release(sdr) -> None:
    """Deja el SDR sin interferir: máxima atenuación, TX y RX en frecuencias distintas y TX en
    ceros. El TX del Pluto no se puede apagar."""

    sdr.tx_destroy_buffer()
    set_tx_atten(sdr, TX_ATTEN_RANGE_DB[0])
    sdr.tx_lo = int(2400e6)
    sdr.rx_lo = int(950e6)
    sdr.tx(np.zeros(2048, dtype=complex))


class SimulatedPluto:
    """Pluto ideal, para probar sin hardware: `rx` devuelve el buffer del TX repetido a partir de
    una muestra al azar (canal plano, sin ruido propio ni corrimiento de frecuencia), saturado y cuantizado como en el
    ADC.

    Con la ganancia GAIN_REF_DB el pico del TX llega a la mitad del fondo de escala, como en el
    loopback digital del Pluto. En "slow_attack" la ganancia lleva el valor eficaz a AGC_TARGET_DB, como
    el AGC."""

    rx_buffer_size = 0
    GAIN_REF_DB = 30.0
    rx_hardwaregain_chan0 = GAIN_REF_DB
    gain_control_mode_chan0 = "manual"

    def tx(self, x: np.ndarray) -> None:
        self.x = x

    def rx(self) -> np.ndarray:
        if self.gain_control_mode_chan0 == "slow_attack":
            self.rx_hardwaregain_chan0 = round(
                self.GAIN_REF_DB
                + AGC_TARGET_DB
                - 10 * np.log10(power(self.x) / (2 * TX_SCALE) ** 2)
            )
        r = np.resize(np.roll(self.x, params.rng.integers(self.x.size)), self.rx_buffer_size)
        gain = 10 ** ((self.rx_hardwaregain_chan0 - self.GAIN_REF_DB) / 20)
        r = np.round(r * ADC_FULL_SCALE / (2 * TX_SCALE) * gain)
        lo, hi = -ADC_FULL_SCALE, ADC_FULL_SCALE - 1
        return np.clip(np.real(r), lo, hi) + 1j * np.clip(np.imag(r), lo, hi)

    def tx_destroy_buffer(self) -> None:
        pass

    def rx_destroy_buffer(self) -> None:
        pass


# --- Transmisión y captura -------------------------------------------------------------------


def frames(symbols: np.ndarray, gaps: np.ndarray, p: LoRaParams) -> np.ndarray:
    """Buffer del TX sin ruido: una trama por fila de `symbols`, cada una con su silencio antes
    del preámbulo.

    Args:
        symbols (np.ndarray): Símbolos del payload de cada trama (n_frames, num_symbols).
        gaps (np.ndarray): Muestras de silencio antes de cada trama (n_frames).
        p (LoRaParams): Parámetros del sistema.

    Returns:
        np.ndarray: Muestras del buffer.
    """

    return np.concatenate(
        [np.concatenate([np.zeros(g), preamble(p), modulator(s, p)]) for s, g in zip(symbols, gaps)]
    )


def record(
    sdr, x: np.ndarray, n_samples: int, calibrate: bool = True, scale_dB: float = 0.0
) -> tuple[np.ndarray, float]:
    """Transmite `x` en forma cíclica y captura `n_samples` muestras de la señal nueva.

    Si `calibrate`, deja actuar al control automático de ganancia (AGC) y después fija la ganancia
    que eligió: con el AGC activo, el silencio y la trama se recibirían con ganancias distintas.

    Returns:
        tuple[np.ndarray, float]: Captura y ganancia del RX, en dB.
    """

    sdr.tx_destroy_buffer()
    sdr.rx_destroy_buffer()
    sdr.rx_buffer_size = SETTLE_SIZE
    sdr.tx(x / adc_peak(x) * TX_SCALE * 10 ** (scale_dB / 20))
    if calibrate:
        sdr.gain_control_mode_chan0 = "slow_attack"
        for _ in range(AGC_SETTLE):
            sdr.rx()
        gain = sdr.rx_hardwaregain_chan0
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = gain
    for _ in range(FLUSH):  # Buffers con la señal anterior
        sdr.rx()
    sdr.rx_destroy_buffer()
    sdr.rx_buffer_size = n_samples
    return sdr.rx(), sdr.rx_hardwaregain_chan0


def measure(
    sdr,
    p: LoRaParams,
    n_frames: int,
    num_symbols: int,
    gap_range: tuple[int, int],
    calibrate: bool = True,
    scale_dB: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Transmite un buffer de `n_frames` tramas con payloads y silencios al azar, y lo captura.

    Cada silencio tiene un largo al azar, en muestras, entre gap_range[0] y gap_range[1] símbolos: el receptor
    no puede predecir dónde empieza la trama siguiente. La captura tiene el buffer completo y por lo
    menos tres períodos más (silencio y trama): cada trama aparece entera al menos una vez después
    de los dos primeros (`evaluate`).

    Args:
        sdr (adi.Pluto): SDR configurado.
        p (LoRaParams): Parámetros del sistema.
        n_frames (int): Tramas del buffer.
        num_symbols (int): Símbolos del payload por trama.
        gap_range (tuple[int, int]): Largo mínimo y máximo del silencio antes de cada trama, en símbolos.
        calibrate (bool): Calibrar la ganancia del RX con el AGC (`record`).
        scale_dB (float): Escala digital de las muestras del TX respecto del pico TX_SCALE, en dB.
            Baja la señal sin bajar el ruido del RX.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]: Símbolos transmitidos
        (n_frames, num_symbols), muestras de silencio antes de cada trama, captura y ganancia del
        RX en dB.
    """

    symbols = params.rng.integers(0, p.M, size=(n_frames, num_symbols))
    gaps = params.rng.integers(gap_range[0] * p.M, gap_range[1] * p.M, size=n_frames)
    # Con un buffer de largo impar el TX del Pluto agrega una muestra en cada repetición
    gaps[0] += gaps.sum() % 2  # El preámbulo y el payload tienen largo par
    x = frames(symbols, gaps, p)
    # Largo de la captura: el de n_frames + 3 períodos con el silencio máximo. Es mayor o igual que
    # el buffer más tres períodos, y no depende de los silencios: las capturas se pueden apilar
    longest = gap_range[1] * p.M + p.preamble_len + num_symbols * p.M
    r, gain = record(sdr, x, (n_frames + 3) * longest, calibrate, scale_dB)
    return symbols, gaps, r, gain


def measure_at_snr(
    sdr,
    p: LoRaParams,
    target_dB: float,
    scale_dB: float,
    n_frames: int,
    num_symbols: int,
    gap_range: tuple[int, int],
) -> tuple[tuple, float, float]:
    """`measure` con la escala del TX que lleva el SNR medido a `target_dB`.

    Por el aire, el SNR que da una misma escala varía alrededor de 1 dB entre capturas. Se mide con
    `scale_dB` y, si el SNR medido se aleja más de `SNR_TOL_DB` del buscado, se corrige la escala
    en la diferencia y se mide de nuevo, hasta `SNR_MAX_TRIES` veces. Cada corrección se limita a
    `SNR_MAX_STEP_DB`, para que una captura con interferencia no lleve la escala lejos. Se queda la
    medición más cercana al SNR buscado.

    Args:
        sdr (adi.Pluto): SDR configurado, con el ruido térmico del RX por encima de la señal.
        p (LoRaParams): Parámetros del sistema.
        target_dB (float): SNR buscado, en dB.
        scale_dB (float): Escala inicial, en dB (`measure`).
        n_frames, num_symbols, gap_range: Como en `measure`.

    Returns:
        tuple[tuple, float, float]: Salida de `measure`, escala usada en dB y SNR medido en dB.
    """

    best = None  # (|error|, salida de measure, escala, SNR) de la medición más cercana
    for _ in range(SNR_MAX_TRIES):
        run = measure(sdr, p, n_frames, num_symbols, gap_range, scale_dB=scale_dB)
        snr = capture_snr(run[2], run[0], run[1], p)
        error = target_dB - snr
        if best is None or abs(error) < best[0]:
            best = (abs(error), run, scale_dB, snr)
        if abs(error) <= SNR_TOL_DB:
            break
        # La escala no puede pasar el pico del TX (0 dB)
        scale_dB = min(scale_dB + float(np.clip(error, -SNR_MAX_STEP_DB, SNR_MAX_STEP_DB)), 0.0)
    return best[1], best[2], best[3]


def measure_snr_sweep(
    sdr,
    p: LoRaParams,
    targets_dB: np.ndarray,
    n_frames: int,
    num_symbols: int,
    gap_range: tuple[int, int],
) -> tuple[list, np.ndarray]:
    """Un punto por SNR de `targets_dB`, con `measure_at_snr`.

    La primera escala sale de una medición corta con `SNR_REF_SCALE_DB`, donde el SNR se mide con
    poco error. Con el ruido térmico del RX fijo, el SNR cambia lo mismo que la escala, así que la
    escala para un SNR es el SNR menos la diferencia entre SNR y escala medida. Esa diferencia se
    actualiza con cada punto que llega al SNR buscado, para seguir las variaciones del canal.

    Args:
        sdr (adi.Pluto): SDR configurado.
        p (LoRaParams): Parámetros del sistema.
        targets_dB (np.ndarray): SNR buscado de cada punto, en dB.
        n_frames, num_symbols, gap_range: Como en `measure`.

    Returns:
        tuple[list, np.ndarray]: Salidas de `measure`, una por punto (para `stack`), y escala
        usada en cada punto, en dB.
    """

    symbols, gaps, r, _ = measure(
        sdr, p, SNR_REF_FRAMES, num_symbols, gap_range, scale_dB=SNR_REF_SCALE_DB
    )
    offset_dB = capture_snr(r, symbols, gaps, p) - SNR_REF_SCALE_DB  # SNR medido menos escala
    runs, scales_dB = [], []
    for target_dB in targets_dB:
        run, scale_dB, snr = measure_at_snr(
            sdr, p, target_dB, target_dB - offset_dB, n_frames, num_symbols, gap_range
        )
        print(
            f"SNR buscado {target_dB:5.1f} dB: escala {scale_dB:5.1f} dB, SNR medido {snr:5.1f} dB"
        )
        if abs(snr - target_dB) <= SNR_TOL_DB:  # Una captura que no llegó no corrige la referencia
            offset_dB = snr - scale_dB
        runs.append(run)
        scales_dB.append(scale_dB)
    return runs, np.array(scales_dB)


def adc_peak(x: np.ndarray) -> float:
    """Mayor valor absoluto de I y Q."""

    return float(np.abs(np.r_[x.real, x.imag]).max())


def stack(runs: list, **meta) -> tuple[np.ndarray, dict]:
    """Junta las salidas de `measure` de un barrido, una por punto, en capturas y metadatos."""

    symbols, gaps, captures, gains = map(np.array, zip(*runs))
    return captures, {"symbols": symbols, "gaps": gaps, "gain": gains, **meta}


# --- Recepción y medición (sin SDR) ----------------------------------------------------------


def receive(
    r: np.ndarray, p: LoRaParams, noise_power: float, P_FA: float, num_symbols: int
) -> np.ndarray:
    """Receptor: busca las tramas en una captura sin saber dónde están.

    Recorre la captura con una ventana de dos tramas de largo y busca un preámbulo en ella con
    `detect_full`. Si lo encuentra, la ventana siguiente empieza donde termina la trama. Si no, la ventana avanza una trama: una trama que empieza en la primera
    mitad de la ventana entra entera en ella, y la que empieza en la segunda mitad, en la ventana
    siguiente.

    La primera ventana empieza en el símbolo de menor energía de las dos primeras tramas de largo,
    que con SNR alto cae en un silencio: si empezara a mitad de un payload, podría confundir sus
    chirps con el preámbulo.

    Args:
        r (np.ndarray): Captura.
        p (LoRaParams): Parámetros del sistema.
        noise_power (float): Potencia del ruido por muestra (sigma^2).
        P_FA (float): Probabilidad de falsa alarma por símbolo.
        num_symbols (int): Símbolos del payload por trama.

    Returns:
        np.ndarray: Muestra de inicio de cada trama detectada.
    """

    frame_len = p.preamble_len + num_symbols * p.M
    energy = np.convolve(np.abs(r[: 2 * frame_len]) ** 2, np.ones(p.M), "valid")
    pos = int(np.argmin(energy))
    starts = []
    while r.size - pos >= frame_len:
        start = detect_full(r[pos : pos + 2 * frame_len], p, noise_power, P_FA)
        if start is None:
            pos += frame_len
            continue
        pos += start + frame_len
        if pos > r.size:
            break  # Trama cortada al final de la captura
        starts.append(pos - frame_len)
    return np.array(starts, dtype=int)


def _delay(r: np.ndarray, x: np.ndarray) -> int:
    """Retardo de la captura `r` respecto del buffer `x`: la muestra i del buffer está en la muestra
    (i + delay) mod L de la captura. Es el máximo de la correlación circular."""

    return int(np.argmax(np.abs(np.fft.ifft(np.fft.fft(r[: x.size]) * np.conj(np.fft.fft(x))))))


def capture_snr(r: np.ndarray, symbols: np.ndarray, gaps: np.ndarray, p: LoRaParams) -> float:
    """SNR de una captura de `measure`, como en `evaluate`: alinea el buffer transmitido con la
    captura y usa `snr_dB`."""

    x = frames(symbols, gaps, p)
    return snr_dB(np.roll(r[: x.size], -_delay(r, x)), x)


def snr_dB(r: np.ndarray, x: np.ndarray) -> float:
    """SNR de una captura alineada con el buffer transmitido sin ruido `x`, respecto de la
    potencia de la trama (como en la simulación).

    Estimador asistido por datos: se estima por mínimos cuadrados la amplitud compleja de la señal
    recibida (ganancia y fase del canal), y lo que queda al restarla es el ruido, incluyendo la
    distorsión del equipo.
    """

    a = np.vdot(x, r) / np.vdot(x, x)
    noise = max(power(r - a * x), ADC_NOISE_FLOOR)
    return float(10 * np.log10(np.abs(a) ** 2 * power(x[x != 0]) / noise))


class Evaluation(NamedTuple):
    """Resultado de `evaluate` para una captura."""

    errors: np.ndarray  # Trama transmitida perdida: inicio no detectado en la muestra exacta
    timing: np.ndarray  # Inicio detectado menos inicio real, en muestras, de las detectadas
    snr: float  # SNR medido, en dB


def evaluate(
    r: np.ndarray,
    symbols: np.ndarray,
    gaps: np.ndarray,
    p: LoRaParams,
    P_FA: float,
) -> Evaluation:
    """Pasa una captura de `measure` por el receptor y la compara con lo transmitido.

    El receptor (`receive`) no sabe dónde están las tramas. La medición sí: ubica el buffer
    transmitido en la captura con el máximo de la correlación circular, y de ahí sale el inicio
    real de cada trama. La potencia del ruido del receptor se estima en el silencio antes de cada
    trama, donde solo hay ruido.

    Args:
        r (np.ndarray): Captura de `measure`.
        symbols (np.ndarray): Símbolos transmitidos (n_frames, num_symbols).
        gaps (np.ndarray): Muestras de silencio antes de cada trama (n_frames).
        p (LoRaParams): Parámetros del sistema.
        P_FA (float): Probabilidad de falsa alarma por símbolo.

    Returns:
        Evaluation: Errores, error del inicio detectado, SNR e inicios detectados.
    """

    x = frames(symbols, gaps, p)
    L, num_symbols = x.size, symbols.shape[1]
    frame_len = p.preamble_len + num_symbols * p.M
    longest = gaps.max() + frame_len

    # Inicio de cada trama en el buffer: los silencios hasta ella inclusive y las tramas anteriores
    in_buffer = np.cumsum(gaps) + np.arange(len(gaps)) * frame_len
    # Retardo de la captura: la muestra i del buffer está en la muestra (i + delay) mod L
    delay = _delay(r, x)
    # Inicio real de cada trama, una vez cada una, después de los dos primeros períodos más
    # largos: el receptor se sincroniza en ellos y puede perder alguna trama
    real = 2 * longest + (in_buffer + delay - 2 * longest) % L

    noise = power(np.concatenate([r[t - g : t] for t, g in zip(real, gaps)]))
    starts = receive(r, p, max(noise, ADC_NOISE_FLOOR), P_FA, num_symbols)
    snr = snr_dB(np.roll(r[:L], -delay), x)
    if starts.size == 0:
        return Evaluation(np.ones(len(symbols), dtype=bool), starts, snr)

    # Cada trama transmitida se compara con la detección más cercana a su inicio real
    nearest = np.abs(starts[:, None] - real).argmin(axis=0)
    timing = starts[nearest] - real
    found = np.abs(timing) < frame_len // 2
    return Evaluation(timing != 0, timing[found], snr)


def save(path: str | Path, captures: np.ndarray, **meta) -> None:
    """Guarda capturas y metadatos en un .npz. Las muestras del ADC son enteras: van en int16."""

    iq = np.stack([np.real(captures), np.imag(captures)], axis=-1).round().astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, iq=iq, **meta)


def load(path: str | Path) -> tuple[np.ndarray, dict]:
    """Carga un .npz de `save`: capturas complejas y metadatos."""

    with np.load(path) as npz:
        data = dict(npz)
    iq = data.pop("iq").astype(float)
    return iq[..., 0] + 1j * iq[..., 1], data


def evaluate_all(captures: np.ndarray, meta: dict, p: LoRaParams, P_FA: float) -> list[Evaluation]:
    """`evaluate` para cada captura de `stack`."""

    return [evaluate(r, s, g, p, P_FA) for r, s, g in zip(captures, meta["symbols"], meta["gaps"])]


def acquire(
    path: str, run: Callable, p: LoRaParams, uri: str | None, loopback: int = 0
) -> tuple[np.ndarray, dict]:
    """Capturas de un paso del experimento y sus metadatos.

    Con `uri`, mide con el SDR (`run(pluto)` devuelve capturas y metadatos) y las guarda en `path`.
    Sin `uri`, carga las guardadas o, si no hay, mide con `SimulatedPluto` sin guardar.
    """

    if uri is not None:
        with session(uri, p, loopback) as pluto:
            captures, meta = run(pluto)
        save(path, captures, **meta)
    elif not Path(path).exists():
        print(f"Sin {path}: Pluto simulado (canal plano, sin ruido ni distorsión del equipo)")
        return run(SimulatedPluto())
    return load(path)
