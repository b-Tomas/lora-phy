"""Diagramas orientativos del TP, en SVG, en la carpeta `diagramas/` del repositorio.

Requiere el extra `diagramas` (schemdraw):

    pip install -e .[diagramas]
    python -m lora.diagramas
"""

from pathlib import Path

import matplotlib.pyplot as plt
import schemdraw
from schemdraw import dsp
from schemdraw import elements as elm

OUT = Path(__file__).parent.parent / "diagramas"
FONT = 13
BOX_H = 1.3


def save_drawing(d: schemdraw.Drawing, name: str) -> None:
    d.save(str(OUT / f"{name}.svg"), transparent=False)


def save_figure(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --- 1. Modelo del sistema ------------------------------------------------------------------


def sistema() -> None:
    L = 2.4  # Largo de las flechas entre bloques
    with schemdraw.Drawing(show=False) as d:
        d.config(fontsize=FONT, bgcolor="white")

        a = dsp.Arrow().right().length(L).label("bits", "top")
        enc = dsp.Box(w=3.0, h=BOX_H).at(a.end).anchor("W").label("Codificador")
        a = dsp.Arrow().at(enc.E).right().length(L).label("símbolos", "top")
        mod = dsp.Box(w=3.0, h=BOX_H).at(a.end).anchor("W").label("Conformador\nde onda")
        a = dsp.Arrow().at(mod.E).right().length(L).label("chirps", "top")
        h = dsp.Box(w=1.6, h=BOX_H).at(a.end).anchor("W").label("$h$")
        a = dsp.Arrow().at(h.E).right().length(1.2)
        s = dsp.Sum().at(a.end).anchor("W")
        z = dsp.Arrow().endpoints((s.S[0], s.S[1] - 1.8), s.S)
        elm.Label().at((z.start[0], z.start[1] - 0.45)).label("AWGN")
        a = dsp.Arrow().at(s.E).right().length(L).label("señal\nrecibida", "top")
        dem = dsp.Box(w=3.0, h=BOX_H).at(a.end).anchor("W").label("Demodulador")
        a = dsp.Arrow().at(dem.E).right().length(L).label("símbolos\nestimados", "top")
        dec = dsp.Box(w=3.0, h=BOX_H).at(a.end).anchor("W").label("Decodificador")
        dsp.Arrow().at(dec.E).right().length(L + 0.6).label("bits\nestimados", "top")

        pad = {"padx": 0.4, "pady": 0.45}
        for grupo, nombre in [
            ([enc, mod], "Transmisor"),
            ([h, s], "Canal"),
            ([dem, dec], "Receptor"),
        ]:
            elm.EncircleBox(grupo, **pad).linestyle("--").color("gray").label(
                nombre, "top", fontsize=FONT, color="gray"
            )

    save_drawing(d, "sistema")


# --- 2. Modelo de canal ruidoso -------------------------------------------------------------


def canal() -> None:
    with schemdraw.Drawing(show=False) as d:
        d.config(fontsize=FONT, bgcolor="white")

        a = dsp.Arrow().right().length(3.0).label("$c(nT_s + kT)$", "top")
        h = dsp.Box(w=2.6, h=BOX_H).at(a.end).anchor("W").label("$h[n]$\n(SLIT)")
        a = dsp.Arrow().at(h.E).right().length(2.2).label("$(c * h)$", "top")
        s = dsp.Sum().at(a.end).anchor("W")
        z = dsp.Arrow().endpoints((s.S[0], s.S[1] - 1.8), s.S)
        elm.Label().at((z.start[0], z.start[1] - 0.75)).label(
            "$z$ (AWGN)\n$\\sigma^2 = P_c\\,/\\,\\mathrm{SNR}$"
        )
        dsp.Arrow().at(s.E).right().length(3.0).label("$r(nT_s + kT)$", "top")

    save_drawing(d, "canal")

    # Respuesta al impulso de cada canal, un SVG por canal
    casos = [
        ("canal_h_plano", "Plano\n$h[n] = \\delta[n]$", [1.0, 0.0]),
        (
            "canal_h_selectivo",
            "Selectivo en frecuencia\n$h[n] = \\sqrt{0.8}\\,\\delta[n] + \\sqrt{0.2}\\,\\delta[n-1]$",
            [0.8**0.5, 0.2**0.5],
        ),
    ]
    for nombre, titulo, h in casos:
        fig, ax = plt.subplots(figsize=(3.8, 2.6))
        k = range(len(h))
        ax.stem(k, h, linefmt="k-", markerfmt="ko", basefmt="k-")
        ax.set_title(titulo, fontsize=11)
        ax.set_xticks(list(k))
        ax.set_xlabel("$n$")
        ax.set_ylabel("$h[n]$")
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.15)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        save_figure(fig, nombre)


# --- 3. Detección de trama (diagrama de secuencia) ------------------------------------------
# Común a los tres métodos del notebook: solo cambian el estadístico y su umbral


def deteccion() -> None:
    actores = ["Transmisor", "Canal", "Detector", "Demodulador"]
    x = {a: i * 3.4 for i, a in enumerate(actores)}

    # Pasos de arriba hacia abajo: mensajes entre actores, mensajes a sí mismo y notas
    pasos = [
        (
            "nota",
            "Detector",
            (
                "Estadístico del método en cada posición\n"
                "(muestra en el tiempo, ventana en frecuencia)\n"
                "Umbral de Neyman-Pearson para la $P_{FA}$"
            ),
        ),
        ("msg", "Transmisor", "Canal", "Silencio (5 símbolos)"),
        ("msg", "Canal", "Detector", "Solo ruido"),
        ("nota", "Detector", "Estadístico < umbral\n(salvo falsa alarma)"),
        ("msg", "Transmisor", "Canal", "Preámbulo: 8 up-chirps"),
        ("msg", "Canal", "Detector", "Up-chirps + ruido"),
        (
            "nota",
            "Detector",
            "En general, el primer cruce\ndel umbral llega antes del inicio",
        ),
        (
            "self",
            "Detector",
            (
                "Máximo del estadístico en los 8 símbolos siguientes,\n"
                "repetido hasta que no aumente $\\rightarrow$ inicio $\\hat{k}$"
            ),
        ),
        ("msg", "Transmisor", "Canal", "2.25 down-chirps"),
        ("msg", "Canal", "Detector", "Down-chirps + ruido"),
        ("nota", "Detector", "No los usa"),
        ("msg", "Transmisor", "Canal", "Payload (10 símbolos)"),
        ("msg", "Canal", "Demodulador", "Payload + ruido"),
        (
            "msg",
            "Detector",
            "Demodulador",
            "$\\hat{k}$: el payload empieza en $\\hat{k} + 10.25 \\cdot 2^{SF}$",
        ),
        ("alt",),
    ]

    alto = {"nota": 1.35, "msg": 0.8, "self": 1.3, "alt": 2.7}
    total = sum(alto[p[0]] for p in pasos) + 0.6
    fig, ax = plt.subplots(figsize=(11, total * 0.62))
    ax.set_xlim(-1.6, x["Demodulador"] + 2.6)
    ax.set_ylim(-total - 0.9, 1.6)
    ax.axis("off")
    ax.text(
        (x["Transmisor"] + x["Demodulador"]) / 2,
        1.35,
        "Detección de una trama",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )

    caja = {"boxstyle": "square,pad=0.45", "fc": "white", "ec": "black", "lw": 1.2}
    for a in actores:
        ax.text(x[a], 0.4, a, ha="center", va="center", fontsize=12, bbox=caja)
        ax.plot([x[a], x[a]], [0, -total], color="0.55", lw=1, ls="--", zorder=0)
        ax.text(x[a], -total - 0.45, a, ha="center", va="center", fontsize=12, bbox=caja)

    y = -0.4
    for p in pasos:
        dy = alto[p[0]]
        yc = y - dy / 2
        if p[0] == "msg":
            _, a, b, texto = p
            ax.annotate(
                "",
                xy=(x[b], yc - 0.12),
                xytext=(x[a], yc - 0.12),
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
            )
            ax.text(
                (x[a] + x[b]) / 2,
                yc + 0.02,
                texto,
                ha="center",
                va="bottom",
                fontsize=11,
                bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"},
            )
        elif p[0] == "self":
            _, a, texto = p
            ax.annotate(
                "",
                xy=(x[a], yc - 0.3),
                xytext=(x[a], yc + 0.3),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "black",
                    "lw": 1.2,
                    "connectionstyle": "arc,angleA=0,angleB=0,armA=28,armB=28,rad=6",
                },
            )
            ax.text(
                x[a] + 0.65,
                yc,
                texto,
                ha="left",
                va="center",
                fontsize=10.5,
                bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"},
            )
        elif p[0] == "nota":
            _, a, texto = p
            ax.text(
                x[a],
                yc,
                texto,
                ha="center",
                va="center",
                fontsize=10.5,
                bbox={"boxstyle": "square,pad=0.4", "fc": "0.93", "ec": "0.4", "lw": 0.8},
            )
        elif p[0] == "alt":
            x0, x1 = x["Detector"] - 1.85, x["Demodulador"] + 1.85
            top, mid, bot = y - 0.1, y - dy / 2, y - dy + 0.1
            # Fondo blanco para tapar las líneas de vida dentro del recuadro
            ax.add_patch(
                plt.Rectangle(
                    (x0, bot), x1 - x0, top - bot, fc="white", ec="black", lw=1, zorder=2.5
                )
            )
            ax.plot([x0, x1], [mid, mid], color="black", lw=0.8, ls="--", zorder=3)
            ax.text(x0 + 0.1, top - 0.08, "alt", ha="left", va="top", fontsize=10, weight="bold")
            ax.text(
                (x0 + x1) / 2,
                top - 0.12,
                "[$\\hat{k}$ = inicio real, $5 \\cdot 2^{SF}$]",
                ha="center",
                va="top",
                fontsize=10.5,
            )
            ax.text(
                (x0 + x1) / 2,
                (top + mid) / 2 - 0.22,
                "Trama detectada",
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.text(
                (x0 + x1) / 2,
                mid - 0.12,
                "[sin cruce del umbral, o $\\hat{k}$ corrido]",
                ha="center",
                va="top",
                fontsize=10.5,
            )
            ax.text(
                (x0 + x1) / 2,
                (mid + bot) / 2 - 0.22,
                "Trama perdida (cuenta para el FER)",
                ha="center",
                va="center",
                fontsize=11,
            )
        y -= dy

    save_figure(fig, "deteccion")


# --- 4. Filtro adaptado de la detección de trama ---------------------------------------------
# Método 1 del notebook: correlación con el preámbulo completo (detection.detect_full)


def _trama(ax) -> None:
    """Trama en el tiempo: silencio, 8 up-chirps, 2.25 down-chirps y payload, con la cota del
    desplazamiento desde el inicio estimado hasta el payload."""
    ax.set_xlim(-2.2, 21.5)
    ax.set_ylim(-2.6, 0.8)
    ax.axis("off")

    def bloque(x0, x1, texto, fc):
        ax.add_patch(plt.Rectangle((x0, -0.5), x1 - x0, 1.0, fc=fc, ec="black", lw=1))
        if texto:
            ax.text((x0 + x1) / 2, 0, texto, ha="center", va="center", fontsize=11)

    bloque(-2.0, 0, "silencio", "0.95")
    for n in range(8):  # Up-chirps: frecuencia que sube
        bloque(n, n + 1, "", "white")
        ax.plot([n + 0.15, n + 0.85], [-0.3, 0.3], color="black", lw=1)
    for n, ancho in enumerate([1, 1, 0.25]):  # 2.25 down-chirps: frecuencia que baja
        x0 = 8 + n
        bloque(x0, x0 + ancho, "", "0.88")
        ax.plot([x0 + 0.15 * ancho, x0 + 0.85 * ancho], [0.3, -0.3], color="black", lw=1)
    bloque(10.25, 21.0, "payload (10 símbolos)", "0.95")

    def cota(x0, x1, y, texto, va):
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.2},
        )
        for x in (x0, x1):
            ax.plot([x, x], [y - 0.2, y + 0.2], color="black", lw=1)
        dy = 0.25 if va == "bottom" else -0.25
        ax.text((x0 + x1) / 2, y + dy, texto, ha="center", va=va, fontsize=11)

    def marca(x, texto, fontsize):
        ax.annotate(
            texto,
            xy=(x, -0.55),
            xytext=(x, -2.35),
            ha="center",
            fontsize=fontsize,
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1},
        )

    marca(0, "$\\hat{k}$", 12)
    cota(0, 10.25, -1.25, "$10.25 \\cdot 2^{SF}$ muestras: 8 up-chirps + 2.25 down-chirps", "top")
    marca(10.25, "inicio del payload", 11)


def filtro_adaptado() -> None:
    # El detector, desde la señal recibida hasta el inicio estimado de la trama
    L = 2.6
    d = schemdraw.Drawing(show=False)
    d.config(fontsize=FONT, bgcolor="white")
    a = d.add(dsp.Arrow().right().length(L).label("$r[k]$", "top"))
    mf = d.add(
        dsp.Box(w=3.6, h=1.6)
        .at(a.end)
        .anchor("W")
        .label("Filtro adaptado\nplantilla $p$:\n8 up-chirps")
    )
    a = d.add(dsp.Arrow().at(mf.E).right().length(L).label("$\\langle r_k, p \\rangle$", "top"))
    mod = d.add(dsp.Box(w=1.8, h=BOX_H).at(a.end).anchor("W").label("$|\\cdot|^2$"))
    a = d.add(dsp.Arrow().at(mod.E).right().length(L).label("$T[k]$", "top"))
    cmp_ = d.add(dsp.Box(w=2.4, h=BOX_H).at(a.end).anchor("W").label("$T[k] \\geq \\eta$"))
    eta = d.add(dsp.Arrow().endpoints((cmp_.S[0], cmp_.S[1] - 1.1), cmp_.S))
    d.add(
        elm.Label()
        .at((eta.start[0], eta.start[1] - 0.5))
        .label("$\\eta = -8\\sigma^2 \\ln(P_{FA} / 2^{SF})$")
    )
    a = d.add(dsp.Arrow().at(cmp_.E).right().length(L).label("primer\ncruce", "top"))
    mx = d.add(
        dsp.Box(w=3.6, h=1.6)
        .at(a.end)
        .anchor("W")
        .label("Máximo de $T$ en\nlos $8 \\cdot 2^{SF}$\nsiguientes")
    )
    d.add(dsp.Arrow().at(mx.E).right().length(L).label("$\\hat{k}$", "top"))
    save_drawing(d, "filtro_adaptado")


def busqueda_impulsos() -> None:
    # Método 2: búsqueda de 8 impulsos (detection.detect_peaks)
    L = 2.0
    d = schemdraw.Drawing(show=False)
    d.config(fontsize=FONT, bgcolor="white")
    a = d.add(dsp.Arrow().right().length(L).label("$r[k]$", "top"))
    corr = d.add(
        dsp.Box(w=3.2, h=1.6)
        .at(a.end)
        .anchor("W")
        .label("Correlación\ncon un up-chirp\n$c|_{s=0}$")
    )
    a = d.add(dsp.Arrow().at(corr.E).right().length(L).label("$y[k]$", "top"))
    mod = d.add(dsp.Box(w=1.8, h=BOX_H).at(a.end).anchor("W").label("$|\\cdot|^2$"))
    a = d.add(dsp.Arrow().at(mod.E).right().length(L).label("$|y[k]|^2$", "top"))
    suma = d.add(
        dsp.Box(w=3.8, h=1.6).at(a.end).anchor("W").label("Suma de 8 valores\nseparados $2^{SF}$")
    )
    a = d.add(dsp.Arrow().at(suma.E).right().length(L).label("$S[k]$", "top"))
    cmp_ = d.add(dsp.Box(w=2.4, h=BOX_H).at(a.end).anchor("W").label("$S[k] \\geq \\eta_S$"))
    eta = d.add(dsp.Arrow().endpoints((cmp_.S[0], cmp_.S[1] - 1.1), cmp_.S))
    d.add(
        elm.Label()
        .at((eta.start[0], eta.start[1] - 0.5))
        .label("$P(S[k] > \\eta_S \\mid H_0) = P_{FA} / 2^{SF}$")
    )
    a = d.add(dsp.Arrow().at(cmp_.E).right().length(L).label("primer\ncruce", "top"))
    mx = d.add(
        dsp.Box(w=3.6, h=1.6)
        .at(a.end)
        .anchor("W")
        .label("Máximo de $S$ en\nlos $8 \\cdot 2^{SF}$\nsiguientes")
    )
    d.add(dsp.Arrow().at(mx.E).right().length(L).label("$\\hat{k}$", "top"))
    save_drawing(d, "busqueda_impulsos")


def demodulacion_preambulo() -> None:
    # Método 3: demodulación del preámbulo (detection.detect_freq)
    L = 2.0
    d = schemdraw.Drawing(show=False)
    d.config(fontsize=FONT, bgcolor="white")
    a = d.add(dsp.Arrow().right().length(L).label("$r[k]$", "top"))
    ven = d.add(dsp.Box(w=3.0, h=1.6).at(a.end).anchor("W").label("Ventanas de\n$2^{SF}$ muestras"))
    a = d.add(dsp.Arrow().at(ven.E).right().length(L))
    dem = d.add(dsp.Box(w=3.0, h=1.6).at(a.end).anchor("W").label("Demodulador\n(Ec. 16)"))
    a = d.add(dsp.Arrow().at(dem.E).right().length(L).label("$\\hat{s}[n]$", "top"))
    cnt = d.add(
        dsp.Box(w=3.4, h=1.6)
        .at(a.end)
        .anchor("W")
        .label("Ventanas con\n$\\hat{s}[n] = 0$ entre\n8 consecutivas")
    )
    a = d.add(dsp.Arrow().at(cnt.E).right().length(L).label("$V[m]$", "top"))
    cmp_ = d.add(dsp.Box(w=2.4, h=BOX_H).at(a.end).anchor("W").label("$V[m] \\geq K$"))
    k = d.add(dsp.Arrow().endpoints((cmp_.S[0], cmp_.S[1] - 1.1), cmp_.S))
    d.add(
        elm.Label()
        .at((k.start[0], k.start[1] - 0.5))
        .label("$K$: el menor con $P(V[m] \\geq K \\mid H_0) \\leq P_{FA}$")
    )
    a = d.add(dsp.Arrow().at(cmp_.E).right().length(L).label("primer\ncruce", "top"))
    mx = d.add(
        dsp.Box(w=3.6, h=1.6)
        .at(a.end)
        .anchor("W")
        .label("Máximo de $V$ en\nlas 8 ventanas\nsiguientes")
    )
    d.add(
        dsp.Arrow().at(mx.E).right().length(L + 0.8).label("$\\hat{k} = \\hat{m}\\, 2^{SF}$", "top")
    )
    save_drawing(d, "demodulacion_preambulo")


def inicio_payload() -> None:
    # Desplazamiento desde el inicio estimado hasta el payload, sobre la trama
    fig, ax = plt.subplots(figsize=(13, 2.9))
    _trama(ax)
    fig.tight_layout()
    save_figure(fig, "inicio_payload_trama")


if __name__ == "__main__":
    sistema()
    canal()
    deteccion()
    filtro_adaptado()
    busqueda_impulsos()
    demodulacion_preambulo()
    inicio_payload()
    print("Diagramas generados en", OUT)
