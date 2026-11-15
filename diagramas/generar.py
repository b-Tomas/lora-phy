"""Genera los diagramas orientativos del TP en SVG, en esta misma carpeta.

Requiere schemdraw y matplotlib:

    pip install schemdraw matplotlib
    python diagramas/generar.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import schemdraw  # noqa: E402
from schemdraw import dsp  # noqa: E402
from schemdraw import elements as elm  # noqa: E402

OUT = Path(__file__).parent
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

        pad = dict(padx=0.4, pady=0.45)
        for grupo, nombre in [([enc, mod], "Transmisor"), ([h, s], "Canal"), ([dem, dec], "Receptor")]:
            elm.EncircleBox(grupo, **pad).linestyle("--").color("gray").label(
                nombre, "top", fontsize=FONT, color="gray"
            )

    save_drawing(d, "sistema")


# --- 2. Modelo de canal ruidoso -------------------------------------------------------------


def canal() -> None:
    with schemdraw.Drawing(show=False) as d:
        d.config(fontsize=FONT, bgcolor="white")

        a = dsp.Arrow().right().length(3.0).label("$c(nT_s + kT)$", "top")
        h = dsp.Box(w=2.6, h=BOX_H).at(a.end).anchor("W").label("$h[k]$\n(SLIT)")
        a = dsp.Arrow().at(h.E).right().length(2.2).label("$(c * h)$", "top")
        s = dsp.Sum().at(a.end).anchor("W")
        z = dsp.Arrow().endpoints((s.S[0], s.S[1] - 1.8), s.S)
        elm.Label().at((z.start[0], z.start[1] - 0.75)).label(
            "$z$ (AWGN)\n$\\sigma^2 = P_c\\,/\\,\\mathrm{SNR}$"
        )
        dsp.Arrow().at(s.E).right().length(3.0).label("$r(nT_s + kT)$", "top")

    save_drawing(d, "canal")

    # Respuestas al impulso de los dos canales
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4), sharey=True)
    casos = [
        ("Plano\n$h[k] = \\delta[k]$", [1.0, 0.0]),
        ("Selectivo en frecuencia\n$h[k] = \\sqrt{0.8}\\,\\delta[k] + \\sqrt{0.2}\\,\\delta[k-1]$",
         [0.8**0.5, 0.2**0.5]),
    ]
    for ax, (titulo, h) in zip(axes, casos):
        k = range(len(h))
        ax.stem(k, h, linefmt="k-", markerfmt="ko", basefmt="k-")
        ax.set_title(titulo, fontsize=11)
        ax.set_xticks(list(k))
        ax.set_xlabel("$k$")
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.15)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("$h[k]$")
    fig.tight_layout()
    save_figure(fig, "canal_h")


# --- 3. Detección de trama (diagrama de secuencia) ------------------------------------------
# Método 1 del notebook: correlación con el preámbulo completo (detection.detect_full)


def deteccion() -> None:
    actores = ["Transmisor", "Canal", "Detector", "Demodulador"]
    x = {a: i * 3.4 for i, a in enumerate(actores)}

    # Pasos de arriba hacia abajo: mensajes entre actores, mensajes a sí mismo y notas
    pasos = [
        ("nota", "Detector",
         "Plantilla $p$: los 8 up-chirps del preámbulo\n"
         "Para cada muestra $k$: $T[k] = |\\langle r_k, p \\rangle|^2$\n"
         "Umbral: $\\eta = -8\\sigma^2 \\ln(P_{FA} / 2^{SF})$"),
        ("msg", "Transmisor", "Canal", "Silencio (5 símbolos)"),
        ("msg", "Canal", "Detector", "Solo ruido"),
        ("nota", "Detector", "$T[k] < \\eta$\n(salvo falsa alarma)"),
        ("msg", "Transmisor", "Canal", "Preámbulo: 8 up-chirps"),
        ("msg", "Canal", "Detector", "Up-chirps + ruido"),
        ("nota", "Detector",
         "$T[k]$ tiene picos cada $2^{SF}/2$:\n"
         "en general, el primer cruce $T[k] \\geq \\eta$\nllega antes del inicio"),
        ("self", "Detector",
         "Máximo de $T$ en los 8 símbolos siguientes,\n"
         "repetido hasta que no aumente $\\rightarrow$ inicio $\\hat{k}$"),
        ("msg", "Transmisor", "Canal", "2.25 down-chirps"),
        ("msg", "Canal", "Detector", "Down-chirps + ruido"),
        ("nota", "Detector", "No los usa"),
        ("msg", "Transmisor", "Canal", "Payload (10 símbolos)"),
        ("msg", "Canal", "Demodulador", "Payload + ruido"),
        ("msg", "Detector", "Demodulador", "$\\hat{k}$: el payload empieza en $\\hat{k} + 10.25 \\cdot 2^{SF}$"),
        ("alt",),
    ]

    alto = {"nota": 1.35, "msg": 0.8, "self": 1.3, "alt": 2.7}
    total = sum(alto[p[0]] for p in pasos) + 0.6
    fig, ax = plt.subplots(figsize=(11, total * 0.62))
    ax.set_xlim(-1.6, x["Demodulador"] + 1.8)
    ax.set_ylim(-total - 0.9, 1.6)
    ax.axis("off")
    ax.text((x["Transmisor"] + x["Demodulador"]) / 2, 1.35,
            "Detección por correlación con el preámbulo completo",
            ha="center", va="center", fontsize=13, weight="bold")

    caja = dict(boxstyle="square,pad=0.45", fc="white", ec="black", lw=1.2)
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
            ax.annotate("", xy=(x[b], yc - 0.12), xytext=(x[a], yc - 0.12),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
            ax.text((x[a] + x[b]) / 2, yc + 0.02, texto, ha="center", va="bottom", fontsize=11,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        elif p[0] == "self":
            _, a, texto = p
            ax.annotate("", xy=(x[a], yc - 0.3), xytext=(x[a], yc + 0.3),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2,
                                        connectionstyle="arc,angleA=0,angleB=0,armA=28,armB=28,rad=6"))
            ax.text(x[a] + 0.65, yc, texto, ha="left", va="center", fontsize=10.5,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        elif p[0] == "nota":
            _, a, texto = p
            ax.text(x[a], yc, texto, ha="center", va="center", fontsize=10.5,
                    bbox=dict(boxstyle="square,pad=0.4", fc="0.93", ec="0.4", lw=0.8))
        elif p[0] == "alt":
            x0, x1 = x["Detector"] - 1.6, x["Demodulador"] + 1.6
            top, mid, bot = y - 0.1, y - dy / 2, y - dy + 0.1
            # Fondo blanco para tapar las líneas de vida dentro del recuadro
            ax.add_patch(plt.Rectangle((x0, bot), x1 - x0, top - bot, fc="white", ec="black",
                                       lw=1, zorder=2.5))
            ax.plot([x0, x1], [mid, mid], color="black", lw=0.8, ls="--", zorder=3)
            ax.text(x0 + 0.1, top - 0.08, "alt", ha="left", va="top", fontsize=10, weight="bold")
            ax.text((x0 + x1) / 2, top - 0.12, "[$\\hat{k}$ = inicio real, $5 \\cdot 2^{SF}$]",
                    ha="center", va="top", fontsize=10.5)
            ax.text((x0 + x1) / 2, (top + mid) / 2 - 0.22, "Trama detectada",
                    ha="center", va="center", fontsize=11)
            ax.text((x0 + x1) / 2, mid - 0.12,
                    "[sin cruce del umbral, o $\\hat{k}$ corrido por un pico lateral]",
                    ha="center", va="top", fontsize=10.5)
            ax.text((x0 + x1) / 2, (mid + bot) / 2 - 0.22, "Trama perdida (cuenta para el FER)",
                    ha="center", va="center", fontsize=11)
        y -= dy

    save_figure(fig, "deteccion")


if __name__ == "__main__":
    sistema()
    canal()
    deteccion()
    print("Diagramas generados en", OUT)
