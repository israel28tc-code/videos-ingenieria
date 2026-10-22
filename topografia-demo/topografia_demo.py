"""Video vertical (1080x1920) para Reels / TikTok / Shorts:
terreno 3D girando + plano de curvas de nivel con eje vial + perfil longitudinal.

Uso:  python3 topografia_demo.py   ->  topografia_demo.mp4 (en esta carpeta)
"""
import os
import subprocess
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Parámetros del video ---
FPS = 30
DURACION_S = 12
N_FRAMES = FPS * DURACION_S
CARPETA_FRAMES = "salida/frames"
VIDEO_SALIDA = "topografia_demo.mp4"
FIRMA = "@israeltapia70"

FONDO = "#0f1720"
TEXTO = "#e8eef4"
ACENTO = "#ffb000"


# --- Terreno sintético (reemplazar por puntos reales de estación total / GPS) ---
def cota(x, y):
    """Cota en metros para coordenadas locales x, y (0-500 m)."""
    z = 2780.0
    z += 60 * np.exp(-((x - 330) ** 2 + (y - 320) ** 2) / (2 * 90 ** 2))
    z += 35 * np.exp(-((x - 120) ** 2 + (y - 380) ** 2) / (2 * 70 ** 2))
    z -= 25 * np.exp(-((x - 220) ** 2 + (y - 120) ** 2) / (2 * 110 ** 2))
    z += 0.04 * x + 0.03 * y
    return z


xs = np.linspace(0, 500, 80)
X, Y = np.meshgrid(xs, xs)
Z = cota(X, Y)

# Eje vial: curva paramétrica de inicio a fin
t = np.linspace(0, 1, 400)
eje_x = 40 + 420 * t
eje_y = 60 + 380 * t + 70 * np.sin(2.2 * np.pi * t)
eje_z = cota(eje_x, eje_y)
abscisa = np.concatenate([[0], np.cumsum(np.hypot(np.diff(eje_x), np.diff(eje_y)))])


def dibujar_frame(i):
    avance = i / (N_FRAMES - 1)  # 0 -> 1
    k = max(2, int(avance * len(t)))  # puntos del eje ya dibujados

    fig = plt.figure(figsize=(10.8, 19.2), dpi=100, facecolor=FONDO)

    fig.text(0.5, 0.955, "LEVANTAMIENTO TOPOGRÁFICO", ha="center", color=TEXTO,
             fontsize=34, fontweight="bold")
    fig.text(0.5, 0.93, "Del terreno al plano y al perfil del eje vial", ha="center",
             color=ACENTO, fontsize=22)

    # 1) Terreno 3D girando
    ax3 = fig.add_axes([0.0, 0.58, 1.0, 0.34], projection="3d", facecolor=FONDO)
    ax3.plot_surface(X, Y, Z, cmap="gist_earth", linewidth=0, antialiased=True,
                     rstride=2, cstride=2, alpha=0.95)
    ax3.contour(X, Y, Z, levels=15, zdir="z", offset=Z.min() - 20, cmap="gist_earth",
                linewidths=1)
    ax3.plot(eje_x[:k], eje_y[:k], eje_z[:k] + 2, color=ACENTO, lw=3)
    ax3.view_init(elev=32, azim=-60 + 120 * avance)
    ax3.set_zlim(Z.min() - 20, Z.max() + 5)
    ax3.set_axis_off()

    # 2) Plano de curvas de nivel
    ax2 = fig.add_axes([0.1, 0.27, 0.8, 0.29], facecolor=FONDO)
    cs = ax2.contour(X, Y, Z, levels=np.arange(2760, 2900, 5), colors="#7fa3b8",
                     linewidths=0.6)
    ax2.contour(X, Y, Z, levels=np.arange(2760, 2900, 25), colors=TEXTO, linewidths=1.3)
    ax2.clabel(cs, levels=cs.levels[::5], fontsize=10, colors=TEXTO, fmt="%.0f")
    ax2.plot(eje_x[:k], eje_y[:k], color=ACENTO, lw=3)
    ax2.plot(eje_x[k - 1], eje_y[k - 1], "o", color=ACENTO, ms=12)
    ax2.set_aspect("equal")
    ax2.set_title("PLANO · curvas de nivel cada 5 m", color=TEXTO, fontsize=18, pad=10)
    ax2.tick_params(colors="#8899aa", labelsize=11)
    for s in ax2.spines.values():
        s.set_color("#334455")

    # 3) Perfil longitudinal
    ax1 = fig.add_axes([0.12, 0.08, 0.78, 0.14], facecolor=FONDO)
    ax1.fill_between(abscisa[:k], eje_z.min() - 10, eje_z[:k], color="#2b4a3a")
    ax1.plot(abscisa[:k], eje_z[:k], color=ACENTO, lw=2.5)
    ax1.set_xlim(0, abscisa[-1])
    ax1.set_ylim(eje_z.min() - 10, eje_z.max() + 10)
    ax1.set_title("PERFIL LONGITUDINAL DEL EJE", color=TEXTO, fontsize=18, pad=10)
    ax1.set_xlabel("Abscisa (m)", color="#8899aa", fontsize=13)
    ax1.set_ylabel("Cota (m)", color="#8899aa", fontsize=13)
    ax1.tick_params(colors="#8899aa", labelsize=11)
    for s in ax1.spines.values():
        s.set_color("#334455")
    ax1.text(abscisa[k - 1], eje_z[k - 1] + 3,
             f"K0+{abscisa[k - 1]:03.0f}\n{eje_z[k - 1]:.1f} m",
             color=TEXTO, fontsize=13, ha="right" if avance > 0.7 else "left")

    fig.text(0.5, 0.02, FIRMA, ha="center", color="#8899aa", fontsize=18)

    fig.savefig(f"{CARPETA_FRAMES}/f{i:04d}.png", facecolor=FONDO)
    plt.close(fig)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(CARPETA_FRAMES, exist_ok=True)
    with Pool() as pool:
        pool.map(dibujar_frame, range(N_FRAMES))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", f"{CARPETA_FRAMES}/f%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", VIDEO_SALIDA,
    ], check=True)
    print("Listo:", VIDEO_SALIDA)
