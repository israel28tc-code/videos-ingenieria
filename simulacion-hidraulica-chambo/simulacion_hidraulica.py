"""Simulación hidráulica del río Chambo (Chimborazo, Ecuador) con datos reales.

Modelo: flujo uniforme 1D por secciones transversales (ecuación de Manning)
sobre el terreno Copernicus DEM 30 m, con el hidrograma diario de GEOGLOWS v2.
Salida (en esta carpeta): video vertical 1080x1920 (Reels / TikTok / Shorts),
portada y tabla de secciones.

Antes:  python3 datos_reales.py
Uso:    python3 simulacion_hidraulica.py
"""
import os
import subprocess
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource
from matplotlib.ticker import MaxNLocator
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.graph import route_through_array

# --- Datos de entrada ---
DEM = "datos/Copernicus_DSM_COG_10_S02_00_W079_00_DEM.tif"
CAUDALES = "datos/chambo_caudal_diario.csv"
PERIODOS = "datos/chambo_periodos_retorno.csv"
LIMITES = (-78.62, -1.68, -78.52, -1.57)   # lon_min, lat_min, lon_max, lat_max
INICIO_TRAMO = (-78.578, -1.649)           # lon, lat aguas arriba
FIN_TRAMO = (-78.5487, -1.5918)            # lon, lat aguas abajo
EVENTO = None      # fecha "AAAA-MM-DD" o None = mayor creciente desde ANIO_MIN
ANIO_MIN = 1980    # antes de 1980 la reanálisis ERA5 es menos confiable

# --- Parámetros hidráulicos ---
N_MANNING = 0.045  # cauce de montaña con gravas + llanura con cultivos
SEPARACION = 90.0  # m entre secciones transversales
SEMIANCHO = 700.0  # m a cada lado del eje
PASO = 10.0        # m entre puntos de cada sección
CAUDALES_TABLA = np.geomspace(10, 1500, 60)

# --- Video ---
FPS = 30
CARPETA_FRAMES = "salida/frames"
VIDEO_SALIDA = "simulacion_rio_chambo.mp4"
PORTADA = "portada_chambo.jpg"
TABLA = "secciones_pico_chambo.csv"
FIRMA = "@israeltapia70"

FONDO = "#0f1720"
TEXTO = "#e8eef4"
TENUE = "#8899aa"
ACENTO = "#ffb000"
AGUA = np.array([0.12, 0.45, 0.90])
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

G = {}  # datos preparados, compartidos con los procesos que dibujan los frames


def fecha_larga(f):
    return f"{f.day} de {MESES[f.month - 1]} de {f.year}"


# ---------------------------------------------------------------- terreno y eje
def leer_terreno():
    with rasterio.open(DEM) as src:
        ventana = from_bounds(*LIMITES, src.transform)
        z = src.read(1, window=ventana).astype(float)
        transform = src.window_transform(ventana)
    lat0 = (LIMITES[1] + LIMITES[3]) / 2
    dx = transform.a * 111320 * np.cos(np.radians(lat0))
    dy = -transform.e * 110574
    return z, transform, dx, dy


def eje_del_rio(z, transform):
    """Camino de menor cota entre inicio y fin del tramo (sigue el fondo del valle)."""
    inv = ~transform

    def celda(lon, lat):
        c, r = inv * (lon, lat)
        return int(r), int(c)

    costo = (z - z.min()) ** 2 + 1
    camino, _ = route_through_array(costo, celda(*INICIO_TRAMO), celda(*FIN_TRAMO),
                                    fully_connected=True, geometric=True)
    return np.array(camino, dtype=float)


def suavizar_y_remuestrear(x, y, separacion, ventana=9):
    k = np.ones(ventana) / ventana
    xs = np.convolve(np.pad(x, ventana // 2, mode="edge"), k, "valid")
    ys = np.convolve(np.pad(y, ventana // 2, mode="edge"), k, "valid")
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    abscisas = np.arange(0, d[-1], separacion)
    return abscisas, np.interp(abscisas, d, xs), np.interp(abscisas, d, ys)


# ---------------------------------------------------------------- hidráulica
def region_mojada(zs, h, ic):
    """Índices extremos del tramo mojado conectado al punto más bajo del cauce."""
    i0 = i1 = ic
    while i0 > 0 and zs[i0 - 1] < h:
        i0 -= 1
    while i1 < len(zs) - 1 and zs[i1 + 1] < h:
        i1 += 1
    return i0, i1


def caudal_manning(zs, h, ic, pendiente):
    i0, i1 = region_mojada(zs, h, ic)
    tramo = zs[i0:i1 + 1]
    area = np.sum(h - tramo) * PASO
    perimetro = np.sum(np.hypot(PASO, np.diff(tramo))) + PASO
    radio = area / perimetro
    return area * radio ** (2 / 3) * np.sqrt(pendiente) / N_MANNING


def nivel_para_caudal(zs, ic, q, pendiente):
    """Nivel de agua (m s.n.m.) con el que la sección conduce el caudal q (bisección)."""
    bajo, alto = zs[ic], zs[ic] + 40
    for _ in range(40):
        h = (bajo + alto) / 2
        if caudal_manning(zs, h, ic, pendiente) < q:
            bajo = h
        else:
            alto = h
    return alto


def preparar():
    z, transform, dx, dy = leer_terreno()
    nf, nc = z.shape

    # Eje del río en metros (x hacia el este, y hacia el norte)
    camino = eje_del_rio(z, transform)
    abscisas, ex, ey = suavizar_y_remuestrear(camino[:, 1] * dx, (nf - 1 - camino[:, 0]) * dy,
                                              SEPARACION)
    tx, ty = np.gradient(ex), np.gradient(ey)
    norma = np.hypot(tx, ty)
    tx, ty = tx / norma, ty / norma
    nx, ny = -ty, tx  # normal: hacia la margen izquierda mirando aguas abajo

    # Secciones transversales muestreadas del terreno
    s = np.arange(-SEMIANCHO, SEMIANCHO + PASO, PASO)
    px = ex[:, None] + s[None, :] * nx[:, None]
    py = ey[:, None] + s[None, :] * ny[:, None]
    secciones = map_coordinates(z, [nf - 1 - py / dy, px / dx], order=1, mode="nearest")
    centro = len(s) // 2
    rango = int(90 / PASO)
    ic = centro - rango + np.argmin(secciones[:, centro - rango:centro + rango + 1], axis=1)
    fondo = secciones[np.arange(len(ic)), ic]
    pendiente = -np.polyfit(abscisas, fondo, 1)[0]

    # Tabla nivel de agua vs caudal en cada sección, suavizada a lo largo del río
    niveles = np.array([[nivel_para_caudal(secciones[k], ic[k], q, pendiente)
                         for q in CAUDALES_TABLA] for k in range(len(abscisas))])
    k5 = np.ones(5) / 5
    niveles = np.apply_along_axis(
        lambda c: np.convolve(np.pad(c, 2, mode="edge"), k5, "valid"), 0, niveles)
    niveles = np.maximum(niveles, fondo[:, None] + 0.05)
    lim_izq = np.zeros_like(niveles)
    lim_der = np.zeros_like(niveles)
    for k in range(len(abscisas)):
        for j in range(len(CAUDALES_TABLA)):
            i0, i1 = region_mojada(secciones[k], niveles[k, j], ic[k])
            lim_izq[k, j], lim_der[k, j] = s[i0] - PASO / 2, s[i1] + PASO / 2

    # Recorte del terreno alrededor del tramo y asignación de cada celda a su sección
    margen = 1200
    c0 = max(0, int((ex.min() - margen) / dx)); c1 = min(nc, int((ex.max() + margen) / dx))
    f0 = max(0, int(nf - 1 - (ey.max() + margen) / dy)); f1 = min(nf, int(nf - 1 - (ey.min() - margen) / dy))
    zr = z[f0:f1, c0:c1]
    cx = np.arange(c0, c1) * dx
    cy = (nf - 1 - np.arange(f0, f1)) * dy
    X, Y = np.meshgrid(cx, cy)
    _, cercana = cKDTree(np.c_[ex, ey]).query(np.c_[X.ravel(), Y.ravel()])
    cercana = cercana.reshape(X.shape)
    desplaz = (X - ex[cercana]) * nx[cercana] + (Y - ey[cercana]) * ny[cercana]
    along = (X - ex[cercana]) * tx[cercana] + (Y - ey[cercana]) * ty[cercana]
    valida = (np.abs(desplaz) <= SEMIANCHO) & (np.abs(along) <= SEPARACION * 0.75)

    # Hidrograma del evento
    serie = pd.read_csv(CAUDALES, parse_dates=["fecha"])
    if EVENTO:
        pico = pd.Timestamp(EVENTO)
    else:
        recientes = serie[serie.fecha.dt.year >= ANIO_MIN]
        pico = recientes.loc[recientes.caudal_m3s.idxmax(), "fecha"]
    evento = serie[(serie.fecha >= pico - pd.Timedelta(days=12)) &
                   (serie.fecha <= pico + pd.Timedelta(days=8))]
    dias_evento = ((evento.fecha - pico).dt.days).to_numpy(float)

    # Ajuste de Gumbel a los caudales por periodo de retorno (para estimar Tr de cada caudal)
    retorno = pd.read_csv(PERIODOS)
    y_t = -np.log(-np.log(1 - 1 / retorno.periodo_anios))
    alfa, u = np.polyfit(y_t, retorno.caudal_m3s, 1)

    # Sección representativa: la más ancha al pico, en la parte central del tramo
    j_pico = np.interp(evento.caudal_m3s.max(), CAUDALES_TABLA, np.arange(len(CAUDALES_TABLA)))
    ancho_pico = (lim_der - lim_izq)[:, int(round(j_pico))]
    medio = slice(len(abscisas) // 5, 4 * len(abscisas) // 5)
    k_rep = medio.start + int(np.argmax(ancho_pico[medio]))

    # Colores del terreno con sombreado (vista 3D a la mitad de resolución)
    paso3d = 2
    z3 = zr[::paso3d, ::paso3d]
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(z3, cmap=plt.get_cmap("gist_earth"), vert_exag=2, blend_mode="soft",
                   vmin=zr.min() - 150, vmax=zr.max() + 50)[..., :3]

    G.update(dict(
        dx=dx, dy=dy, zr=zr, X=X, Y=Y, cercana=cercana, desplaz=desplaz, valida=valida,
        abscisas=abscisas, ex=ex, ey=ey, nx=nx, ny=ny, s=s, secciones=secciones, ic=ic,
        fondo=fondo, pendiente=pendiente, niveles=niveles, lim_izq=lim_izq, lim_der=lim_der,
        evento=evento, dias_evento=dias_evento, pico=pico, alfa=alfa, u=u, retorno=retorno,
        k_rep=k_rep, paso3d=paso3d, z3=z3, rgb=rgb,
    ))


# ---------------------------------------------------------------- estado en cada instante
def estado(dia):
    q = float(np.interp(dia, G["dias_evento"], G["evento"].caudal_m3s))
    j = np.interp(q, CAUDALES_TABLA, np.arange(len(CAUDALES_TABLA)))
    j0 = int(np.floor(j)); j1 = min(j0 + 1, len(CAUDALES_TABLA) - 1); w = j - j0

    def mezclar(a):
        return a[:, j0] * (1 - w) + a[:, j1] * w

    nivel, izq, der = mezclar(G["niveles"]), mezclar(G["lim_izq"]), mezclar(G["lim_der"])
    c = G["cercana"]
    nivel_celda = nivel[c]
    mojada = (G["valida"] & (G["desplaz"] >= izq[c]) & (G["desplaz"] <= der[c])
              & (G["zr"] < nivel_celda))
    calado = np.where(mojada, nivel_celda - G["zr"], 0.0)
    area_ha = mojada.sum() * G["dx"] * G["dy"] / 1e4
    p = np.exp(-np.exp(-(q - G["u"]) / G["alfa"]))
    tr = 1 / (1 - p) if p < 1 else np.inf
    return q, nivel, calado, area_ha, tr


def texto_tr(tr):
    if tr < 2:
        return "< 2 años"
    if tr > 100:
        return "> 100 años"
    return f"{tr:.0f} años"


def estilo_ejes(ax):
    ax.set_facecolor(FONDO)
    ax.tick_params(colors=TENUE, labelsize=12)
    for borde in ax.spines.values():
        borde.set_color("#334455")


# ---------------------------------------------------------------- dibujo
def dibujar(i, dia, azimut, ruta):
    q, nivel, calado, area_ha, tr = estado(dia)
    fecha = G["pico"] + pd.Timedelta(days=dia)
    paso_pico = dia >= 0

    fig = plt.figure(figsize=(10.8, 19.2), dpi=100, facecolor=FONDO)
    fig.text(0.5, 0.958, "SIMULACIÓN HIDRÁULICA", ha="center", color=TEXTO,
             fontsize=36, fontweight="bold")
    fig.text(0.5, 0.934, "Río Chambo · Chimborazo, Ecuador", ha="center", color=ACENTO,
             fontsize=23)
    fig.text(0.5, 0.913, f"Creciente del {fecha_larga(G['pico'])} · datos reales", ha="center",
             color=TENUE, fontsize=17)

    # 1) Terreno 3D con el agua
    ax3 = fig.add_axes([-0.08, 0.545, 1.16, 0.37], projection="3d", facecolor=FONDO)
    ax3.computed_zorder = False  # para que la línea de la sección quede sobre el terreno
    p3 = G["paso3d"]
    cal3 = calado[::p3, ::p3]
    zplot = G["z3"] + np.where(cal3 > 0, cal3, 0)
    fuerza = np.clip(0.55 + cal3 / 4, 0, 0.9)[..., None] * (cal3 > 0)[..., None]
    colores = G["rgb"] * (1 - fuerza) + AGUA * fuerza
    Xk, Yk = G["X"][::p3, ::p3] / 1000, G["Y"][::p3, ::p3] / 1000
    ax3.plot_surface(Xk, Yk, zplot, facecolors=colores, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, shade=False, zorder=1)
    k = G["k_rep"]
    sx = (G["ex"][k] + G["s"] * G["nx"][k]) / 1000
    sy = (G["ey"][k] + G["s"] * G["ny"][k]) / 1000
    dentro = np.abs(G["s"]) <= 600
    ax3.plot(sx[dentro], sy[dentro], G["secciones"][k][dentro] + 15, color=ACENTO, lw=3,
             zorder=5)
    ax3.view_init(elev=34, azim=azimut)
    rx, ry = np.ptp(Xk), np.ptp(Yk)
    ax3.set_box_aspect((rx, ry, np.ptp(zplot) / 1000 * 2.2), zoom=1.55)
    ax3.set_axis_off()

    # 2) Cifras del instante
    cifras = [("FECHA", f"{fecha.day} {MESES[fecha.month - 1][:3]} {fecha.year}"),
              ("CAUDAL", f"{q:,.0f} m³/s"),
              ("PERIODO RETORNO", texto_tr(tr)),
              ("ÁREA CON AGUA", f"{area_ha:,.0f} ha")]
    for n, (titulo, valor) in enumerate(cifras):
        xc = 0.125 + n * 0.25
        fig.text(xc, 0.523, titulo, ha="center", color=TENUE, fontsize=13)
        fig.text(xc, 0.497, valor, ha="center", color=TEXTO, fontsize=25, fontweight="bold")

    # 3) Hidrograma
    ev, de = G["evento"], G["dias_evento"]
    axh = fig.add_axes([0.12, 0.335, 0.8, 0.125])
    estilo_ejes(axh)
    axh.plot(de, ev.caudal_m3s, color="#3d5266", lw=2)
    hasta = de <= dia
    dd = np.append(de[hasta], dia)
    qq = np.append(ev.caudal_m3s.to_numpy()[hasta], q)
    axh.fill_between(dd, 0, qq, color="#1f73e6", alpha=0.35)
    axh.plot(dd, qq, color="#5aa9ff", lw=2.5)
    axh.plot(dia, q, "o", color=ACENTO, ms=11)
    ret = G["retorno"].set_index("periodo_anios").caudal_m3s
    for t in (2, 10, 100):
        axh.axhline(ret[t], color=TENUE, lw=1, ls="--")
        axh.text(de[-1], ret[t], f" Tr {t}", color=TENUE, fontsize=12, va="center")
    axh.xaxis.set_major_locator(MaxNLocator(integer=True))
    axh.set_xlim(de[0], de[-1])
    axh.set_ylim(0, max(ret[100], ev.caudal_m3s.max()) * 1.1)
    axh.set_title("HIDROGRAMA (caudal diario)", color=TEXTO, fontsize=17, pad=8)
    axh.set_xlabel("Días respecto al pico", color=TENUE, fontsize=13)
    axh.set_ylabel("m³/s", color=TENUE, fontsize=13)

    # 4) Sección transversal representativa
    zs, s = G["secciones"][k], G["s"]
    h = nivel[k]
    i0, i1 = region_mojada(zs, h, G["ic"][k])
    axs = fig.add_axes([0.12, 0.1, 0.8, 0.17])
    estilo_ejes(axs)
    base = zs.min() - 5
    axs.fill_between(s, base, zs, color="#4a3b2c")
    axs.plot(s, zs, color="#c9a77c", lw=2)
    axs.fill_between(s[i0:i1 + 1], zs[i0:i1 + 1], h, color="#1f73e6", alpha=0.85)
    if paso_pico:
        q_max = G["evento"].caudal_m3s.max()
        jm = np.interp(q_max, CAUDALES_TABLA, np.arange(len(CAUDALES_TABLA)))
        h_max = np.interp(jm, np.arange(len(CAUDALES_TABLA)), G["niveles"][k])
        axs.axhline(h_max, color=ACENTO, lw=1.5, ls="--")
        axs.text(-480, h_max + 0.3, "nivel máximo", color=ACENTO, fontsize=12)
    axs.set_xlim(-500, 500)
    axs.set_ylim(base, G["fondo"][k] + 18)
    km = G["abscisas"][k]
    axs.set_title(f"SECCIÓN K{int(km // 1000)}+{km % 1000:03.0f} · calado {h - zs[G['ic'][k]]:.1f} m"
                  f" · ancho {s[i1] - s[i0] + PASO:.0f} m", color=TEXTO, fontsize=17, pad=8)
    axs.set_xlabel("Distancia al eje (m)", color=TENUE, fontsize=13)
    axs.set_ylabel("Cota (m s.n.m.)", color=TENUE, fontsize=13)

    fig.text(0.5, 0.047, "Terreno: Copernicus DEM GLO-30 (ESA)  ·  Caudales: GEOGLOWS v2 (ECMWF/ERA5)",
             ha="center", color=TENUE, fontsize=12)
    fig.text(0.5, 0.033, f"Modelo 1D de flujo uniforme (Manning n = {N_MANNING})  ·  "
             "Fines educativos, no es un estudio oficial de riesgo", ha="center", color=TENUE,
             fontsize=12)
    fig.text(0.5, 0.01, FIRMA, ha="center", color=TEXTO, fontsize=19)

    fig.savefig(ruta, facecolor=FONDO, pil_kwargs={"quality": 92} if ruta.endswith(".jpg") else None)
    plt.close(fig)


def dibujar_frame(args):
    i, dia, azimut = args
    dibujar(i, dia, azimut, f"{CARPETA_FRAMES}/f{i:04d}.png")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(CARPETA_FRAMES, exist_ok=True)
    preparar()
    print(f"Pendiente media: {G['pendiente'] * 100:.2f} %  ·  secciones: {len(G['abscisas'])}"
          f"  ·  pico: {G['pico'].date()} ({G['evento'].caudal_m3s.max():.0f} m³/s)")

    # Tabla de resultados al pico
    q_max = G["evento"].caudal_m3s.max()
    _, nivel, _, area_ha, tr = estado(0.0)
    jm = int(round(np.interp(q_max, CAUDALES_TABLA, np.arange(len(CAUDALES_TABLA)))))
    pd.DataFrame({
        "abscisa_m": G["abscisas"].round(0), "cota_fondo_m": G["fondo"].round(2),
        "nivel_agua_m": nivel.round(2), "calado_m": (nivel - G["fondo"]).round(2),
        "ancho_m": (G["lim_der"] - G["lim_izq"])[:, jm].round(0),
    }).to_csv(TABLA, index=False)
    print(f"Al pico: {q_max:.0f} m³/s · Tr ≈ {tr:.0f} años · {area_ha:.0f} ha con agua")

    # Línea de tiempo: subida (9 s), pausa en el pico (2 s), bajada (6 s), cierre (2 s)
    dias = np.concatenate([np.linspace(-12, 0, 9 * FPS), np.zeros(2 * FPS),
                           np.linspace(0, 8, 6 * FPS), np.full(2 * FPS, 8.0)])
    azimuts = np.linspace(-115, -55, len(dias))
    with Pool() as pool:
        pool.map(dibujar_frame, list(zip(range(len(dias)), dias, azimuts)))

    dibujar(0, 0.0, -85, PORTADA)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", f"{CARPETA_FRAMES}/f%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
        VIDEO_SALIDA,
    ], check=True)
    print("Listo:", VIDEO_SALIDA)
