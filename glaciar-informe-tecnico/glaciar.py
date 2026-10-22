# -*- coding: utf-8 -*-
"""Video "Informe tecnico del glaciar del Cayambe" (1080x1920, 30 fps, ~29 s).
Todo se dibuja aqui (PIL + numpy) y el audio se sintetiza (viento, pad, efectos): sin material con derechos.
Datos: Ecociencia (mar-2026) Cayambe 1.225 ha en 2024, -41,9 % vs 1985; estudio UTN (Landsat) 6,1 km2 en 2050, 0 hacia 2080.
uso: python glaciar.py [frame <t> <salida.png>]   (sin argumentos renderiza todo)
"""
import math, os, subprocess, sys, wave
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "out" else "/home/user/glaciar_informe"
WORK = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
DUR = 29.0
NF = int(DUR * FPS)

FD = "/usr/share/fonts/opentype/inter/"
BLACK, BOLD, MED = FD + "InterDisplay-Black.otf", FD + "Inter-Bold.otf", FD + "Inter-Medium.otf"
MONO = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf"

BG = (9, 24, 41)
CYAN = (94, 200, 255)
ICE = (198, 236, 255)
WHITE = (246, 250, 255)
RED = (238, 66, 72)
GREEN = (52, 168, 96)
GRAY = (138, 164, 190)
NAVY = (9, 24, 41)

# escenas: (inicio, fin)
SC = [(0.0, 3.2), (3.2, 7.4), (7.4, 13.0), (13.0, 18.6), (18.6, 23.6), (23.6, 29.0)]
STAMP1, STAMP2 = 1.30, 23.85          # inicio del golpe del sello; impacto 0.16 s despues
SLAM = 0.16
TYPE = [3.55, 4.25, 4.95]             # inicio de cada linea de la ficha
TYPE_D = 0.5
SHRINK = (8.2, 11.2)                  # glaciar 1985 -> 2024
COUNT = (14.0, 16.8)                  # canchas
SOLID = (19.3, 20.3)                  # grafico 1985-2024
DASH = (20.5, 21.7)                   # proyeccion 2024-2080
A85, A24 = 21.08, 12.25               # km2
N_CANCHAS = 1237


# ---------------------------------------------------------------- utilidades
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, a, b):
    return clamp((t - a) / (b - a))


def eo(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def eio(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def eback(x):
    x = clamp(x)
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


_fc = {}


def font(path, size):
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]


_tc = {}


def timg(s, f, col):
    k = (s, f.path, f.size, col)
    if k not in _tc:
        b = f.getbbox(s, anchor="ls")
        im = Image.new("RGBA", (b[2] - b[0] + 8, b[3] - b[1] + 8), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((4 - b[0], 4 - b[1]), s, font=f, fill=col + (255,), anchor="ls")
        _tc[k] = im
    return _tc[k]


def with_alpha(im, a):
    im = im.copy()
    al = (np.asarray(im.getchannel("A"), dtype=np.float32) * a).astype(np.uint8)
    im.putalpha(Image.fromarray(al))
    return im


def comp(base, im, x, y):
    x, y = int(round(x)), int(round(y))
    w, h = im.size
    x0, y0, x1, y1 = max(0, x), max(0, y), min(base.width, x + w), min(base.height, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    base.alpha_composite(im.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def el(fr, s, f, col, x, y, t, t0, sa, dur=0.28, anchor="m", slide=26, pop=True):
    """Texto que entra con fade + subida + rebote. x = centro (m), izquierda (l) o derecha (r); y = arriba."""
    p = prog(t, t0, t0 + dur)
    if p <= 0 or sa <= 0:
        return
    im = timg(s, f, col)
    a = clamp(p * 2.2) * sa
    sc = 0.86 + 0.14 * eback(p) if pop else 1.0
    if abs(sc - 1) > 0.004:
        im = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.BICUBIC)
    if a < 0.999:
        im = with_alpha(im, a)
    x0 = x - im.width / 2 if anchor == "m" else (x if anchor == "l" else x - im.width)
    comp(fr, im, x0, y + (1 - eo(p)) * slide)


def dashed(d, p0, p1, fill, width=3, dash=14, gap=10, frac=1.0):
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0) * frac
    if L <= 0:
        return
    ux, uy = (x1 - x0) / math.hypot(x1 - x0, y1 - y0), (y1 - y0) / math.hypot(x1 - x0, y1 - y0)
    s = 0.0
    while s < L:
        e = min(s + dash, L)
        d.line((x0 + ux * s, y0 + uy * s, x0 + ux * e, y0 + uy * e), fill=fill, width=width)
        s += dash + gap


def miles(n):
    return f"{n:,}".replace(",", ".")


def coma(v, dec):
    return f"{v:.{dec}f}".replace(".", ",")


# ---------------------------------------------------------------- fondo + cajetin
def build_bg():
    a = np.zeros((H, W, 3), np.float32)
    a[:] = BG
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - W / 2) / W, (yy - H * 0.45) / H)
    a *= (1.08 - 0.55 * r)[..., None]
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    for x in range(0, W, 54):
        d.line((x, 0, x, H), fill=(40, 90, 140, 38 if x % 270 else 70), width=1)
    for y in range(0, H, 54):
        d.line((0, y, W, y), fill=(40, 90, 140, 38 if y % 270 else 70), width=1)
    # cajetin superior
    d.rectangle((40, 196, 1040, 258), outline=CYAN + (170,), width=2)
    d.line((470, 196, 470, 258), fill=CYAN + (170,), width=2)
    d.line((860, 196, 860, 258), fill=CYAN + (170,), width=2)
    f = font(MONO, 24)
    d.text((56, 227), "INFORME TÉCNICO N.º 001", font=f, fill=CYAN + (230,), anchor="lm")
    d.text((665, 227), "OBRA: GLACIAR CAYAMBE", font=f, fill=CYAN + (230,), anchor="mm")
    return im


BGIMG = build_bg()

# ---------------------------------------------------------------- montana
MR0, MR1 = 780, 1345
CTRL = [(0, 1330), (80, 1272), (160, 1206), (230, 1140), (300, 1076), (360, 1012), (410, 962), (455, 926),
        (500, 892), (540, 862), (575, 843), (605, 834), (640, 847), (675, 839), (712, 860), (752, 886),
        (800, 926), (850, 976), (900, 1036), (955, 1110), (1010, 1188), (1080, 1262)]


def smooth(v, k):
    ker = np.hanning(k)
    ker /= ker.sum()
    return np.convolve(v, ker, mode="same")


def build_mountain():
    rng = np.random.default_rng(11)
    xs = np.arange(W)
    ridge = np.interp(xs, [p[0] for p in CTRL], [p[1] for p in CTRL])
    ridge += 5 * smooth(rng.normal(0, 1.6, W), 9) + 9 * smooth(rng.normal(0, 2.2, W), 45)
    # mascara con antialias (2x)
    m2 = Image.new("L", (W * 2, (MR1 - MR0) * 2), 0)
    pts = [(x * 2, (ridge[x] - MR0) * 2) for x in xs] + [(W * 2, (MR1 - MR0) * 2 + 40), (0, (MR1 - MR0) * 2 + 40)]
    ImageDraw.Draw(m2).polygon(pts, fill=255)
    malpha = np.asarray(m2.resize((W, MR1 - MR0), Image.LANCZOS), dtype=np.float32) / 255
    rows = np.arange(MR0, MR1)[:, None].astype(np.float32)
    fade = np.clip((1342 - rows) / 160, 0, 1)
    # roca: degradado vertical + cara izquierda iluminada
    tv = np.clip((rows - 830) / 500, 0, 1)
    base = np.array([30, 66, 100], np.float32) * (1 - tv[..., None]) + np.array([13, 32, 54], np.float32) * tv[..., None]
    lit = 1 + 0.16 * np.clip((605 - xs[None, :]) / 605, -1, 1)
    rock = np.clip(base * lit[..., None], 0, 255)
    mt = Image.fromarray(np.dstack([rock, malpha * fade * 255]).astype(np.uint8), "RGBA")
    # curvas de nivel
    lines = Image.new("L", (W, MR1 - MR0), 0)
    dl = ImageDraw.Draw(lines)
    for k, y in enumerate(range(870, 1330, 34)):
        wob = [(x, y - MR0 + 5 * math.sin(x / 47 + k)) for x in range(0, W + 8, 8)]
        dl.line(wob, fill=255, width=2)
    la = np.asarray(lines, np.float32) / 255 * (malpha > 0.5) * fade * 0.20
    cy = Image.fromarray(np.dstack([np.broadcast_to(np.array(CYAN, np.uint8), (MR1 - MR0, W, 3)),
                                    (la * 255).astype(np.uint8)]), "RGBA")
    mt.alpha_composite(cy)
    # contorno de cresta (va encima de la nieve)
    ol = Image.new("RGBA", (W, MR1 - MR0), (0, 0, 0, 0))
    ImageDraw.Draw(ol).line([(x, ridge[x] - MR0) for x in xs], fill=CYAN + (225,), width=3)
    ola = np.asarray(ol, np.float32)
    ola[..., 3] *= fade
    ol = Image.fromarray(ola.astype(np.uint8), "RGBA")
    # linea de nieve dentada con lenguas glaciares
    jag = (14 * np.sin(xs / 31 + 0.5) + 9 * np.sin(xs / 11.3 + 2) + 6 * smooth(rng.normal(0, 2, W), 15))
    for c, amp in ((380, 46), (522, 38), (700, 52), (792, 34)):
        jag += amp * np.exp(-((xs - c) / 20.0) ** 2)
    # color de la nieve (con grietas)
    tv2 = np.clip((rows - 830) / 260, 0, 1)
    snow = np.array(WHITE, np.float32) * (1 - tv2[..., None]) + np.array([182, 222, 247], np.float32) * tv2[..., None]
    shade = 1 - 0.13 * np.clip((xs[None, :] - 605) / 475, 0, 1)
    snow = np.clip(snow * shade[..., None], 0, 255).astype(np.uint8)
    simg = Image.fromarray(snow, "RGB")
    ds = ImageDraw.Draw(simg)
    for _ in range(46):
        x = rng.uniform(330, 860)
        y = rng.uniform(850, 1060) - MR0
        ln = rng.uniform(14, 40)
        ang = rng.uniform(-0.5, 0.5)
        ds.line((x, y, x + ln * math.cos(ang), y + ln * math.sin(ang) + 6), fill=(150, 198, 230), width=2)
    msub = malpha > 0.5
    colcum = np.vstack([np.zeros((1, W), np.int64), np.cumsum(msub, axis=0)])
    return ridge, malpha, fade, mt, ol, jag, np.asarray(simg), msub, colcum


RIDGE, MALPHA, MFADE, MTIMG, OLIMG, JAG, SNOWRGB, MSUB, COLCUM = build_mountain()
ROWS = np.arange(MR0, MR1)[:, None]


def snow_count(ys):
    idx = np.clip(np.round(ys + JAG).astype(int) - MR0, 0, MR1 - MR0)
    return COLCUM[idx, np.arange(W)].sum()


YS_TABLE = np.arange(MR0, MR1)
CNT_TABLE = np.array([snow_count(y) for y in YS_TABLE])
YS0 = 1012
A0 = snow_count(YS0)


def ys_for_area(km2):
    target = A0 * km2 / A85
    return float(np.interp(target, CNT_TABLE, YS_TABLE))


def snow_layer(ys, a):
    lim = ys + JAG
    m = (MSUB & (ROWS < lim[None, :])).astype(np.uint8) * 255
    al = np.asarray(Image.fromarray(m).filter(ImageFilter.BoxBlur(1)), np.float32) / 255
    al = al * np.clip(MALPHA * 1.4, 0, 1) * MFADE * a
    return Image.fromarray(np.dstack([SNOWRGB, (al * 255).astype(np.uint8)]), "RGBA")


def snow_left(ys):
    lim = ys + JAG
    on = np.where(RIDGE < lim - 2)[0]
    x = int(on.min())
    return x, float(lim[x])


def build_ghost():
    lim = YS0 + JAG
    m = (MSUB & (ROWS < lim[None, :])).astype(np.float32) * 0.24
    g = Image.fromarray(np.dstack([np.broadcast_to(np.array(ICE, np.uint8), (MR1 - MR0, W, 3)),
                                   (m * 255).astype(np.uint8)]), "RGBA")
    d = ImageDraw.Draw(g)
    on = np.where(RIDGE < lim - 2)[0]
    pts = [(x, lim[x] - MR0) for x in range(on.min(), on.max() + 1)]
    for i in range(0, len(pts) - 1, 22):
        d.line(pts[i:i + 12], fill=CYAN + (235,), width=3)
    return g


GHOST = build_ghost()


def draw_mountain(fr, t, ma, oy, ys, ghost_a=0.0):
    if ma <= 0:
        return
    comp(fr, MTIMG if ma >= 0.999 else with_alpha(MTIMG, ma), 0, MR0 + oy)
    if ghost_a > 0:
        comp(fr, with_alpha(GHOST, ghost_a * ma), 0, MR0 + oy)
    comp(fr, snow_layer(ys, ma), 0, MR0 + oy)
    comp(fr, OLIMG if ma >= 0.999 else with_alpha(OLIMG, ma), 0, MR0 + oy)


# ---------------------------------------------------------------- sello
def build_stamp():
    f = font(BLACK, 116)
    s = "NO APROBADO"
    b = f.getbbox(s, anchor="lt")
    w, h = b[2] - b[0] + 110, b[3] - b[1] + 92
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((6, 6, w - 7, h - 7), radius=24, outline=RED + (255,), width=12)
    d.rounded_rectangle((27, 27, w - 28, h - 28), radius=12, outline=RED + (255,), width=4)
    d.text((w / 2, h / 2 + 2), s, font=f, fill=RED + (255,), anchor="mm")
    rng = np.random.default_rng(3)
    n = np.asarray(Image.fromarray((rng.random((h, w)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6)),
                   np.float32)
    n = (n - n.mean()) / n.std()
    a = np.asarray(im.getchannel("A"), np.float32) * np.clip((n + 1.7) * 1.4, 0, 1) * 0.96
    im.putalpha(Image.fromarray(a.astype(np.uint8)))
    return im.rotate(7, expand=True, resample=Image.BICUBIC)


STAMP = build_stamp()


def draw_stamp(fr, t, ts, cx, cy, base_sc=1.0, sa=1.0):
    p = prog(t, ts, ts + SLAM)
    if p <= 0 or sa <= 0:
        return
    if p < 1:
        sc = 1 + 1.4 * (1 - p) ** 2
    else:
        q = prog(t, ts + SLAM, ts + SLAM + 0.3)
        sc = 1 - 0.045 * math.sin(math.pi * q) * (1 - q)
    sc *= base_sc
    im = STAMP.resize((round(STAMP.width * sc), round(STAMP.height * sc)), Image.BICUBIC)
    a = clamp(p * 3) * sa
    if a < 0.999:
        im = with_alpha(im, a)
    comp(fr, im, cx - im.width / 2, cy - im.height / 2)


def shake(t):
    for ts in (STAMP1, STAMP2):
        q = prog(t, ts + SLAM, ts + SLAM + 0.28)
        if 0 < q < 1:
            amp = 16 * (1 - q)
            return amp * math.sin(t * 173), amp * math.cos(t * 131)
    return 0, 0


# ---------------------------------------------------------------- nieve que cae
_r = np.random.default_rng(5)
FLAKES = [(_r.uniform(0, W), _r.uniform(0, H), _r.uniform(40, 110), _r.uniform(8, 26), _r.uniform(0.3, 0.8),
           _r.uniform(1.4, 3.8), _r.uniform(60, 150), _r.uniform(0, 6.3)) for _ in range(70)]


def draw_flakes(d, t, a):
    for x0, y0, v, amp, fq, r, al, ph in FLAKES:
        y = (y0 + v * t) % (H + 40) - 20
        x = x0 + amp * math.sin(2 * math.pi * fq * t + ph)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(235, 246, 255, int(al * a)))


# ---------------------------------------------------------------- escenas
def scene_alpha(t, i):
    a, b = SC[i]
    if t < a or t >= b:
        return 0.0
    fin = 1.0 if i == 0 else eo(prog(t, a, a + 0.22))
    fout = 1.0 if i == len(SC) - 1 else 1 - eio(prog(t, b - 0.22, b))
    return fin * fout


def head(fr, s, t, t0, sa, y=292):
    el(fr, s, font(MONO, 36), CYAN, 540, y, t, t0, sa, pop=False, slide=14)


def s1(fr, d, t, sa):
    el(fr, "Soy ingeniero civil", font(BOLD, 60), ICE, 540, 318, t, 0.0, sa, dur=0.22)
    el(fr, "y FISCALICÉ", font(BLACK, 132), WHITE, 540, 400, t, 0.12, sa, dur=0.24)
    el(fr, "un glaciar", font(BLACK, 132), WHITE, 540, 548, t, 0.24, sa, dur=0.24)
    draw_stamp(fr, t, STAMP1, 540, 1050, 1.0, sa)
    el(fr, "¿Por qué? Te explico ↓", font(BOLD, 52), ICE, 540, 1392, t, 2.2, sa)


FICHA = [("UBICACIÓN : ", "Ecuador · Lat 0°", WHITE),
         ("ALTURA    : ", "5.790 m s.n.m.", WHITE),
         ("ESTADO    : ", "en retroceso", RED)]


def s2(fr, d, t, sa):
    head(fr, "FICHA TÉCNICA DE LA OBRA", t, 3.3, sa)
    f = font(MONO, 44)
    for i, (lab, val, col) in enumerate(FICHA):
        full = lab + val
        p = prog(t, TYPE[i], TYPE[i] + TYPE_D)
        n = int(len(full) * p)
        if n <= 0:
            continue
        y = 362 + i * 62
        shown_l, shown_v = full[:min(n, len(lab))], full[len(lab):n]
        d.text((110, y), shown_l, font=f, fill=CYAN + (int(255 * sa),), anchor="la")
        if shown_v:
            d.text((110 + f.getlength(lab), y), shown_v, font=f, fill=col + (int(255 * sa),), anchor="la")
        if p < 1 and int(t * 8) % 2 == 0:
            cx = 110 + f.getlength(full[:n])
            d.rectangle((cx + 2, y + 6, cx + 24, y + 50), fill=ICE + (int(200 * sa),))
    # insignia
    p = prog(t, 5.8, 6.1)
    if p > 0:
        a = clamp(p * 2) * sa
        sc = 0.9 + 0.1 * eback(p)
        bw, bh = 860 * sc, 118 * sc
        cy = 600
        d.rounded_rectangle((540 - bw / 2, cy - bh / 2, 540 + bw / 2, cy + bh / 2), radius=18 * sc,
                            fill=ICE + (int(235 * a),))
        if p > 0.5:
            el(fr, "El ÚNICO glaciar del planeta", font(BOLD, 44), NAVY, 540, cy - 48, 1, 0, a, pop=False, slide=0)
            el(fr, "justo sobre la línea ecuatorial", font(BOLD, 44), NAVY, 540, cy + 4, 1, 0, a, pop=False, slide=0)
    # linea ecuatorial + cumbre
    pe = eo(prog(t, 3.6, 4.4))
    if pe > 0:
        dashed(d, (455, 735), (455, 1335), CYAN + (int(230 * sa),), width=3, frac=pe)
        el(fr, "LÍNEA ECUATORIAL · 0°", font(MONO, 26), CYAN, 455, 690, t, 3.6, sa, pop=False, slide=8)
    sx = int(np.argmin(RIDGE))
    sy = RIDGE[sx]
    pl = prog(t, 4.4, 4.75)
    if pl > 0:
        a = int(255 * clamp(pl * 2) * sa)
        d.polygon([(sx, sy - 12), (sx - 10, sy + 4), (sx + 10, sy + 4)], fill=RED + (a,))
        d.line((sx + 8, sy - 10, sx + 70, sy - 70, sx + 120, sy - 70), fill=WHITE + (a,), width=2)
        el(fr, "5.790 m", font(MONO, 34), WHITE, sx + 128, sy - 92, t, 4.4, sa, anchor="l", pop=False, slide=6)


def s3(fr, d, t, sa):
    head(fr, "MEDICIÓN DE ÁREA DE HIELO", t, 7.5, sa)
    p = eio(prog(t, *SHRINK))
    area = A85 + (A24 - A85) * p
    year = round(1985 + (2024 - 1985) * p)
    el(fr, coma(area, 2) + " km²", font(BLACK, 150), WHITE, 540, 346, t, 7.6, sa, pop=False, slide=18)
    el(fr, f"AÑO {year}", font(MONO, 50), ICE, 540, 535, t, 7.7, sa, pop=False, slide=12)
    # etiquetas 1985 / 2024 a la izquierda de cada borde
    if t >= SHRINK[0] - 0.2:
        a = clamp(prog(t, SHRINK[0] - 0.2, SHRINK[0] + 0.1)) * sa
        x, y = snow_left(YS0)
        d.line((x - 12, y, x - 60, y), fill=CYAN + (int(220 * a),), width=2)
        d.text((x - 68, y), "1985", font=font(MONO, 30), fill=CYAN + (int(240 * a),), anchor="rm")
        if p > 0.15:
            ys = ys_for_area(area)
            x2, y2 = snow_left(ys)
            ly = min(y2, y - 38)
            a2 = clamp((p - 0.15) * 4) * sa
            d.line((x2 - 12, y2, x2 - 40, ly, x2 - 60, ly), fill=WHITE + (int(230 * a2),), width=2)
            d.text((x2 - 68, ly), str(year), font=font(MONO, 30), fill=WHITE + (int(250 * a2),), anchor="rm")
    el(fr, "−41,9 %", font(BLACK, 150), RED, 540, 1338, t, 11.35, sa, dur=0.3)
    el(fr, "menos hielo en 39 años", font(BOLD, 48), WHITE, 540, 1494, t, 11.7, sa)


CW, CH, COLS = 22, 15, 34
GX0, GY0 = (W - COLS * CW) // 2, 790


def s4(fr, d, t, sa):
    head(fr, "HIELO PERDIDO 1985–2024", t, 13.1, sa)
    el(fr, "Perdió 8,8 km² de hielo.", font(BOLD, 62), WHITE, 540, 345, t, 13.2, sa)
    el(fr, "Eso equivale a…", font(MED, 46), GRAY, 540, 430, t, 13.55, sa)
    n = round(N_CANCHAS * eio(prog(t, *COUNT)))
    el(fr, miles(n), font(BLACK, 170), ICE, 540, 488, t, 13.8, sa, pop=False, slide=14)
    el(fr, "canchas de fútbol", font(BLACK, 72), WHITE, 540, 688, t, 13.9, sa)
    a = int(255 * sa)
    for i in range(n):
        r, c = divmod(i, COLS)
        x, y = GX0 + c * CW, GY0 + r * CH
        fresh = i >= n - 30 and n < N_CANCHAS
        d.rectangle((x + 1, y + 1, x + CW - 2, y + CH - 2), fill=((150, 240, 175) if fresh else GREEN) + (a,))
        d.line((x + CW // 2, y + 2, x + CW // 2, y + CH - 3), fill=(230, 255, 235, a // 2), width=1)
    el(fr, "≈ 1 cancha cada 11,5 días", font(BOLD, 58), ICE, 540, 1378, t, 17.1, sa)
    el(fr, "(cancha FIFA 105 × 68 m · promedio 1985–2024)", font(MONO, 25), GRAY, 540, 1462, t, 17.4, sa, pop=False)


CX0, CX1, CY0, CY1 = 170, 900, 640, 1180


def X(yr):
    return CX0 + (yr - 1980) / (2085 - 1980) * (CX1 - CX0)


def Y(v):
    return CY1 - v / 24 * (CY1 - CY0)


def s5(fr, d, t, sa):
    head(fr, "PROYECCIÓN", t, 18.75, sa)
    el(fr, "Si sigue así…", font(BLACK, 104), WHITE, 540, 345, t, 18.9, sa)
    pa = clamp(prog(t, 18.9, 19.2)) * sa
    if pa > 0:
        fm = font(MONO, 26)
        for v in (0, 6, 12, 18, 24):
            d.line((CX0, Y(v), CX1, Y(v)), fill=(70, 130, 185, int(110 * pa)), width=1)
            d.text((CX0 - 14, Y(v)), str(v), font=fm, fill=GRAY + (int(255 * pa),), anchor="rm")
        d.text((CX0 - 14, CY0 - 34), "km²", font=fm, fill=GRAY + (int(255 * pa),), anchor="rm")
        d.line((CX0, CY1, CX1, CY1), fill=CYAN + (int(230 * pa),), width=3)
        d.line((CX0, CY0 - 10, CX0, CY1), fill=CYAN + (int(230 * pa),), width=3)
        for yr in (1985, 2024, 2050, 2080):
            d.line((X(yr), CY1, X(yr), CY1 + 10), fill=CYAN + (int(230 * pa),), width=2)
            d.text((X(yr), CY1 + 18), str(yr), font=fm, fill=GRAY + (int(255 * pa),), anchor="ma")
    ps = eio(prog(t, *SOLID))
    if ps > 0:
        x1 = 1985 + (2024 - 1985) * ps
        v1 = A85 + (A24 - A85) * ps
        d.polygon([(X(1985), Y(0)), (X(1985), Y(A85)), (X(x1), Y(v1)), (X(x1), Y(0))], fill=ICE + (int(55 * sa),))
        d.line((X(1985), Y(A85), X(x1), Y(v1)), fill=ICE + (int(255 * sa),), width=7)
        d.ellipse((X(1985) - 10, Y(A85) - 10, X(1985) + 10, Y(A85) + 10), fill=ICE + (int(255 * sa),))
        el(fr, "21,1", font(MONO, 34), ICE, X(1985) + 14, Y(A85) - 54, t, SOLID[0], sa, anchor="l", pop=False, slide=6)
        if ps >= 1:
            d.ellipse((X(2024) - 10, Y(A24) - 10, X(2024) + 10, Y(A24) + 10), fill=ICE + (int(255 * sa),))
            el(fr, "12,25", font(MONO, 34), ICE, X(2024), Y(A24) - 60, t, SOLID[1], sa, pop=False, slide=6)
    pd = prog(t, *DASH)
    if pd > 0:
        seg = [(2024, A24), (2050, 6.1), (2080, 0.0)]
        tot = (X(2080) - X(2024))
        cut = X(2024) + tot * eio(pd)
        for (ya, va), (yb, vb) in zip(seg, seg[1:]):
            xa, xb = X(ya), X(yb)
            if cut <= xa:
                break
            fr_ = clamp((cut - xa) / (xb - xa))
            dashed(d, (xa, Y(va)), (xb, Y(vb)), RED + (int(255 * sa),), width=6, dash=18, gap=12, frac=fr_)
        if eio(pd) > 0.46:
            d.ellipse((X(2050) - 9, Y(6.1) - 9, X(2050) + 9, Y(6.1) + 9), fill=RED + (int(255 * sa),))
            el(fr, "6,1", font(MONO, 34), RED, X(2050) + 10, Y(6.1) - 62, t, DASH[0] + 0.5, sa, anchor="l",
               pop=False, slide=6)
        if pd >= 1:
            pulse = 14 + 5 * math.sin((t - DASH[1]) * 9)
            d.ellipse((X(2080) - pulse, Y(0) - pulse, X(2080) + pulse, Y(0) + pulse), fill=RED + (int(255 * sa),))
    el(fr, "2080: 0 km²", font(BLACK, 128), RED, 540, 1262, t, 21.85, sa, dur=0.3)
    el(fr, "proyección: estudio UTN (imágenes Landsat)", font(MONO, 26), GRAY, 540, 1420, t, 22.2, sa, pop=False)


def s6(fr, d, t, sa):
    head(fr, "DICTAMEN TÉCNICO", t, 23.7, sa)
    draw_stamp(fr, t, STAMP2, 540, 540, 1.08, sa)
    el(fr, "OBSERVACIÓN:", font(MONO, 40), CYAN, 540, 790, t, 24.6, sa, pop=False)
    el(fr, "No se repara con hormigón", font(BOLD, 64), WHITE, 540, 852, t, 24.75, sa)
    el(fr, "ni con varilla de 12 mm.", font(BOLD, 64), WHITE, 540, 936, t, 25.1, sa)
    el(fr, "¿Lo visitarías", font(BLACK, 96), ICE, 540, 1092, t, 26.0, sa)
    el(fr, "antes de 2080?", font(BLACK, 96), ICE, 540, 1200, t, 26.12, sa)
    el(fr, "@israeltapia70", font(BOLD, 44), WHITE, 540, 1345, t, 26.6, sa)
    el(fr, "Fuentes: Ecociencia (2026) · UTN (Landsat)", font(MONO, 25), GRAY, 540, 1420, t, 26.7, sa, pop=False)


SCENES = [s1, s2, s3, s4, s5, s6]


def render(t):
    fr = BGIMG.copy()
    d = ImageDraw.Draw(fr, "RGBA")
    draw_flakes(d, t, 1.0)
    # montana (escenas 1-3)
    if t < SC[2][1]:
        ma = eo(prog(t, 0, 0.25)) * (1 - eio(prog(t, SC[2][1] - 0.22, SC[2][1])))
        oy = (1 - eo(prog(t, 0, 0.5))) * 80
        area = A85 + (A24 - A85) * eio(prog(t, *SHRINK))
        ghost = clamp(prog(t, SHRINK[0] - 0.2, SHRINK[0] + 0.2)) if t >= SC[2][0] else 0.0
        draw_mountain(fr, t, ma, oy, ys_for_area(area), ghost)
    d = ImageDraw.Draw(fr, "RGBA")
    for i, fn in enumerate(SCENES):
        sa = scene_alpha(t, i)
        if sa > 0:
            fn(fr, d, t, sa)
            d.text((950, 227), f"LÁM. {i + 1}/6", font=font(MONO, 24), fill=CYAN + (int(230 * sa),), anchor="mm")
    dx, dy = shake(t)
    if dx or dy:
        sh = Image.new("RGBA", (W, H), BG + (255,))
        sh.paste(fr, (int(dx), int(dy)))
        fr = sh
    fo = prog(t, DUR - 0.4, DUR)
    if fo > 0:
        fr = Image.blend(fr, Image.new("RGBA", (W, H), BG + (255,)), eio(fo))
    return fr.convert("RGB")


# ---------------------------------------------------------------- audio
SR = 48000


def build_audio(path):
    n = int((DUR + 0.5) * SR)
    t = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)
    rng = np.random.default_rng(7)
    f = np.fft.rfftfreq(n, 1 / SR)

    def band(x, lo, hi, tilt=0.0):
        g = 1 / (1 + (lo / np.maximum(f, 1)) ** 4) / (1 + (f / hi) ** 4)
        if tilt:
            g = g * (np.maximum(f, 20) / 1000) ** tilt
        return np.fft.irfft(np.fft.rfft(x) * g, n=n)

    # viento estereo con rafagas
    gust = 0.55 + 0.25 * np.sin(2 * np.pi * 0.11 * t) + 0.2 * np.sin(2 * np.pi * 0.27 * t + 1.3)
    for ch, ph in ((L, 0.0), (R, 1.7)):
        w = band(rng.normal(0, 1, n), 70, 650, -0.6)
        w /= np.abs(w).max()
        hi = band(rng.normal(0, 1, n), 900, 2600)
        hi /= np.abs(hi).max()
        g2 = 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + ph)
        ch += 0.20 * w * gust + 0.035 * hi * g2 * gust
    # pad: La menor -> Fa mayor en el dictamen
    def chord(freqs, amp):
        s = np.zeros(n)
        for k, fq in enumerate(freqs):
            for det in (-0.35, 0.35):
                s += np.sin(2 * np.pi * (fq + det) * t + k) / (k + 1.6)
        return s * amp
    am = chord([55, 110, 130.81, 164.81, 220], 0.075)
    fmaj = chord([43.65, 87.31, 110, 130.81, 174.61], 0.075)
    xf = np.clip((t - SC[5][0]) / 0.8, 0, 1)
    swell = 0.7 + 0.3 * np.clip((t - SC[4][0]) / 3, 0, 1) * (1 - xf)
    trem = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
    pad = (am * (1 - xf) + fmaj * xf) * swell * trem * np.clip(t / 1.5, 0, 1)
    L += pad
    R += pad

    def add(sig, t0, gain=1.0, pan=0.0):
        i0 = int(t0 * SR)
        if i0 >= n:
            return
        sig = sig[: n - i0] * gain
        L[i0:i0 + len(sig)] += sig * (1 - pan) if pan > 0 else sig
        R[i0:i0 + len(sig)] += sig * (1 + pan) if pan < 0 else sig

    def env_exp(m, tau):
        return np.exp(-np.arange(m) / SR / tau)

    def whoosh(t0, dur=0.5, g=0.35):
        m = int(dur * SR)
        x = rng.normal(0, 1, m)
        X = np.fft.rfft(x)
        ff = np.fft.rfftfreq(m, 1 / SR)
        lo = np.fft.irfft(X / (1 + (ff / 700) ** 2), n=m)
        hi = np.fft.irfft(X * (ff / 2500) ** 2 / (1 + (ff / 2500) ** 2) / (1 + (ff / 9000) ** 4), n=m)
        lo /= np.abs(lo).max()
        hi /= np.abs(hi).max()
        u = np.linspace(0, 1, m)
        e = np.sin(np.pi * u) ** 2
        add((lo * (1 - u) + hi * u) * e, t0, g)

    def ding(t0, fq=1568, g=0.22):
        m = int(1.2 * SR)
        tt = np.arange(m) / SR
        s = (np.sin(2 * np.pi * fq * tt) + 0.32 * np.sin(2 * np.pi * fq * 2.01 * tt) * np.exp(-tt / 0.15)
             + 0.12 * np.sin(2 * np.pi * fq * 3.02 * tt) * np.exp(-tt / 0.08))
        s *= np.exp(-tt / 0.32) * np.clip(tt / 0.003, 0, 1)
        add(s, t0, g)

    def thump(t0, g=0.9):
        m = int(0.7 * SR)
        tt = np.arange(m) / SR
        fq = 42 + 110 * np.exp(-tt / 0.06)
        s = np.sin(2 * np.pi * np.cumsum(fq) / SR) * np.exp(-tt / 0.2)
        nz = rng.normal(0, 1, m) * np.exp(-tt / 0.02)
        nzl = np.convolve(nz, np.ones(12) / 12, mode="same")
        snap = np.diff(rng.normal(0, 1, m + 1)) * np.exp(-tt / 0.012) * 0.35
        add(s + 0.8 * nzl + snap, t0, g)

    def click(t0, g=0.10, fq=2400):
        m = int(0.03 * SR)
        tt = np.arange(m) / SR
        s = (np.diff(rng.normal(0, 1, m + 1)) * 0.5 + np.sin(2 * np.pi * fq * tt)) * np.exp(-tt / 0.006)
        add(s, t0, g, pan=rng.uniform(-0.3, 0.3))

    def crack(t0, g=0.5):
        k = rng.integers(7, 14)
        for j in range(k):
            dt = rng.uniform(0, 0.14)
            m = int(rng.uniform(0.004, 0.012) * SR)
            tt = np.arange(m) / SR
            s = np.diff(rng.normal(0, 1, m + 1)) * np.exp(-tt / rng.uniform(0.002, 0.006))
            add(s, t0 + dt, g * rng.uniform(0.4, 1.0), pan=rng.uniform(-0.5, 0.5))
        m = int(0.25 * SR)
        tt = np.arange(m) / SR
        add(np.sin(2 * np.pi * 95 * tt) * np.exp(-tt / 0.07), t0, g * 0.5)

    def fall(t0, dur, f0, f1, g=0.10):
        m = int(dur * SR)
        tt = np.arange(m) / SR
        fq = f0 * (f1 / f0) ** (tt / dur)
        s = np.sin(2 * np.pi * np.cumsum(fq) / SR) * (0.8 + 0.2 * np.sin(2 * np.pi * 7 * tt))
        s *= np.sin(np.pi * tt / dur) ** 0.7
        add(s, t0, g)

    # --- linea de tiempo (mismos tiempos que el video)
    thump(0.02, 0.4)
    whoosh(0.0, 0.45, 0.3)
    for tt0 in (0.0, 0.12, 0.24):
        click(tt0, 0.12, 1800)
    thump(STAMP1 + SLAM, 0.6)
    ding(2.2, 1318, 0.2)
    for i, sc in enumerate(SC[1:], 1):
        whoosh(sc[0] - 0.12, 0.5, 0.32)
    for i, (lab, val, _) in enumerate(FICHA):
        full = lab + val
        for c in range(len(full)):
            if full[c] != " ":
                click(TYPE[i] + TYPE_D * c / len(full), 0.07, rng.uniform(1900, 2900))
        ding(TYPE[i] + TYPE_D + 0.02, 1760, 0.12)
    ding(5.8, 2093, 0.22)
    whoosh(3.6, 0.6, 0.12)
    for k in range(int((SHRINK[1] - SHRINK[0]) / 0.09)):
        click(SHRINK[0] + k * 0.09, 0.05, 3000)
    tc = SHRINK[0] + 0.1
    while tc < SHRINK[1]:
        crack(tc, 0.3)
        tc += rng.uniform(0.28, 0.55)
    thump(11.35 + 0.05, 0.5)
    click(11.7, 0.1)
    prev = 0
    for k in range(1, 2000):
        tk = COUNT[0] + (COUNT[1] - COUNT[0]) * k / 2000
        nn = round(N_CANCHAS * eio(prog(tk, *COUNT)))
        if nn // COLS > prev:
            prev = nn // COLS
            click(tk, 0.11, 3200 if prev % 2 else 2600)
    ding(COUNT[1] + 0.05, 1760, 0.26)
    click(17.1, 0.1)
    fall(SOLID[0], SOLID[1] - SOLID[0], 330, 300, 0.05)
    fall(DASH[0], DASH[1] - DASH[0] + 0.15, 520, 130, 0.12)
    thump(21.85 + 0.05, 0.5)
    thump(STAMP2 + SLAM, 0.6)
    click(24.75, 0.1)
    click(25.1, 0.1)
    ding(26.0, 2093, 0.24)
    ding(26.12, 1568, 0.16)
    # salida
    fo = np.clip((DUR - t) / 0.8, 0, 1)
    L *= fo
    R *= fo
    pk = max(np.abs(L).max(), np.abs(R).max())
    st = np.stack([L, R], 1) / pk * 0.89
    with wave.open(path, "wb") as wv:
        wv.setnchannels(2)
        wv.setsampwidth(2)
        wv.setframerate(SR)
        wv.writeframes((st * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- render
def work(rng_):
    k, a, b = rng_
    seg = os.path.join(WORK, f"seg_{k}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                          "-pix_fmt", "yuv420p", "-g", "60", seg], stdin=subprocess.PIPE)
    for i in range(a, b):
        p.stdin.write(render(i / FPS).tobytes())
    p.stdin.close()
    p.wait()
    return seg


def main():
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "frame":
        render(float(sys.argv[2])).save(sys.argv[3])
        return
    wav = os.path.join(WORK, "audio.wav")
    build_audio(wav)
    k = 4
    cuts = [round(NF * j / k) for j in range(k + 1)]
    with Pool(k) as pool:
        segs = pool.map(work, [(j, cuts[j], cuts[j + 1]) for j in range(k)])
    lst = os.path.join(WORK, "segs.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{s}'\n" for s in segs)
    mp4 = os.path.join(OUT, "glaciar_informe_tecnico.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                    "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.85:level=disabled", "-ar", "48000", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "-movflags", "+faststart", mp4], check=True)
    render(2.7).save(os.path.join(OUT, "glaciar_portada.jpg"), quality=92)
    print(mp4)


if __name__ == "__main__":
    main()
