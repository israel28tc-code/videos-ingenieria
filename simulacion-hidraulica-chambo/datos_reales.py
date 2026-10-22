"""Descarga de datos reales (gratuitos y públicos) para las simulaciones.

- Terreno: Copernicus DEM GLO-30 (resolución 30 m), servido desde AWS.
- Caudales: GEOGLOWS v2, simulación retrospectiva diaria desde 1940
  (ERA5 + ruteo Muskingum) y caudales por periodo de retorno (Gumbel).
  Licencia CC-BY-NC-SA 4.0: citar la fuente y no usar con fines comerciales.

Uso:  python3 datos_reales.py   ->  archivos en datos/
"""
import json
import os
import subprocess

import numcodecs
import numpy as np
import pandas as pd

GEOGLOWS = "https://geoglows-v2.s3.us-west-2.amazonaws.com/retrospective"
COPERNICUS = "https://copernicus-dem-30m.s3.amazonaws.com"
CARPETA = "datos"


def descargar(url, destino=None):
    """Descarga con curl (respeta el proxy del sistema). Devuelve bytes o guarda en archivo."""
    cmd = ["curl", "-sSfL", "--retry", "3", url]
    if destino:
        subprocess.run(cmd + ["-o", destino], check=True)
        return None
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def leer_bloque_zarr(zarr_url, variable, indice_bloque, meta):
    """Lee un bloque (chunk) de un arreglo zarr v2 sin instalar zarr."""
    info = meta[f"{variable}/.zarray"]
    codec = numcodecs.get_codec(info["compressor"])
    sep = info.get("dimension_separator", ".")
    nombre = sep.join(str(i) for i in indice_bloque)
    crudo = codec.decode(descargar(f"{zarr_url}/{variable}/{nombre}"))
    forma = list(info["chunks"])
    datos = np.frombuffer(crudo, dtype=info["dtype"])
    # El último bloque de un eje puede venir completo (relleno) en zarr v2
    return datos.reshape(forma)


def buscar_indice_rio(zarr_url, meta, river_id):
    info = meta["river_id/.zarray"]
    n, paso = info["shape"][0], info["chunks"][0]
    for b in range(-(-n // paso)):
        ids = leer_bloque_zarr(zarr_url, "river_id", (b,), meta)
        pos = np.nonzero(ids == river_id)[0]
        if pos.size:
            return b * paso + int(pos[0])
    raise ValueError(f"river_id {river_id} no encontrado")


def caudales_geoglows(river_id):
    """Serie diaria de caudal (m3/s) y caudales por periodo de retorno."""
    url_d = f"{GEOGLOWS}/daily.zarr"
    meta_d = json.loads(descargar(f"{url_d}/.zmetadata"))["metadata"]
    idx = buscar_indice_rio(url_d, meta_d, river_id)

    nt = meta_d["time/.zarray"]["shape"][0]
    paso_t, paso_r = meta_d["Q/.zarray"]["chunks"]
    tiempo = np.concatenate([leer_bloque_zarr(url_d, "time", (b,), meta_d)
                             for b in range(-(-nt // paso_t))])[:nt]
    q = np.concatenate([leer_bloque_zarr(url_d, "Q", (b, idx // paso_r), meta_d)[:, idx % paso_r]
                        for b in range(-(-nt // paso_t))])[:nt]
    fechas = pd.Timestamp("1940-01-01") + pd.to_timedelta(tiempo, unit="s")
    serie = pd.DataFrame({"fecha": fechas.normalize(), "caudal_m3s": q.astype(float)})

    url_r = f"{GEOGLOWS}/return-periods.zarr"
    meta_r = json.loads(descargar(f"{url_r}/.zmetadata"))["metadata"]
    idx_r = buscar_indice_rio(url_r, meta_r, river_id)
    periodos = leer_bloque_zarr(url_r, "return_period", (0,), meta_r)
    paso = meta_r["gumbel_daily/.zarray"]["chunks"][1]
    gumbel = leer_bloque_zarr(url_r, "gumbel_daily", (0, idx_r // paso), meta_r)[:, idx_r % paso]
    retorno = pd.DataFrame({"periodo_anios": periodos, "caudal_m3s": gumbel})
    return serie, retorno


def dem_copernicus(lat, lon):
    """Descarga la hoja de 1°x1° del Copernicus DEM que contiene (lat, lon)."""
    la, lo = int(np.floor(lat)), int(np.floor(lon))
    ns = f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
    ew = f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
    nombre = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    destino = os.path.join(CARPETA, f"{nombre}.tif")
    if not os.path.exists(destino):
        descargar(f"{COPERNICUS}/{nombre}/{nombre}.tif", destino)
    return destino


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(CARPETA, exist_ok=True)

    # Río Chambo (Chimborazo), tramo entre Chambo y Penipe. ID del tramo en GEOGLOWS v2.
    RIO_CHAMBO = 620926418
    serie, retorno = caudales_geoglows(RIO_CHAMBO)
    serie.to_csv(os.path.join(CARPETA, "chambo_caudal_diario.csv"), index=False)
    retorno.to_csv(os.path.join(CARPETA, "chambo_periodos_retorno.csv"), index=False)
    print(serie.describe())
    print(retorno)

    print("DEM:", dem_copernicus(-1.62, -78.57))
