# Simulación hidráulica · río Chambo

Simula la creciente del 28 de febrero de 1988 del río Chambo (Chimborazo, Ecuador): el agua sube sobre el terreno 3D, con el hidrograma, el periodo de retorno y una sección transversal. Usa terreno y caudales reales. El resultado es un video vertical de 1080x1920, listo para Reels, TikTok y Shorts.

| Archivo | Qué es |
|---|---|
| `simulacion_rio_chambo.mp4` | El video (19 s, con audio propio sin derechos de autor) |
| `portada_chambo.jpg` | Portada: el pico de la creciente |
| `copys_simulacion_chambo.txt` | Textos para publicar y datos para responder comentarios |
| `secciones_pico_chambo.csv` | Resultado por sección al pico: cota de fondo, nivel de agua, calado, ancho |
| `datos/` | Caudales diarios 1940–2026 y caudales por periodo de retorno (el DEM se descarga aparte) |
| `datos_reales.py` | Descarga el terreno y los caudales |
| `simulacion_hidraulica.py` | Modelo hidráulico y generación del video |

## Cómo usarlo

Requiere Python 3.10 o superior y `ffmpeg`.

```bash
pip install -r requirements.txt
python3 datos_reales.py           # descarga el DEM (41 MB) y actualiza los caudales en datos/
python3 simulacion_hidraulica.py  # genera el video, la portada y la tabla (~4 min)
```

### Cambiar el evento o el río

Al inicio de `simulacion_hidraulica.py`:
- `EVENTO = "2017-03-18"` simula esa fecha. Con `None` toma la mayor creciente desde `ANIO_MIN`.
- `N_MANNING`, `SEPARACION` y `SEMIANCHO` son los parámetros hidráulicos.
- `INICIO_TRAMO` y `FIN_TRAMO` (lon, lat) definen el tramo del río.

Para otro río, cambia `RIO_CHAMBO` en `datos_reales.py` por el `river_id` de GEOGLOWS del tramo que quieras.

## Modelo hidráulico

1. **Eje del río:** se traza sobre el terreno como el camino de menor cota entre el inicio y el fin del tramo.
2. **Secciones transversales:** una cada 90 m, de ±700 m, muestreadas del DEM.
3. **Pendiente:** media del tramo, por regresión de la cota de fondo (0,78 %).
4. **Nivel de agua:** en cada sección y para cada caudal se resuelve la ecuación de Manning (flujo uniforme), considerando solo el área mojada conectada al cauce.
5. **Mancha de agua:** cada celda del terreno toma el nivel de su sección más cercana; está mojada si queda bajo ese nivel.
6. **Periodo de retorno:** se estima ajustando Gumbel a los caudales Tr 2–100 años de GEOGLOWS.

### Limitaciones (importante antes de publicar)

- Es un modelo de **flujo uniforme**. No es un modelo de remanso (HEC-RAS 1D) ni bidimensional.
- El DEM de 30 m no ve el fondo del cauce bajo el agua ni detalles finos (muros, puentes, diques).
- Los caudales de GEOGLOWS son **simulados**, no medidos en estación. Pueden diferir de los registros del INAMHI. Los valores de 1940–1979 son menos confiables.
- Los resultados sirven para **divulgación y educación**. No deben usarse para diseño ni como estudio oficial de riesgo.

## Fuentes de datos y licencias

- **Terreno:** Copernicus DEM GLO-30. © DLR e.V. 2010–2014 y © Airbus Defence and Space GmbH 2014–2018, provisto bajo COPERNICUS por la Unión Europea y la ESA.
- **Caudales:** GEOGLOWS v2, simulación retrospectiva (ECMWF ERA5 + ruteo Muskingum), licencia **CC BY-NC-SA 4.0**. Hay que citar la fuente y no usarlo con fines comerciales. Si publicas un video con estos datos en una cuenta monetizada o para un cliente, revisa la licencia.
