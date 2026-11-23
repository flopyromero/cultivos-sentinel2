# Filtro "¿esto es un lote agrícola?" para "Probalo en tu lote".
#
# La red siempre elige una de sus 4 clases (maíz, soja 1ra, soja 2da, otros): no sabe decir "acá no hay un lote".
# Este filtro va ANTES de la red y no la cambia. Usa Dynamic World (Google), un mapa de cobertura del suelo hecho con
# Sentinel-2 a 10 m que da, para cada imagen, la probabilidad de 9 coberturas (agua, árboles, pasto, cultivo,
# construido, etc.). Se promedia en el parche de 160 m durante toda la campaña.
#
# Criterio conservador: solo se rechaza el punto si la mayor parte es claramente NO agrícola (construido, agua,
# árboles o bañado). Pasto, suelo desnudo y arbustos NO se rechazan, porque un lote en barbecho o una pastura
# pueden verse así, y la clase "Otros" de la red ya incluye pasturas.
#
# Si Dynamic World falla o no tiene datos, el filtro se saltea y la página funciona como antes.

CLASES_DW = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']

NO_AGRICOLAS = {                       # cobertura -> cómo se lo decimos a la persona
    'built': 'zona construida (pueblo, ciudad, galpones o ruta)',
    'water': 'agua (río, laguna o canal)',
    'trees': 'árboles o monte',
    'flooded_vegetation': 'bañado o isla',
    'snow_and_ice': 'nieve',
}

UMBRAL = 0.5                           # se rechaza si más del 50 % del parche es no agrícola, en promedio en la campaña


def probabilidades(inicio, fin, x0, y0, lado=160):
    """Promedio de las probabilidades de Dynamic World en el parche (UTM 20S, esquina superior izquierda x0, y0)
    entre las fechas inicio y fin. Devuelve un dict {cobertura: probabilidad} o None si no hay datos o falla."""
    try:
        import ee
        parche = ee.Geometry.Rectangle([x0, y0 - lado, x0 + lado, y0], 'EPSG:32720', False)
        dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
              .filterBounds(parche)
              .filterDate(inicio, fin)
              .select(CLASES_DW))
        prom = dw.mean().reduceRegion(reducer=ee.Reducer.mean(), geometry=parche, scale=10,
                                      crs='EPSG:32720', maxPixels=1e6).getInfo()
        if not prom or any(prom.get(c) is None for c in CLASES_DW):
            return None
        return {c: float(prom[c]) for c in CLASES_DW}
    except Exception:
        return None


def decidir(prob, umbral=UMBRAL):
    """Devuelve (es_lote, motivo, porcentaje_no_agricola). Sin datos (prob None) se considera lote: el filtro no bloquea."""
    if not prob:
        return True, None, None
    total = sum(prob.values()) or 1.0
    no_agricola = sum(prob.get(c, 0.0) for c in NO_AGRICOLAS) / total
    if no_agricola <= umbral:
        return True, None, round(100 * no_agricola, 1)
    principal = max(NO_AGRICOLAS, key=lambda c: prob.get(c, 0.0))
    return False, principal, round(100 * no_agricola, 1)
