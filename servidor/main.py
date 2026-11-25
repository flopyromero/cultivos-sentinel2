# Servidor de la demo "Probalo en tu lote" (TP final de IA II, UCA Rosario 2026).
#
# Única tarea: recibir un punto y una campaña, bajar de Earth Engine la serie de Sentinel-2 de ese lugar
# preparada EXACTAMENTE como en el notebook 02 (nubes con Cloud Score+, mediana cada 10 días, 27 períodos,
# parche de 16 x 16 píxeles de 10 m y huecos interpolados) y devolverla. El modelo NO corre acá: corre en el
# navegador, en la página de Hugging Face, con los mismos modelos ONNX y la misma validación que los lotes de ejemplo.
#
# Variables de entorno:
#   EE_CLAVE   contenido del .json de la cuenta de servicio de Earth Engine (secreto: se carga en Render, nunca en git)
#   ORIGENES   páginas que pueden usar el servidor, separadas por coma (por defecto, la del Space)
#   FILTRO_COBERTURA  1 (por defecto) para avisar cuando el punto no es un lote agrícola (Dynamic World, ver
#              cobertura.py); 0 para apagarlo y volver al comportamiento anterior sin tocar el código
#
# Local:  EE_CLAVE="$(cat clave.json)" uvicorn main:app --reload

import base64
import datetime
import json
import os
import threading
import time
from collections import OrderedDict, defaultdict, deque

import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pyproj import Transformer

import cobertura

# ------------------------------------------------------------------ constantes del notebook 02 (no cambiar)
REGION_LIMITES = (-63.6, -34.3, -59.5, -31.4)          # oeste, sur, este, norte: la zona donde se entrenó y evaluó
# Se acepta cualquier punto de Argentina (rectángulo aproximado del país). Fuera de REGION_LIMITES la red responde igual,
# pero no se entrenó ni se evaluó ahí: la respuesta lo marca con 'fuera_de_zona' y la página lo avisa.
# No se acepta el resto del mundo: las 27 fechas van de septiembre a mayo, que es el verano del hemisferio sur.
PERMITIDO = (-73.6, -55.1, -53.6, -21.7)               # oeste, sur, este, norte
# El rectángulo incluye partes de Chile, Uruguay, Paraguay, Bolivia y Brasil. Fuera de la zona del trabajo, el país se
# confirma con FAO GAUL (la misma fuente de límites que usa el 02 para los departamentos): ver controlar_pais().
PAISES_ACEPTADOS = {'Argentina'}                        # ADM0_NAME en FAO GAUL 2015
PATCH = 16                                               # 16 x 16 píxeles de 10 m = 160 m
BANDAS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
CS_UMBRAL = 0.6                                          # Cloud Score+: se descarta todo píxel con puntaje menor
CAMPANAS = ['2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25', '2025_26']


def periodos(inicio):
    # 27 períodos de septiembre a mayo: días 1-10, 11-20 y 21-fin de cada mes (fin exclusivo, como filterDate)
    a, out = datetime.date.fromisoformat(inicio), []
    for k in range(9):
        y, m = a.year + (a.month - 1 + k) // 12, (a.month - 1 + k) % 12 + 1
        fin_mes = datetime.date(y + (m == 12), m % 12 + 1, 1)
        for d0, d1 in [(1, 11), (11, 21), (21, None)]:
            out.append((datetime.date(y, m, d0).isoformat(), (datetime.date(y, m, d1) if d1 else fin_mes).isoformat()))
    return out


def inicio_de(campana):
    return f'20{campana[2:4]}-09-01'


T = len(periodos('2020-09-01'))
NOMBRES_BANDAS = [f't{t:02d}_{b}' for t in range(T) for b in BANDAS]


def interpolar(Xc):
    # Igual al notebook 02. Xc: (n, T, H, W, C) uint16 con 0 donde no hay dato. Interpola cada píxel en el tiempo.
    Tn = Xc.shape[1]
    valido = ~(Xc == 0).all(axis=-1)
    t = np.arange(Tn).reshape(1, Tn, 1, 1)
    prev = np.maximum.accumulate(np.where(valido, t, -1), axis=1)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(valido, t, Tn), 1), axis=1), 1)
    hay_prev, hay_next = prev >= 0, nxt < Tn
    Xf = Xc.astype(np.float32)
    Xp = np.take_along_axis(Xf, np.clip(prev, 0, Tn - 1)[..., None], axis=1)
    Xn = np.take_along_axis(Xf, np.clip(nxt, 0, Tn - 1)[..., None], axis=1)
    w = ((t - prev) / np.maximum(nxt - prev, 1))[..., None]
    out = np.where((hay_prev & hay_next)[..., None], Xp + w * (Xn - Xp),
          np.where(hay_prev[..., None], Xp, np.where(hay_next[..., None], Xn, 0)))
    out = np.where(valido[..., None], Xf, out)
    return np.round(out).astype(np.uint16), ~(hay_prev | hay_next)


a_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32720', always_xy=True)
a_geo = Transformer.from_crs('EPSG:32720', 'EPSG:4326', always_xy=True)


def grilla(lon, lat):
    # Esquina superior izquierda del parche, alineada a la grilla de 10 m, como en bajar_parche() del 02
    x, y = a_utm.transform(lon, lat)
    x0 = round(x / 10) * 10 - PATCH * 5
    y0 = round(y / 10) * 10 + PATCH * 5
    return x0, y0


def esquinas(x0, y0):
    lado = PATCH * 10
    return [list(reversed(a_geo.transform(x, y))) for x, y in
            [(x0, y0), (x0 + lado, y0), (x0 + lado, y0 - lado), (x0, y0 - lado)]]     # [lat, lon]


# ------------------------------------------------------------------ Earth Engine
EE = {'listo': False, 'error': None, 'proyecto': None}


def iniciar_ee():
    clave = os.environ.get('EE_CLAVE', '').strip()
    if not clave:
        EE['error'] = 'Falta la variable EE_CLAVE con la clave de la cuenta de servicio.'
        return
    try:
        import ee
        info = json.loads(clave)
        cred = ee.ServiceAccountCredentials(info['client_email'], key_data=clave)
        ee.Initialize(cred, project=info['project_id'])
        EE.update(listo=True, error=None, proyecto=info['project_id'])
    except json.JSONDecodeError:
        EE['error'] = 'EE_CLAVE no es un JSON válido: pegá el contenido completo del archivo .json de la clave.'
    except KeyError as e:
        EE['error'] = f'A la clave le falta el campo {e}: pegá el contenido completo del archivo .json.'
    except Exception as e:
        # el mensaje de la librería puede incluir la clave: se muestra solo el tipo de error
        EE['error'] = f'No se pudo conectar con Earth Engine ({type(e).__name__}). Revisar la clave y los permisos de la cuenta de servicio.'


def parche_ee(x0, y0):
    import ee
    lado = PATCH * 10
    return ee.Geometry.Rectangle([x0, y0 - lado, x0 + lado, y0], 'EPSG:32720', False)


def dentro_de(limites, lat, lon):
    oeste, sur, este, norte = limites
    return oeste <= lon <= este and sur <= lat <= norte


def stack_campana(inicio, zona):
    # Igual al notebook 02. La única diferencia es filterBounds: el 02 filtraba por la zona del trabajo y acá se filtra
    # por el parche pedido, para que funcione en cualquier lugar. Dentro de la zona la serie es la misma: las imágenes
    # que no tocan el parche no tienen píxeles ahí y no cambian la mediana.
    import ee
    region = zona
    cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    per = periodos(inicio)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(region)
          .filterDate(per[0][0], per[-1][1])
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
          .linkCollection(cs_plus, ['cs_cdf'])
          .map(lambda img: img.updateMask(img.select('cs_cdf').gte(CS_UMBRAL)))
          .select(BANDAS))
    vacio = ee.Image.constant([0] * len(BANDAS)).rename(BANDAS).updateMask(0)
    capas = []
    for t, (a, b) in enumerate(per):
        col = s2.filterDate(a, b)
        comp = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), vacio))
        capas.append(comp.select(BANDAS).rename([f't{t:02d}_{b}' for b in BANDAS]))
    return ee.Image.cat(capas).unmask(0).toUint16()


def bajar_crudo(campana, x0, y0, intentos=4):
    # Igual a bajar_parche() del 02, pero con la grilla ya calculada. Devuelve (T, H, W, C) uint16 con 0 donde no hay dato.
    import ee
    pedido = {'expression': stack_campana(inicio_de(campana), parche_ee(x0, y0)), 'fileFormat': 'NUMPY_NDARRAY',
              'grid': {'dimensions': {'width': PATCH, 'height': PATCH},
                       'affineTransform': {'scaleX': 10, 'shearX': 0, 'translateX': x0,
                                           'shearY': 0, 'scaleY': -10, 'translateY': y0},
                       'crsCode': 'EPSG:32720'}}
    ultimo = None
    for k in range(intentos):
        try:
            arr = ee.data.computePixels(pedido)
            a = np.stack([arr[n] for n in NOMBRES_BANDAS], axis=-1)
            a = a.reshape(PATCH, PATCH, T, len(BANDAS))
            return a.transpose(2, 0, 1, 3).astype(np.uint16)
        except Exception as e:
            ultimo = e
            time.sleep(2 ** k)
    raise RuntimeError(f'Earth Engine no respondió: {type(ultimo).__name__}: {str(ultimo)[:200]}')


def ubicacion_de(lon, lat):
    # (departamento, provincia, país) según FAO GAUL, como en el 02. Si el punto no cae en ningún país (mar o un río
    # ancho), los tres son None. Si Earth Engine falla, la excepción sigue: quien llama decide qué hacer.
    import ee
    gaul = ee.FeatureCollection('FAO/GAUL/2015/level2').filterBounds(ee.Geometry.Point([lon, lat]))
    f = gaul.first().getInfo()
    p = f['properties'] if f else {}
    return p.get('ADM2_NAME'), p.get('ADM1_NAME'), p.get('ADM0_NAME')


def departamento_de(lon, lat):
    # Opcional: nombre del departamento. Si falla, la serie igual se devuelve.
    try:
        depto, prov, _ = ubicacion_de(lon, lat)
        return depto, prov
    except Exception:
        return None, None


# ------------------------------------------------------------------ app
@asynccontextmanager
async def al_iniciar(app):
    iniciar_ee()
    yield


app = FastAPI(title='Serie Sentinel-2 de un lote', version='1.0', lifespan=al_iniciar)
ORIGENES = [o.strip() for o in os.environ.get(
    'ORIGENES', 'https://final-ia-final-ia-ii.static.hf.space,https://huggingface.co').split(',') if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=ORIGENES, allow_methods=['GET'], allow_headers=['*'])

CACHE_MAX = 300
_cache = OrderedDict()
_cache_lock = threading.Lock()
_simultaneos = threading.Semaphore(3)            # 512 MB de RAM: pocos pedidos a la vez
LIMITE_POR_IP, VENTANA_S = 40, 3600              # pedidos nuevos (no en caché) por IP y por hora
LIMITE_TOTAL = 400                               # y en total, por hora: tope que no depende de identificar a nadie
_pedidos = defaultdict(deque)
_todos = deque()
_pedidos_lock = threading.Lock()


def filtro_activo():
    return os.environ.get('FILTRO_COBERTURA', '1').strip().lower() not in ('0', 'no', 'false', 'off')


def revisar_cobertura(campana, x0, y0):
    # Filtro previo (cobertura.py): si el parche es mayormente construido, agua, monte o bañado, no se clasifica.
    # Si Dynamic World falla, devuelve None y el pedido sigue como antes.
    inicio = inicio_de(campana)
    prob = cobertura.probabilidades(inicio, periodos(inicio)[-1][1], x0, y0)
    es_lote, motivo, pct = cobertura.decidir(prob)
    return prob, es_lote, motivo, pct


def mensaje_no_lote(campana, motivo, pct):
    return (f'Este punto no parece un lote con cultivo: según el mapa de cobertura Dynamic World, en la campaña '
            f'{campana.replace("_", "/")} el {pct:.0f} % del recuadro no es agrícola: es sobre todo '
            f'{cobertura.NO_AGRICOLAS[motivo]}. '
            'La red solo sabe reconocer cultivos, así que no lo clasifica. Elegí un punto en el medio de un lote.')


def ip_de(request: Request):
    # Render está detrás de Cloudflare, que pone la IP real en CF-Connecting-IP y pisa cualquier valor que mande el
    # cliente. Si no está, se usa la ÚLTIMA dirección de X-Forwarded-For (la agrega el proxy, no el cliente): la
    # primera la puede inventar cualquiera y saltearía el límite.
    cf = request.headers.get('cf-connecting-ip', '').strip()
    if cf:
        return cf
    reenviada = [x.strip() for x in request.headers.get('x-forwarded-for', '').split(',') if x.strip()]
    if reenviada:
        return reenviada[-1]
    return request.client.host if request.client else '?'


def controlar_limite(ip):
    ahora = time.time()
    with _pedidos_lock:
        for q in (_pedidos[ip], _todos):
            while q and ahora - q[0] > VENTANA_S:
                q.popleft()
        if len(_todos) >= LIMITE_TOTAL:
            raise HTTPException(429, 'El servidor recibió demasiados pedidos en la última hora. Probá de nuevo en un rato.')
        if len(_pedidos[ip]) >= LIMITE_POR_IP:
            raise HTTPException(429, 'Demasiados pedidos desde esta conexión. Probá de nuevo en un rato.')
        _pedidos[ip].append(ahora)
        _todos.append(ahora)


@app.get('/')
@app.get('/salud')
def salud(request: Request):
    return {'ok': True, 'tu_ip': ip_de(request), 'earth_engine': EE['listo'], 'error': EE['error'], 'campanas': CAMPANAS,
            'region': dict(zip(['oeste', 'sur', 'este', 'norte'], REGION_LIMITES)),
            'permitido': dict(zip(['oeste', 'sur', 'este', 'norte'], PERMITIDO)), 'filtro_cobertura': filtro_activo()}


def validar(lat, lon, campana):
    if campana not in CAMPANAS:
        raise HTTPException(422, f'Campaña desconocida: {campana}. Opciones: {", ".join(CAMPANAS)}.')
    if not dentro_de(PERMITIDO, lat, lon):
        raise HTTPException(422, 'El punto está fuera de Argentina. La serie va de septiembre a mayo, el verano del '
                                 'hemisferio sur, así que el modelo solo tiene sentido en Argentina.')


def controlar_pais(lat, lon):
    # Fuera de la zona del trabajo, confirma con FAO GAUL que el punto está en Argentina (el rectángulo PERMITIDO
    # también abarca países vecinos). Devuelve (departamento, provincia) para no repetir la consulta, o None si GAUL
    # no respondió: en ese caso queda solo el control del rectángulo, para que una falla de Earth Engine no frene la demo.
    # Dentro de la zona no se llama: ahí todo es Argentina.
    try:
        depto, prov, pais = ubicacion_de(lon, lat)
    except Exception:
        return None
    if pais not in PAISES_ACEPTADOS:
        donde = f'en {pais}' if pais else 'en el agua (mar o río), fuera de cualquier lote'
        raise HTTPException(422, f'Ese punto está {donde}. La página solo analiza lotes de Argentina: las 27 fechas van '
                                 'de septiembre a mayo y el modelo se armó con datos argentinos.')
    return depto, prov


def asegurar_ee():
    if not EE['listo']:
        iniciar_ee()
        if not EE['listo']:
            raise HTTPException(503, EE['error'] or 'Earth Engine no está disponible.')


@app.get('/cobertura')
def ver_cobertura(request: Request,
                  lat: float = Query(..., description='Latitud del centro del lote (grados)'),
                  lon: float = Query(..., description='Longitud del centro del lote (grados)'),
                  campana: str = Query(..., description='Campaña, por ejemplo 2025_26')):
    # Para probar el filtro desde el navegador: muestra qué ve Dynamic World y qué decidiría, sin bajar la serie.
    validar(lat, lon, campana)
    asegurar_ee()
    controlar_limite(ip_de(request))
    x0, y0 = grilla(lon, lat)
    with _simultaneos:
        if not dentro_de(REGION_LIMITES, lat, lon):
            controlar_pais(lat, lon)
        prob, es_lote, motivo, pct = revisar_cobertura(campana, x0, y0)
    return {'campana': campana, 'lat': lat, 'lon': lon, 'filtro_activo': filtro_activo(),
            'fuera_de_zona': not dentro_de(REGION_LIMITES, lat, lon), 'es_lote': es_lote, 'pct_no_agricola': pct, 'motivo': motivo,
            'mensaje': None if es_lote else mensaje_no_lote(campana, motivo, pct),
            'probabilidades': None if prob is None else {k: round(v, 3) for k, v in prob.items()}}


@app.get('/serie')
def serie(request: Request,
          lat: float = Query(..., description='Latitud del centro del lote (grados)'),
          lon: float = Query(..., description='Longitud del centro del lote (grados)'),
          campana: str = Query(..., description='Campaña, por ejemplo 2025_26')):
    validar(lat, lon, campana)
    x0, y0 = grilla(lon, lat)
    clave = (campana, x0, y0)
    with _cache_lock:
        if clave in _cache:
            _cache.move_to_end(clave)
            return {**_cache[clave], 'en_cache': True}
    asegurar_ee()
    controlar_limite(ip_de(request))
    t0 = time.time()
    prob = None
    with _simultaneos:
        # el país se controla primero: si el punto no es de Argentina, no se baja nada
        ubicacion = None if dentro_de(REGION_LIMITES, lat, lon) else controlar_pais(lat, lon)
        if filtro_activo():
            prob, es_lote, motivo, pct = revisar_cobertura(campana, x0, y0)
            if not es_lote:
                raise HTTPException(422, mensaje_no_lote(campana, motivo, pct))
        try:
            crudo = bajar_crudo(campana, x0, y0)
        except RuntimeError as e:
            raise HTTPException(502, str(e))
        depto, prov = ubicacion if ubicacion else departamento_de(lon, lat)
    relleno = (crudo == 0).all(axis=-1).mean(axis=(1, 2))                    # fracción de píxeles sin dato por período
    X, sin_dato = interpolar(crudo[None])
    X = X[0]
    res = {
        'campana': campana, 'lat': lat, 'lon': lon,
        'fuera_de_zona': not dentro_de(REGION_LIMITES, lat, lon),     # fuera de donde se entrenó y evaluó el modelo
        'forma': list(X.shape), 'dtype': 'uint16',
        'bandas_b64': base64.b64encode(X.astype('<u2').tobytes()).decode(),   # (T, H, W, C), little-endian
        'relleno_por_periodo': [round(float(v), 4) for v in relleno],
        'pct_interpolado': round(float(100 * relleno.mean()), 1),
        'pixeles_sin_dato': int(sin_dato.sum()),
        'esquinas': esquinas(x0, y0),
        'departamento': depto, 'provincia': prov,
        'cobertura': None if prob is None else {k: round(v, 3) for k, v in prob.items()},
        'segundos': round(time.time() - t0, 1),
    }
    with _cache_lock:
        _cache[clave] = res
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)
    return {**res, 'en_cache': False}
