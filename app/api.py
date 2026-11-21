# API del TP final de IA II (UCA Rosario, 2026): sirve los 12 modelos de "año afuera" con FastAPI.
#
# Correr desde la carpeta app/ (usa app/datos_app: los 12 modelos y los 192 lotes de ejemplo del notebook 05):
#     pip install -r ../requirements.txt
#     uvicorn api:app --reload
# Documentación interactiva en http://127.0.0.1:8000/docs
#
# POST /predict acepta:
#   - JSON {"id": 1234}: un lote de ejemplo (ejemplos/series.npz); se usa el modelo que no vio su campaña.
#   - multipart con "archivo" (.npz con "bandas" de forma (27, 16, 16, 10) en uint16) y "campana" opcional.
import io
import os

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ValidationError

from inferencia import BANDAS_S2, Modelos, SerieInvalida, validar_serie

DATOS = os.environ.get('DATOS_APP', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datos_app'))
AVISO_SIN_CAMPANA = ('El archivo no dice de qué campaña es, así que se promediaron los 6 modelos. Si el lote es de una '
                     'de las campañas del trabajo (2019/20 a 2024/25), 5 de esos 6 modelos lo vieron al entrenar y el '
                     'resultado es optimista: pasá su campaña para usar el modelo que no la vio. Para un lote de una '
                     'campaña nueva, el promedio es lo correcto.')

app = FastAPI(
    title='La red en enero contra el mapa del INTA',
    description='Clasifica el cultivo (maíz, soja de 1ra, soja de 2da u otros) y estima el rinde del departamento a '
                'partir de la serie Sentinel-2 de un parche de 16 × 16 píxeles, con el modelo hasta enero y con la '
                'campaña completa. Evaluado sobre puntos de muestra en lotes puros.',
    version='1.0',
)

_cache = {}


def modelos():
    if 'modelos' not in _cache:
        _cache['modelos'] = Modelos(DATOS)
    return _cache['modelos']


def series():
    # id -> (campaña, bandas) de los lotes de ejemplo
    if 'series' not in _cache:
        z = np.load(os.path.join(DATOS, 'ejemplos', 'series.npz'), allow_pickle=True)
        _cache['series'] = {int(i): (str(c), b) for i, c, b in zip(z['id'], z['campana'], z['bandas'])}
    return _cache['series']


def correr(bandas, campana):
    if campana and campana not in modelos().campanas:
        raise HTTPException(422, f'No hay modelo para la campaña {campana}. Campañas: {modelos().campanas}')
    try:
        r = modelos().predecir(bandas, campana)
    except SerieInvalida as e:
        raise HTTPException(422, str(e))
    if not campana:
        r['aviso'] = AVISO_SIN_CAMPANA
    return r


class PedidoPorId(BaseModel):
    id: int


@app.get('/salud')
def salud():
    return {'ok': True, 'campanas': modelos().campanas, 'lotes_de_ejemplo': len(series())}


@app.get('/ejemplos')
def ejemplos():
    return [{'id': i, 'campana': c} for i, (c, _) in sorted(series().items())]


@app.post('/predict/id')
def predecir_por_id(pedido: PedidoPorId):
    if pedido.id not in series():
        raise HTTPException(404, f'No hay un lote de ejemplo con id {pedido.id}. Ver GET /ejemplos.')
    campana, bandas = series()[pedido.id]
    return {'id': pedido.id, **correr(bandas, campana)}


@app.post('/predict/archivo')
async def predecir_archivo(archivo: UploadFile = File(...), campana: str | None = Form(None)):
    try:
        z = np.load(io.BytesIO(await archivo.read()), allow_pickle=False)
        bandas = z['bandas']
    except Exception:
        raise HTTPException(422, 'El archivo tiene que ser un .npz con un arreglo "bandas" de forma (27, 16, 16, 10).')
    if 'bandas_orden' in z.files and [str(b) for b in z['bandas_orden']] != BANDAS_S2:
        raise HTTPException(422, f'Las bandas tienen que venir en este orden: {BANDAS_S2}.')
    if bandas.ndim == 5 and bandas.shape[0] == 1:
        bandas = bandas[0]
    try:
        validar_serie(bandas)
    except SerieInvalida as e:
        raise HTTPException(422, str(e))
    campana = campana or (str(z['campana']) if 'campana' in z.files else None)
    salida = correr(bandas, campana)
    if 'id' in z.files:
        salida['id'] = int(z['id'])
    return salida


@app.post('/predict', summary='Por id (JSON) o con un archivo .npz (multipart)')
async def predecir(request: Request):
    # Una sola dirección para los dos casos, como la usa la página
    if request.headers.get('content-type', '').startswith('application/json'):
        try:
            pedido = PedidoPorId.model_validate(await request.json())
        except (ValueError, ValidationError):
            raise HTTPException(422, 'Se espera un JSON de la forma {"id": 1234}.')
        return predecir_por_id(pedido)
    form = await request.form()
    if 'archivo' not in form:
        raise HTTPException(422, 'Mandá {"id": ...} en JSON o un "archivo" .npz en multipart.')
    return await predecir_archivo(form['archivo'], form.get('campana') or None)
