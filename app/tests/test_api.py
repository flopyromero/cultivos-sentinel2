# Tests de la API. Usan app/datos_app (los 12 modelos y los 192 lotes de ejemplo que exporta el notebook 05,
# incluidos en el repositorio) y comparan contra las predicciones guardadas por los notebooks en resultados/.
# Correr desde app/:
#     pytest -q
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
DATOS = os.environ.get('DATOS_APP', os.path.join(AQUI, '..', 'datos_app'))
RESULTADOS = os.path.join(AQUI, '..', '..', 'resultados')
if not os.path.exists(os.path.join(DATOS, 'modelos', 'indice.json')):
    pytest.skip(f'No está la carpeta datos_app en {DATOS} (la arma el notebook 05)', allow_module_level=True)
os.environ['DATOS_APP'] = DATOS

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

cliente = TestClient(api.app)


def npz(**arreglos):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arreglos)
    return buf.getvalue()


@pytest.fixture(scope='module')
def por_id():
    # Los 192 lotes de ejemplo, cada uno con el modelo que no vio su campaña
    return {e['id']: cliente.post('/predict', json={'id': e['id']}).json() for e in cliente.get('/ejemplos').json()}


def test_salud():
    r = cliente.get('/salud').json()
    assert r['ok'] and len(r['campanas']) == 6 and r['lotes_de_ejemplo'] == 192


def test_coincide_con_los_notebooks(por_id):
    # La clase con la campaña completa tiene que ser la que guardó el notebook 03 (año afuera) en al menos el 99,5 %
    # de los lotes, y el rinde estimado, prácticamente el mismo.
    p = pd.read_csv(os.path.join(RESULTADOS, '03v2', 'predicciones.csv')).set_index('id')
    p4 = pd.read_csv(os.path.join(RESULTADOS, '04', 'predicciones.csv')).set_index('id')
    ids = list(por_id)
    clase = np.mean([por_id[i]['completa']['cultivo'] == p.loc[i, 'anio | cultivo | CNN + LSTM | rinde_sembrado'] for i in ids])
    assert clase >= 0.995, f'misma clase en el {100 * clase:.1f} % de los lotes'
    for modo, col, tabla in [('completa', 'anio | rinde | CNN + LSTM | rinde_sembrado', p),
                             ('enero', 'anio | rinde | CNN + LSTM, hasta ene', p4)]:
        dif = [abs(por_id[i][modo]['rinde_kg_ha'] - tabla.loc[i, col]) for i in ids
               if por_id[i][modo]['rinde_kg_ha'] is not None and not np.isnan(tabla.loc[i, col])]
        assert len(dif) > 50 and max(dif) < 25, f'{modo}: diferencia máxima de rinde {max(dif):.1f} kg/ha'


def test_archivo_da_lo_mismo_que_el_id(por_id):
    i, (campana, bandas) = next(iter(api.series().items()))
    r = cliente.post('/predict', files={'archivo': ('lote.npz', npz(bandas=bandas, campana=campana, id=i))}).json()
    for modo in ('enero', 'completa'):
        assert r[modo] == por_id[i][modo]
    assert 'aviso' not in r


def test_sin_campana_avisa():
    _, (_, bandas) = next(iter(api.series().items()))
    r = cliente.post('/predict', files={'archivo': ('lote.npz', npz(bandas=bandas))}).json()
    assert r['campana_modelo'] == 'promedio de los 6 modelos' and 'optimista' in r['aviso']


def test_rechaza_series_con_huecos():
    _, (campana, bandas) = next(iter(api.series().items()))
    con_hueco = bandas.copy()
    con_hueco[5] = 0
    r = cliente.post('/predict', files={'archivo': ('lote.npz', npz(bandas=con_hueco, campana=campana))})
    assert r.status_code == 422 and 'sin dato' in r.json()['detail']


def test_rechaza_forma_y_tipo_incorrectos():
    r = cliente.post('/predict', files={'archivo': ('lote.npz', npz(bandas=np.ones((9, 16, 16, 10), np.uint16)))})
    assert r.status_code == 422
    r = cliente.post('/predict', files={'archivo': ('lote.npz', npz(bandas=np.ones((27, 16, 16, 10), np.float32)))})
    assert r.status_code == 422


def test_rechaza_enteros_que_no_son_uint16():
    _, (campana, bandas) = next(iter(api.series().items()))
    negativas = bandas.astype(np.int16)
    negativas[0, 0, 0, 0] = -5
    r = cliente.post('/predict', files={'archivo': ('lote.npz', npz(bandas=negativas, campana=campana))})
    assert r.status_code == 422 and 'uint16' in r.json()['detail']


@pytest.mark.parametrize('cuerpo', ['{"id": ', '{}', '[1, 2]', '{"id": "abc"}'])
def test_json_mal_formado_da_422(cuerpo):
    r = cliente.post('/predict', content=cuerpo, headers={'Content-Type': 'application/json'})
    assert r.status_code == 422


def test_id_inexistente():
    assert cliente.post('/predict', json={'id': -1}).status_code == 404
