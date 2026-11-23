# Tests sin Earth Engine: la descarga se reemplaza por una serie inventada con huecos.
import base64
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402

ORIGEN = 'https://final-ia-final-ia-ii.static.hf.space'
PUNTO = {'lat': -33.0, 'lon': -61.0}


def crudo_falso(campana, x0, y0, intentos=4):
    rng = np.random.default_rng(0)
    x = rng.integers(200, 4000, size=(main.T, main.PATCH, main.PATCH, len(main.BANDAS)), dtype=np.uint16)
    x[3] = 0                     # un período entero nublado
    x[0, :4] = 0                 # huecos al principio (sin dato anterior)
    x[10:14, 5, 5] = 0           # un píxel nublado varios períodos seguidos
    return x


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setattr(main, 'bajar_crudo', crudo_falso)
    monkeypatch.setattr(main, 'departamento_de', lambda lon, lat: ('CASEROS', 'Santa Fe'))
    monkeypatch.setattr(main.cobertura, 'probabilidades', lambda *a, **k: None)   # sin Earth Engine: el filtro se saltea
    monkeypatch.setitem(main.EE, 'listo', True)
    main._cache.clear()
    main._pedidos.clear()
    main._todos.clear()
    with TestClient(main.app) as c:
        main.EE['listo'] = True          # el arranque sin EE_CLAVE lo marca como no listo
        yield c


def decodificar(r):
    return np.frombuffer(base64.b64decode(r['bandas_b64']), '<u2').reshape(r['forma'])


def test_periodos_iguales_al_02():
    p = main.periodos('2025-09-01')
    assert len(p) == 27 and p[0] == ('2025-09-01', '2025-09-11') and p[-1] == ('2026-05-21', '2026-06-01')


def test_serie_interpolada_sin_huecos(cliente):
    r = cliente.get('/serie', params={**PUNTO, 'campana': '2025_26'}).json()
    x = decodificar(r)
    assert x.shape == (27, 16, 16, 10) and x.dtype == np.uint16
    assert not (x == 0).all(axis=-1).any()                       # ningún píxel sin dato: pasa la validación del modelo
    crudo = crudo_falso(None, 0, 0)
    assert np.array_equal(x[5], crudo[5])                        # lo que tenía dato no cambia
    esperado = np.round((crudo[2].astype(np.float32) + crudo[4]) / 2).astype(np.uint16)
    assert np.array_equal(x[3], esperado)                        # período nublado: punto medio entre vecinos
    assert np.array_equal(x[0, :4], crudo[1, :4])                # sin dato anterior: se copia el siguiente
    assert r['relleno_por_periodo'][3] == 1.0 and r['pixeles_sin_dato'] == 0
    assert r['departamento'] == 'CASEROS' and len(r['esquinas']) == 4


def test_esquinas_forman_un_cuadrado_de_160_m(cliente):
    r = cliente.get('/serie', params={**PUNTO, 'campana': '2024_25'}).json()
    (la1, lo1), (_, lo2), _, (la4, _) = r['esquinas']        # sup. izq., sup. der., inf. der., inf. izq.
    assert abs((lo2 - lo1) * 111320 * np.cos(np.radians(la1)) - 160) < 3
    assert abs((la1 - la4) * 110574 - 160) < 3
    lats, lons = zip(*r['esquinas'])
    assert min(lats) < PUNTO['lat'] < max(lats) and min(lons) < PUNTO['lon'] < max(lons)


def test_cache(cliente):
    a = cliente.get('/serie', params={**PUNTO, 'campana': '2023_24'}).json()
    b = cliente.get('/serie', params={'lat': PUNTO['lat'] + 0.00001, 'lon': PUNTO['lon'], 'campana': '2023_24'}).json()
    assert a['en_cache'] is False and b['en_cache'] is True     # el mismo parche de la grilla de 10 m


@pytest.mark.parametrize('params', [{**PUNTO, 'campana': '2026_27'}, {'lat': -38.0, 'lon': -61.0, 'campana': '2025_26'},
                                    {'lat': -33.0, 'lon': -65.0, 'campana': '2025_26'}, {'lat': 'x', 'lon': -61, 'campana': '2025_26'}])
def test_rechaza_pedidos_invalidos(cliente, params):
    assert cliente.get('/serie', params=params).status_code == 422


def test_limite_por_ip(cliente, monkeypatch):
    monkeypatch.setattr(main, 'LIMITE_POR_IP', 2)
    for k in range(2):
        assert cliente.get('/serie', params={'lat': -33 + k / 100, 'lon': -61, 'campana': '2025_26'}).status_code == 200
    assert cliente.get('/serie', params={'lat': -33.5, 'lon': -61, 'campana': '2025_26'}).status_code == 429


def test_sin_clave_responde_503(cliente, monkeypatch):
    monkeypatch.setitem(main.EE, 'listo', False)
    monkeypatch.delenv('EE_CLAVE', raising=False)
    r = cliente.get('/serie', params={**PUNTO, 'campana': '2025_26'})
    assert r.status_code == 503 and 'EE_CLAVE' in r.json()['detail']


def test_cors_solo_la_pagina(cliente):
    ok = cliente.get('/salud', headers={'Origin': ORIGEN})
    otro = cliente.get('/salud', headers={'Origin': 'https://otra-pagina.com'})
    assert ok.headers.get('access-control-allow-origin') == ORIGEN
    assert 'access-control-allow-origin' not in otro.headers


@pytest.mark.parametrize('clave', ['no es json', '{"project_id": "x"}',
                                   '{"client_email": "a@b.iam.gserviceaccount.com", "project_id": "x", "private_key": "SECRETO"}'])
def test_error_de_clave_no_muestra_la_clave(monkeypatch, clave):
    monkeypatch.setenv('EE_CLAVE', clave)
    monkeypatch.setitem(main.EE, 'listo', False)
    main.iniciar_ee()
    assert main.EE['listo'] is False and 'SECRETO' not in main.EE['error'] and 'private_key' not in main.EE['error']


def test_limite_no_se_saltea_inventando_la_ip(cliente, monkeypatch):
    monkeypatch.setattr(main, 'LIMITE_POR_IP', 2)
    real = {'CF-Connecting-IP': '200.1.1.1'}
    for k in range(2):
        falsa = {**real, 'X-Forwarded-For': f'9.9.9.{k}'}         # el cliente inventa otra IP en cada pedido
        assert cliente.get('/serie', params={'lat': -33 + k / 100, 'lon': -61, 'campana': '2025_26'}, headers=falsa).status_code == 200
    r = cliente.get('/serie', params={'lat': -33.5, 'lon': -61, 'campana': '2025_26'}, headers={**real, 'X-Forwarded-For': '8.8.8.8'})
    assert r.status_code == 429


def test_sin_cloudflare_usa_la_ultima_ip_reenviada(cliente):
    r = cliente.get('/salud', headers={'X-Forwarded-For': '6.6.6.6, 190.2.2.2'}).json()
    assert r['tu_ip'] == '190.2.2.2'


def test_limite_total(cliente, monkeypatch):
    monkeypatch.setattr(main, 'LIMITE_TOTAL', 3)
    codigos = [cliente.get('/serie', params={'lat': -33 + k / 100, 'lon': -61, 'campana': '2025_26'},
                           headers={'CF-Connecting-IP': f'10.0.0.{k}'}).status_code for k in range(4)]
    assert codigos == [200, 200, 200, 429]


# ------------------------------------------------------------------ filtro de cobertura (Dynamic World)
def prob_dw(**dominante):
    base = {c: 0.02 for c in main.cobertura.CLASES_DW}
    base.update(dominante)
    return base


def test_decidir_lote_y_no_lote():
    assert main.cobertura.decidir(prob_dw(crops=0.8))[0] is True
    assert main.cobertura.decidir(prob_dw(grass=0.5, bare=0.3))[0] is True        # barbecho o pastura: no se rechaza
    es, motivo, pct = main.cobertura.decidir(prob_dw(built=0.7))
    assert es is False and motivo == 'built' and pct > 50
    assert main.cobertura.decidir(prob_dw(trees=0.4, water=0.3))[1] == 'trees'
    assert main.cobertura.decidir(None) == (True, None, None)                       # sin datos no bloquea


def test_ciudad_se_rechaza_con_mensaje(cliente, monkeypatch):
    monkeypatch.setattr(main.cobertura, 'probabilidades', lambda *a, **k: prob_dw(built=0.8))
    r = cliente.get('/serie', params={**PUNTO, 'campana': '2024_25'})
    assert r.status_code == 422 and 'no parece un lote' in r.json()['detail'] and 'construida' in r.json()['detail']


def test_lote_pasa_y_devuelve_cobertura(cliente, monkeypatch):
    monkeypatch.setattr(main.cobertura, 'probabilidades', lambda *a, **k: prob_dw(crops=0.8))
    r = cliente.get('/serie', params={**PUNTO, 'campana': '2024_25'})
    assert r.status_code == 200 and r.json()['cobertura']['crops'] == 0.8


def test_filtro_apagado_vuelve_al_comportamiento_anterior(cliente, monkeypatch):
    monkeypatch.setenv('FILTRO_COBERTURA', '0')
    llamado = []
    monkeypatch.setattr(main.cobertura, 'probabilidades', lambda *a, **k: llamado.append(1) or prob_dw(built=0.9))
    r = cliente.get('/serie', params={**PUNTO, 'campana': '2024_25'})
    assert r.status_code == 200 and not llamado and r.json()['cobertura'] is None
    assert cliente.get('/salud').json()['filtro_cobertura'] is False


def test_ruta_para_probar_el_filtro(cliente, monkeypatch):
    monkeypatch.setattr(main.cobertura, 'probabilidades', lambda *a, **k: prob_dw(water=0.9))
    r = cliente.get('/cobertura', params={**PUNTO, 'campana': '2025_26'}).json()
    assert r['es_lote'] is False and r['motivo'] == 'water' and 'agua' in r['mensaje']
