# Inferencia de los modelos del TP final de IA II (UCA Rosario, 2026).
#
# Uso:
#     from inferencia import Modelos
#     m = Modelos('datos_app')                      # carga los 12 modelos (tarda unos segundos)
#     r = m.predecir(bandas, campana='2022_23')     # bandas: (27, 16, 16, 10) uint16
#     r = m.predecir(bandas)                        # sin campaña: promedio de los 6 modelos
#
# Ojo con el modo sin campaña: si el lote es de una de las 6 campañas del trabajo, 5 de esos 6 modelos lo vieron
# al entrenar y el resultado es optimista. Para evaluar un lote del dataset, pasar su campaña.
#
# La serie tiene que venir preparada como en el notebook 02: nubes enmascaradas, mediana cada 10 días y huecos
# interpolados. validar_serie() rechaza series con huecos (píxeles en 0), que el modelo respondería sin sentido.
import json, os
import numpy as np
import keras
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

CLASES = ['Maíz', 'Soja 1ra', 'Soja 2da', 'Otros']
BANDAS_S2 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
T_FULL, T_ENE, PATCH = 27, 15, 16


class SerieInvalida(ValueError):
    pass


def validar_serie(bandas, max_ceros=0.01):
    # Rechaza series que no vienen preparadas como en el notebook 02. En los 192 lotes reales de ejemplo no hay
    # ningún píxel-período en 0 en todas las bandas, y como mucho un 0,07 % tiene un 0 en alguna banda.
    x = np.asarray(bandas)
    if x.ndim == 4:
        x = x[None]
    if x.shape[1:] != (T_FULL, PATCH, PATCH, len(BANDAS_S2)):
        raise SerieInvalida(f'Se esperaba (27, 16, 16, 10) por lote y llegó {x.shape[1:]}')
    if x.dtype != np.uint16:
        raise SerieInvalida(f'Se esperaban las bandas crudas en uint16 (valores de 0 a 65535) y llegó {x.dtype}')
    huecos = (x == 0).all(-1)                       # píxel-período sin dato en ninguna banda
    if huecos.any():
        n, t = int(huecos.sum()), sorted(set(np.where(huecos)[1].tolist()))
        raise SerieInvalida(f'La serie tiene {n} píxeles sin dato (0 en todas las bandas) en los períodos {t}. '
                            'El modelo espera la serie preparada como en el notebook 02: nubes enmascaradas, '
                            'mediana cada 10 días y huecos interpolados.')
    frac = float((x == 0).any(-1).mean())
    if frac > max_ceros:
        raise SerieInvalida(f'El {100 * frac:.1f} % de los píxeles tiene alguna banda en 0 (se tolera hasta '
                            f'{100 * max_ceros:.0f} %). Revisar el enmascarado de nubes y la interpolación.')


def preprocesar(bandas):
    # (n, 27, 16, 16, 10) o (27, 16, 16, 10) en uint16 -> (n, 27, 16, 16, 12) float32 sin normalizar.
    # Igual que el notebook 03: reflectancia = valor / 10000, y se agregan NDVI y NDRE.
    x = np.asarray(bandas)
    if x.ndim == 4:
        x = x[None]
    if x.shape[1:] != (T_FULL, PATCH, PATCH, len(BANDAS_S2)):
        raise ValueError(f'Se esperaba (27, 16, 16, 10) por lote y llegó {x.shape[1:]}')
    x = x.astype(np.float32) / 10000.0
    b = {n: i for i, n in enumerate(BANDAS_S2)}
    ndvi = (x[..., b['B8']] - x[..., b['B4']]) / np.maximum(x[..., b['B8']] + x[..., b['B4']], 1e-6)
    ndre = (x[..., b['B8A']] - x[..., b['B5']]) / np.maximum(x[..., b['B8A']] + x[..., b['B5']], 1e-6)
    # el 03 guardaba la serie en float16: se redondea igual para reproducir exactamente lo que vio el modelo
    return np.concatenate([x, ndvi[..., None], ndre[..., None]], axis=-1).astype(np.float16).astype(np.float32)


class Modelos:
    def __init__(self, carpeta):
        self.carpeta = carpeta
        self.indice = json.load(open(os.path.join(carpeta, 'modelos', 'indice.json')))
        self.campanas = sorted(self.indice)
        self._m = {}
        for camp, modos in self.indice.items():
            for modo, info in modos.items():
                modelo = keras.models.load_model(os.path.join(carpeta, 'modelos', info['archivo']), compile=False)
                self._m[(camp, modo)] = (modelo, info)

    def _correr(self, x12, camp, modo):
        modelo, info = self._m[(camp, modo)]
        T = info['periodos']
        mu = np.asarray(info['mu_canales'], np.float32)
        sd = np.asarray(info['sd_canales'], np.float32)
        x = ((x12[:, :T] - mu) / sd).astype(np.float32)
        entrada = {'serie': x} if isinstance(modelo.input, dict) else x
        out = modelo.predict(entrada, verbose=0, batch_size=64)
        prob = np.asarray(out['cultivo'], np.float64)
        rinde = np.asarray(out['rinde'], np.float64).ravel() * info['rinde_desvio'] + info['rinde_media']
        return prob, rinde

    def predecir_lote(self, bandas, campana=None, modo='completa'):
        # Probabilidades (n, 4) y rinde (n,) en kg/ha. modo: 'completa' (27 períodos) o 'enero' (los primeros 15).
        # Con campana=None se promedian los 6 modelos (uno por campaña dejada afuera).
        validar_serie(bandas)
        x12 = preprocesar(bandas)
        camps = [campana] if campana else self.campanas
        for c in camps:
            if (c, modo) not in self._m:
                raise ValueError(f'No hay modelo para campaña={c} y modo={modo}. Campañas: {self.campanas}')
        res = [self._correr(x12, c, modo) for c in camps]
        prob = np.mean([p for p, _ in res], axis=0)
        rinde = np.mean([r for _, r in res], axis=0)
        return prob, rinde

    def predecir(self, bandas, campana=None):
        # Resultado legible para un solo lote, con el modelo en enero y con la campaña completa.
        salida = {'campana_modelo': campana or 'promedio de los 6 modelos'}
        for modo in ('enero', 'completa'):
            prob, rinde = self.predecir_lote(bandas, campana, modo)
            k = int(prob[0].argmax())
            salida[modo] = {
                'cultivo': CLASES[k],
                'probabilidades': {c: round(float(p), 4) for c, p in zip(CLASES, prob[0])},
                # el rinde solo tiene sentido para maíz y soja
                'rinde_kg_ha': None if k == 3 else round(float(rinde[0])),
            }
        return salida
