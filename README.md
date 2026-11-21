# Clasificación de cultivos y estimación de rinde con Sentinel-2 y CNN + LSTM

Trabajo final de **Inteligencia Artificial y Aprendizaje Automático II** · Licenciatura en Ciencias de Datos · UCA Rosario · 2026

**Integrantes:** María Florencia Romero, Juan Ignacio Flores y Sol Mansilla
**Docentes:** Javier Fornari y Antonio Rial

- 🌾 **Página con el modelo funcionando en vivo:** https://huggingface.co/spaces/Final-IA/Final_IA_II
- 📄 **Informe completo:** [`informe/Informe_TP_final_IA2.pdf`](informe/Informe_TP_final_IA2.pdf)

## Qué hace

A partir de la serie de imágenes Sentinel-2 de un parche de 160 × 160 m a lo largo de la campaña, una red **CNN + LSTM** clasifica el cultivo (**maíz, soja de primera, soja de segunda u otros**) y estima el **rinde** del departamento, en la zona núcleo pampeana (campañas 2019/20 a 2024/25).

La propuesta de valor: **sobre puntos de muestra en lotes puros, la red anticipa en enero la clase que después publica el INTA**, en una campaña que nunca vio. El Mapa Nacional de Cultivos del INTA llega unos 10 meses después de enero (el de 2023/24 se publicó en Zenodo el 31 de octubre de 2024).

**Qué no medimos:** un mapa completo. Los puntos se eligieron en zonas puras de unas 10 ha, con la muestra balanceada por clase y sin ríos ni ciudades. Un mapa completo tendría bordes de lote, lotes mezclados y la proporción real de cada clase, y es esperable un F1 más bajo.

## Datos

| Fuente | Qué aporta |
|---|---|
| Mapa Nacional de Cultivos (INTA), 6 campañas | Etiquetas. Soja de 1ra y de 2da se separan cruzando el mapa de verano con el de invierno |
| Sentinel-2 nivel 2A (Google Earth Engine) | 27 imágenes por campaña (una cada 10 días, de septiembre a mayo), 10 bandas + NDVI + NDRE, nubes enmascaradas con Cloud Score+ |
| Estimaciones agrícolas (MAGyP) | Rinde por departamento, sobre superficie sembrada |
| ERA5-Land | Lluvia y temperaturas por período (notebook 04) |

Dataset final: **18.692 parches** de 16 × 16 píxeles (160 × 160 m) en 37 departamentos, balanceados por clase. Los datos no están en el repositorio porque son pesados: los notebooks 01 y 02 los descargan y arman desde las fuentes (ver *Cómo reproducir*).

## Modelo

- **CNN por fecha** (`TimeDistributed`): dos bloques de convoluciones 3 × 3 con *max pooling* y una convolución final con *global average pooling* resumen cada imagen en 64 valores.
- **LSTM** de 64 unidades sobre la secuencia de 27 fechas.
- **Dos salidas:** el cultivo (softmax, 4 clases) y el rinde (regresión).
- 76.133 parámetros. La versión "hasta enero" ve solo los primeros 15 períodos.

Se comparó contra un **Random Forest** sobre el promedio de cada canal en cada período, contra referencias simples (promedio histórico del cultivo y del departamento) y contra variantes que cambian una sola cosa (LSTM sola, CNN sin LSTM, imágenes mensuales, 6 bandas, hasta enero, con clima).

## Evaluación

- **Por departamento (5 particiones):** ¿funciona en lugares que el modelo nunca vio?
- **Dejando un año afuera (6 particiones):** ¿funciona en una campaña que el modelo nunca vio? Es la prueba más parecida al uso real.

## Resultados principales

F1 macro sobre los puntos de muestra (coincidencia con el mapa del INTA):

| | Por departamento | Año afuera |
|---|---|---|
| CNN + LSTM, campaña completa | **0,947** | **0,919** |
| Random Forest, campaña completa | 0,936 | 0,894 |
| CNN + LSTM, **solo hasta enero** | **0,933** | **0,912** |
| Random Forest, solo hasta enero | 0,903 | 0,864 |

- **Pronóstico temprano:** en una campaña nueva, la red en enero ya iguala al Random Forest con la campaña completa, y le gana al Random Forest de enero en 5 de las 6 campañas (la excepción es 2024/25).
- **Qué aporta cada parte:** lo que distingue a los cultivos es la serie de fechas. Ni la forma del lote ni la recurrencia de la LSTM mostraron una mejora mayor al ruido. Las bandas extra (borde rojo y B8A) sí: chica pero en las 5 particiones.
- **Rinde:** en una campaña nueva, la red (MAE 1.022 kg/ha) empata con el promedio histórico del departamento (1.032 kg/ha). Aprende cuánto rinde cada lugar, pero no anticipa cuánto se aparta un año de lo normal (la sequía de 2022/23 es el caso extremo).
- **Clima:** mejora el rinde dentro de una campaña conocida, pero con solo seis campañas empeora el pronóstico de un año nuevo en 4 de 6.

Todas las tablas se pueden verificar con los archivos de [`resultados/`](#resultados).

## Estructura

```
notebooks/     01 a 06, en el orden en que se corren
resultados/    tablas, particiones y predicciones guardadas por los notebooks 03, 04 y 05
app/           API con FastAPI, inferencia.py, tests y el código de la página
informe/       informe en PDF
requirements.txt
LICENSE
```

## Notebooks

Se corren en orden en Google Colab, con los datos en una carpeta de Google Drive (`MyDrive/TP_final_IA2`).

| Notebook | Qué hace | Entorno |
|---|---|---|
| `01_mapas_inta_etiquetas_v2` | Descarga el MNC del INTA, traduce las leyendas y arma las etiquetas de 4 clases | CPU |
| `02_dataset_sentinel2_v2` | Elige los puntos y descarga los parches Sentinel-2, el rinde y el clima | CPU + Earth Engine |
| `03v2_a1_por_departamento` | Registro de ejecución: redes por departamento | GPU |
| `03v2_a2_por_departamento_hasta_ene` | Registro de ejecución: redes "hasta enero" por departamento | GPU |
| `03v2_b_anio_afuera` | Registro de ejecución: redes dejando un año afuera | GPU |
| `03v2_resultados` | Carga todas las corridas del 03 y arma tablas, gráficos y conclusiones | GPU (sin entrenar) |
| `04_modelo_con_clima` | Variantes con clima y redes "hasta enero" dejando un año afuera | GPU |
| `05_exportar_app` | Exporta los datos y los 12 modelos para la página y la API, y verifica que reproducen las predicciones guardadas | CPU |
| `06_exportar_onnx` | Convierte los 12 modelos a ONNX para la página y verifica que dan lo mismo que en Keras | CPU |

El entrenamiento suma más de 10 horas de GPU, así que el 03 se dividió en partes (a1, a2 y b) para correrlas **en paralelo desde varias cuentas de Colab**; sus resultados se juntan en `03v2_resultados`.

## Cómo reproducir

1. **Drive:** crear la carpeta `MyDrive/TP_final_IA2` (o un acceso directo a ella). Todos los notebooks leen y escriben ahí.
2. **Earth Engine (notebook 02):** hace falta una cuenta de Google Earth Engine con un proyecto de Google Cloud habilitado. En el 02, cambiar `ee.Initialize(project='tp-cultivos-ia2')` por el proyecto propio.
3. **Entrenamiento (03 y 04):** con GPU (T4 o L4). La variable `PARTE` elige qué esquema corre cada sesión (`'depto'`, `'anio'` o `'todo'`).
4. **Retoma automática:** cada entrenamiento se guarda en Drive con una *huella* (configuración + código del modelo + ids de las particiones); al volver a correr, lo que ya está hecho se carga en segundos. **Limitación:** la huella no incluye el contenido del dataset. Si se vuelve a generar el dataset con el 02, hay que borrar las carpetas `corridas/` para que se reentrene todo. El notebook 05 verifica, de todos modos, que los modelos exportados reproducen las predicciones guardadas.
5. **Versiones:** las de [`requirements.txt`](requirements.txt) (TensorFlow 2.21.0 y Keras 3.13.2 en Colab).

## Resultados

| Archivo | Qué tiene |
|---|---|
| `resultados/03v2/resultados_por_particion.csv` | Métricas de cada modelo en cada partición, de los dos esquemas (solo imágenes) |
| `resultados/03v2/predicciones.csv` | Predicción de cada modelo para cada parche, en su partición de test |
| `resultados/03v2/particiones_03.csv` | A qué partición pertenece cada parche (las usan el 04 y el 05) |
| `resultados/03v2/tabla_por_departamento.csv` | Tabla general por departamento |
| `resultados/04/resultados_por_particion.csv` | Lo mismo con clima y las redes "hasta enero" dejando un año afuera. Incluye el **Random Forest hasta enero por campaña** (filas `anio`, modelo `Random Forest, hasta ene`) |
| `resultados/04/predicciones.csv` | Rinde predicho por cada modelo del 04 |
| `resultados/04/tabla_depto.csv`, `tabla_anio.csv` | Tablas resumen del 04 |
| `resultados/05/metricas.json` | Métricas por campaña que muestra la página |
| `resultados/05/versiones.json` | Versiones con las que se exportaron los modelos |

## La página y la API

**La página** está publicada gratis en Hugging Face Spaces: https://huggingface.co/spaces/Final-IA/Final_IA_II. Muestra lo que dijo la red en enero contra el mapa del INTA, los errores y los casos dudosos, y permite **correr el modelo en vivo** sobre 192 lotes de ejemplo (32 por campaña, vista "Correr en vivo") o sobre un `.npz` propio. Como Hugging Face cobra los Spaces con servidor, el modelo corre **en el navegador**: los 12 modelos convertidos a ONNX (notebook 06) se ejecutan con *onnxruntime-web*, con el mismo preprocesamiento que `inferencia.py`. Su código está en `app/pagina/`; los datos y los modelos que usa están en el Space.

**La API** (`app/api.py`, FastAPI) usa la carpeta `datos_app` que arma el notebook 05:

```bash
pip install -r requirements.txt
cd app
DATOS_APP=ruta/a/datos_app uvicorn api:app --reload     # documentación en http://127.0.0.1:8000/docs
DATOS_APP=ruta/a/datos_app pytest -q                    # tests
```

- `POST /predict` con `{"id": 1234}` predice un lote de ejemplo con **el modelo que no vio su campaña**.
- `POST /predict` con un archivo `.npz` (`bandas` de forma (27, 16, 16, 10) en uint16, las 10 bandas crudas B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 cada 10 días de septiembre a mayo) y `campana` opcional.
- **La serie tiene que venir preparada como en el notebook 02** (nubes enmascaradas, mediana cada 10 días y huecos interpolados). `inferencia.validar_serie()` rechaza series con píxeles sin dato (0 en todas las bandas) o con más del 1 % de píxeles con alguna banda en 0. La página hace el mismo control.
- **Sin campaña** se promedian los 6 modelos. Si el lote es de una de las campañas del trabajo, 5 de esos 6 modelos lo vieron al entrenar y el resultado es optimista: la API y la página lo avisan. Para un lote de una campaña nueva, el promedio es lo correcto.
- **Tests** (`app/tests/test_api.py`): la API predice los 192 lotes de ejemplo y se compara con lo que guardaron los notebooks en `resultados/` (misma clase en al menos el 99,5 % y el mismo rinde, con una tolerancia de 25 kg/ha). También prueban que un archivo da lo mismo que su id, el aviso sin campaña y que se rechazan series con huecos o con forma o tipo incorrectos.

`app/inferencia.py` es el mismo que exporta el notebook 05, más la validación de la serie.

## Limitaciones

- Puntos de muestra en zonas puras: no evalúa bordes de lote ni lotes mezclados, y la muestra balanceada no sirve para estimar superficie.
- Se mide la coincidencia con el mapa del INTA, que tiene sus propios errores (más en la sequía de 2022/23).
- El rinde es por departamento: para un lote, la estimación es en la práctica el rinde típico de su zona.
- Seis campañas son pocas para aprender cómo un año anormal afecta el rinde.

## Datos: fuentes y licencias

- **INTA, Mapa Nacional de Cultivos**, campañas 2019/20 a 2024/25, publicados en Zenodo bajo **Creative Commons Attribution 4.0 International**.
- **Copernicus Sentinel-2** nivel 2A, vía Google Earth Engine: acceso libre, completo y abierto. *Contains modified Copernicus Sentinel data 2019-2025.*
- **MAGyP, Estimaciones agrícolas** (datos.gob.ar), bajo **Creative Commons Attribution 4.0**.
- **ERA5-Land** (Muñoz-Sabater, 2019; Copernicus Climate Change Service, Climate Data Store), bajo **CC BY 4.0**. *Generated using Copernicus Climate Change Service information 2019-2025.*

La lista completa de referencias está en el informe.

## Licencia

El código de este repositorio se publica bajo la licencia MIT (ver [`LICENSE`](LICENSE)). Los datos conservan la licencia de su fuente.
