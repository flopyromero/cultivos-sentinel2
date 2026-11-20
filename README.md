# Clasificación de cultivos y estimación de rinde con Sentinel-2 y CNN + LSTM

Trabajo final de **Inteligencia Artificial y Aprendizaje Automático II** · Licenciatura en Ciencias de Datos · UCA Rosario · 2026

**Integrantes:** María Florencia Romero, Juan Ignacio Flores y Sol Mansilla
**Docentes:** Javier Fornari y Antonio Rial

- 🌾 **Página con el modelo funcionando en vivo:** https://huggingface.co/spaces/Final-IA/Final_IA_II
- 📄 **Informe completo:** [`informe/Informe_TP_final_IA2.pdf`](informe/Informe_TP_final_IA2.pdf)

## Qué hace

A partir de la serie de imágenes Sentinel-2 de un lote a lo largo de la campaña, una red **CNN + LSTM** clasifica el cultivo (**maíz, soja de primera, soja de segunda u otros**) y estima el **rinde** del departamento, en la zona núcleo pampeana (campañas 2019/20 a 2024/25).

La propuesta de valor es **adelantar el mapa del INTA**: viendo imágenes **solo hasta enero**, y en una campaña que nunca vio, la red reconoce el cultivo de cada lote. El Mapa Nacional de Cultivos del INTA se publica cerca de 10 meses después.

## Datos

| Fuente | Qué aporta |
|---|---|
| Mapa Nacional de Cultivos (INTA), 6 campañas | Etiquetas. Soja de 1ra y de 2da se separan cruzando el mapa de verano con el de invierno |
| Sentinel-2 nivel 2A (Google Earth Engine) | 27 imágenes por campaña (una cada 10 días, de septiembre a mayo), 10 bandas + NDVI + NDRE, nubes enmascaradas con Cloud Score+ |
| Estimaciones agrícolas (MAGyP) | Rinde por departamento, sobre superficie sembrada |
| ERA5-Land | Lluvia y temperaturas por período (notebook 04) |

Dataset final: **18.692 parches** de 16 × 16 píxeles (160 × 160 m) en 37 departamentos, de zonas puras de unas 10 ha, balanceados por clase.

Los datos no están en este repositorio porque son pesados: los notebooks 01 y 02 los descargan y arman desde las fuentes.

## Modelo

- **CNN por fecha** (`TimeDistributed`): dos bloques de convoluciones 3 × 3 con *max pooling* y una convolución final con *global average pooling* resumen cada imagen en 64 valores.
- **LSTM** de 64 unidades sobre la secuencia de 27 fechas.
- **Dos salidas:** el cultivo (softmax, 4 clases) y el rinde (regresión).
- 76.133 parámetros. La versión "hasta enero" ve solo los primeros 15 períodos.

Se comparó contra un **Random Forest** sobre el promedio de cada canal en cada período, contra referencias simples (promedio histórico del cultivo y del departamento) y contra variantes que cambian una sola cosa (LSTM sola, CNN sin LSTM, imágenes mensuales, 6 bandas, hasta enero, con clima).

## Evaluación

Dos esquemas que miden cosas distintas:

- **Por departamento (5 particiones):** ¿funciona en lugares que el modelo nunca vio?
- **Dejando un año afuera (6 particiones):** ¿funciona en una campaña que el modelo nunca vio? Es la prueba más parecida al uso real.

## Resultados principales

| | Por departamento | Año afuera |
|---|---|---|
| CNN + LSTM, campaña completa (F1 macro) | **0,947** | **0,919** |
| Random Forest, campaña completa | 0,936 | 0,894 |
| CNN + LSTM, **solo hasta enero** | **0,933** | **0,912** |
| Random Forest, solo hasta enero | 0,903 | 0,864 |

- **Pronóstico temprano:** en una campaña nueva, la red en enero ya iguala al Random Forest con la campaña completa, y le gana al Random Forest de enero en 5 de las 6 campañas (la excepción es 2024/25).
- **Qué aporta cada parte:** lo que distingue a los cultivos es la serie de fechas. Ni la forma del lote ni la recurrencia de la LSTM mostraron una mejora mayor al ruido. Las bandas extra (borde rojo y B8A) sí, chica pero consistente.
- **Rinde:** en una campaña nueva, la red (MAE 1.022 kg/ha) empata con el promedio histórico del departamento (1.032 kg/ha). Aprende cuánto rinde cada lugar, pero no anticipa cuánto se aparta un año de lo normal (la sequía de 2022/23 es el caso extremo).
- **Clima:** mejora el rinde dentro de una campaña conocida, pero con solo seis campañas empeora el pronóstico de un año nuevo en 4 de 6.

## Notebooks

Se corren en orden en Google Colab, con los datos en una carpeta de Google Drive (`MyDrive/TP_final_IA2`).

| Notebook | Qué hace | Entorno |
|---|---|---|
| `01_mapas_inta_etiquetas_v2` | Descarga el MNC del INTA, traduce las leyendas y arma las etiquetas de 4 clases | CPU |
| `02_dataset_sentinel2_v2` | Elige los puntos y descarga los parches Sentinel-2, el rinde y el clima | CPU + Earth Engine |
| `03v2_a1_por_departamento` | Redes por departamento (5 variantes × 5 particiones) | GPU |
| `03v2_a2_por_departamento_hasta_ene` | Redes "hasta enero" por departamento | GPU |
| `03v2_b_anio_afuera` | Redes dejando un año afuera | GPU |
| `03v2_resultados` | Carga todas las corridas y arma tablas y gráficos | GPU (sin entrenar) |
| `04_modelo_con_clima` | Variantes con clima y redes "hasta enero" dejando un año afuera | GPU |
| `05_exportar_app` | Exporta los datos y los 12 modelos para la página y la API | CPU |
| `06_exportar_onnx` | Convierte los 12 modelos a ONNX y verifica que dan lo mismo que en Keras | CPU |

El entrenamiento suma más de 10 horas de GPU, así que el notebook 03 se dividió para correr **en paralelo desde varias cuentas de Colab**. Cada entrenamiento se guarda en Drive con una *huella* (configuración + código + particiones): al volver a correr, lo que ya está hecho se carga en segundos, y la variable `PARTE` elige qué esquema corre cada sesión.

## La página

El servicio está publicado gratis en Hugging Face Spaces: https://huggingface.co/spaces/Final-IA/Final_IA_II

- **Mapa:** lo que dijo la red en enero contra el mapa del INTA, los errores y los casos dudosos. Cada punto muestra su curva de NDVI, las probabilidades de la red y el rinde de su departamento.
- **El modelo:** métricas por campaña contra el Random Forest, limitaciones y para qué no usarlo.
- **Probá la API:** se sube un `.npz` con las 10 bandas crudas de un parche (27 × 16 × 16 × 10, uint16) o se elige un lote de ejemplo, y la red lo clasifica en ese momento.

Como Hugging Face cobra los Spaces con servidor, el modelo corre **en el navegador** de quien visita la página: los modelos se convirtieron a ONNX (notebook 06) y se ejecutan con *onnxruntime-web*, con el mismo preprocesamiento que en Python. La misma lógica está como API con FastAPI (`POST /predict` y `/docs`) para correr en una computadora.

## Limitaciones

- Puntos de muestra en zonas puras: no evalúa bordes de lote ni lotes mezclados, y la muestra balanceada no sirve para estimar superficie.
- Se mide la coincidencia con el mapa del INTA, que tiene sus propios errores (más en la sequía de 2022/23).
- El rinde es por departamento: para un lote, la estimación es en la práctica el rinde típico de su zona.
- Seis campañas son pocas para aprender cómo un año anormal afecta el rinde.

## Fuentes

- INTA, *Mapa Nacional de Cultivos*, campañas 2019/20 a 2024/25 (Zenodo).
- ESA, *Copernicus Sentinel-2* nivel 2A, vía Google Earth Engine.
- MAGyP, *Estimaciones agrícolas* por departamento (datos.gob.ar).
- Muñoz-Sabater et al. (2021), *ERA5-Land*.

La lista completa de referencias está en el informe.
