# Servidor de "Probalo en tu lote"

Recibe un punto y una campaña, baja de Earth Engine la serie de Sentinel-2 de ese lugar preparada **igual que en el
notebook 02** (Cloud Score+ ≥ 0,6, mediana cada 10 días, 27 períodos de septiembre a mayo, parche de 16 × 16 píxeles
de 10 m en UTM 20S, huecos interpolados en el tiempo) y la devuelve. El modelo no corre acá: corre en el navegador, en
la página del Space, con los mismos modelos ONNX y la misma validación que los lotes de ejemplo.

| Ruta | Qué hace |
|---|---|
| `GET /salud` | Estado del servidor y de la conexión con Earth Engine |
| `GET /serie?lat=-33.04&lon=-61.17&campana=2025_26` | Serie `(27, 16, 16, 10)` en `uint16`, en base64, con el porcentaje interpolado, las esquinas del parche y el departamento |
| `GET /cobertura?lat=-33.04&lon=-61.17&campana=2025_26` | Para probar el filtro: qué ve Dynamic World en el parche y si lo consideraría un lote |

**Filtro de cobertura** (`cobertura.py`). La red siempre elige una de sus 4 clases, aunque el punto sea un pueblo o un
río. Antes de bajar la serie, el servidor mira Dynamic World (cobertura del suelo de Google, Sentinel-2, 10 m) en el
parche durante la campaña: si más de la mitad es construido, agua, árboles o bañado, responde `422` con un mensaje que
la página muestra tal cual, y no clasifica. Pasto, suelo desnudo o arbustos no se rechazan (un barbecho o una pastura
pueden verse así). Si Dynamic World falla, el pedido sigue como antes. Se apaga con la variable `FILTRO_COBERTURA=0`.

Acepta cualquier punto de Argentina (rectángulo aproximado: longitud −73,6 a −53,6, latitud −55,1 a −21,7) y las
campañas 2019/20 a 2025/26. Fuera de la zona del trabajo (longitud −63,6 a −59,5, latitud −34,3 a −31,4) la serie se
arma igual, pero la respuesta trae `fuera_de_zona: true` y la página avisa que ahí el modelo no se entrenó ni se evaluó
(y no muestra el rinde). El rectángulo de Argentina también abarca partes de los países vecinos: por eso, fuera de la zona del trabajo, el servidor confirma el país con FAO GAUL (la misma fuente de límites que usa el 02) antes de bajar las imágenes, y aprovecha esa consulta para el departamento. Si el punto cae en otro país o en el agua responde `422`; si GAUL no responde, queda solo el control del rectángulo para no frenar la demo. Dentro de la zona no se consulta. Fuera del rectángulo también responde `422`: las 27 fechas van de septiembre a mayo, el verano del
hemisferio sur. Para todos los puntos el parche se arma en UTM 20S, como en el 02; lejos de su meridiano central se
deforma muy poco (menos del 2 % de tamaño y unos grados de giro, que no importan en un parche homogéneo). Guarda en memoria los últimos 300 parches y limita los pedidos nuevos por conexión.

## Correrlo

Necesita una cuenta de servicio de Google Cloud con los roles *Lector de recursos de Earth Engine* y *Consumidor de
Service Usage*, en un proyecto registrado en Earth Engine. Su clave (el `.json`) va en la variable `EE_CLAVE`:
**nunca se sube al repositorio**.

```bash
pip install -r requirements.txt
EE_CLAVE="$(cat clave.json)" uvicorn main:app --reload
pytest -q tests        # no necesitan Earth Engine
```

En Render (plan gratuito): *Root Directory* `servidor`, *Build Command* `pip install -r requirements.txt`,
*Start Command* `uvicorn main:app --host 0.0.0.0 --port $PORT` y la variable secreta `EE_CLAVE`. El servicio gratuito
se apaga a los 15 minutos sin uso; la página lo despierta al abrirse.
