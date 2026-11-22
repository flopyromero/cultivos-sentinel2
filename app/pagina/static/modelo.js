'use strict';
// El modelo corriendo en el navegador, sin servidor.
// Los 12 modelos de Keras (6 campañas × enero/completa) se convirtieron a ONNX con el notebook 06 y se verificó que dan
// lo mismo que en Keras. Acá se repite exactamente el preprocesamiento de inferencia.py:
// reflectancia = valor / 10000, se agregan NDVI y NDRE, se redondea a float16 (como lo guardó el notebook 03) y se
// normaliza con la media y el desvío de cada canal del entrenamiento de ese modelo.
(function () {
  const ORT_VERSION = '1.22.0';
  // El motor se sirve desde el mismo Space (no depende de internet externo); jsdelivr queda solo como respaldo.
  // dirección completa (la pide el motor); fuera del navegador (tests en Node) no hay document
  const ORT_LOCAL = typeof document !== 'undefined' ? new URL('static/vendor/onnxruntime/', document.baseURI).href : 'static/vendor/onnxruntime/';
  const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const CARPETA = 'onnx/';
  const T_FULL = 27, PATCH = 16, NB = 10, NC = 12;
  const BANDAS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
  const B = Object.fromEntries(BANDAS.map((b, i) => [b, i]));

  let ort = (typeof globalThis !== 'undefined' && globalThis.ort) || null;
  let indice = null;
  const sesiones = {};

  // ---------------------------------------------------------------- lectura del .npz (zip de archivos .npy)
  async function inflar(bytes) {
    const ds = new DecompressionStream('deflate-raw');
    return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer());
  }

  async function leerZip(buf) {
    const dv = new DataView(buf), u8 = new Uint8Array(buf);
    let fin = -1;
    for (let i = buf.byteLength - 22; i >= Math.max(0, buf.byteLength - 66000); i--) {
      if (dv.getUint32(i, true) === 0x06054b50) { fin = i; break; }
    }
    if (fin < 0) throw new Error('El archivo no es un .npz válido (no es un zip).');
    let n = dv.getUint16(fin + 10, true);
    let pos = dv.getUint32(fin + 16, true);
    if (pos === 0xffffffff && dv.getUint32(fin - 20, true) === 0x07064b50) {   // zip64
      const z64 = Number(dv.getBigUint64(fin - 20 + 8, true));
      n = Number(dv.getBigUint64(z64 + 32, true));
      pos = Number(dv.getBigUint64(z64 + 48, true));
    }
    const archivos = {};
    for (let k = 0; k < n; k++) {
      if (dv.getUint32(pos, true) !== 0x02014b50) throw new Error('El .npz está dañado.');
      const metodo = dv.getUint16(pos + 10, true);
      let comp = dv.getUint32(pos + 20, true), sinComp = dv.getUint32(pos + 24, true), local = dv.getUint32(pos + 42, true);
      const ln = dv.getUint16(pos + 28, true), le = dv.getUint16(pos + 30, true), lc = dv.getUint16(pos + 32, true);
      const nombre = new TextDecoder().decode(u8.subarray(pos + 46, pos + 46 + ln));
      for (let e = pos + 46 + ln; e < pos + 46 + ln + le;) {   // campo extra zip64 con los tamaños reales
        const id = dv.getUint16(e, true), largo = dv.getUint16(e + 2, true);
        if (id === 1) {
          let p = e + 4;
          if (sinComp === 0xffffffff) { sinComp = Number(dv.getBigUint64(p, true)); p += 8; }
          if (comp === 0xffffffff) { comp = Number(dv.getBigUint64(p, true)); p += 8; }
          if (local === 0xffffffff) { local = Number(dv.getBigUint64(p, true)); }
        }
        e += 4 + largo;
      }
      const ini = local + 30 + dv.getUint16(local + 26, true) + dv.getUint16(local + 28, true);
      const datos = u8.subarray(ini, ini + comp);
      if (metodo === 0) archivos[nombre] = datos.slice();
      else if (metodo === 8) archivos[nombre] = await inflar(datos);
      else throw new Error(`El .npz usa una compresión que no se puede leer (${metodo}).`);
      pos += 46 + ln + le + lc;
    }
    return archivos;
  }

  function leerNpy(u8) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    if (u8[0] !== 0x93 || String.fromCharCode(...u8.subarray(1, 6)) !== 'NUMPY') throw new Error('Arreglo .npy inválido.');
    const v = u8[6];
    const largo = v === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
    const ini = v === 1 ? 10 : 12;
    const cab = new TextDecoder('latin1').decode(u8.subarray(ini, ini + largo));
    const descr = /'descr':\s*'([^']+)'/.exec(cab)[1];
    if (/'fortran_order':\s*True/.test(cab)) throw new Error('Arreglo en orden Fortran: guardalo con np.ascontiguousarray.');
    const forma = /'shape':\s*\(([^)]*)\)/.exec(cab)[1].split(',').map((s) => s.trim()).filter(Boolean).map(Number);
    const datos = u8.slice(ini + largo);   // copia alineada
    const b = datos.buffer;
    let valores;
    if (descr === '<u2' || descr === '|u2') valores = new Uint16Array(b);
    else if (descr === '<i8') valores = Array.from(new BigInt64Array(b), Number);
    else if (descr === '<i4') valores = new Int32Array(b);
    else if (descr === '<f4') valores = new Float32Array(b);
    else if (descr === '<f8') valores = new Float64Array(b);
    else if (/^<U\d+$/.test(descr)) {
      const ancho = Number(descr.slice(2)), u32 = new Uint32Array(b), cant = u32.length / ancho;
      valores = Array.from({ length: cant }, (_, i) => String.fromCodePoint(...u32.subarray(i * ancho, (i + 1) * ancho)).replace(/\0+$/, ''));
    } else valores = null;   // otros tipos no se usan
    return { descr, forma, valores };
  }

  // indice: para archivos con varios lotes (bandas de forma (n, 27, 16, 16, 10)), cuál leer
  async function leerNpz(buf, indice = null) {
    const arch = await leerZip(buf);
    const a = {};
    for (const [nombre, bytes] of Object.entries(arch)) a[nombre.replace(/\.npy$/, '')] = leerNpy(bytes);
    if (!a.bandas) throw new Error('El .npz no tiene un arreglo llamado "bandas".');
    let { forma, valores, descr } = a.bandas;
    let k = 0;
    if (forma.length === 5) {
      if (forma[0] > 1 && indice === null) throw new Error(`El archivo tiene ${forma[0]} lotes; se espera uno solo, de forma (27, 16, 16, 10).`);
      k = forma[0] > 1 ? indice : 0;
      if (k < 0 || k >= forma[0]) throw new Error('No existe ese lote en el archivo.');
      forma = forma.slice(1);
    }
    if (forma.join(',') !== `${T_FULL},${PATCH},${PATCH},${NB}`) {
      throw new Error(`"bandas" tiene forma (${forma.join(', ')}); se espera (27, 16, 16, 10): 27 períodos, 16 × 16 píxeles, 10 bandas.`);
    }
    if (!(valores instanceof Uint16Array)) throw new Error(`"bandas" es ${descr}; se espera uint16 (las bandas crudas de Sentinel-2).`);
    const tam = T_FULL * PATCH * PATCH * NB;
    valores = valores.subarray(k * tam, (k + 1) * tam);
    const orden = a.bandas_orden?.valores;
    if (orden && orden.join(',') !== BANDAS.join(',')) throw new Error(`Las bandas vienen en otro orden (${orden.join(', ')}); se espera ${BANDAS.join(', ')}.`);
    const de = (x) => (x && x.valores && x.valores.length ? x.valores[x.valores.length > 1 ? k : 0] : null);
    return { bandas: valores, campana: de(a.campana), id: de(a.id) };
  }

  // ---------------------------------------------------------------- preprocesamiento (igual que inferencia.py)
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  // redondea un float32 a la precisión de float16 (al par más cercano, como numpy) y lo devuelve como número
  function aF16(v) {
    if (v === 0 || !Number.isFinite(v)) return v;
    const a = Math.abs(v);
    if (a >= 65520) return Math.sign(v) * Infinity;
    f32[0] = a;
    const e = ((u32[0] >>> 23) & 0xff) - 127;
    const q = 2 ** (Math.max(e, -14) - 10);
    const r = a / q, n = Math.floor(r), d = r - n;
    const m = d > 0.5 || (d === 0.5 && n % 2 === 1) ? n + 1 : n;
    return Math.sign(v) * m * q;
  }

  function preprocesar(bandas) {
    const n = T_FULL * PATCH * PATCH;
    const x = new Float32Array(n * NC);
    const eps = Math.fround(1e-6);
    for (let p = 0; p < n; p++) {
      const o = p * NB, d = p * NC;
      const r = new Array(NB);
      for (let c = 0; c < NB; c++) { r[c] = Math.fround(bandas[o + c] / 10000); x[d + c] = aF16(r[c]); }
      const ndvi = Math.fround(Math.fround(r[B.B8] - r[B.B4]) / Math.max(Math.fround(r[B.B8] + r[B.B4]), eps));
      const ndre = Math.fround(Math.fround(r[B.B8A] - r[B.B5]) / Math.max(Math.fround(r[B.B8A] + r[B.B5]), eps));
      x[d + 10] = aF16(ndvi); x[d + 11] = aF16(ndre);
    }
    return x;
  }

  // La serie tiene que venir preparada como en el notebook 02 (nubes enmascaradas, mediana cada 10 días, huecos
  // interpolados). Igual que validar_serie() de inferencia.py: se rechazan píxeles sin dato.
  function validarSerie(bandas) {
    const npx = T_FULL * PATCH * PATCH;
    let huecos = 0, conCero = 0;
    const periodos = new Set();
    for (let p = 0; p < npx; p++) {
      let ceros = 0;
      for (let c = 0; c < NB; c++) if (bandas[p * NB + c] === 0) ceros++;
      if (ceros === NB) { huecos++; periodos.add(Math.floor(p / (PATCH * PATCH))); }
      if (ceros > 0) conCero++;
    }
    if (huecos) {
      throw new Error(`La serie tiene ${huecos} píxeles sin dato (0 en todas las bandas) en los períodos ${[...periodos].join(', ')}. `
        + 'El modelo espera la serie preparada igual que en el trabajo: nubes enmascaradas, mediana cada 10 días y huecos completados.');
    }
    if (conCero / npx > 0.01) {
      throw new Error(`El ${(100 * conCero / npx).toFixed(1).replace('.', ',')} % de los píxeles tiene alguna banda en 0 `
        + '(se tolera hasta 1 %). Revisar el enmascarado de nubes y la interpolación.');
    }
  }

  // ---------------------------------------------------------------- modelos
  function cargarScript(url) {
    return new Promise((ok, mal) => {
      const s = document.createElement('script');
      s.src = url; s.onload = ok; s.onerror = () => mal(new Error('No se pudo descargar el motor del modelo. Probá recargar la página.'));
      document.head.appendChild(s);
    });
  }

  async function preparar() {
    if (!indice) {
      const r = await fetch(CARPETA + 'indice.json');
      if (!r.ok) throw new Error('Los modelos todavía no están publicados en esta página.');
      indice = await r.json();
    }
    if (!ort) {
      let base = ORT_LOCAL;
      try { await cargarScript(ORT_LOCAL + 'ort.wasm.min.js'); } catch { base = ORT_CDN; await cargarScript(ORT_CDN + 'ort.wasm.min.js'); }
      ort = globalThis.ort;
      if (ort.env?.wasm) ort.env.wasm.wasmPaths = base;
    }
    if (ort.env?.wasm) ort.env.wasm.numThreads = 1;
    return indice;
  }

  async function disponible() {
    try {
      const r = await fetch(CARPETA + 'indice.json');
      if (!r.ok) return false;
      indice = await r.json();
      return true;
    } catch { return false; }
  }

  function sesion(camp, modo) {
    const clave = `${camp}_${modo}`;
    if (!sesiones[clave]) {
      const info = indice.modelos[camp][modo];
      sesiones[clave] = (async () => {
        const r = await fetch(CARPETA + info.archivo);
        if (!r.ok) throw new Error(`No se pudo descargar el modelo ${info.archivo}.`);
        return ort.InferenceSession.create(new Uint8Array(await r.arrayBuffer()), { executionProviders: ['wasm'] });
      })().catch((e) => { delete sesiones[clave]; throw e; });
    }
    return sesiones[clave];
  }

  async function correr(x12, camp, modo) {
    const info = indice.modelos[camp][modo];
    const T = info.periodos, n = T * PATCH * PATCH;
    const mu = info.mu_canales.map(Math.fround), sd = info.sd_canales.map(Math.fround);
    const x = new Float32Array(n * NC);
    for (let i = 0; i < n * NC; i++) { const c = i % NC; x[i] = Math.fround(Math.fround(x12[i] - mu[c]) / sd[c]); }
    const s = await sesion(camp, modo);
    const out = await s.run({ serie: new ort.Tensor('float32', x, [1, T, PATCH, PATCH, NC]) });
    return { prob: Array.from(out.cultivo.data, Number), rinde: Number(out.rinde.data[0]) * info.rinde_desvio + info.rinde_media };
  }

  // Igual que Modelos.predecir de inferencia.py: con campaña, el modelo que no la vio; sin campaña, el promedio de los 6.
  async function predecir(bandas, campana) {
    validarSerie(bandas);
    await preparar();
    const campanas = Object.keys(indice.modelos).sort();
    if (campana && !indice.modelos[campana]) throw new Error(`No hay modelo para la campaña ${campana}.`);
    const usar = campana ? [campana] : campanas;
    const x12 = preprocesar(bandas);
    const salida = { campana_modelo: campana || 'promedio de los 6 modelos' };
    for (const modo of ['enero', 'completa']) {
      const res = [];
      for (const c of usar) res.push(await correr(x12, c, modo));
      const prob = indice.clases.map((_, k) => res.reduce((a, r) => a + r.prob[k], 0) / res.length);
      const rinde = res.reduce((a, r) => a + r.rinde, 0) / res.length;
      const k = prob.indexOf(Math.max(...prob));
      salida[modo] = {
        cultivo: indice.clases[k],
        probabilidades: Object.fromEntries(indice.clases.map((c, i) => [c, Math.round(prob[i] * 1e4) / 1e4])),
        rinde_kg_ha: k === 3 ? null : Math.round(rinde),
      };
    }
    return salida;
  }

  const api = { leerNpz, preprocesar, validarSerie, predecir, preparar, disponible, aF16, _set: (o, i) => { ort = o; indice = i; } };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else globalThis.ModeloEnVivo = api;
})();
