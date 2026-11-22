'use strict';

const COLORES = { 'Maíz': '#1f77b4', 'Soja 1ra': '#2ca02c', 'Soja 2da': '#ff7f0e', 'Otros': '#9467bd' };
const MESES = ['sep', 'oct', 'nov', 'dic', 'ene', 'feb', 'mar', 'abr', 'may'];
const T_ENERO = 15;   // la red "hasta enero" ve los períodos 0 a 14

const estado = { campana: null, vista: 'red', modo: 'enero', lotes: [], metricas: null, clases: [], seleccionado: null };
const $ = (s) => document.querySelector(s);
const camp = (c) => c.replace('_', '/');
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toLocaleString('es-AR', { maximumFractionDigits: d, minimumFractionDigits: d });
const nombreModo = (m) => m === 'enero' ? 'en enero' : 'con la campaña completa';
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// En la versión publicada como Space estático (gratis) no hay servidor: los datos se leen de archivos JSON
// generados con la misma API, y el modelo corre en el navegador con los modelos ONNX (static/modelo.js).
const ESTATICO = window.ESTATICO === true;
if (ESTATICO) document.body.classList.add('estatico');
// Sin servidor, el modelo igual puede correr en el navegador si están publicados los modelos ONNX (static/modelo.js)
let EN_VIVO = false;
// id del lote -> { archivo, indice }: las series publicadas para correr el modelo en vivo (un .npz por campaña)
const SERIES = {};
const archivoDe = (id) => SERIES[id]?.archivo || ejemplos.find((e) => e.id === id && e.archivo)?.archivo || null;
const cacheDetalle = {};

async function leerJson(f, segundos = 30) {
  // con tiempo límite: si la descarga se traba, se muestra un error en vez de quedar "Cargando…" para siempre
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), segundos * 1000);
  try {
    const r = await fetch(f, { signal: ctrl.signal });
    if (!r.ok) throw new Error(`No se pudo leer ${f} (${r.status})`);
    return await r.json();
  } catch (e) {
    throw new Error(e.name === 'AbortError' ? `La descarga de ${f} tardó demasiado. Volvé a hacer clic en el punto.` : e.message);
  } finally {
    clearTimeout(t);
  }
}

// Detalle (curvas, probabilidades, rinde): está partido en tandas de POR_TANDA lotes, en el orden de lotes_<c>.json.
// Se baja solo la tanda del punto abierto (unos 300 KB) y las curvas típicas de la campaña, una vez.
// Si una descarga falla, no queda guardada, así el próximo clic la vuelve a intentar.
const POR_TANDA = 320;
function cacheado(clave, cargar) {
  if (!cacheDetalle[clave]) cacheDetalle[clave] = cargar().catch((e) => { delete cacheDetalle[clave]; throw e; });
  return cacheDetalle[clave];
}

// Detalle de un lote de cualquier campaña (para comparar lo que calcula el navegador con lo guardado por los notebooks)
async function detalleDe(id, c) {
  let ids;
  if (c === estado.campana) ids = estado.lotes.map((l) => l.id);
  else {
    const d = await cacheado(`lotes_${c}`, () => leerJson(`data/lotes_${c}.json`));
    const k = d.columnas.indexOf('id');
    ids = d.lotes.map((f) => f[k]);
  }
  const pos = ids.findIndex((x) => String(x) === String(id));
  if (pos < 0) throw new Error(`No existe el lote ${id} en ${camp(c)}`);
  const k = Math.floor(pos / POR_TANDA);
  const det = await cacheado(`det_${c}_${k}`, () => leerJson(`data/detalle_${c}_${k}.json`));
  if (!det[id]) throw new Error(`No existe el lote ${id}`);
  return det[id];
}

// POST /predict sin servidor: lee el .npz y corre el modelo en el navegador, con la misma respuesta que la API
async function predecirEnNavegador(opciones) {
  const M = window.ModeloEnVivo;
  let buf, campana = null, id = null, indice = null;
  if (opciones.body instanceof FormData) {
    buf = await opciones.body.get('archivo').arrayBuffer();
    campana = opciones.body.get('campana') || null;
  } else {
    id = JSON.parse(opciones.body).id;
    const archivo = archivoDe(id);
    if (!archivo) throw new Error(`El lote ${id} no tiene su serie de imágenes publicada en esta página.`);
    buf = await cacheado(`npz_${archivo}`, async () => {
      const r = await fetch(archivo.replace(/^\//, ''));
      if (!r.ok) throw new Error(`No se pudo descargar ${archivo}.`);
      return r.arrayBuffer();
    });
    indice = SERIES[id]?.archivo === archivo ? SERIES[id].indice : null;
  }
  const d = await M.leerNpz(buf, indice);
  campana = campana || d.campana || null;
  if (id === null) id = d.id;
  const r = await M.predecir(d.bandas, campana);
  if (id !== null && d.campana && campana === d.campana) {
    try {
      const det = await detalleDe(id, d.campana);
      Object.assign(r, { guardado: det.guardado, clase_inta: det.lote.clase_inta, rinde_real: det.lote.rinde_real });
    } catch { /* el lote no está en la muestra publicada: no hay con qué comparar */ }
  }
  r.nota = 'Calculado en tu navegador, en este momento.';
  if (!campana) {
    r.aviso = 'El archivo no dice de qué campaña es, así que se promediaron los 6 modelos. Si el lote es de una de las campañas '
      + 'del trabajo (2019/20 a 2024/25), 5 de esos 6 modelos lo vieron al entrenar y el resultado es optimista: elegí su campaña '
      + 'para usar el modelo que no la vio. Para un lote de una campaña nueva, el promedio es lo correcto.';
  }
  return r;
}

async function apiEstatica(url, opciones) {
  if (url === '/api/campanas') return leerJson('data/campanas.json');
  if (url === '/api/metricas') return leerJson('data/metricas.json');
  if (url === '/api/ejemplos') return leerJson('data/ejemplos.json');
  let m = url.match(/^\/api\/lotes\?campana=(.+)$/);
  if (m) return leerJson(`data/lotes_${decodeURIComponent(m[1])}.json`);
  m = url.match(/^\/api\/lote\/(\d+)$/);
  if (m) {
    const c = estado.campana;
    const pos = estado.lotes.findIndex((l) => String(l.id) === m[1]);
    if (pos < 0) throw new Error(`No existe el lote ${m[1]} en ${camp(c)}`);
    const k = Math.floor(pos / POR_TANDA);
    const [det, tip] = await Promise.all([cacheado(`det_${c}_${k}`, () => leerJson(`data/detalle_${c}_${k}.json`)),
      cacheado(`tip_${c}`, () => leerJson(`data/tipicas_${c}.json`))]);
    const d = det[m[1]];
    if (!d) throw new Error(`No existe el lote ${m[1]}`);
    return { ...d, tipica: tip[d.clave_tipica] };
  }
  if (url === '/predict' && EN_VIVO) return predecirEnNavegador(opciones);
  throw new Error('El modelo no está disponible en este momento. Probá recargar la página.');
}

async function api(url, opciones) {
  if (ESTATICO) return apiEstatica(url, opciones);
  const r = await fetch(url, opciones);
  const datos = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(datos.detail ? (typeof datos.detail === 'string' ? datos.detail : JSON.stringify(datos.detail)) : `Error ${r.status}`);
  return datos;
}

// ------------------------------------------------------------------ pestañas
document.querySelectorAll('.pestana').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.pestana').forEach((x) => { x.classList.toggle('activa', x === b); x.setAttribute('aria-selected', String(x === b)); });
  document.querySelectorAll('.panel-pestana').forEach((p) => p.classList.toggle('activa', p.id === `pestana-${b.dataset.pestana}`));
  if (b.dataset.pestana === 'mapa') mapa.invalidateSize();
}));

// ------------------------------------------------------------------ mapa
const mapa = L.map('mapa', { preferCanvas: true, zoomControl: true, zoomSnap: 0.25 }).setView([-32.9, -61.5], 7);
// Fondos sin clave de API: OpenStreetMap (por defecto) e imágenes satelitales de Esri
const fondoMapa = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap', maxZoom: 19,
}).addTo(mapa);
const fondoSatelite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Imágenes &copy; Esri', maxZoom: 19,
});
L.control.layers({ 'Mapa': fondoMapa, 'Satélite': fondoSatelite }, null, { position: 'topright' }).addTo(mapa);
const render = L.canvas({ padding: 0.3 });
const capa = L.layerGroup().addTo(mapa);
let marcaSel = null;

function colorDe(l) {
  if (estado.vista === 'inta' || estado.vista === 'errores' || estado.vista === 'vivo') return COLORES[l.clase_inta];
  // en "dudosos" el color es lo que dijo la red
  return COLORES[estado.modo === 'enero' ? l.pred_enero : l.pred_completa];
}

// Los puntos se achican cuando el mapa está alejado, para que no se tapen entre sí
function radioPuntos() {
  const z = mapa.getZoom();
  const base = z < 7.75 ? 2.6 : z < 8.75 ? 3.6 : z < 10 ? 5 : 7;
  return (estado.vista === 'errores' || estado.vista === 'dudosos') ? base + 1 : estado.vista === 'vivo' ? base + 3 : base;
}
mapa.on('zoomend', () => { const r = radioPuntos(); capa.eachLayer((m) => m.setRadius(r)); });

function dibujarPuntos() {
  capa.clearLayers();
  const acierto = estado.modo === 'enero' ? 'acierto_enero' : 'acierto_completa';
  const dudoso = estado.modo === 'enero' ? 'dudoso_enero' : 'dudoso_completa';
  const r = radioPuntos();
  let n = 0;
  for (const l of estado.lotes) {
    if (estado.vista === 'errores' && l[acierto]) continue;
    if (estado.vista === 'dudosos' && !l[dudoso]) continue;
    if (estado.vista === 'vivo' && !archivoDe(l.id)) continue;
    n++;
    const borde = estado.vista === 'errores' ? '#b3261e' : estado.vista === 'dudosos' ? '#9a6700' : '#ffffff';
    const m = L.circleMarker([l.lat, l.lon], {
      renderer: render, radius: r, weight: estado.vista === 'errores' || estado.vista === 'dudosos' ? 1.5 : 0.6,
      color: borde, fillColor: colorDe(l), fillOpacity: 0.9,
    });
    m.on('click', () => abrirLote(l.id, [l.lat, l.lon]));
    m.addTo(capa);
  }
  dibujarLeyenda(n);
}

function dibujarLeyenda(n) {
  const titulo = estado.vista === 'inta' ? 'Clase según el INTA'
    : estado.vista === 'errores' ? `Errores de la red ${nombreModo(estado.modo)} (color: lo que dijo el INTA)`
      : estado.vista === 'dudosos' ? `Dudosos de la red ${nombreModo(estado.modo)} (color: lo que dijo la red)`
      : estado.vista === 'vivo' ? 'Puntos que se pueden correr en vivo (color: lo que dijo el INTA). Tocá uno y “Correr el modelo”'
        : `Lo que dijo la red ${nombreModo(estado.modo)}`;
  $('#leyenda').innerHTML = `<strong>${titulo}</strong>` +
    estado.clases.map((c) => `<div><span class="punto" style="background:${COLORES[c]}"></span>${c}</div>`).join('') +
    `<div class="nota" style="margin:4px 0 0">${fmt(n)} puntos${notaDudosos()}</div>`;
}

function notaDudosos() {
  const du = estado.metricas?.dudosos?.por_campana?.[estado.campana]?.[estado.modo];
  if (estado.vista !== 'dudosos' || !du) return '';
  return ` con menos de ${fmt(100 * estado.metricas.dudosos.umbral)} puntos de diferencia entre las dos clases más probables.
    Ahí coincide con el INTA en el ${fmt(100 * du.coinciden_dudosos / Math.max(du.dudosos, 1))} %; en el resto, en el
    ${fmt(100 * du.coinciden_seguros / Math.max(du.lotes - du.dudosos, 1))} %.`;
}

function dibujarResumen() {
  const m = estado.metricas?.por_campana?.[estado.campana]?.[estado.modo];
  if (!m) { $('#resumen').innerHTML = ''; return; }
  const clases = estado.metricas.clases || estado.clases;
  const filas = clases.map((c, k) => {
    const fila = m.matriz_inta_vs_red[k];
    const total = fila.reduce((a, b) => a + b, 0);
    const pct = total ? (100 * fila[k] / total) : 0;
    return `<div class="fila" title="${fmt(fila[k])} de ${fmt(total)} puntos de ${c} según el INTA">
      <span class="etiqueta-clase"><span class="punto" style="background:${COLORES[c]}"></span>${c}</span><span class="num">${fmt(pct, 0)} %</span>
      <div class="barra"><span style="width:${pct}%;background:${COLORES[c]}"></span></div></div>`;
  }).join('');
  const rf = estado.metricas.random_forest?.por_campana?.[estado.campana]?.[estado.modo];
  const lineaRf = rf === undefined ? '' : `<div class="cifra-sub">Random Forest con los mismos datos: ${fmt(rf, 3)}
      (media de las 6 campañas: red ${fmt(mediaRed(estado.modo), 3)}, Random Forest ${fmt(mediaRf(estado.modo), 3)}).</div>`;
  $('#resumen').innerHTML = `
    <div class="tarjeta">
      <div class="cifra">${fmt(m.f1_macro, 3)}</div>
      <div class="cifra-sub">F1 macro de la red ${nombreModo(estado.modo)} en ${camp(estado.campana)}, campaña que no vio.
        Coincide con el INTA en ${fmt(m.aciertos)} de ${fmt(m.lotes)} puntos.</div>${lineaRf}
    </div>
    <div class="tarjeta"><div class="cab-clases">Puntos de cada clase del INTA que la red clasificó igual</div>
      <div class="por-clase">${filas}</div>
    </div>`;
}

function mediaRed(modo) {
  const v = Object.values(estado.metricas?.por_campana || {}).map((x) => x[modo]?.f1_macro).filter((x) => typeof x === 'number');
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}
function mediaRf(modo) {
  const v = Object.values(estado.metricas?.random_forest?.por_campana || {}).map((x) => x[modo]).filter((x) => typeof x === 'number');
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

async function cargarCampana(c) {
  estado.campana = c;
  const d = await api(`/api/lotes?campana=${encodeURIComponent(c)}`);
  const cols = d.columnas;
  estado.lotes = d.lotes.map((f) => Object.fromEntries(cols.map((k, i) => [k, f[i]])));
  dibujarPuntos();
  dibujarResumen();
  if (estado.lotes.length) mapa.fitBounds(L.latLngBounds(estado.lotes.map((l) => [l.lat, l.lon])), { padding: [20, 20] });
  if (ESTATICO) cacheado(`tip_${c}`, () => leerJson(`data/tipicas_${c}.json`)).catch(() => {});   // se adelantan las curvas típicas
  $('#inspector').className = 'inspector vacio';
  $('#inspector').innerHTML = '<p class="ayuda">Hacé clic en un punto del mapa para ver su curva de verdor, qué dijo la red y el rinde de su departamento.</p>';
  if (marcaSel) { mapa.removeLayer(marcaSel); marcaSel = null; }
}

document.querySelectorAll('[data-vista]').forEach((b) => b.addEventListener('click', () => {
  estado.vista = b.dataset.vista;
  document.querySelectorAll('[data-vista]').forEach((x) => { x.classList.toggle('activo', x === b); x.setAttribute('aria-pressed', String(x === b)); });
  dibujarPuntos();
}));
document.querySelectorAll('[data-modo]').forEach((b) => b.addEventListener('click', () => {
  estado.modo = b.dataset.modo;
  document.querySelectorAll('[data-modo]').forEach((x) => { x.classList.toggle('activo', x === b); x.setAttribute('aria-pressed', String(x === b)); });
  $('#btn-vista-red').textContent = estado.modo === 'enero' ? 'Red en enero' : 'Red con la campaña completa';
  dibujarPuntos();
  dibujarResumen();
}));

// ------------------------------------------------------------------ inspector
function graficoNdvi(ndvi, tipica, clase) {
  const W = 368, H = 210, m = { i: 34, d: 8, a: 10, b: 34 };
  const usada = tipica?.usada;
  const vals = [...(ndvi || []), ...(usada ? usada.media.map((v, i) => v + (usada.desvio[i] || 0)) : [])].filter((v) => v !== null);
  const ymin = Math.min(0, ...vals.map((v) => v - 0.05)), ymax = Math.max(1, ...vals);
  const x = (t) => m.i + (W - m.i - m.d) * t / 26;
  const y = (v) => m.a + (H - m.a - m.b) * (1 - (v - ymin) / (ymax - ymin));
  const linea = (arr) => arr.map((v, t) => (v === null ? null : `${x(t).toFixed(1)},${y(v).toFixed(1)}`)).filter(Boolean).join(' ');
  let banda = '', media = '';
  const color = COLORES[clase] || '#888';
  if (usada) {
    const sup = usada.media.map((v, t) => `${x(t).toFixed(1)},${y(v + (usada.desvio[t] || 0)).toFixed(1)}`);
    const inf = usada.media.map((v, t) => `${x(t).toFixed(1)},${y(v - (usada.desvio[t] || 0)).toFixed(1)}`).reverse();
    banda = `<polygon points="${sup.concat(inf).join(' ')}" fill="${color}" opacity="0.16"/>`;
    media = `<polyline points="${linea(usada.media)}" fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="5 4"/>`;
  }
  const xe = (x(T_ENERO - 1) + x(T_ENERO)) / 2;
  let ejes = '';
  for (let v = Math.ceil(ymin * 5) / 5; v <= ymax + 1e-9; v += 0.2) {
    ejes += `<line x1="${m.i}" x2="${W - m.d}" y1="${y(v)}" y2="${y(v)}" stroke="#ece9e1"/><text x="${m.i - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  MESES.forEach((mes, k) => { ejes += `<text x="${x(3 * k + 1)}" y="${H - m.b + 16}" text-anchor="middle">${mes}</text>`; });
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Curva de NDVI del lote y curva típica">
    ${ejes}
    <rect x="${xe}" y="${m.a}" width="${W - m.d - xe}" height="${H - m.a - m.b}" fill="#000" opacity="0.035"/>
    <line x1="${xe}" x2="${xe}" y1="${m.a}" y2="${H - m.b}" stroke="#8a8a8a" stroke-dasharray="2 3"/>
    <text x="${xe + 4}" y="${m.a + 10}">después de enero</text>
    ${banda}${media}
    ${ndvi ? `<polyline points="${linea(ndvi)}" fill="none" stroke="#1f2a24" stroke-width="2.2"/>` : ''}
    <text x="${m.i}" y="${H - 4}">— este punto   - - típica de ${esc(clase)} (${usada ? (usada.nivel === 'zona' ? 'zona' : 'departamento') : 'sin datos'}, ± desvío)</text>
  </svg>`;
}

function bloqueProbs(titulo, p, real) {
  const filas = Object.entries(p.probabilidades).map(([c, v]) => `
    <div class="prob"><span>${c}</span><div class="barra"><span style="width:${(100 * v).toFixed(1)}%;background:${COLORES[c]}"></span></div>
    <span class="num">${fmt(100 * v, 0)} %</span></div>`).join('');
  const ok = real ? (p.cultivo === real) : null;
  const orden = Object.entries(p.probabilidades).sort((a, b) => b[1] - a[1]);
  const margen = orden.length > 1 ? orden[0][1] - orden[1][1] : 1;
  const umbral = estado.metricas?.dudosos?.umbral ?? 0.5;
  const dudoso = p.dudoso ?? (margen < umbral);
  const veredicto = ok === null ? '' : ok
    ? (dudoso ? '<span class="dudoso">coincide con el INTA, pero dudando</span>' : '<span class="ok">✓ coincide con el INTA</span>')
    : '<span class="mal">✗ no coincide</span>';
  const aviso = dudoso ? `<div class="nota" style="margin:0 0 4px"><span class="chip-dudoso">Dudoso</span>
    entre ${esc(orden[0][0])} y ${esc(orden[1][0])} hay ${fmt(100 * margen, 0)} puntos de diferencia (menos de ${fmt(100 * umbral)}).</div>` : '';
  return `<div class="modo-prob"><div class="cab">${titulo}: ${esc(p.cultivo)} ${veredicto}</div>${aviso}${filas}</div>`;
}

async function abrirLote(id, latlng) {
  const ins = $('#inspector');
  ins.className = 'inspector';
  ins.innerHTML = '<p class="ayuda">Cargando…</p>';
  if (marcaSel) mapa.removeLayer(marcaSel);
  // mismo canvas que los puntos y sin capturar clics: si no, el anillo crea un canvas encima que tapa los demás puntos
  marcaSel = L.circleMarker(latlng, { renderer: render, interactive: false, radius: 10, color: '#1f2a24', weight: 2.5, fill: false }).addTo(mapa);
  try {
    const d = await api(`/api/lote/${id}`);
    const l = d.lote, g = d.guardado, rd = d.rinde_depto;
    const otros = l.clase_inta === 'Otros';
    const puedeCorrer = ESTATICO ? (EN_VIVO && !!archivoDe(l.id)) : d.tiene_serie;
    const rinde = otros ? '<p class="sub">La clase Otros no tiene rinde.</p>' : `
      <table>
        <tr><th>Rinde del departamento (kg/ha)</th><th class="num">${esc(l.clase_inta)}</th></tr>
        <tr><td>Real (MAGyP, sobre superficie sembrada)</td><td class="num">${fmt(l.rinde_real)}</td></tr>
        <tr><td>Estimado por la red en enero${rd ? ` (promedio de ${rd.n} puntos)` : ''}</td><td class="num">${fmt(rd ? rd.enero : l.rinde_enero)}</td></tr>
        <tr><td>Estimado por la red, campaña completa</td><td class="num">${fmt(rd ? rd.completa : l.rinde_completa)}</td></tr>
        <tr><td>Promedio histórico del departamento</td><td class="num">${fmt(l.rinde_hist_depto)}</td></tr>
      </table>
      <p class="nota">El rinde es por departamento: todos los puntos de ${esc(l.clase_inta)} de ${esc(l.departamento)} comparten el mismo valor real.
        Con campañas nuevas, la red no le ganó al promedio histórico del departamento.</p>`;
    ins.innerHTML = `
      <h2>Punto ${l.id} · ${camp(l.campana)}</h2>
      <p class="sub">${esc(l.departamento)}, ${esc(l.provincia)} · Según el INTA:
        <span class="etiqueta-clase"><span class="punto" style="background:${COLORES[l.clase_inta]}"></span>${esc(l.clase_inta)}</span></p>
      <h3>Verdor (NDVI) a lo largo de la campaña</h3>
      ${graficoNdvi(d.ndvi, d.tipica, l.clase_inta)}
      ${l.pct_interpolado !== null ? `<p class="nota">${fmt(l.pct_interpolado, 1)} % de la serie se completó por nubes.</p>` : ''}
      <h3>Qué dijo la red (modelo que no vio ${camp(l.campana)})</h3>
      ${bloqueProbs('En enero', g.enero, l.clase_inta)}
      ${bloqueProbs('Con la campaña completa', g.completa, l.clase_inta)}
      <details class="plegable"><summary>Rinde del departamento</summary>${rinde}</details>
      ${EN_VIVO && !puedeCorrer ? `<p class="nota">Este punto no tiene su serie de imágenes publicada, así que no se puede
        correr el modelo en vivo. Los que sí se pueden están en la vista “Correr en vivo”.</p>` : ''}
      ${puedeCorrer ? `<h3>Correr el modelo ahora</h3>
        <p class="sub">Este punto tiene su serie de imágenes guardada: se le pasa a la red en este momento, con el modelo que no vio ${camp(l.campana)}.</p>
        <button class="principal" id="btn-correr">Correr el modelo</button><div id="res-correr"></div>` : ''}`;
    if (puedeCorrer) $('#btn-correr').addEventListener('click', () => correrPorId(l.id, $('#btn-correr'), $('#res-correr')));
  } catch (e) {
    ins.innerHTML = `<div class="resultado error">${esc(e.message)}</div>`;
  }
}

function difMax(a, b) {
  return Math.max(...Object.keys(a).map((k) => Math.abs(a[k] - (b[k] ?? 0))));
}

function htmlResultado(r) {
  const modos = ['enero', 'completa'].map((m) => {
    const p = r[m];
    let comp = '';
    if (r.guardado) {
      const dm = difMax(p.probabilidades, r.guardado[m].probabilidades);
      comp = ` · ${p.cultivo === r.guardado[m].cultivo && dm < 0.01 ? '<span class="ok">igual al resultado de la evaluación del trabajo</span>' : `<span class="mal">difiere de lo guardado (máx. ${dm.toFixed(3)})</span>`}`;
    }
    const top = Math.max(...Object.values(p.probabilidades));
    return `<div><strong>${m === 'enero' ? 'En enero' : 'Campaña completa'}:</strong> ${esc(p.cultivo)} (${fmt(100 * top, 0)} %)
      · rinde ${p.rinde_kg_ha === null ? '—' : fmt(p.rinde_kg_ha) + ' kg/ha'}${comp}</div>`;
  }).join('');
  const inta = r.clase_inta ? `<div>Según el INTA: <strong>${esc(r.clase_inta)}</strong>${r.rinde_real ? ` · rinde real del departamento ${fmt(r.rinde_real)} kg/ha` : ''}</div>` : '';
  const modelo = `<div class="nota" style="margin:2px 0 6px">Modelo usado: ${r.campana_modelo && r.campana_modelo.includes('_') ? 'el que no vio ' + camp(r.campana_modelo) : esc(r.campana_modelo)}</div>`;
  const aviso = r.aviso ? `<p class="nota"><span class="chip-dudoso">Ojo</span> ${esc(r.aviso)}</p>` : '';
  return `<div class="resultado">${modelo}${modos}${inta}${aviso}${r.nota ? `<p class="nota">${esc(r.nota)}</p>` : ''}</div>`;
}

async function correrPorId(id, boton, destino) {
  const texto = boton.textContent;
  boton.disabled = true; boton.textContent = 'Corriendo…';
  try {
    const r = await api('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }) });
    destino.innerHTML = htmlResultado(r);
  } catch (e) {
    destino.innerHTML = `<div class="resultado error">${esc(e.message)}</div>`;
  } finally { boton.disabled = false; boton.textContent = texto; }
}

// ------------------------------------------------------------------ ficha del modelo
function dibujarMetricas() {
  const pc = estado.metricas?.por_campana || {};
  const cs = Object.keys(pc).sort();
  const val = (c, m, k) => pc[c]?.[m]?.[k];
  const media = (m, k) => { const v = cs.map((c) => val(c, m, k)).filter((x) => typeof x === 'number'); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const fila = (et, f) => `<tr><td>${et}</td>${f}</tr>`;
  const rf = estado.metricas.random_forest?.por_campana || {};
  const celdas = (g, rfe, rfc) => `<td class="num"><strong>${fmt(g('enero', 'f1_macro'), 3)}</strong></td><td class="num">${fmt(rfe, 3)}</td>
    <td class="num"><strong>${fmt(g('completa', 'f1_macro'), 3)}</strong></td><td class="num">${fmt(rfc, 3)}</td>
    <td class="num">${fmt(g('completa', 'mae_rinde_depto'))}</td><td class="num">${fmt(g('completa', 'mae_promedio_depto'))}</td>`;
  $('#tabla-metricas').innerHTML = `<table>
    <tr><th>Campaña (no vista)</th><th class="num">F1 red enero</th><th class="num">RF enero</th><th class="num">F1 red completa</th>
      <th class="num">RF completa</th><th class="num">MAE rinde red</th><th class="num">MAE prom. depto</th></tr>
    ${cs.map((c) => fila(camp(c), celdas((m, k) => val(c, m, k), rf[c]?.enero, rf[c]?.completa))).join('')}
    ${fila('<strong>Media</strong>', celdas(media, mediaRf('enero'), mediaRf('completa')))}</table>`;
  // limitaciones con los números reales
  const tasas = cs.map((c) => 1 - val(c, 'enero', 'aciertos') / val(c, 'enero', 'lotes'));
  const tot = cs.reduce((a, c) => [a[0] + val(c, 'enero', 'aciertos'), a[1] + val(c, 'enero', 'lotes')], [0, 0]);
  const unoCada = (t) => Math.round(1 / t);
  $('#lim-desacuerdo').textContent = `En enero, la red no coincide con el INTA en 1 de cada ${unoCada(1 - tot[0] / tot[1])} puntos en promedio:
    desde 1 de cada ${unoCada(Math.min(...tasas))} en la mejor campaña hasta 1 de cada ${unoCada(Math.max(...tasas))} en la sequía de 2022/23.`;
  const du = estado.metricas.dudosos?.total?.enero;
  if (du) $('#lim-dudosos').textContent = `Cuando la red duda (diferencia menor a ${fmt(100 * estado.metricas.dudosos.umbral)} puntos entre las dos
    clases más probables), coincide con el INTA solo en el ${fmt(100 * du.coinciden_dudosos / du.dudosos, 0)} % de los casos, contra el
    ${fmt(100 * du.coinciden_seguros / (du.lotes - du.dudosos), 0)} % cuando está segura. Sirve para decidir qué revisar, pero ojo: muchos
    desacuerdos son con la red segura.`;
  $('#aclaracion-metricas').textContent = (estado.metricas.aclaracion || '') +
    ' RF: Random Forest sobre el promedio de cada banda en cada período, con las mismas particiones. MAE: error absoluto medio del' +
    ' rinde por departamento, en kg/ha, con la campaña completa; “prom. depto” es el promedio histórico del departamento.';
}

// ------------------------------------------------------------------ probá la API
let ejemplos = [];
function dibujarEjemplos() {
  if (!$('#lista-ejemplos')) return;   // la pestaña de ejemplos se sacó: los lotes guardados se corren desde el mapa
  const c = $('#sel-campana-ej').value;
  const lista = ejemplos.filter((e) => e.campana === c && (!ESTATICO || archivoDe(e.id)));
  $('#lista-ejemplos').innerHTML = `<table><tr><th>Id</th><th>Según el INTA</th><th></th><th></th></tr>
    ${lista.map((e) => `<tr><td>${e.id}</td><td><span class="etiqueta-clase"><span class="punto" style="background:${COLORES[e.clase_inta]}"></span>${esc(e.clase_inta)}</span></td>
      <td>${e.archivo ? `<a href="${ESTATICO ? e.archivo.replace(/^\//, '') : e.archivo}" download>descargar .npz</a>` : ''}</td>
      <td>${ESTATICO && !EN_VIVO ? '' : `<button class="principal" data-id="${e.id}" style="padding:4px 10px;font-size:14px">probar por id</button>`}</td></tr>
      <tr><td colspan="4" id="res-ej-${e.id}"></td></tr>`).join('')}</table>`;
  document.querySelectorAll('#lista-ejemplos button[data-id]').forEach((b) =>
    b.addEventListener('click', () => correrPorId(Number(b.dataset.id), b, $(`#res-ej-${b.dataset.id}`))));
}

$('#form-subir')?.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const f = $('#archivo').files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append('archivo', f);
  if ($('#sel-campana-subir').value) fd.append('campana', $('#sel-campana-subir').value);
  $('#resultado-subir').innerHTML = '<p class="nota">Corriendo el modelo…</p>';
  try {
    $('#resultado-subir').innerHTML = htmlResultado(await api('/predict', { method: 'POST', body: fd }));
  } catch (e) {
    $('#resultado-subir').innerHTML = `<div class="resultado error">${esc(e.message)}</div>`;
  }
});

// ------------------------------------------------------------------ inicio
(async function iniciar() {
  try {
    const [campanas, metricas] = await Promise.all([api('/api/campanas'), api('/api/metricas')]);
    estado.metricas = metricas;
    estado.clases = metricas.clases || Object.keys(COLORES);
    const opciones = campanas.map((c) => `<option value="${c}">${camp(c)}</option>`).join('');
    $('#sel-campana').innerHTML = opciones;
    if ($('#sel-campana-ej')) $('#sel-campana-ej').innerHTML = opciones;
    if ($('#sel-campana-subir')) $('#sel-campana-subir').innerHTML += opciones;
    const inicial = campanas[campanas.length - 1];
    $('#sel-campana').value = inicial;
    $('#sel-campana').addEventListener('change', (e) => cargarCampana(e.target.value));
    $('#sel-campana-ej')?.addEventListener('change', dibujarEjemplos);
    dibujarMetricas();
    await cargarCampana(inicial);
    ejemplos = await api('/api/ejemplos');
    if (ESTATICO && window.ModeloEnVivo) {
      EN_VIVO = await window.ModeloEnVivo.disponible();
      document.body.classList.toggle('en-vivo', EN_VIVO);
      try {
        for (const s of await leerJson('data/series_vivo.json')) SERIES[s.id] = s;
      } catch { /* sin la lista ampliada quedan los lotes de ejemplo */ }
    }
    dibujarEjemplos();
  } catch (e) {
    $('#resumen').innerHTML = `<div class="resultado error">No se pudieron cargar los datos: ${esc(e.message)}</div>`;
  }
})();

// el link a las aclaraciones las despliega
document.querySelectorAll('a[href="#aclaraciones"]').forEach((a) => a.addEventListener('click', () => { $('#aclaraciones').open = true; }));
