'use strict';
// "Probalo en tu lote": se elige un punto en el mapa, un servidor chico (Render) baja de Earth Engine la serie de
// Sentinel-2 de ese lugar preparada igual que en el notebook 02, y la red la clasifica acá, en el navegador, con los
// mismos modelos ONNX y la misma validación que los lotes de ejemplo. Usa COLORES, MESES, T_ENERO, $, esc, fmt, camp,
// leerJson, cacheado y bloqueProbs de app.js.
(function () {
  const SERVIDOR = (window.SERVIDOR_LOTES || '').replace(/\/+$/, '');
  const REGION = { oeste: -63.6, sur: -34.3, este: -59.5, norte: -31.4 };     // donde se entrenó y evaluó el modelo
  const PERMITIDO = { oeste: -73.6, sur: -55.1, este: -53.6, norte: -21.7 };  // Argentina (rectángulo aproximado)
  const CAMPANAS_TRABAJO = ['2019_20', '2020_21', '2021_22', '2022_23', '2023_24', '2024_25'];
  const CAMPANAS = [...CAMPANAS_TRABAJO, '2025_26'];
  const T = 27, PATCH = 16, NB = 10, I_B4 = 2, I_B8 = 6;
  const ins = $('#resultado-lote');
  const boton = $('#btn-analizar');
  let punto = null, cuadro = null, marca = null, despierto = false;

  // ------------------------------------------------------------ mapa
  const m = L.map('mapa-lote', { zoomSnap: 0.5 }).setView([-32.9, -61.4], 8);
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Imágenes &copy; Esri', maxZoom: 19,
  }).addTo(m);
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19, opacity: 0.9,
  }).addTo(m);
  L.rectangle([[REGION.sur, REGION.oeste], [REGION.norte, REGION.este]], {
    color: '#ffd54f', weight: 2, dashArray: '6 6', fill: false, interactive: false,
  }).addTo(m);
  m.attributionControl.addAttribution('Búsqueda &copy; OpenStreetMap');
  document.querySelector('[data-pestana="lote"]').addEventListener('click', () => setTimeout(() => m.invalidateSize(), 0));

  const dentroDe = (z, lat, lon) => lon >= z.oeste && lon <= z.este && lat >= z.sur && lat <= z.norte;
  const dentro = (lat, lon) => dentroDe(REGION, lat, lon);
  // Aviso fuera de la zona: texto fijo (sin datos del usuario), por eso va como HTML y no pasa por esc()
  const AVISO_FUERA = `<p class="nota"><span class="chip-dudoso">Fuera de la zona</span> Este lote está fuera del recuadro
    amarillo, donde se entrenó el modelo. La red responde igual, pero el resultado puede fallar: acá el calendario de siembra
    puede ser otro (y si se siembra más tarde, la versión de enero todavía no lo ve), puede haber cultivos que casi no aparecen
    en la zona núcleo, y cambian el clima y los suelos.</p>`;

  // Cuadrado aproximado de 160 m mientras no llega el exacto del servidor (grilla UTM de 10 m)
  function cuadradoAprox(lat, lon) {
    const dlat = 80 / 110574, dlon = 80 / (111320 * Math.cos(lat * Math.PI / 180));
    return [[lat + dlat, lon - dlon], [lat + dlat, lon + dlon], [lat - dlat, lon + dlon], [lat - dlat, lon - dlon]];
  }

  function elegir(lat, lon, centrar) {
    if (!dentroDe(PERMITIDO, lat, lon)) {
      mostrarError('Ese punto está fuera de Argentina. Las imágenes van de septiembre a mayo, el verano del hemisferio sur, '
        + 'así que el modelo solo tiene sentido en Argentina.');
      return;
    }
    punto = { lat, lon };
    if (cuadro) m.removeLayer(cuadro);
    if (marca) m.removeLayer(marca);
    cuadro = L.polygon(cuadradoAprox(lat, lon), { color: '#ffffff', weight: 2, fill: false, interactive: false }).addTo(m);
    marca = L.circleMarker([lat, lon], { radius: 3, color: '#ffffff', fillColor: '#ffffff', fillOpacity: 1, interactive: false }).addTo(m);
    if (centrar) m.setView([lat, lon], Math.max(m.getZoom(), 15));
    else if (m.getZoom() < 14) m.setView([lat, lon], 15);
    boton.disabled = !SERVIDOR;
    ins.className = 'inspector';
    ins.innerHTML = `<p class="ayuda">Punto elegido: ${lat.toFixed(5)}, ${lon.toFixed(5)}. Revisá que el cuadrado blanco quede
      todo adentro de un lote y tocá <strong>Analizar este lote</strong>.</p>
      ${dentro(lat, lon) ? '' : '<p class="nota"><span class="chip-dudoso">Fuera de la zona</span> Está fuera del recuadro amarillo, '
        + 'donde se entrenó el modelo: la red va a responder, pero el resultado no está verificado y puede fallar por un calendario '
        + 'de siembra distinto o cultivos que no conoce. Al analizarlo se explica en detalle.</p>'}
      <p class="nota" id="estado-servidor"></p>`;
    mostrarEstadoServidor();
  }
  m.on('click', (e) => elegir(e.latlng.lat, e.latlng.lng, false));

  // ------------------------------------------------------------ búsqueda (OpenStreetMap / Nominatim) o coordenadas
  $('#form-buscar').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const q = $('#buscar-lugar').value.trim();
    if (!q) return;
    const c = q.match(/^\s*(-?\d+(?:[.,]\d+)?)\s*[,;\s]\s*(-?\d+(?:[.,]\d+)?)\s*$/);
    if (c) {
      const a = Number(c[1].replace(',', '.')), b = Number(c[2].replace(',', '.'));
      const [lat, lon] = Math.abs(a) < Math.abs(b) ? [a, b] : [b, a];   // acepta "lat, lon" o "lon, lat"
      elegir(lat, lon, true);
      return;
    }
    try {
      const url = 'https://nominatim.openstreetmap.org/search?format=jsonv2&countrycodes=ar&limit=1&accept-language=es&q=' + encodeURIComponent(q);
      const r = await (await fetch(url)).json();
      if (!r.length) throw new Error(`No encontré “${q}”.`);
      m.setView([Number(r[0].lat), Number(r[0].lon)], 13);
    } catch (e) {
      mostrarError(e.message.startsWith('No encontré') ? e.message : 'No se pudo buscar el lugar. Probá con coordenadas o moviendo el mapa.');
    }
  });

  // ------------------------------------------------------------ campañas
  $('#sel-campana-lote').innerHTML = CAMPANAS.map((c) => `<option value="${c}">${camp(c)}${c === '2025_26' ? ' (nueva)' : ''}</option>`).join('');
  $('#sel-campana-lote').value = '2025_26';

  // ------------------------------------------------------------ servidor (el gratuito se duerme a los 15 minutos)
  async function pedir(ruta, segundos) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), segundos * 1000);
    try {
      const r = await fetch(SERVIDOR + ruta, { signal: ctrl.signal });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(d.detail || `El servidor respondió con error ${r.status}.`);
      return d;
    } catch (e) {
      if (e.name === 'AbortError') throw new Error('El servidor tardó demasiado en responder. Probá de nuevo en un minuto.');
      if (e instanceof TypeError) throw new Error('No se pudo conectar con el servidor que baja las imágenes. Probá de nuevo en un minuto.');
      throw e;
    } finally { clearTimeout(t); }
  }

  function mostrarEstadoServidor() {
    const el = document.getElementById('estado-servidor');
    if (!el) return;
    el.textContent = !SERVIDOR ? 'El servidor que baja las imágenes todavía no está configurado: por ahora solo se puede elegir el punto.'
      : despierto ? '' : 'El servidor gratuito se apaga cuando nadie lo usa: la primera consulta puede tardar hasta un minuto.';
  }

  async function despertar() {
    if (!SERVIDOR) return false;
    try {
      const s = await pedir('/salud', 90);
      despierto = !!s.earth_engine;
      if (!despierto && s.error) console.warn('Servidor sin Earth Engine:', s.error);
    } catch { despierto = false; }
    mostrarEstadoServidor();
    return despierto;
  }
  // Se despierta al abrir la página y se lo mantiene despierto mientras la página esté abierta (para la exposición)
  if (SERVIDOR) { despertar(); setInterval(despertar, 10 * 60 * 1000); }
  mostrarEstadoServidor();

  // ------------------------------------------------------------ análisis
  function decodificar(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Uint16Array(u8.buffer);   // little-endian, (T, H, W, C)
  }

  function ndviPorPeriodo(b) {
    const out = [];
    for (let t = 0; t < T; t++) {
      let s = 0;
      for (let p = 0; p < PATCH * PATCH; p++) {
        const i = (t * PATCH * PATCH + p) * NB;
        const r = b[i + I_B4], n = b[i + I_B8];
        s += (n - r) / (n + r + 1e-6);
      }
      out.push(s / (PATCH * PATCH));
    }
    return out;
  }

  // Curva típica de cada clase en la campaña (promedio de los departamentos, pesado por cantidad de lotes).
  // Para 2025/26 no hay mapa del INTA: se usa el promedio de las 6 campañas del trabajo.
  async function tipicasDe(c) {
    return cacheado(`tipicas_clase_${c}`, async () => {
      const fuentes = CAMPANAS_TRABAJO.includes(c) ? [c] : CAMPANAS_TRABAJO;
      const suma = {}, peso = {};
      for (const f of fuentes) {
        const d = await cacheado(`tip_${f}`, () => leerJson(`data/tipicas_${f}.json`));
        for (const [clave, v] of Object.entries(d)) {
          const u = v.usada;
          if (!u || u.nivel !== 'departamento') continue;
          const clase = clave.split('|').pop(), n = u.n_lotes || 1;
          suma[clase] = suma[clase] || new Array(T).fill(0);
          u.media.forEach((x, t) => { suma[clase][t] += n * x; });
          peso[clase] = (peso[clase] || 0) + n;
        }
      }
      return Object.fromEntries(Object.keys(suma).map((k) => [k, suma[k].map((x) => x / peso[k])]));
    });
  }

  function grafico(ndvi, tipicas) {
    const W = 368, H = 230, mg = { i: 34, d: 8, a: 10, b: 62 };
    const vals = [...ndvi, ...Object.values(tipicas).flat()];
    const ymin = Math.min(0, ...vals.map((v) => v - 0.05)), ymax = Math.max(1, ...vals);
    const x = (t) => mg.i + (W - mg.i - mg.d) * t / 26;
    const y = (v) => mg.a + (H - mg.a - mg.b) * (1 - (v - ymin) / (ymax - ymin));
    const linea = (arr) => arr.map((v, t) => `${x(t).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    let ejes = '';
    for (let v = Math.ceil(ymin * 5) / 5; v <= ymax + 1e-9; v += 0.2) {
      ejes += `<line x1="${mg.i}" x2="${W - mg.d}" y1="${y(v)}" y2="${y(v)}" stroke="#ece9e1"/><text x="${mg.i - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`;
    }
    MESES.forEach((mes, k) => { ejes += `<text x="${x(3 * k + 1)}" y="${H - mg.b + 16}" text-anchor="middle">${mes}</text>`; });
    const xe = (x(T_ENERO - 1) + x(T_ENERO)) / 2;
    const tip = Object.entries(tipicas).map(([c, v]) =>
      `<polyline points="${linea(v)}" fill="none" stroke="${COLORES[c] || '#888'}" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.9"/>`).join('');
    const ley = Object.keys(tipicas).map((c, k) =>
      `<g transform="translate(${mg.i + 82 * k},${H - 4})"><line x1="0" x2="16" y1="-4" y2="-4" stroke="${COLORES[c]}" stroke-width="2" stroke-dasharray="5 3"/><text x="20" y="0">${esc(c)}</text></g>`).join('');
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Curva de NDVI del lote y curvas típicas de cada cultivo">
      ${ejes}
      <rect x="${xe}" y="${mg.a}" width="${W - mg.d - xe}" height="${H - mg.a - mg.b}" fill="#000" opacity="0.035"/>
      <line x1="${xe}" x2="${xe}" y1="${mg.a}" y2="${H - mg.b}" stroke="#8a8a8a" stroke-dasharray="2 3"/>
      <text x="${xe + 4}" y="${mg.a + 10}">después de enero</text>
      ${tip}
      <polyline points="${linea(ndvi)}" fill="none" stroke="#1f2a24" stroke-width="2.4"/>
      ${ley}
      <text x="${mg.i}" y="${H - 22}">— este lote   - - típica de cada cultivo (promedio de la zona núcleo)</text>
    </svg>`;
  }

  function mostrarError(msg) {
    ins.className = 'inspector';
    ins.innerHTML = `<div class="resultado error">${esc(msg)}</div><p class="nota" id="estado-servidor"></p>`;
    mostrarEstadoServidor();
  }

  function esperando(msg) {
    ins.className = 'inspector';
    ins.innerHTML = `<p class="esperando">${esc(msg)}</p>`;
  }

  boton.addEventListener('click', async () => {
    if (!punto || !SERVIDOR) return;
    const c = $('#sel-campana-lote').value;
    const p = { ...punto };
    boton.disabled = true;
    try {
      if (!despierto) {
        esperando('Despertando el servidor (el gratuito se apaga cuando nadie lo usa; puede tardar hasta un minuto)…');
        if (!(await despertar())) throw new Error('El servidor no pudo conectarse con Earth Engine. Probá de nuevo en un minuto.');
      }
      esperando(`Bajando las imágenes satelitales de ${camp(c)} para este lote (27 fechas, de septiembre a mayo)…`);
      const s = await pedir(`/serie?lat=${p.lat}&lon=${p.lon}&campana=${c}`, 180);
      if (cuadro) m.removeLayer(cuadro);
      cuadro = L.polygon(s.esquinas, { color: '#ffffff', weight: 2, fill: false, interactive: false }).addTo(m);
      esperando('Corriendo la red en tu navegador…');
      const bandas = decodificar(s.bandas_b64);
      const M = window.ModeloEnVivo;
      const usar = CAMPANAS_TRABAJO.includes(c) ? c : null;
      const [r, tip] = await Promise.all([M.predecir(bandas, usar), tipicasDe(c).catch(() => ({}))]);
      const ndvi = ndviPorPeriodo(bandas);
      const fuera = s.fuera_de_zona ?? !dentro(p.lat, p.lon);
      const avisos = [];
      if (Math.max(...ndvi) < 0.45) {
        avisos.push('La curva de verdor es muy baja toda la campaña: el punto puede no ser un lote de cultivo (pueblo, camino, '
          + 'agua o monte). El modelo responde igual, pero el resultado no tiene sentido.');
      }
      if (s.pct_interpolado > 30) avisos.push(`Campaña muy nublada en este lugar: el ${fmt(s.pct_interpolado, 0)} % de la serie se completó interpolando.`);
      const modelo = usar ? `el modelo que no vio ${camp(c)}`
        : 'el promedio de los 6 modelos: ninguno vio esta campaña y el INTA todavía no publicó su mapa, así que el resultado todavía no se puede verificar';
      const rinde = (q) => q.rinde_kg_ha === null ? '' : `<tr><td>${q === r.enero ? 'En enero' : 'Campaña completa'}</td><td class="num">${fmt(q.rinde_kg_ha)}</td></tr>`;
      const lugar = s.departamento ? `${esc(s.departamento)}, ${esc(s.provincia)} · ` : '';
      ins.className = 'inspector';
      ins.innerHTML = `
        <h2>Lote elegido · ${camp(c)}</h2>
        <p class="sub">${lugar}${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}</p>
        ${fuera ? AVISO_FUERA : ''}
        ${avisos.map((a) => `<p class="nota"><span class="chip-dudoso">Ojo</span> ${esc(a)}</p>`).join('')}
        <h3>Verdor (NDVI) a lo largo de la campaña</h3>
        ${grafico(ndvi, tip)}
        <p class="nota">La línea negra es este lote; las punteadas, la curva típica de cada cultivo en la zona núcleo. El lote se parece
          al cultivo cuya curva sigue${fuera ? ' (fuera de la zona, las fechas de siembra pueden correr la curva)' : ''}. ${fmt(s.pct_interpolado, 1)} % de la serie se completó por nubes.</p>
        <h3>Qué dijo la red</h3>
        <p class="sub">Con ${modelo}. <strong>En enero</strong> es el pronóstico temprano, con las imágenes hasta fin de enero;
          <strong>con la campaña completa</strong> es la confirmación, con todas las imágenes hasta mayo.</p>
        ${bloqueProbs('En enero', r.enero, null)}
        ${bloqueProbs('Con la campaña completa', r.completa, null)}
        ${fuera || (r.enero.rinde_kg_ha === null && r.completa.rinde_kg_ha === null) ? '' : `<details class="plegable"><summary>Rinde estimado</summary>
          <table><tr><th>Rinde estimado del departamento (kg/ha)</th><th class="num"></th></tr>${rinde(r.enero)}${rinde(r.completa)}</table>
          <p class="nota">Es el rinde medio del departamento para ese cultivo, no el de este lote. Con campañas nuevas, la red no le
            ganó al promedio histórico del departamento.</p></details>`}
        <p class="nota">Imágenes satelitales bajadas en ${fmt(s.segundos, 0)} s${s.en_cache ? ' (ya estaban guardadas)' : ''}
          y clasificadas en tu navegador.</p>`;
    } catch (e) {
      mostrarError(e.message);
    } finally {
      boton.disabled = !punto;
    }
  });
})();
