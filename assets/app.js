import { parseCSV, parseFile, prepareData, priceConfig, plotEligible, adjacentConnections, connectionKey, costDomain } from './data.mjs';

const $ = id => document.getElementById(id);
const colors = ['#a67718', '#c35e81', '#258995', '#705ca2', '#5275b6', '#62772a', '#226b57', '#bc5140', '#916543', '#62899b', '#a55ea0', '#66815d', '#545995', '#b6814d', '#2b7890', '#a45559', '#877332', '#617070', '#8c6fac', '#487953'];
const pct = n => n === null ? '—' : `${(n * 100).toFixed(1)}%`;
const usd = n => n === null ? '—' : n === 0 ? '$0' : '$' + n.toFixed(n < .001 ? 7 : n < .1 ? 5 : 3).replace(/0+$/, '').replace(/\.$/, '');
const ms = n => n === null ? '—' : `${(n * 1000).toFixed(2)} ms`;
const escape = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const name = c => String(c.raw.model_alias || c.model.split('/').at(-1)).replace(/_/g, ' ');
const safePath = s => typeof s === 'string' && /^(?:results|data)\/[\p{L}\p{N}_ .\/+-]+$/u.test(s) && !s.split('/').includes('..');
const link = path => safePath(path) ? `<a href="${encodeURI(path)}" target="_blank" rel="noopener">${escape(path)}</a>` : escape(path);
const hidden = { cost: new Set(), confidence: new Set() };
let data, manifest, rates, registry, modelColors, loading = false;
async function fetchText(path) {
  if (!safePath(path)) throw new Error(`Invalid source path: ${path}`);
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.text();
}
async function load() {
  if (loading) return;
  loading = true; $('refresh').disabled = true; $('error').hidden = true;
  $('load-status').textContent = 'Reading original result files and checking full-test coverage…';
  try {
    [manifest, rates, registry] = await Promise.all(['results/manifest.json', 'results/cloud_rates.json', 'results/formats.json'].map(async p => JSON.parse(await fetchText(p))));
    const [truth, ...files] = await Promise.all([fetchText(manifest.ground_truth).then(parseCSV), ...manifest.files.map(async f => ({ ...f, data: parseFile(f.path, await fetchText(f.path)) }))]);
    data = prepareData(files, truth, registry, manifest.ground_truth);
    modelColors = new Map([...new Set(data.configs.map(c => c.model))].map((model, i) => [model, colors[i] || `hsl(${Math.round(13 + (i - colors.length) * 137.508) % 360} 50% 40%)`]));
    fillControls(); renderOverview(); render();
    $('content').hidden = false;
    $('load-status').textContent = `Read ${files.length} source files · ${data.sampleCount} test sentences · scores recomputed from predictions where available.`;
  } catch (error) {
    $('content').hidden = true; $('error').hidden = false;
    $('error').textContent = `Could not load results: ${error.message}. Serve the repository over HTTP and run uv run python scripts/build_results_manifest.py after importing results.`;
    $('load-status').textContent = 'Results could not be loaded.';
  } finally { loading = false; $('refresh').disabled = false; }
}
function options(id, entries) {
  const previous = $(id).value;
  $(id).replaceChildren(...entries.map(([value, title]) => { const el = document.createElement('option'); el.value = value; el.textContent = title; return el; }));
  if (entries.some(([value]) => value === previous)) $(id).value = previous;
}
function fillControls() {
  const models = [...new Map(data.configs.map(c => [c.model, name(c)])).entries()];
  options('cost-model', [['all', 'All DL + LLM models'], ['dl', 'DL models only'], ['llm', 'LLM models only'], ...models]);
  options('confidence-model', [['all', 'All LLM models'], ...models.filter(([m]) => data.configs.some(c => c.model === m && c.kind === 'llm'))]);
  const efforts = [...new Set(data.configs.filter(c => c.kind === 'llm').map(c => c.effort))].sort((a, b) => ['low', 'medium', 'high', 'default'].indexOf(a) - ['low', 'medium', 'high', 'default'].indexOf(b));
  for (const id of ['cost-reasoning', 'confidence-reasoning']) options(id, [['all', 'All reasoning levels'], ...efforts.map(e => [e, e[0].toUpperCase() + e.slice(1)])]);
  options('dl-rate', rates.presets.map(p => [p.id, p.title]));
}
function renderOverview() {
  const complete = data.configs.filter(c => c.complete), excluded = data.configs.filter(c => !c.complete);
  $('stats').innerHTML = [[data.sampleCount, 'Test sentences'], [new Set(data.configs.map(c => c.model)).size, 'Distinct models'], [complete.length, 'Complete configurations'], [excluded.length, 'Excluded configurations']].map(([v, label]) => `<div class="stat"><strong>${v}</strong><span>${label}</span></div>`).join('');
  const best = complete.length ? Math.max(...complete.map(c => c.accuracy)) : null;
  const bestModels = [...new Set(complete.filter(c => c.accuracy === best).map(name))];
  const cheapest = complete.filter(c => c.kind === 'llm' && c.cost !== null).sort((a, b) => a.cost - b.cost)[0];
  $('summary').innerHTML = `<strong>${best === null ? 'No complete runs yet.' : `${pct(best)} is the highest observed accuracy.`}</strong> ${escape(bestModels.join(', '))}${cheapest ? `. Cheapest complete LLM run: ${escape(name(cheapest))} / ${escape(cheapest.effort)}, ${usd(cheapest.cost)} for all ${data.sampleCount} sentences.` : ''} With ${data.sampleCount} examples, one error changes accuracy by ${(100 / data.sampleCount).toFixed(2)} percentage points.`;
  const pairs = complete.filter(c => c.kind === 'llm' && c.effort === 'low').map(low => {
    const high = complete.find(c => c.effort === 'high' && connectionKey(c) === connectionKey(low));
    return high ? high.accuracy - low.accuracy : null;
  }).filter(v => v !== null);
  if (pairs.length) $('summary').innerHTML += `<p>For ${pairs.length} matched complete low/high reasoning pairs: ${pairs.filter(v => v > 1e-9).length} improved accuracy, ${pairs.filter(v => Math.abs(v) <= 1e-9).length} stayed equal, and ${pairs.filter(v => v < -1e-9).length} decreased. These are saved request settings; inference cost and confidence remain separate measurements.</p>`;
  $('excluded-list').innerHTML = excluded.map(c => `<li><strong>${escape(name(c))} / ${escape(c.effort)}</strong>: ${escape(c.reasons.join('; '))}. ${c.sources.map(link).join(' · ')}</li>`).concat(data.warnings.map(w => `<li>${escape(w)}</li>`)).join('') || '<li>All recognized runs match the full test set.</li>';
  $('sources').innerHTML = [manifest.ground_truth, 'results/manifest.json', 'results/formats.json', 'results/cloud_rates.json', ...manifest.files.map(f => f.path)].map(path => `<li>${link(path)}</li>`).join('');
}
function priced() {
  const preset = rates.presets.find(p => p.id === $('dl-rate').value) || rates.presets[0];
  return data.configs.map(c => priceConfig(c, preset, $('custom-rate').value, $('cost-basis').value));
}
function renderRate() {
  const p = rates.presets.find(p => p.id === $('dl-rate').value) || rates.presets[0], wall = $('cost-basis').value === 'wall';
  $('custom-rate').disabled = p.id !== 'custom';
  const rate = p.id === 'custom' ? $('custom-rate').value : p.hourly_usd;
  $('rate-note').textContent = `${p.scope}. Rate: ${rate === '' || rate === null || !Number.isFinite(Number(rate)) || Number(rate) < 0 ? 'enter a nonnegative custom rate' : `$${Number(rate).toFixed(6)}/hour`}. Sources checked ${rates.checked_on}. Estimate = recorded ${wall ? 'model wall' : 'inference'} seconds ÷ 3,600 × hourly rate. ${wall ? 'Includes measured model loading, warmup and inference.' : 'Excludes model loading and warmup.'} Excludes provisioning, idle time, storage, network, taxes and cloud billing minimums. Estimates reuse saved runtimes on the recorded devices; cloud hardware may run at a different speed.${p.id === 'gcp-t4-gpu-only' ? ' GPU-only rate excludes host CPU/RAM; it is not the price of a complete VM.' : ''}`;
  $('rate-sources').innerHTML = p.sources.filter(s => /^https:\/\//.test(s.url)).map(s => `<a href="${escape(s.url)}" target="_blank" rel="noopener">${escape(s.title)}</a>`).join(' · ');
}
function filtered(configs, chart) {
  const model = $(`${chart}-model`).value, effort = $(`${chart}-reasoning`).value;
  return configs.filter(c => (model === 'all' || c.kind === model || c.model === model) && (c.kind === 'dl' || effort === 'all' || c.effort === effort));
}
function render() {
  if (!data) return;
  renderRate(); const configs = priced();
  for (const chart of ['cost', 'confidence']) {
    const candidates = filtered(configs.filter(c => chart === 'cost' || c.kind === 'llm'), chart);
    const eligible = candidates.filter(c => chart === 'cost' ? plotEligible(c, $('cost-scale').value) : c.complete && c.confidence !== null);
    const points = eligible.filter(c => !hidden[chart].has(c.model));
    renderChart(chart, points);
    renderLegend(chart, eligible);
    const excluded = candidates.filter(c => !c.complete).length, unavailable = candidates.filter(c => c.complete && !eligible.includes(c)).length;
    $(`${chart}-note`).textContent = `${points.length} of ${eligible.length} eligible configurations shown, each covering all ${data.sampleCount} sentences. ${excluded} incomplete runs excluded.${unavailable ? ` ${unavailable} complete runs omitted: ${chart === 'cost' ? 'missing cost/timing, incompatible GPU rate, or zero cost on a logarithmic axis (use Linear for zero).' : 'confidence missing for one or more examples.'}` : ''} ${chart === 'cost' ? 'Exact overlapping points share a tooltip; coordinates are never shifted. DL points represent saved inference configurations.' : 'Confidence is the mean model-reported probability, not the fraction of correct answers. Lines connect adjacent available reasoning levels only.'}`;
  }
  renderTable(configs);
}
function renderLegend(chart, configs) {
  const models = [...new Map(configs.map(c => [c.model, name(c)])).entries()];
  $(`${chart}-legend`).replaceChildren(...models.map(([model, title]) => {
    const button = document.createElement('button'); button.className = 'legend-item'; button.type = 'button'; button.title = 'Show / hide ' + title;
    button.setAttribute('aria-pressed', String(!hidden[chart].has(model))); button.style.opacity = hidden[chart].has(model) ? '.4' : '1';
    const dot = document.createElement('span'); dot.className = 'legend-dot'; dot.style.background = modelColors.get(model); button.append(dot, document.createTextNode(title));
    button.onclick = () => { hidden[chart].has(model) ? hidden[chart].delete(model) : hidden[chart].add(model); render(); }; return button;
  }));
}
const NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v)); if (text !== undefined) node.textContent = text; return node;
}
function renderChart(chart, points) {
  const W = 1160, H = 560, L = 85, R = 1090, T = 64, B = 463;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg', role: 'group', 'aria-label': `${chart === 'cost' ? 'Total cost versus accuracy' : 'Mean confidence versus accuracy'}, ${points.length} complete configurations` });
  svg.append(svgEl('title', {}, chart === 'cost' ? 'Total cost versus full-test sentiment accuracy' : 'Mean LLM confidence versus full-test sentiment accuracy'));
  const add = (tag, attrs, text) => { const n = svgEl(tag, attrs, text); svg.append(n); return n; };
  const log = chart === 'cost' && $('cost-scale').value === 'log';
  let xMin = log ? .00001 : 0, xMax = log ? .1 : 1, yMin = 0, yMax = 1;
  if (chart === 'confidence' && $('confidence-range').value === 'focus') { xMin = .8; yMin = .8; }
  if (chart === 'cost') [xMin, xMax] = costDomain(points.map(c => c.pricedCost), log);
  if (chart === 'cost') {
    if ($('accuracy-range').value === 'top') yMin = .7;
    if ($('accuracy-range').value === 'focus' && points.length) yMin = Math.max(0, Math.floor((Math.min(...points.map(c => c.accuracy)) - .05) * 10) / 10);
  }
  const x = v => L + (log ? (Math.log(v) - Math.log(xMin)) / (Math.log(xMax) - Math.log(xMin)) : (v - xMin) / (xMax - xMin)) * (R - L);
  const y = v => B - (v - yMin) / (yMax - yMin) * (B - T);
  add('text', { x: L, y: 24, class: 'axis-title' }, 'Accuracy on the complete test set (%)');
  add('text', { x: L, y: 44, class: 'axis-text', fill: '#226b57' }, '↑ HIGHER IS BETTER');
  for (let i = 0; i <= 4; i++) {
    const value = yMin + (yMax - yMin) * i / 4;
    add('line', { x1: L, x2: R, y1: y(value), y2: y(value), class: 'grid-line' });
    add('text', { x: L - 12, y: y(value) + 4, 'text-anchor': 'end', class: 'axis-text' }, (value * 100).toFixed(1).replace(/\.0$/, ''));
  }
  let ticks;
  if (log) {
    ticks = []; for (let power = Math.log10(xMin); power <= Math.log10(xMax) + .001; power++) ticks.push(10 ** power);
    if (ticks.length > 8) ticks = ticks.filter((v, i) => i % Math.ceil(ticks.length / 8) === 0 || i === ticks.length - 1);
  } else ticks = Array.from({ length: 6 }, (_, i) => xMin + (xMax - xMin) * i / 5);
  for (const value of ticks) {
    add('line', { x1: x(value), x2: x(value), y1: T, y2: B, class: 'grid-line' });
    add('text', { x: x(value), y: B + 25, 'text-anchor': 'middle', class: 'axis-text' }, chart === 'cost' ? usd(value) : `${Math.round(value * 100)}%`);
  }
  add('text', { x: (L + R) / 2, y: H - 23, 'text-anchor': 'middle', class: 'axis-title' }, chart === 'cost' ? `Total cost for all ${data.sampleCount} sentences · USD${log ? ' · logarithmic scale' : ' · linear scale'}` : 'Mean reported confidence (%)');
  if (chart === 'confidence') {
    add('line', { x1: L, x2: R, y1: B, y2: T, class: 'reference-line' });
    add('text', { x: L + 12, y: T + 75, class: 'axis-text' }, 'Underconfident');
    add('text', { x: R - 12, y: B - 12, 'text-anchor': 'end', class: 'axis-text' }, 'Overconfident');
  }
  const visible = points.filter(c => c.accuracy >= yMin && c.accuracy <= yMax && (chart === 'cost' || c.confidence >= xMin));
  const coord = c => [x(chart === 'cost' ? c.pricedCost : c.confidence), y(c.accuracy)];
  for (const [a, b] of adjacentConnections(visible)) {
    const [ax, ay] = coord(a), [bx, by] = coord(b);
    add('line', { x1: ax, y1: ay, x2: bx, y2: by, stroke: modelColors.get(a.model), class: 'connector' });
  }
  // Group exact coordinates for access to overlapping configurations, without distorting measurements.
  const clusters = new Map();
  for (const c of visible) { const key = JSON.stringify(coord(c)); if (!clusters.has(key)) clusters.set(key, []); clusters.get(key).push(c); }
  for (const cluster of clusters.values()) {
    cluster.sort((a, b) => ['high', 'medium', 'low', 'dl'].indexOf(a.effort) - ['high', 'medium', 'low', 'dl'].indexOf(b.effort));
    cluster.forEach((c, i) => {
      const [cx, cy] = coord(c), size = cluster.length > 1 ? Math.max(3, 10 - i * 2.5) : 6;
      const attrs = { fill: modelColors.get(c.model), stroke: '#fff', 'stroke-width': 1.3, class: 'plot-point', tabindex: '0', role: 'button', 'aria-label': `${name(c)}, ${c.effort}, accuracy ${pct(c.accuracy)}, ${chart === 'cost' ? usd(c.pricedCost) : 'confidence ' + pct(c.confidence)}` };
      let marker;
      if (c.kind === 'dl') marker = add('path', { ...attrs, d: `M${cx},${cy - size} L${cx + size},${cy + size} L${cx - size},${cy + size} Z` });
      else if (c.effort === 'medium') marker = add('path', { ...attrs, d: `M${cx},${cy - size} L${cx + size},${cy} L${cx},${cy + size} L${cx - size},${cy} Z` });
      else if (c.effort === 'high') marker = add('rect', { ...attrs, x: cx - size, y: cy - size, width: size * 2, height: size * 2 });
      else marker = add('circle', { ...attrs, cx, cy, r: size });
      marker.append(svgEl('title', {}, cluster.map(p => `${name(p)} / ${p.effort}: ${pct(p.accuracy)} accuracy, ${chart === 'cost' ? usd(p.pricedCost) : pct(p.confidence) + ' confidence'}`).join('\n')));
      const show = () => showDetail(chart, cluster);
      marker.addEventListener('mouseenter', show); marker.addEventListener('focus', show); marker.addEventListener('click', show);
      marker.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } });
    });
  }
  if ($(`${chart}-labels`).checked) {
    const labels = [...new Map(visible.map(c => [c.model, c])).values()].sort((a, b) => coord(a)[1] - coord(b)[1]);
    const occupied = visible.map(c => { const [cx, cy] = coord(c); return { x: cx - 9, y: cy - 9, w: 18, h: 18 }; });
    for (const c of labels) {
      const [cx, cy] = coord(c), text = name(c), w = Math.min(260, text.length * 6.4 + 8), h = 17;
      let box;
      for (const distance of [14, 34, 56, 82, 112, 145, 180, 215, 255, 300]) {
        for (const [dx, dy] of [[1, 1], [1, -1], [-1, 1], [-1, -1], [0, 1], [0, -1]]) {
          const candidate = { x: dx < 0 ? cx - distance - w : dx === 0 ? cx - w / 2 : cx + distance, y: cy + dy * distance - h / 2, w, h };
          if (candidate.x < L + 3 || candidate.x + w > R - 3 || candidate.y < T + 3 || candidate.y + h > B - 3) continue;
          if (occupied.some(b => candidate.x < b.x + b.w + 4 && candidate.x + w + 4 > b.x && candidate.y < b.y + b.h + 3 && candidate.y + h + 3 > b.y)) continue;
          box = candidate; break;
        }
        if (box) break;
      }
      if (!box) continue; occupied.push(box);
      add('line', { x1: cx, y1: cy, x2: Math.max(box.x, Math.min(cx, box.x + w)), y2: box.y + h / 2, stroke: modelColors.get(c.model), 'stroke-width': .8, opacity: .65, 'pointer-events': 'none' });
      add('text', { x: box.x + 3, y: box.y + 12, class: 'point-label', 'pointer-events': 'none' }, text.length > 39 ? text.slice(0, 38) + '…' : text);
    }
  }
  if (!points.length) add('text', { x: (L + R) / 2, y: (T + B) / 2, 'text-anchor': 'middle', class: 'axis-title' }, 'No eligible points for these filters.');
  if (visible.length < points.length) add('text', { x: L, y: B + 49, class: 'axis-text' }, `${points.length - visible.length} configurations outside the selected axis range. Select Full to see every point.`);
  $(`${chart}-chart`).replaceChildren(svg);
}
function showDetail(chart, configs) {
  $(`${chart}-detail`).innerHTML = configs.map(c => `<div><strong>${escape(name(c))} · ${escape(c.effort.toUpperCase())}</strong> — ${pct(c.accuracy)} accuracy (${Math.round(c.accuracy * c.samples)}/${c.samples}); ${pct(c.confidence)} mean confidence; ${usd(c.pricedCost)} ${escape(c.costKind)}; ${ms(c.totalSeconds === null ? null : c.totalSeconds / c.samples)} mean inference latency.<br><span>Provider pin: ${escape(c.provider || 'not applicable')} · returned: ${escape(c.actualProvider || 'not recorded')} · pin verified: ${c.kind === 'dl' ? 'not applicable' : c.providerVerified ? 'yes' : 'unverified'} · temperature: ${escape(c.temperature ?? 'API default')} · device: ${escape(c.device || 'API')} · ${escape(c.scoring)}. ${c.notes.map(escape).join(' ')} ${c.sources.map(link).join(' · ')}</span></div>`).join('<hr>');
}
function tableRows(configs) { return configs.filter(c => $('table-filter').value === 'all' || c.complete === ($('table-filter').value === 'complete')); }
function renderTable(configs) {
  $('result-table').innerHTML = tableRows(configs).map(c => `<tr><td><strong>${escape(name(c))}</strong><br><small>${escape(c.provider || c.device)}</small></td><td>${c.kind.toUpperCase()} / ${escape(c.effort)}</td><td>${pct(c.accuracy)}</td><td>${pct(c.confidence)}</td><td>${usd(c.pricedCost)}<br><small>${escape(c.costKind)}</small></td><td>${ms(c.totalSeconds === null ? null : c.totalSeconds / c.samples)}</td><td>${c.coverage}/${c.samples}</td><td><span class="status-badge ${c.complete ? 'complete' : 'excluded'}">${c.complete ? 'Complete' : 'Excluded'}</span><br><small>${escape(c.reasons.join('; '))}</small></td></tr>`).join('');
}
$('export').onclick = () => {
  const fields = ['model', 'kind', 'effort', 'provider', 'accuracy', 'confidence', 'pricedCost', 'costKind', 'totalSeconds', 'coverage', 'samples', 'complete', 'sources'];
  const quote = v => '"' + String(Array.isArray(v) ? v.join('; ') : v ?? '').replace(/"/g, '""') + '"';
  const text = [fields.join(','), ...tableRows(priced()).map(c => fields.map(k => quote(c[k])).join(','))].join('\r\n');
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' })), a = document.createElement('a');
  a.href = url; a.download = 'benchmark-comparison.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('refresh').onclick = load;
for (const id of ['cost-model', 'cost-reasoning', 'cost-scale', 'accuracy-range', 'dl-rate', 'custom-rate', 'cost-basis', 'cost-labels', 'confidence-model', 'confidence-reasoning', 'confidence-labels', 'confidence-range', 'table-filter']) $(id).addEventListener('change', render);
load();
