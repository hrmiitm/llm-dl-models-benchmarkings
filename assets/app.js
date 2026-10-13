import { parseCSV, parseFile, prepareData, priceConfig, plotEligible, adjacentConnections, connectionKey, costDomain, familyLabel, providerLabel, filterConfigs, sortConfigs } from './data.mjs';

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
const collapsed = new Set();
const families = () => ['llm', 'dl', 'decision'].filter(kind => $(`family-${kind}`).checked);
const confidenceNames = {model_reported: 'LLM self-reported', classifier_probability: 'DL class probability', top_class_probability: 'Decision class probability'};
const tableFields = [['model', 'Model'], ['kind', 'Family'], ['effort', 'Mode / reasoning'], ['provider', 'Provider / pin'], ['accuracy', 'Accuracy'], ['macroF1', 'Macro F1'], ['confidence', 'Confidence'], ['nativeConfidence', 'Native confidence'], ['pricedCost', 'Cost'], ['latencyMs', 'Mean latency'], ['totalSeconds', 'Total inference time'], ['coverage', 'Coverage'], ['samples', 'Test samples'], ['complete', 'Status'], ['costKind', 'Cost source'], ['modified', 'Source modified time']];
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
  for (const id of ['cost-model', 'confidence-model', 'table-model']) options(id, [['all', 'All models in selected families'], ...models]);
  options('table-sort', tableFields);
  const efforts = [...new Set(data.configs.filter(c => c.kind === 'llm').map(c => c.effort))].sort((a, b) => ['low', 'medium', 'high', 'default'].indexOf(a) - ['low', 'medium', 'high', 'default'].indexOf(b));
  for (const id of ['cost-reasoning', 'confidence-reasoning']) options(id, [['all', 'All reasoning levels'], ...efforts.map(e => [e, e[0].toUpperCase() + e.slice(1)])]);
  options('dl-rate', rates.presets.map(p => [p.id, p.title]));
}
function renderOverview() {
  const complete = data.configs.filter(c => c.complete), excluded = data.configs.filter(c => !c.complete);
  $('stats').innerHTML = [[data.sampleCount, 'Test sentences'], [new Set(data.configs.map(c => c.model)).size, 'Distinct models'], [complete.length, 'Complete configurations'], [excluded.length, 'Excluded configurations']].map(([v, label]) => `<div class="stat"><strong>${v}</strong><span>${label}</span></div>`).join('');
  const best = complete.length ? Math.max(...complete.map(c => c.accuracy)) : null;
  const bestModels = [...new Set(complete.filter(c => c.accuracy === best).map(name))];
  const cheapest = complete.filter(c => c.kind !== 'dl' && c.cost !== null).sort((a, b) => a.cost - b.cost)[0];
  $('summary').innerHTML = `<strong>${best === null ? 'No complete runs yet.' : `${pct(best)} is the highest observed accuracy.`}</strong> ${escape(bestModels.join(', '))}${cheapest ? `. Cheapest complete API run: ${escape(name(cheapest))} / ${escape(cheapest.effort)}, ${usd(cheapest.cost)} for all ${data.sampleCount} sentences.` : ''} With ${data.sampleCount} examples, one error changes accuracy by ${(100 / data.sampleCount).toFixed(2)} percentage points.`;
  const pairs = complete.filter(c => c.kind === 'llm' && c.effort === 'low').map(low => {
    const high = complete.find(c => c.effort === 'high' && connectionKey(c) === connectionKey(low));
    return high ? high.accuracy - low.accuracy : null;
  }).filter(v => v !== null);
  if (pairs.length) $('summary').innerHTML += `<p>For ${pairs.length} matched complete low/high reasoning pairs: ${pairs.filter(v => v > 1e-9).length} improved accuracy, ${pairs.filter(v => Math.abs(v) <= 1e-9).length} stayed equal, and ${pairs.filter(v => v < -1e-9).length} decreased. These are saved request settings; inference cost and confidence remain separate measurements.</p>`;
  const unpinned = complete.filter(c => c.kind === 'decision' && !c.provider).length;
  if (unpinned) $('summary').innerHTML += `<p>${unpinned} decision configurations used unpinned routing. Their recorded provider mix is shown in the table and point details.</p>`;
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
  return filterConfigs(configs, {families: families(), model: $(`${chart}-model`).value, effort: $(`${chart}-reasoning`).value, source: chart === 'confidence' ? $('confidence-source').value : 'all'});
}
function render() {
  if (!data) return;
  renderRate(); const configs = priced(), selected = families();
  $('family-all').checked = selected.length === 3; $('family-all').indeterminate = selected.length > 0 && selected.length < 3;
  $('family-preset').value = selected.length === 3 ? 'all' : selected.length === 1 ? selected[0] : 'custom';
  const counts = selected.map(kind => `${configs.filter(c => c.kind === kind).length} ${familyLabel(kind)}`);
  $('family-note').textContent = selected.length ? `Selected: ${counts.join(' + ')} configurations. These family filters apply to both graphs and the table; each view has additional independent filters.` : 'No families selected. Select a family or All to show results.';
  for (const chart of ['cost', 'confidence']) {
    const candidates = filtered(configs, chart);
    const eligible = candidates.filter(c => chart === 'cost' ? plotEligible(c, $('cost-scale').value) : c.complete && c.confidence !== null);
    const points = eligible.filter(c => !hidden[chart].has(c.model));
    const plotted = renderChart(chart, points); renderLegend(chart, eligible);
    const excluded = candidates.filter(c => !c.complete).length, unavailable = candidates.filter(c => c.complete && !eligible.includes(c)).length;
    $(`${chart}-detail`).textContent = 'Hover, tap, or keyboard-focus a point for its metrics, confidence source and provider details.';
    $(`${chart}-note`).textContent = `${plotted} of ${eligible.length} eligible configurations in view, each covering all ${data.sampleCount} sentences. ${excluded} incomplete runs excluded.${unavailable ? ` ${unavailable} complete runs omitted: ${chart === 'cost' ? 'missing cost/timing, incompatible GPU rate, or zero cost on a logarithmic axis (use Linear for zero).' : 'confidence missing for one or more examples.'}` : ''} ${chart === 'cost' ? 'Exact overlapping points share a tooltip; coordinates are never shifted.' : 'Confidence sources differ: LLM self-reported estimates vs DL/decision class probabilities. Native decision confidence is kept separate.'} Reasoning filters affect LLMs; DL/decision configurations stay visible when their families are selected. Dotted guides attach model names to markers.`;
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
  svg.append(svgEl('title', {}, chart === 'cost' ? 'Total cost versus full-test sentiment accuracy' : 'Mean selected-label confidence versus full-test sentiment accuracy'));
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
  add('text', { x: (L + R) / 2, y: H - 23, 'text-anchor': 'middle', class: 'axis-title' }, chart === 'cost' ? `Total cost for all ${data.sampleCount} sentences · USD${log ? ' · logarithmic scale' : ' · linear scale'}` : 'Mean selected-label confidence (%)');
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
    cluster.sort((a, b) => ['high', 'medium', 'low', 'dl', 'decision', 'default'].indexOf(a.effort) - ['high', 'medium', 'low', 'dl', 'decision', 'default'].indexOf(b.effort));
    cluster.forEach((c, i) => {
      const [cx, cy] = coord(c), size = cluster.length > 1 ? Math.max(3, 10 - i * 2.5) : 6;
      const attrs = { fill: modelColors.get(c.model), stroke: '#fff', 'stroke-width': 1.3, class: 'plot-point', tabindex: '0', role: 'button', 'data-family': c.kind, 'data-model': c.model, 'data-effort': c.effort, 'aria-label': `${name(c)}, ${c.effort}, accuracy ${pct(c.accuracy)}, ${chart === 'cost' ? usd(c.pricedCost) : 'confidence ' + pct(c.confidence)}` };
      let marker;
      if (c.kind === 'dl') marker = add('path', { ...attrs, d: `M${cx},${cy - size} L${cx + size},${cy + size} L${cx - size},${cy + size} Z` });
      else if (c.kind === 'decision') marker = add('polygon', { ...attrs, points: Array.from({length: 6}, (_, j) => `${cx + size * Math.cos(j * Math.PI / 3)},${cy + size * Math.sin(j * Math.PI / 3)}`).join(' ') });
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
    const context = document.createElement('canvas').getContext('2d');
    context.font = '600 10.5px ' + getComputedStyle(document.body).fontFamily;
    const labels = [...new Map(visible.map(c => [c.model, c])).values()].sort((a, b) => coord(a)[1] - coord(b)[1]);
    const occupied = visible.map(c => { const [cx, cy] = coord(c); return { x: cx - 9, y: cy - 9, w: 18, h: 18 }; });
    for (const c of labels) {
      const [cx, cy] = coord(c), text = name(c).length > 39 ? name(c).slice(0, 38) + '…' : name(c), w = context.measureText(text).width + 8, h = 17;
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
      add('line', { x1: cx, y1: cy, x2: Math.max(box.x, Math.min(cx, box.x + w)), y2: box.y + h / 2, stroke: '#82968b', 'stroke-width': .8, 'stroke-dasharray': '1 3', opacity: .8, 'pointer-events': 'none' });
      add('text', { x: box.x + 3, y: box.y + 12, class: 'point-label', 'pointer-events': 'none' }, text);
    }
  }
  if (!points.length) add('text', { x: (L + R) / 2, y: (T + B) / 2, 'text-anchor': 'middle', class: 'axis-title' }, 'No eligible points for these filters.');
  if (visible.length < points.length) add('text', { x: L, y: B + 49, class: 'axis-text' }, `${points.length - visible.length} configurations outside the selected axis range. Select Full to see every point.`);
  svg.setAttribute('aria-label', `${chart === 'cost' ? 'Cost versus accuracy' : 'Confidence versus accuracy'}: ${visible.length} visible full-test configurations; ${points.length - visible.length} outside the selected axis range`);
  $(`${chart}-chart`).replaceChildren(svg);
  return visible.length;
}
function showDetail(chart, configs) {
  $(`${chart}-detail`).innerHTML = configs.map(c => `<div><strong>${escape(name(c))} · ${escape(familyLabel(c.kind))} / ${escape(c.effort)}</strong> — ${pct(c.accuracy)} accuracy (${Math.round(c.accuracy * c.samples)}/${c.samples}); ${pct(c.macroF1)} macro F1; ${pct(c.confidence)} mean selected-label confidence (${escape(confidenceNames[c.confidenceSource] || c.confidenceSource)})${c.nativeConfidence !== null ? `; ${pct(c.nativeConfidence)} native decision confidence (separate metric)` : ''}; ${usd(c.pricedCost)} ${escape(c.costKind)}; ${ms(c.latencyMs === null ? null : c.latencyMs / 1000)} mean inference latency.<br><span>${escape(providerLabel(c))} · returned: ${escape(c.actualProviders.join(', ') || 'not recorded')} ${c.responseModels.length ? ' · returned model: ' + escape(c.responseModels.join(', ')) : ''} · pin verified: ${c.kind === 'dl' ? 'not applicable' : !c.provider ? 'not pinned' : c.providerVerified ? 'yes' : 'unverified'} · temperature: ${escape(c.kind === 'decision' ? 'not requested' : c.kind === 'dl' ? 'not applicable' : c.temperature ?? 'API default')} · device: ${escape(c.device || 'API')} · ${escape(c.scoring)}. ${c.notes.map(escape).join(' ')} ${c.sources.map(link).join(' · ')}</span></div>`).join('<hr>');
}
function tableRows(configs) {
  return sortConfigs(filterConfigs(configs, {families: families(), model: $('table-model').value, status: $('table-filter').value, query: $('table-search').value}), $('table-sort').value, $('table-direction').value);
}
function groupLabel(c) {
  return $('table-group').value === 'model' ? c.model : $('table-group').value === 'kind' ? familyLabel(c.kind) : $('table-group').value === 'provider' ? providerLabel(c) : '';
}
function renderTable(configs) {
  const rows = tableRows(configs), grouped = $('table-group').value !== 'none', groups = new Map();
  for (const c of rows) { const label = groupLabel(c); if (!groups.has(label)) groups.set(label, []); groups.get(label).push(c); }
  const entries = [...groups.entries()];
  $('result-table').innerHTML = entries.map(([label, members], i) => {
    const key = $('table-group').value + ':' + label, folded = collapsed.has(key);
    const header = grouped ? `<tr class="group-row"><td colspan="11"><button type="button" data-group-index="${i}" aria-expanded="${!folded}">${folded ? '▸' : '▾'} ${escape(label)} <small>${members.length} configuration${members.length === 1 ? '' : 's'}</small></button></td></tr>` : '';
    return header + (folded && grouped ? '' : members.map(c => `<tr class="result-row" data-model="${escape(c.model)}" data-family="${c.kind}"><td><strong>${escape(name(c))}</strong></td><td>${escape(familyLabel(c.kind))} / ${escape(c.effort)}</td><td class="provider-cell">${escape(providerLabel(c))}</td><td>${pct(c.accuracy)}</td><td>${pct(c.macroF1)}</td><td>${pct(c.confidence)}<br><small>${escape(confidenceNames[c.confidenceSource] || c.confidenceSource)}</small></td><td>${pct(c.nativeConfidence)}</td><td>${usd(c.pricedCost)}<br><small>${escape(c.costKind)}</small></td><td>${ms(c.latencyMs === null ? null : c.latencyMs / 1000)}</td><td>${c.coverage}/${c.samples}</td><td><span class="status-badge ${c.complete ? 'complete' : 'excluded'}">${c.complete ? 'Complete' : 'Excluded'}</span><br><small>${escape(c.reasons.join('; '))}</small></td></tr>`).join(''));
  }).join('') || '<tr><td colspan="11">No configurations match these filters.</td></tr>';
  $('result-table').querySelectorAll('[data-group-index]').forEach(button => button.onclick = () => {
    const key = $('table-group').value + ':' + entries[Number(button.dataset.groupIndex)][0];
    collapsed.has(key) ? collapsed.delete(key) : collapsed.add(key); renderTable(priced());
  });
  $('table-fold').disabled = !grouped || !rows.length;
  $('table-fold').textContent = entries.some(([label]) => collapsed.has($('table-group').value + ':' + label)) ? 'Expand groups' : 'Collapse groups';
  $('table-note').textContent = `${rows.length} matching configurations in ${grouped ? entries.length + ' groups' : 'a flat list'}. Groups follow their first ranked configuration and keep original metrics. Choose No grouping for a global ranking of individual configurations.`;
  document.querySelectorAll('th[data-key]').forEach(th => {
    th.removeAttribute('aria-sort'); th.querySelector('.sort-indicator').textContent = '';
    if (th.dataset.key === $('table-sort').value) { th.setAttribute('aria-sort', $('table-direction').value === 'asc' ? 'ascending' : 'descending'); th.querySelector('.sort-indicator').textContent = $('table-direction').value === 'asc' ? '↑' : '↓'; }
  });
}
$('export').onclick = () => {
  const fields = ['model', 'kind', 'effort', 'provider', 'actualProviders', 'responseModels', 'accuracy', 'macroF1', 'confidence', 'confidenceSource', 'nativeConfidence', 'pricedCost', 'costKind', 'latencyMs', 'totalSeconds', 'coverage', 'samples', 'complete', 'scoring', 'sources'];
  const quote = v => '"' + String(Array.isArray(v) ? v.join('; ') : v ?? '').replace(/"/g, '""') + '"';
  const text = [fields.join(','), ...tableRows(priced()).map(c => fields.map(k => quote(c[k])).join(','))].join('\r\n');
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' })), a = document.createElement('a');
  a.href = url; a.download = 'benchmark-comparison.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('refresh').onclick = load;
for (const id of ['cost-model', 'cost-reasoning', 'cost-scale', 'accuracy-range', 'dl-rate', 'custom-rate', 'cost-basis', 'cost-labels', 'confidence-model', 'confidence-reasoning', 'confidence-labels', 'confidence-range', 'confidence-source', 'table-filter', 'table-model', 'table-group', 'table-sort', 'table-direction']) $(id).addEventListener('change', render);
$('table-search').addEventListener('input', () => renderTable(priced()));
for (const kind of ['llm', 'dl', 'decision']) $(`family-${kind}`).addEventListener('change', render);
$('family-all').onchange = () => { for (const kind of ['llm', 'dl', 'decision']) $(`family-${kind}`).checked = $('family-all').checked; render(); };
$('family-preset').onchange = () => {
  const value = $('family-preset').value;
  if (value !== 'custom') {
    for (const kind of ['llm', 'dl', 'decision']) $(`family-${kind}`).checked = value === 'all' || value === kind;
    for (const id of ['cost-model', 'confidence-model', 'table-model', 'cost-reasoning', 'confidence-reasoning', 'confidence-source', 'table-filter']) $(id).value = 'all';
    $('table-search').value = '';
  }
  render();
};
$('table-fold').onclick = () => {
  if ($('table-fold').textContent.startsWith('Expand')) collapsed.clear();
  else for (const c of tableRows(priced())) collapsed.add($('table-group').value + ':' + groupLabel(c));
  renderTable(priced());
};
document.querySelectorAll('button[data-sort]').forEach(button => button.onclick = () => {
  const key = button.dataset.sort;
  $('table-direction').value = $('table-sort').value === key ? ($('table-direction').value === 'asc' ? 'desc' : 'asc') : ['accuracy', 'macroF1', 'confidence', 'nativeConfidence'].includes(key) ? 'desc' : 'asc';
  $('table-sort').value = key; renderTable(priced());
});
$('reset-filters').onclick = () => {
  for (const kind of ['llm', 'dl', 'decision']) $(`family-${kind}`).checked = true;
  for (const id of ['cost-model', 'cost-reasoning', 'confidence-model', 'confidence-reasoning', 'confidence-source', 'table-model', 'table-filter']) $(id).value = 'all';
  $('cost-scale').value = 'log'; $('accuracy-range').value = 'full'; $('confidence-range').value = 'full';
  $('dl-rate').value = rates.presets[0].id; $('custom-rate').value = ''; $('cost-basis').value = 'inference';
  $('table-group').value = 'none'; $('table-sort').value = 'model'; $('table-direction').value = 'asc'; $('table-search').value = '';
  $('cost-labels').checked = $('confidence-labels').checked = true;
  hidden.cost.clear(); hidden.confidence.clear(); collapsed.clear(); render();
};
load();
