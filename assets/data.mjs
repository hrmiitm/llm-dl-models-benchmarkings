/** File adapters, full-test scoring and pricing. No model scores are embedded here. */
export function parseCSV(text) {
  const rows = [], row = []; let value = '', quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { value += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && (c === ',' || c === '\n' || c === '\r')) {
      row.push(value); value = '';
      if (c !== ',') { if (row.some(v => v !== '')) rows.push(row.splice(0)); else row.length = 0; }
      if (c === '\r' && text[i + 1] === '\n') i++;
    } else value += c;
  }
  if (quoted) throw new Error('Unclosed quote in CSV');
  if (value || row.length) { row.push(value); rows.push(row); }
  const header = rows.shift() || [];
  if (new Set(header).size !== header.length) throw new Error('Duplicate CSV columns');
  return rows.map(values => Object.fromEntries(header.map((key, i) => [key, values[i] ?? ''])));
}
export function parseFile(path, text) {
  if (path.endsWith('.csv')) return parseCSV(text);
  if (path.endsWith('.jsonl')) return text.split(/\r?\n/).filter(s => s.trim()).map(JSON.parse);
  return JSON.parse(text);
}
const aliases = {
  model: ['model', 'model_id', 'model_name', 'name'], kind: ['model_kind', 'model_type', 'family'],
  effort: ['reasoning_effort', 'effort', 'reasoning.effort', 'settings.reasoning_effort'],
  id: ['id', 'sample_id', 'sentence_id'], sentence: ['sentence', 'text', 'input.text'],
  label: ['predicted_label', 'prediction.label', 'prediction', 'label', 'sentiment'],
  confidence: ['confidence', 'mean_confidence', 'avg_confidence', 'confidence_percent', 'prediction.confidence'],
  nativeConfidence: ['native_confidence', 'mean_native_confidence'], macroF1: ['macro_f1', 'metrics.macro_f1'],
  accuracy: ['accuracy', 'metrics.accuracy', 'accuracy_percent', 'accuracy_pct'],
  cost: ['api_cost_usd', 'total_cost_usd', 'cost_usd', 'usage.cost', 'cost'],
  seconds: ['inference_seconds', 'latency_seconds', 'latency_ms'],
  totalSeconds: ['total_inference_seconds', 'inference_total_seconds', 'total_latency_seconds'],
  wallSeconds: ['model_run_wall_seconds', 'wall_seconds'],
  samples: ['samples', 'sample_count', 'n_samples', 'total_samples'],
  successful: ['successful', 'success_count', 'completed'], failed: ['failed', 'failures', 'error_count'],
  provider: ['requested_provider', 'provider_pin', 'provider'], endpoint: ['endpoint'],
  device: ['device', 'gpu_name'], temperature: ['temperature'], revision: ['model_revision'],
};
const get = (r, key) => key.split('.').reduce((v, k) => v != null && Object.hasOwn(Object(v), k) ? v[k] : undefined, r);
export const number = value => value === null || value === undefined || typeof value === 'boolean' || String(value).trim() === '' || !Number.isFinite(Number(value)) ? null : Number(value);
const nonnegative = v => { const n = number(v); return n !== null && n >= 0 ? n : null; };
export function fraction(v, unit = 'fraction') {
  const n = number(v); if (n === null) return null;
  const result = unit === 'percent' ? n / 100 : n;
  return result >= 0 && result <= 1 ? result : null;
}
const glob = (pattern, path) => new RegExp('^' + pattern.split('*').map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*') + '$').test(path);
function flatten(data, role = 'auto', inherited = {}) {
  if (Array.isArray(data)) return data.flatMap(r => flatten(r, role, inherited));
  if (!data || typeof data !== 'object') return [];
  const containers = ['predictions', 'records', 'summaries', 'results', 'data'];
  const shared = { ...inherited, ...data }; containers.forEach(k => delete shared[k]);
  const entries = containers.filter(k => Array.isArray(data[k]));
  if (entries.length) return entries.flatMap(k => flatten(data[k], k === 'predictions' ? 'predictions' : k === 'summaries' ? 'summary' : role, shared));
  return [{ row: shared, role }];
}
function normalize(row, file, adapter, registry) {
  const pick = key => {
    const paths = adapter.fields?.[key] ? [adapter.fields[key]] : [...(registry.aliases?.[key] || []), ...(aliases[key] || [key])];
    for (const path of paths) { const value = get(row, path); if (value !== undefined) return { value, path }; }
    return { value: undefined, path: '' };
  };
  const v = key => pick(key).value;
  const ratio = key => { const { value, path } = pick(key); return fraction(value, adapter.units?.[key] || (/percent|_pct$/.test(path) ? 'percent' : 'fraction')); };
  const seconds = key => { const { value, path } = pick(key); const n = nonnegative(value); return n === null ? null : n / (adapter.units?.[key] === 'ms' || path.endsWith('_ms') ? 1000 : 1); };
  let kind = String(adapter.family || v('kind') || '').toLowerCase();
  if (!['dl', 'llm', 'decision'].includes(kind)) kind = /\/decision_models\//.test(file.path) || /\/alpha\/decisions(?:$|[/?])/.test(row.endpoint || '') ? 'decision' : /(?:^|\/)dl_/.test(file.path) || row.device ? 'dl' : /(?:^|\/)llm_/.test(file.path) || row.usage || row.reasoning_effort || row.endpoint ? 'llm' : row.probabilities ? 'dl' : null;
  const provider = pick('provider');
  const pin = kind === 'decision' && provider.path === 'provider' && !adapter.fields?.provider ? '' : String(provider.value || '');
  const currency = String(row.currency || adapter.currency || 'USD').toUpperCase();
  return {
    model: String(v('model') || ''), kind, effort: kind !== 'llm' ? kind : String(v('effort') || 'default').toLowerCase(),
    id: v('id') == null ? null : String(v('id')), sentence: v('sentence'), label: String(v('label') || '').toLowerCase(),
    costPresent: pick('cost').path !== '' || number(row.api_calls_without_cost) > 0, secondsPresent: pick('seconds').path !== '',
    confidence: ratio('confidence'), nativeConfidence: ratio('nativeConfidence'), macroF1: ratio('macroF1'),
    confidenceSource: row.confidence_source || (kind === 'decision' ? 'top_class_probability' : kind === 'dl' ? 'classifier_probability' : 'model_reported'), accuracy: ratio('accuracy'), cost: currency === 'USD' && !(number(row.api_calls_without_cost) > 0) ? nonnegative(v('cost')) : null,
    seconds: seconds('seconds'), totalSeconds: seconds('totalSeconds'), wallSeconds: seconds('wallSeconds'),
    samples: number(v('samples')), successful: number(v('successful')), failed: number(v('failed')),
    confidenceSamples: number(row.confidence_samples), provider: pin,
    actualProvider: String(row.provider || ''), providerVerified: row.provider_verified,
    routing: row.provider_routing || null, endpoint: String(v('endpoint') || ''), device: String(v('device') || ''),
    temperature: v('temperature') ?? null, dtype: String(row.dtype || ''), revision: String(v('revision') || ''),
    failedRow: Boolean(row.error) || row.success === false || ['failed', 'error'].includes(row.status),
    datasetIds: row.dataset_ids || row.sample_ids, testSet: row.test_set || adapter.test_set,
    source: file.path, modified: file.modified_at || '', raw: row,
  };
}
const configKey = r => JSON.stringify([r.model, r.kind, r.effort, r.provider, r.endpoint, r.temperature, r.revision, r.device, r.dtype, r.routing]);
export const connectionKey = r => JSON.stringify([r.model, r.kind, r.provider, r.endpoint, r.temperature, r.revision, r.device, r.dtype, r.routing]);
const runKey = path => path.replace(/(?:predictions|summary|evaluation)/g, 'run').replace(/\.(jsonl?|csv)$/, '');
const sumKnown = rows => rows.length && rows.every(n => n !== null) ? rows.reduce((a, b) => a + b, 0) : null;
export function prepareData(files, truthRows, registry = {}, groundTruth = 'data/answer_key_30.csv') {
  const truth = new Map(truthRows.map(r => [String(r.id), r]));
  if (!truth.size || truth.size !== truthRows.length || truthRows.some(r => !r.id || !['negative', 'neutral', 'positive'].includes(r.label))) throw new Error('Answer key is empty, has duplicate IDs or invalid labels');
  const groups = new Map(), warnings = [];
  for (const file of files) {
    if (/api_calls|environment|\/(?:manifest|formats|cloud_rates)\.json$/.test(file.path)) continue;
    const adapter = (registry.adapters || []).find(a => glob(a.pattern, file.path)) || {};
    const role = adapter.role || file.role || (/predictions/.test(file.path) ? 'predictions' : /summary|evaluation/.test(file.path) ? 'summary' : 'auto');
    let recognized = 0;
    for (const entry of flatten(file.data, role)) {
      const r = normalize(entry.row, file, adapter, registry);
      if (!r.model || !r.kind) continue;
      const type = entry.role === 'auto' ? r.id !== null ? 'predictions' : r.samples !== null || r.accuracy !== null ? 'summary' : null : entry.role;
      if (!['predictions', 'summary'].includes(type)) continue;
      recognized++;
      const signature = configKey(r), key = runKey(file.path) + signature;
      if (!groups.has(key)) groups.set(key, { signature, r, rows: [], summaries: [], sources: new Set(), modified: '' });
      const g = groups.get(key); g[type === 'predictions' ? 'rows' : 'summaries'].push(r); g.sources.add(file.path); g.modified = [g.modified, r.modified].sort().at(-1);
    }
    if (!recognized) warnings.push(`Unrecognized result schema: ${file.path}. Add an adapter in results/formats.json.`);
  }
  const candidates = [...groups.values()].map(g => {
    const summary = g.summaries.at(-1), base = g.rows[0] || summary, reasons = [], notes = [];
    let complete = false, accuracy = null, confidence = null, nativeConfidence = null, macroF1 = null, cost = null, totalSeconds = null, coverage = 0;
    if (g.rows.length) {
      const ids = new Set(); let correct = 0; const valid = [];
      for (const r of g.rows) {
        const expected = truth.get(r.id);
        const ok = expected && !ids.has(r.id) && !r.failedRow && ['negative', 'neutral', 'positive'].includes(r.label) && (r.sentence == null || String(r.sentence).trim() === expected.sentence.trim());
        if (!expected) reasons.push('Unknown sample ID');
        if (ids.has(r.id)) reasons.push('Duplicate sample ID');
        if (r.failedRow || !['negative', 'neutral', 'positive'].includes(r.label)) reasons.push('Failed or invalid prediction');
        if (expected && r.sentence != null && String(r.sentence).trim() !== expected.sentence.trim()) reasons.push('Sentence differs from answer key');
        ids.add(r.id);
        if (ok) { coverage++; valid.push(r); correct += r.label === expected.label; }
      }
      if (coverage !== truth.size || g.rows.length !== truth.size) reasons.push(`Incomplete matching predictions (${coverage}/${truth.size})`);
      complete = !reasons.length;
      if (complete) {
        accuracy = correct / truth.size;
        confidence = valid.every(r => r.confidence !== null) ? valid.reduce((s, r) => s + r.confidence, 0) / truth.size : null;
        nativeConfidence = valid.every(r => r.nativeConfidence !== null) ? valid.reduce((s, r) => s + r.nativeConfidence, 0) / truth.size : null;
        macroF1 = ['negative', 'neutral', 'positive'].reduce((sum, label) => {
          const tp = valid.filter(r => r.label === label && truth.get(r.id).label === label).length;
          const fp = valid.filter(r => r.label === label && truth.get(r.id).label !== label).length;
          const fn = valid.filter(r => r.label !== label && truth.get(r.id).label === label).length;
          return sum + (2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 0);
        }, 0) / 3;
      }
      cost = sumKnown(g.rows.map(r => r.cost)); totalSeconds = sumKnown(g.rows.map(r => r.seconds));
      // A summary may supply totals only when the corresponding raw field is absent, never when explicitly unknown.
      if (cost === null && g.rows.every(r => !r.costPresent)) cost = summary?.cost ?? null;
      if (totalSeconds === null && g.rows.every(r => !r.secondsPresent)) totalSeconds = summary?.totalSeconds ?? null;
      if (summary && complete && summary.accuracy !== null && Math.abs(summary.accuracy - accuracy) > 1e-9) notes.push('Saved summary accuracy disagrees; raw predictions used');
    } else {
      coverage = summary.successful ?? 0;
      const idsMatch = Array.isArray(summary.datasetIds) && summary.datasetIds.length === truth.size && new Set(summary.datasetIds.map(String)).size === truth.size && summary.datasetIds.every(id => truth.has(String(id)));
      if (summary.samples !== truth.size || !(summary.failed === 0 || summary.successful === truth.size) || summary.failedRow || (summary.failed !== null && summary.failed > 0)) reasons.push('Incomplete summary coverage');
      if (!idsMatch && summary.testSet !== groundTruth) reasons.push('Summary test set unverified; declare test_set in an adapter');
      if (summary.accuracy === null) reasons.push('Missing or invalid summary accuracy');
      complete = !reasons.length;
      if (complete) { accuracy = summary.accuracy; confidence = summary.confidenceSamples !== null && summary.confidenceSamples !== truth.size ? null : summary.confidence; nativeConfidence = number(summary.raw.native_confidence_samples) !== null && number(summary.raw.native_confidence_samples) !== truth.size ? null : summary.nativeConfidence; macroF1 = summary.macroF1; coverage = truth.size; }
      cost = summary.cost; totalSeconds = summary.totalSeconds;
      notes.push('Saved aggregate; not independently rescored from predictions');
    }
    const actualProviders = [...new Set(g.rows.length ? g.rows.map(r => r.actualProvider).filter(Boolean) : (Array.isArray(base.raw.providers) ? base.raw.providers : [base.actualProvider]).filter(Boolean))].sort();
    if (base.kind === 'decision' && !base.provider) notes.push('Provider routing was not pinned for this saved run. Returned providers: ' + (actualProviders.join(', ') || 'not recorded') + '.');
    const responseModels = [...new Set(g.rows.map(r => r.raw.response_model).filter(Boolean))].sort();
    const repairedSamples = g.rows.filter(r => r.raw.replaced_attempt).length;
    if (repairedSamples) notes.push(`${repairedSamples} failed prediction(s) replaced. Cost and mean inference latency cover retained attempts; earlier attempts are in the API logs. ${summary?.unpriced_api_attempts ?? 'Unknown number of'} earlier/current API attempt(s) have unknown charges.`);
    return { ...base, repairedSamples, id: g.signature, complete, accuracy, confidence, nativeConfidence, macroF1, actualProviders, responseModels, cost, totalSeconds,
      latencyMs: totalSeconds === null ? null : 1000 * totalSeconds / (g.rows.length || summary?.samples || truth.size), wallSeconds: summary?.wallSeconds ?? base.wallSeconds,
      coverage, samples: truth.size, reasons: [...new Set(reasons)], notes, sources: [...g.sources], modified: g.modified,
      scoring: g.rows.length ? 'Raw predictions / answer key' : 'Declared full-test summary',
      providerVerified: g.rows.length ? g.rows.every(r => r.providerVerified === true) : base.providerVerified === true,
    };
  });
  // Select a whole latest run, never combine separate datasets or duplicate exports.
  const latest = new Map();
  for (const c of candidates.sort((a, b) => a.modified.localeCompare(b.modified))) {
    const previous = latest.get(c.id);
    if (previous && previous.modified === c.modified && previous.scoring.startsWith('Raw') && !c.scoring.startsWith('Raw')) continue;
    latest.set(c.id, c);
  }
  return { configs: [...latest.values()].sort((a, b) => a.model.localeCompare(b.model) || a.effort.localeCompare(b.effort)), warnings, sampleCount: truth.size };
}
export function priceConfig(config, preset, customRate, basis = 'inference') {
  if (config.kind !== 'dl') return { ...config, pricedCost: config.cost, costKind: config.repairedSamples ? 'Recorded retained-attempt charges' : 'Recorded API charge' };
  const rate = preset.id === 'custom' ? nonnegative(customRate) : nonnegative(preset.hourly_usd);
  const seconds = basis === 'wall' ? config.wallSeconds : config.totalSeconds;
  const matches = !preset.gpu_match || config.device.toLowerCase().includes(preset.gpu_match.toLowerCase());
  return { ...config, pricedCost: matches && rate !== null && seconds !== null ? seconds * rate / 3600 : null, costKind: `DL estimate · ${basis === 'wall' ? 'model wall time' : 'inference only'}` };
}
export function plotEligible(c, scale = 'linear') { return c.complete && c.accuracy !== null && c.pricedCost !== null && (scale !== 'log' || c.pricedCost > 0); }
export function adjacentConnections(points) {
  const order = ['low', 'medium', 'high'], groups = new Map(), lines = [];
  for (const p of points) { const key = connectionKey(p); if (!groups.has(key)) groups.set(key, new Map()); groups.get(key).set(p.effort, p); }
  for (const group of groups.values()) for (let i = 1; i < order.length; i++) if (group.has(order[i - 1]) && group.has(order[i])) lines.push([group.get(order[i - 1]), group.get(order[i])]);
  return lines;
}

export function costDomain(values, log = false) {
  values = values.filter(v => Number.isFinite(v) && (log ? v > 0 : v >= 0));
  if (!values.length) return log ? [.00001, .1] : [0, 1];
  const minimum = Math.min(...values), maximum = Math.max(...values);
  if (!log) return [0, maximum > 0 ? maximum * 1.12 : .001];
  let min = 10 ** Math.floor(Math.log10(minimum)), max = 10 ** Math.ceil(Math.log10(maximum));
  if (min === max) { min /= 10; max *= 10; }
  return [min, max];
}

export const familyLabel = kind => ({llm: 'LLM', dl: 'DL', decision: 'Decision'}[kind] || kind);
export function providerLabel(c) {
  if (c.kind === 'dl') return c.device || 'Local compute';
  return c.provider ? `Pinned: ${c.provider}` : `Unpinned: ${(c.actualProviders || []).join(', ') || 'not recorded'}`;
}
export function filterConfigs(configs, {families = ['llm', 'dl', 'decision'], model = 'all', effort = 'all', status = 'all', query = '', source = 'all'} = {}) {
  query = query.trim().toLowerCase();
  return configs.filter(c => families.includes(c.kind) && (model === 'all' || c.model === model || c.kind === model)
    && (effort === 'all' || c.kind !== 'llm' || c.effort === effort)
    && (status === 'all' || c.complete === (status === 'complete'))
    && (source === 'all' || c.confidenceSource === source)
    && (!query || [c.model, c.raw?.model_alias, c.kind, c.effort, providerLabel(c), c.costKind, c.scoring].join(' ').toLowerCase().includes(query)));
}
export function sortConfigs(configs, key = 'model', direction = 'asc') {
  const value = c => key === 'provider' ? providerLabel(c) : key === 'kind' ? familyLabel(c.kind) : c[key];
  const multiplier = direction === 'desc' ? -1 : 1;
  return [...configs].sort((a, b) => {
    const av = value(a), bv = value(b), missing = v => v == null || typeof v === 'number' && !Number.isFinite(v);
    if (missing(av) || missing(bv)) return Number(missing(av)) - Number(missing(bv));
    const order = typeof av === 'number' && typeof bv === 'number' || typeof av === 'boolean' && typeof bv === 'boolean' ? Number(av) - Number(bv) : String(av).localeCompare(String(bv));
    return order * multiplier || a.model.localeCompare(b.model) || a.effort.localeCompare(b.effort);
  });
}
