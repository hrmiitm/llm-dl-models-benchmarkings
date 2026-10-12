export const EFFORTS = ['low', 'medium', 'high'];
export const LABELS = ['negative', 'neutral', 'positive'];
export const finite = value => typeof value === 'number' && Number.isFinite(value);
const validCount = n => finite(n) && n >= 0;
const probability = n => finite(n) && n >= 0 && n <= 1;
function savedCharge(row) {
  if(!row || (validCount(row.api_calls_without_cost) && row.api_calls_without_cost > 0)) return null;
  return [row.api_cost_usd,row.cost_usd,row.total_cost_usd,row.usage?.cost].find(validCount) ?? null;
}
const mean = values => values.length ? values.reduce((a,b) => a+b,0)/values.length : null;
const sumIfKnown = values => values.length && values.every(validCount) ? values.reduce((a,b)=>a+b,0) : null;
export const nameOf = model => ({'yiyanghkust/finbert-tone':'FinBERT Tone', 'cardiffnlp/twitter-roberta-base-sentiment-latest':'Twitter RoBERTa', 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0':'DeBERTa Zero-shot', 'facebook/bart-large-mnli':'BART MNLI', 'gemini-3.1-flash-lite':'Gemini 3.1 Flash Lite', 'gemini-3.8-flash':'Gemini 3.8 Flash', 'qwen/qwen3.8-27b':'Qwen 3.8 27B'}[model] ?? model);

export function parseCSV(text) {
  const rows=[]; let row=[], field='', quoted=false;
  text=text.replace(/^\uFEFF/,'');
  for(let i=0;i<text.length;i++) {
    const c=text[i];
    if(quoted) { if(c==='"' && text[i+1]==='"') {field+='"'; i++;} else if(c==='"') quoted=false; else field+=c; }
    else if(c==='"' && !field) quoted=true;
    else if(c===',') {row.push(field); field='';}
    else if(c==='\n' || c==='\r') {if(c==='\r' && text[i+1]==='\n') i++; row.push(field); if(row.some(Boolean)) rows.push(row); row=[];field='';}
    else field+=c;
  }
  if(quoted) throw new Error('Unclosed quote in ground-truth CSV.');
  if(field || row.length) {row.push(field);rows.push(row);}
  const headers=rows.shift()??[];
  if(!headers.includes('id') || !headers.includes('label')) throw new Error('Ground truth needs id and label columns.');
  return rows.map(r=>Object.fromEntries(headers.map((h,i)=>[h,r[i]??''])));
}

export function scorePredictions(records, truth) {
  const byId=new Map(), warnings=[];
  for(const r of records) {
    const id=String(r.id);
    if(byId.has(id)) warnings.push(`Duplicate prediction ID ${id}.`);
    byId.set(id,r);
    if(!truth.has(id)) warnings.push(`Unrecognized prediction ID ${id}.`);
  }
  const matrix=LABELS.map(()=>Array(4).fill(0)), rows=[];
  let successful=0, correct=0, sentenceMismatch=false;
  for(const [id,t] of truth) {
    const r=byId.get(id), p=r?.predicted_label;
    const sameSentence=!r?.sentence || !t.sentence || r.sentence.trim()===t.sentence.trim();
    if(!sameSentence) sentenceMismatch=true;
    const valid=!!r && LABELS.includes(p) && !r.error && sameSentence;
    const column=valid?LABELS.indexOf(p):3;
    matrix[LABELS.indexOf(t.label)][column]++;
    if(valid) {successful++; if(p===t.label) correct++;}
    rows.push({id, truth:t.label, predicted:valid?p:null, sentence:t.sentence, raw:r??null, valid});
  }
  if(sentenceMismatch) warnings.push('Prediction sentences disagree with the answer key for the same IDs.');
  const perClass=LABELS.map((label,i)=>{
    const tp=matrix[i][i], support=matrix[i].reduce((a,b)=>a+b,0);
    const predicted=matrix.reduce((n,row)=>n+row[i],0);
    const precision=predicted?tp/predicted:0, recall=support?tp/support:0;
    return {label,support,precision,recall,f1:precision+recall?2*precision*recall/(precision+recall):0};
  });
  const n=truth.size;
  const confidences=rows.filter(r=>r.valid && probability(r.raw?.confidence)).map(r=>r.raw.confidence);
  return {rows,matrix,perClass,samples:n,successful,failed:n-successful,
    accuracy:correct/n, macro_f1:mean(perClass.map(c=>c.f1)), precision:mean(perClass.map(c=>c.precision)),
    recall:mean(perClass.map(c=>c.recall)), balanced_accuracy:mean(perClass.filter(c=>c.support).map(c=>c.recall)),
    error_rate:1-correct/n, success_rate:successful/n,
    confidence:confidences.length===n?mean(confidences):null,confidence_samples:confidences.length,
    complete:byId.size===n && records.length===n && successful===n && !warnings.length, warnings};
}

function fileInfo(path) {
  const leaf=path.split('/').pop(), match=leaf.match(/^(llm|dl)_(predictions|summary|evaluation)(.*?)\.(jsonl|json)$/);
  if(!match) return null;
  return {family:match[1],type:match[2],suffix:match[3],run:path.slice(0,path.lastIndexOf('/')+1)+match[1]+match[3]};
}
const effortOf=(row,info)=>info.family==='dl'?'none':row.reasoning_effort??row.effort??(info.suffix.match(/(?:^|_)(low|medium|high)(?:_|$)/)?.[1]??'unrecorded');
function keyOf(row,info) {return JSON.stringify([row.model,effortOf(row,info),row.endpoint??'',row.temperature??null,row.requested_provider??null]);}

export function prepareData(loaded, truthRows) {
  const truth=new Map(), warnings=[], candidates=[], runs=new Map();
  for(const r of truthRows) {
    if(!r.id || !LABELS.includes(r.label) || truth.has(r.id)) throw new Error('Ground truth contains missing/duplicate IDs or invalid sentiment labels.');
    truth.set(String(r.id),r);
  }
  if(!truth.size) throw new Error('The answer key is empty.');
  for(const file of loaded) {
    const info=fileInfo(file.path); if(!info) continue;
    if(!Array.isArray(file.data)) {warnings.push(`${file.path}: expected an array or JSONL records.`);continue;}
    if(!runs.has(info.run)) runs.set(info.run,{info,files:[],predictions:[],summaries:[],evaluation:[]});
    const run=runs.get(info.run);run.files.push(file);
    if(info.type==='predictions') run.predictions.push(...file.data);
    else if(info.type==='summary') run.summaries.push(...file.data);
    else run.evaluation.push(...file.data);
  }
  for(const run of runs.values()) {
    const groups=new Map();
    for(const r of [...run.summaries,...run.predictions]) {
      if(typeof r.model!=='string') {warnings.push(`${run.info.run}: record missing a model name.`);continue;}
      const key=keyOf(r,run.info); if(!groups.has(key)) groups.set(key,{seed:r,records:[],summary:null});
    }
    for(const r of run.predictions) groups.get(keyOf(r,run.info))?.records.push(r);
    for(const r of run.summaries) {const g=groups.get(keyOf(r,run.info));if(g) g.summary=r;}
    for(const [key,g] of groups) {
      const isDL=run.info.family==='dl', seed=g.summary??g.seed, evaluation=run.evaluation.find(r=>r.model===seed.model), records=g.records;
      let metrics;
      if(records.length) metrics=scorePredictions(records,truth);
      else {
        const s={...evaluation,...seed}, n=s.samples;
        const successful=validCount(s.successful)?s.successful:validCount(s.failed)&&validCount(n)?n-s.failed:isDL&&finite(s.accuracy)?n:null;
        metrics={samples:n,successful,failed:validCount(n)&&validCount(successful)?n-successful:null,
          accuracy:finite(s.accuracy)?s.accuracy:null,macro_f1:finite(s.macro_f1)?s.macro_f1:null,
          success_rate:validCount(successful)&&n>0?successful/n:null,precision:null,recall:null,balanced_accuracy:null,
          error_rate:finite(s.accuracy)?1-s.accuracy:null,confidence:null,rows:[],matrix:null,perClass:[],complete:false,warnings:[]};
        warnings.push(`${nameOf(seed.model)}: prediction file unavailable; summary-only metrics shown in table, excluded from matched-sentence plots.`);
      }
      const matched=metrics.rows.filter(r=>r.raw).map(r=>r.raw);
      const latencies=metrics.rows.filter(r=>r.valid && validCount(r.raw?.inference_seconds)).map(r=>r.raw.inference_seconds);
      const recordedLatencies=matched.map(r=>r.inference_seconds).filter(validCount);
      const sumToken=field=>sumIfKnown(matched.map(r=>r[field]))??(validCount(seed[field])?seed[field]:null);
      const environment=loaded.find(f=>f.path===run.files[0].path.slice(0,run.files[0].path.lastIndexOf('/')+1)+'environment'+run.info.suffix+'.json')?.data ?? loaded.find(f=>f.path===run.files[0].path.slice(0,run.files[0].path.lastIndexOf('/')+1)+'environment.json')?.data;
      const c={...metrics,id:key,model:seed.model,name:nameOf(seed.model),family:isDL?'dl':'llm',isDL,
        requested_provider:seed.requested_provider??null,provider:seed.provider??null,provider_verified:seed.provider_verified??null,
        effort:effortOf(seed,run.info),endpoint:seed.endpoint??g.seed.endpoint??'',temperature:seed.temperature??g.seed.temperature??null,
        device:seed.device??matched.find(r=>r.device)?.device??environment?.gpu??'Unrecorded',
        mean_latency:latencies.length?mean(latencies):finite(seed.mean_latency_ms)?seed.mean_latency_ms/1000:finite(seed.average_inference_ms)?seed.average_inference_ms/1000:null,
        total_inference:recordedLatencies.length?recordedLatencies.reduce((a,b)=>a+b,0):seed.total_inference_seconds??seed.inference_total_seconds??null,
        wall_seconds:seed.model_run_wall_seconds??null,load_seconds:seed.load_seconds??null,warmup_seconds:seed.warmup_seconds??null,
        prompt_tokens:sumToken('prompt_tokens'),completion_tokens:sumToken('completion_tokens'),total_tokens:sumToken('total_tokens'),
        hourly_usd:validCount(seed.hourly_cost_usd)?seed.hourly_cost_usd:null,
        saved_cost:validCount(seed.api_calls_without_cost)&&seed.api_calls_without_cost>0?null:sumIfKnown(matched.map(savedCharge))??savedCharge(seed),
        unknown_api_attempts:validCount(seed.api_calls_without_cost)?seed.api_calls_without_cost:matched.reduce((n,r)=>n+(validCount(r.api_calls_without_cost)?r.api_calls_without_cost:0),0),
        sources:run.files.map(f=>f.path),prediction_source:run.files.find(f=>fileInfo(f.path)?.type==='predictions')?.path??null,
        modified_at:run.files.reduce((newest,f)=>Date.parse(f.modified_at??0)>Date.parse(newest)?f.modified_at:newest,'1970-01-01T00:00:00Z')};
      c.signature=JSON.stringify([c.endpoint,c.temperature,c.complete,metrics.rows.map(r=>r.id).sort()]);
      c.warnings.forEach(w=>warnings.push(`${c.name}, ${c.effort}: ${w}`));
      if(finite(seed.accuracy)&&finite(c.accuracy)&&Math.abs(seed.accuracy-c.accuracy)>0.0001) warnings.push(`${c.name}, ${c.effort}: saved accuracy differs from the score recomputed with this answer key.`);
      candidates.push(c);
    }
  }
  const unique=new Map();
  for(const c of candidates) {
    const prior=unique.get(c.id);
    if(prior) warnings.push(`${c.name}, ${c.effort}: multiple saved runs found; using the newest file set without pooling duplicate predictions.`);
    if(!prior || Date.parse(c.modified_at)>Date.parse(prior.modified_at)) unique.set(c.id,c);
  }
  const configs=[...unique.values()].sort((a,b)=>a.family.localeCompare(b.family)||a.name.localeCompare(b.name)||EFFORTS.indexOf(a.effort)-EFFORTS.indexOf(b.effort));
  for(const c of configs) if(!c.complete) warnings.push(`${c.name}, ${c.effort}: ${c.successful??'?'} / ${c.samples??'?'} valid answers; excluded when “Matched complete runs” is selected.`);
  return {configs,truth,warnings:[...new Set(warnings)]};
}

export function priceConfig(config, pricing, state) {
  const c={...config,cost:null,cost_kind:'Unknown',cost_note:''};
  if(c.isDL) {
    const preset=pricing.dl_presets?.find(p=>p.id===state.dlRate);
    const rate=state.dlRate==='custom'?state.customRate:state.dlRate==='saved'?c.hourly_usd:preset?.hourly_usd;
    const compatible=!preset?.gpu_match || c.device.toLowerCase().includes(preset.gpu_match.toLowerCase());
    if(validCount(rate)&&validCount(c.total_inference)&&compatible) {c.cost=c.total_inference/3600*rate;c.cost_kind='DL compute estimate';c.cost_note=`${rate.toFixed(6)} USD/hour × recorded inference time. ${c.device}.`;} else c.cost_note=compatible?'No valid hourly rate or inference timing.':'Selected rate does not match recorded GPU.';
    return c;
  }
  if(validCount(c.saved_cost)) {c.cost=c.saved_cost;c.cost_kind='Saved API charge';return c;}
  if(state.costMode==='saved') {c.cost_note='API charges were not saved.';return c;}
  const rate=pricing.llm_rates?.[c.model];
  if(!rate || (rate.endpoint_match && !c.endpoint.includes(rate.endpoint_match))) {c.cost_note='No matching endpoint price in results/pricing.json.';return c;}
  if(rate.valid_through && new Date().toISOString().slice(0,10)>rate.valid_through) {c.cost_note='Price reference expired; update results/pricing.json.';return c;}
  if(validCount(c.total_tokens)&&validCount(c.prompt_tokens)&&c.total_tokens<c.prompt_tokens) {c.cost_note='Recorded total tokens are less than input tokens.';return c;}
  const output=validCount(c.total_tokens)&&validCount(c.prompt_tokens)?c.total_tokens-c.prompt_tokens:c.completion_tokens;
  if(validCount(c.prompt_tokens)&&validCount(output)&&validCount(rate.input_usd_per_million)&&validCount(rate.output_usd_per_million)) {
    c.cost=(c.prompt_tokens*rate.input_usd_per_million+output*rate.output_usd_per_million)/1e6;
    c.cost_kind='LLM token estimate';c.billed_output_tokens=output;
    c.cost_note=`Uncached input + output/thinking tokens, standard paid rate checked ${pricing.checked_on}. Excludes proxy fees, discounts and any unreported retry usage.${c.unknown_api_attempts?' Some API attempts have unknown charges.':''}`;
  } else c.cost_note='Token totals are missing or inconsistent.';
  return c;
}

export function overallConfigs(configs) {
  const groups=new Map();for(const c of configs) {const key=JSON.stringify([c.model,c.requested_provider??null]);if(!groups.has(key)) groups.set(key,[]);groups.get(key).push(c);}
  return [...groups.values()].map(levels=>{
    if(levels[0].isDL) return levels[0];
    const average=key=>levels.every(c=>finite(c[key]))?mean(levels.map(c=>c[key])):null;
    const c={...levels[0],id:levels[0].model+'|'+(levels[0].requested_provider??'unrecorded')+'|overall',effort:'overall',levels:levels.map(c=>c.effort),
      configurations:levels.length,sources:[...new Set(levels.flatMap(c=>c.sources))],
      prediction_source:null,rows:levels.flatMap(c=>c.rows),complete:levels.every(c=>c.complete),
      successful:levels.reduce((s,c)=>s+(c.successful??0),0),failed:levels.reduce((s,c)=>s+(c.failed??0),0),
      attempts:levels.reduce((s,c)=>s+(c.samples??0),0)};
    for(const key of ['accuracy','macro_f1','balanced_accuracy','precision','recall','error_rate','success_rate','confidence','mean_latency','total_inference','wall_seconds','cost','prompt_tokens','completion_tokens','total_tokens']) c[key]=average(key);
    c.cost_kind=levels.every(l=>l.cost_kind===levels[0].cost_kind)?levels[0].cost_kind:'Mixed cost sources';
    c.matrix=levels.every(l=>l.matrix)?LABELS.map((_,i)=>Array.from({length:4},(_,j)=>levels.reduce((sum,l)=>sum+l.matrix[i][j],0))):null;
    c.cost_note='Mean cost of one run over the same sentences, averaged across selected available reasoning levels. Not an ensemble.';
    return c;
  });
}
