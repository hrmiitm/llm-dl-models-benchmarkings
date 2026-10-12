import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parseCSV, parseFile, prepareData, priceConfig, plotEligible, adjacentConnections, costDomain } from '../assets/data.mjs';
const truth = [{id:'1',sentence:'Loss.',label:'negative'},{id:'2',sentence:'Meeting.',label:'neutral'},{id:'3',sentence:'Profit.',label:'positive'}];
const rows = truth.map(r => ({...r, model:'example', predicted_label:r.label, confidence:.9, inference_seconds:.1, api_cost_usd:.01, reasoning_effort:'low', requested_provider:'pin'}));
const file = (data, path='results/llm_predictions_low.jsonl', modified_at='2026-10-13T00:00:00Z') => ({path, data, modified_at});
const config = (r=rows, extra=[], registry={}) => prepareData([file(r), ...extra],truth,registry).configs[0];
const summary = {model:'example', samples:3, successful:3, failed:0, accuracy:1, mean_confidence:.9, reasoning_effort:'low',requested_provider:'pin',api_cost_usd:.03};
test('CSV preserves commas, quotes, multiline fields, BOM and CRLF', () => {
 assert.deepEqual(parseCSV('\uFEFFid,sentence,label\r\n1,"A, ""B""\nC",positive\r\n'), [{id:'1',sentence:'A, "B"\nC',label:'positive'}]);
 assert.throws(()=>parseCSV('a,b\n1,"oops'),/Unclosed quote/);
});
test('raw predictions are independently scored on all test examples', () => {
 const c=config(rows.map((r,i)=>i===0?{...r,predicted_label:'neutral'}:r));
 assert.equal(c.accuracy,2/3); assert.equal(c.confidence,.9); assert.equal(c.cost,.03); assert.ok(c.complete);
});
test('failed, missing, duplicate, foreign and text-mismatched examples exclude the whole run', () => {
 for (const r of [rows.slice(1), [...rows,rows[0]], rows.map((r,i)=>i===0?{...r,error:'timeout'}:r), rows.map((r,i)=>i===0?{...r,id:'other'}:r), rows.map((r,i)=>i===0?{...r,sentence:'Different test'}:r)]) {
  const c=config(r); assert.equal(c.complete,false); assert.equal(c.accuracy,null); assert.equal(c.confidence,null);
 }
});
test('summary cannot repair failed raw predictions or replace recomputed accuracy', () => {
 assert.equal(config(rows.map((r,i)=>i===0?{...r,error:'timeout'}:r),[file([summary],'results/llm_summary_low.json')]).complete,false);
 const c=config(rows,[file([{...summary,accuracy:0}],'results/llm_summary_low.json')]); assert.equal(c.accuracy,1); assert.match(c.notes.join(),/disagrees/);
});
test('partial or invalid confidence is not represented as a full-test mean', () => {
 for(const confidence of [null,2,-.1,'',true]) assert.equal(config(rows.map((r,i)=>i===0?{...r,confidence}:r)).confidence,null);
});
test('unknown cost remains unknown, zero is a real value, unbilled calls do not become subtotals', () => {
 assert.equal(config(rows.map(r=>({...r,api_cost_usd:0}))).cost,0);
 const extra=[file([summary],'results/llm_summary_low.json')];
 assert.equal(config(rows.map(r=>({...r,api_cost_usd:null,usage:{cost:.01}})),extra).cost,null);
 assert.equal(config(rows.map(r=>({...r,api_calls_without_cost:1})),extra).cost,null);
 const absent=rows.map(({api_cost_usd,...r})=>r); assert.equal(config(absent,extra).cost,.03);
});
test('API logs never double-count charges or samples', () => {
 const c=config(rows,[file(rows,'results/llm_api_calls_low.jsonl')]); assert.equal(c.cost,.03); assert.equal(c.coverage,3);
});
test('pins, temperatures and revisions stay separate; latest duplicate run replaces a whole run', () => {
 const a=file(rows), b=file(rows.map(r=>({...r,requested_provider:'other'})),'results/run2/llm_predictions_low.jsonl');
 assert.equal(prepareData([a,b],truth).configs.length,2);
 assert.equal(prepareData([a,file(rows.map(r=>({...r,temperature:.2})),'results/run2/llm_predictions_low.jsonl')],truth).configs.length,2);
 const latest=file(rows.slice(1),'results/run2/llm_predictions_low.jsonl','2026-10-14T00:00:00Z');
 const result=prepareData([a,latest],truth); assert.equal(result.configs.length,1); assert.equal(result.configs[0].complete,false);
});
test('summary-only data requires full coverage AND explicit matching test set', () => {
 const f=s=>file([s],'results/llm_summary_low.json');
 assert.equal(prepareData([f(summary)],truth).configs[0].complete,false);
 assert.equal(prepareData([f({...summary,test_set:'data/answer_key_30.csv'})],truth).configs[0].complete,true);
 assert.equal(prepareData([f({...summary,test_set:'data/answer_key_30.csv',failed:1})],truth).configs[0].complete,false);
 assert.equal(prepareData([f({...summary,dataset_ids:['1','2','3']})],truth).configs[0].complete,true);
});
test('future nested JSON and CSV field adapters normalize percentages and milliseconds', () => {
 const registry={adapters:[{pattern:'results/vendor/*',family:'llm',fields:{model:'model_name',accuracy:'metrics.score',confidence:'metrics.certainty',cost:'billing.usd',samples:'n',failed:'failures',totalSeconds:'timing.ms'},units:{accuracy:'percent',confidence:'percent',totalSeconds:'ms'},test_set:'data/answer_key_30.csv'}]};
 const result=prepareData([file({results:[{model_name:'new',metrics:{score:100,certainty:95},billing:{usd:.2},n:3,failures:0,timing:{ms:300}}]},'results/vendor/new.json')],truth,registry);
 const c=result.configs[0]; assert.ok(c.complete); assert.equal(c.accuracy,1); assert.equal(c.confidence,.95); assert.equal(c.cost,.2); assert.equal(c.totalSeconds,.3);
 const csv=parseFile('results/vendor/new.csv','model_name,score,n,failures\nnew,100,3,0\n');
 registry.adapters[0].fields.accuracy='score'; assert.ok(prepareData([file(csv,'results/vendor/new.csv')],truth,registry).configs[0].complete);
});
test('mapped explicit null costs and times cannot be masked by summary totals', () => {
 const registry={adapters:[{pattern:'results/llm_predictions*',fields:{cost:'billing.usd',seconds:'timing.seconds'}}]};
 const c=config(rows.map(({api_cost_usd,inference_seconds,...r})=>({...r,billing:{usd:null},timing:{seconds:null}})),[file([{...summary,total_inference_seconds:3}],'results/llm_summary_low.json')],registry);
 assert.equal(c.cost,null); assert.equal(c.totalSeconds,null);
});
test('DL pricing uses selected measured timing, honors GPU scope and custom zero', () => {
 const c={kind:'dl',complete:true,accuracy:1,device:'Tesla T4',totalSeconds:3600,wallSeconds:7200};
 const p={id:'t4',hourly_usd:.526,gpu_match:'T4'};
 assert.equal(priceConfig(c,p,'').pricedCost,.526); assert.equal(priceConfig(c,p,'','wall').pricedCost,1.052);
 assert.equal(priceConfig({...c,device:'A100'},p,'').pricedCost,null);
 const zero=priceConfig(c,{id:'custom'},0); assert.equal(zero.pricedCost,0); assert.ok(plotEligible(zero)); assert.equal(plotEligible(zero,'log'),false);
 assert.equal(priceConfig(c,{id:'custom'},'').pricedCost,null); assert.equal(priceConfig(c,{id:'custom'},-1).pricedCost,null);
 assert.equal(priceConfig({kind:'llm',cost:.1},p,'').pricedCost,.1);
});
test('lines connect adjacent reasoning levels only, never cross pins or skip missing levels', () => {
 const c=config(); assert.equal(adjacentConnections([c,{...c,effort:'high'}]).length,0);
 assert.equal(adjacentConnections([c,{...c,effort:'medium'},{...c,effort:'high'}]).length,2);
 assert.equal(adjacentConnections([c,{...c,effort:'medium',provider:'another'}]).length,0);
});
test('invalid truth and unrecognized files are reported instead of silently scored', () => {
 assert.throws(()=>prepareData([file(rows)],[...truth,truth[0]]),/duplicate IDs/);
 assert.equal(prepareData([file({foo:42},'results/new.json')],truth).warnings.length,1);
});
test('repository metrics agree with saved summaries for selected source runs', () => {
 const manifest=JSON.parse(fs.readFileSync(new URL('../results/manifest.json',import.meta.url)));
 const files=manifest.files.map(f=>({...f,data:parseFile(f.path,fs.readFileSync(new URL('../'+f.path,import.meta.url),'utf8'))}));
 const result=prepareData(files,parseCSV(fs.readFileSync(new URL('../'+manifest.ground_truth,import.meta.url),'utf8')));
 assert.ok(result.configs.length>0); assert.equal(new Set(result.configs.map(c=>c.id)).size,result.configs.length);
 assert.equal(result.sampleCount,30);
 assert.ok(result.configs.filter(c=>c.complete).every(c=>c.coverage===result.sampleCount));
 for(const f of files.filter(f=>/summary|evaluation/.test(f.path))) for(const row of f.data) {
  const c=result.configs.find(c=>c.model===row.model&&c.effort===(row.reasoning_effort||'dl')&&c.sources.includes(f.path));
  if(c?.complete) { assert.ok(Math.abs(c.accuracy-row.accuracy)<1e-12); if(c.kind==='llm') assert.ok(Math.abs(c.cost-row.api_cost_usd)<1e-12); }
 }
 assert.ok(manifest.files.every(f=>!f.path.includes('api_calls')&&!f.path.includes('smoke')));
});

test('empty and zero-only log domains remain finite and positive', () => {
 for(const values of [[],[0],[null,NaN,-1]]) { const [min,max]=costDomain(values,true); assert.ok(min>0&&max>min&&Number.isFinite(max)); }
 assert.deepEqual(costDomain([.001],true),[.0001,.01]);
 assert.deepEqual(costDomain([0],false),[0,.001]);
});
