import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {parseCSV,scorePredictions,prepareData,priceConfig,overallConfigs} from '../assets/benchmark-data.mjs';

const truthRows=[{id:'1',label:'negative',sentence:'Loss increased.'},{id:'2',label:'neutral',sentence:'Meeting announced.'},{id:'3',label:'positive',sentence:'Profit increased.'}];
const truth=new Map(truthRows.map(r=>[r.id,r]));
const records=truthRows.map(r=>({id:r.id,sentence:r.sentence,predicted_label:r.label,model:'example',reasoning_effort:'low',endpoint:'https://proxy/gemini/chat/completions',temperature:0,inference_seconds:.1,prompt_tokens:10,completion_tokens:1,total_tokens:100}));
const loaded=(rows=records,extra=[])=>[{path:'results/llm_predictions_low.jsonl',modified_at:'2026-10-12T00:00:00Z',data:rows},...extra];
const baseConfig=()=>prepareData(loaded(),truthRows).configs[0];
const prices={checked_on:'2026-10-12',dl_presets:[{id:'test',hourly_usd:2,gpu_match:'T4'}],llm_rates:{example:{input_usd_per_million:1,output_usd_per_million:2,endpoint_match:'/gemini/'}}};
const state={costMode:'estimate',dlRate:'test'};

test('CSV handles quoted commas, escaped quotes, BOM and CRLF',()=>{
 const rows=parseCSV('\uFEFFid,sentence,label\r\n1,"He said ""profit, up""",positive\r\n');
 assert.deepEqual(rows,[{id:'1',sentence:'He said "profit, up"',label:'positive'}]);
 assert.throws(()=>parseCSV('id,label\n1,"positive'),/Unclosed quote/);
});
test('missing predictions remain errors in accuracy, F1 and coverage',()=>{
 const result=scorePredictions(records.slice(1),truth);
 assert.equal(result.accuracy,2/3);assert.equal(result.macro_f1,2/3);assert.equal(result.success_rate,2/3);
 assert.equal(result.matrix[0][3],1);assert.equal(result.complete,false);
});
test('sentence mismatch invalidates an otherwise correct label',()=>{
 const bad=records.map((r,i)=>i===0?{...r,sentence:'Another dataset.'}:r);
 const result=scorePredictions(bad,truth);assert.equal(result.accuracy,2/3);assert.ok(result.warnings.some(w=>w.includes('disagree')));
});
test('duplicate predictions never qualify as a complete matched run',()=>{
 const result=scorePredictions([...records,records[0]],truth);assert.equal(result.samples,3);assert.equal(result.complete,false);assert.ok(result.warnings.some(w=>w.includes('Duplicate')));
});
test('invalid confidence values do not become valid percentages',()=>{
 assert.equal(scorePredictions(records.map(r=>({...r,confidence:2})),truth).confidence,null);
});
test('partial confidence coverage is not presented as a full-set mean',()=>{
 const result=scorePredictions(records.map((r,i)=>({...r,confidence:i===0?.9:null})),truth);
 assert.equal(result.confidence,null);assert.equal(result.confidence_samples,1);
});
test('fresh duplicate files replace old runs instead of doubling samples',()=>{
 const newer=records.map(r=>({...r,predicted_label:'neutral'}));
 const result=prepareData([...loaded(),{path:'results/new-run/llm_predictions_low.jsonl',modified_at:'2026-10-12T01:00:00Z',data:newer}],truthRows);
 assert.equal(result.configs.length,1);assert.equal(result.configs[0].accuracy,1/3);assert.equal(result.configs[0].samples,3);
});
test('summary-only scores remain visible but do not qualify for matched plots',()=>{
 const result=prepareData([{path:'results/llm_summary_low.json',data:[{model:'example',reasoning_effort:'low',samples:3,accuracy:1,successful:3}]}],truthRows);
 assert.equal(result.configs[0].accuracy,1);assert.equal(result.configs[0].complete,false);
});
test('duplicate truth IDs stop scoring instead of inventing a denominator',()=>{
 assert.throws(()=>prepareData(loaded(),[...truthRows,truthRows[0]]),/duplicate IDs/);
});
test('token estimates include thinking hidden from completion_tokens',()=>{
 const config=baseConfig(),priced=priceConfig(config,prices,state);
 assert.equal(priced.billed_output_tokens,270);assert.equal(priced.cost,(30+270*2)/1e6);assert.equal(priced.cost_kind,'LLM token estimate');
});
test('saved API charges take priority, including actual zero-cost calls',()=>{
 const result=prepareData(loaded(records.map(r=>({...r,api_cost_usd:0}))),truthRows);
 const priced=priceConfig(result.configs[0],prices,state);assert.equal(priced.cost,0);assert.equal(priced.cost_kind,'Saved API charge');
});
test('known subtotal is not treated as full cost when retry charges are missing',()=>{
 const rs=records.map(r=>({...r,api_cost_usd:null,known_api_cost_usd:.2,api_calls_without_cost:1}));
 const config=prepareData(loaded(rs),truthRows).configs[0];assert.equal(config.saved_cost,null);
 assert.equal(priceConfig(config,prices,{...state,costMode:'saved'}).cost,null);
 assert.equal(priceConfig(config,prices,state).cost_kind,'LLM token estimate');
});
test('summary API totals are used when per-sentence costs are absent',()=>{
 const summaries={path:'results/llm_summary_low.json',data:[{...records[0],api_cost_usd:.5,api_calls_without_cost:0}]};
 const config=prepareData(loaded(records,[summaries]),truthRows).configs[0];assert.equal(priceConfig(config,prices,state).cost,.5);
});
test('unknown cost does not become zero and saved-only disables token estimates',()=>{
 assert.equal(priceConfig(baseConfig(),{llm_rates:{}},state).cost,null);
 assert.equal(priceConfig(baseConfig(),prices,{...state,costMode:'saved'}).cost,null);
});
test('inconsistent token totals and mismatched providers stay unpriced',()=>{
 assert.equal(priceConfig({...baseConfig(),total_tokens:2,prompt_tokens:3},prices,state).cost,null);
 assert.equal(priceConfig({...baseConfig(),endpoint:'https://proxy/other/'},prices,state).cost,null);
});
test('DL prices recorded inference only and respects GPU compatibility',()=>{
 const config={isDL:true,device:'Tesla T4',total_inference:3600,hourly_usd:null};
 assert.equal(priceConfig(config,prices,state).cost,2);
 assert.equal(priceConfig({...config,device:'A100'},prices,state).cost,null);
 assert.equal(priceConfig(config,prices,{dlRate:'custom',customRate:0}).cost,0);
 assert.equal(priceConfig(config,prices,{dlRate:'custom',customRate:-1}).cost,null);
});
test('overall means are per run, not an ensemble or a three-run charge',()=>{
 const config=baseConfig(),configs=[0,1,2].map((i)=>({...config,effort:['low','medium','high'][i],cost:(i+1)*.01,cost_kind:'LLM token estimate',accuracy:(i+1)/3}));
 const overall=overallConfigs(configs)[0];assert.equal(overall.samples,3);assert.equal(overall.attempts,9);assert.equal(overall.cost,.02);assert.equal(overall.accuracy,2/3);assert.equal(overall.matrix.flat().reduce((a,b)=>a+b,0),9);
});
test('actual result metrics agree with the independently saved evaluation', {skip: !fs.existsSync(new URL('../results/manifest.json',import.meta.url)) ? 'Import benchmark results and rebuild the manifest first.' : false}, ()=>{
 const manifest=JSON.parse(fs.readFileSync(new URL('../results/manifest.json',import.meta.url)));
 const files=manifest.files.map(f=>{const text=fs.readFileSync(new URL('../'+f.path,import.meta.url),'utf8');return {...f,data:f.path.endsWith('.jsonl')?text.trim().split(/\r?\n/).map(JSON.parse):JSON.parse(text)};});
 const result=prepareData(files,parseCSV(fs.readFileSync(new URL('../'+manifest.ground_truth,import.meta.url),'utf8')));
 const evaluation=files.filter(f=>f.path.endsWith('dl_evaluation.json')).flatMap(f=>f.data);
 for(const row of evaluation){const config=result.configs.find(c=>c.model===row.model);assert.ok(Math.abs(config.accuracy-row.accuracy)<1e-12);assert.ok(Math.abs(config.macro_f1-row.macro_f1)<1e-12);}
 assert.ok(result.configs.every(c=>c.complete));
 assert.equal(new Set(result.configs.map(c=>c.id)).size,result.configs.length);
 assert.ok(!manifest.files.some(f=>f.path.includes('smoke')));
});

test('provider pins keep configurations and overall averages separate',()=>{
 const a=records.map(r=>({...r,requested_provider:'openai',provider:'OpenAI',provider_verified:true}));
 const b=records.map(r=>({...r,requested_provider:'azure',provider:'Azure',provider_verified:true}));
 const configs=prepareData([...loaded(a),{path:'results/other/llm_predictions_reasoning_low.jsonl',data:b}],truthRows).configs;
 assert.equal(configs.length,2);assert.equal(overallConfigs(configs).length,2);
 assert.deepEqual(new Set(configs.map(c=>c.requested_provider)),new Set(['openai','azure']));
});
