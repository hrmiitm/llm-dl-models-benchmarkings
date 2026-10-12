import {EFFORTS, LABELS, finite, nameOf, parseCSV, prepareData, priceConfig, overallConfigs} from './benchmark-data.mjs';
const $=s=>document.querySelector(s);
const METRICS={accuracy:'Accuracy',macro_f1:'Macro F1',balanced_accuracy:'Balanced accuracy',precision:'Macro precision',recall:'Macro recall',error_rate:'Error rate',success_rate:'Successful answer rate',confidence:'Mean confidence'};
const XMETRICS={cost:'Total cost / run, USD',mean_latency:'Mean latency, seconds',total_inference:'Total recorded inference, seconds',wall_seconds:'Model run wall time, seconds'};
const palette=['#41729b','#be5947','#267b67','#9b7421','#835ab0','#278597','#74852f','#cb6588','#465c91','#966349'];
const q=new URLSearchParams(location.search);
const state={model:q.get('model')??'all',efforts:q.has('efforts')?q.get('efforts').split(',').filter(e=>EFFORTS.includes(e)):[...EFFORTS],view:q.get('view')==='overall'?'overall':'levels',x:XMETRICS[q.get('x')]?q.get('x'):'cost',y:METRICS[q.get('y')]?q.get('y'):'accuracy',xScale:q.get('xscale')==='linear'?'linear':'log',yScale:q.get('yscale')==='log'?'log':'linear',dlRate:q.get('dlrate')??'azure',customRate:q.has('custom')&&q.get('custom').trim()?Number(q.get('custom')):null,costMode:q.get('costmode')==='saved'?'saved':'estimate',completeOnly:q.get('complete')!=='false',labels:q.get('labels')!=='false',hidden:new Set(),sort:'accuracy',descending:true,inspect:null};
let data=null,pricing={dl_presets:[],llm_rates:{}}, sources=[], loading=false,loadWarnings=[],colors=new Map(),loadedAt=null;
const pct=n=>finite(n)?`${(n*100).toFixed(2)}%`:'Unknown';
const money=n=>finite(n)?`$${n.toFixed(n<.01?6:4)}`:'Unknown';
const duration=n=>finite(n)?n<1?`${(n*1000).toFixed(1)} ms`:`${n.toFixed(2)} s`:'Unknown';
const integer=n=>finite(n)?Math.round(n).toLocaleString():'—';
const level=c=>c.isDL?'GPU':c.effort==='overall'?'Overall':c.effort[0].toUpperCase()+c.effort.slice(1);
const node=(tag,text,attrs={})=>{const e=document.createElement(tag);if(text!==null)e.textContent=text;for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));return e;};
const svgNode=(tag,attrs={},text=null)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==null)e.textContent=text;return e;};
const color=c=>colors.get(typeof c==='string'?c:c.model)??palette[0];
const lowerIsBetter=()=>state.y==='error_rate';

function writeState(){const params=new URLSearchParams({model:state.model,efforts:state.efforts.join(','),view:state.view,x:state.x,y:state.y,xscale:state.xScale,yscale:state.yScale,dlrate:state.dlRate,costmode:state.costMode});if(state.customRate!==null)params.set('custom',state.customRate);if(!state.completeOnly)params.set('complete','false');if(!state.labels)params.set('labels','false');history.replaceState(null,'',location.pathname+'?'+params+location.hash);}
function syncControls(){
  $('#model-filter').value=state.model;for(const e of document.querySelectorAll('input[name=effort]'))e.checked=state.efforts.includes(e.value);
  for(const [id,key]of [['x-metric','x'],['y-metric','y'],['x-scale','xScale'],['y-scale','yScale'],['dl-rate','dlRate'],['cost-mode','costMode']])$('#'+id).value=state[key];
  $('#complete-only').checked=state.completeOnly;$('#point-labels').checked=state.labels;
  $('#custom-control').hidden=state.dlRate!=='custom';if(document.activeElement!==$('#custom-rate'))$('#custom-rate').value=state.customRate??'';
  $('#view-levels').setAttribute('aria-pressed',state.view==='levels');$('#view-overall').setAttribute('aria-pressed',state.view==='overall');
}
const familyFilter=c=>state.model==='all'||state.model==='family:'+c.family||state.model===c.model;
const baseSelection=()=>data.configs.filter(c=>familyFilter(c)&&(c.isDL||state.efforts.includes(c.effort)||!EFFORTS.includes(c.effort)));
const configSelection=()=>baseSelection().filter(c=>!state.hidden.has(c.model)).map(c=>priceConfig(c,pricing,state));
const selected=()=>state.view==='overall'?overallConfigs(configSelection()):configSelection();

async function getText(path){const response=await fetch(path,{cache:'no-store',signal:AbortSignal.timeout(20000)});if(!response.ok)throw new Error(`HTTP ${response.status}: ${path}`);return response.text();}
const getJSON=async path=>JSON.parse(await getText(path));
const safePath=path=>typeof path==='string'&&/^results\/[\w./-]+\.(json|jsonl)$/.test(path)&&!path.split('/').includes('..')&&!/(^|\/)(llm_smoke|smoke|smoke-test|decision_models|extended|initial-run)(\/|$)/.test(path);
async function load(){
  if(loading)return;loading=true;$('#refresh').disabled=true;$('#load-status').textContent='Reading saved results and recomputing metrics…';$('#error').hidden=true;
  try {
    const manifest=await getJSON('results/manifest.json');
    if(!Array.isArray(manifest.files))throw new Error('Invalid results manifest. Run python scripts/build_results_manifest.py.');
    if(typeof manifest.ground_truth!=='string'||!/^data\/[\w./-]+\.csv$/.test(manifest.ground_truth)||manifest.ground_truth.includes('..'))throw new Error('Invalid ground-truth path in manifest.');
    const [truthText,priceData]=await Promise.all([getText(manifest.ground_truth),getJSON('results/pricing.json').catch(()=>null)]);
    pricing=priceData?.currency==='USD'?priceData:{dl_presets:[],llm_rates:{}};
    loadWarnings=priceData?[]:['No pricing file loaded; cost remains unknown unless a saved charge/custom rate is available.'];
    const files=manifest.files.filter(f=>safePath(f.path));
    const results=await Promise.allSettled(files.map(async f=>{
      const text=await getText(f.path);
      const parsed=f.path.endsWith('.jsonl')?text.split(/\r?\n/).filter(line=>line.trim()).map((line,i)=>{try{return JSON.parse(line);}catch{throw new Error(`Invalid JSONL at ${f.path}:${i+1}`);}}):JSON.parse(text);
      return {...f,data:parsed};
    }));
    const loaded=[];results.forEach((r,i)=>{if(r.status==='fulfilled')loaded.push(r.value);else loadWarnings.push(`${files[i].path}: ${r.reason.message}`);});
    data=prepareData(loaded,parseCSV(truthText));
    if(!data.configs.length)throw new Error('No benchmark configurations found. Check the results manifest and saved files.');
    sources=[manifest.ground_truth,'results/manifest.json',...(priceData?['results/pricing.json']:[]),...loaded.map(f=>f.path)];
    loadedAt=new Date();
    const models=[...new Set(data.configs.map(c=>c.model))].sort((a,b)=>nameOf(a).localeCompare(nameOf(b)));
    colors=new Map(models.map((m,i)=>[m,palette[i%palette.length]]));
    const filter=$('#model-filter');filter.replaceChildren(node('option','All models · DL + LLM',{value:'all'}),node('option','Deep learning only',{value:'family:dl'}),node('option','Language models only',{value:'family:llm'}));
    models.forEach(m=>filter.append(node('option',nameOf(m),{value:m})));
    if(![...filter.options].some(o=>o.value===state.model))state.model='all';
    const rateSelect=$('#dl-rate');rateSelect.replaceChildren(node('option','Saved JSON hourly rate',{value:'saved'}));
    for(const p of pricing.dl_presets??[]){const option=node('option',p.label,{value:p.id});option.disabled=!!p.gpu_match&&data.configs.filter(c=>c.isDL).some(c=>!c.device.toLowerCase().includes(p.gpu_match.toLowerCase()));rateSelect.append(option);}
    rateSelect.append(node('option','Custom hourly rate',{value:'custom'}));
    if(![...rateSelect.options].some(o=>o.value===state.dlRate&&!o.disabled))state.dlRate='saved';
    $('#dataset-stats').replaceChildren(...[[data.truth.size,'sentences'],[models.length,'models'],[data.configs.length,'configurations']].map(([v,label])=>{const e=node('div',null);e.append(node('strong',v),node('small',label));return e;}));
    $('#content').hidden=false;syncControls();render();renderSources();
    $('#load-status').textContent=`${loaded.length} result files loaded · scores recomputed from saved predictions · updated ${loadedAt.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}`;
  }catch(error){$('#error').replaceChildren(node('strong','Could not load benchmark files. '),document.createTextNode(error.message+' Serve the project root with python -m http.server 8000, and generate the file index with python scripts/build_results_manifest.py.'));$('#error').hidden=false;$('#load-status').textContent='Data loading failed; no example scores have been substituted.';}
  finally{loading=false;$('#refresh').disabled=false;}
}

function renderPricing(){
  const note=$('#rate-note');note.replaceChildren();const preset=pricing.dl_presets?.find(p=>p.id===state.dlRate);
  const devices=[...new Set(data.configs.filter(c=>c.isDL).map(c=>c.device))].join(', ');
  if(preset){note.append(document.createTextNode(`${preset.hourly_usd?.toFixed(6)} USD/hour. ${preset.description} `));if(preset.source)note.append(node('a','Rate source ↗',{href:preset.source,target:'_blank',rel:'noopener'}));}
  else if(state.dlRate==='custom'){const valid=finite(state.customRate)&&state.customRate>=0;note.textContent=valid?`Custom rate: ${state.customRate.toFixed(6)} USD/hour. Applied to recorded DL inference time.`:'Enter a finite, nonnegative hourly rate. DL cost stays unknown until the rate is valid.';$('#custom-rate').setAttribute('aria-invalid',!valid);}
  else note.textContent='Use saved hourly rates if recorded; otherwise DL costs remain unknown. Select a provider or enter a custom scenario.';
  note.append(document.createTextNode(` Measurements: Colab / ${devices || 'GPU unrecorded'}. Pricing references checked ${pricing.checked_on??'date unrecorded'}.`));
}

function renderHeadlines(configs){
  const complete=configs.filter(c=>c.complete), quality=complete.filter(c=>finite(c[state.y]));
  const best=quality.length?(lowerIsBetter()?Math.min(...quality.map(c=>c[state.y])):Math.max(...quality.map(c=>c[state.y]))):null;
  const winners=quality.filter(c=>Math.abs(c[state.y]-best)<1e-8);
  const fastest=complete.filter(c=>finite(c.mean_latency)).sort((a,b)=>a.mean_latency-b.mean_latency)[0];
  const cheapest=complete.filter(c=>finite(c.cost)).sort((a,b)=>a.cost-b.cost)[0];
  const card=(title,value,note)=>{const d=node('div',null,{class:'headline'});d.append(node('span',title),node('strong',value),node('small',note));return d;};
  $('#headlines').replaceChildren(card(state.y==='confidence'?'HIGHEST REPORTED CONFIDENCE':`BEST ${METRICS[state.y].toUpperCase()}`,pct(best),winners.length?[...new Set(winners.map(c=>c.name))].join(' · '):'No complete comparable runs'),card('LOWEST RECORDED LATENCY',duration(fastest?.mean_latency),fastest?`${fastest.name} · ${level(fastest)}`:'No recorded latency'),card('LOWEST SAME-RUN COST',money(cheapest?.cost),cheapest?`${cheapest.name} · ${cheapest.cost_kind}`:'No known costs for complete runs'));
}

function pointDetails(c){
  const box=$('#point-detail');box.replaceChildren(node('strong',`${c.name} · ${level(c)}. `),document.createTextNode(`${METRICS[state.y]} ${pct(c[state.y])}; mean latency ${duration(c.mean_latency)}; ${money(c.cost)} / run (${c.cost_kind.toLowerCase()}). `));
  if(c.effort==='overall')box.append(document.createTextNode(`Mean across ${c.levels.join(', ')}; ${c.samples} unique sentences, ${c.attempts} repeated predictions. `));
  else box.append(document.createTextNode(`${c.successful??'?'} / ${c.samples??'?'} valid answers. `));
  box.append(document.createTextNode(c.cost_note+' '));
  if(c.prediction_source)box.append(node('a','Raw predictions ↗',{href:c.prediction_source,target:'_blank',rel:'noopener'}));
  state.inspect=c.id;$('#inspect').value=c.id;renderMatrix();
}
function pointAria(c){return `${c.name}, ${level(c)}: ${METRICS[state.y]} ${pct(c[state.y])}, ${XMETRICS[state.x]} ${state.x==='cost'?money(c.cost):duration(c[state.x])}. ${c.cost_kind}.`;}
function renderChart(configs){
  const root=$('#chart');root.replaceChildren();
  const eligible=configs.filter(c=>(!state.completeOnly||c.complete)&&finite(c[state.x])&&finite(c[state.y])&&(state.xScale!=='log'||c[state.x]>0)&&(state.yScale!=='log'||c[state.y]>0));
  const excluded=configs.length-eligible.length;
  $('#chart-heading').textContent=`${state.x==='cost'?'Cost':state.x==='mean_latency'?'Latency':state.x==='wall_seconds'?'Wall time':'Inference time'} vs ${METRICS[state.y]}`;
  $('#direction').textContent=state.y==='confidence'?'↑ More confident, not more accurate':'↖ Top-left is best';
  $('#chart-intro').textContent=`Left = ${state.x==='cost'?'cheaper':'faster'}. Up = ${state.y==='confidence'?'higher reported confidence (not necessarily more accurate)':lowerIsBetter()?'fewer errors':'better '+METRICS[state.y].toLowerCase()}. ${state.view==='levels'?'Thin lines connect Low → Medium → High when recorded non-effort settings and sentence sets match.':'Each LLM point averages the selected available reasoning levels. DL remains a single GPU configuration.'}`;
  $('#symbols').textContent=state.view==='levels'?'● Low   ◆ Medium   ■ High   ▲ DL · model = color':'● LLM overall   ▲ DL · model = color';
  $('#chart-caption').textContent=`${eligible.length} plotted configurations; ${excluded} excluded by coverage, missing values or log-axis limits. ${state.xScale==='log'?'Log spacing compares ratios; zero values are omitted.':'Linear spacing compares absolute differences.'} ${state.view==='overall'?'Overall cost/time is the mean per same-size run, not the sum of repeated runs.':''}`;
  if(!eligible.length){root.append(node('div','No points match this view. Try changing the metric, cost source, axis scale, or filters.',{class:'empty-chart'}));renderLegend();return;}
  const W=Math.max(360,Math.round(root.getBoundingClientRect().width)),H=W<600?385:445,L=W<600?56:76,R=28,T=72,B=H-69,PW=W-L-R,PH=B-T;
  let maxX=Math.max(...eligible.map(c=>c[state.x])),minX=Math.min(...eligible.map(c=>c[state.x]));
  let lo,hi;
  if(state.xScale==='log'){lo=Math.log10(minX);hi=Math.log10(maxX);const pad=Math.max((hi-lo)*.07,.08);lo-=pad;hi+=pad;}else{lo=0;hi=maxX>0?maxX*1.12:1;}
  let yLo=0,yHi=1;if(state.yScale==='log'){yLo=Math.min(-.05,Math.floor(Math.log10(Math.min(...eligible.map(c=>c[state.y]))))-.05);yHi=0;}
  const x=v=>L+((state.xScale==='log'?Math.log10(v):v)-lo)/(hi-lo)*PW;
  const y=v=>{const t=state.yScale==='log'?(Math.log10(v)-yLo)/(yHi-yLo):v;return lowerIsBetter()?T+t*PH:B-t*PH;};
  const svg=svgNode('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':`${XMETRICS[state.x]} versus ${METRICS[state.y]}, ${eligible.length} points`,'data-point-count':eligible.length,'data-x-scale':state.xScale});
  svg.append(svgNode('text',{x:L,y:27,class:'axis-title'},`${METRICS[state.y]} (%)`),svgNode('text',{x:L,y:45,class:'axis-guide'},state.y==='confidence'?'↑ MORE CONFIDENT':`↑ ${lowerIsBetter()?'LOWER':'HIGHER'} IS BETTER`));
  for(let i=0;i<5;i++){
    const value=state.yScale==='log'?10**(yLo+(yHi-yLo)*i/4):i/4,py=y(value);
    svg.append(svgNode('line',{x1:L,x2:W-R,y1:py,y2:py,class:'grid-line'}),svgNode('text',{x:L-9,y:py+4,'text-anchor':'end',class:'axis-text'},`${(value*100).toFixed(state.yScale==='log'?1:0)}`));
    const vx=state.xScale==='log'?10**(lo+(hi-lo)*i/4):lo+(hi-lo)*i/4,px=x(vx);
    const label=state.x==='cost'?`$${vx.toFixed(vx<.001?5:vx<.01?4:3)}`:vx<1?vx.toPrecision(2):vx.toFixed(vx<10?1:0);
    svg.append(svgNode('line',{x1:px,x2:px,y1:T,y2:B,class:'grid-line'}),svgNode('text',{x:px,y:B+23,'text-anchor':'middle',class:'axis-text'},label));
  }
  svg.append(svgNode('text',{x:L+PW/2,y:H-18,'text-anchor':'middle',class:'axis-title'},W<600?(state.x==='cost'?'Cost / run, USD →':state.x==='mean_latency'?'Mean latency, seconds →':XMETRICS[state.x]):XMETRICS[state.x]));
  if(state.view==='levels')for(const route of new Set(eligible.filter(c=>!c.isDL).map(c=>JSON.stringify([c.model,c.requested_provider])))){const levels=eligible.filter(c=>JSON.stringify([c.model,c.requested_provider])===route);for(let i=0;i<2;i++){const a=levels.find(c=>c.effort===EFFORTS[i]),b=levels.find(c=>c.effort===EFFORTS[i+1]);if(a&&b&&a.signature===b.signature)svg.append(svgNode('path',{d:`M${x(a[state.x])},${y(a[state.y])}L${x(b[state.x])},${y(b[state.y])}`,class:'connector',stroke:color(a)}));}}
  const occupiedLabels=[];
  for(const c of eligible){
    const px=x(c[state.x]),py=y(c[state.y]);let mark;
    if(c.isDL)mark=svgNode('polygon',{points:`${px},${py-7} ${px-6.5},${py+5.5} ${px+6.5},${py+5.5}`});
    else if(c.effort==='medium')mark=svgNode('polygon',{points:`${px},${py-6.5} ${px-6.5},${py} ${px},${py+6.5} ${px+6.5},${py}`});
    else if(c.effort==='high')mark=svgNode('rect',{x:px-5.5,y:py-5.5,width:11,height:11});
    else mark=svgNode('circle',{cx:px,cy:py,r:c.effort==='overall'?6.5:5.6});
    for(const[k,v]of Object.entries({fill:color(c),class:'mark',tabindex:0,role:'button','aria-label':pointAria(c),'data-model':c.model,'data-effort':c.effort,'data-x':c[state.x],'data-y':c[state.y]}))mark.setAttribute(k,v);
    mark.append(svgNode('title',{},pointAria(c)));
    for(const event of ['pointerenter','focus','click'])mark.addEventListener(event,()=>pointDetails(c));
    mark.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pointDetails(c);}});svg.append(mark);
    if(state.labels&&(c.isDL||((c.effort==='high'||c.effort==='overall')&&W>=620))){
      const preferLeft=px>W-180, textWidth=c.name.length*6.2;
      const options=[17,-13,31,-27,45].flatMap(dy=>[preferLeft,!preferLeft].map(left=>({left,dy})));
      let placement;
      for(const candidate of options){
        const tx=px+(candidate.left?-9:9),ty=py+candidate.dy;
        const box={left:candidate.left?tx-textWidth:tx,right:candidate.left?tx:tx+textWidth,top:ty-12,bottom:ty+2};
        if(box.left<L||box.right>W-R||box.top<T||box.bottom>B)continue;
        if(occupiedLabels.some(b=>box.left<b.right+5&&box.right>b.left-5&&box.top<b.bottom+3&&box.bottom>b.top-3))continue;
        placement={...candidate,tx,ty,box};break;
      }
      if(placement){occupiedLabels.push(placement.box);svg.append(svgNode('text',{x:placement.tx,y:placement.ty,'text-anchor':placement.left?'end':'start',class:'point-label'},c.name));}
    }
  }
  root.append(svg);renderLegend();
}
function renderLegend(){const root=$('#legend');root.replaceChildren();for(const model of new Set(baseSelection().map(c=>c.model))){const button=node('button',null,{type:'button','aria-pressed':!state.hidden.has(model)});button.append(node('span','',{class:'legend-dot',style:`--color:${color(model)}`}),document.createTextNode(nameOf(model)));button.addEventListener('click',()=>{state.hidden.has(model)?state.hidden.delete(model):state.hidden.add(model);render();});root.append(button);}}

function renderReasoning(){
  const root=$('#reasoning-cards');root.replaceChildren();const configs=configSelection().filter(c=>!c.isDL);
  for(const route of new Set(configs.map(c=>JSON.stringify([c.model,c.requested_provider])))){
    const group=configs.filter(c=>JSON.stringify([c.model,c.requested_provider])===route),model=group[0].model;
    const card=node('article',null,{class:'reason-card',style:`--color:${color(model)}`});card.append(node('h3',`${nameOf(model)}${group[0].requested_provider?' · '+group[0].requested_provider:''}`));
    for(const effort of EFFORTS){const c=group.find(c=>c.effort===effort);if(!c)continue;
      const row=node('div',null,{class:'reason-row'}),left=node('div',null),right=node('div',null);
      left.append(node('strong',`${effort==='low'?'●':effort==='medium'?'◆':'■'} ${level(c)}`));
      right.append(node('strong',pct(c[state.y])));const track=node('div',null,{class:'reason-track'});const visual=finite(c[state.y])?(lowerIsBetter()?1-c[state.y]:c[state.y]):0;track.append(node('span','',{style:`width:${Math.max(0,Math.min(100,visual*100))}%`}));right.append(track,node('small',`${duration(c.mean_latency)} · ${money(c.cost)} / run`));row.append(left,right);card.append(row);
    }root.append(card);
  }
  if(!root.children.length)root.append(node('p','Select an LLM to compare its reasoning levels.',{class:'reason-empty'}));
}
function sortConfigs(configs){return [...configs].sort((a,b)=>{const av=a[state.sort],bv=b[state.sort];if(av==null)return bv==null?0:1;if(bv==null)return-1;const compare=typeof av==='string'?av.localeCompare(bv):av-bv;return state.descending?-compare:compare;});}
function renderTable(configs){for(const button of document.querySelectorAll('[data-sort]'))button.parentElement.setAttribute('aria-sort',button.dataset.sort===state.sort?(state.descending?'descending':'ascending'):'none');const root=$('#table-body');root.replaceChildren();for(const c of sortConfigs(configs)){
  const tr=node('tr',null,{style:`--color:${color(c)}`,class:c.complete?'':'partial'}),name=node('td',null);name.append(node('strong',c.name),node('span',c.isDL?'DL':'LLM',{class:'tag'}),node('small',c.model));if(c.requested_provider)name.append(node('small',`Provider: ${c.requested_provider}${c.provider_verified?' · verified':''}`));tr.append(name);
  tr.append(node('td',level(c)),node('td',pct(c.accuracy)),node('td',pct(c.macro_f1)),node('td',duration(c.mean_latency)));
  const cost=node('td',null);cost.append(document.createTextNode(money(c.cost)),node('small',c.cost_kind));cost.title=c.cost_note;tr.append(cost);
  const answers=node('td',null);answers.append(document.createTextNode(`${c.successful??'?'} / ${c.attempts??c.samples??'?'}`));if(c.effort==='overall')answers.append(node('small',`${c.samples} unique · ${c.configurations} levels`));else if(c.failed)answers.append(node('small',`${c.failed} failed/missing`));tr.append(answers);
  const tokens=node('td',integer(c.total_tokens));if(c.isDL)tokens.textContent='Not applicable';else if(c.effort==='overall')tokens.append(node('small','mean / run'));tr.append(tokens);
  const source=node('td',null);c.sources.forEach((path,i)=>{if(i)source.append(document.createElement('br'));source.append(node('a',path.split('/').pop(),{href:path,target:'_blank',rel:'noopener'}));});tr.append(source);root.append(tr);
}if(!configs.length){const tr=node('tr',null);tr.append(node('td','No configurations selected.',{colspan:9}));root.append(tr);}}
function renderMatrix(){const configs=selected(),c=configs.find(c=>c.id===state.inspect)??configs[0];if(c){state.inspect=c.id;$('#inspect').value=c.id;}
  const root=$('#confusion');root.replaceChildren();if(!c?.matrix){root.append(node('div','Individual predictions are required for this breakdown.',{class:'empty'}));$('#matrix-note').textContent='';return;}
  $('#matrix-note').textContent=`${c.name} · ${level(c)} · ${c.effort==='overall'?`${c.attempts} repeated predictions across ${c.levels.length} selected levels (${c.samples} unique sentences). Counts are pooled here; chart metrics are means across runs.`:`${c.samples} sentences. Accuracy ${pct(c.accuracy)}.`}`;
  const table=node('table',null),thead=node('thead',null),header=node('tr',null);header.append(node('th','Actual ↓ / Predicted →',{scope:'col'}));[...LABELS,'No answer'].forEach(label=>header.append(node('th',label,{scope:'col'})));thead.append(header);table.append(thead);const body=node('tbody',null),max=Math.max(1,...c.matrix.flat());
  c.matrix.forEach((values,i)=>{const row=node('tr',null);row.append(node('th',LABELS[i],{scope:'row'}));values.forEach((v,j)=>{const cell=node('td',v);cell.style.background=`color-mix(in srgb, ${j===i?'var(--accent)':'#b46c4b'} ${Math.round(v/max*32)}%, var(--surface))`;row.append(cell);});body.append(row);});table.append(body);root.append(table);
}
function renderSources(){
  $('#source-list').replaceChildren(...sources.map(path=>{const li=node('li',null);li.append(node('a',path,{href:path,target:'_blank',rel:'noopener'}));return li;}));
  $('#loaded-at').textContent=`Loaded ${loadedAt.toLocaleString()}. Add new result files and use uv run python scripts/build_results_manifest.py to update the static file index; metrics are recalculated in your browser.`;
  const root=$('#pricing-sources');root.replaceChildren(node('h3','Reference pricing'));const list=node('ul',null);
  for(const [model,rate]of Object.entries(pricing.llm_rates??{})){const li=node('li',null);li.append(document.createTextNode(`${nameOf(model)}: $${rate.input_usd_per_million} / $${rate.output_usd_per_million} per million input / output tokens. `),node('a','Provider source ↗',{href:rate.source,target:'_blank',rel:'noopener'}));if(rate.valid_through)li.append(document.createTextNode(` Rate valid through ${rate.valid_through}.`));list.append(li);}root.append(list);if(pricing.note)root.append(node('p',pricing.note));
}
function render(){if(!data)return;
  const configs=selected();syncControls();renderPricing();renderHeadlines(configs);
  $('#point-detail').textContent='Hover, tap or keyboard-focus a point to inspect its score, timing and cost source.';
  $('#scope-note').textContent=`${configs.length} configurations shown · ${data.truth.size} unique sentences · ${state.view==='overall'?'LLM means across selected available levels; GPU models keep their single run.':'Low / medium / high are separate runs. GPU models have one configuration.'} Legend buttons toggle model visibility.`;
  const allNotes=[...loadWarnings,...data.warnings];if(configs.some(c=>c.cost_kind==='LLM token estimate'))allNotes.push('LLM billed charges are absent. Cost estimates include output/thinking tokens and assume uncached input, without Foundry markup or discounts.');if(configs.some(c=>!finite(c.cost)))allNotes.push('Some costs are unknown; those configurations remain in the table and can be compared on the latency axis.');
  $('#notices').hidden=!allNotes.length;$('#notice-list').replaceChildren(...[...new Set(allNotes)].map(note=>node('li',note)));
  const inspect=$('#inspect');inspect.replaceChildren(...configs.map(c=>node('option',`${c.name} · ${level(c)}`,{value:c.id})));
  renderChart(configs);renderReasoning();renderTable(configs);renderMatrix();
  const good=configs.filter(c=>c.complete&&finite(c.accuracy)),best=good.length?Math.max(...good.map(c=>c.accuracy)):null,names=[...new Set(good.filter(c=>Math.abs(c.accuracy-best)<1e-8).map(c=>c.name))];
  $('#reading').textContent=names.length?`${names.join(' and ')} ${names.length>1?'share the highest':'has the highest'} observed accuracy (${pct(best)}) in this selection. Use the latency axis to inspect speed separately; changing the DL rental rate changes estimated cost, not measured predictions or performance.`:'Select at least one complete configuration to compare scores on the same sentence set.';
}

$('#refresh').addEventListener('click',load);
for(const [id,key]of [['model-filter','model'],['x-metric','x'],['y-metric','y'],['x-scale','xScale'],['y-scale','yScale'],['cost-mode','costMode']])$('#'+id).addEventListener('change',event=>{state[key]=event.target.value;if(id==='model-filter')state.hidden.clear();render();writeState();});
$('#dl-rate').addEventListener('change',event=>{if(event.target.value==='custom'&&state.customRate===null)state.customRate=pricing.dl_presets?.find(p=>p.id===state.dlRate)?.hourly_usd??null;state.dlRate=event.target.value;render();writeState();});
$('#custom-rate').addEventListener('input',event=>{state.customRate=event.target.value.trim()?Number(event.target.value):null;render();writeState();});
for(const input of document.querySelectorAll('input[name=effort]'))input.addEventListener('change',()=>{state.efforts=[...document.querySelectorAll('input[name=effort]:checked')].map(e=>e.value);render();writeState();});
for(const [id,key]of [['complete-only','completeOnly'],['point-labels','labels']])$('#'+id).addEventListener('change',e=>{state[key]=e.target.checked;render();writeState();});
for(const view of ['levels','overall'])$('#view-'+view).addEventListener('click',()=>{state.view=view;state.inspect=null;render();writeState();});
$('#inspect').addEventListener('change',event=>{state.inspect=event.target.value;renderMatrix();});
for(const button of document.querySelectorAll('[data-sort]'))button.addEventListener('click',()=>{const field=button.dataset.sort;state.descending=state.sort===field?!state.descending:true;state.sort=field;renderTable(selected());});
$('#export').addEventListener('click',()=>{const columns=['model','requested_provider','provider','provider_verified','family','effort','samples','attempts','configurations','successful','failed','accuracy','macro_f1','balanced_accuracy','precision','recall','mean_latency','total_inference','cost','cost_kind','prompt_tokens','completion_tokens','total_tokens'];const escape=value=>'"'+String(value??'').replaceAll('"','""')+'"';const rows=[columns,...sortConfigs(selected()).map(c=>columns.map(key=>c[key]))];const url=URL.createObjectURL(new Blob([rows.map(r=>r.map(escape).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=node('a','',{href:url,download:`sentiment-benchmark-${state.view}.csv`});document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
function themeLabel(){$('#theme').textContent=document.documentElement.dataset.theme==='dark'?'Light mode':'Dark mode';}
$('#theme').addEventListener('click',()=>{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';try{localStorage.setItem('sentiment-theme',dark?'dark':'light');}catch{}themeLabel();});themeLabel();
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(data)renderChart(selected());},100);});
load();
