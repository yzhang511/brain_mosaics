// Exercise standalone atlas interactions with a minimal DOM/canvas harness.
// This checks JavaScript execution, not visual browser layout.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const elements=new Map();
const ctx=new Proxy({}, {get:()=>()=>{}});
function el(id){if(!elements.has(id))elements.set(id,{id,value:'',textContent:'',innerHTML:'',add(o){if(this.value==='')this.value=String(o.value);},getContext(){return ctx;},getBoundingClientRect(){return {left:0,top:0,width:620,height:330};}});return elements.get(id);}
el('data').textContent=data;el('metric').value='mean_conflict_other_TV';el('condition').value='0';
const sandbox={document:{getElementById:el,querySelectorAll:()=>[]},Option:function(text,value){this.text=text;this.value=value;},console};
vm.createContext(sandbox);vm.runInContext(code,sandbox,{timeout:20000});
let n=0;
for(let c=0;c<6;c++)for(let l of [0,6,7,11])for(let h of [0,8,11])for(let cond=0;cond<4;cond++){
    el('case').value=String(c);el('layer').value=String(l);el('head').value=String(h);el('condition').value=String(cond);
    el('condition').onchange();assert(el('context').innerHTML.includes('Neutral'));assert(el('profile').innerHTML.includes('Most positive token alignments'));n++;
}
el('keys').onclick({clientX:100,clientY:100});el('values').onclick({clientX:300,clientY:150});
for(const metric of ['context_half_life','persistent_half_life','mean_effective_slots','key_nearest_cos_mean','value_cos_among_key_neighbors']){el('metric').value=metric;el('metric').onchange();}
assert(!/\b(?:NaN|undefined)\b/.test(el('stats').innerHTML));
console.log(JSON.stringify({status:'passed',interaction_states:n,plot_clicks:2,overview_metrics:6,scope:'DOM/canvas stub smoke test; not a browser rendering test'}));
