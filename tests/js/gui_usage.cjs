const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.attrs = {}; this.className = ''; this.hidden = true; this.classList = {add(){}, remove(){}}; }
  append(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  setAttribute(k,v) { this.attrs[k] = v; }
  focus() { document.activeElement = this; }
}
const nodes = Object.fromEntries(['usage-window','usage-dialog','usage-heading','usage-form','usage-close'].map(k => [k,new Element('div')]));
global.$ = id => nodes[id];
global.document = {activeElement: new Element('button'), body: new Element('body')};
global.text = (tag,value,parent) => { const e = new Element(tag); e.textContent = value; parent.append(e); return e; };
global.words = {usage:'Usage', close:'Close', usage_window:'Window', all_time:'All time', no_calls:'No calls', no_calls_since:'No calls in {since}', older_limits:'Older limits', zero_limit:'Blocked {key}', removed_keys_note:'Removed keys'};
const source = fs.readFileSync(process.argv[2], 'utf8');
vm.runInThisContext(source.slice(source.indexOf('let usageFocus ='),source.indexOf('$("usage-close").addEventListener')));
let paths = [];
const data = {rows:[{provider:'groq',key:'#1 masked',calls:4,failed:1,limited:1,tokens_in:42,tokens_out:8,last_call:'2026-11-09T12:00'}],removed:true,limits:[{provider:'groq',key:'#1 masked',name:'requests',left:'0 / 0',reset:'',limits_current:true,exhausted_for_good:true},{provider:'groq',key:'removed',name:'tokens',left:'3 / 8',limits_current:false,exhausted_for_good:false}]};
global.fetch = async path => { paths.push(path); return {ok:true,json:async () => data}; };
const descendants = node => [node, ...node.children.flatMap(descendants)];
(async () => {
  openUsage();
  await new Promise(setImmediate);
  assert.equal(nodes['usage-window'].hidden,false);
  assert.equal(document.activeElement,nodes['usage-close']);
  const [selector,content] = nodes['usage-form'].children;
  assert.deepEqual(selector.children.map(o => o.value),['24h','7d','30d','all']);
  assert.equal(paths[0],'/usage?since=24h');
  assert.equal(descendants(content).filter(n => n.tagName === 'table').length,2);
  assert.ok(descendants(content).some(n => n.className === 'older'));
  assert.ok(descendants(content).some(n => n.textContent === 'Blocked groq #1 masked'));
  assert.ok(descendants(content).some(n => n.className === 'usage-totals'));
  assert.ok(descendants(content).some(n => n.textContent === 'Removed keys'));
  selector.children.find(o => o.value === '7d').onclick();
  await new Promise(setImmediate);
  assert.equal(paths.at(-1),'/usage?since=7d');
  data.rows = []; data.limits = [];
  selector.children.find(o => o.value === 'all').onclick();
  await new Promise(setImmediate);
  assert.equal(content.children[0].textContent,'No calls');
  assert.equal(selector.children.find(o => o.value === 'all').attrs['aria-checked'],'true');
  assert.equal(selector.children.find(o => o.value === '24h').attrs['aria-checked'],'false');
  const previous = new Element('button'); usageFocus = previous;
  closeUsage();
  assert.equal(nodes['usage-window'].hidden,true);
  assert.equal(document.activeElement,previous);
  const sidebar = source.slice(source.indexOf('function setupPanels()'),source.indexOf('let panelsLoading'));
  assert.ok(sidebar.indexOf('words.usage, openUsage') < sidebar.indexOf('words.settings}`'));
  assert.ok(!source.includes('type: "usage"'));
  assert.ok(!source.includes('moduleSnapshot.removed'));
})().catch(error => {console.error(error);process.exitCode=1;});
