const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.attrs = {}; this.listeners = {}; this.hidden = false; this.value = ''; this.detaches = 0; this.className = ''; this.classList = {toggle() {}}; }
  append(child) {
    if (child.parentElement) { child.parentElement.children = child.parentElement.children.filter(c => c !== child); child.detaches++; }
    this.children.push(child); child.parentElement = this;
  }
  replaceChildren() { for (const child of this.children) { child.detaches++; child.parentElement = null; } this.children = []; }
  setAttribute(k, v) { this.attrs[k] = v; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(k, v) { (this.listeners[k] ||= []).push(v); }
  dispatchEvent(e) { if (e.type === "change") this.onchange?.(e); for (const fn of this.listeners[e.type] || []) fn(e); if (e.bubbles) this.parentElement?.dispatchEvent(e); }
  contains(node) { return node === this || this.children.some(c => c.contains(node)); }
  closest(selector) { return this.className.split(' ').includes(selector.slice(1)) ? this : this.parentElement?.closest(selector); }
  focus() { document.activeElement = this; }
  querySelectorAll(selector) {
    const all = this.children.flatMap(c => [c, ...c.querySelectorAll('*')]);
    if (selector === '*') return all;
    if (selector === '[role=option]') return all.filter(c => c.attrs.role === 'option');
    if (selector === '.module-editor:not([hidden])') return all.filter(c => c.className === 'module-editor' && !c.hidden);
    if (selector === 'textarea') return all.filter(c => c.tagName === 'TEXTAREA');
    return [];
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
global.document = {activeElement: null, listeners: {}, createElement: tag => new Element(tag), addEventListener(name, fn) {this.listeners[name] = fn;}};
global.text = (tag, value, parent) => { const e = new Element(tag); e.textContent = value; parent.append(e); return e; };
global.button = (p, label, fn) => {const b = text('button', label, p); b.addEventListener('click', fn); return b;};
global.words = {auto: 'Auto', model: 'Model', provider: 'Provider', key: 'Key', search: 'Search'};
const source = fs.readFileSync(process.argv[2], 'utf8');
vm.runInThisContext(source.slice(source.indexOf('let openPicker ='), source.indexOf('async function post(')));
const panel = new Element('section');
const p = picker(panel, 'Scope', [{value:'here', label:'Only here', disabled:true, hint:'This role cannot use it'}, {value:'role', label:'Everywhere'}, {value:'more', label:'Another choice', group:'Others'}], 'role');
let changes = 0; p.addEventListener('change', () => changes++);
const trigger = p.children[1], popup = p.children[2], input = popup.children[0], list = popup.children[1];
trigger.dispatchEvent({type:'click'});
assert.equal(p.open, true); assert.equal(trigger.attrs['aria-expanded'], 'true');
assert.equal(preserveEditor(panel), true);
assert.equal(list.children[0].children[1].textContent, 'This role cannot use it');
const key = name => input.dispatchEvent({type:'keydown', key:name, preventDefault(){}, stopPropagation(){}});
key('ArrowDown'); key('Enter');
assert.equal(p.value, 'role'); assert.equal(changes, 1); assert.equal(p.open, false);
trigger.dispatchEvent({type:'click'}); input.value = 'another'; input.dispatchEvent({type:'input'});
assert.equal(list.querySelectorAll('[role=option]').length, 1);
key('ArrowDown'); key('Enter'); assert.equal(p.value, 'more');
trigger.dispatchEvent({type:'click'}); key('Escape'); assert.equal(p.open, false);
trigger.dispatchEvent({type:'click'}); document.listeners.pointerdown({target: new Element('div')}); assert.equal(p.open, false);
const role = {providers:[
 {id:'groq',name:'Groq',models:['groq/one','groq/two'],keys:[{fingerprint:'a',label:'A'}]},
 {id:'ollama',name:'Ollama',models:['ollama/local'],keys:[]}
], candidate_keys:[{provider:'groq',fingerprint:'a',label:'A'},{provider:'other',fingerprint:'b',label:'B'}]};
const choices = rolePickers(panel, role, null, 'b');
assert.equal(choices.provider.value, ''); assert.equal(choices.model.value, ''); assert.equal(choices.key.value, 'b');
choices.provider.value = 'groq'; choices.provider.onchange();
assert.equal(choices.model.value, 'groq/one'); assert.equal(choices.key.value, '');
choices.key.value = 'a'; choices.model.value = 'groq/two'; choices.model.onchange(); assert.equal(choices.key.value, 'a');
choices.provider.value = 'ollama'; choices.provider.onchange(); assert.equal(choices.key.hidden, true); assert.equal(choices.key.value, '');
choices.set('groq/two', 'a'); assert.equal(choices.key.hidden, false); assert.equal(choices.model.value, 'groq/two');
// A poll must not detach even a closed picker while its editor stays open.
const tasks = new Element('section'), card = new Element('article'), editor = new Element('div');
editor.className = 'module-editor'; tasks.append(card); card.append(editor);
global.$ = id => {assert.equal(id, 'tasks'); return tasks;};
vm.runInThisContext(source.slice(source.indexOf('function showTasks('), source.indexOf('let panelsLoading')));
showTasks({tasks:[]}); assert.equal(tasks.children[0], card); assert.equal(card.detaches, 0); assert.equal(editor.detaches, 0);
editor.hidden = true; document.activeElement = null;
assert.equal(preserveEditor(tasks), false);
// Settings' 1.5-second refresh defers a requested redraw until the picker loses focus.
global.settingsData = null; global.settingsPending = null; global.settingsRedraw = false;
global.fetch = async () => ({ok: true, json: async () => ({})});
let draws = 0; global.drawSettings = () => draws++;
global.$ = () => panel;
vm.runInThisContext(source.slice(source.indexOf('async function refreshSettings('), source.indexOf('function openSettings(')));
(async () => {
  trigger.dispatchEvent({type:'click'});
  await refreshSettings(true); assert.equal(draws, 0);
  await refreshSettings(); assert.equal(draws, 0);
  p.close(); document.activeElement = null;
  await refreshSettings(); assert.equal(draws, 1);
  // Language changes override focus preservation and keep the current settings section.
  trigger.dispatchEvent({type:'click'});
  await refreshSettings(true, true); assert.equal(draws, 2);
  const originalRefresh = refreshSettings;
  let refreshed = null, modulesDrawn = 0, panelsDrawn = 0;
  global.refreshSettings = async (...args) => { refreshed = args; };
  global.setupPanels = () => panelsDrawn++;
  global.showModules = () => modulesDrawn++;
  const roleLabel = new Element('strong'); roleLabel.dataset.moduleRole = 'interlocutor';
  roleLabel.textContent = 'Собеседник';
  document.querySelectorAll = () => [roleLabel];
  global.$ = () => ({hidden:false});
  vm.runInThisContext(source.slice(source.indexOf('async function applyStrings('), source.indexOf('function setupPanels(')));
  await applyStrings({lang:'en', role_interlocutor:'Interlocutor'});
  assert.equal(roleLabel.textContent, 'Interlocutor');
  assert.equal(modulesDrawn, 1); assert.equal(panelsDrawn, 1);
  assert.deepEqual(refreshed, [true, true]);
  refreshed = null;
  await applyStrings({lang:'en', role_interlocutor:'Interlocutor'});
  assert.equal(refreshed, null);
  global.refreshSettings = originalRefresh;
})().catch(e => {console.error(e); process.exitCode = 1;});
