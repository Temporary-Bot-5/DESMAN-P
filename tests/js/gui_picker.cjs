const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.attrs = {}; this.listeners = {}; this.hidden = false; this.value = ''; this.detaches = 0; this.className = ''; this.classList = {toggle() {}}; }
  // A node leaving the page loses its scroll, as in a browser.
  unscroll() { this.scrollTop = 0; for (const c of this.children) c.unscroll(); }
  append(child) {
    if (child.parentElement) { child.parentElement.children = child.parentElement.children.filter(c => c !== child); child.detaches++; child.unscroll(); }
    this.children.push(child); child.parentElement = this;
  }
  replaceChildren() { for (const child of this.children) { child.detaches++; child.parentElement = null; child.unscroll(); } this.children = []; }
  setAttribute(k, v) { this.attrs[k] = v; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(k, v) { (this.listeners[k] ||= []).push(v); }
  dispatchEvent(e) { if (e.type === "change") this.onchange?.(e); for (const fn of this.listeners[e.type] || []) fn(e); if (e.bubbles) this.parentElement?.dispatchEvent(e); }
  contains(node) { return node === this || this.children.some(c => c.contains(node)); }
  closest(selector) { return this.className.split(' ').includes(selector.slice(1)) ? this : this.parentElement?.closest(selector); }
  setCustomValidity() {}
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
  focus() { document.activeElement = this; }
  querySelectorAll(selector) {
    const all = this.children.flatMap(c => [c, ...c.querySelectorAll('*')]);
    if (selector === '*') return all;
    if (selector === '[role=option]') return all.filter(c => c.attrs.role === 'option');
    if (selector === '[role=listbox]') return all.filter(c => c.attrs.role === 'listbox');
    if (selector === 'article.task') return all.filter(c => c.tagName === 'ARTICLE' && c.className.split(' ')[0] === 'task');
    if (selector === 'article.module-card') return all.filter(c => c.tagName === 'ARTICLE' && c.className === 'module-card');
    if (selector === 'section') return all.filter(c => c.tagName === 'SECTION');
    if (selector === '[data-search]') return all.filter(c => c.dataset.search);
    if (selector === '.module-editor:not([hidden])') return all.filter(c => c.className === 'module-editor' && !c.hidden);
    if (selector === 'textarea') return all.filter(c => c.tagName === 'TEXTAREA');
    return [];
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
global.document = {createTextNode: text => {const e = new Element("span"); e.textContent=text;return e;}, activeElement: null, listeners: {}, createElement: tag => new Element(tag), addEventListener(name, fn) {this.listeners[name] = fn;}};
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
// Each word of a search on its own, in any order.
const models = picker(new Element('section'), 'Model', [{value:'a', label:'Qwen/Qwen2.5-32B-Instruct-GGUF'}, {value:'b', label:'Qwen/Qwen2.5-7B-Instruct-GGUF'}, {value:'c', label:'meta/Llama-3-32B'}]);
models.children[1].dispatchEvent({type:'click'});
const modelSearch = models.children[2].children[0], modelList = models.children[2].children[1];
modelSearch.value = '  32b   QWEN '; modelSearch.dispatchEvent({type:'input'});
assert.deepEqual(modelList.querySelectorAll('[role=option]').map(n => n.children[0].textContent), ['Qwen/Qwen2.5-32B-Instruct-GGUF']);
models.close();
trigger.dispatchEvent({type:'click'}); document.listeners.pointerdown({target: new Element('div')}); assert.equal(p.open, false);
const role = {providers:[
 {id:'groq',name:'Groq',models:['groq/one','groq/two'],keys:[{fingerprint:'a',label:'A'}]},
 {id:'ollama',name:'Ollama',models:['ollama/local'],keys:[]}
], candidate_keys:[{provider:'groq',fingerprint:'a',label:'A'},{provider:'other',fingerprint:'b',label:'B'}]};
global.fetch = async () => ({ok:true, json: async () => ({models:[{value:'groq/live',label:'live',known:false}]})});
const choices = rolePickers(panel, role, null, 'a');
assert.equal(choices.provider.value, 'a'); assert.equal(choices.model.value, ''); assert.equal(choices.key.value, 'a');
assert.equal(choices.model.requestValue, ''); // Auto is sent as Auto, never as the first model
assert.equal(panel.children.filter(n => n.dataset.picker).length, 3); // scope plus two module dropdowns
choices.model.value = 'groq/two'; choices.model.onchange({}); assert.equal(choices.key.value, 'a');
choices.provider.value = 'ollama'; choices.provider.onchange(); assert.equal(choices.key.value, '');
choices.model.value = ''; assert.equal(choices.model.requestValue, 'ollama/local'); // Ollama + Auto stays on the machine
choices.set('groq/two', 'a'); assert.equal(choices.model.value, 'groq/two');
// Saved custom labels and removed Ollama models stay selectable.
for (const saved of ['groq/custom-label', 'ollama/removed']) {
  choices.set(saved, saved.startsWith('ollama/') ? null : 'a');
  assert.equal(choices.model.value, saved);
  choices.model.children[1].dispatchEvent({type:'click'});
  assert.ok(choices.model.querySelectorAll('[role=option]').some(n => n.children[0].textContent === saved.split('/')[1]));
  choices.model.close();
}
const tasks = new Element('section');
const waitingRoot = new Element('section');
const panelSearch = new Element('input'), modulesRoot = new Element('div');
global.$ = id => ({waiting: waitingRoot, 'panel-search': panelSearch, modules: modulesRoot})[id] || (assert.equal(id, 'tasks'), tasks);
global.currentSession = 's1'; global.terminalOpen = false;
global.expanded = new Set(['t']);
global.moduleEditors = new Map();
global.moduleSnapshot = {roles:[role], roles_by_scope:{task:['executor']}};
role.role = 'executor';
Object.assign(words, {tasks:'Tasks', pinned:'Pinned', pin:'Pin', unpin:'Unpin', status_needs_human:'Needs human', model_usage:'{calls}/{tokens}', role_usage_task:'{calls}/{tokens}', change:'Change', pause:'Pause', resume:'Resume', replace:'Replace', chats_label:'Chats: {names}', tasks_label:'Tasks: {names}', none:'none'});
global.iconAction = (parent, icon, label, action) => button(parent, label, () => action);
global.actionButton = (p, label, value) => button(p, label, () => typeof value === 'function' ? value() : value);
global.updateActions = () => {};
vm.runInThisContext(source.slice(source.indexOf('function moduleCard('), source.indexOf('function showModules(')));
vm.runInThisContext(source.slice(source.indexOf('function showTasks('), source.indexOf('let panelsLoading')));
const moduleData = {module_id:'a1b2c3',scope:'task',scope_id:'t',role:'executor',choice_model:'groq/custom-label',choice_key:'a',model:'groq/openai/gpt-oss-20b', provider_name:'Groq', provider:'groq', key_label:'Key 1', module_source:'a'};
const task = {id:'t',session_id:'s1',goal:'Work',status:'running',done:0,total:1,subtasks:[{seq:1,ref:'t/1',description:'Step',status:'running',version:1}],cards:[moduleData]};
showTasks({tasks:[task]});
let moduleNode = tasks.querySelectorAll('*').find(n => n.className === 'module-card');
const controls = moduleNode.querySelectorAll('*').filter(n => n.className === 'module-control');
assert.deepEqual(controls.map(n => n.textContent), ['Change', 'Pause', 'Replace']);
assert.ok(controls.every(n => n.tagName === 'BUTTON'));
assert.ok(moduleNode.querySelectorAll('*').some(n => n.textContent === 'gpt-oss-20b' && n.title === moduleData.model));
assert.ok(moduleNode.querySelectorAll('*').some(n => n.textContent === 'Groq · Key 1'));
assert.ok(moduleNode.querySelectorAll('*').some(n => n.className === 'module-id' && n.textContent === '#a1b2c3'));
// The panel search: every word in any order, over what a card shows and names (#id, ref).
panelSearch.value = 'A1B2C3 work'; filterPanels();
assert.equal(tasks.querySelectorAll('article.task')[0].hidden, false);
panelSearch.value = 'work nothing-like-this'; filterPanels();
assert.equal(tasks.querySelectorAll('article.task')[0].hidden, true);
panelSearch.value = ''; filterPanels();
assert.equal(tasks.querySelectorAll('article.task')[0].hidden, false);
controls[0].dispatchEvent({type:'click'});
const editor = tasks.querySelector('.module-editor:not([hidden])');
const modelPicker = editor.children[2];
assert.equal(modelPicker.value, 'groq/custom-label'); // "Only here" uses the same preservation.
modelPicker.children[1].dispatchEvent({type:'click'});
const search = modelPicker.children[2].children[0]; search.value = 'custom'; search.dispatchEvent({type:'input'});
const draft = tasks.querySelector('textarea'); draft.value = 'unsent';
const openList = modelPicker.children[2].children[1]; openList.scrollTop = 140;
showTasks({tasks:[{...task,status:'needs_human',needs_human:true,done:1,subtasks:[{...task.subtasks[0],status:'needs_human',version:2}]}]});
assert.equal(tasks.children[1].className, 'task waiting');
assert.equal(tasks.children[1].children[0].children[0].children[2].textContent, '1/1');
assert.equal(tasks.querySelector('.module-editor:not([hidden])'), editor);
assert.equal(modelPicker.open, true); assert.equal(search.value, 'custom'); assert.equal(document.activeElement, search);
assert.equal(openList.scrollTop, 140); // a poll does not send the open list back to its top
assert.equal(tasks.querySelector('textarea').value, 'unsent');
const decisions = tasks.querySelectorAll('*').find(n => n.className === 'decisions');
assert.equal(decisions.children[0].listeners.click[0]().version, 2);
assert.ok(tasks.querySelectorAll('*').some(n => n.className === 'status-chip' && n.textContent === words.status_needs_human));
// The chat the task was asked in shows the waiting step under it, with the same decisions.
assert.equal(waitingRoot.hidden, false);
const step = waitingRoot.children.find(n => n.className === 'waiting-step');
const stepDecisions = step.children.find(n => n.className === 'decisions');
const stepAnswer = step.children.find(n => n.tagName === 'TEXTAREA');
stepAnswer.value = 'use the other key';
assert.deepEqual(stepDecisions.children[0].listeners.click[0](), {type:'decide', ref:'t/1', version:2, decision:'retry', note:'use the other key'});
showTasks({tasks:[{...task,status:'needs_human',needs_human:true,done:1,subtasks:[{...task.subtasks[0],status:'needs_human',version:2,ref:'t/1',title:'Step'}]}]});
assert.equal(waitingRoot.children.find(n => n.className === 'waiting-step').children.find(n => n.tagName === 'TEXTAREA').value, 'use the other key');
// Focus in the field under the chat stays there when the Tasks panel redraws the same step.
const chatAnswer = waitingRoot.children.find(n => n.className === 'waiting-step').children.find(n => n.tagName === 'TEXTAREA');
chatAnswer.focus();
showTasks({tasks:[{...task,status:'needs_human',needs_human:true,done:1,subtasks:[{...task.subtasks[0],status:'needs_human',version:2,ref:'t/1'}]}]});
assert.equal(document.activeElement.parentElement.className, 'waiting-step');
// A step the person failed is their verdict: no field under the chat.
showTasks({tasks:[{...task,status:'failed',done:1,subtasks:[{...task.subtasks[0],status:'failed',version:3,ref:'t/1'}]}]});
assert.equal(waitingRoot.hidden, true);
document.activeElement = null;
currentSession = 'other';
showTasks({tasks:[{...task,status:'needs_human',needs_human:true,done:1,subtasks:[{...task.subtasks[0],status:'needs_human',version:2,ref:'t/1'}]}]});
assert.equal(waitingRoot.hidden, true); // another chat does not show it
currentSession = 's1';
modelPicker.close(); document.activeElement = null;
expanded.clear();
showTasks({tasks:[{...task,id:'normal',pinned:false}, {...task,id:'pinned',pinned:true}, {...task,id:'normal2',pinned:false}]});
assert.deepEqual(tasks.children.filter(n => n.tagName === 'H2').map(n => n.textContent), ['Pinned', 'Tasks']);
const taskCards = tasks.children.filter(n => n.tagName === 'ARTICLE');
assert.deepEqual(taskCards.map(n => n.children[0].children[1].attrs['aria-pressed']), ['true', 'false', 'false']);
assert.deepEqual(taskCards.map(n => n.children[0].children[1].title), ['Unpin', 'Pin', 'Pin']);

// Settings' 1.5-second refresh defers a requested redraw until the picker loses focus.
global.settingsData = null; global.settingsPending = null; global.settingsRedraw = false;
global.fetch = async () => ({ok: true, json: async () => ({})});
let draws = 0; global.drawSettings = () => draws++;
global.$ = () => panel;
vm.runInThisContext(source.slice(source.indexOf('async function refreshSettings('), source.indexOf('function openSettings(')));
(async () => {
  global.lineOpen = true; global.sending = false; global.settingsPending = null;
  global.crypto = {randomUUID: () => 'download-action'};
  let download = null;
  global.act = action => {download = action;};
  const local = rolePickers(new Element('section'), role, 'ollama/local', null);
  local.model.value = 'hf:author/repo';
  let stopped = false;
  local.model.onchange({stopImmediatePropagation() {stopped = true;}});
  assert.equal(stopped, true); assert.equal(local.model.value, 'ollama/local');
  assert.equal(download.change.model, 'author/repo');
  assert.equal(settingsPending, null); // a card download cannot wait forever for a closed settings modal
  const originalFetch = global.fetch;
  let deliver;
  global.fetch = () => new Promise(resolve => { deliver = resolve; });
  const changing = rolePickers(new Element('section'), role, 'groq/one', 'a');
  const loading = changing.model.onopen();
  changing.model.value = 'groq/two'; changing.model.onchange({});
  deliver({ok:true, json:async () => ({models:[{value:'groq/one',label:'one'}, {value:'groq/two',label:'two'}]})});
  await loading;
  assert.equal(changing.model.value, 'groq/two'); // a delayed listing never undoes a user's selection
  global.fetch = originalFetch;
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

// The visible command list is the cycling set until typing or Escape closes it.
{
  const commandInput = new Element('textarea');
  const suggestions = new Element('div');
  Element.prototype.scrollIntoView = function() {};
  const sandbox = {
    input: commandInput, suggestions, button, text, resize() {}, terminalOpen: false,
    p: {commands: {'/context': 'Request', '/config reset': 'Reset', '/config': 'Settings'}},
    complete: null,
  };
  const start = source.indexOf('      let matches = [];');
  const end = source.indexOf('\n    }\n    if (p.kind === "line" &&', start);
  vm.runInNewContext(source.slice(start, end), sandbox);
  commandInput.value = '/con'; commandInput.dispatchEvent({type:'input'});
  assert.equal(suggestions.children[0].children[0].textContent, '/config');
  function complete(key, shiftKey = false) {
    let prevented = false;
    const handled = sandbox.complete({key, shiftKey, preventDefault() {prevented = true;}});
    assert.equal(handled, prevented);
    return handled;
  }
  assert.equal(complete('Tab'), true); assert.equal(commandInput.value, '/config');
  complete('Tab'); assert.equal(commandInput.value, '/context');
  complete('Tab', true); assert.equal(commandInput.value, '/config');
  complete('Tab', true); assert.equal(commandInput.value, '/config reset');
  assert.equal(complete('Enter'), false); // Composer submits the current value.
  complete('Escape'); assert.equal(suggestions.children.length, 0);
  assert.equal(commandInput.value, '/config reset');
  assert.equal(complete('Tab'), false);
}

(async () => {
  const root = new Element('section'), saved = [];
  const sandbox = {
    root, d: {providers: [{id:'ollama', settings:[]}]}, text,
    words: {model:'Model', delete:'Delete', status_ready:'Ready', off:'Off'},
    settingSave(row, change, label) {saved.push({row, change, label});},
    picker() {return {dataset:{}, isConnected:true, value:'', setOptions() {}};},
    fetch: async () => ({ok:true, json:async () => ({
      options:[{downloaded:true, value:'tiny:latest', label:'tiny:latest', hint:'5 GB'},
        {downloaded:false, value:'hf.co/other', label:'other', hint:'8 GB'}],
      notes:[], reachable:true, host:'http://configured.test', selected:'tiny:latest',
    })}),
  };
  const start = source.indexOf('  } else if (settingsSection === "models") {') + '  } else if (settingsSection === "models") {'.length;
  const end = source.indexOf('  } else if (settingsSection === "network") {', start);
  vm.runInNewContext(source.slice(start, end), sandbox);
  await new Promise(setImmediate);
  assert.equal(saved.length, 1);
  assert.equal(saved[0].label, 'Delete');
  assert.equal(saved[0].change.operation, 'remove');
  assert.equal(saved[0].change.model, 'tiny:latest');
})().catch(e => {console.error(e); process.exitCode = 1;});
