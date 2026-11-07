"use strict";

const $ = id => document.getElementById(id);
let words = {};
let after = 0;
let state = 0;
let promptId = null;
let pendingId = null;
let transcriptVersion = null;
let ended = false;
let sending = false;
let lineOpen = false;
let terminalOpen = false;
const expanded = new Set();

function text(tag, value, parent) {
  const node = document.createElement(tag);
  node.textContent = value;
  parent.append(node);
  return node;
}

// Linkify text nodes only; Rich HTML has already been escaped by the server.
function linkify(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) {
    if (!walker.currentNode.parentElement.closest("a")) {
      nodes.push(walker.currentNode);
    }
  }
  for (const node of nodes) {
    const regex = /https?:\/\/[^\s<>"']+/g;
    let match;
    let last = 0;
    const fragment = document.createDocumentFragment();
    while ((match = regex.exec(node.data))) {
      fragment.append(document.createTextNode(node.data.slice(last, match.index)));
      const a = document.createElement("a");
      a.href = match[0];
      a.textContent = match[0];
      a.rel = "noopener noreferrer";
      a.target = "_blank";
      fragment.append(a);
      last = regex.lastIndex;
    }
    if (last) {
      fragment.append(document.createTextNode(node.data.slice(last)));
      node.replaceWith(fragment);
    }
  }
}

function rich(target, html) {
  target.innerHTML = html;
  linkify(target);
}

function button(parent, label, action) {
  const b = text("button", label, parent);
  b.type = "button";
  b.addEventListener("click", action);
  return b;
}

// One searchable picker for models, providers, keys, scopes and settings fields.
let openPicker = null;
let pickerSerial = 0;
function picker(parent, label, options = [], current = "") {
  const root = text("div", "", parent); root.className = "picker"; root.dataset.picker = "true";
  const title = text("span", label, root); title.id = `picker-label-${++pickerSerial}`;
  const trigger = button(root, "", () => root.open ? close() : show());
  trigger.setAttribute("role", "combobox"); trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-labelledby", title.id); trigger.setAttribute("aria-expanded", "false");
  const panel = text("div", "", root); panel.className = "picker-panel"; panel.hidden = true;
  const search = document.createElement("input"); search.type = "search"; search.placeholder = words.search;
  search.setAttribute("aria-label", words.search); search.setAttribute("role", "combobox"); search.setAttribute("aria-autocomplete", "list");
  panel.append(search);
  const list = text("div", "", panel); list.id = `picker-list-${pickerSerial}`; list.setAttribute("role", "listbox");
  list.setAttribute("aria-labelledby", title.id);
  for (const input of [trigger, search]) input.setAttribute("aria-controls", list.id);
  let value = current, all = options, matches = [], index = -1;
  function active() {
    [...list.querySelectorAll("[role=option]")].forEach((node, i) => {
      node.classList.toggle("selected", i === index);
      if (i === index) { search.setAttribute("aria-activedescendant", node.id); node.scrollIntoView?.({block: "nearest"}); }
    });
    if (index < 0) search.removeAttribute("aria-activedescendant");
  }
  function choose(option) {
    if (option.disabled) return;
    root.value = option.value;
    close(); trigger.focus({preventScroll: true});
    root.dispatchEvent(new Event("change", {bubbles: true}));
  }
  function render() {
    const q = search.value.toLocaleLowerCase();
    matches = all.filter(o => `${o.label} ${o.hint || ""}`.toLocaleLowerCase().includes(q));
    list.replaceChildren(); let group;
    for (const [i, option] of matches.entries()) {
      if (option.group && option.group !== group) { text("strong", option.group, list).className = "picker-group"; group = option.group; }
      const node = button(list, "", () => choose(option)); node.id = `${list.id}-${i}`;
      node.setAttribute("role", "option"); node.setAttribute("aria-selected", String(value === option.value));
      node.setAttribute("aria-disabled", String(!!option.disabled)); node.tabIndex = -1;
      text("span", option.label, node);
      if (option.hint) text("small", option.hint, node);
    }
    index = -1; active();
  }
  function close() {
    panel.hidden = true; root.open = false;
    trigger.setAttribute("aria-expanded", "false"); search.setAttribute("aria-expanded", "false");
    if (openPicker === root) openPicker = null;
  }
  function show() {
    openPicker?.close(); openPicker = root; root.open = true;
    panel.hidden = false; search.value = "";
    trigger.setAttribute("aria-expanded", "true"); search.setAttribute("aria-expanded", "true");
    render(); position(); search.focus(); root.onopen?.(); root.onsearch?.("");
  }
  function keys(e) {
    if (["ArrowDown", "ArrowUp"].includes(e.key)) {
      e.preventDefault(); if (!root.open) show();
      const delta = e.key === "ArrowDown" ? 1 : -1;
      for (let n = 0; n < matches.length; n++) {
        index = (index + delta + matches.length) % matches.length;
        if (!matches[index].disabled) break;
      }
      active();
    } else if (e.key === "Enter" && root.open) {
      e.preventDefault(); if (index >= 0) choose(matches[index]);
    } else if (e.key === "Escape" && root.open) {
      e.preventDefault(); e.stopPropagation(); close(); trigger.focus();
    } else if (e.key === "Tab") close();
  }
  trigger.addEventListener("keydown", keys); search.addEventListener("keydown", keys);
  search.addEventListener("change", e => e.stopPropagation());
  search.addEventListener("input", () => { render(); root.onsearch?.(search.value); });
  Object.defineProperty(root, "value", {get: () => value, set: next => {
    value = next ?? ""; trigger.textContent = all.find(o => o.value === value)?.label || value || words.auto;
  }});
  root.setOptions = next => { all = next; root.value = value; if (root.open) render(); };
  function position() {
    if (!trigger.getBoundingClientRect) return;
    const rect = trigger.getBoundingClientRect();
    const width = Math.min(Math.max(rect.width, 280), innerWidth - 16);
    panel.style.width = `${width}px`;
    panel.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - width - 8))}px`;
    const below = innerHeight - rect.bottom;
    panel.style.top = below >= 200 ? `${rect.bottom + 4}px` : "auto";
    panel.style.bottom = below >= 200 ? "auto" : `${innerHeight - rect.top + 4}px`;
  }
  root.position = position;
  root.close = close; root.checkValidity = () => true; root.value = current;
  return root;
}
document.addEventListener("pointerdown", e => { if (openPicker && !openPicker.contains(e.target)) openPicker.close(); });
document.addEventListener("scroll", () => openPicker?.position(), true);
globalThis.addEventListener?.("resize", () => openPicker?.position());
function preserveEditor(root) {
  return !!root.querySelector('.module-editor:not([hidden])') ||
    !!(openPicker && root.contains(openPicker)) ||
    !!document.activeElement?.closest?.('.picker') && root.contains(document.activeElement);
}

// Shared source/model editor. Only fingerprints and model labels leave this component.
function rolePickers(parent, role, chosen, fingerprint) {
  const providers = role.providers || [];
  const sources = [{value: "", label: words.auto}];
  for (const p of providers) {
    for (const k of p.keys) sources.push({value: k.fingerprint, label: `${p.name} · ${k.label}`, provider: p});
    if (p.id === "ollama") sources.push({value: "ollama", label: "Ollama", provider: p});
  }
  const provider = picker(parent, words.source, sources);
  const model = picker(parent, words.model);
  // Existing action contracts receive the selected source's fingerprint.
  const key = {get value() { return provider.value === "ollama" ? "" : provider.value; }};
  const note = text("small", "", parent); note.className = "model-note";
  let generation = 0, searchTimer, localChoices = [], previousModel = chosen || "";
  // Auto stays Auto for a key (the router asks that key's provider first); Auto with the
  // Ollama source means Ollama's own default model, so the choice stays on the machine.
  Object.defineProperty(model, "requestValue", {get: () => model.value ||
    (provider.value === "ollama" ? sources.find(s => s.value === "ollama")?.provider?.models?.[0] || "" : "")});
  function fill(preferred = "", rows = null) {
    const p = sources.find(s => s.value === provider.value)?.provider;
    const models = rows || (p?.models || []).map(m => ({value: m, label: m.slice(m.indexOf("/") + 1)}));
    if (preferred && !models.some(m => m.value === preferred)) models.push({value: preferred, label: preferred.slice(preferred.indexOf("/") + 1)});
    model.setOptions([{value: "", label: words.auto}, ...models.map(m => ({...m, label: m.label + (m.known ? ` · ${words.known}` : "")}))]);
    model.value = preferred;
  }
  async function load(query = "") {
    const source = sources.find(s => s.value === provider.value);
    const p = source?.provider;
    if (!p) return;
    const current = ++generation;
    try {
      const url = p.id === "ollama" ? `/local-models?q=${encodeURIComponent(query)}${query ? "" : "&installed=1"}` : `/source-models?provider=${encodeURIComponent(p.id)}&fingerprint=${encodeURIComponent(source.value)}`;
      const response = await fetch(url);
      if (!response.ok || current !== generation) return;
      const data = await response.json();
      if (current !== generation) return;
      localChoices = p.id === "ollama" ? data.options || [] : [];
      const rows = p.id === "ollama" ? localChoices.map(m => ({...m, value: m.downloaded ? `ollama/${m.value}` : `hf:${m.value}`})) : data.models;
      fill(model.value, rows);
      note.textContent = data.note || (data.notes || []).join(" · ");
    } catch {}
  }
  function set(modelValue, keyValue) {
    ++generation; previousModel = modelValue || "";
    const pid = modelValue?.split("/", 1)[0];
    provider.value = pid === "ollama" ? "ollama" : keyValue || "";
    if (pid === "ollama" && !sources.some(s => s.value === "ollama")) { sources.push({value: "ollama", label: "Ollama", provider: {id: "ollama", models: [modelValue]}}); provider.setOptions(sources); }
    if (keyValue && !sources.some(s => s.value === keyValue)) {
      sources.push({value: keyValue, label: words.missing}); provider.setOptions(sources);
    }
    fill(modelValue || "");
  }
  provider.onchange = () => { fill(); note.textContent = ""; load(); };
  provider.onopen = async () => {
    if (sources.some(s => s.value === "ollama")) return;
    try {
      const response = await fetch("/local-models?installed=1");
      if (!response.ok) return;
      const data = await response.json();
      if (!(data.options || []).some(m => m.downloaded)) return;
      const p = {id: "ollama", models: data.options.filter(m => m.downloaded).map(m => `ollama/${m.value}`)};
      sources.push({value: "ollama", label: "Ollama", provider: p}); provider.setOptions(sources);
    } catch {}
  };
  model.onopen = () => load();
  model.onsearch = query => {
    if (provider.value !== "ollama") return;
    clearTimeout(searchTimer); searchTimer = setTimeout(() => load(query), 300);
  };
  model.onchange = event => {
    if (model.value.startsWith("hf:")) {
      event.stopImmediatePropagation();
      const name = model.value.slice(3); model.value = previousModel;
      if (lineOpen && !sending && !settingsPending) {
        const id = crypto.randomUUID();
        act({type: "settings", id, change: {section: "models", operation: "add", model: name, server: "ollama"}});
      }
    } else previousModel = model.value;
  };
  set(chosen, fingerprint);
  return {provider, model, key, set};
}

async function post(path, body) {
  return fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
}

// Windows reflect server prompts; only the server validates and applies answers.
async function answer(p, value, cancel = false) {
  if (sending) return;
  sending = true;
  try {
    const body = cancel ? {id: p.id, cancel: true} : {id: p.id, value};
    const response = await post("/answer", body);
    if (!response.ok) throw new Error();
    if (pendingId === p.id) {
      $("pending").replaceChildren();
      pendingId = null;
    }
    if (promptId === p.id) {
      lineOpen = false;
      updateActions();
      $("prompt").replaceChildren();
      promptId = null;
    }
  } catch {
    $("status").textContent = words.invalid;
  } finally {
    sending = false;
  }
}

function showPrompt(p, pending = false) {
  const target = pending ? "pending" : "prompt";
  let shownId = pending ? pendingId : promptId;
  if (!pending) lineOpen = !!p && p.kind === "line";
  if (!pending) document.body.classList.toggle("settings-prompt", !!p && p.kind !== "line");
  updateActions();
  if (!p) {
    $(target).replaceChildren();
    if (pending) pendingId = null; else promptId = null;
    return;
  }
  if (p.id === shownId) return;
  if (pending) pendingId = p.id; else promptId = p.id;
  const form = $(target);
  form.replaceChildren();
  form.classList.toggle("line-prompt", p.kind === "line");
  if (p.question) text("h2", p.question, form);
  let input;
  if (p.kind === "select" || p.kind === "checkbox") {
    let index = 0;
    const checked = [];
    for (const item of p.items) {
      if ("heading" in item) {
        text("h2", item.heading, form);
        continue;
      }
      const i = index++;
      if (p.kind === "select") {
        const b = button(form, item.label, () => answer(p, i));
        b.className = "choice";
        if (item.hint) text("small", item.hint, b);
      } else {
        const label = text("label", "", form);
        const tick = document.createElement("input");
        tick.type = "checkbox";
        tick.checked = (p.default || []).includes(i);
        label.append(tick);
        label.append(document.createTextNode(item.label));
        if (item.hint) text("small", item.hint, label);
        checked.push(tick);
      }
    }
    if (p.kind === "checkbox") {
      button(form, words.ok, () => answer(p, checked.flatMap((c, i) => c.checked ? [i] : [])));
    }
  } else if (p.kind === "confirm") {
    button(form, words.yes, () => answer(p, true));
    button(form, words.no, () => answer(p, false));
  } else {
    input = document.createElement(p.kind === "line" ? "textarea" : "input");
    if (p.kind === "secret") {
      input.type = "password";
      input.autocomplete = "off";
    } else {
      input.value = p.default || "";
    }
    input.maxLength = 20000;
    input.setAttribute("aria-label", p.question || "DESMAN");
    if (p.kind === "line") {
      input.rows = 1;
      input.placeholder = terminalOpen ? words.terminal_input : words.answer;
    }
    form.append(input);
    const submit = () => {
      const value = input.value;
      if (p.kind === "secret") input.value = "";
      answer(p, value);
    };
    let complete = () => false;
    input.addEventListener("keydown", e => {
      if (complete(e)) return;
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        submit();
      }
    });
    const send = button(form, p.kind === "line" ? "↑" : words.send, submit);
    send.title = words.send;
    send.setAttribute("aria-label", words.send);
    if (p.kind === "line") {
      const suggestions = text("div", "", form);
      suggestions.id = "suggestions";
      const resize = () => {
        input.rows = 1;
        const height = parseFloat(getComputedStyle(input).lineHeight);
        input.rows = Math.min(8, Math.max(1, Math.ceil((input.scrollHeight - 24) / height)));
      };
      input.addEventListener("input", resize);
      resize();
      let matches = [];
      let selected = -1;
      const close = () => { matches = []; selected = -1; suggestions.replaceChildren(); };
      const choose = name => { input.value = name; resize(); close(); input.focus(); };
      const draw = () => {
        suggestions.replaceChildren();
        matches.forEach(([name, hint], i) => {
          const b = button(suggestions, "", () => choose(name));
          text("span", name, b).className = "command-name";
          text("span", hint, b).className = "command-hint";
          b.classList.toggle("selected", i === selected);
          if (i === selected) b.scrollIntoView({block: "nearest"});
        });
      };
      input.addEventListener("input", () => {
        const value = input.value.toLowerCase();
        const bare = terminalOpen && value.length > 0 && !/^[!/]|\s/.test(value);
        const commands = Object.entries(p.commands || {}).sort((a, b) =>
          Number(a[0].includes(" ")) - Number(b[0].includes(" ")) || a[0].localeCompare(b[0]));
        matches = bare ?
          commands.filter(([name]) => name.startsWith("!"))
            .map(([name, hint]) => [name.slice(1), hint])
            .filter(([name]) => name.startsWith(value)) :
          (value.startsWith("!") || (value.startsWith("/") && !/\s/.test(value))) ?
            commands.filter(([name]) => name.startsWith(value)) : [];
        selected = -1;
        draw();
      });
      complete = e => {
        if (!matches.length || e.isComposing) return false;
        if (e.key === "Escape") { close(); e.preventDefault(); return true; }
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          selected = selected < 0 ? (e.key === "ArrowDown" ? 0 : matches.length - 1) :
            (selected + (e.key === "ArrowDown" ? 1 : -1) + matches.length) % matches.length;
          draw(); e.preventDefault(); return true;
        }
        if (e.key === "Tab") {
          selected = selected < 0 ? (e.shiftKey ? matches.length - 1 : 0) :
            (selected + (e.shiftKey ? -1 : 1) + matches.length) % matches.length;
          input.value = matches[selected][0]; resize(); draw();
          e.preventDefault(); return true;
        }
        return false;
      };
    }
    if (p.kind === "line" && !$("settings-window").hidden) {
      $("settings-dialog").focus({preventScroll: true});
    } else input.focus();
  }
  if (p.kind !== "line") button(form, words.cancel, () => {
    if (input && p.kind === "secret") input.value = "";
    answer(p, null, true);
  });
  if (p.kind !== "line" && !input) form.querySelector("button")?.focus({preventScroll: true});
  form.onsubmit = e => e.preventDefault();
}

function removeGreeting() {
  document.querySelectorAll(".greeting").forEach(n => n.remove());
}

function event(e) {
  if (e.type === "reset") $("messages").replaceChildren();
  if (e.type === "prompt_closed") {
    if (e.kind === "line" && e.echo !== null) removeGreeting();
    refreshPanels();
  }
  if (e.type === "out") {
    const node = text("div", "", $("messages"));
    node.className = e.context ? "out context-output" : e.greeting ? "out greeting" : "out";
    rich(node, e.html);
  }
  if (e.type === "prompt_closed" && e.echo !== null) {
    const value = Array.isArray(e.echo) ? e.echo.join(", ") :
      (typeof e.echo === "boolean" ? (e.echo ? words.yes : words.no) : String(e.echo));
    const node = text("div", value, $("messages"));
    node.className = "user";
    linkify(node);
  }
  if (e.type === "end") ended = true;
}

// Poll events with their cursor and transient state with its separate revision.
async function poll() {
  while (!ended) {
    try {
      const response = await fetch(`/events?after=${after}&state=${state}`);
      if (!response.ok) throw new Error();
      const data = await response.json();
      $("status").textContent = "";
      // Setup and language commands can change labels during the session.
      if (data.events.length && !data.ended) {
        try {
          const labels = await fetch("/strings.json");
          if (labels.ok) { await applyStrings(await labels.json()); }
        } catch {}
        $("stop").textContent = words.stop;
      }
      const transcript = $("messages");
      const nearBottom = transcript.scrollTop + transcript.clientHeight >= transcript.scrollHeight - 120;
      for (const e of data.events) {
        event(e);
        after = Math.max(after, e.seq);
      }
      state = data.state;
      if (data.transcript && data.transcript_version !== transcriptVersion) {
        transcriptVersion = data.transcript_version;
        transcript.replaceChildren();
        for (const html of data.transcript) rich(text("div", "", transcript), html);
      }
      let live = data.session_live || data.live || "";
      if (data.turn?.state === "queued") {
        $("live").textContent = words.waiting_module.replace("{n}", data.turn.ahead);
      } else rich($("live"), live);
      showPrompt(data.pending || null, true);
      showPrompt(data.prompt);
      $("stop").hidden = !data.busy;
      ended ||= data.ended;
      if (data.events.length && nearBottom && !document.querySelector(".greeting")) transcript.scrollTop = transcript.scrollHeight;
    } catch {
      $("status").textContent = words.offline;
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  }
  $("status").textContent = words.ended;
  $("prompt").replaceChildren();
  $("stop").hidden = true;
}

(async () => {
  try {
    const response = await fetch("/strings.json");
    if (!response.ok) throw new Error();
    words = await response.json();
    // Stop asks the main chat thread to use its ordinary cancellation path.
    $("stop").textContent = words.stop;
    $("stop").onclick = () => post("/stop", {});
    setupPanels();
    refreshPanels();
    setInterval(refreshPanels, 4000);
    document.addEventListener("visibilitychange", refreshPanels);
    await poll();
  } catch {
    $("status").textContent = "DESMAN";
  }
})();


function updateActions() {
  if (lineOpen && !sending && !settingsPending) {
    for (const save of settingsDeferred) queueMicrotask(save);
    settingsDeferred.clear();
  }
  $("settings-wait").textContent = lineOpen ? "" : words.wait;
  document.querySelectorAll("[data-action]").forEach(b => {
    b.disabled = !lineOpen || (!!b.closest("#settings-window") && !!settingsPending);
    b.title = lineOpen ? (b.dataset.label || "") : words.wait;
  });
}

function actionButton(parent, label, value) {
  const b = button(parent, label, () => act(typeof value === "function" ? value() : value));
  b.dataset.action = "true";
  b.dataset.label = label;
  return b;
}

async function act(action) {
  if (!action || !lineOpen || sending) return;
  sending = true;
  try {
    const response = await post("/action", action);
    if (!response.ok) throw new Error();
    lineOpen = false;
    updateActions();
  } catch {
    if (action.type === "settings") {
      settingsPending = null;
      settingsReceipt?.settled?.(false);
      $("settings-result").textContent = words.invalid;
    }
    $("status").textContent = words.invalid;
  } finally { sending = false; }
}

async function applyStrings(next) {
  const changed = words.lang !== next.lang;
  words = next;
  setupPanels();
  if (!changed) return;
  showModules();
  for (const label of document.querySelectorAll("[data-module-role]")) {
    label.textContent = words[`role_${label.dataset.moduleRole}`] || label.textContent;
  }
  if (!$("settings-window").hidden) await refreshSettings(true, true);
}

function setupPanels() {
  document.documentElement.lang = words.lang;
  $("panel-switch").replaceChildren();
  for (const mode of ["tasks", "modules"]) {
    const control = button($("panel-switch"), words[mode], () => {
      panelMode = mode;
      $("tasks").hidden = mode !== "tasks";
      $("modules").hidden = mode !== "modules";
      setupPanels();
    });
    control.setAttribute("aria-pressed", String(panelMode === mode));
  }
  $("panel-footer").replaceChildren();
  actionButton($("panel-footer"), panelMode === "tasks" ? words.delete_tasks : words.stop_modules,
    {type: panelMode === "tasks" ? "hide_finished" : "pause_tasks"});
  $("sidebar").setAttribute("aria-label", words.sessions);
  $("tasks-panel").setAttribute("aria-label", words.tasks);
  $("session-tools").replaceChildren();
  actionButton($("session-tools"), words.new_chat, {type: "new_session", temporary: false});
  actionButton($("session-tools"), words.temporary_chat, {type: "new_session", temporary: true});
  const terminal = actionButton($("session-tools"), words.terminal, {type: "terminal"});
  terminal.id = "terminal-button";
  $("sidebar-bottom").replaceChildren();
  actionButton($("sidebar-bottom"), words.delete_chats, {type: "delete_all"});
  actionButton($("sidebar-bottom"), `⚑ ${words.report_problem}`, {type: "report_problem"});
  actionButton($("sidebar-bottom"), `✉ ${words.messenger}`, {type: "messenger"});
  button($("sidebar-bottom"), `⚙ ${words.settings}`, openSettings);
  $("topbar").replaceChildren();
  for (const [label, id] of [[words.sessions, "sidebar"], [words.tasks, "tasks-panel"]]) {
    const b = button($("topbar"), label, () => {
      const open = $(id).classList.toggle("open");
      b.setAttribute("aria-expanded", String(open));
    });
    b.setAttribute("aria-expanded", String($(id).classList.contains("open")));
    b.setAttribute("aria-controls", id);
  }
  updateActions();
}

function relativeDate(value) {
  const date = new Date(value);
  const now = new Date();
  const day = d => Date.UTC(d.getFullYear(), d.getMonth(), d.getDate());
  const days = (day(date) - day(now)) / 86400000;
  if (days === 0) return new Intl.DateTimeFormat(words.lang, {hour: "2-digit", minute: "2-digit", hourCycle: "h23"}).format(date);
  if (days === -1) return new Intl.RelativeTimeFormat(words.lang, {numeric: "auto"}).format(-1, "day");
  return new Intl.DateTimeFormat(words.lang, {dateStyle: "short"}).format(date);
}

function iconAction(parent, icon, label, value) {
  const b = actionButton(parent, label, value);
  b.textContent = icon;
  b.className = "icon-action";
  b.setAttribute("aria-label", label);
  return b;
}

function showSessions(data) {
  terminalOpen = !!data.terminal && !!data.terminal.id && data.current === data.terminal.id;
  $("session-title").textContent = terminalOpen ? words.terminal : "";
  const terminal = $("terminal-button");
  if (terminal) {
    terminal.textContent = words.terminal;
    if (data.terminal && data.terminal.unread) {
      text("span", String(data.terminal.unread), terminal).className = "unread";
    }
    terminal.classList.toggle("current", terminalOpen);
    if (terminalOpen) terminal.setAttribute("aria-current", "true");
    else terminal.removeAttribute("aria-current");
  }
  const input = $("prompt").querySelector("textarea");
  if (input && lineOpen) input.placeholder = terminalOpen ? words.terminal_input : words.answer;
  const root = $("sessions");
  root.replaceChildren();
  for (const pinned of [true, false]) {
    const sessions = data.sessions.filter(s => s.pinned === pinned);
    if (pinned && !sessions.length) continue;
    text("h2", pinned ? words.pinned : words.sessions, root);
    for (const s of sessions) {
      const row = text("div", "", root);
      row.className = s.id === data.current ? "session current" : "session";
      const open = actionButton(row, (s.unread ? "● " : "") + (s.name || "…"), {type: "open_session", id: s.id});
      open.className = "session-name";
      if (s.state) {
        const label = s.state === "pending" ? words.waits_for_you :
          s.state === "queued" ? words.queued_turns.replace("{n}", s.queued) : words.answering;
        text("small", label + (s.state === "answering" && s.queued ? " · " + words.queued_turns.replace("{n}", s.queued) : ""), row);
      }
      if (s.id === data.current) open.setAttribute("aria-current", "true");
      text("time", relativeDate(s.last_at), row).dateTime = new Date(s.last_at).toISOString();
      iconAction(row, "⌖", s.pinned ? words.unpin : words.pin, {type: s.pinned ? "unpin" : "pin", id: s.id});
      iconAction(row, "×", words.delete, {type: "delete_session", id: s.id});
    }
  }

  updateActions();
}

function showTasks(data) {
  // Preserve unsent answers and keyboard focus when a poll updates task state.
  const drafts = new Map([...$("tasks").querySelectorAll("textarea")].map(n => [n.dataset.ref, n.value]));
  const active = document.activeElement;
  const focused = active && active.dataset.ref;
  const selection = focused ? [active.selectionStart, active.selectionEnd] : null;
  const root = $("tasks");
  // moduleCard reattaches open editors from moduleEditors; redraw the live task state.
  root.replaceChildren();
  let section;
  for (const task of [...data.tasks].sort((a, b) => Number(b.pinned) - Number(a.pinned))) {
    const heading = task.pinned ? words.pinned : words.tasks;
    if (heading !== section) { text("h2", heading, root); section = heading; }
    const card = text("article", "", root);
    card.className = `task ${task.needs_human ? "waiting" : task.status}`;
    const titleRow = text("div", "", card);
    titleRow.className = "task-title-row";
    const toggle = button(titleRow, "", () => {
      if (expanded.has(task.id)) expanded.delete(task.id); else expanded.add(task.id);
      showTasks(data);
    });
    toggle.className = "task-toggle";
    text("span", "●", toggle).className = "status-dot";
    text("span", task.goal, toggle).className = "task-goal";
    text("span", `${task.done}/${task.total}`, toggle);
    text("span", expanded.has(task.id) ? "⌃" : "⌄", toggle);
    toggle.title = task.goal + " · " + words[`status_${task.status}`];
    toggle.setAttribute("aria-expanded", String(expanded.has(task.id)));
    const pin = iconAction(titleRow, "⌖", task.pinned ? words.unpin : words.pin, {type: "task_control", operation: task.pinned ? "unpin" : "pin", id: task.id});
    pin.title = task.pinned ? words.unpin : words.pin;
    pin.setAttribute("aria-pressed", String(!!task.pinned));
    if (task.queued) text("span", words.worker_queue, card).className = "status-chip";
    if (task.paused_at) {
      text("span", words.paused, card).className = "status-chip";
      actionButton(card, words.continue_task, {type: "task_control", operation: "resume", id: task.id});
    }
    if (!expanded.has(task.id)) continue;
    for (const sub of task.subtasks) {
      const row = text("div", "", card);
      row.className = "subtask";
      text("span", words[`status_${sub.status}`], row).className = "status-chip";
      const heading = text("span", `${sub.seq} · ${sub.title}`, row);
      heading.title = sub.description;
      text("p", sub.description, card);
      if (sub.version === null) continue;
      text("p", sub.reason, card);
      const input = document.createElement("textarea");
      input.rows = 3;
      input.maxLength = 20000;
      input.placeholder = words.answer;
      input.setAttribute("aria-label", words.answer);
      input.dataset.ref = sub.ref;
      input.value = drafts.get(sub.ref) || "";
      card.append(input);
      const decisions = text("div", "", card);
      decisions.className = "decisions";
      for (const decision of ["retry", "accept", "fail"]) {
        const b = actionButton(decisions, words[decision], () => {
          if (decision === "fail" && !input.value.trim()) {
            input.setCustomValidity(words.answer); input.reportValidity(); return null;
          }
          input.setCustomValidity("");
          return {type: "decide", ref: sub.ref, version: sub.version, decision, note: input.value};
        });
        if (decision === "fail") b.className = "danger";
      }
    }
    const modules = task.cards || [];
    if (modules.length) text("h3", words.modules, card);
    for (const m of modules) moduleCard(card, m, "task");

  }
  updateActions();
  if (openPicker && root.contains(openPicker)) openPicker.position();
  if (active && root.contains(active) && !focused) active.focus({preventScroll: true});
  if (focused) {
    const input = [...root.querySelectorAll("textarea")].find(n => n.dataset.ref === focused);
    if (input) { input.focus({preventScroll: true}); input.setSelectionRange(...selection); }
  }
}

let panelsLoading = false;
async function refreshPanels() {
  if (document.visibilityState !== "visible" || panelsLoading || ended) return;
  panelsLoading = true;
  try {
    const responses = await Promise.all([fetch("/sessions"), fetch("/tasks"), fetch("/modules")]);
    if (responses.every(r => r.ok)) {
      moduleSnapshot = await responses[2].json();
      showSessions(await responses[0].json());
      showModules();
      showTasks(await responses[1].json());
    }
  } catch {} finally { panelsLoading = false; }
}


let panelMode = "tasks";
let moduleSnapshot = {modules: [], roles: [], interlocutor: null};
// Keep open editors intact across snapshot polls, including their unsent selections.
const moduleEditors = new Map();

function moduleCard(parent, m, placement = "modules") {
  const identity = `${placement}:${m.scope}:${m.scope_id}:${m.role}`;
  const card = text("article", "", parent);
  card.className = "module-card";
  const portrait = document.createElement("img");
  portrait.src = `/img/${m.icon || "module"}.png`;
  portrait.alt = "";
  card.append(portrait);
  const info = text("div", "", card);
  info.className = "module-info";
  text("strong", words[`role_${m.role}`] || m.label, info).dataset.moduleRole = m.role;
  text("small", m.model || words.auto, info);
  text("small", `${words.key}: ${m.key_label && m.key_label !== "auto" ? m.key_label : words.auto}`, info);
  const usage = text("div", "", info);
  usage.className = "module-usage";
  if (m.pinned) text("span", "⌖", usage).title = words.pinned;
  if (m.last_model && m.last_model !== m.model) text("small", `${words.answered_by} ${m.last_model}`, info);
  text("span", words[`source_${m.source}`], usage).className = "status-chip";
  text("small", words.model_usage.replace("{calls}", m.model_calls).replace("{tokens}", m.model_tokens), usage);
  usage.title = (m.scope === "session" ? words.role_usage_chat : words.role_usage_task)
    .replace("{calls}", m.calls).replace("{tokens}", m.tokens);
  if (m.state) text("small", m.state, info).className = `module-state ${m.state_style || "muted"}`;
  const change = button(card, "✎", () => {
    moduleEditors.set(identity, editor);
    editor.hidden = false;
    change.hidden = true;
  });
  change.className = "module-change";
  change.dataset.action = "true";
  change.dataset.label = words.change;
  change.setAttribute("aria-label", words.change);
  const editor = moduleEditors.get(identity) || text("div", "", card);
  if (moduleEditors.has(identity)) {
    card.append(editor);
    change.hidden = true;
    return card;
  }
  editor.className = "module-editor";
  editor.hidden = true;
  const role = moduleSnapshot.roles.find(r => r.role === m.role);
  if (!role) return card;
  const allowedHere = (moduleSnapshot.roles_by_scope?.[m.scope] || []).includes(m.role);
  const scopeLabel = text("label", "", editor);
  const everywhere = document.createElement("input"); everywhere.type = "checkbox";
  everywhere.checked = !allowedHere; everywhere.disabled = !allowedHere;
  if (!allowedHere) everywhere.title = words.scope_unavailable;
  scopeLabel.append(everywhere, document.createTextNode(m.scope === "session" ? words.every_chat : words.every_task));
  const scope = {get value() { return everywhere.checked ? "role" : m.scope; }};
  const choices = rolePickers(editor, role, null, null);
  const {model, key} = choices;
  function defaults() {
    const global = scope.value === "role";
    choices.set(global ? role.model : m.choice_model, global ? role.key_fingerprint : m.choice_key);
  }
  everywhere.onchange = defaults;
  defaults();
  actionButton(editor, words.apply, () => {
    moduleEditors.delete(identity);
    editor.hidden = true;
    change.hidden = false;
    return {type: "module", role: m.role, scope: scope.value,
      scope_id: scope.value === "role" ? null : m.scope_id,
      model: model.requestValue || null, key: key.value || null};
  });
  button(editor, words.cancel, () => {
    moduleEditors.delete(identity);
    editor.hidden = true;
    change.hidden = false;
  });
  return card;
}

function showModules() {
  const root = $("modules");
  if (!preserveEditor(root)) {
  root.replaceChildren();
  const groups = new Map();
  for (const m of moduleSnapshot.modules) {
    if (!groups.has(m.task_id)) {
      const group = text("section", "", root);
      text("h3", `${m.task_ref} · ${m.goal}`, group);
      groups.set(m.task_id, group);
    }
    moduleCard(groups.get(m.task_id), m);
  }
  }
  $("chat-module").hidden = terminalOpen || !moduleSnapshot.interlocutor;
  $("chat-module-title").textContent = words.role_interlocutor;
  $("chat-module-title").onclick = e => {
    e.preventDefault();
    const collapsed = $("chat-module").classList.toggle("collapsed");
    $("chat-module-title").setAttribute("aria-expanded", String(!collapsed));
  };
  if (!preserveEditor($("chat-module-card"))) {
  $("chat-module-card").replaceChildren();
  if (!terminalOpen && moduleSnapshot.interlocutor) {
    moduleCard($("chat-module-card"), moduleSnapshot.interlocutor, "chat");
  }
  }
  updateActions();
}

// Settings are local UI state; authority stays with the chat's open line.
let settingsData = null;
let settingsRedraw = false;
let settingsSection = "general";
let settingsPending = null;
let settingsTimer = null;
let settingsFocus = null;
let settingsReceipt = null;
let settingsReceiptTimer = null;
const settingsDeferred = new Set();
const settingsSections = ["general", "keys", "roles", "permissions", "personalize", "behaviour", "limits", "models", "network", "protected"];

function closeSettings() {
  $("settings-window").hidden = true;
  document.body.classList.remove("settings-open");
  clearInterval(settingsTimer);
  settingsFocus?.focus({preventScroll: true});
}

async function refreshSettings(redraw = false, languageChanged = false) {
  try {
    const response = await fetch("/settings");
    if (!response.ok) throw new Error();
    settingsData = await response.json();
    const result = settingsData.result;
    if (settingsPending && result?.id === settingsPending) {
      settingsPending = null;
      settingsReceipt?.settled?.(result.ok);
      clearTimeout(settingsReceiptTimer);
      if (settingsReceipt?.isConnected) {
        settingsReceipt.textContent = result.ok ? `✓ ${words.saved}` : result.message;
        settingsReceipt.className = `setting-receipt ${result.ok ? "good" : "bad"}`;
        const receipt = settingsReceipt;
        settingsReceiptTimer = setTimeout(() => { receipt.textContent = ""; }, 2000);
      }
      $("settings-result").textContent = settingsReceipt?.isConnected ? "" : result.message;
      $("settings-result").className = result.ok ? "good" : "bad";
      updateActions();
      redraw ||= !settingsReceipt?.isConnected;
      if (result.ok) {
        $("settings-form").querySelector("[data-local-models]")?.reload?.();
        const labels = await fetch("/strings.json");
        if (labels.ok) { await applyStrings(await labels.json()); }
      }
    }
    settingsRedraw ||= redraw;
    if (settingsRedraw && (languageChanged || !preserveEditor($("settings-form")))) {
      settingsRedraw = false;
      const focused = $("settings-window").contains(document.activeElement);
      drawSettings();
      if (languageChanged && focused) $("settings-nav").querySelector('[aria-current="true"]')?.focus({preventScroll: true});
    }
  } catch { $("settings-result").textContent = words.invalid; }
}

function openSettings() {
  settingsFocus = document.activeElement;
  $("settings-window").hidden = false;
  document.body.classList.add("settings-open");
  if (document.body.classList.contains("settings-prompt")) {
    $("prompt").querySelector("input, textarea, button")?.focus({preventScroll: true});
  } else $("settings-close").focus();
  refreshSettings(true);
  clearInterval(settingsTimer);
  settingsTimer = setInterval(() => {
    if (!document.hidden) refreshSettings();
  }, 1500);
}

function settingSelect(parent, label, options, current) {
  return picker(parent, label, options, current ?? "auto");
}

function settingInput(parent, label, value, type = "text") {
  const row = text("label", "", parent);
  row.className = "setting-row";
  text("span", label, row);
  const input = document.createElement(type === "textarea" ? "textarea" : "input");
  if (type !== "textarea") input.type = type;
  if (type === "checkbox") { input.checked = value; input.setAttribute("role", "switch"); }
  else input.value = value ?? "";
  row.append(input);
  return input;
}

function settingAuto(parent, change, controls = null) {
  controls ||= [...parent.querySelectorAll("[data-picker], input:not(.picker input), select, textarea")].filter(c => !c.dataset.autosave);
  const receipt = text("small", "", controls.at(-1)?.parentElement || parent);
  receipt.className = "setting-receipt";
  let previous = JSON.stringify(change());
  const save = () => {
    if (!receipt.isConnected) return;
    const value = change();
    const next = JSON.stringify(value);
    if (next === previous || controls.some(c => !c.checkValidity())) return;
    if (!lineOpen || sending || settingsPending) { settingsDeferred.add(save); return; }
    const id = crypto.randomUUID();
    settingsPending = id;
    settingsReceipt = receipt;
    const before = previous;
    receipt.settled = ok => { if (!ok) previous = before; };
    previous = next;
    act({type: "settings", id, change: value});
  };
  for (const control of controls) {
    control.dataset.autosave = "true";
    if (control.dataset.picker || control.tagName === "SELECT" || control.type === "checkbox") control.addEventListener("change", save);
    else {
      control.addEventListener("blur", save);
      control.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); save(); } });
    }
  }
}

function settingSave(parent, change, label = words.save) {
  const b = actionButton(parent, label, () => {
    if (!lineOpen || sending || settingsPending) return null;
    const id = crypto.randomUUID();
    settingsPending = id;
    settingsReceipt = null;
    $("settings-result").textContent = "";
    return {type: "settings", id, change: typeof change === "function" ? change() : change};
  });
  if ([words.add_key, words.join, words.lend, words.download, words.approve].includes(label)) b.classList.add("primary");
  return b;
}

function drawSettings() {
  if (!settingsData) return;
  const d = settingsData;
  if (settingsSection === "protected" && !d.protected.blocked) settingsSection = "general";
  const root = $("settings-form");
  root.replaceChildren();
  $("settings-nav").replaceChildren();
  $("settings-heading").textContent = words.settings;
  $("settings-close").setAttribute("aria-label", words.close);
  for (const section of settingsSections) {
    if (section === "protected" && !d.protected.blocked) continue;
    const b = button($("settings-nav"), words[`settings_${section}`], () => {
      settingsSection = section; drawSettings();
    });
    b.setAttribute("aria-current", String(section === settingsSection));
  }
  text("h2", words[`settings_${settingsSection}`], root);
  if (settingsSection === "general") {
    const language = settingSelect(root, words.language, d.general.languages, d.general.language);
    settingAuto(root, () => ({section: "general", language: language.value}));
    settingSave(root, {section: "wizard"}, words.setup_wizard);
  } else if (settingsSection === "keys") {
    const others = text("details", "", root);
    text("summary", words.other_providers, others);
    for (const p of [...d.providers].sort((a, b) => Number(b.ready) - Number(a.ready))) {
      const card = text("section", "", p.ready ? root : others);
      card.className = "provider-block";
      const heading = text("div", "", card); heading.className = "provider-heading";
      text("strong", p.name, heading);
      text("small", p.ready ? words.status_ready : words.missing, heading).className = "setting-badge";
      if (/^https?:\/\//.test(p.url)) {
        const a = text("a", words.get_key, heading);
        a.href = p.url; a.target = "_blank"; a.rel = "noopener noreferrer";
      }
      for (const key of p.keys) {
        const row = text("div", "", card); row.className = "key-row";
        text("span", `${key.label} ${key.mask}`, row);
        settingSave(row, {section: "keys", operation: "check", provider: p.id}, words.check);
        settingSave(row, {section: "keys", operation: "remove", provider: p.id, key: key.fingerprint}, words.delete);
      }
      if (p.keyed) button(heading, `+ ${words.add_key}`, () => {
        if (card.querySelector(".key-entry")) return;
        const entry = text("div", "", card); entry.className = "key-entry";
        const label = settingInput(entry, words.key_label, ""); label.maxLength = 60;
        settingSave(entry, () => ({section: "keys", operation: "add", provider: p.id, label: label.value}), words.add_key);
        label.focus();
      });
      for (const field of p.settings) {
        const value = settingInput(card, field.label, field.value); value.maxLength = 1000;
        settingAuto(card, () => ({section: "provider", provider: p.id, setting: field.name, value: value.value || "-"}), [value]);
      }
    }
    root.append(others);
  } else if (settingsSection === "roles") {
    const head = text("div", "", root); head.className = "role-row role-head";
    for (const label of [words.settings_roles, words.source, words.model]) text("small", label, head);
    for (const role of d.roles) {
      const card = text("section", "", root); card.className = "role-row"; text("strong", role.label, card);
      const {provider, model, key} = rolePickers(card, role, role.model, role.key_fingerprint);
      settingAuto(card, () => ({section: "roles", role: role.role, model: model.requestValue || null, key: key.value || null}), [provider, model]);
    }
  } else if (settingsSection === "permissions") {
    for (const p of d.permissions) {
      const on = settingInput(root, p.label, p.on, "checkbox");
      settingAuto(root, () => ({section: "permissions", name: p.name, on: on.checked}));
    }
  } else if (settingsSection === "personalize") {
    text("p", d.personalize.consent, root);
    const on = settingInput(root, words.settings_personalize, d.personalize.on, "checkbox");
    settingAuto(root, () => ({section: "personalize", on: on.checked}));
    for (const note of d.personalize.notes) {
      const row = text("div", note.text, root);
      settingSave(row, {section: "forget", note: note.id}, words.forget);
    }
  } else if (settingsSection === "behaviour") {
    const values = {};
    for (const part of ["length", "tone", "style"]) values[part] = settingSelect(root, d.behaviour.labels[part], d.behaviour.choices[part], d.behaviour.values[part]);
    const custom = settingInput(root, d.behaviour.labels.custom, d.behaviour.values.custom, "textarea"); custom.maxLength = 500;
    settingAuto(root, () => ({section: "behaviour", ...Object.fromEntries(Object.entries(values).map(([k, v]) => [k, v.value])), custom: custom.value}));
    settingSave(root, {section: "behaviour", reset: true}, words.reset);
  } else if (settingsSection === "limits") {
    for (const limit of d.limits) {
      const row = text("section", "", root);
      const input = settingInput(row, limit.label, limit.value, "number");
      input.placeholder = String(limit.default);
      text("small", `${limit.min}–${limit.max} · ${words.auto}: ${limit.default}`, input.parentElement.firstChild).className = "setting-description";
      input.min = limit.min; input.max = limit.max; input.step = 1;
      settingAuto(row, () => ({section: "limits", name: limit.name, value: input.value === "" ? "auto" : Number(input.value)}));
      const reset = settingSave(input.parentElement, {section: "limits", name: limit.name, value: "auto"}, "↺");
      reset.title = words.auto; reset.setAttribute("aria-label", words.auto);
    }
  } else if (settingsSection === "models") {
    const p = d.providers.find(p => p.id === "ollama");
    const state = text("p", `Ollama · ${p?.settings.find(s => s.name === "OLLAMA_HOST")?.value || ""}`, root);
    const notes = text("small", "", root);
    const downloaded = text("div", "", root);
    const model = picker(root, words.model); model.dataset.localModels = "true";
    model.reload = () => load("");
    let choices = [], timer, request = 0;
    async function load(q) {
      const id = ++request;
      try {
        const response = await fetch(`/local-models?q=${encodeURIComponent(q)}`);
        if (!response.ok) throw new Error();
        const data = await response.json();
        if (id !== request || !model.isConnected) return;
        choices = data.options;
        downloaded.replaceChildren();
        for (const option of choices.filter(o => o.downloaded)) {
          const row = text("div", "", downloaded); row.className = "setting-row";
          text("span", `${option.label} · ${option.hint}`, row);
          settingSave(row, {section: "models", operation: "remove", model: option.value}, words.delete);
        }
        model.setOptions(choices.map(o => ({...o, hint: `${o.selected ? "✓ · " : ""}${o.hint}`})));
        model.value ||= data.selected;
        state.textContent = `Ollama · ${data.host} · ${data.reachable ? words.status_ready : words.off}`;
        notes.textContent = data.notes.join(" · ");
      } catch { if (id === request) notes.textContent = words.local_unreachable; }
    }
    model.onsearch = q => { clearTimeout(timer); timer = setTimeout(() => load(q), 300); };
    model.onchange = () => {
      const option = choices.find(o => o.value === model.value);
      if (!option || !lineOpen || sending || settingsPending) return;
      const id = crypto.randomUUID(); settingsPending = id; settingsReceipt = null;
      act({type: "settings", id, change: {section: "models", operation: option.downloaded ? "use" : "add", model: option.value, server: "ollama"}});
    };
    load("");
  } else if (settingsSection === "network") {
    const status = text("div", "desman-global", root); status.className = "provider-heading";
    text("small", d.network.enabled ? words.status_active : words.off, status).className = "setting-badge";
    const advanced = text("details", "", root); text("summary", words.advanced, advanced);
    const registry = settingInput(advanced, words.registry, d.network.registry || ""); registry.maxLength = 200;
    settingSave(status, () => ({section: "network", operation: d.network.enabled ? "leave" : "join", ...(d.network.enabled ? {} : {registry: registry.value || null})}), d.network.enabled ? words.leave : words.join);
    for (const p of d.providers.filter(p => p.lendable)) {
      const card = text("section", "", root); card.className = "lending-row"; text("strong", p.name, card);
      const quota = d.network.donated[p.id];
      const calls = settingInput(card, words.calls_daily, quota?.calls_per_day ?? d.network.default_quota.calls_per_day, "number"); calls.min = 1; calls.max = 100000;
      const tokens = settingInput(card, words.tokens_daily, quota?.tokens_per_day ?? d.network.default_quota.tokens_per_day, "number"); tokens.min = 1000; tokens.max = 100000000;
      const change = () => ({section: "network", operation: quota ? "quota" : "donate", provider: p.id, calls: Number(calls.value), tokens: Number(tokens.value)});
      if (quota) settingAuto(card, change);
      else settingSave(card, change, words.lend);
      if (quota) settingSave(card, {section: "network", operation: "undonate", provider: p.id}, words.unlend);
    }
  } else if (settingsSection === "protected") {
    for (const line of d.protected.lines) text("p", line, root);
    for (const op of ["show", "approve"]) settingSave(root, {section: "protected", operation: op}, words[op]);
  }
  updateActions();
}

$("settings-close").addEventListener("click", closeSettings);
document.addEventListener("keydown", e => {
  if ($("settings-window").hidden || document.body.classList.contains("settings-prompt")) return;
  if (e.key === "Escape") { e.preventDefault(); closeSettings(); }
  if (e.key === "Tab") {
    const controls = [...$("settings-dialog").querySelectorAll("button, input, select, textarea, a[href]")].filter(n => !n.disabled && n.getClientRects().length);
    const first = controls[0], last = controls[controls.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
});
