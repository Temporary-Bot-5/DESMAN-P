"use strict";

const $ = id => document.getElementById(id);
let words = {};
let after = 0;
let state = 0;
let promptId = null;
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

function showPrompt(p) {
  lineOpen = !!p && p.kind === "line";
  updateActions();
  if (!p) {
    $("prompt").replaceChildren();
    promptId = null;
    return;
  }
  if (p.id === promptId) return;
  promptId = p.id;
  const form = $("prompt");
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
        const commands = Object.entries(p.commands || {});
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
        if ((e.key === "Tab" && (selected >= 0 || matches.length === 1)) || (e.key === "Enter" && selected >= 0)) {
          choose(matches[selected < 0 ? 0 : selected][0]); e.preventDefault(); return true;
        }
        return false;
      };
    }
    input.focus();
  }
  if (p.kind !== "line") button(form, words.cancel, () => {
    if (input && p.kind === "secret") input.value = "";
    answer(p, null, true);
  });
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
    node.className = e.greeting ? "out greeting" : "out";
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
          if (labels.ok) { words = await labels.json(); setupPanels(); }
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
      rich($("live"), data.live || "");
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
  document.querySelectorAll("[data-action]").forEach(b => {
    b.disabled = !lineOpen;
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
    $("status").textContent = words.invalid;
  } finally { sending = false; }
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
  $("sidebar").setAttribute("aria-label", words.sessions);
  $("tasks-panel").setAttribute("aria-label", words.tasks);
  $("session-tools").replaceChildren();
  actionButton($("session-tools"), words.new_chat, {type: "new_session", temporary: false});
  actionButton($("session-tools"), words.temporary_chat, {type: "new_session", temporary: true});
  const terminal = actionButton($("session-tools"), "General_Terminal", {type: "terminal"});
  terminal.id = "terminal-button";
  $("sidebar-bottom").replaceChildren();
  actionButton($("sidebar-bottom"), `✉ ${words.messenger}`, {type: "messenger"});
  actionButton($("sidebar-bottom"), `⚙ ${words.settings}`, {type: "settings"});
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
  $("session-title").textContent = terminalOpen ? "General_Terminal" : "";
  const terminal = $("terminal-button");
  if (terminal) {
    terminal.textContent = "General_Terminal";
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
      const open = actionButton(row, s.name || "…", {type: "open_session", id: s.id});
      open.className = "session-name";
      if (s.id === data.current) open.setAttribute("aria-current", "true");
      text("time", relativeDate(s.last_at), row).dateTime = new Date(s.last_at).toISOString();
      iconAction(row, "⌖", s.pinned ? words.unpin : words.pin, {type: s.pinned ? "unpin" : "pin", id: s.id});
      iconAction(row, "×", words.delete, {type: "delete_session", id: s.id});
    }
  }
  actionButton(root, words.delete_chats, {type: "delete_all"}).className = "delete-chats";
  updateActions();
}

function showTasks(data) {
  // Preserve unsent answers and keyboard focus when a poll updates task state.
  const drafts = new Map([...$("tasks").querySelectorAll("textarea")].map(n => [n.dataset.ref, n.value]));
  const active = document.activeElement;
  const focused = active && active.dataset.ref;
  const selection = focused ? [active.selectionStart, active.selectionEnd] : null;
  const root = $("tasks");
  root.replaceChildren();
  for (const task of data.tasks) {
    const card = text("article", "", root);
    card.className = `task ${task.needs_human ? "waiting" : task.status}`;
    const toggle = button(card, "", () => {
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
    if (!expanded.has(task.id)) continue;
    for (const sub of task.subtasks) {
      const row = text("div", "", card);
      row.className = "subtask";
      text("span", words[`status_${sub.status}`], row).className = "status-chip";
      text("span", `${sub.seq} · ${sub.description}`, row);
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
  text("strong", words[`role_${m.role}`] || m.label, info);
  text("small", m.model || words.auto, info);
  text("small", `${words.key}: ${m.key_label && m.key_label !== "auto" ? m.key_label : words.auto}`, info);
  const usage = text("div", "", info);
  usage.className = "module-usage";
  text("span", words[`source_${m.source}`], usage).className = "status-chip";
  text("small", `${words.calls}: ${m.calls} · ${words.tokens}: ${m.tokens}`, usage);
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
  function select(label) {
    const wrapper = text("label", label, editor);
    return wrapper.appendChild(document.createElement("select"));
  }
  function option(select, value, label) {
    const o = document.createElement("option");
    o.value = value || "";
    o.textContent = label;
    select.append(o);
    return o;
  }
  const model = select(words.model);
  const key = select(words.key);
  const scope = select(words.scope);
  option(scope, m.scope, words.only_here).disabled = !m.scope_id;
  option(scope, "role", words.everywhere);
  scope.value = m.scope_id ? m.scope : "role";
  function keys(preferred) {
    key.replaceChildren();
    option(key, "", words.auto);
    const provider = (model.value || role.candidate_models[0] || "").split("/", 1)[0];
    for (const k of role.candidate_keys.filter(k => k.provider === provider)) {
      option(key, k.fingerprint, k.label);
    }
    key.value = [...key.options].some(o => o.value === preferred) ? preferred : "";
  }
  function defaults() {
    const global = scope.value === "role";
    const chosen = global ? role.model : m.choice_model;
    model.replaceChildren();
    option(model, "", words.auto);
    for (const candidate of new Set([...role.candidate_models, ...(chosen ? [chosen] : [])])) {
      option(model, candidate, candidate);
    }
    model.value = chosen || "";
    keys((global ? role.key_fingerprint : m.choice_key) || "");
  }
  scope.onchange = defaults;
  model.onchange = () => keys("");
  defaults();
  actionButton(editor, words.apply, () => {
    moduleEditors.delete(identity);
    return {type: "module", role: m.role, scope: scope.value,
      scope_id: scope.value === "role" ? null : m.scope_id,
      model: model.value || null, key: key.value || null};
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
  $("chat-module").hidden = terminalOpen || !moduleSnapshot.interlocutor;
  $("chat-module-title").textContent = words.role_interlocutor;
  $("chat-module-title").onclick = e => {
    e.preventDefault();
    const collapsed = $("chat-module").classList.toggle("collapsed");
    $("chat-module-title").setAttribute("aria-expanded", String(!collapsed));
  };
  $("chat-module-card").replaceChildren();
  if (!terminalOpen && moduleSnapshot.interlocutor) {
    moduleCard($("chat-module-card"), moduleSnapshot.interlocutor, "chat");
  }
  updateActions();
}
