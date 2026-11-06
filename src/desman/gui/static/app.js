"use strict";

const $ = id => document.getElementById(id);
let words = {};
let after = 0;
let state = 0;
let promptId = null;
let ended = false;
let sending = false;

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
  if (!p) {
    $("prompt").replaceChildren();
    promptId = null;
    return;
  }
  if (p.id === promptId) return;
  promptId = p.id;
  const form = $("prompt");
  form.replaceChildren();
  text("h2", p.question, form);
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
    form.append(input);
    const submit = () => {
      const value = input.value;
      if (p.kind === "secret") input.value = "";
      answer(p, value);
    };
    input.addEventListener("keydown", e => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        submit();
      }
    });
    button(form, words.send, submit);
    if (p.kind === "line") {
      const suggestions = text("div", "", form);
      suggestions.id = "suggestions";
      input.addEventListener("input", () => {
        suggestions.replaceChildren();
        const value = input.value.toLowerCase();
        if (value.startsWith("!") || (value.startsWith("/") && !/\s/.test(value))) {
          for (const [name, hint] of Object.entries(p.commands || {})) {
            if (name.startsWith(value)) {
              button(suggestions, `${name} — ${hint}`, () => {
                input.value = name;
                input.focus();
                suggestions.replaceChildren();
              });
            }
          }
        }
      });
    }
    input.focus();
  }
  button(form, words.cancel, () => {
    if (input && p.kind === "secret") input.value = "";
    answer(p, null, true);
  });
  form.onsubmit = e => e.preventDefault();
}

function event(e) {
  if (e.type === "out") {
    const node = text("div", "", $("messages"));
    node.className = "out";
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
          if (labels.ok) words = await labels.json();
        } catch {}
        $("stop").textContent = words.stop;
      }
      for (const e of data.events) {
        event(e);
        after = Math.max(after, e.seq);
      }
      state = data.state;
      rich($("live"), data.live || "");
      showPrompt(data.prompt);
      $("stop").hidden = !data.busy;
      ended ||= data.ended;
      if (data.events.length) window.scrollTo(0, document.body.scrollHeight);
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
    await poll();
  } catch {
    $("status").textContent = "DESMAN";
  }
})();
