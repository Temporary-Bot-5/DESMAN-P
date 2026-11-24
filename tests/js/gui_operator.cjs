const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
  }

  append(node) { this.children.push(node); }
  replaceChildren() { this.children = []; }
  setAttribute() {}
  addEventListener(event, callback) { this[event] = callback; }
  focus() {}
  set innerHTML(_) { throw new Error("Operator content must be rendered as text"); }
  get isConnected() { return true; }
}

const elements = new Map();
const actions = [];
const document = {createElement: tag => new Element(tag)};
const context = vm.createContext({
  document, URLSearchParams,
  words: new Proxy({lang: "en"}, {get: (object, key) => object[key] ?? key}),
  $: id => {
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  },
  button: (parent, label, onclick) => {
    const button = new Element("button");
    Object.assign(button, {textContent: label, onclick});
    parent.append(button);
    return button;
  },
  settingInput: (parent, label, value) => {
    const input = new Element("input");
    Object.assign(input, {label, value});
    parent.append(input);
    return input;
  },
  settingSave: (parent, change, label) => {
    const button = new Element("button");
    button.textContent = label;
    button.onclick = () => actions.push(JSON.parse(JSON.stringify(change())));
    parent.append(button);
  },
  updateActions() {},
});

const source = fs.readFileSync(process.argv[2], "utf8");
const run = code => vm.runInContext(code, context);
run(source.slice(source.indexOf("function text("), source.indexOf("// Linkify")));
run(source.slice(source.indexOf("let operatorOpen = false;")));

const children = parent => [parent, ...parent.children.flatMap(children)];
const nav = () => elements.get("settings-nav").children.map(button => button.textContent);

run("showOperatorButton(false)");
assert.equal(elements.get("operator-button").hidden, true);
run("showOperatorButton(true)");
assert.equal(elements.get("operator-button").hidden, false);
run('operatorStatus = {scopes: ["tasks"]}; drawOperator()');
assert.ok(!nav().includes("op_moderation"));
run('operatorStatus.scopes.push("moderation"); drawOperator()');
assert.ok(nav().includes("op_moderation"));

const script = '<script>alert("registry result")</script>';
const root = new Element("div");
context.root = root;
context.inbox = {
  jobs: [{job: "job", task: {description: script}, result: script}],
  clusters: [{task: "cluster", goal: {goal: "Goal"}}],
  job_results: {job: script},
  cluster_results: {cluster: script},
};
run("operatorInbox(root, inbox)");
const quotes = children(root).filter(node => node.tag === "pre");
assert.deepEqual(quotes.map(node => node.textContent), [script, script]);
assert.ok(quotes.every(node => node.children.length === 0));

const press = (parent, label) => children(parent).find(node => node.tag === "button" && node.textContent === label).onclick();
const [job, cluster] = root.children;
cluster.children.find(node => node.label === "op_note").value = "Recheck";
for (const card of [job, cluster]) {
  for (const label of ["op_accept", "op_again", "op_drop"]) press(card, label);
}
assert.deepEqual(actions.splice(0), [
  {section: "operator", operation: "job settle", args: {id: "job", decision: "accept"}},
  {section: "operator", operation: "job settle", args: {id: "job", decision: "again"}},
  {section: "operator", operation: "job settle", args: {id: "job", decision: "drop"}},
  {section: "operator", operation: "cluster settle", args: {id: "cluster", decision: "accept", note: "Recheck"}},
  {section: "operator", operation: "cluster settle", args: {id: "cluster", decision: "again", note: "Recheck"}},
  {section: "operator", operation: "cluster settle", args: {id: "cluster", decision: "drop", note: "Recheck"}},
]);

// Results read as the CLI prints them.
assert.equal(run(`operatorText(${JSON.stringify(JSON.stringify({text: "Line 1\nLine 2"}))})`), "Line 1\nLine 2");
assert.equal(run(`operatorText(${JSON.stringify(JSON.stringify({a: 1, b: "x"}))})`), '{\n  "a": 1,\n  "b": "x"\n}');
assert.equal(run("operatorText('plain words')"), "plain words");
assert.equal(run("operatorText(null)"), null);

for (const [kind, label] of [["job", "op_description"], ["cluster", "op_goal"]]) {
  root.replaceChildren();
  run(`operatorOffer(root, ${kind === "cluster"})`);
  const form = root.children[0];
  const values = {[label]: "Work", op_acceptance: "Done", op_inputs: "NAME=TEXT\n\nTAG= value ", op_days: "4"};
  for (const input of form.children.filter(node => node.label in values)) {
    input.value = values[input.label];
  }
  if (kind === "job") {
    const runs = form.children.find(node => node.label === "op_runs_code");
    runs.checked = true;
    children(form).find(node => node.tag === "select").value = "code";
  }
  press(form, kind === "job" ? "op_job_offer" : "op_cluster_offer");
  assert.deepEqual(actions.pop(), {
    section: "operator", operation: `${kind} add`, args: {
      [kind === "job" ? "description" : "goal"]: "Work", acceptance: "Done",
      inputs: ["NAME=TEXT", "TAG= value "], days: 4,
      ...(kind === "job" ? {kind: "code", runs_code: true} : {}),
    },
  });
}

const field = (parent, label) => children(parent).find(node => node.label === label);
for (const installed of [false, true]) {
  root.replaceChildren();
  run(`operatorKeySettings(root, ${installed})`);
  field(root, "op_file").value = "/operator.pem";
  field(root, "registry").value = "https://registry.test";
  press(root, "op_import");
  assert.deepEqual(actions.pop(), {
    section: "operator", operation: "key import", args: {
      file: "/operator.pem", registry: "https://registry.test", replace: installed,
    },
  });
  press(root, installed ? "remove" : "op_create");
  assert.deepEqual(actions.pop(), {
    section: "operator", operation: installed ? "key remove" : "key create",
    args: installed ? {} : {registry: "https://registry.test"},
  });
}

context.nodes = {nodes: [{
  name: script, node: "node", status: "active", owner: "Old owner", capabilities: [],
}]};
root.replaceChildren();
run("operatorNodes(root, nodes)");
field(root, "op_name").value = "New owner";
press(root, "save");
press(root, "op_clear");
assert.deepEqual(actions.splice(0), [
  {section: "operator", operation: "owner", args: {node: "node", name: "New owner"}},
  {section: "operator", operation: "owner", args: {node: "node", clear: true}},
]);
assert.ok(children(root).some(node => node.tag === "td" && node.textContent === script));

context.suspensions = {suspensions: [{node: "node", action: "suspend", lifted: false, expires_at: "2999-01-02T00:00:00Z"}]};
root.replaceChildren();
run("operatorSuspensions(root, suspensions)");
field(root, "op_reason").value = "Resolved";
press(root, "op_lift");
assert.deepEqual(actions.pop(), {
  section: "operator", operation: "lift", args: {node: "node", reason: "Resolved"},
});

context.complaint = {id: "complaint", about_node: "node", status: "open", text: script, message_text: script};
context.fetch = async path => {
  assert.equal(path, "/operator?view=complaint&id=complaint");
  return {ok: true, json: async () => ({has_key: true, ...context.complaint})};
};
(async () => {
  root.replaceChildren();
  run("operatorOpen = true; operatorComplaints(root, {complaints: [complaint]})");
  const panel = children(root).find(node => node.tag === "details");
  panel.open = true;
  await panel.toggle();
  assert.deepEqual(children(panel).filter(node => node.tag === "pre").map(node => node.textContent), [script, script]);
  press(panel, "op_dismiss");
  assert.equal(field(panel, "op_complaints").value, "complaint");
  assert.equal(field(panel, "op_complaints").readOnly, true);
  field(panel, "op_reason").value = "Abuse";
  field(panel, "op_days").value = "30";
  press(panel, "op_suspend");
  assert.deepEqual(actions.splice(0), [
    {section: "operator", operation: "complaint dismiss", args: {id: "complaint"}},
    {
      section: "operator", operation: "suspend",
      args: {node: "node", reason: "Abuse", days: 30, complaints: ["complaint"]},
    },
  ]);

  for (const view of ["jobs", "clusters", "complaints", "suspensions", "nodes", "reports"]) {
    context.fetch = async path => {
      assert.equal(path, `/operator?view=${view}`);
      return {ok: true, json: async () => ({has_key: true, [view]: []})};
    };
    root.replaceChildren();
    run(`operatorList(root, "${view}", () => { throw new Error("Empty lists should skip rendering rows"); })`);
    await root.children[1].reloadOperator();
    assert.deepEqual(root.children[1].children.map(node => node.textContent), [`op_empty_${view}`]);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
