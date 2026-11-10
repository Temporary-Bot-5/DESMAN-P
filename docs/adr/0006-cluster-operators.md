# ADR 0006: Cluster operators — DESMAN appoints a node to lead a cluster

Status: accepted (ROADMAP T.4; owner's requirements 2026-11-05 in `PROJECT_PLAN.md` and the
T.4 entry of `ROADMAP.md`)

## Context

ADR 0004 left the cluster to the registry: a global operator (the owner, with an operator key
of scope `tasks`) signs single, self-contained jobs, and the registry lends each one to a node,
then a node of another owner rechecks the result (`registry/work.py`). Nobody plans: a goal
larger than one subtask has to be cut into jobs by the person.

The end state in `PROJECT_PLAN.md` has a middle level: **cluster operators**, nodes appointed by
DESMAN itself, which assemble free modules into a planner, executors and reviewers for a task of
the global operator and supervise them. The friends' test (phase T) must show that DESMAN
appoints a cluster and its operator by itself when there are enough free modules. The open
questions the plan left for this ADR, and the answers T.4 already fixed:

- **Trust.** People write the rules of appointment (how long the node has worked, how its
  results held up when rechecked). An appointment is signed, short-lived, revocable; its scope
  is the work of its cluster only: never releases, moderation, or appointments.
- **Reachability.** Through the registry, no peer-to-peer (`AGENTS.md`: no p2p).
- **Storage.** Only network records leave a machine.
- **Disappearance.** The role is held under a lease; another node takes it over.

What must keep holding: a node acts only on records that pass `sign.Trust`; network work stays
untrusted, LOW and preempted by the person's own (ADR 0004, decision 6); keys never leave their
machine; every step is journaled (`network_log` on the node, `registry_log` on the server).

## Decision

### 1. A cluster is formed for one cluster task, and dissolved when it is settled

- **A cluster task** (`cluster_task` record, new kind) is a goal with acceptance criteria and
  optional inputs (`protocol.ClusterTask`: like `Job`, but a goal to plan, not one subtask). The
  global operator signs it with its operator key (scope `tasks`, as for a job):
  `desman-registry cluster add`.
- **A cluster** is the set of nodes working on one cluster task: one **operator** and its
  **members** (the nodes its jobs are lent to). It exists from the first appointment to the
  task's settlement, then dissolves. Long-lived clusters that keep information for each other
  (the plan's end state) are a later ADR; nothing here prevents them.
- A node is in at most one cluster at a time, as operator or member. It still takes the
  registry's ordinary jobs when its cluster has none for it.

### 2. Who appoints: an appointer key, assigned by the root, with one scope

- The root assigns a new scope, `appoint`, to a key of its own (the **appointer**). The
  appointer signs only `appointment` records (`sign.OPERATOR_KINDS`); with `appoint` alone it
  cannot sign tasks, moderation, revocations, releases or assignments. The root keeps the
  `appoint` scope apart from `tasks` and `moderation` by refusing to sign an assignment that
  mixes them (`admin.assign`); `assign` without `--scope` keeps meaning `tasks` and
  `moderation`.
- The appointer key lives on the registry's server (`appointer.key` in the registry's directory,
  `0600`, made by `desman-registry appointer create`; the person assigns it with the root,
  `desman-registry assign --scope appoint`, like any operator). It is the only key the
  registry's code uses **by itself**: "DESMAN appoints" means the registry signs appointments
  with it, under the rules of decision 3, without a person in the loop. The assignment is in the
  trust list, like every assignment.
- **Why an online key is acceptable:** whoever holds the server can already hide or replay
  work; with the appointer key they can additionally name a cluster operator, and an operator can
  only cut a cluster task **the global operator signed** into jobs for that task (decision 4).
  The root revokes the appointer like any operator, and every appointment it signed falls with
  it.

### 3. No thresholds: every result is reviewed on another device

The owner (2026-11-10): no thresholds of age, history or cluster size; what matters is that a
result is never reviewed on the device that made it, so that a malicious node cannot go
unnoticed. One node = one device (its node key, `net/sign.py`).

- **Free** = active, takes jobs (`Claim.jobs`), heartbeat within the registry's usual window,
  not revoked, not in another cluster. To lead it must also allow it (`Claim.lead`, decision 9).
  These are the facts of being online, not trust thresholds; nothing in a file to tune.
- **The one rule (in code, `registry/work.py`; a test holds it):** the recheck of a cluster job is
  lent only to a node that is neither the node that executed it nor the cluster's operator. A
  plan, a job's inputs and the final result are records of the operator, so the operator never
  reviews; an executor never reviews its own result. A recheck may go to a member or to any free
  node outside the cluster; it never waits for the cluster to have one.
- **Enough free modules:** a free node that may lead and two other free nodes (one to execute,
  another to recheck). Then the registry appoints. The leader is the free node with the most
  results accepted on recheck (then the oldest); members are taken in the same order, at most
  six (a constant).
- Every appointment and the reason for the choice (counts, not content) go to `registry_log`.
  `desman-registry clusters` shows what waits and why ("2 of 3 free nodes").

### 4. Authority: one more link, narrower at each step

```
root ──assignment(scope tasks)────▶ global operator ──cluster_task─────────────┐
root ──assignment(scope appoint)──▶ appointer ──appointment(cluster, task, ───▶ cluster operator
                                               members, term ≤ 30 min)          │
                     cluster_plan, cluster_job, cluster_result ◀────────────────┘
```

- **Appointment** (new kind, signed by an operator with scope `appoint`): the cluster id, the
  `cluster_task` record id, the operator's public key, the members' fingerprints, a **term of at
  most 30 minutes**. `Trust` gains this one second link: it keeps appointments it received as
  state, like assignments, and drops them when their signer is revoked or loses `appoint`.
  Short terms take the place of revocation; an appointment is not in the trust list (a hidden
  "revocation" is hidden for at most its term).
- **What a cluster operator signs,** only inside the term of its current appointment and only
  naming that appointment's cluster task:
  - `cluster_plan`: the subtask graph for the cluster task (as `planner.Plan`: ≤8 nodes, a DAG,
    each with its `Kind`), versioned (decision 6);
  - `cluster_job`: one node of the plan, as a `Job`, plus the cluster id, the plan version and
    node, and its inputs (the accepted results of the node's dependencies, copied by code);
  - `cluster_result`: the cluster task is done, naming the accepted result of the plan's last
    node.

  Nothing else: never a `task` outside its cluster, a moderation decision, a revocation, a
  release, an assignment or an appointment. `Trust` checks the kind against the signer's
  appointment, the term, and that the record names the appointment's cluster and task; a node
  also checks it is one of the appointment's members before it takes a `cluster_job`.
- The registry checks every cluster record the way a node will (`store.trust`) before keeping
  it, as `admin` does for records it signs.

### 5. All traffic goes through the registry

The operator and the members never connect to each other. The registry is the mailbox and the
lender, with the same pull-only HTTPS, signed requests and leases as ADR 0004:

- the operator polls its cluster (`GET /clusters/mine`): the cluster task, its appointment, the
  current plan, every job of the cluster with its state, results and verdicts;
- it hands in plans, jobs and the final result (`POST /clusters/{id}/records`);
- a `cluster_job` is lent with `registry/work.py`'s machinery (lease and token, `execute` then
  `recheck`, `MAX_LOST`, `MAX_FAILED`), only to the cluster's members for `execute`. The recheck
  goes to a node other than its executor and the operator (decision 3), preferably a member,
  otherwise any free node: a recheck must never wait for the cluster to have one.

### 6. What the operator does: plan, lend, supervise, finish

The operator's work is a loop in `net/node.py`, between a worker's steps and only while the
person's own work leaves room (as for jobs, decision 6 of ADR 0004):

1. **Plan.** No plan yet: the planner (role `network_planner`: no tools, lent modules only,
   within their quotas) turns the cluster task into a `Plan`, checked as locally
   (`planner.py`); the template plan if the model fails. Signed as `cluster_plan` version 1.
2. **Lend.** Every plan node whose dependencies are accepted becomes a `cluster_job` (its inputs
   = those results, within `MAX_INPUT_BYTES`, cut by code with a marker if larger).
3. **Supervise.** A job the registry gave up on (failed, leases lost, disputed) comes back to
   the operator once: it may split it (the decomposer, 2–4 parts) into a new plan version that
   keeps every accepted node as it is (as ADR 0010: history is not rewritten). At most two new
   versions per cluster task; after that, and for a job of a `runs_code` node, the cluster task
   goes to the global operator's inbox (`desman-registry inbox`), never settled silently.
4. **Finish.** The last node's result accepted on recheck → `cluster_result`. The global
   operator sees it in the inbox and accepts it, sends it back with a note (a new plan version,
   as 3), or drops it: the person is the last word on a cluster task, as on a disputed job.

The operator does not execute or review its cluster's jobs itself (it may be lent ordinary jobs
of other clusters). Its model calls are journaled in `model_calls` with the role and the cluster
task, and count against the lent modules' daily quotas.

### 7. The role is a lease: an operator that disappears is replaced from the records

- The appointment's term is the lease. While the operator polls (`GET /clusters/mine` counts as
  its heartbeat), the registry signs it a fresh appointment before the old one is past half its
  term, with the current members (gone members replaced by free nodes).
- An operator not seen for one term is replaced: the registry appoints another free node that may lead
  (not the old one, for this task) for the same cluster. What the old one signs later is refused
  by the registry and by the nodes: its appointment is past its term or superseded (the newest
  appointment of a cluster is the only one that counts).
- **Nothing is lost,** because everything the operator does is a signed record the registry
  keeps: the new operator reads the cluster's plan, jobs, results and verdicts and continues from
  them. Jobs already lent keep running under their leases; their results stand.
- After three operators lost on one cluster task, it goes to the inbox.

### 8. What is stored where

- **Only network records** cross machines: cluster tasks, appointments, plans, jobs, results,
  verdicts. Never a person's own task, chat, profile, knowledge node or key. A record is signed,
  so whoever holds a copy cannot change it.
- In this ADR the registry keeps the copies, and its backup (`desman-registry backup`) is the
  global operator's backup of everything a cluster holds. Copies on several nodes with disk
  quotas the person donates (the plan's end state) are a later step, with long-lived clusters.
- The operator node keeps nothing a successor needs; it keeps what it did in `network_log` and
  its calls in `model_calls`, like any node.

### 9. The person decides whether their node may lead

- `Claim.lead` (protocol) says the node allows being appointed. It comes from a new setting
  `network.lead` (on by default once a person joins and lends a module; `Config` schema bump):
  leading spends only what the person already lent, under the same quotas, and sends only
  network records. `desman global lead on|off`, the catalog, the page's DESMAN-Global settings.
  Off → the node is never appointed; an operator turned off mid-way stops renewing and is
  replaced (decision 7).
- What the person sees: `desman global` / the page say "leads cluster C (task …), N members"
  or "member of cluster C"; the cluster operator's work appears in Modules under DESMAN-Global
  like other lent work, with its role. The manual's desman-global topic says what leading means.

### 10. Policy

New role `network_planner` in `policy.toml` (protected): no tools. Cluster jobs run as ordinary
network jobs (`network_executor`, `network_reviewer`) under the same policy; the person's
`permissions` stay the upper bound.

### 11. Tests without the network

Registry and nodes in one process (`MockTransport`, `FakeClock`), as in ADR 0004. Fault scenarios
in `tests/faults.py`: the operator killed mid-cluster (another takes over, the plan continues,
no job lost or done twice), a late record of a replaced operator (refused), a member gone
(replaced, its lease lent again), a forged appointment, an appointment past 30 minutes, the
appointer revoked (all its clusters stop), an operator signing outside its cluster (refused).

## Consequences

- `net/sign.py` and `policy.toml` change: protected files;
  the owner approves them after each merge (`desman protected approve`).
- The "at most one link below the root" rule of `sign.Trust` becomes "one link, plus
  appointments signed by an operator with `appoint`"; nothing else gains a second link.
- The appointer key is a second key on the server, used without a person; its assignment can
  be revoked by the root like any other.
- Split into PRs (ROADMAP T.4a–T.4d): records and trust; the registry's side; the node's side;
  fault scenarios and the in-process demo.
