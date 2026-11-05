# Demo D10 on a live registry

`tests/test_demo_d10.py` runs the story with a real in-process registry, three node identities,
scripted models, benchmark subprocesses and temporary git repositories. This page checks it
on the owner's server and two test machines. Three identities are necessary: a node is never
lent the same job again, even after losing its lease. Use A on the first machine and B on the
second; C can run on the second machine with separate config/data directories and node key.
B and C must represent different owners. Keep their folders and terminal environments distinct.

## Before you start

The live registry checkout `~/desman-registry-app` was at G.4. Upgrade it to current `master`
before this check, following the upgrade and backup instructions in
[registry-deploy.md](registry-deploy.md). Restart its service after installing the upgraded
package. Keep the root private key on the owner's server; every machine must have the matching
`src/desman/net/root.pub`. Synchronize the machines' clocks.

Use disposable node data/config folders and clean git clones of the same base commit. The
operator's release checkout and A's checkout must fetch from the same remote. Arrange a test
remote the owner can push to; nodes need only read access. Do not publish demo changes to the
production remote. Install each checkout in its own virtual environment with `pip install -e .`.
Run A's commands from A's checkout so updates move that installation.

On the server, with the intended `DESMAN_REGISTRY_DIR` set:

```bash
desman-registry assign --scope tasks
desman-registry trust sign --days 7
desman-registry trust show
```

Expected: an active operator assignment and a fresh root-signed trust list. Releases are signed
by the root directly; no additional operator scope is needed for this story. Renew the list
daily as described in the deployment guide.

On each node terminal, set unique directories (replace `A` with `B` or `C` as appropriate):

```bash
export DESMAN_CONFIG_DIR="$HOME/desman-d10/A/config"
export DESMAN_DATA_DIR="$HOME/desman-d10/A/data"
desman config
```

Configure a running local Ollama server and a model through the wizard. The live run uses real
models; the automated test uses FakeLLM. If B and C share one machine, use separate node folders
and workers. Default registry owners are their distinct key fingerprints; do not assign B and
C the same owner with `desman-registry owner`.

## The check

1. **Join and lend a module.** On A, B and C (substitute the actual HTTPS address):

   ```bash
   desman global join --registry https://registry.example.org
   desman global donate ollama --calls 100 --tokens 100000
   desman global status
   ```

   Expected: each node is enabled and lends its local module within the displayed daily quotas.
   Start only A's worker for now. B and C must not claim the first execution.

2. **Add a network task.** On the server:

   ```bash
   desman-registry job add --description "Sum 2 and 3" --acceptance "Says 5"
   desman-registry jobs
   ```

   On A, run `desman worker`. On the server, use `desman-registry job show JOB --json`
   (replace JOB with the returned job ID) to observe its lease.

   Expected: A registers, takes the job and starts executing it. Network work has low priority
   and uses only donated modules. Record A's worker PID before the next step.

3. **Crash A before hand-in.** From another terminal, run `kill -KILL PID` against A's worker.
   A small sum can finish quickly: if the server already received its result, repeat with a new
   job and kill immediately after the lease appears. Leave A stopped until the lease expires
   (currently ten minutes after its last extension). Inspect with:

   ```bash
   desman-registry jobs
   desman-registry job show JOB --json
   desman-registry inbox
   ```

   Expected: the old lease ends as expired and the job remains available for execution. One
   lost lease does not send the job to the operator's inbox. The automated test crashes A
   after local completion but before transmission, and verifies its late result is refused.

4. **B executes, C rechecks.** Start `desman worker` on B. After the server shows a result and
   the job is awaiting recheck, restart `desman worker` on A with its original folders/key.
   Then start `desman worker` on C.

   ```bash
   desman-registry job show JOB --json
   desman-registry jobs
   desman-registry inbox
   ```

   Expected: A receives no second lease for JOB; any late hand-in is refused as `lease_lost`.
   C reviews B's signed result under its own lease, and JOB becomes `accepted`. Result and
   verdict come from different owners. No node waits for a human decision. The node databases'
   `network_log` records work taken, handed in or dropped; the registry log records the expired
   lease. `desman-registry log` shows server events.

5. **A proposes and previews an upload.** On A:

   ```bash
   desman proposals analyze
   desman proposals list
   desman proposals show N
   desman proposals evaluate N
   desman global submit-proposal N
   ```

   Replace N with a proposed change to text targets only. Analysis can return no proposal when
   there is no useful failure evidence: prepare ordinary local tasks that expose a repeatable
   planning problem first, then analyze again. Do not manufacture a production database row.

   Expected: evaluation finishes locally; the upload command displays the exact signed payload,
   destination and confirmation hash, and sends nothing. Inspect the payload for private task
   text. A proposal with tuning settings cannot be submitted. On the server,
   `desman-registry proposal list` must still show no new submission from this preview.

   Confirm the displayed payload on A:

   ```bash
   desman global submit-proposal N --confirm HASH
   ```

   Expected: exactly that submission is sent and gets a registry proposal ID. A wrong hash
   sends nothing. The local `network_log` records `proposal_submitted`.

6. **Another node checks it.** Leave B's worker running; stop C's worker if B must do this check.
   On the server:

   ```bash
   desman-registry proposal list
   desman-registry proposal show ID --json
   ```

   Expected: another owner's node gets a proposal-check lease, validates the permitted text
   diff, runs the benchmark in isolated worktrees and hands in a signed check. State becomes
   `reviewed`, check outcome is `checked`, and `matches_submission` is true. The normal local
   evaluation defaults to fake mode; the network independently repeats that mode.

7. **The operator releases it.** Stop A's worker before moving its checkout. On the server,
   supply the clean release checkout cloned from the shared test remote:

   ```bash
   desman-registry proposal accept ID --repo /path/to/release-checkout
   desman-registry proposal accept ID --repo /path/to/release-checkout --confirm HASH
   desman-registry release list
   git -C /path/to/release-checkout push origin HEAD
   ```

   Expected: the first command only previews the commit and hash. Only the second command,
   with that hash, creates the commit and root-signed release. Inspect the changed planner
   text and commit ID. Pushing is the owner's separate action; the registry never pushes.

8. **A receives, explicitly installs, then reverts.** Restart A's worker long enough to receive
   the release, then stop it. In A's checkout:

   ```bash
   git rev-parse HEAD
   desman global update --show
   git rev-parse HEAD
   desman global update --apply 1
   git rev-parse HEAD
   desman global update --revert
   git rev-parse HEAD
   ```

   Substitute the displayed release number for 1 if this registry already has releases.
   Expected: receipt records `release_received` and moves no code. `--show` fetches/verifies and
   displays the commit/files, leaving HEAD unchanged. The person's `--apply NUMBER` command
   fast-forwards to the signed commit and changes the planner text; there is no extra hash
   prompt for updates. `--revert` returns HEAD and the text to their pre-update state.
   The node's `applications` journal records `applied`, then `reverted`, and `network_log` has
   `update_applied` and `update_reverted`. This text-only release changes no protected files;
   if a different release lists protected changes, inspect and run `desman protected approve`
   as instructed before resuming tasks. Restart workers after moving code.

Record the job, submission and release IDs, observed commit IDs, and each Expected result.
A disputed result, missing fresh trust list, exhausted quota or no analyst proposal needs an
operator decision before the live check can be called complete.
