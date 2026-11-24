# Operator mode

A desman-global operator holds an Ed25519 key with a time-limited assignment signed
by the registry's root owner. Either operator scope lets you inspect the inbox, work,
reports, nodes and registry log. The `tasks` scope lets you offer and settle work and
set node owners. The `moderation` scope covers complaints and suspensions.

Import the server's existing `operator.key` from a securely transferred local copy,
or create a fresh key and give the displayed public key to the root owner. The root
owner runs `desman-registry assign PUBLIC_KEY --scope tasks` (use the required scope
or repeat `--scope` for both). DESMAN keeps the private key in its data directory and
signs requests and records locally. Use `--registry URL` during import or creation to
specify the registry.

## Commands

Each line starts with `desman operator`. Read commands that take `--json` emit the
registry reply as JSON. Text views show times in your local timezone.
NODE is a node fingerprint or its unique prefix.

| Command | What it does |
| --- | --- |
| `desman operator status` | Show the key fingerprint, registry and current assignment. |
| `desman operator key import FILE [--registry URL] [--replace]` | Import an existing operator key. |
| `desman operator key create [--registry URL]` | Create a key and print the public key for assignment. |
| `desman operator key remove [--yes]` | Preview or remove the local key. |
| `desman operator inbox [--json]` | Show jobs and clusters waiting for a decision. |
| `desman operator jobs [--status S] [--json]` | List jobs, optionally filtered by status. |
| `desman operator job show ID [--json]` | Show a job, its result and reviews. |
| `desman operator job add --description TEXT --acceptance TEXT [--kind KIND] [--runs-code] [--input NAME=TEXT]... [--days N]` | Sign and offer a job, capped by the assignment term. |
| `desman operator job settle ID DECISION` | Accept, send back (`again`), or drop a job. |
| `desman operator clusters [--status S] [--json]` | List cluster tasks, optionally filtered by status. |
| `desman operator cluster show ID [--json]` | Show a cluster task, its plan, jobs and result. |
| `desman operator cluster add --goal TEXT --acceptance TEXT [--input NAME=TEXT]... [--days N]` | Sign and offer a cluster goal, capped by the assignment term. |
| `desman operator cluster settle ID DECISION [--note TEXT]` | Accept, send back (`again`), or drop a cluster task. Attach a note when sending it back. |
| `desman operator complaints [--all] [--json]` | List open complaints, or include settled complaints. |
| `desman operator complaint show ID [--json]` | Show a complaint, its quoted message and decisions. |
| `desman operator complaint dismiss ID` | Dismiss a complaint. |
| `desman operator suspend NODE --reason TEXT [--days N] [--complaint ID]...` | Sign a suspension, optionally citing complaints. |
| `desman operator lift NODE --reason TEXT` | Sign a lift of the active suspension. |
| `desman operator suspensions [--json]` | List moderation records. |
| `desman operator reports [--kind KIND] [--json]` | List shared reports. Kind can be `bug` or `memory`. |
| `desman operator report show ID [--json]` | Read a shared report. |
| `desman operator nodes [--json]` | List nodes, owners and capabilities. |
| `desman operator owner NODE [NAME] [--clear]` | Set a node's owner name or clear it. |
| `desman operator log [--limit N] [--json]` | Read recent registry journal entries. |

The same commands work in chat with `/operator`. When the interlocutor proposes a
change, DESMAN shows the proposed record or decision and asks you to confirm it.
DESMAN displays registry text directly and asks before sharing other people's words
with the interlocutor as untrusted data.

## Server administration

The root owner still handles assignments and revocations, trust signing, appointer
setup, proposals and releases, backups, serving, and updates on the registry server.
The server's operator commands remain available there. See the
[registry deployment guide](registry-deploy.md).
