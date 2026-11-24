# ADR 0016: Operator mode in DESMAN

Status: accepted (2026-11-24, ROADMAP O.1–O.3)

## Context

A global operator (ADR 0004, 0006) holds `operator.key`, assigned by the root for a term with
scopes (`tasks`, `moderation`). Today every operator action is a `desman-registry` command run
on the registry's own machine: the inbox of jobs and cluster tasks waiting for a word, job and
cluster task records, settling them, complaints and suspensions, bug reports, node owners.

The owner wants operators to work from an ordinary DESMAN, on any machine where their key is,
the way a participant uses desman-global: in the terminal, the chat and the page. The phase T
end state (`desman-global-vision`) has human global operators as the backup of the operators
DESMAN appoints, and they cannot all have a shell on the server.

## Decision

**Operator mode is part of DESMAN, switched on by giving it an operator key. It talks to the
registry over new `/v1/operator/*` paths, each request signed by the operator key (the same
`net.request` signatures as a node's). The registry acts only for a key with a current root
assignment holding the scope the action needs.**

1. **The key.** `desman operator key import FILE` copies an Ed25519 private key into the data
   directory as `operator.key` (0600, `getattr(os, "O_BINARY", 0)`), checks it, and shows its
   fingerprint; `operator key create` makes a new one and prints the public half for the root
   owner (`desman-registry assign`, unchanged). `operator key remove` deletes it after a
   confirmation. The key never reaches a model, the chat history, a log or `config.json`; the
   chat shows only its fingerprint. Without the key nothing of the mode is shown (the page's
   section, the chat's catalog entries answer "no operator key").
2. **Authority stays in the registry.** A request is verified as any other (signature, time,
   nonce, revocation). Then the registry finds the caller's newest root-signed `assignment`:
   none, expired or revoked → `not_operator` (403), journaled. Each path names its scope:
   reading reports, nodes, jobs, clusters, inbox, log and whoami needs either operator scope.
   Listing or reading complaints, listing suspensions, dismissing complaints, suspending and
   lifting suspensions need `moderation`. Adding or settling jobs and cluster tasks, and
   changing node owners, need `tasks`. Missing scope → `scope` (403), journaled.
   The operator need not be a registered node.
3. **Records are signed on the operator's machine.** A job (`task`), a cluster task
   (`cluster_task`), a suspension and its lifting (`moderation`) are built and signed by
   DESMAN with the operator key, exactly as `desman-registry` builds them (shared builders, not
   copies), and sent whole; the registry checks them as it checks records today. Decisions that
   are not records (settle a job or a cluster task, dismiss a complaint, set a node's owner) are
   authorised by the signed request and journaled with the operator's fingerprint.
4. **What stays on the server.** Everything that needs the root key or the server's files:
   `assign`, `revoke`, `trust sign`, the appointer, proposals and releases, `backup`, `update`,
   `serve`. The mode never asks for the root key.
5. **The surfaces.** CLI `desman operator ...` (status, inbox, jobs, job add|show|settle,
   clusters, cluster add|show|settle, complaints, complaint show|dismiss, suspend, lift,
   suspensions, reports, report show, nodes, owner, log); every command has its catalog entry,
   reading → shown by code, changing → the confirmation window built by code with the exact
   record or decision; the page gets an "Operator" section. What the registry returns from
   other people (complaint texts, quoted messages, reports, results) is shown as untrusted and
   reaches the interlocutor only when the person asks, marked untrusted. No tool lets an agent
   act as the operator; every operator action is the person's command or confirmation.
6. **Journals.** The registry journals operator mutations with the caller's fingerprint,
   action path and record id. It also journals successful complaint detail reads as
   `operator_read_complaint` with the caller's fingerprint and complaint id. Routine reads
   such as inbox and whoami add no operator entries. Refusals are journaled as above.
   Journal entries omit private message text. The operator's local `network_log` records
   operator activity using fingerprints and record ids.

## Consequences

- The operator no longer needs a shell on the registry's machine; the root owner still does.
- Two places can now settle the same job: the registry applies the first decision and answers
  the second with `already_settled` (409).
- A stolen operator key gives the scopes it was assigned until the root revokes it, as today;
  keeping it outside `config.json` keeps it out of config backups and screenshots.
- `desman-registry`'s operator commands stay (they are the server's fallback) and share the
  record builders and the settle logic with the new paths.
