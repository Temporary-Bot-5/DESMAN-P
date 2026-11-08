# ADR 0008: Modules the person stops or removes

Status: accepted (owner's request and answers, 2026-11-08)

## Context

The Modules tab (T.2h) shows one card per task and role, under a long heading with the task's
whole goal: it repeats the Tasks tab and does not answer what the owner asks of it — which modules
exist, which work for desman-global and which for the person's own tasks and chats. DESMAN expects
modules to disappear now and then (a key revoked, a quota used up, a local server gone); the
person needs the same two moves by hand: stop a module for a while, or take it away so that the
work goes on with another one.

## Decision

1. **A module is a source and a model**: a configured key (by fingerprint) or a local server
   (Ollama, llama.cpp), and the model it serves. The Modules tab shows one card per module that is
   assigned or worked recently (the first of a role's lineup for an unfinished task or an open
   chat, the modules lent to desman-global, and those that answered in the last hour), in two
   sections: "For DESMAN-Global" and "For your tasks and chats". Under a card, one short line per
   task or chat it works for: the task's title cut to about 30 characters (or the chat's name)
   and the role. Interlocutors are modules like the others: a chat's interlocutor is listed under
   its module with the chat's name. The long goal stays in the Tasks tab only.
2. **Stop (no confirmation) = pause.** A stopped module gets no new calls. The work it was doing
   waits: the unfinished tasks it works for are paused (`Store.pause_task`, as "Stop all
   modules"), the turns of the chats it answers wait in their queue with the reason shown. The
   card shows "Continue": it lifts the stop and resumes exactly the tasks this stop paused (a task
   paused by the person in another way stays paused). A stopped module is never chosen for new
   work meanwhile: the router skips it.
3. **Remove (confirmation in a code-built window) = replace.** A removed module is excluded for
   the work it is listed under (from a task's or a chat's card: that task or chat only; from the
   Modules tab: every task and chat it works for now, and new work) — the router drops it from
   the lineup, so the role continues with the next model of its lineup, never the same one. If
   the lineup is then empty the work waits, shown as "no module left for ROLE" (the person adds
   a key or restores the module in Settings). If no work is left for that role, nothing replaces
   it. A removal can be undone from Settings → Roles ("Removed modules: restore").
4. **Stored in the database**, not in `config.json`: table `module_holds` (source, model, scope
   `all | task | session | network`, scope id, state `stopped | removed`, at, paused task ids for
   a stop), changed only by the person's page/CLI/chat action (a chat request only after the
   person's yes in the window), each change in `operator_log`. Agents never write it.
5. **desman-global.** Modules lent (`Network.donated`, by provider) are listed under "For
   DESMAN-Global" with the network jobs they work. Stop/Remove apply to them too (scope
   `network`; Remove does not change the person's donation, it excludes that model from the
   crew). When the last lent module is stopped or removed, the node is **suspended**: it asks for
   no jobs and tells the registry nothing new (its held job leases are released as on a worker's
   stop), recorded in `network_log`. `Network.enabled` (the person's consent) is not touched. The
   node continues by itself as soon as a lent module is active again (continued, or a new provider
   lent).
6. **Tasks** get the same two small buttons next to ✎/⌖ in their title row: Stop (= pause, no
   confirmation; "Continue" resumes) and Remove (confirmation; cancels the task and hides it).

## Consequences

- One more migration and table; `router.lineup` takes the holds of its scope (a read of the
  database: callers pass a connection or the holds already read).
- The CLI gets the same bodies (`desman modules stop|continue|remove|restore`), in
  `commands.CATALOG`; the manual's modules topic describes them.
