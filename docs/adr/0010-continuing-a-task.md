# ADR 0010: Continuing a task — more subtasks, or a step done again

Status: accepted (owner's request and answer, 2026-11-09)

## Context

A task ends when its last subtask is accepted. If the result falls short ("add a contacts
page", "step 2 used the wrong data"), the only way today is a new task, which starts from
nothing: it does not see the old task's results unless files are copied (`files_from`), and the
person has to restate the goal. The owner wants the agent to be able to finish a task by adding
subtasks to it or by redoing them, both when the person asks and on DESMAN's own initiative,
always with the person's confirmation (answer of 2026-11-09: "both").

What must hold:
- **History is never rewritten.** A done subtask stays done with its accepted result; the
  subtask state machine (`core/states.py`, protected) has no edge out of DONE, and none is added.
- **Human control.** Nothing is added to a task on a model's word: the person confirms each
  continuation in a window rendered by code, whether they asked for it or DESMAN proposed it.
- **One way of working.** New subtasks are ordinary subtasks: claimed, executed, reviewed,
  retried and recovered exactly like the first ones.

## Decision

1. **A continuation appends subtasks to the task.** `Store.extend(task_id, parts, basis)`
   inserts the planned parts after the task's last subtask (higher `seq`), atomically, with the
   same graph checks as a task. The parts that depend on nothing among themselves depend on the
   `basis`: the task's current last subtask (its final result) and, for a redo, the subtask being
   redone. So the new work sees what was done through the usual dependency context. The task's
   final result becomes the new last subtask's (`final_result` already takes the last one); the
   planner is told that this last part must produce the complete result, the earlier one
   included. The task becomes active again (its status is derived from its subtasks).
2. **Redo = a continuation, not a rewind.** "Redo step N because …" plans new parts with that
   reason as the goal and step N in the basis. Step N keeps its result in the history; the new
   final result replaces it as the task's answer.
3. **Planning** reuses the planner with a continuation prompt: the task's goal, its subtasks
   (descriptions and statuses, results only through the basis), and what to add or redo. It gets
   the same limits as a plan (≤ 8 parts, tags, template on failure).
4. **The person asks:** new catalog commands `task extend TASK GOAL` and `task redo SUBTASK
   NOTE` (CLI, chat, page), class *change*: the window shows the task and what will be added
   before anything is planned. In the chat, "add a contacts page to that site" or "redo step 2,
   the data is wrong" becomes such a command turn of the interlocutor, confirmed in the window.
5. **DESMAN proposes:** when it retells a finished task, the interlocutor may add one sentence
   of what it would add or redo (`Retelling.continue_with`) when the result clearly falls short of
   the goal. The chat then shows a window "Continue task REF: …?". Only the person's yes starts
   the continuation. A declined proposal is journaled (`operator_log`) and not proposed again for
   that task.
6. **Not for desman-global's jobs** (origin `network`): a lent job is never extended locally.
   Cancelled tasks are not continued.

## Consequences

- No protected file changes; no new subtask status. One store entry point (`extend`), one
  planner function, two catalog commands, one optional field of the retelling.
- Audit: `task_extended` (task-level note with the new subtask ids, the basis, who asked:
  person or proposal accepted) through `Store.note()`; each new subtask's creation as for a
  task.
- The page's Tasks panel shows the new subtasks under the same task; its module cards follow.

## Split

- **T.2q-a** `Store.extend`, the continuation planner, `task extend` / `task redo` (CLI,
  catalog, chat turn with the window), manual, 12 catalogs.
- **T.2q-b** DESMAN's own proposal at the retelling (`continue_with`), its window and journal.
