"""The interlocutor's manual of DESMAN (ROADMAP C.10): what it may tell the person about how
DESMAN works, so that it never has to make it up. G.7b describes independent fake proposal checks.

Short texts for the model, in English like every prompt, one per topic. The interlocutor's
prompt lists only the topics, a line each (small models have little context); it asks for one
or two with a `lookup` turn, and gets their texts with the same message again
(`desman.interlocutor`).

The manual says only what DESMAN does now: never plans (those are in ROADMAP.md, written for
developers). A change of code that changes what a topic says changes the topic in the same
commit. A test checks it against the code: every command it names (written `desman ...` in
backticks, with its options and upper-case placeholders for arguments) is in the CLI and the
catalog, every chat command (`/...`) is the chat's, and each topic fits
`TOPIC_CHARS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

TOPIC_CHARS: Final = 1600
"""The longest a topic's text may be: two of them have to fit a small model's context."""

MAX_LOOKUP: Final = 2
"""The most topics one lookup may ask for."""


@dataclass(frozen=True)
class Topic:
    name: str
    about: str
    """What it is about, in the line the interlocutor's prompt lists it with."""
    text: str


TOPICS: Final = (
    Topic(
        "page",
        "the browser page, panels, settings, problem reports and desktop shortcuts",
        "`desman gui` opens the local browser page with the same chat and confirmation windows. Tasks run in background workers, leaving the chat free after planning. Results arrive in their own chats; other chats show unread marks. Settings → Background workers (1–8, default 3) applies on the next page start. Pause tasks on their cards. The terminal chat waits as before. The left sidebar lists chats, pinned first; Delete all chats sits above Report a problem. The right column switches between Tasks and Modules. Its footer stays visible: Delete all tasks hides finished unpinned tasks, preserving their history; Stop all modules pauses unfinished unpinned tasks. Both show a confirmation with the affected tasks. Pin tasks to protect them from bulk actions. Paused tasks have Continue. Module cards show state, usage, the chosen model and the last answering model when different. Their source/model editor is shared with Settings → Roles. Settings saves changes with receipts. General Terminal shows command output. Report a problem collects a description and shows the exact payload for confirmation. `desman gui --shortcut` creates launch shortcuts on Windows and Linux without starting the page.",
    ),
    Topic(
        "tasks",
        "how work is planned, run and checked; the states of tasks and subtasks",
        "Work becomes a durable task: a planner makes up to 8 subtasks by default, tagged text/code/analysis; a template is used if planning fails. Executors work, reviewers accept or return feedback. Dependencies form a DAG. Stopped work resumes from checkpoints with `desman resume TASK`; completed work is never repeated. `desman worker` works across unfinished tasks. The page starts its own worker pool: task requests return after planning, queued cards wait for workers, and results are retold once in their own chats. States: pending (dependencies), ready, running, review, done, paused (provider limit), needs_human, failed, blocked, cancelled. Task status is derived from subtasks. `desman list` shows visible tasks; `desman list --subtasks` includes subtasks; `desman list --all` includes hidden tasks. `desman status TASK`, `desman audit TASK` and `desman stats` retain access to history. `desman task pin TASK` and `desman task unpin TASK` control a task's pin and its modules' pin. `desman task hide TASK` hides finished work only; `desman task hide --finished` hides finished unpinned tasks. Nothing is erased. `desman task pause TASK` pauses unfinished work; `desman task pause --all` pauses unfinished unpinned tasks. Paused tasks are not scheduled; running executors yield at their next step, keeping checkpoints and replaying recorded effects on resume. `desman resume TASK` clears the operator pause and continues. List marks pinned/paused tasks. Rows show short titles; details and audit keep full descriptions.",
    ),
    Topic(
        "files",
        "where a task's files are, giving them to a later task, running code",
        "Each task has a folder of its own (workspaces/TASK next to the database), where the "
        "executor reads and writes files with its tools; nothing outside it. After a task, the "
        "chat lists its files and their folder itself. A task sees only its own folder: when "
        'a request builds on what an earlier task of the same chat wrote ("test it", "change '
        'that file"), the new task gets copies of those files, in from-TASK/ in its folder; '
        "the original files stay as they were. Saving something is work too: a task writes it.\n"
        "Running code (a test, a crash test, checking what a program prints) needs the "
        "run_command tool, which needs the person's shell_execution switch (`desman "
        "permissions`) and a working sandbox. A subtask that must run code is marked so by the "
        "planner (`desman run --runs-code` holds a task to it): without the tool it waits for the person at once, with the reason, and "
        "with it its result counts only if the code really ran. No part of DESMAN reports "
        "what code would print without running it.\n"
        "The interlocutor itself does nothing: only DESMAN's own messages tell what was done. "
        "A retelling shows code only as the task left it.",
    ),
    Topic(
        "retries",
        "what happens after a failed attempt; what waits for the person's decision",
        "A rejected or unusable attempt is tried again, each time differently: with the "
        "reviewer's feedback, then with the instructions reworded, then with another model, "
        "then the subtask is split into 2-4 smaller parts whose results are combined. By "
        "default a subtask gets 5 attempts and 5 interruptions (network errors, crashes); "
        "the tuning may change both. A provider's limit costs no attempt: the subtask is "
        "paused until the limit resets. A failure retrying cannot help (a refused key, a "
        "denied permission) goes to the person at once.\n"
        "DESMAN never gives up by itself: when the attempts run out the subtask waits for the "
        "person (needs_human). `desman inbox` lists what waits, with the reason, and in a terminal or the chat lets the person pick a subtask and decide; `desman inbox --list` only lists. The person "
        "decides: `desman retry SUBTASK` sends it back to work with fresh attempts (a note becomes a "
        "hint to the next attempt), `desman accept SUBTASK` takes it as it is, `desman fail SUBTASK --note TEXT` gives "
        "up on it (what depends on it is blocked), `desman cancel TASK` stops the whole task. A "
        "failed subtask can still be sent back with retry. The chat tells when something "
        "comes to wait; asked there, each decision is confirmed by the person in a window.",
    ),
    Topic(
        "providers",
        "models, API keys, free tiers, limits and local models",
        "DESMAN uses providers whose keys the person added. Only free tiers are chosen "
        "automatically; paid providers require explicit choice. Quotas change "
        "and every provider is treated as unreliable.\n"
        "Each role has a model lineup. Limits, outages, refused keys and small contexts move "
        "a call to the next model. Chat notes name the answering model and skipped reasons. Keys rotate. Limits "
        "are shared across workers and restarts; exhausted quotas pause work without spending attempts. "
        "Choose the model and key asked first per role with `desman tuning` or `desman config`. "
        "The selected key goes first.\n"
        "For one task (reviewer, decomposer, text, code, analysis) use `desman modules set ROLE --task REF --model PROVIDER/MODEL`; "
        "for one chat (interlocutor only) use `desman modules set ROLE --session ID --key PROVIDER:KEY`. "
        "Auto inherits the role again. "
        "`desman modules` shows active tasks; `desman modules clear --task REF` resets choices.\n"
        "For installed models and downloads see the localmodels topic.\n"
        "Keys enter only through a hidden field (`desman keys add PROVIDER`), never the chat, "
        "model or logs. Inspect with `desman keys list`, `desman keys check` (no tokens) "
        "and `desman usage`.",
    ),
    Topic(
        "localmodels",
        "Ollama models: downloads, removal and memory",
        "Auto uses configured local models last; chosen Ollama models go first, with fallback if removed. "
        "`desman models show MODEL` checks a Hugging Face model's size; `desman models add MODEL` "
        "downloads it after confirmation. Settings lists installed Ollama and HF GGUF models. Remove one installed model there or with `desman models remove NAME`; confirmation shows size, saved choices stay, and missing references are reported. Ollama context starts at 16384, capped by the model maximum and reduced for free memory after weights and KV cache, down to 8192. Calls still run if memory is insufficient; fallback notes warn. OLLAMA_CONTEXT overrides it. llama.cpp keeps its server context. Agents only suggest models.\n",
    ),
    Topic(
        "permissions",
        "what agents may do, their tools, the sandbox and the web",
        "Agents act only through tools, and each tool needs two things: the role may use it "
        "(a policy file that people change in the source, never an agent), and the person's "
        "switch for what it needs is on. The switches are network_access, shell_execution, "
        "fs_outside_access and admin_rights; all are off until the person turns them on "
        "(`desman permissions`, or the setup wizard). Only the person sets them; in the chat "
        "a change is confirmed in a window.\n"
        "Only the executor has tools now: reading, writing and listing files in its task's "
        "own folder (always allowed; nothing outside it, no links); run_command "
        "(shell_execution), which runs code only in a sandbox with no network, a cleared "
        "environment and limits of time and memory, and is not offered at all where the "
        "sandbox cannot run. On Windows agents cannot run code: the sandbox needs Linux. "
        "web_search and web_fetch (network_access): reading public pages "
        "only, results marked untrusted; web_search asks Wikipedia, Stack Overflow, arXiv and "
        "GitHub, and the whole web only through the person's SearXNG server or a Tavily "
        "key); suggest_model, which "
        "only records a suggestion for the person. Every tool call is recorded, and a refused "
        "one too, with the reason. The planner, the reviewer and the interlocutor have no "
        "tools. admin_rights is for test machines: while it is on, every run warns.",
    ),
    Topic(
        "protected",
        "what judges the agents, and why tasks may be blocked",
        "Some files decide what agents may do and whether their work is good: the tool "
        "policy and its code, the reviewer and the executor, the sandbox, the guard of the "
        "task folders, the rules of who may decide what, the benchmark and its comparison, "
        "the test harness. DESMAN keeps what the person approved of each of them. While one "
        "of them differs from the approved version (an edit, an update), no task runs at "
        "all, and the chat says so. The person looks at what changed (`desman protected "
        "show --diff`) and approves the files as they are (`desman protected "
        "approve`), after which tasks run again. Agents cannot change these files or approve "
        "them; in the chat the approval is confirmed by the person in a window, after seeing "
        "which files changed.",
    ),
    Topic(
        "proposals",
        "the Improvement Analyst, its proposed changes and their evaluation",
        "The Improvement Analyst studies what went wrong (failures, reviewers' feedback, "
        "plans that fell back to the template, the totals of stats) and proposes a change to "
        "DESMAN: new texts for some of its prompts (shown as a diff), or settings of the "
        "tuning (given as a `desman tuning` command). It runs as an ordinary task without "
        "tools when the person asks (`desman proposals analyze`), never on protected files, "
        "keys or permissions. Its outcome: proposed, nothing (no change worth making) or "
        "unusable. `desman proposals list` and `desman proposals show N` show proposals; a "
        "proposal never changes once made.\n"
        "`desman proposals evaluate N` runs a fixed benchmark on the current version and on it "
        "with the proposal, each in a separate copy of the source, and says which went "
        "better; by default on a scripted fake model (does it still work, what its prompts "
        "cost), with real on the person's models (spends their quota). The analyst sees "
        "how earlier proposals did.\n"
        "Nothing is applied by DESMAN itself: only the person applies a proposal, with "
        "`desman proposals apply N` (in the chat after confirming in a window). Its diff "
        "becomes a git commit of the checkout DESMAN runs from, on its branch, and its "
        "settings are changed; refused if the person has uncommitted changes in those files. "
        "Workers and chats already running keep the old prompts until they start again. "
        "`desman proposals revert N` takes it back: a commit reverting that one, and the "
        "settings as they were (unless the person changed them since).",
    ),
    Topic(
        "chat",
        "sessions, the chat's own commands, how long conversations are kept",
        "Bare `desman` opens a new chat; `desman -c` continues the last one, `/sessions` an earlier one. `desman gui` opens the same chat in a browser. `desman -t` opens a temporary chat, erased on exit with no profile or summaries; its tasks stay. Messages become replies, clarifications, tasks or catalog commands. Model-requested changes require the person's confirmation in a window; keys use hidden fields. Decisions are journaled. Explicit /COMMAND (also !COMMAND) runs the same body as desman COMMAND; chat commands take precedence. `/help` lists both sets, as does `desman help`. Chat commands: `/new`, `/sessions`, `/rename`, `/compact`, `/profile`, `/delete`, `/pin`, `/delete-all`, `/temporary`, `/messages`, `/exit`. Pins protect chats from delete-all. `/terminal` opens General Terminal; `/back` returns. Terminal input must start with / and never calls a model. `/clear` clears the current screen only, keeping history and model context; reopening a chat restores messages. `/context` shows the first model, fallbacks, total/system/message token estimates and the exact request with role headers and [your next message]; /context full is an alias. Neither calls a model. Older history is summarized to fit the budget; failed summaries leave older messages out of view. A fallback reply is followed by the actual answering model and skipped models' reasons. Pasted API keys are redacted before storage.",
    ),
    Topic(
        "gui",
        "browser page, chat and task panels, buttons and keyboard completion",
        "The page shares the terminal chat, commands and confirmation windows. Source and Model are the only two module dropdowns. Source offers Auto, each configured provider key (masked), and configured or discovered Ollama. Cloud model lists are fetched only when a source is selected or the picker opens; known catalog models are marked and come first. Failed discovery shows catalog models with a note; saved missing models remain selectable. Paid keys require explicit selection. Ollama lists installed models and supports Hugging Face GGUF search/download with confirmation. A checkbox applies the choice to every chat/task; otherwise it is local where that role allows it. New chats retain pending choices until their first message. Editors survive polling. Cards show the last successful answering model when different. Modules inherit their task's pin. Settings and module edits keep the chat open, with output in General Terminal. `/clear` clears only this chat's screen; history remains and reopening restores it. Chat and General Terminal use the same /COMMAND catalog and `/help`; !COMMAND remains an alias. Tab cycles commands from the first match, Shift+Tab goes back, Enter runs the line and Esc closes suggestions. General Terminal has no model. `/context` there explains this.",
    ),
    Topic(
        "profile",
        "how the interlocutor talks, and the notes it keeps about the person",
        "How the interlocutor talks is the person's setting: length of answers (short, "
        "detailed), tone (formal, informal), style (simple, technical) and their own words; "
        "`desman behaviour` shows and changes it (in the chat, after the person's yes). It "
        "holds in every session.\n"
        "Personalisation is off unless the person agrees (`desman personalize`, or the "
        "wizard). With it on, the interlocutor keeps short notes about the person between "
        "conversations (at most 20, one line each): how they like to be answered, their "
        "interests, requests they make often. Every change is shown when made, the person "
        "sees, changes and erases them in `/profile`, and turning it off hides them. Notes "
        "never hold secrets, never allow anything and never confirm a change. The journal "
        "records that a note changed, never its text.",
    ),
    Topic(
        "language",
        "the chat language and confirmed language changes",
        "Configure the language with `desman language CODE`. Replies follow the person’s language; when it differs from configuration, a language command proposes switching, with an explanation in the person’s language before the confirmation window. Declining stops further switch proposals in that session. Settings keep their configured language until confirmation.",
    ),
    Topic(
        "data",
        "where DESMAN keeps its configuration, database and task files",
        "The configuration (language, API keys, permissions, behaviour, personalisation, "
        "tuning) is one file, config.json, readable only by its owner: in the project folder, "
        "or in ~/.config/desman/ for an installed package. `desman config` is the setup "
        "wizard, `desman config show` shows it without keys, `desman config reset` deletes it "
        "after the person's yes.\n"
        "Tasks, their history, the chat's sessions, the profile, model calls and limits are "
        "in a SQLite database, ~/.local/share/desman/desman.db; each task's files are in a "
        "folder of its own in workspaces/ next to it. `desman init-db` creates the database "
        "or brings it up to date. Everything stays on the person's machine; only the "
        "requests to model providers, to Hugging Face and, when allowed, to the web leave it.",
    ),
    Topic(
        "global",
        "desman-global: the opt-in network of DESMANs, joining, leaving, lending providers",
        "desman-global is off until `desman global join` (use --registry URL if unknown). "
        "Joining makes node.key, shows only its fingerprint, and lends nothing. "
        "`desman global donate PROVIDER --calls N --tokens N` lends a local model or an "
        "allowed free provider, never paid; keys stay here. `desman global quota PROVIDER` "
        "shows/changes quotas; `desman global undonate PROVIDER` takes it back. "
        "`desman global leave` withdraws everything and switches off; `desman global status` "
        "shows state.\n"
        "While `desman worker` runs, it registers and fetches revocations. Jobs need a fresh "
        "root-signed trust list. An idle node works on one operator-signed job with lent "
        "models (own folder, sandbox, no web), reviews a result or checks a proposal with an "
        "isolated fake benchmark. Own tasks go first. Unreachable: retry later; revoked: stop.\n"
        "`desman global submit-proposal N` previews a text proposal and benchmarks (no settings "
        "or private evidence; keys block sending; 5/day); --confirm HASH sends that payload "
        "to that registry. The chat asks in a window.\n"
        "For bug reports and optional shared memory, see topic globalsend. "
        "Root-signed operator releases arrive with the worker: `desman global update --show N` "
        "inspects (omit N for newest); `desman global update --apply N` installs only by the "
        "person, never automatically; `desman global update --revert` undoes the newest installed release, one at a time, if HEAD/schema "
        "have not advanced. Restart workers/chats; approve protected changes and reinstall "
        "changed dependencies. Chat changes require the confirmation window.",
    ),
    Topic(
        "globalsend",
        "Sending bug reports and shared memory to the registry",
        "`desman global report TEXT` previews your problem description and system/version; "
        "--task ID adds bounded task status, descriptions and failure causes, never workspace files. "
        "`desman global share` previews aggregate stats; --failures includes failures and plan "
        "fallbacks quoting your tasks and reviewers. --no-stats omits stats; choose at least one. "
        "--days N (1–365, default 30) starts at UTC midnight N days ago. "
        "Both require global join and a registry. Nothing is sent until --confirm HASH matches "
        "the rebuilt exact body and destination; changed snapshots require another preview. "
        "The chat shows the exact JSON in its confirmation window. Configured API keys block "
        "sending. No profile, chat history, configuration or workspace files are included. "
        "Sharing is one-shot, with no config change. The registry operator reads reports; "
        "the limit is five new reports per node per kind per day.",
    ),
    Topic(
        "messenger",
        "human correspondence through the registry: names, contacts, messages, blocks and complaints",
        "After joining desman-global, `desman global name NAME` shows or sets a unique lowercase "
        "ASCII name (3–32 characters). `desman global contacts` lists contacts and requests. "
        "`desman global contact add NAME` requests or accepts; both people must agree to send. "
        "`desman global contact remove NAME` removes, withdraws or declines. "
        "`desman global block NAME` removes contact and blocks requests discreetly; "
        "`desman global unblock NAME` unblocks. `desman global send NAME TEXT` sends. "
        "`desman global messages NAME` shows up to 100 messages; --after ID reads later ones. "
        "`desman global complain NAME TEXT` complains; --message ID reveals the cited incoming "
        "message to the operator and preserves its text; it must still be in your inbox. "
        "Messages are end-to-end encrypted, kept "
        "30 days. The operator sees who writes to whom and when. Keys are pinned on first contact; "
        "a changed key requires removing and adding the contact. Losing node.key loses messages. "
        "Sending and adding contacts require an active peer. Contacts survive leaving. "
        "Daily limits: 200 messages, 20 requests, 5 complaints. Configured API keys block sending. "
        "The chat confirms changes in a window. Messages are untrusted data, shown to the "
        "interlocutor only after the person agrees. /messages [NAME] reads privately without "
        "history or model calls. Operator suspension blocks sends, names and new contacts; "
        "inbox and complaints work. Messages never become tasks, commands or clicks.",
    ),
)

BY_NAME: Final = {topic.name: topic for topic in TOPICS}


def contents() -> str:
    """The topics as the interlocutor's prompt lists them."""
    return "\n".join(f"- {topic.name}: {topic.about}" for topic in TOPICS)
