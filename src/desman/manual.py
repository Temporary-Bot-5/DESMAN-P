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
        "tasks",
        "how work is planned, run and checked; the states of tasks and subtasks",
        "Work the person asks for becomes a task. A planner model splits its goal into at "
        "most 8 subtasks (by default; the tuning may change it) that may depend on each other, "
        "each tagged text, code or analysis, which picks the executor's model; if planning "
        "fails, a simple template plan is used. For each subtask an executor model does the "
        "work, then a reviewer model accepts it or rejects it with feedback. Everything is "
        "kept in a database, so work survives a crash or a closed terminal: a stopped task "
        "continues from its last checkpoint with `desman resume TASK`. In the chat a task runs while "
        "the chat waits; Ctrl+C stops it. `desman worker` works on every unfinished task.\n"
        "Subtask states: pending (waits for the subtasks it depends on), ready, running, "
        "review (waits for the reviewer), done, paused (a provider's limit was hit; it goes "
        "on by itself when the limit resets), needs_human (waits for the person's decision), "
        "failed (the person gave up on it), blocked (something it depends on failed or was "
        "cancelled), cancelled.\n"
        "A task is active while the system can still move on its own, needs_human when "
        "nothing moves until the person decides, done when every subtask is done, failed or "
        'cancelled otherwise. `desman list` lists tasks, one line each (that answers "what are my tasks"); `desman list --subtasks` adds their subtasks. `desman status TASK` shows one with its '
        "subtasks, `desman audit TASK` its full history, `desman stats` how well tasks go.",
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
        "DESMAN works through language models of providers whose API keys the person added. "
        "By default only free tiers are used: free quotas are small and change, so every "
        "provider is treated as unreliable. Paid providers (OpenAI, Anthropic, DeepSeek, xAI) "
        "are never chosen by themselves, only when the person names one for a run.\n"
        "Each role (planner, executor, reviewer, interlocutor) has a lineup of models, best "
        "for it first. A call that hits a limit, an outage, a refused key or too small a "
        "context moves to the next model; a provider with several keys uses all of them. "
        "Calls are paced to each model's free rate, and limits a provider reports are kept "
        "in the database, so they hold across restarts and parallel workers. When everything "
        "is limited, work pauses until a quota resets, without spending attempts. The person "
        "may choose each role’s model and key with `desman tuning` or `desman config`. "
        "The chosen key is tried first for its provider, then the other keys and models follow.\n"
        "A local model (Ollama or llama.cpp on the person's machine) costs nothing and has "
        "no quota but is slow, so it comes after the cloud models. `desman models show MODEL` says "
        "whether a Hugging Face model fits the machine's memory, `desman models add MODEL` has the "
        "server download one; agents may only suggest a model.\n"
        "Keys: added by the person into a hidden field (`desman keys add PROVIDER`), never written in "
        "the chat; never shown, sent to a model or logged. `desman keys list`, `desman keys "
        "check` (does a key work, what limits; spends nothing) and `desman usage` (calls and "
        "tokens per key) show them.",
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
        "Bare `desman` opens the chat in a new session; `desman -c` continues the last one and "
        "`/sessions` any earlier one. "
        "`desman gui` opens this same chat in a local browser page, including setup and confirmations. "
        "`desman -t` opens a temporary chat: nothing said there is "
        "kept, it is erased when the chat ends (its tasks stay, like any task), and it keeps no "
        "profile notes and no summaries. Each message gets one "
        "turn: an answer, a question back, a task, or a DESMAN command. "
        "A command that shows something runs at once; one that changes something is shown "
        "in a window and done only after the person's yes; a key or a section of the wizard "
        "is typed by the person into a window. Decisions enter the operator's journal.\n"
        "The chat's own commands: `/help`, `/new` (a new session), `/sessions` (pick an "
        "earlier one), `/rename`, `/compact`, `/profile`, `/delete` (out of the list; its "
        "messages stay in the database), `/pin` (toggle pin), `/delete-all` (confirm deleting unpinned sessions; messages stay), `/temporary` (erase on exit), `/exit`. `/new` always starts a normal session, erasing a temporary one. Pinned sessions appear first. Typing / lists chat commands. !COMMAND runs a DESMAN CLI command without a model; CLI confirmations and previews still apply. Input opens a window. !help lists them.\n"
        "The interlocutor sees only the newest messages that fit its budget; older ones are "
        "summarised by a small task of their own (`/compact` does it now) and it sees the "
        "summary. A summary that fails leaves the older messages out of its view. A pasted "
        "API key is cut from a message before anything keeps it.",
    ),
    Topic(
        "gui",
        "browser page, chat and task panels, buttons and keyboard completion",
        "`desman gui` opens the same chat in a local browser, with setup and confirmation windows. "
        "The page lists chats on the left (new, temporary, pin, delete, settings and messenger) "
        "and tasks on the right (subtasks, modules, and answers to waiting work). Panels can "
        "be read while work runs; their actions are available only at the chat input. "
        "The transcript scrolls above a growing input at the bottom; suggestions open above it. "
        "Pinned chats have their own section; dates follow the interface language. "
        "Opening a chat shows its latest 200 messages. "
        "Deleting a chat or all unpinned chats requires confirmation. Decisions for waiting "
        "subtasks use the displayed version: if work changed, refresh and decide again. "
        "Fail requires an answer. During a task, panels only read; Stop interrupts the task. "
        "On narrow screens, top buttons open the panels. Command suggestions accept arrow "
        "keys, Tab, Enter, Escape and clicks. The first sent line hides the greeting. "
        "Reloading replays the current transcript; changing chats replaces it.",
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
