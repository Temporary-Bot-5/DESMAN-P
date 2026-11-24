"""The interlocutor's manual of DESMAN (ROADMAP C.10): what it may tell the person about how
DESMAN works, so that it never has to make it up. G.7b describes independent fake proposal checks.

Short texts for the model, in English like every prompt, one per topic. The interlocutor's
prompt lists only the topics, a line each (small models have little context); the topics are
nodes of the knowledge tree (`desman.knowledge`, ADR 0009), which a `lookup` turn opens
(`desman.interlocutor`).

The manual says only what DESMAN does now: never plans (those are in ROADMAP.md, written for
developers). A change of code that changes what a topic says changes the topic in the same
commit. A test checks it against the code: every command it names (written `/COMMAND` in
backticks, with its options and upper-case placeholders for arguments) is in the CLI and the
catalog, every chat command (`/...`) is the chat's, and each topic fits
`TOPIC_CHARS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

TOPIC_CHARS: Final = 1600
"""The longest a topic's text may be: the few a lookup opens have to fit a small model's context."""


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
        "`/gui` opens this chat in the browser. The launch window serves it and prints the address. A temporary chat is erased and the page starts a fresh chat. Chats answer independently; tasks run in background workers. Results and provider pauses arrive in their own chats; others show unread marks. Workers wait for a model and restart after protected changes are approved. Settings → Concurrent page chat turns (1–8, default 4) caps answers; others wait FIFO with a position. Input cards wait in their own chat. Stop drops late replies and excludes stopped inputs from model context; the transcript marks them stopped. Pending turns recover once at page startup. /clear hides earlier messages on screen; the model and /context still see them. Page catalog commands open General Terminal, which saves their input and output. Sidebar marks answering, queued, waits for you and unread. Background workers (1–8, default 3) applies on next page start. Terminal chat waits as before. Chats are listed pinned first; Delete all chats is above Report a problem. The right column has Tasks and Modules. Its footer confirms bulk actions: Delete all tasks hides finished unpinned tasks, keeping history; Stop all modules pauses unfinished unpinned tasks. Pin tasks to protect them. Paused tasks have Continue. Module cards show roles, model, source, usage, chats and active tasks; their editor is shared with Settings → Roles. Report a problem shows the payload for confirmation. `/gui shortcut` creates Windows/Linux shortcuts to start DESMAN. Use `/interface MODE` with gui to make the page the default.",
    ),
    Topic(
        "tasks",
        "how work is planned, run and checked; the states of tasks and subtasks",
        "Planner: 1 subtask for simple work, up to 8 for tools/dependencies; template fallback. Simple code changes and tests share one subtask. Executors work; reviewers check. Dependencies form a DAG. Stopped work resumes from checkpoints with `/resume TASK`; done work is kept. `/worker` works across unfinished tasks. The page queues tasks for its worker pool and retells results once in their chats. States: pending (dependencies), ready, running, review, done, paused (provider limit), needs_human, failed, blocked, cancelled. `/run GOAL` and `/status TASK` show full text answers (data inside) and file folders; status --json includes the full result, files and folder. `/list` shows visible tasks (with --subtasks, --all: hidden too). `/status TASK`, `/audit TASK` and `/stats` retain access to history. `/task pin TASK` and `/task unpin TASK` control a task's pin and its modules' pin. `/task hide TASK` hides finished work only; `/task hide --finished` hides finished unpinned ones; nothing is erased. `/task pause TASK` pauses unfinished work; `/task pause --all` pauses unfinished unpinned tasks. Paused tasks are not scheduled; running executors yield at their next step, keeping checkpoints and replaying recorded effects on resume. `/resume TASK` clears the operator pause and continues. `/task extend TASK GOAL` and `/task redo SUBTASK NOTE` add subtasks starting from its result; done work stays. Retellings may propose these (window first). Subtasks get ≤3 knowledge nodes (see status).",
    ),
    Topic(
        "master",
        "a master module: the whole goal given to one strong model instead of split work",
        "`/run --master GOAL` gives the whole goal to one qualified model, with up to "
        "24 tool rounds per attempt by default (8–48 with `/tuning --master-rounds N`). In chat, ask for one agent or type /master on (see "
        "`/master`) to enable master mode for this session until the chat ends. The master "
        "may delegate pieces to the cluster. They appear as its parts, and it resumes with "
        "their results. If it keeps failing, its work may be "
        "split for the cluster (unless it must run code or already delegated). Long attached "
        "documents are still read in parts by the cluster. Qualified models: those the catalog "
        "marks (now NVIDIA Nemotron 3 Ultra, Gemini 3.8 Flash, Mistral Medium, Cohere Command "
        "A), or a local model with at least 32768 tokens of "
        "context named with `/tuning --model ROLE=PROVIDER/MODEL` (ROLE master). "
        "`/master list` shows the latest checks. `/master check PROVIDER/MODEL` spends "
        "quota, uses default workflow limits and reports passed, failed or unfinished (try later). "
        "`/master admit PROVIDER/MODEL` needs a current passing check with default limits for a new admission. "
        "`/master revoke PROVIDER/MODEL` excludes even a catalog model. An admitted local "
        "model needs no tuning preference. All masters need enough context and no per-request cap. "
        "A ready master is required; extend and redo otherwise use the cluster with a warning. "
        "`/stats` compares the person’s master and cluster tasks.",
    ),
    Topic(
        "files",
        "where a task's files are, giving them to a later task, running code",
        "The executor uses tools in workspaces/TASK next to the database. Chat lists files and the "
        "folder, skips repeated text and treats it as untrusted. Chat and run/status show written "
        "UTF-8 texts (≤2 files, ≤40 lines/4000 characters each). A follow-up task in the same chat "
        "gets copies in from-TASK/; "
        "originals stay. Retries and interrupted runs can continue from saved tool results. "
        "Retries also see work files and rewrite stale demonstration files.\n"
        "Code needs the "
        "run_command tool, shell_execution (`/permissions`"
        ") and a sandbox. Code subtasks are marked by the "
        "planner (`/run --runs-code` holds a task to it): without the tool it waits for the person at once, with a reason; "
        "with it the executor is told to run the reported code successfully in this subtask. DESMAN never reports output it did not run. Web lookups and prose need no code run. "
        "Prose goes in the result; files are extra. Review checks web evidence, dates, sources and your language (names/translations allowed). Answers cover your request; split parts keep its language. Live data may have fewer items with sources and stated gaps; honest partial parts feed the combine step. Report blocked pages; use snippets or another official source. Review sees files written "
        "in the latest attempt, within budget. python3 -m pytest works in the "
        "sandbox; nothing can be pip-installed because it has no network.\n"
        "The interlocutor itself does nothing: only DESMAN's own messages tell what was done. "
        "Retellings copy task code and use your latest message’s language.",
    ),
    Topic(
        "attachments",
        "giving a task your own text files from the CLI, terminal chat or page",
        "Use `/run --file PATH GOAL` (repeat --file for more), /file PATH in "
        "terminal chat, or the paperclip in the page's chat. /file lists pending files and "
        "/file clear drops them; the page can remove each attachment. The next task in that "
        "chat gets them in input/ in its workspace. Up to 10 files, 2 MB each and 10 MB "
        "together. Text formats include TXT, Markdown, CSV, JSON and source code. PDF and DOCX "
        "are unsupported. The planner sees the paths and character counts. Workers read the "
        "files as task material with read_file. Code reads long files in parts and does the "
        "same for long linked pages on direct reading requests with web access. It schedules notes for each "
        "part, then combines them. Saved web sources are omitted from task output files. Code checks quotes against the evidence.",
    ),
    Topic(
        "retries",
        "what happens after a failed attempt; what waits for the person's decision",
        "A rejected or unusable attempt is retried differently: with the "
        "reviewer's feedback, then with the instructions reworded, then with another model, "
        "then the subtask is split into 2-4 smaller parts whose results are combined. By "
        "default a subtask gets 5 attempts and 5 interruptions (network errors, crashes); "
        "tuning may change both. A provider's limit costs no attempt: the subtask is "
        "paused until the limit resets. Run/resume, chat and planning wait ≤15 min when every model is limited; "
        "after longer task pauses, use `/resume TASK` or `/worker` when the limit resets. A failure retrying cannot help (a refused key, a "
        "denied permission) goes to the person at once.\n"
        "DESMAN never gives up by itself: when the attempts run out the subtask waits for the "
        "person (needs_human). `/inbox` lists what waits, with the reason, and lets the person pick a subtask and decide; `/inbox --list` only lists. The person "
        "decides: `/retry SUBTASK` sends it back to work with fresh attempts (a note becomes a "
        "hint to the next attempt), `/accept SUBTASK` takes it as it is, `/fail SUBTASK` (optional --note) gives "
        "up on it (what depends on it is blocked), `/cancel TASK` stops the whole task. A "
        "failed subtask can still be sent back with retry. A step of a chat's task that comes "
        "to wait is told in that chat at once, even while the rest runs; the page shows a field "
        "under that chat (answer, Retry, Accept, Fail); an answer in words becomes retry with it "
        "as the note. Confirm decisions; retry/accept resumes with models, else shows a resume hint.",
    ),
    Topic(
        "providers",
        "models, API keys, free tiers, limits and local models",
        "Longer requests give models more time to answer. DESMAN uses configured providers. "
        "Auto chooses free tiers. To use a paid provider, choose its model and key for a role under Models and keys for roles in `/config`. Quotas change "
        "and providers may fail.\n"
        "Auto spreads work over fitting free models. Repeated transient failures move models behind healthy ones for 15 minutes, until a success. "
        "Choosing a model for a role pins it first. Limits, outages, refused keys and small contexts move "
        "a call to the next model. Chat notes name the answering model and skipped reasons. Keys rotate. Limits "
        "are shared across workers and restarts; run, chat and planning wait up to 15 minutes for limits (see retries), without spending attempts. "
        "Choose the model and key asked first per role with `/tuning` or `/config`. "
        "The selected key goes first.\n"
        "For one task (reviewer, decomposer, text, code, analysis) use `/modules set ROLE --task REF --model PROVIDER/MODEL`; "
        "for one chat (interlocutor only) use `/modules set ROLE --session ID --key PROVIDER:KEY`. "
        "Auto inherits the role again. "
        "`/modules` shows active tasks; `/modules clear --task REF` resets choices.\n"
        "See localmodels for downloads.\n"
        "Keys enter only through a hidden field (`/keys add PROVIDER`), never the chat, "
        "model or logs. DESMAN never shows a key in full, only masked. The full key is in config.json in the project folder (or on the provider’s site). Inspect with `/keys list`, `/keys check` (no tokens) "
        "and `/usage` (ends with totals). Account/model access refusals last 24h; usage shows reasons, keys check clears them. Token windows are shared.",
    ),
    Topic(
        "modules",
        "module cards, stopping, removal and restoration",
        "Modules lists source/model pairs for local tasks and recent chats, including interlocutors, "
        "and a separate DESMAN-Global section with lent modules; shared modules have a card in each. Each card shows short work titles. "
        "Stop pauses unfinished tasks served by that module; chat turns queue. Continue resumes only "
        "tasks paused by that stop once all overlapping stops are lifted. Removed modules are skipped, including explicit preferences; "
        "the next model answers, or work waits without spending attempts when no module remains. "
        "Explicit model choices in Change or Settings → Roles restore replaced sources in scope; paused sources stay stopped. "
        "Use `/modules stop SOURCE MODEL`, `/modules continue SOURCE MODEL`, "
        "`/modules remove SOURCE MODEL`, `/modules restore SOURCE MODEL`: default all "
        "work, optionally --task REF, --chat ID or --network. SOURCE is a key fingerprint, ollama "
        "or llamacpp; MODEL is provider/model. CLI is immediate; chat proposals need confirmation. "
        "Page buttons of cloud modules act at once; a local module's Pause, Resume, Replace and a "
        "Change to or from a local model ask first (size vs free memory; text may leave the "
        "computer); replacing the last lent module asks too. A task's ✕ hides it at once, "
        "retaining history. Removing a network module preserves donation and consent; "
        "no active lent module suspends registry contacts until one becomes available again. "
        "The JSON shape of `/modules` remains the task/role list; the page aggregates modules.",
    ),
    Topic(
        "localmodels",
        "Ollama models: downloads, removal and memory",
        "Auto uses configured local models last; chosen Ollama models go first, with fallback if removed. "
        "`/models show MODEL` checks a Hugging Face model's size; `/models add MODEL` "
        "downloads it after confirmation. Settings lists installed Ollama and HF GGUF models. Remove one installed model there or with `/models remove NAME`; confirmation shows size, saved choices stay, and missing references are reported. Ollama context starts at 16384, capped by the model maximum and reduced for free memory after weights and KV cache, down to 8192. Calls still run if memory is insufficient; fallback notes warn. OLLAMA_CONTEXT overrides it. llama.cpp keeps its server context. Agents only suggest models.\n",
    ),
    Topic(
        "permissions",
        "what agents may do, their tools, the sandbox and the web",
        "Tools require both role policy and user permission. "
        "network_access, shell_execution, fs_outside_access and admin_rights default to off. "
        "Use `/permissions` or the wizard. Chat changes require confirmation.\n"
        "The executor can read, write and list task files without following links. "
        "run_command needs shell_execution and an available "
        "sandbox with no network, a cleared environment and time and memory limits. Linux uses "
        "bubblewrap. Windows and macOS prefer Docker Desktop in Linux-container mode. "
        "Windows tries WSL2 next, then WebAssembly. WASI supports Python, pytest and unittest "
        "without packages requiring C extensions. "
        "`/sandbox setup --yes` installs the selected backend's files. "
        "On Windows, `/sandbox setup --wsl --yes` installs the WSL sandbox. "
        "`/sandbox setup --wasi --yes` installs the WebAssembly sandbox. "
        "`/sandbox remove --wsl --yes` deletes its sandbox distribution. "
        "web_search and web_fetch (network_access) read public pages and mark results untrusted. "
        "Search covers Wikipedia, Stack Overflow, arXiv and GitHub. Whole-web search needs local "
        "SearXNG (`/search setup`), a Tavily key in `/config`, or borrowing via "
        "`/global search STATE`. If missing search setup blocks work, the subtask waits "
        "without spending a retry. Set it up and retry. "
        "suggest_model records a suggestion for the person. read_node reads the knowledge tree "
        "and is always allowed. Every tool call and refusal is recorded, including the reason "
        "for refusal. The planner, reviewer and interlocutor have no tools. admin_rights is for "
        "test machines and warns on every run.",
    ),
    Topic(
        "search",
        "set up and manage whole-web search",
        "`/search setup` previews local SearXNG. Start Docker if it is installed. "
        "Without Docker, or with --native (also an option in the wizard), setup uses a Python virtual environment. "
        "DESMAN starts the Python service when needed. "
        "SearXNG binds to 127.0.0.1 on a free port from 8888; --port chooses one and --yes confirms setup. "
        "The URL is saved after a JSON search answers. `/search status` shows the installation "
        "type, commit, URL and health (--json for scripts). `/search start` and `/search stop` "
        "control local services. `/search remove` previews removing both local installs; --yes confirms it. "
        "Docker settings are kept. "
        "`/keys list` shows its URL and `/keys check PROVIDER` with searxng checks searches. "
        "Tavily with a key is the alternative; an existing SearXNG URL can be set in the wizard.",
    ),
    Topic(
        "protected",
        "what judges the agents, and why tasks may be blocked",
        "Some files decide what agents may do and whether their work is good: the tool "
        "policy and its code, the reviewer and the executor, the sandbox, the guard of the "
        "task folders, the rules of who may decide what, the benchmark and its comparison, "
        "the test harness. DESMAN keeps what the person approved of each of them. While one "
        "of them differs from the approved version (an edit, an update), no task runs at "
        "all, and the chat says so. The person looks at what changed (`/protected "
        "show --diff`) and approves the files as they are (`/protected "
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
        "tuning (given as a `/tuning` command). It runs as an ordinary task without "
        "tools when the person asks (`/proposals analyze`), never on protected files, "
        "keys or permissions. Its outcome: proposed, nothing (no change worth making) or "
        "unusable. `/proposals list` and `/proposals show N` show proposals; a "
        "proposal never changes once made.\n"
        "`/proposals evaluate N` runs a fixed benchmark on the current version and on it "
        "with the proposal, each in a separate copy of the source, and says which went "
        "better; by default on a scripted fake model (does it still work, what its prompts "
        "cost), with real on the person's models (spends their quota). The analyst sees "
        "how earlier proposals did.\n"
        "Nothing is applied by DESMAN itself: only the person applies a proposal, with "
        "`/proposals apply N` (in the chat after confirming in a window). Its diff "
        "becomes a git commit of the checkout DESMAN runs from, on its branch, and its "
        "settings are changed; refused if the person has uncommitted changes in those files. "
        "Workers and chats already running keep the old prompts until they start again. "
        "`/proposals revert N` takes it back: a commit reverting that one, and the "
        "settings as they were (unless the person changed them since).",
    ),
    Topic(
        "chat",
        "sessions, the chat's own commands, how long conversations are kept",
        "First launch opens setup, then chat. Set gui with `/interface MODE` for the page. `/sessions` lists and opens earlier chats. `/temporary` opens a temporary chat, erased on exit with no profile or summaries; its tasks stay. Messages become replies, clarifications, tasks or catalog commands. Model-requested changes require the person's confirmation in a window; keys use hidden fields. Decisions are journaled. A shared read can continue the answer, up to three commands per message. Command advice without a lookup gets one correction. Explicit /COMMAND (or !COMMAND) runs in place in terminal chat and logs to General Terminal. `/help` lists both sets; chat commands take precedence. Chat commands: `/new`, `/sessions`, `/rename`, `/compact`, `/profile`, `/delete`, `/pin`, `/delete-all`, `/temporary`, `/messages`, `/exit`. Pins protect chats from delete-all. `/terminal` opens General Terminal; `/back` returns. Notices count unseen events since the chat opened and are hidden in temporary chats. General Terminal takes commands only. `/clear` clears the current screen only, keeping history and model context; reopening a chat restores messages. `/context` shows the first model, fallbacks, total/system/message token estimates and the exact request; /context full is an alias, and it names the knowledge-tree nodes (manual, commands, docs; plans marked) the last answer read. Neither calls a model. Older history is summarized; failed summaries leave it out of view. Fallback replies name the answering model and skipped models' reasons. Pasted keys are redacted.",
    ),
    Topic(
        "gui",
        "browser page, chat and task panels, buttons and keyboard completion",
        "The page shares the terminal chat, commands and confirmation windows. Source and Model are the only two module dropdowns. Source offers Auto, each configured provider key (masked), and configured or discovered Ollama. Cloud model lists load when a source is chosen or the picker opens; catalog models come first. Failed discovery shows catalog models with a note; saved missing models remain selectable. Paid keys require explicit selection. Ollama lists installed models and supports Hugging Face GGUF search by words and download with confirmation. A checkbox applies the choice to every chat/task; otherwise it is local where that role allows it. New chats retain pending choices until their first message. Cards: #ID, roles, model, key, usage, chats, live tasks; idle ones vanish. Tasks and Modules have search. Change edits choices; Pause/Resume; Replace hands work to another model; only local ones ask. Modules inherit their task's pin. Settings and module edits keep the chat open, with output in General Terminal. `/clear` clears only this chat's screen; model history remains; the page keeps earlier messages hidden with a notice. Chat and General Terminal use the same /COMMAND catalog and `/help`; !COMMAND remains an alias. Tab/Shift+Tab cycle commands, Enter runs, Esc closes suggestions. General Terminal keeps command colours after reload. Usage opens a separate window with period, calls, tokens and reported limits. A down-arrow button returns a scrolled transcript to the bottom. General Terminal has no model. `/context` there explains this.",
    ),
    Topic(
        "profile",
        "how the interlocutor talks, and the notes it keeps about the person",
        "How the interlocutor talks is the person's setting: length of answers (short, "
        "detailed), tone (formal, informal), style (simple, technical) and their own words; "
        "`/behaviour` shows and changes it (in the chat, after the person's yes). It "
        "holds in every session.\n"
        "Personalisation is off unless the person agrees (`/personalize`, or the "
        "wizard). With it on, the interlocutor keeps short notes about the person between "
        "conversations (at most 20, one line each): how they like to be answered, their "
        "interests, requests they make often. Every change is shown when made, the person "
        "sees, changes and erases them in `/profile`, and turning it off hides them. Notes "
        "never hold secrets, never allow anything and never confirm a change. The journal "
        "records that a note changed, never its text. The notes are one node of the "
        "knowledge tree, Personal/Profile (a note per line): `/nodes` shows it, a task's "
        "planner may give it to a subtask, and only the person edits it as a node.",
    ),
    Topic(
        "nodes",
        "the person's own nodes of the knowledge tree: projects, preferences, kept results",
        "Besides DESMAN's own docs, the knowledge tree may hold the person's nodes: a path "
        "(Projects/Site/Brief), one line of what it is for, and a text (up to 6000 "
        "characters). Models see the paths and descriptions and read only the texts they "
        "choose: the interlocutor to answer, the planner for a task's subtasks (an executor "
        "may read one with a tool). They exist only with personalisation on (`/personalize`"
        "); turned off, the nodes stay unused until removed. A temporary chat "
        "never proposes one, and neither it nor the plans of its tasks see them. `/nodes` lists them, `/nodes show PATH` "
        "shows any node's text, `/nodes add PATH --about TEXT --text TEXT`, `/nodes"
        " change PATH` (--to, --about, --text) and `/nodes remove PATH` edit them; "
        "the page has them in Settings, Personalisation. The interlocutor may propose a "
        "change with any answer, and a done task's retelling may propose keeping part of its "
        "result; nothing is kept before the person confirms it in a window. A node kept from "
        "a task's result is marked: models read it as data, not instructions. Nodes are "
        "changed and erased in place; the journal records which node changed, never its text.",
    ),
    Topic(
        "language",
        "the chat language and confirmed language changes",
        "Configure the language with `/language CODE`. Replies follow the latest message’s language. Retellings fall back to the goal’s language when the latest message gives no signal (ok, /status), then to the interface language. Other replies fall back to the interface language. The language command is proposed only when the person explicitly asks to change the interface language.",
    ),
    Topic(
        "data",
        "where DESMAN keeps its configuration, database and task files",
        "The configuration (language, API keys, permissions, behaviour, personalisation, "
        "tuning) is one file, config.json, readable only by its owner: in the project folder, "
        "or in ~/.config/desman/ for an installed package. `/config` is the setup "
        "wizard, `/config show` shows it without keys, `/config reset` deletes it "
        "after the person's yes. For installer installs, `/uninstall` previews removal. "
        "Add --purge to delete data and keys, and --yes to confirm. Removal cancels if DESMAN "
        "windows or workers stay open for ten minutes. A failure is reported on next startup. "
        "Managed Docker SearXNG is removed; search setup restores it. Settings → General → "
        "Remove DESMAN shows the preview and asks for confirmation.\n"
        "Tasks, their history, the chat's sessions, the profile, model calls and limits are "
        "in a SQLite database, ~/.local/share/desman/desman.db; each task's files are in a "
        "folder of its own in workspaces/ next to it. `/init-db` creates the database "
        "or brings it up to date. Everything stays on the person's machine; only the "
        "requests to model providers, to Hugging Face and, when allowed, to the web leave it.",
    ),
    Topic(
        "global",
        "desman-global: the opt-in network of DESMANs, joining, leaving, lending providers",
        "`/global join` opts in (--registry URL). `/global leave` withdraws all. "
        "`/global status` shows this node's state and key fingerprint. "
        "Operators list nodes with `/operator nodes`. "
        "`/global donate PROVIDER --calls N --tokens N` lends a local or eligible free provider. "
        "Keys stay here. `/global cloud on` lets you lend free cloud keys for the network "
        "operator's jobs if your terms allow it. "
        "`/global quota PROVIDER` sets quotas and shows the UTC-day reset "
        "in local time. `/global undonate PROVIDER` withdraws it. "
        "`/global donate PROVIDER --calls N` with searxng lends managed search (calls only). "
        "`/global search STATE` (default off) controls borrowing when joined without SearXNG or Tavily. "
        "Queries omit the asker’s identity. Results are untrusted.\n"
        "Page workers register and fetch revocations. Idle nodes with fresh trust may "
        "do or review jobs in a sandbox offline, or check proposals with a fake "
        "benchmark. Personal work comes first. `/global lead STATE` (on/off, default on) "
        "allows idle appointed nodes to lead clusters. Status shows the role. Personal work or "
        "turning leadership off hands the role back.\n"
        "`/global submit-proposal N` previews texts and benchmarks (5/day, no private "
        "evidence or settings). Use --confirm HASH to send it. For reports and memory see globalsend. "
        "`/global update --show N` inspects root-signed releases (omit N for latest); "
        "--apply N installs the release. --revert undoes the latest install if HEAD and schema still match. "
        "Restart workers/chats after updating. Approve protected changes and reinstall changed "
        "dependencies. Chat changes require confirmation.",
    ),
    Topic(
        "operator",
        "Operator setup, network work, moderation and registry records",
        "`/operator status` shows the key fingerprint, public key, registry, scopes and assignment expiry. "
        "Use `/operator key import FILE` for an existing Ed25519 PEM key. "
        "`/operator key create` generates a key and shows its public key "
        "and the assignment command for the root owner. Import and create accept --registry URL. "
        "`/operator key remove` previews removal in the terminal. Add --yes to remove it. "
        "The private key stays on this machine. The tasks scope can offer and settle work and set "
        "node owners; moderation covers complaints and suspensions. "
        "Use `/operator inbox` for work waiting for a decision and its available results, `/operator jobs` and "
        "`/operator clusters` to list work, and `/operator job show ID` or "
        "`/operator cluster show ID` for results. Decisions are accept, again or drop. "
        "Moderation lists: `/operator complaints`, `/operator suspensions`, `/operator reports`. "
        "`/operator nodes` lists network nodes and owners. `/operator log` shows registry history. "
        "A lift targets one suspension. If multiple are in force, the node lift is refused. "
        "The confirmation warns if a lift ends before its suspension. "
        "Chat changes requested through the interlocutor require confirmation. "
        "On the page, manage the key in Settings → desman-global. With a key, the Operator "
        "section shows Status, Inbox, Work, Network and, with the scope, Moderation. Confirm changes in its windows. "
        "Chat reads display tables or text. Registry replies containing other people's words "
        "are shared as untrusted data after agreement. Metadata-only replies need no agreement.",
    ),
    Topic(
        "globalsend",
        "Sending bug reports and shared memory to the registry",
        "`/global report TEXT` previews your problem description and system/version; "
        "--task ID adds bounded task status, descriptions and failure causes, never workspace files. "
        "`/global share` previews aggregate stats; --failures includes failures and plan "
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
        "In DESMAN, the messenger and messages mean desman-global's messenger. "
        "After joining, `/global name NAME` shows or sets a unique lowercase "
        "ASCII name (3–32 characters). `/global contacts` lists contacts and requests. "
        "`/global contact add NAME` requests or accepts; both people must agree to send. "
        "`/global contact remove NAME` removes, withdraws or declines. "
        "`/global block NAME` removes contact and blocks requests discreetly; "
        "`/global unblock NAME` unblocks. `/global send NAME TEXT` sends. "
        "`/global messages NAME` shows up to 100 messages; --after ID reads later ones. "
        "`/global complain NAME TEXT` complains; --message ID reveals the cited incoming "
        "message to the operator and preserves its text; it must still be in your inbox. "
        "First successful use records the encryption notice. Message headers show the "
        "other person, a short fingerprint and local time. "
        "Messages are end-to-end encrypted and kept 30 days. The operator sees who writes to whom and when. Keys are pinned on first contact; "
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
