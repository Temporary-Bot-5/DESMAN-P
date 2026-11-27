# Demand: what people want from agents

Desk research through 27 November 2026 (Claude).

Sources here are mostly surveys by vendors and investors; their samples differ. Exact numbers are
indicative only; the direction they agree on matters more.

### The numbers

**Consumers** — [Menlo Ventures, State of Consumer AI 2026](https://menlovc.com/perspective/2026-the-state-of-consumer-ai/)
(5,067 US adults, July 2026):
- 64% of US adults use AI, 25% daily. Among users, 41% have tried an agent and 24% use one
  regularly. The agents they name most are coding agents (Codex, Claude Code/Cowork) and Perplexity.
- People give agents email (36%), the browser (33%), messengers (31%), cloud storage (29%) and
  calendars (27%). Fewer give health (23%) and finance (20%) access.
- What matters to users: accuracy 45%, trustworthiness 40%, security/privacy 36%, ease of use 32%.
  67% want the AI to *explain and walk them through* a topic, not just give the answer.
- Non-users: 70% distrust what AI says, 76% name privacy, 47% "don't know how".
- 55% of users pay something, typically $20–49 a month. Depth of use is growing (spending tripled)
  faster than reach (+3 points).

**Enterprises** — [G2, 3,222 enterprise reviews](https://learn.g2.com/ai-agent-adoption-trends-enterprises-2026)
(Jan 2025–Oct 2026):
- Uses: process automation 34%, customer service 27%, software engineering 21%.
- 79% asked for more *visibility or control*: explanations 32%, guardrails 29%, warnings before
  risky actions 18%. Only 21% asked for better reliability.
- Autonomy wanted: 60% "with guardrails", 21% limited, 11% approve every action, 7% full autonomy.
- Top complaint: setup and complexity (25%). Cost ~20% (usage pricing nobody can forecast).
  "Distrust tax": 31% spend 6+ hours a week overseeing agents.

**Builders** — [LangChain, State of Agent Engineering](https://www.langchain.com/state-of-agent-engineering)
(1,340 answers, Nov–Dec 2025): 57% have agents in production. Top blocker is quality (32%),
then latency (20%). Main use cases: customer service 26.5%, research and data analysis 24.4%,
internal workflow automation 18%. More than ¾ use several models; about a third run their own
models (for cost, data residency, regulation).

**Developers** — Stack Overflow 2026, as reported by [Docker](https://www.docker.com/blog/stack-overflow-survey-2026-ai-agent-trust/):
few trust AI output for important decisions, and about half trust it only when they can check it
themselves. Security/privacy is the second factor in adoption, after accuracy.

**Usage patterns** — [Anthropic Economic Index, March 2026](https://www.anthropic.com/research/economic-index-march-2026-report):
in the chat app, *augmentation* (feedback, validation, learning) is slowly growing. Through the
API, coding is moving to automated workflows (agents such as Claude Code).

**Automation** — n8n's growth came from mixing deterministic workflow steps with model calls and
agents ([Sacra](https://sacra.com/chat/h/db0ed46f-9f0e-48ed-8319-e5724eca23a5/); user and revenue
figures unverified).

### The phenomenon of 2026: personal agents in messengers

[OpenClaw](https://github.com/openclaw/openclaw) (formerly Clawdbot, by Peter Steinberger; MIT,
now under a foundation) is the most-starred repository on GitHub: ~395k stars by late
November 2026, past React in March 2026. What it is: "an AI assistant that runs on your own
computer and meets you in the channels you already use". Its parts:
- **Gateway**, one per host, is the local control plane (WebSocket `127.0.0.1:18789`) for
  sessions, tools and channels: WhatsApp, Telegram, Slack, Discord, Signal, iMessage and others.
  Device nodes (phone, desktop) add voice, camera, screen and local actions.
  ([architecture](https://docs.openclaw.ai/concepts/architecture))
- **Proactive work**: one `GatewayScheduler` with `heartbeat` and `cron` events. The agent can write
  to the person on its own, not only answer.
- **Skills**: a folder with `SKILL.md` (the AgentSkills format, as in Claude Code), installed from
  the public ClawHub registry ([skills](https://docs.openclaw.ai/tools/skills)).
- **Models** are plugins: hosted, local, or even Claude Code/Codex as a "harness".
- **Safety**: tools run *on the host* unless sandboxing is configured (Docker/Podman/SSH),
  and the docs say it "is not a perfect security boundary" ([sandboxing](https://docs.openclaw.ai/gateway/sandboxing)).

That last point produced the year's biggest agent security story. Hundreds of malicious skills
were found on ClawHub (the "ClawHavoc" campaign found by Koi Security; estimates range from ~340
to 800+, up to ~20% of the registry, by different researchers), there was a one-click RCE CVE-2026-25253, and the
"ClawJacked" attack let websites hijack a local agent through its WebSocket
([Dark Reading](https://www.darkreading.com/cyber-risk/malicious-openclaw-skills-clawhub-threaten-ai-supply-chain),
[Nebius hardening guide](https://nebius.com/blog/posts/openclaw-security)). These numbers come
from security vendors, but the direction is clear.

[Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research, MIT, ~267k stars)
is the main alternative, and the one closest to DESMAN's idea of evolution: "the self-improving AI
agent… with a built-in learning loop". It **creates skills by itself after complex tasks** and
improves them while using them. Its memory is curated by the agent, with periodic nudges to save
what it learned, FTS5 search over past sessions with summaries, and a user model (Honcho). It has
a cron scheduler that delivers results to any messenger, isolated subagents, and six terminal
backends (local, Docker, SSH, Modal, Daytona, Vercel Sandbox). It can import an OpenClaw setup.
Lighter relatives: **nanobot** (HKU, ~4k lines of Python), and **NanoClaw/ZeroClaw**, which isolate
by default ([comparison](https://agentpedia.codes/blog/best-openclaw-alternatives-self-hosted-ai-agents-2026)).

### What this means for DESMAN

1. **The demand is for a personal assistant that lives where the person already is (messenger),
   works on its own schedule, and remembers.** DESMAN already has the hard parts that OpenClaw lacks:
   durable tasks, a sandbox by default, permissions set by the human only, confirmation windows. It
   does not have channels (Telegram etc.), scheduled or recurring tasks, or proactive messages.
   *A Telegram channel and a `cron`-like recurring task are the most direct answer to this demand.*
2. **Safety is now a selling point, not a cost.** OpenClaw's skill-supply-chain disaster is the
   counterexample DESMAN can point to: bwrap+seccomp by default, protected files, signed
   releases, no third-party code without review. Anything like a skills hub must be signed and
   reviewed, never "trusted code".
3. **People want visibility and control more than autonomy** (G2: 79% vs 21%). DESMAN's design
   (inbox, audit, a window for each change, explanations) matches this. It should be *shown*:
   explain what was done and why, warn before risky steps, give clear task reports.
4. **Setup and complexity is the top complaint**, and 47% of non-users "don't know how". DESMAN's
   wizard, the chat front door and the free-keys flow are on target. Every extra step costs users.
5. **Cost**: most people pay $20–49 a month or nothing. Zero budget is a real niche (students,
   countries where $20 is a lot). So is privacy through local models. Unpredictable usage pricing
   is a named complaint, and DESMAN's limits and usage display fit that.
6. **Learning, not just answers** (67% want explanations): the interlocutor could offer
   "walk me through it" as a mode, cheaply.
7. **Hermes shows that "self-improvement" users actually feel is skills and memory learned from
   experience, not prompt rewriting.** DESMAN's Improvement Analyst works on prompts and settings
   under human approval. A *skill library learned from finished tasks*, proposed to the person and
   kept as knowledge-tree nodes, would fit DESMAN's principles and its existing knowledge tree.
