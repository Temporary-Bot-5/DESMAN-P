# Decentralized Self-Evolving Modular Agents Network (Prototype) - DESMAN(P)

## Project brief for AI assistants

DESMAN(P) is a research prototype for a **decentralized, fault-tolerant, self-evolving network of small AI agents**. Its long-term goal is to coordinate many short-lived, specialized agents—LLM-based workers, tools, and lightweight services—so that the system can decompose large tasks, recover from agent failures, prioritize critical work, and gradually improve its own components under human supervision.

The project is inspired conceptually by self-improving coding agents such as Ouroboros and decentralized multi-agent coordination systems such as AgentNet, but it is **not** an attempt to reproduce either project. DESMAN(P) combines their ideas into a practical prototype focused on durable task state, recovery, prioritization, and controlled self-improvement.

## Core concept

DESMAN(P) is not a single large monolithic agent. It is a modular network in which:

- A **task** is broken into a graph of small, independently resumable subtasks.
- Each subtask has explicit inputs, dependencies, acceptance criteria, priority, status, and output artifacts.
- **Agent workers** are temporary or dynamically created executors that claim a subtask, perform bounded work, save results externally, and terminate or release resources.
- A **planner / dispatcher** assigns work, tracks dependencies, handles retries, and can preempt low-priority work when a critical task appears.
- **Specialized agents or clusters** may focus on roles such as planning, research, coding, testing, tool use, review, or self-improvement analysis.
- The system state lives outside any individual agent, so an agent can disappear at any time without losing completed work.

The intended analogy is a simplified, software-engineered version of a brain: specialized regions, distributed processing, attention-like prioritization, and recovery from local failures. This is an engineering metaphor, not a claim of biological accuracy.

## Mandatory design principles

1. **Durable state first.**  
    No agent may be the sole holder of important state. Tasks, subtasks, plans, logs, results, and checkpoints must be stored in a persistent shared store, initially SQLite or file-based storage.
2. **Resumability.**  
    Every nontrivial task must be decomposed into checkpoints. If an executor stops unexpectedly, another executor must be able to resume from the last valid checkpoint without corrupting results.
3. **Fault tolerance.**  
    Agent failure is normal. The system must detect stale tasks through leases or timeouts, reassign work safely, and avoid duplicate side effects.
4. **Prioritization and preemption.**  
    Tasks carry priority levels. Critical tasks may interrupt, pause, or delay lower-priority work. The scheduler must make this behavior explicit and auditable.
5. **Minimal agent communication.**  
    Agents should not chat freely. They communicate through structured task records, artifacts, status updates, and narrowly defined messages. Excessive free-form multi-agent conversation increases cost, confusion, and failure modes.
6. **Human control.**  
    The human operator is the final authority. The system may propose changes, but it must not silently rewrite its own security rules, permissions, evaluation criteria, or core execution logic.
7. **Controlled self-evolution.**  
    Self-improvement is allowed only through a reviewable change pipeline: propose a change, test it in isolation, compare it with the current version, record the result, and apply it only after approval or passing strict automated checks. Every accepted change must be reversible through version control.
8. **Budget awareness.**  
    The prototype must assume limited, rate-limited, and potentially unstable free LLM API access. It must degrade gracefully, pause work when quotas are exhausted, and never assume unlimited model calls.

## Initial architecture

For the first prototype, avoid Kubernetes, peer-to-peer networking, complex message brokers, and a large persistent swarm. Build a small, reliable foundation:

- **Task Store:** persistent database of tasks, subtasks, dependencies, priorities, statuses, artifacts, and audit logs.
- **Task Queue / Scheduler:** assigns ready subtasks to available executors, manages leases, retries, priorities, and failure recovery.
- **Agent Runtime:** a lightweight process or service that receives a subtask, calls the selected LLM or tool, performs bounded work, and writes results back.
- **Model Router:** selects an available LLM provider/model based on task type, availability, rate limits, and configured fallbacks.
- **Tool Registry:** a controlled list of permitted tools and capabilities, initially file operations, code execution in a sandbox, search, and simple utilities.
- **Reviewer / Validator:** checks outputs against acceptance criteria and can request retry, revision, or human review.
- **Evolution Module:** later-stage component that analyzes failures and performance, proposes patches, prompts, workflows, or new specialized agents—but does not directly deploy unreviewed changes.

## Roles in the first version

Start with only a few roles:

- **Planner:** decomposes a user goal into a task graph.
- **Executor:** performs one bounded subtask.
- **Reviewer:** validates the result and decides whether to accept, retry, or escalate.
- **Coordinator:** manages priorities, dependencies, retries, and assignment.
- **Improvement Analyst:** later role that proposes improvements, never directly deploys them.

Do not begin with hundreds of agents. A useful first configuration is one coordinator, one planner, one executor, and one reviewer.

## Free-tier constraints

The prototype is being built with zero financial budget and without relying on local model inference. Therefore:

- Use free or trial-tier cloud LLM APIs where available, such as Gemini API, GroqCloud, OpenRouter free models, Cloudflare Workers AI, Cohere trial access, or Mistral free/experiment access.
- Treat every provider as unreliable and rate-limited.
- Implement provider fallback, request throttling, caching, and task pausing.
- Do not send secrets, credentials, private code, or personal data to free APIs without reviewing their data policies.
- Design for intermittent execution: the system may run in short sessions and resume later.

## Security and safety rules

- Never allow an agent to modify its own permissions, safety constraints, test harness, or approval rules without explicit human approval.
- Never run generated code with broad filesystem, network, payment, messaging, or system-administration access.
- Use sandboxing and least-privilege tool access.
- Log every meaningful decision, model call, tool use, retry, and state transition.
- Make all self-modification proposals reviewable diffs or pull requests.
- Preserve a rollback path to the last known-good version.
- If evaluation criteria conflict with safety constraints, safety constraints win.

## What the prototype must demonstrate

A successful early DESMAN(P) prototype should demonstrate that it can:

1. Accept a goal and create a structured task graph.
2. Execute subtasks using one or more free LLM APIs.
3. Save intermediate results and continue after a simulated agent crash.
4. Reassign a stale or failed subtask to another executor.
5. Prioritize an urgent task over lower-priority work.
6. Avoid unsafe duplicate execution of side-effecting actions.
7. Produce an auditable history of decisions and artifacts.
8. Propose—but not automatically deploy—an improvement to a prompt, workflow, tool, or agent configuration.

## Explicit non-goals for the first prototype

- No fully autonomous, continuously running swarm.
- No unsupervised recursive self-rewriting of the core system.
- No peer-to-peer network or blockchain component.
- No attempt to replace a large frontier model with many small models for every task.
- No complex web UI before the CLI and core workflow are stable.
- No assumption that free API quotas will remain available indefinitely.

## Current development approach

The project begins with a portable, cross-platform development environment: a repository, Python-based runtime, SQLite-backed task store, configuration files, CLI commands, workspace directory, logs, and tests. The first milestone is not self-evolution; it is a reliable single-agent execution loop that can create, execute, checkpoint, resume, and audit a task.

Only after that foundation works should the project add multiple executors, model routing, priority preemption, fault injection tests, and finally a tightly constrained self-improvement workflow.

## Instruction for AI assistants working on DESMAN(P)

When assisting with this project:

- Treat recoverability, auditability, and human control as higher priorities than autonomy or speed.
- Propose the smallest next implementable milestone rather than generating the entire system at once.
- Avoid unnecessary dependencies and complex infrastructure.
- Assume API limits, outages, and agent crashes.
- Explain architectural decisions in plain language.
- Never provide code that grants an agent unrestricted system access.
- Never implement silent self-modification of safety rules or evaluation logic.
- Prefer explicit state machines, task records, and testable components over informal agent-to-agent conversation.

DESMAN(P) is therefore a **prototype for a resilient modular agent network**, not a finished autonomous AI. Its immediate purpose is to prove that a small, carefully organized swarm can work reliably under free-tier constraints while preserving recoverability, prioritization, and human oversight.
