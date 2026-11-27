# Multi-agent, decentralized, and fault-tolerant systems for DESMAN(P)

Research snapshot, 27 November 2026. Dates identify the paper, specification, or release discussed. DESMAN already has the basic machinery for delegation and recovery. The highest-value changes are to make a resumed task safe to repeat, measure which jobs merit several model calls, and make remote results auditable. These address different problems. A checkpoint records progress, a signature attributes a result, and an acceptance check establishes whether the result is usable. The coordinator should own all three decisions, including escalation to the human inbox. [Temporal activity semantics](https://docs.temporal.io/activity-definition) · [A2A signature specification](https://a2a-protocol.org/latest/specification/#84-agent-card-signature) · [MAST study](https://arxiv.org/html/2503.13657)

## Multi-agent architecture

### AgentNet

The project that inspired DESMAN appeared as a 1 April 2025 preprint and at NeurIPS 2025. Each AgentNet agent combines a router and an executor, with separate bounded stores of useful trajectory fragments. The algorithm chooses an entry agent by similarity between task requirements and capability profiles. A router can execute locally, pass the task on, or split it. Experience updates capability profiles and the collaboration graph, and similar past trajectories become examples for future tasks. The paper evaluates math, logical QA, and function calling. Its decentralized routing and reduced context exchange are design properties. Production recovery across unreliable, mutually distrustful machines and enforceable confidentiality would need separate evidence. [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2025/hash/9a379c1b05793d1c42dc832269834515-Abstract-Conference.html) · [Algorithm and evaluation](https://arxiv.org/html/2504.00587) · [Reference implementation](https://github.com/JohnSili/AgentNet)

For DESMAN: Record verified completion, quality, and provider availability by capability and task family. Have the Improvement Analyst propose routing changes from those observations, evaluate them in the existing worktrees, and leave deployment to a human commit.

### MetaGPT

MetaGPT models software delivery as a sequence of standard operating procedures. A product role writes requirements, architecture turns those into design artifacts, and engineering consumes the artifacts. Roles subscribe to relevant published messages and use structured documents for handoffs. The paper appeared at ICLR 2024, and Python v0.8.2 shipped on 9 March 2025. Its most transferable idea is the required intermediate work product that downstream roles can inspect. [ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/6507b115562bb0a305f1958ccc87355a-Paper-Conference.pdf) · [v0.8.2](https://pypi.org/project/metagpt/0.8.2/)

For DESMAN: Give each DAG edge an output contract. Include the input artifact revision, deliverable type, acceptance criteria, and evidence locations, so the reviewer knows what the executor was asked to produce.

### CAMEL and OWL

CAMEL's 31 March 2023 paper explored role-playing with separately prompted agents, including an assistant and a user role that gives it directions. Its later Workforce implementation adds a task coordinator, a task-planning agent and worker nodes communicating over a task channel. OWL uses the CAMEL stack for tool-assisted tasks such as search and document processing. On 18 May 2025 the team published an initial experiment for reproducing its Workforce results on GAIA. This is an author-reported benchmark effort. The practical mechanism is explicit task assignment and collection of results behind a coordinator. [CAMEL paper](https://arxiv.org/abs/2303.17760) · [Workforce implementation](https://github.com/camel-ai/camel/blob/master/camel/societies/workforce/workforce.py) · [OWL repository and timeline](https://github.com/camel-ai/OWL)

For DESMAN: Treat OWL-style specialists as bounded workers, such as an evidence extractor that returns source snippets and citations. Keep shared writes under the executor's guarded workspace policy.

### Microsoft AutoGen, AG2, and Microsoft Agent Framework

Microsoft AutoGen v0.4, announced 14 January 2025, reworked the original conversational framework around an asynchronous, message-driven Core, with higher-level AgentChat teams and extensions for models and tools. It introduced a distributed runtime option alongside local execution. Microsoft put it in maintenance mode in October 2025, and its documentation now says it is archived (the date of archiving was not verified). Use its architecture as a reference and budget for the implications of adopting an archived dependency. [Microsoft Research release](https://www.microsoft.com/en-us/research/blog/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) · [Archived repository](https://github.com/microsoft/autogen)

For DESMAN: Borrow explicit message envelopes for worker start, completion, and failure. SQLite can keep the authoritative record as long as the existing deployment model meets the workload.

AG2 is the community lineage of the original AutoGen API. AG2 Classic retains `ConversableAgent` and group-chat patterns. AG2 1.0.0 shipped on 27 July 2026 and uses an asynchronous `Agent` and `Network` design. Its channels include request/reply consultation, round-robin discussion, and conditional transitions. Classic code needs migration to the newer API. [AG2 repository and migration overview](https://github.com/ag2ai/ag2) · [AG2 1.0.0](https://pypi.org/project/ag2/1.0.0/)

For DESMAN: Use a typed reviewer consultation with an explicit reply and deadline. Prefer that narrow protocol when the task needs one judgement.

Microsoft Agent Framework 1.0 reached general availability on 3 April 2026. The Python and .NET framework brings together patterns from AutoGen and Semantic Kernel. It supplies agent abstractions and graph workflows with executors, typed edges, checkpointing and human interaction, plus A2A and MCP interoperability. Graph workflow state and production deployment still depend on the configured runtime and storage. [1.0 announcement](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/) · [Workflow overview](https://learn.microsoft.com/agent-framework/concepts/workflows/)

For DESMAN: Reuse the distinction between an executor's typed input/output and the workflow's routing decision. Revisit adoption if external integrations justify migrating the existing scheduler.

### CrewAI

CrewAI 1.0.0 shipped on 20 October 2025. Crews organize role-bearing agents and tasks into sequential or manager-led processes. Flows use event-triggered methods, listeners, routers, and typed or dictionary state. The open-source persistence layer can store flow state through SQLite. A persisted flow state is a useful application boundary. Determining whether an interrupted external effect already happened remains the application's responsibility. [1.0.0 package](https://pypi.org/project/crewai/1.0.0/) · [CrewAI processes](https://docs.crewai.com/en/concepts/processes) · [Flows](https://docs.crewai.com/en/concepts/flows) · [SQLite persistence code](https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/flow/persistence/sqlite.py)

For DESMAN: Expose meaningful flow transitions in the inbox, including what artifact changed and what decision is needed. Treat branch decisions as persisted results, so a restart cannot silently choose a different branch.

### LangGraph

LangGraph 1.0 shipped on 22 October 2025. A checkpointer stores state keyed by a thread ID at graph super-step boundaries. Pending writes from nodes that finished in a partially failed super-step allow those results to be reused. An interrupt persists the pause payload and yields control. Resuming the same thread with a command re-enters the node from the start and supplies the human response at the interrupt. Replay therefore requires idempotent pre-interrupt effects or separate recorded steps. Synchronous persistence writes before the next step starts. Asynchronous persistence can overlap execution with writes. Exit-only persistence can lose progress from before a crash. [1.0 announcement](https://www.langchain.com/blog/langchain-langgraph-1dot0) · [Persistence and pending writes](https://docs.langchain.com/oss/python/langgraph/persistence) · [Durability modes](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py) · [Interrupt semantics](https://docs.langchain.com/oss/python/langgraph/interrupts)

For DESMAN: Make approval a separate checkpointed transition tied to the exact artifact digest and proposed action. Resume with that decision and reuse completed sibling results after preemption or worker loss.

### OpenAI Agents SDK

Launched on 11 March 2025, the open-source SDK runs the agent/tool loop, with guardrails, tracing, and sessions. A handoff transfers the run to another agent, whereas an agent called as a tool returns to the caller for synthesis. Handoff filters can restrict the context forwarded. The April 2026 update introduced a model-native harness with sandbox execution. The SDK's session and resumable run facilities should be assessed separately from a coordinator's guarantees about competing workers, external effects and provider retries. Temporal's integration explicitly runs model calls as Activities. [Launch](https://openai.com/index/new-tools-for-building-agents/) · [SDK repository](https://github.com/openai/openai-agents-python) · [Handoffs](https://openai.github.io/openai-agents-python/handoffs/) · [15 April 2026 update](https://openai.com/index/the-next-evolution-of-the-agents-sdk/) · [Temporal integration](https://docs.temporal.io/develop/python/integrations/openai-agents)

For DESMAN: Treat research and review specialists as tools of the coordinating role. Use explicit handoffs when responsibility really transfers. Trace tool calls and approvals against stable task and attempt IDs.

### Google ADK

Google introduced Agent Development Kit at Cloud Next in April 2025. ADK's runner mediates agent execution and session events. LLM agents can select tools or delegate, while workflow agents encode sequences, parallel branches, and loops. ADK Python 2.0.0 reached GA on 19 May 2026 and added graph-based workflows with nodes and explicit routing. ADK is an application framework and can expose remote agent interfaces through A2A. [Google Cloud Next recap](https://cloud.google.com/blog/topics/google-cloud-next/next25-developer-keynote-recap) · [ADK repository](https://github.com/google/adk-python) · [Python 2.0.0 release](https://github.com/google/adk-python/releases/tag/v2.0.0) · [2.0 documentation](https://github.com/google/adk-docs/blob/main/docs/2.0/index.md)

For DESMAN: Borrow the explicit distinction between model-selected actions and scheduler-controlled edges. Use an adapter to try ADK-backed agents without giving them ownership of the task database.

## Protocols across trust boundaries

### A2A

The Agent2Agent protocol shipped v1.0 on 12 March 2026. The v1.0.1 changelog is dated 26 May. An Agent Card describes skills, interfaces, and authentication requirements and can carry a JSON Web Signature. The wire protocol supports messages, stateful tasks, artifacts, and status updates through polling, streaming or optional push. That contract can support a remote task that needs more input before it can finish. The receiving service implements the execution and recovery semantics behind its task IDs. A card signature authenticates the advertised document under the appropriate key. [1.0 announcement](https://a2a-protocol.org/dev/blog/2026/03/12/a2a-protocol-ships-v10-production-ready-standard-for-agent-to-agent-communication/) · [v1.0.1 notes](https://github.com/a2aproject/A2A/releases/tag/v1.0.1) · [Specification](https://a2a-protocol.org/latest/specification/)

For DESMAN: Try read-only capability advertisement and a single externally visible research-task lifecycle. Map remote task IDs to local job IDs durably and keep authorization and acceptance in DESMAN.

### MCP

Model Context Protocol standardizes host/client access to server-provided tools, resources, and prompts using JSON-RPC. The 28 July 2026 revision removed transport sessions and initialization from the core, sending protocol version and client capabilities on each request. Tasks moved into an extension. Broken HTTP response streams require a new request ID, which means DESMAN needs its own stable operation identity when a retry might repeat a write. A tool description and its provider's identity are inputs to an authorization decision. [2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28) · [Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)

For DESMAN: Introduce MCP through an allowlisted adapter for read-only tools. Route any tool that could mutate state through existing workspace, sandbox, and confirmation checks, and reconcile interrupted calls before retrying.

### ANP

Agent Network Protocol is an open protocol suite for cross-domain agent identity and communication. Its September 2026 v1.2 catalog includes HTTP request authentication using `did:wba` or `did:web`. Its discovery and description work can advertise interfaces such as OpenRPC, and its optional meta-protocol addresses negotiation. The repository distinguishes released documents from draft or candidate profiles. Specification status is different from field-tested interoperability. Web-based identifiers still depend on control of and trust in the publishing domain and keys. [ANP releases](https://github.com/agent-network-protocol/AgentNetworkProtocol/releases) · [Specification catalog](https://github.com/agent-network-protocol/AgentNetworkProtocol) · [Getting-started guide](https://github.com/agent-network-protocol/AgentNetworkProtocol/blob/main/docs/anp-getting-started-guide.md)

For DESMAN: Start with explicit key rotation, revocation, and capability expiry in the signed registry. Add DID-based identity when a federation partner needs it, and keep task authorization scoped to a job.

## Decentralized inference, training, and quota sharing

### Petals

Petals' 13 December 2023 paper describes peers hosting ranges of transformer blocks. A distributed hash table advertises available ranges, and clients choose a pipeline through them. On a peer failure the client can find a substitute and send the inputs needed to reconstruct that stage's attention cache. This recovers from churn given alternative serving capacity. The paper discusses privacy risks with public nodes, and its routing mechanism gives no general proof that each peer computed honestly. Python v2.2.0 shipped on 6 September 2023. Present-day public swarm capacity would need measurement. [Petals paper, including recovery and privacy](https://arxiv.org/html/2312.08361) · [v2.2.0](https://pypi.org/project/petals/2.2.0/)

For DESMAN: Carry the durable inputs needed to retry a subtask on another node and maintain alternatives by capability. Avoid putting layer-sharded public inference in the critical path of quota lending.

### Bittensor

Bittensor coordinates incentivized commodity markets in subnets. Miners perform subnet-specific work and validators evaluate them off chain. Validator weights go to the chain and Yuma Consensus aggregates them with stake weighting and clipping to allocate rewards. The February 2025 Dynamic TAO upgrade introduced subnet tokens. The key distinction for DESMAN is between a subnet's scoring mechanism and the mechanism aggregating its scores. Quality and collusion resistance depend on who validates and what they check. [Official network overview](https://preview.bittensor.com/docs/concepts/network) · [Yuma Consensus](https://www.bittensor.com/docs/internals/consensus) · [Dynamic TAO timeline](https://www.yumaai.com/updates/state-of-bittensor-volume-2.pdf)

For DESMAN: Maintain a reputation history from verified assignments by task family. Assign checks across independent operators and providers where possible. Keep the raw check outcomes available to the human.

### Prime Intellect

INTELLECT-1, released on 29 November 2024, used low-communication distributed pretraining with local optimization and periodic exchange of updates. INTELLECT-2, released 11 May 2025, separated inference-generated rollouts from training and used TOPLOC validators to check rollouts before admitting them. TOPLOC, announced 28 January 2025, fingerprints activation information and checks it through model recomputation with tolerances for execution differences. It concerns model execution under specified settings. It does not establish whether the text is correct, and using it requires access that hosted free-tier APIs generally do not expose to DESMAN. [INTELLECT-1 release](https://www.primeintellect.ai/blog/intellect-1-release) · [INTELLECT-2 release](https://www.primeintellect.ai/blog/intellect-2-release) · [TOPLOC design](https://www.primeintellect.ai/blog/toploc)

For DESMAN: Borrow the intake gate between remote generation and accepted artifacts. Use model-execution verification only in a separate experiment with open weights and instrumentation.

### Nous Psyche

Nous published Psyche's architecture on 14 May 2025. It coordinates distributed training using reduced-communication updates, commitments, and witness reports. A Solana program can coordinate rounds, or a TCP coordinator can be used in centralized development runs. Witnesses observe results and quorum advances the round. The protocol's evidence about participation and update availability should not be read as certification of every gradient. Nous's September 2025 update describes the Consilience run. Continued capacity for arbitrary workloads is a separate question. [Architecture](https://nousresearch.com/nous-psyche) · [Protocol internals](https://docs.psyche.network/explain/index.html) · [September 2025 update](https://nousresearch.com/the-next-phase-of-psyche)

For DESMAN: Use a bounded review window with explicit outcomes for checker silence, disconnects, and disagreements. Escalate unresolved jobs instead of allowing a cluster to hold the coordinator indefinitely.

### Exo

Exo is a local distributed inference runtime. It discovers devices, uses topology information to place model shards, and supports tensor parallelism over fast links including RDMA over Thunderbolt on supported Macs. The April 2026 v1.0.71 release records continuing sampling defaults and RDMA fixes. Its published architecture emphasizes cooperating devices in a local cluster, with no documented mechanism comparable to randomized audits of public providers. [Exo repository](https://github.com/exo-explore/exo) · [v1.0.71 release notes](https://github.com/exo-explore/exo/releases/tag/v1.0.71)

For DESMAN: Model a trusted Exo deployment as one optional inference backend, like Ollama. Measure its availability as a cluster and expose it under the same provider-routing interface.

### Hyperbolic

Hyperbolic operates a GPU marketplace and proposed Proof of Sampling. The research paper appeared on 1 May 2024 and was revised in May 2025. The proposal samples work for recomputation or challenge and uses incentives and penalties to make cheating costly under its model assumptions. The marketplace's supplier program and the verification paper establish different things. I found no public evidence in those sources that every production GPU or inference call is covered by the paper's verification scheme. [GPU supplier program](https://www.hyperbolic.ai/suppliers) · [Proof of Sampling paper](https://arxiv.org/abs/2405.00295)

For DESMAN: Sample low-risk jobs for independent rechecks, use deterministic validators wherever possible, and escalate important disagreements. Sampling reduces checking cost but can miss bad results and needs genuinely independent checkers.

### Volunteer compute and API quota

AI Horde is a working architectural analogue for volunteer capacity. A shared queue distributes image and text jobs to volunteer machines. Kudos affect priority, and a trusted-worker setting restricts worker eligibility. Those controls provide accountability and queue incentives. A volunteer operator can still see the request. [AI Horde server and kudos](https://github.com/Haidra-org/AI-Horde) · [Trusted-worker limits](https://beta.stablehorde.net/docs/image/settings/trusted_workers/) · [Privacy policy](https://aihorde.net/privacy/)

For DESMAN: Use contribution history to inform queue priority only after measuring fairness and failure rates. Limit volunteer jobs to inputs explicitly approved for that trust class.

AI Commons is a closer design analogue. A self-hosted coordinator leases schema-defined tasks to volunteers running Ollama or an opted-in commercial API. Workers hold their own keys and enforce local budgets, and the coordinator cross-checks outputs. Its README describes an early MVP with local testing, no published release, and no public coordinator. [AI Commons repository](https://github.com/tartalonllc/ai-commons)

For DESMAN: Lend bounded execution from the owner's node and keep its upstream credentials local. Start with Ollama and upstream offerings whose terms permit the intended use. Enforce caps at both the lender and scheduler, and bind signed result receipts to job, input digest, attempt, model claim, and expiry. Treat a model name in the receipt as a claim, and refuse sensitive inputs on volunteer nodes. Provider terms and free-tier data handling vary. [Gemini terms](https://ai.google.dev/gemini-api/terms) · [OpenAI Services Agreement](https://openai.com/policies/services-agreement/)

Aweshare documents an invite-only hub. Producer agents publish offerings, and distinct consumer credentials let the hub meter rate and usage and revoke access. The hub is a trust and availability dependency, and the documented controls govern access rather than inference quality. [Aweshare repository](https://github.com/Wehuman01/aweshare)

For DESMAN: Borrow the separation of producer admission, offering caps and consumer authorization. Keep acceptance of outputs with the coordinator.

Token Bank pairs a local provider gateway with an optional community mesh and advertises sharing idle quota. Its design depicts local provider keys and separate relay credentials. Its materials did not establish public usage at scale or independent validation of returned answers. I did not verify a mature public federation of donated free-tier API quota. [Token Bank repository](https://github.com/wink-run/tokenbank)

For DESMAN: Use a local-only default and explicit opt-in for each outbound offering. Test the peer lifecycle with known operators before relying on anonymous capacity.

## Durable execution runtimes

### Temporal

Temporal Server v1.0.0 shipped on 30 September 2020. A service records an event history, and workers reconstruct deterministic workflow control flow by replaying it. Activities handle external I/O, including LLM calls. A recorded Activity completion is reused during replay, but an attempt that made an external change before losing its completion acknowledgement can run again. Signals allow external decisions to reach a waiting workflow. The separate service is useful when durable orchestration is itself shared infrastructure. [v1.0 announcement](https://community.temporal.io/t/official-v1-announcement/680) · [Workflows and replay](https://docs.temporal.io/workflows) · [Activity idempotency](https://docs.temporal.io/activity-definition) · [Python message passing](https://docs.temporal.io/develop/python/message-passing)

For DESMAN: Record operation intent before dispatch and the observed result afterwards. For an ambiguous outcome, reconcile against the target or use its supported idempotency key before retrying. Fault-inject a worker crash between the external effect and the acknowledgement.

### DBOS

The Python DBOS 2.0.0 package shipped on 25 September 2025. DBOS uses decorators to mark workflows and steps, then records step outcomes and workflow state in a system database. Current Python configuration supports SQLite, with Postgres recommended for production. Datasource operations can store a transaction's result with its database writes, which is where DBOS's exactly-once guarantee applies. A separate model API call has its own failure semantics. [2.0.0 package](https://pypi.org/project/dbos/2.0.0/) · [Workflow semantics](https://docs.dbos.dev/python/tutorials/workflow-tutorial) · [Database configuration](https://docs.dbos.dev/python/reference/configuration) · [Transaction semantics](https://docs.dbos.dev/python/tutorials/transaction-tutorial)

For DESMAN: DBOS is the closest migration candidate for a synchronous SQLite proof of concept. First try its transaction pattern locally by committing the artifact pointer, step result, and ready-state updates together, conditional on the worker's current lease generation.

### Restate

Restate 1.0 launched on 10 June 2024. A separate runtime journals service invocations. A Python handler wraps nondeterministic operations in durable steps, and replay reads recorded results from the journal. A one-shot awakeable or workflow promise can hold a pending operation until an external callback arrives. Waits inside exclusive handlers can queue other calls for that object, an important design detail for approvals. [1.0 announcement](https://restate.dev/blog/announcing-restate-1.0-restate-cloud-and-our-seed-funding-round) · [Python durable steps](https://docs.restate.dev/develop/python/durable-steps) · [External events](https://docs.restate.dev/develop/python/external-events)

For DESMAN: Persist a single-use approval ID with expiry, authenticate the responding human, and suspend execution without holding a worker lease. Bind the response to the exact action displayed in the inbox.

### Inngest

Inngest announced self-hosting alongside 1.0 on 23 September 2024. Events trigger functions, and the executor persists named step results. A fresh invocation can then skip completed step work. Durable waits match external events or wake at scheduled times. Flow control includes keyed concurrency and throttling. These controls fit quota-driven workloads, with the caveat that durable step results and event payloads consume persisted state. [1.0 announcement source](https://github.com/inngest/website/blob/main/content/blog/inngest-1-0-announcing-self-hosting-support.mdx) · [Durable primitives](https://www.inngest.com/docs/durable-execution/primitives) · [Flow control](https://www.inngest.com/docs/durable-execution/flow-control) · [Step limits and best practices](https://www.inngest.com/docs/durable-execution/best-practices)

For DESMAN: Schedule by upstream quota scope, such as provider and account/project, and persist the next eligible time after a rate limit. Keep checkpoint payloads small by referencing content-addressed artifacts.

## Agent memory

### Letta and MemGPT

The 12 October 2023 MemGPT paper described an operating-system-inspired split between in-context memory and external storage, with model-invoked operations for moving information into and out of context. Letta exposes named core memory blocks with bounded content and searchable archival passages. Directly attached blocks can affect every future turn, so a write to a shared block is a consequential state change. [MemGPT paper](https://arxiv.org/abs/2310.08560) · [Letta core memory](https://docs.letta.com/v1-sdk/memory/memory-blocks) · [Archival memory](https://docs.letta.com/v1-sdk/memory/archival-memory)

For DESMAN: Keep a small human-editable profile and job-specific brief in context. Retrieve other knowledge by reference. Send durable profile edits to the inbox and preserve an audit trail.

### mem0

The mem0 paper appeared on 28 April 2025. Its extraction stage turns new conversation material into candidate memories using recent context and a summary. The update stage compares candidates with retrieved memories and decides whether to add, update, delete, or leave an entry alone. The paper also evaluates a graph variant. The system optimizes reuse of compact facts, with an inference step that can introduce bad merges or omit important qualifications. [Mem0 paper](https://arxiv.org/html/2504.19413) · [Mem0 implementation](https://github.com/mem0ai/mem0)

For DESMAN: Store extracted facts as proposals with source and subject scope. Require confirmation before they update the profile or become shared instructions.

### Zep and Graphiti

Zep's paper appeared on 20 January 2025. Graphiti ingests episodes, extracts entities and relationships, and tracks when facts were valid and when the system learned or superseded them. Hybrid retrieval combines semantic, lexical, and graph signals. A temporal edge can keep an old statement available for audit after later evidence supersedes it. Extraction and conflict resolution still depend on the ingested evidence and models. [Zep paper](https://arxiv.org/abs/2501.13956) · [Graphiti repository](https://github.com/getzep/graphiti)

For DESMAN: Add observation date, validity window, evidence URI and supersession to knowledge-tree records. Start with SQLite and test whether time-aware retrieval answers operational questions about provider changes or outdated documents.

## When multiple agents help, and why they fail

**MAST.** The NeurIPS 2025 study identifies 14 failure modes grouped into design and specification, inter-agent misalignment, and verification and termination. Examples include losing context, repeating work, ending before acceptance criteria are met, and relying on inadequate checks. The taxonomy came from expert trace analysis and was applied more broadly with automated annotation. Its benchmark patterns are diagnostic evidence, with prevalence tied to the studied frameworks and tasks. [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/b1041e52d3be19f0a9bc491657488e4a-Abstract-Datasets_and_Benchmarks_Track.html) · [Full study and failure examples](https://arxiv.org/html/2503.13657)

For DESMAN: Label benchmark traces with observed failure categories and collect explicit termination evidence. Include cases where a well-formed artifact fails a substantive requirement.

**Cognition and Anthropic.** Cognition's 12 June 2025 practitioner account argues that independent agents doing tightly coupled work lose decision context, especially when editing code, and favors a coherent controlling trajectory. Anthropic's 13 June 2025 research write-up describes a lead agent that delegates independent search directions, synthesizes findings and uses a citation agent. Anthropic reports a 90.2% relative improvement over single-agent Opus 4 on its internal research evaluation with Opus 4 leading Sonnet 4 subagents. It also reports that multi-agent runs used about 15 times the tokens of ordinary chats in its data. The evaluation and token comparison have different baselines. Together the write-ups suggest fan-out when useful branches can work independently and hand back evidence that can be combined. [Cognition account](https://cognition.com/blog/dont-build-multi-agents) · [Anthropic architecture and internal evaluation](https://www.anthropic.com/research/multi-agent-research-system)

For DESMAN: Parallelize independent evidence collection and candidate generation, then give a single owner responsibility for synthesis or shared writes. Keep tightly coupled workspace edits on one trajectory.

**Many smaller models versus one larger model.** Mixture-of-Agents, posted 7 June 2024, fed outputs from several models into a later aggregator and reported beating GPT-4 Omni on AlpacaEval 2.0 with an open-model mixture. That is evidence for the evaluated model mix and preference benchmark, with no equal-call or equal-compute guarantee. The December 2025 scaling study, revised 8 April 2026, standardized compute, prompts and tools across architectures. It found diminishing returns from collaboration as single agents became more capable, benefits on some parallelizable workloads, and error propagation when centralized verification was absent. FrugalGPT's May 2023 cascade study supplies a complementary baseline in which a cheap model attempts the task first and a stronger one handles uncertain cases. [Mixture-of-Agents](https://arxiv.org/abs/2406.04692) · [Controlled scaling study](https://arxiv.org/abs/2512.08296) · [FrugalGPT](https://arxiv.org/abs/2305.05176)

A more direct size comparison comes from an April 2026 workshop study. Its Qwen3 experiments allocated specialist tools to small subagents and concentrated reasoning in the orchestrator. Small teams beat larger single-model configurations on some tested workloads, and upgrading the orchestrator helped more consistently than enlarging the workers. The multi-agent setup also used structured task memory, so the study evaluates a package of design choices. An 8 November 2026 study of two models of at most one billion parameters found that fixed-size voting could even lower accuracy on one tested math setting. Correlated wrong answers make voting a weak default. [Small-agent collaboration study](https://arxiv.org/html/2601.11327v2) · [Small-model self-consistency study](https://arxiv.org/abs/2611.11201)

For DESMAN: Reserve stronger reasoning for planning, synthesis and adjudication. Compare a small single model, a stronger single model, a cascade, independent samples with a verifier, and bounded specialist fan-out. Control calls and tokens where possible and report concurrency, wall time, throttled attempts, human escalations, and verified completion separately. Test task families independently. Allow fan-out when branches add distinct evidence or a checker can reliably choose among candidates.

## Top ideas for DESMAN

Effort estimates are relative to the current synchronous SQLite prototype and reflect implementation and verification work.

1. **Make retries safe after uncertain completion** — M, coordinator and executor. Persist a stable operation ID, attempt and lease generation. Reuse recorded results, reject stale commits, and reconcile ambiguous effects before retry. Test crashes around dispatch and acknowledgement. [Temporal Activity semantics](https://docs.temporal.io/activity-definition)
2. **Require evidence at DAG handoffs** — S, planner, executor and reviewer. Put the expected output, artifact revision, and acceptance check on each edge. Require a verification outcome before terminal success. [MetaGPT](https://proceedings.iclr.cc/paper_files/paper/2024/file/6507b115562bb0a305f1958ccc87355a-Paper-Conference.pdf) · [MAST](https://arxiv.org/html/2503.13657)
3. **Bind approvals to an exact action** — M, coordinator, inbox and Improvement Analyst. Store the artifact digest, operation, approver and expiry. Release the worker while waiting. A changed revision requires a fresh decision, and prompt or settings changes still require the human's git commit. [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) · [Restate external events](https://docs.restate.dev/develop/python/external-events)
4. **Schedule around real quota boundaries** — M, model router and coordinator. Track backoff and concurrency per upstream quota scope, stop filling the queue with unready retries, and make fallback eligibility explicit. [Inngest flow control](https://www.inngest.com/docs/durable-execution/flow-control)
5. **Benchmark when to fan out** — M, planner, reviewer and Improvement Analyst. Compare a single trajectory, a cascade, and parallel candidates under the same workload. Use benchmark results to propose routing changes for human approval. [Controlled scaling study](https://arxiv.org/abs/2512.08296) · [FrugalGPT](https://arxiv.org/abs/2305.05176)
6. **Pilot signed receipts and selective audits** — L, desman-global and reviewer. Use local key custody, explicit workload opt-in, per-peer caps, result receipts, and randomly assigned independent checks. Record disputes and make sensitive jobs ineligible by default. [AI Commons](https://github.com/tartalonllc/ai-commons) · [Proof of Sampling](https://arxiv.org/abs/2405.00295)
7. **Add provenance and expiry to memory** — M, knowledge tree and profile. Store when a fact was observed, when it was valid, and what superseded it. Keep proposed shared instructions subject to human review. [Graphiti](https://github.com/getzep/graphiti) · [Letta memory blocks](https://docs.letta.com/v1-sdk/memory/memory-blocks)
8. **Keep interoperability at the edges** — M, executor and desman-global. Use MCP for an allowlisted tool and A2A for one remote research skill, proving that both can resume through local IDs and authorization checks. Evaluate ANP federation when another operator needs domain-based identity. [MCP specification](https://modelcontextprotocol.io/specification/2026-07-28) · [A2A specification](https://a2a-protocol.org/latest/specification/) · [ANP catalog](https://github.com/agent-network-protocol/AgentNetworkProtocol)

## Sources

- Orchestration papers and implementations. [AgentNet](https://arxiv.org/html/2504.00587) · [MetaGPT](https://proceedings.iclr.cc/paper_files/paper/2024/file/6507b115562bb0a305f1958ccc87355a-Paper-Conference.pdf) · [CAMEL](https://arxiv.org/abs/2303.17760) · [OWL](https://github.com/camel-ai/OWL)
- Orchestration runtimes and release notes. [AutoGen](https://github.com/microsoft/autogen) · [AG2](https://github.com/ag2ai/ag2) · [Microsoft Agent Framework](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/) · [CrewAI](https://docs.crewai.com/en/concepts/flows) · [LangGraph](https://docs.langchain.com/oss/python/langgraph/persistence) · [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) · [Google ADK](https://github.com/google/adk-python)
- Protocol specifications. [A2A](https://a2a-protocol.org/latest/specification/) · [MCP](https://modelcontextprotocol.io/specification/2026-07-28) · [ANP](https://github.com/agent-network-protocol/AgentNetworkProtocol)
- Compute and verification. [Petals](https://arxiv.org/html/2312.08361) · [Bittensor](https://www.bittensor.com/docs/internals/consensus) · [TOPLOC](https://www.primeintellect.ai/blog/toploc) · [Psyche](https://docs.psyche.network/explain/index.html) · [Exo](https://github.com/exo-explore/exo) · [Proof of Sampling](https://arxiv.org/abs/2405.00295)
- Volunteer infrastructure. [AI Horde](https://github.com/Haidra-org/AI-Horde) · [AI Commons](https://github.com/tartalonllc/ai-commons) · [Aweshare](https://github.com/Wehuman01/aweshare) · [Token Bank](https://github.com/wink-run/tokenbank)
- Durable execution. [Temporal](https://docs.temporal.io/workflows) · [DBOS](https://docs.dbos.dev/python/tutorials/workflow-tutorial) · [Restate](https://docs.restate.dev/develop/python/durable-steps) · [Inngest](https://www.inngest.com/docs/durable-execution/primitives)
- Memory. [MemGPT](https://arxiv.org/abs/2310.08560) · [Letta](https://docs.letta.com/v1-sdk/memory/memory-blocks) · [Mem0](https://arxiv.org/abs/2504.19413) · [Zep](https://arxiv.org/abs/2501.13956) · [Graphiti](https://github.com/getzep/graphiti)
- Research and practitioner accounts. [MAST](https://arxiv.org/html/2503.13657) · [Cognition](https://cognition.com/blog/dont-build-multi-agents) · [Anthropic](https://www.anthropic.com/research/multi-agent-research-system) · [Mixture-of-Agents](https://arxiv.org/abs/2406.04692) · [Scaling study](https://arxiv.org/abs/2512.08296) · [Small-agent collaboration](https://arxiv.org/html/2601.11327v2) · [Self-consistency](https://arxiv.org/abs/2611.11201) · [FrugalGPT](https://arxiv.org/abs/2305.05176)
