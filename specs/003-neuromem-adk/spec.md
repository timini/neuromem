# Feature Specification: neuromem-adk — Google ADK Integration

**Feature Branch**: `003-neuromem-adk`
**Created**: 2026-04-12
**Status**: Draft
**Input**: User description: "We need separate agent wrappers (e.g., neuromem-adk) so an agent built with the Google Agent Development Kit can install neuromem-adk and get persistent long-term memory with minimal wiring. This wrapper is the foundation the benchmarks will run against."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — One-Line Memory Attachment (Priority: P1)

An agent developer has an existing Google ADK agent (typically a `google.adk.agents.Agent` instance) and wants to add persistent long-term memory to it. They install the `neuromem-adk` package and call a single function to attach memory — no manual wiring of `NeuroMemory`, providers, callbacks, or tools. From that moment, the agent automatically captures every conversation turn into memory and has relevant past context injected into its prompt before each model call.

**Why this priority**: The entire value proposition of the package is "add memory in one line." Without this, the package is no better than hand-wiring neuromem directly — and hand-wiring is exactly what the user is trying to avoid. All four other user stories exist to make this one story's experience work cleanly out of the box.

**Independent Test**: Build a minimal ADK agent with a single instruction string, call the attachment function with a database path, run a multi-turn conversation, and verify that (a) the agent responds sensibly, (b) a memory database file exists at the given path with captured turns, and (c) a later question referencing an earlier turn is answered using retrieved memory.

**Acceptance Scenarios**:

1. **Given** an ADK agent constructed with a model, name, and instruction, **When** the developer calls the attachment function with a database path, **Then** memory is fully wired and the function returns an object the developer can use to inspect or override the memory state later.
2. **Given** no existing database file at the provided path, **When** the attachment function runs, **Then** a new database is created automatically with the correct schema.
3. **Given** an existing database file at the path with prior memories, **When** the attachment function runs, **Then** the agent continues from the existing memory state without losing or duplicating prior content.
4. **Given** the developer has not set any provider arguments, **When** the attachment function runs, **Then** it automatically reads the Google API key from the environment and instantiates the default provider pair.

---

### User Story 2 — Automatic Turn Capture (Priority: P1)

After every completed agent turn — user message in, assistant response out — the (user, assistant) pair flows into memory automatically. The developer never has to call a save method explicitly. Every turn is captured, regardless of how long the conversation runs or how many turns fire in a single session.

**Why this priority**: The single most important invariant for any memory library is "no dropped turns." If the developer has to remember to call a save method after every turn, eventually they will forget, and the memory graph will have gaps. Automatic capture is what makes neuromem's cognitive loop trustworthy for long-running agents.

**Independent Test**: Run a 10-turn conversation against an agent with memory attached. Without any explicit save calls in the test code, verify that the memory database contains exactly 10 consolidated memories after the conversation ends, each with the correct raw content and metadata linking back to its turn.

**Acceptance Scenarios**:

1. **Given** an agent with memory attached, **When** a user sends a message and the agent responds, **Then** the conversation turn is captured into memory without any additional developer action.
2. **Given** an agent processing a long multi-turn conversation, **When** the dream threshold is reached, **Then** the captured turns consolidate in the background without blocking the caller's thread and without dropping any turns.
3. **Given** an exception thrown during an agent turn, **When** control returns to the caller, **Then** the memory store is in a consistent state (no partially-written turns, no phantom "dreaming" status rows left behind).
4. **Given** two consecutive turns with identical raw content, **When** captured, **Then** both are stored as separate memories with distinct IDs (idempotency is the caller's responsibility, not the library's).

---

### User Story 3 — Passive Context Injection (Priority: P1)

Before each call to the underlying language model, the library automatically surfaces relevant past memories and injects them into the model's system prompt. The agent sees prior context "for free" — without the model having to explicitly decide to search memory and without the developer having to write any retrieval logic. When the developer asks a question about an earlier turn, the answer flows naturally because the relevant memory was already present in the prompt before the model started reasoning.

**Why this priority**: Passive injection is the architectural difference between this package and the competing mem0 ADK integration. Mem0 requires the language model to explicitly decide to invoke a search tool, which works for simple cases but misses memories the model doesn't realize it should look for. Passive injection makes memory available by default, which matches how biological prefrontal recall works and is the behaviour users actually expect when they hear "the agent remembers."

**Independent Test**: Have the user mention a specific fact in turn 1 (e.g., "my dog's name is Rex"). At turn 3, ask a related but non-identical question (e.g., "what did I say about my pet?"). Verify that the agent's answer correctly references the dog's name, and that the injection happened automatically without the language model calling any tools.

**Acceptance Scenarios**:

1. **Given** an agent with memory containing prior turns, **When** a new user message arrives, **Then** the relevant subset of the memory graph is rendered and prepended to the system prompt before the model call.
2. **Given** a memory store with no relevant context for the user's message, **When** a model call fires, **Then** no injection happens and the prompt is unchanged (empty-match is not an error).
3. **Given** a user message that matches many memories, **When** injection happens, **Then** the injected context is bounded in size so the prompt does not balloon uncontrollably.
4. **Given** the developer wants to inspect what got injected for debugging, **When** they check the attached memory object, **Then** they can retrieve the most recent injected context block without needing to re-run the conversation.

---

### User Story 4 — LLM-Driven Active Memory Search (Priority: P2)

The agent can also explicitly invoke memory-search operations when it decides it needs to look something up. This complements passive injection: passive injection handles "relevant context I probably need" and active search handles "I need to go look up something specific I can't find in what I already have." The language model sees two tools it can call — one for searching memory for a concept and one for retrieving specific memories by identifier.

**Why this priority**: Passive injection covers the common case but has a recall ceiling set by the size of the injected window. When the agent recognises a gap ("the user said something about their project earlier — let me check my memory for the details"), it should be able to act on that recognition via a tool call. This also provides a testable hook for benchmark suites that want to force explicit memory interactions.

**Independent Test**: Ask the agent a question whose answer is not in the passively-injected context but would surface through a targeted search. Confirm the model invokes the search tool on its own, receives relevant results, and answers correctly using them.

**Acceptance Scenarios**:

1. **Given** an agent with memory attached, **When** the agent's underlying model decides to call the memory search tool, **Then** the tool returns a structured result containing relevant memory identifiers and their summaries.
2. **Given** the agent invokes the memory retrieval tool with a list of identifiers, **When** the tool executes, **Then** it returns the full content of each requested memory and reinforces their recency weight so they are less likely to be forgotten during future decay cycles.
3. **Given** the agent calls a memory tool with an invalid or unknown identifier, **When** the tool executes, **Then** it silently skips the unknown identifier and returns only the found entries, without raising an error that would break the agent turn.

---

### User Story 5 — Native Framework Memory-Slot Integration (Priority: P2)

Google ADK defines a memory-service abstraction that framework-level features use to hook into memory operations. The library plugs into this abstraction so future ADK features that depend on the memory slot — for example, session-end hooks, tenant isolation, or managed cloud integrations — automatically work with neuromem without additional wiring. The developer does not have to choose between "use neuromem directly" and "use ADK's memory features"; they get both at once.

**Why this priority**: As ADK evolves, new features will assume the presence of a functioning memory service. Packages that ignore ADK's memory slot miss out on those features. Plugging into the slot costs a small amount of implementation effort now and saves a potentially large amount of retrofitting later. It also gives advanced users a clean extension point when they want to mix neuromem with other ADK services.

**Independent Test**: Configure the ADK runner to use the neuromem-backed memory service. Run a session through to completion, then verify that the session-end hook consolidates any pending memories (forces a dream cycle) before the session closes, and that the memory service's search method returns results through the ADK-native code path.

**Acceptance Scenarios**:

1. **Given** an ADK runner configured with the neuromem memory service, **When** a session completes, **Then** any memories captured during the session are consolidated before the session closes.
2. **Given** the ADK runner's built-in memory-search flow, **When** an agent invokes it, **Then** results come from the same underlying memory store used by the passive injection path — no divergent state.
3. **Given** two memory operations running concurrently through different ADK code paths (e.g., session-end consolidation and a tool-triggered search), **When** they execute, **Then** they do not race or corrupt the memory state.

---

### User Story 6 — Manual Provider Override (Priority: P3)

An advanced developer wants to use a non-default provider pair — for example, a local embedding provider to avoid network calls during testing, a different Gemini model for cost reasons, or a future provider for a different language model vendor entirely. They pass the providers explicitly to the attachment function, and the package uses them instead of the default auto-instantiated pair.

**Why this priority**: The 90% case is "use the default Gemini providers and go." The override path exists so the other 10% — heavy testers, cost-conscious power users, and future multi-vendor deployments — are not forced to abandon the package and hand-wire everything.

**Independent Test**: Call the attachment function with explicit provider arguments. Confirm the default auto-instantiation is skipped and the provided instances are used for every subsequent summarisation, tag extraction, cluster naming, and embedding call.

**Acceptance Scenarios**:

1. **Given** the developer passes explicit language-model and embedding-provider instances, **When** the attachment function runs, **Then** those instances are used and no default provider is instantiated.
2. **Given** the developer passes only a language-model provider but omits the embedder, **When** the attachment function runs, **Then** the default embedder is instantiated for the missing slot.

---

### Edge Cases

- **No API key in environment**: the attachment function raises a clear, actionable error naming the missing environment variable and pointing to the package README.
- **Agent already has registered tools**: the memory tools are appended to the existing tool list without overwriting any.
- **Agent already has registered callbacks**: the memory callbacks are chained with the existing callbacks so both fire in order.
- **Attaching memory twice to the same agent**: the second call is detected and raises an error (memory should be attached once per agent lifetime).
- **Database file corrupted or unreadable**: the attachment function surfaces the underlying storage error with the file path in the message.
- **Model call fails mid-turn**: any partially-captured turn is rolled back to avoid leaving the memory store in an inconsistent state.
- **Extremely long user messages**: the passive-injection tree is bounded in size so the combined prompt (original instruction + injected context + user message) does not exceed the model's context window.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The package MUST expose a single public function (default name `enable_memory`) that accepts an existing ADK agent and a database path and returns an object the developer can use to inspect or manipulate the attached memory.
- **FR-002**: The package MUST automatically instantiate the default language-model and embedding provider pair from the existing `neuromem-gemini` sibling package when the developer does not pass providers explicitly.
- **FR-003**: The package MUST read the Google API key from the environment when the default provider pair is auto-instantiated. No other credential source is required for the default path.
- **FR-004**: The package MUST capture every completed agent turn into memory automatically, using ADK's post-turn callback extension point.
- **FR-005**: The package MUST inject relevant past-memory context into the model's system prompt before each model call, using ADK's pre-model callback extension point.
- **FR-006**: The package MUST register two memory-related tools on the agent — one for searching memory by query and one for retrieving memories by identifier — so the language model can invoke them directly when it decides to.
- **FR-007**: The memory retrieval tool MUST reinforce the recency weight of any memories it returns so frequently-accessed memories survive decay.
- **FR-008**: The package MUST implement ADK's native memory-service abstraction so framework-level features that hook into the memory slot work automatically.
- **FR-009**: The package MUST NOT modify any file inside the existing `neuromem-core` package.
- **FR-010**: The package MUST NOT add any runtime dependency to `neuromem-core`'s own dependency list.
- **FR-011**: The package MUST provide a unit test suite that runs without any network access, using mocked ADK runners to verify each wiring component in isolation.
- **FR-012**: The package MUST provide one end-to-end integration test that runs against a real ADK agent and a real Google language model, gated behind an explicit test marker so it never runs during normal development flow.
- **FR-013**: The package MUST allow developers to override the default providers by passing explicit provider instances to the attachment function.
- **FR-014**: Attaching memory to the same agent twice MUST raise an actionable error rather than silently double-wiring.
- **FR-015**: Missing credentials at the default provider path MUST raise an actionable error naming the missing environment variable.

### Key Entities

- **Attached Memory Handle**: The object returned by `enable_memory`. Holds a reference to the underlying memory store, the providers, and the registered callbacks. Developers use it to inspect memory state, force consolidation, or override behaviour during tests.
- **Memory-Service Adapter**: The package's implementation of ADK's native memory-service abstraction. Plugs neuromem into the framework-level memory slot. Its session-end hook triggers consolidation of any pending memories.
- **Callback Pair**: The two ADK callbacks registered by the attachment function — one for context injection before each model call, one for turn capture after each agent turn.
- **Tool Pair**: The two function tools registered on the agent — one for search-by-query, one for retrieve-by-identifier. Bound to the attached memory handle so the language model sees only user-facing arguments.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An ADK developer can go from "I have an existing agent" to "my agent has working persistent memory" with a single function call and no manual wiring beyond one import and one function invocation.
- **SC-002**: After a 10-turn conversation through an agent with memory attached, exactly 10 turns are captured in the memory store, with no dropped or duplicated entries.
- **SC-003**: A question asked in turn N+2 about a fact mentioned in turn N is answered correctly using memory — either through passive injection into the prompt or through an active search tool call — without the developer writing any retrieval logic in between.
- **SC-004**: The end-to-end integration test against a real ADK agent and a real language model runs to completion in under 30 seconds per invocation and costs under one US cent in API fees at the current model's pricing.
- **SC-005**: The unit test suite (excluding the integration test) runs in under 10 seconds on a developer's laptop and makes zero network calls.
- **SC-006**: Every existing CI gate on the repository — the forbidden-imports tripwire, the locked-dependencies tripwire, the coverage gate, the default pytest run — stays green when this package is added. No existing test regresses.
- **SC-007**: The package's public namespace exposes no more than three names — the attachment function, the memory-service adapter class, and the version string. Everything else is private.
- **SC-008**: A developer reading the package's README for the first time can understand "what it does, how to install it, and how to use it" in under three minutes of reading.

## Assumptions

- ADK developers are comfortable with Python 3.10+ and with function-call-based tool registration.
- The default provider pair's credentials are supplied via environment variables. Alternative credential mechanisms (explicit `genai.Client` injection, Google Cloud service accounts) are deferred to follow-up work.
- The database path is supplied explicitly by the developer on every attachment call. No implicit "default location" is provided in v1; explicit is better than implicit for a library.
- The package targets the current stable Google ADK release. Minor breaking changes in future ADK releases will be handled in patch releases of this package.
- ADK's built-in memory-service abstraction is stable enough to implement against. If the ADK team changes the abstraction significantly, the impl will be updated, but the public API of this package (the attachment function and the returned handle) MUST remain source-compatible across such updates.
- This package is NOT a replacement for developers who want to build memory-aware agents outside ADK. Those developers should consume `neuromem-core` and (optionally) `neuromem-gemini` directly.
- Multi-user / multi-tenant isolation inside a single attached memory handle is out of scope for v1. Developers with multi-tenant requirements will instantiate one memory handle per tenant.
- Custom storage backends for ADK agents are out of scope for v1. Developers who want Postgres / pgvector storage will implement the storage adapter contract themselves and pass it through to the attachment function via a follow-up extension.
- The follow-up benchmark package (`neuromem-bench`, separate feature) will consume this package as its primary agent backend. That package's requirements are documented separately.
