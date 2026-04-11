# Research — neuromem-adk

Resolved design decisions for the `neuromem-adk` package. Each decision is recorded with the options considered, the chosen path, the rationale, and the alternatives explicitly rejected. Source material for every decision is cited.

## 1. Extension-point strategy: hybrid, not pure tools

**Decision**: Use four distinct ADK extension points, each covering one behaviour that no single other approach covers:

1. `before_model_callback` — passive context injection (prepend the neuromem ASCII tree to the system prompt before each model call).
2. `after_agent_callback` — automatic turn capture (user message + assistant response flow into `NeuroMemory.enqueue`).
3. Function-tool registration — LLM-driven active search (`search_memory`, `retrieve_memories` exposed as callable tools).
4. `NeuromemMemoryService(BaseMemoryService)` — framework-native memory slot, handles session-end consolidation via `add_session_to_memory`.

**Rationale**: The cognitive loop has four different behaviours that each fire at a different point in the ADK request lifecycle. Trying to shoehorn all four into a single extension point either loses behaviours (mem0's tool-only approach loses passive injection and automatic capture) or forces an unnatural wrapping (e.g., wrapping the whole `Agent.run_async` method would fight ADK's native dispatch flow). The hybrid approach maps each behaviour onto its natural ADK hook, which is exactly what the callback and memory-service abstractions were designed for.

**Alternatives considered**:

- **Pure tool-registration, like mem0** — reject. Loses passive injection (the single biggest differentiator from mem0) and automatic turn capture (the single most important invariant for a memory library).
- **Pure `BaseMemoryService` subclassing** — reject. Covers session-end hooks but provides no place to inject context before a model call or capture turns as they happen. `BaseMemoryService` is a framework-level slot, not a per-turn hook.
- **Pure `before_model_callback` and `after_model_callback` without tools** — reject. Covers passive injection and capture but loses the "LLM can explicitly decide to search" path, which is a real capability (spec US4) and free to add via function-tool registration.
- **Subclass `Agent` directly** — reject. Fragile across ADK versions; the callback and tool APIs are the framework's designated public extension points and are more stable than the class hierarchy.

**Sources**:

- Google ADK docs, callback section: [https://adk.dev/callbacks/](https://adk.dev/callbacks/) — documents the six callback hooks and return-value override semantics.
- Google ADK docs, memory section: [https://adk.dev/sessions/memory/](https://adk.dev/sessions/memory/) — documents `BaseMemoryService` ABC and its two abstract methods.
- mem0 ADK integration: [https://docs.mem0.ai/integrations/google-ai-adk](https://docs.mem0.ai/integrations/google-ai-adk) — shows the tool-only approach and its limits.

## 2. Default provider pair: auto-instantiate Gemini from `neuromem-gemini`

**Decision**: `enable_memory(agent, db_path=...)` with no provider arguments auto-instantiates `GeminiLLMProvider` and `GeminiEmbeddingProvider` from the `neuromem-gemini` sibling, reading `GOOGLE_API_KEY` from the environment. Advanced users can override with `enable_memory(agent, db_path=..., llm=..., embedder=...)`.

**Rationale**:

1. **90% case is "use defaults and go"**. Forcing every user to write three lines of provider wiring every time defeats the one-line-wire-up promise (US1).
2. **The default matches ADK's own assumption** — ADK uses `google-genai` under the hood with `GOOGLE_API_KEY`. Sharing the same credentials between ADK and neuromem means users configure auth once and everything works.
3. **`neuromem-gemini` already exists** and is on the same release cadence, so taking it as a runtime dep of `neuromem-adk` is free. It's not a new component we'd have to build.
4. **Override path preserved** for the 10% case (local embedders for testing, different Gemini model for cost, future Anthropic/OpenAI swap-ins).

**Alternatives considered**:

- **Require explicit providers always** — reject. Clean separation (no runtime dep on neuromem-gemini) but every user writes the same 3 lines of boilerplate. Net UX cost > architectural purity benefit.
- **Ship a "mock" default that doesn't hit any real LLM** — reject. The resulting agent wouldn't actually work without manual configuration, which fails US1's "works out of the box" promise.

**Sources**:

- User confirmation during plan review (see `partitioned-gliding-goblet.md`): "Auto-default to Gemini".

## 3. Tool binding: `functools.partial`

**Decision**: Bind `search_memory` and `retrieve_memories` to the live `NeuroMemory` instance using `functools.partial(search_memory, system=the_memory)`. The resulting partial is passed to `Agent(tools=[...])`.

**Rationale**:

1. **Python 3.10+ `inspect.signature` correctly walks `functools.partial`** — pre-bound keyword arguments are excluded from the reported signature, so ADK's schema generator sees only `{query, top_k, depth}` for `search_memory` and `{memory_ids}` for `retrieve_memories`. The LLM never sees the internal `system` plumbing.
2. **Stdlib only** — no new dependency, consistent with Principle II (even though Principle II only binds neuromem-core, the same minimalism is virtuous in sibling packages).
3. **Preserves the source function** — debugging and tracebacks still point at `neuromem.tools.search_memory`, not a one-off wrapper.
4. **Matches the documented "intended wrapper pattern"** — the docstring of `neuromem.tools.search_memory` explicitly says: "Downstream wrappers usually ``functools.partial`` the system into the tool to hide this from the agent."

**Alternatives considered**:

- **Wrapper functions defined inside `enable_memory`** — accepted as a fallback if ADK's schema generator turns out to trip on partials in practice. Slightly more code, slightly noisier tracebacks, but same functionality.
- **Class with bound methods** (`NeuromemAdkSession(memory).search_memory(query)`) — reject. Adds a layer of OO indirection users don't need. Overkill for a two-tool wire-up.

**Sources**:

- `packages/neuromem-core/src/neuromem/tools.py` module docstring — cites the `functools.partial` pattern as the intended wrapper mechanism.
- User confirmation during plan review: "ok partial then".

## 4. Test shape: unit tests + one integration smoke test

**Decision**: Follow the `neuromem-gemini` template exactly:

- **Unit tests**: `test_enable.py`, `test_callbacks.py`, `test_memory_service.py` — fast, deterministic, no network. Use a mocked ADK runner fixture to verify each wiring component in isolation. Run on every PR via the default `uv run pytest` invocation.
- **Integration test**: `test_adk_integration.py` marked `@pytest.mark.integration`. Builds a real `google.adk.agents.Agent`, attaches memory via `enable_memory`, runs a 3–4 turn conversation against real Gemini. Asserts (a) memory was captured and (b) a later-turn question referencing an earlier fact gets answered via injected context.

**Rationale**:

1. **Fast unit tests are the daily development feedback loop.** They must not hit the network or require credentials, or developers will start commenting them out.
2. **One integration test is enough** to prove the end-to-end wiring actually works against real ADK + real Gemini. More is better in theory but each run costs API tokens. The existing `neuromem-gemini` pattern has already proven that one integration test is sufficient for confidence.
3. **Consistent with Constitution Principle V** — "Test-First with Mocks (NON-NEGOTIABLE)". Unit tests with mocks are mandatory; integration tests are additional.
4. **Keeps the default CI fast and free** — integration tests are excluded from the default run via the workspace `-m "not integration"` addopt.

**Alternatives considered**:

- **Only isolated unit tests, no integration** — reject. Mocks can drift from reality; one real run per CI cycle catches mock-vs-real divergence.
- **Full multi-scenario integration suite** — reject for v0.1. Spec SC-004 budgets integration-test runs at <30 seconds and <$0.01 per invocation. A multi-scenario suite would blow that.

**Sources**:

- `packages/neuromem-gemini/tests/` — the template this pattern copies.
- User confirmation during plan review: "both" (implicit via non-objection to the recommendation).

## 5. Workflow: spec-kit formal pass, not informal

**Decision**: Ship this package through the spec-kit workflow (`/speckit.specify → /speckit.plan → /speckit.tasks → /speckit.implement`) landing at `specs/003-neuromem-adk/`.

**Rationale**:

1. **It's a real user-facing feature** with multiple distinct user stories (6 stories across 3 priority tiers), not a thin helper.
2. **It sets the pattern for future framework wrappers** — whatever process this package follows will be copied for `neuromem-anthropic`, `neuromem-langchain`, etc. Doing it the formal way once creates a reusable template.
3. **The Constitution expects new user-facing features to go through the process** (see `.specify/memory/constitution.md` §Workflow).
4. **`neuromem-bench` will NOT follow this path** — the follow-up benchmark harness is experimental ("learn what a benchmark run looks like") and will use an informal feature branch. This plan covers only `neuromem-adk`.

**Alternatives considered**:

- **Skip spec-kit, just ship on a feature branch** — reject for this package. Spec-kit's overhead is justified by the size (6 user stories, 15 functional requirements) and the reusability (template for future wrappers).

**Sources**:

- User confirmation during plan review: "run /speckit.specify".
- Constitution v2.0.0, §Workflow.

## 6. Package dep graph and resolution

**Decision**: `packages/neuromem-adk/pyproject.toml` declares:

```toml
[project.dependencies]
neuromem-core       # workspace source
neuromem-gemini     # workspace source
google-adk >= 1.29
numpy >= 1.26
```

Workspace source resolution via `[tool.uv.sources]`:

```toml
[tool.uv.sources]
neuromem-core   = { workspace = true }
neuromem-gemini = { workspace = true }
```

**Rationale**: This is the exact same pattern `neuromem-gemini` already uses to reference `neuromem-core` as a workspace dep. `uv sync` resolves both as editable installs from local paths during dev, and `uv build` records the names in the wheel metadata so PyPI installs resolve from PyPI.

**Alternatives considered**:

- **Pin to specific versions of the workspace siblings** — reject. Locking to a specific version in a monorepo-with-workspaces breaks the "change all packages together" semantics. Workspace source declarations correctly leave version pinning to the lockfile.

**Sources**:

- `packages/neuromem-gemini/pyproject.toml` — the template this pattern copies.
- uv workspace docs: [https://docs.astral.sh/uv/concepts/projects/workspaces/](https://docs.astral.sh/uv/concepts/projects/workspaces/).

## 7. Credential resolution: env var + `.env` file, no `python-dotenv`

**Decision**: Reuse the credential resolver pattern from `neuromem-gemini/tests/conftest.py` — first check `os.environ["GOOGLE_API_KEY"]`, then fall back to parsing a repo-root `.env` file with a 10-line stdlib parser. No `python-dotenv` dependency.

**Rationale**:

1. **Consistency** — matches the existing `neuromem-gemini` pattern. A developer who has already set up `GEMINI_API_KEY` in the repo's `.env` should not have to configure a second variable for the ADK integration.
2. **No extra dependency** — `python-dotenv` is a small and well-respected package but it's still a transitive dep with its own release cycle. Stdlib parsing is 10 lines and doesn't need maintenance.
3. **`GOOGLE_API_KEY` is the canonical ADK variable name** — this is what both `google-adk` and `google-genai` read by default. We do NOT use `GEMINI_API_KEY` because the existing neuromem-gemini tests already read from both (env + `.env`) and the user's workflow points at `GOOGLE_API_KEY` for the ADK side specifically.

**Alternatives considered**:

- **`python-dotenv` package** — reject. 10 lines of stdlib parsing saves a dependency.
- **Require env var only, no `.env` fallback** — reject. Developers working against a local `.env` file during development would have to export variables manually.

**Sources**:

- `packages/neuromem-gemini/tests/conftest.py::_parse_dotenv` — the 10-line parser this reuses.

## 8. CI strategy: extend existing integration workflow

**Decision**: Add a new job `adk-integration` to the existing `.github/workflows/integration.yml` file alongside the existing `gemini` job. Same secret (`GEMINI_API_KEY` — mapped to `GOOGLE_API_KEY` for the ADK runtime), same cadence (nightly at 03:17 UTC + push to main with path filter + manual dispatch), same timeout policy, same `--no-cov` flag.

**Rationale**:

1. **Single source of truth** for integration test configuration. A new file per sibling package would scatter credentials, schedules, and concurrency settings.
2. **Independent jobs, not matrix entries** — each job runs independently and failures don't mask each other. A Gemini API outage that breaks the gemini job won't cancel the ADK job.
3. **Shared path filter** updates the `paths` trigger to also include `packages/neuromem-adk/**`, so the ADK integration job runs when ADK code changes but not when only Gemini-provider code changes.

**Alternatives considered**:

- **New separate workflow file** — reject. Creates duplicate schedule, concurrency, and permissions config. Two jobs in one file is the simpler maintenance story.
- **Single combined job that runs both suites** — reject. Couples the two integration paths' failure modes; a broken ADK wiring would hide a broken Gemini wiring (or vice-versa).

**Sources**:

- `.github/workflows/integration.yml` (committed in PR #39) — the file this extends.
