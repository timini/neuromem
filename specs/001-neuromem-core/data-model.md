# Phase 1 Data Model: neuromem Core Library

**Feature**: 001-neuromem-core
**Source**: spec.md §Key Entities, §Data Model
**Purpose**: Binding definitions for every entity the library persists or manipulates, including field types, invariants, validation rules, and state transitions.

This document is the authoritative source for the SQLite schema and for what a `StorageAdapter` implementation must preserve. Any divergence between this file and `src/neuromem/storage/sqlite.py` is a bug.

---

## Entity: Memory

A single atomic unit of agent experience. Created by `NeuroMemory.enqueue()` and consumed by the dreaming pipeline, then optionally archived.

### Fields

| Field | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| `id` | `str` (UUID v4) | No | generated | Primary key. Opaque to callers. |
| `raw_content` | `str` (UTF-8) | No | — | The original text enqueued by the agent. Never mutated. |
| `summary` | `str` (UTF-8) | Yes | `NULL` on insert if no summary yet | LLM-generated 1–2 sentence episodic summary. Populated synchronously in `enqueue()` per Resolved Design Decision #3. |
| `status` | `str` enum | No | `'inbox'` | One of `'inbox' \| 'dreaming' \| 'consolidated' \| 'archived'`. |
| `access_weight` | `float` | No | `1.0` | In `[0.0, 1.0]` conceptually; stored as `REAL`. Reset to `1.0` on LTP (retrieval). Decayed exponentially in every dreaming cycle. |
| `last_accessed` | `int` (Unix epoch seconds) | Yes | `NULL` until first access | Updated on LTP. Used as the `t` in the decay formula: `W_new = W_old * exp(-λ * (now - last_accessed))`. |
| `created_at` | `int` (Unix epoch seconds) | No | `time.time()` at insert | Immutable. |
| `metadata` | `str` (JSON) | Yes | `NULL` | Freeform JSON-serialisable dict from the caller. Validated before insert by `json.dumps(metadata)`. |

### Invariants

1. **I-M1**: `id` is globally unique within a single storage backend.
2. **I-M2**: `raw_content` is never empty and never `None`. `enqueue("")` MUST raise `ValueError`.
3. **I-M3**: `status` transitions follow the state machine below. No other transitions are legal.
4. **I-M4**: Once `status = 'archived'`, `raw_content` and `summary` MUST remain intact — archival is never destructive (FR-015).
5. **I-M5**: `access_weight` is reset to exactly `1.0` (not "spiked to some higher value") on every successful `retrieve_memories()` call for memories in `'consolidated'` state (FR-021).
6. **I-M6**: `metadata`, if provided, MUST round-trip through `json.dumps`/`json.loads` without error. Non-JSON-serialisable metadata raises `ValueError` before any DB write.

### State Machine

```
             ┌──────────────────── (rollback on dream failure) ─────────┐
             │                                                           │
             ▼                                                           │
      ┌──────────┐   threshold met    ┌──────────┐   success    ┌──────────────┐
─────▶│  inbox   │ ─────────────────▶ │ dreaming │ ───────────▶ │ consolidated │
      └──────────┘                    └──────────┘              └───────┬──────┘
            ▲                                                           │
            │                                                           │ decay
            │                                                           │ below
            │                                                           │ threshold
            │                                                           ▼
            │                                                    ┌──────────┐
            └────────────────── never returns ─────────────────  │ archived │
                                                                 └──────────┘
```

**Valid transitions** (enforced by `StorageAdapter.update_memory_status`):

| From | To | Trigger | Who |
|---|---|---|---|
| *(new)* | `inbox` | `insert_memory()` | `NeuroMemory.enqueue()` |
| `inbox` | `dreaming` | Double-buffer flip at the start of `_run_dream_cycle` | Background thread |
| `dreaming` | `consolidated` | Dreaming cycle completes successfully (steps 3–11 of Phase B workflow) | Background thread |
| `dreaming` | `inbox` | Dreaming cycle raises — rollback (FR-029) | Background thread exception handler |
| `consolidated` | `archived` | `apply_decay_and_archive()` — `access_weight < archive_threshold` | End of dreaming cycle (Phase C, FR-014) |

**Invalid transitions** (MUST raise or silently refuse):

- `inbox → consolidated` (skipping the dreaming phase)
- `inbox → archived`
- `consolidated → dreaming` (archived memories cannot be re-dreamed; LTP only spikes `access_weight`)
- `archived → consolidated` (no resurrection in v1 per User Story 3 acceptance scenario #2)
- `archived → dreaming`

### Validation rules

- **V-M1**: `enqueue(raw_text)` with `raw_text == ""` or `raw_text is None` → `ValueError`.
- **V-M2**: `enqueue(raw_text, metadata=obj)` where `obj` is not JSON-serialisable → `ValueError` before DB write.
- **V-M3**: `retrieve_memories(ids)` with missing IDs silently skips them (no exception, no empty dict).
- **V-M4**: `retrieve_memories(ids)` on an `inbox`-status memory returns the content but does NOT trigger LTP (spec Edge Case: retrieve before consolidation).
- **V-M5**: `retrieve_memories(ids)` on an `archived`-status memory returns the content and does NOT resurrect the status.

---

## Entity: Node

A concept node in the knowledge graph. Represents either a leaf tag extracted from a memory summary or a centroid parent category generated during agglomerative clustering.

### Fields

| Field | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| `id` | `str` (UUID v4) | No | generated | Primary key. |
| `label` | `str` | No | — | Human-readable concept name (e.g., `"SQLite"`, `"Databases"`). For leaf tags, the string returned by `LLMProvider.extract_tags()`. For centroids, the string returned by `LLMProvider.generate_category_name()`. |
| `embedding` | `np.ndarray` (1-D, float32) | No | — | Serialised as raw float32 bytes in a `BLOB` column in the SQLite adapter via `arr.astype(np.float32).tobytes()`. Reconstructed on read with `np.frombuffer(blob, dtype=np.float32)`. Contracts accept `np.ndarray` or `list[float]` inputs and normalise internally. For leaf tags: the vector from `EmbeddingProvider.get_embeddings([label])[0]`. For centroids: `compute_centroid(member_embeddings)`. |
| `embedding_dim` | `int` | No | — | Companion integer recording `len(embedding)`. Used as a sanity check on read (`len(blob) // 4 == embedding_dim`). Corruption raises `StorageError`. |
| `is_centroid` | `bool` | No | `False` | `True` iff this node was synthesised by the clustering loop. `False` for leaf tag nodes extracted directly from memories. |

### Invariants

1. **I-N1**: `id` is globally unique within a single storage backend.
2. **I-N2**: `label` is non-empty. For centroids, `label` is a single word (no spaces) per spec §Provider Interfaces. If the LLM returns multi-word output, `NeuroMemory` takes the first word and logs a `WARNING`.
3. **I-N3**: `embedding` has identical length to every other node's embedding in the same storage backend. Mixing dimensions is undefined behaviour — the caller is responsible for using a single `EmbeddingProvider` for the lifetime of a storage backend. The adapter validates against the *first* dimension it observes and raises `ValueError` on any subsequent insert with a different length.
4. **I-N4**: Leaf tag nodes (`is_centroid=False`) MUST have at least one `has_tag` edge connecting them to a memory. Centroid nodes (`is_centroid=True`) MUST have at least one `child_of` edge pointing to a member node.
5. **I-N5**: Centroid nodes are never deleted when their members are archived — they persist as long as at least one `child_of` edge points to a non-archived member. Orphan centroid cleanup is a Phase B sub-step (Step 10) that runs inside `apply_decay_and_archive`.

### Validation rules

- **V-N1**: `upsert_node(node_id, label, embedding, is_centroid)` with `embedding` having a different length than any existing node → `ValueError` in v1. (Future: the adapter may auto-resize or refuse based on a configuration flag.)
- **V-N2**: `upsert_node` with `embedding = []` → `ValueError`.
- **V-N3**: `SQLiteAdapter` MUST log a `WARNING` via `neuromem` logger if an incoming embedding has more than **4096** dimensions (spec Assumption).

---

## Entity: Edge

A directional weighted relationship between two graph entities. Connects node-to-node (taxonomy hierarchy) or node-to-memory (memory tagging).

### Fields

| Field | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| `id` | `int` (auto-increment) | No | generated | Primary key. Internal only — callers never reference by edge id. |
| `source_id` | `str` | No | — | Foreign reference. For `child_of`: centroid node id. For `has_tag`: memory id. |
| `target_id` | `str` | No | — | Foreign reference. For `child_of`: child node id. For `has_tag`: tag node id. |
| `weight` | `float` | No | `1.0` | For `child_of`: the cosine similarity that triggered the merge. For `has_tag`: `1.0` (tag membership is binary). |
| `relationship` | `str` enum | No | — | One of `'has_tag' \| 'child_of'`. |

### Invariants

1. **I-E1**: The composite `(source_id, target_id, relationship)` is unique. The SQLite adapter enforces this with a `UNIQUE` constraint; the idempotent `insert_edge` behaviour ignores duplicates silently.
2. **I-E2**: `relationship` is always exactly one of `'has_tag' | 'child_of'` in v1. Unknown relationship types raise `ValueError`.
3. **I-E3**: `weight` is in `[0.0, 1.0]` for cosine-derived edges. No strict validation — the caller is responsible for passing sensible values.
4. **I-E4**: Edges connected to an archived memory are removed from traversal queries by the `SQLiteAdapter.get_subgraph()` implementation (it filters by joined `memories.status != 'archived'`). The edges themselves may be deleted eagerly by `remove_edges_for_memory()` at archival time or left in the table — both are valid adapter strategies.

### Validation rules

- **V-E1**: `insert_edge(source_id, target_id, relationship='unknown')` → `ValueError`.
- **V-E2**: `insert_edge` with `source_id == target_id` → allowed (self-loops permitted but pointless).
- **V-E3**: `insert_edge` with non-existent source or target is NOT validated at insert time in v1 (SQLite has no foreign-key enforcement by default and we don't enable it for simplicity). Traversal code handles missing nodes gracefully.

---

## Relationships (ER diagram, textual)

```
┌───────────────┐                                         ┌──────────────┐
│    Memory     │                                         │     Node     │
│               │                                         │              │
│ id (PK)       │                                         │ id (PK)      │
│ raw_content   │                                         │ label        │
│ summary       │       ┌───────────────────┐             │ embedding    │
│ status        │       │       Edge        │             │ is_centroid  │
│ access_weight │       │                   │             │              │
│ last_accessed │       │ id (PK)           │             └──────────────┘
│ created_at    │◀──────│ source_id (FK)    │                    ▲
│ metadata      │       │ target_id (FK)    │────────────────────┤
└───────────────┘       │ weight            │                    │
                        │ relationship      │  has_tag: memory → node
                        └───────────────────┘  child_of: centroid → node
```

**Edge semantics**:

- `Memory --has_tag--> Node`: memory is tagged with the concept. Source = memory.id, target = node.id, weight = 1.0.
- `Node --child_of--> Node`: the source (always a centroid) is the parent of the target. Source = centroid.id, target = child.id, weight = cosine similarity at merge time.

A node can be both a parent (via outgoing `child_of` edges) and a child (via incoming `child_of` edges from a grand-parent centroid). This is how the clustering tree stacks multiple levels deep (Databases → Programming → Technology).

---

## Canonical SQLite schema

Reproduced from spec.md §Data Model. The `SQLiteAdapter` MUST create exactly this schema on first connect.

```sql
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    raw_content   TEXT NOT NULL,
    summary       TEXT,
    status        TEXT NOT NULL DEFAULT 'inbox'
                   CHECK (status IN ('inbox', 'dreaming', 'consolidated', 'archived')),
    access_weight REAL NOT NULL DEFAULT 1.0,
    last_accessed INTEGER,
    created_at    INTEGER NOT NULL,
    metadata      TEXT
);

CREATE TABLE IF NOT EXISTS nodes (
    id            TEXT PRIMARY KEY,
    label         TEXT NOT NULL,
    embedding     BLOB NOT NULL,           -- raw float32 bytes
    embedding_dim INTEGER NOT NULL,         -- len(embedding) — corruption check
    is_centroid   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    TEXT NOT NULL,
    target_id    TEXT NOT NULL,
    weight       REAL NOT NULL DEFAULT 1.0,
    relationship TEXT NOT NULL
                  CHECK (relationship IN ('has_tag', 'child_of')),
    UNIQUE (source_id, target_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_edges_source    ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target    ON edges(target_id);
```

### Notes on the schema additions vs spec.md

Two `CHECK` constraints were added here that spec.md §Data Model did not explicitly call out:

1. `memories.status CHECK (status IN ('inbox', 'dreaming', 'consolidated', 'archived'))` — enforces the state machine at the DB level as a defence-in-depth against buggy adapters.
2. `edges.relationship CHECK (relationship IN ('has_tag', 'child_of'))` — enforces the relationship enum at the DB level.

These are additions, not contradictions, and they are safe because both value sets are closed in v1. If a future release adds a new status or relationship kind, the `CHECK` constraint is modified in the same migration that introduces the new value.

---

## Counts, indexes, and performance notes

- `idx_memories_status` makes the "how many inbox rows?" count in `enqueue()` an O(log n) operation (SC-002's 50 ms budget).
- `idx_edges_source` and `idx_edges_target` make subgraph traversal O(edges touching frontier) instead of O(all edges).
- No index on `nodes.embedding` — cosine similarity search in the SQLite adapter is a full table scan (FR-024). This is the performance ceiling the 10,000-node / 500 ms SC-003 target was chosen against, and is exactly the bottleneck that motivates a future `PostgresAdapter` with `pgvector`.
