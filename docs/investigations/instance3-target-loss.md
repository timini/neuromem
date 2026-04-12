# Investigation: LongMemEval Instance 3 — "Where did I redeem the coupon?" → Target

**Status**: Reproduced. Root cause identified. **Fix A applied and verified — Instance 3 now scores 1.0 against the reproducer.** Fixes F and G' still to do (tracked separately).

**Reproduce with**: `GOOGLE_API_KEY=... uv run python packages/neuromem-bench/scripts/repro_instance3_target.py`

Runs in ~3 minutes and costs well under $0.01.

---

## TL;DR

On LongMemEval Instance 3 (`question_id=51a45a95`), `NeuromemAgent` answers *correctly* that the user saved $5 on coffee creamer with a coupon, but drops the store name "Target" — which is the actual gold answer.

The headline hypothesis in `docs/benchmarks/RESULTS.md` was that `extract_tags` loses named entities. **That turned out to be wrong.** The real failure is upstream:

> **`generate_summary` compresses named entities out of long turns before the tagging stage ever sees them.** The clustering then has no way to associate the entity-rich memories with the entity-stripped ones, because they share no surviving tags.

The reproducer makes this visible turn-by-turn. Example: assistant turn 5 (1023 raw chars, mentions "Target" once) gets compressed to a 198-char summary where Target has been squeezed out in favour of abstract nouns ("email deals", "coupon organisation"). By the time `extract_tags` runs, there's nothing left to extract.

---

## The data

Instance 3's `answer_session_ids` is `['answer_d61669c7']` — a single 12-turn session. Of those 12 turns, 5 mention Target/Cartwheel (turns 2, 3, 6, 7, 8) and 1 mentions the coffee-creamer-coupon event (turn 4). The evidence is scattered:

| Turn | Role | Content excerpt | Mentions |
|---|---|---|---|
| 2 | user | "...Cartwheel app from Target..." | Target, Cartwheel |
| 3 | assistant | "...saving money...at Target..." | Target |
| 4 | **user** | **"I actually redeemed a $5 coupon on coffee creamer last Sunday..."** | *(no entity — just "email inbox")* |
| 5 | assistant | "...like Target, send exclusive coupons..." | Target |
| 6 | user | "I shop at Target pretty frequently..." | Target, Cartwheel |
| 7 | assistant | "You're a regular Target shopper!" | Target, Cartwheel |

The key insight: **turn 4 — the single turn that contains the coupon-on-creamer event — never mentions Target.** The user assumed (reasonably) that the context carries over. A human reader threads it together from surrounding turns. The cognitive loop does not.

---

## What the reproducer shows

The reproducer runs two pipelines.

### Pipeline 1 — per-turn summary + tag inspection

For each turn, it prints the raw text, the `generate_summary` output, and the `extract_tags` output, with markers showing whether "Target" / "Cartwheel" survive each stage.

**Representative output** (edited for brevity):

```
--- turn 4 [user] ---
RAW  (136 chars) [—]:
     I actually redeemed a $5 coupon on coffee creamer last Sunday,
     which was a nice surprise since I didn't know I had it in my
     email inbox.
SUM  (87 chars) [—]:
     The author unexpectedly saved $5 on coffee creamer using a
     coupon found in their email.
TAGS [Target/Cartwheel NOT in tags]:
     ['Coupon', 'savings', 'coffee creamer']
```

Turn 4 has no entity to preserve — this one's not the summariser's fault.

```
--- turn 5 [assistant] ---
RAW  (1023 chars) [Target]:
     ...Many retailers, like Target, send exclusive coupons and
     promotions to their email subscribers...
SUM  (198 chars) [—]:          ← Target is gone
     Finding a surprise coupon for a needed item like coffee creamer
     is exciting and serves as a reminder to regularly check emails
     for deals, which can be organized by creating a dedicated email
     folder.
TAGS [Target/Cartwheel NOT in tags]:
     ['Surprise coupon', 'needed item', 'coffee creamer', 'deals',
      'email folder']
```

**This is the bug.** Turn 5 directly replies to turn 4 ("your coffee creamer coupon reminds me, retailers like Target..."), the one place in the session where "Target" is associated with "coffee creamer / coupon / email" in the same 1000 characters. The summariser's 1-2 sentence compression discards Target because the prompt asks for *one or two sentences* with no instruction to preserve named entities. Fewer tokens = drop the specific, keep the abstract.

By contrast, turns 2, 6, 7 preserve Target all the way through — but those turns don't mention coffee creamer or the coupon event, so at retrieval time the embedding of the benchmark question ("Where did I redeem a $5 coupon on coffee creamer?") doesn't land near them.

### Pipeline 2 — end-to-end cognitive loop

The reproducer then spins up a real `NeuroMemory` with `GeminiLLMProvider` + `GeminiEmbeddingProvider`, enqueues all 12 turns, forces a dream cycle, runs `ContextHelper.build_prompt_context(question)`, and prints the rendered tree.

The tree **does** contain "Target" once — in memory `mem_b1ddf85…` under the path:

```
📁 stuff
└── 📁 belongings
    └── 📁 containers
        └── 📁 storage
            └── 📁 plastic sleeves
                └── 📄 mem_b1ddf85…: "Combine the Target Cartwheel app with a well-organized coupon and receipt binder…"
```

Note the path: `stuff → belongings → containers → storage → plastic sleeves`. The memory that preserves "Target" has been clustered on the **coupon-binder-organisation** axis (because that summary leads with the binder-plus-app suggestion), not the **shopping-at-Target** axis. At retrieval time it surfaces via the wrong centroid, so the answer LLM doesn't recognise it as relevant to the question.

Meanwhile, the three memories the retrieval ranker *does* surface near the top (all descendants of `discount → voucher → coupon`) are the ones that had Target stripped out. The LLM reads them, sees "coffee creamer", "coupon", "$5", "email" — and correctly produces the verbose-but-incomplete answer we saw in the benchmark:

```
Pred: You unexpectedly saved $5 on coffee creamer using a coupon you
      found. I don't have specific information about where you
      redeemed it.
```

---

## Root cause — not one bug, two interacting ones

### 1. Summary-stage entity loss (primary)

`GeminiLLMProvider.generate_summary` uses this prompt (`packages/neuromem-gemini/src/neuromem_gemini/llm.py:149`):

```python
"Summarise the following text in one or two sentences. "
"Respond with ONLY the summary — no preamble, no markdown, "
"no quotation marks.\n\n"
f"Text: {raw_text}"
```

No instruction to preserve proper nouns / entities / specific brands. When the LLM is squeezed to 1–2 sentences from 1000+ chars, the compression strategy it picks is "keep the high-level concept, drop the specifics". This is the biggest lever.

### 2. Tag-stage entity de-prioritisation (secondary)

`extract_tags` uses this prompt (`packages/neuromem-gemini/src/neuromem_gemini/llm.py:169`):

```python
"Extract between 3 and 5 key concepts from the following text "
"and return them as a comma-separated list. "
"Exclude stopwords, articles, prepositions, and generic words "
'(e.g. "the", "is", "a", "of", "to", "for"). '
```

Negative ("exclude…") but no positive instruction. When a summary mentions both a proper noun and an abstract concept, the tagger is just as happy to keep the concept and drop the noun. In the reproducer output, tag lists like `['Coupon', 'savings', 'coffee creamer']` and `['Maximize savings', 'Cartwheel app', 'coupon binder', 'organized sections']` show this — the tagger happily picks specific-like terms ("Cartwheel app") when they're the topic, but drops entities that appear incidentally.

### 3. Clustering can't bridge summaries that don't share surface tags (knock-on)

The dream cycle clusters on tag-embedding similarity. Memories about the "coupon-on-creamer event" cluster tightly together (they all share `coupon`, `savings`, `creamer`). Memories about Target as a store cluster separately (they share `Target`, `Cartwheel`, `household goods`). Nothing links them. No single turn in this session survives summarisation carrying both "Target" *and* "coffee creamer" in its tag list, so there's no shared tag to pull the two clusters into the same subgraph.

---

## Fixes, ranked by cost/value

### A. Add "preserve named entities" to the summary prompt *(cheap, likely high lift)*

One prompt edit in `generate_summary`:

```python
"Summarise the following text in one or two sentences. "
"Preserve all proper nouns, brand names, places, people, "   # NEW
"and specific numbers as they appear in the text. "          # NEW
"Respond with ONLY the summary — no preamble, no markdown, "
"no quotation marks.\n\n"
```

This is the single lowest-risk, highest-expected-value change. The reproducer is the correctness test: run before, run after, diff the turn-5 summary. If "Target" starts surviving, the downstream tag + cluster + retrieve path has a fighting chance.

### B. Add "preserve proper nouns" to the tag prompt *(cheap, smaller lift)*

Same shape of prompt edit in `extract_tags` / `extract_tags_batch`. Useful as a backstop even after A — in case the summariser still drops an entity occasionally, the tagger will be biased toward keeping any that survive.

### C. Two-phase summary: factual + abstractive *(medium cost, high lift, a design change)*

Split `generate_summary` into two calls:
1. An "entity-preserving factual" summary (1–2 sentences, must name every proper noun / number in the source).
2. An abstractive summary (1–2 sentences, free to generalise).

Store both. Extract tags from the factual summary. Render the factual summary in the context tree. Feed the abstractive one to the clusterer for higher-level grouping.

This would double the summariser cost on the hot path. Not cheap. But it directly addresses the "compression forces a choice between specific and abstract" problem that A and B only bias against.

### D. Co-mention cross-links in the graph *(expensive, graph-schema change)*

When two memories within N turns of each other survive with tags that *co-occur* in the raw text, add an edge between them even if they don't share a tag post-extraction. Effectively: "turn 4 and turn 5 are adjacent and their raw texts share the word `coupon` — link them." This is a whole-design change and should wait until after A/B/C have been tried.

### E. LLM-rerank retrieved memories *(expensive per-query)*

At answer time, pull a larger top-k (say 20 instead of 5), feed them to an LLM with the question, ask "which of these mention the same real-world entity as the question implies?", then re-rank. Standard retrieval-rerank pattern. Expensive and doesn't help the many-hop retrieval failure the instance actually has, so probably not worth it here.

---

## Recommended next step

1. ✅ Apply fix A (one-line prompt edit). **Done** — see "Result of fix A" below.
2. Add a real ADK-backed benchmark agent using `neuromem_adk.enable_memory` (fix F) so the LLM can call `retrieve_memories` to drill into `raw_content` when the summary tree isn't enough.
3. Render named entities (stored per memory) alongside tags in the ContextHelper tree (fix G'), so one-shot agents also get entity visibility without needing tool calls.

## Result of fix A

Applied a one-line addition to both the `generate_summary` and `extract_tags` prompts in `packages/neuromem-gemini/src/neuromem_gemini/llm.py`:

> Summary prompt: *"Preserve all proper nouns — brand names, places, people, organisations — and specific numeric values exactly as they appear in the text."*

> Tag prompt: *"When the text mentions proper nouns — brand names, places, people, organisations — include them as concepts; prefer specific entities over the abstract category they belong to."*

(Same instruction folded into the batch-tag prompt too.)

### Before (reproducer turn 5, pre-fix)

```
SUM: Finding a surprise coupon for a needed item like coffee creamer
     is exciting and serves as a reminder to regularly check emails
     for deals...                           ← Target dropped

TAGS: ['Surprise coupon', 'needed item', 'coffee creamer', 'deals',
       'email folder']                       ← Target/Cartwheel NOT in tags

Final answer:
   "You unexpectedly saved $5 on coffee creamer using a coupon you
    found. I don't have specific information about where you redeemed it."
→ Contains 'Target'? False
```

### After (reproducer turn 5, post-fix)

```
SUM: Check your email for exclusive coupons from retailers like Target
     to catch surprise deals...             ← Target preserved

TAGS: ['Email', 'Coupons', 'Retailers', 'Target', 'Savings']   ← in tags

Final answer:
   "You redeemed a $5 coupon on coffee creamer at Target."
→ Contains 'Target'? True
```

Every Target-mentioning turn (2, 3, 5, 6, 7) now survives summarisation with the entity intact. In the rendered ContextHelper tree, the `$5 coupon / coffee creamer` subtree now sits adjacent to `Target shopper` memories, so the answer LLM can join them.

Full workspace test suite still green (282 passed). No prompt-string assertions to update.

---

## References

- Reproducer script: `packages/neuromem-bench/scripts/repro_instance3_target.py`
- Original failing run: `docs/benchmarks/longmemeval-s-neuromem-n3.llm_judge.jsonl` (instance 3, score 0.0)
- Summary prompt: `packages/neuromem-gemini/src/neuromem_gemini/llm.py:142`
- Tag prompt: `packages/neuromem-gemini/src/neuromem_gemini/llm.py:162`
- Batch tag prompt: `packages/neuromem-gemini/src/neuromem_gemini/llm.py:184`
