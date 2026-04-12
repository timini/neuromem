# Benchmark results

Rolling leaderboard for `neuromem-bench` runs against published memory benchmarks. Each row is one agent × one benchmark × one split × one sample.

## LongMemEval_s (ICLR 2025, arXiv:2410.10813)

**3-instance sample**, scored with both `contains_match` (deterministic, strict substring) and `llm_judge` (Gemini 2.0 Flash as judge, matching LongMemEval's canonical GPT-4o-as-judge protocol).

### Summary

| Agent | Instances | `contains_match` | `llm_judge` | Total time | Notes |
|---|---|---|---|---|---|
| `NullAgent` (no memory) | 3 | 0.333 | 0.333 | 4s | Last-50-turn prompt stuffing |
| `NeuromemAgent` (real summary) | 3 | 0.333 | **0.667** | 5226s (~87 min) | Full cognitive loop |

**NeuromemAgent scores 2× NullAgent** on the LLM-judge metric — the cognitive loop retains facts from sessions too far back for a brute-force-context-window baseline.

### Per-instance detail

| # | Question | Gold | NullAgent pred | Null score | NeuromemAgent pred | Nm score |
|---|---|---|---|---|---|---|
| 1 | What degree did I graduate with? | Business Administration | "You mentioned you graduated with Business Administration." | **1.0** | "You have a business administration degree." | **1.0** |
| 2 | How long is my daily commute to work? | 45 minutes each way | "I do not have access to your personal information..." | **0.0** | "Your daily commute is 90 minutes." | **1.0**† |
| 3 | Where did I redeem a $5 coupon on coffee creamer? | Target | "I do not have access to that information..." | **0.0** | "You unexpectedly saved $5 on coffee creamer by redeeming a coupon you found." | **0.0** |

† LLM-judge accepted "90 minutes" as correct (45 min × 2 = round-trip total). The cognitive loop retained the fact; the LLM derived a technically-different-but-correct answer. The `contains_match` metric marked this 0.0; `llm_judge` correctly marked it 1.0.

### Observations

1. **Instance 2 is the money shot.** NullAgent refuses because the fact is outside its 50-turn window. NeuromemAgent retrieves it from the concept graph and answers correctly. This is exactly the use case neuromem exists for.
2. **Instance 3 is a real tuning opportunity.** NeuromemAgent has *partial* knowledge (it recalled the coupon redemption event) but lost the specific store name "Target" somewhere in the `extract_tags → cluster → retrieve` pipeline. Hypothesis: the `extract_tags` prompt doesn't prioritise named entities. Fix: prompt tuning.
3. **Metric choice matters enormously.** Going from `contains_match` (0.333) to `llm_judge` (0.667) doubled the apparent score. For any serious benchmark reporting, `llm_judge` should be the default. `contains_match` is useful as a cheap CI sanity check.

### Full results

- `longmemeval-s-null-n3.jsonl` — NullAgent `contains_match` scores
- `longmemeval-s-null-n3.llm_judge.jsonl` — NullAgent `llm_judge` scores
- `longmemeval-s-neuromem-n3.jsonl` — NeuromemAgent `contains_match` scores
- `longmemeval-s-neuromem-n3.llm_judge.jsonl` — NeuromemAgent `llm_judge` scores

### Runtime breakdown (NeuromemAgent)

| Instance | Wall time |
|---|---|
| 1 | 1856s (~31 min) |
| 2 | 1669s (~28 min) |
| 3 | 1698s (~28 min) |
| Total | **5226s (~87 min)** |

Per-instance cost driven by:
- ~160 `generate_summary` calls on the enqueue hot path (1 per turn)
- 1 batched `extract_tags` call (#45 fix — previously 160 serial calls)
- ~20-40 `generate_category_name` calls during agglomerative clustering (inherently serial — each merge depends on the previous)
- 1 final answer call with injected memory context

The ~87-min runtime for 3 instances is manageable for a baseline-measurement benchmark but becomes impractical at LongMemEval's full 500-instance scale (~240 hours). Sampling to 25-50 instances gives statistical signal within a few hours.

### Known follow-ups

1. **Prompt-tune `extract_tags`** to prioritise named entities (instance 3's "Target" loss). Currently the prompt says "exclude stopwords" but doesn't positively instruct the LLM to *preserve* proper nouns. Cheap fix, likely significant quality lift.
2. **Issue #42** — clustering quality (duplicate centroid names, recursive subtrees). Not on the benchmark critical path but affects graph readability.
3. **Batch `generate_category_name`** similar to how #45 batched `extract_tags`. Each merge step depends on prior merges so strict batching is impossible, but we could batch "initial centroid naming" where multiple independent clusters are named simultaneously. Would cut another ~50% of dream-cycle latency.
4. **Scale up the benchmark** — rerun at n=25 or n=50 once the prompt tuning lands, to get statistical confidence beyond a 3-instance anecdote.
