# Benchmark run: longmemeval / s

- **Agent**: `NeuromemAgent`
- **Metric**: `llm_judge`
- **Instances**: 60 (59 scored, 1 errored)
- **Mean score**: **0.746** (over 59 scored instances)
- **Wall-clock time**: 4020.0s

## Per-question-type breakdown

| Question type | Count | Mean score |
|---|---|---|
| `knowledge-update` | 10 | 0.600 |
| `multi-session` | 10 | 0.800 |
| `single-session-assistant` | 10 | 0.600 |
| `single-session-preference` | 10 | 0.900 |
| `single-session-user` | 10 | 0.900 |
| `temporal-reasoning` | 9 | 0.667 |
