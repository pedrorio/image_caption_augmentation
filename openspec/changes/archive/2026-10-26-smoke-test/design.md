# Design

## Context

See proposal.md. The check has to download `t5-small` once and then finish well under a few minutes.

## Goals / Non-Goals

- Goals: one optimizer step and one paraphrase batch with a checked tensor shape.
- Non-Goals: a quality score, a captioner, or the `ramsrigouthamg/t5_paraphraser` checkpoint in CI.

## Decisions

- `max_steps: 1`, `max_examples: 20`, `gpus: 0`, `compute_train_gleau: false`, short source length. GLEU decoding inside the step would dominate the job.
- Inference paraphrases the loaded source captions (at most 20) with `max_new_tokens` small enough for CPU.
- Actions caches `HF_HOME` and `data/cache`.

## Risks / Trade-offs

- [First CI run pays for the `t5-small` download] → The cache key is stable so later runs reuse it.
- [Sampling is nondeterministic] → The assertion is shape and a finite loss, not the string.

## Migration Plan

- CI command: `uv sync --frozen --all-groups` then `uv run pytest`.

## Open Questions

- None.
